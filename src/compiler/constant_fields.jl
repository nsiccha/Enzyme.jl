# Pointer fields of stack objects that only ever hold constant data.
#
# An immutable struct that holds a `Const` array beside an active one, such as
# `Overlay(base, active)` or a `view` or `reshape` of it, is passed by reference to
# a function that is not inlined. The struct's stack slot is active memory, so
# Enzyme needs a shadow for each pointer stored into it, and the constant array
# has none. The mixed-activity handler then gives it a zero-filled copy of the
# whole array (`constant_argument_data`), on every call and at every such store,
# however few of its elements are read.
#
# The copy is needed only because the callee's loads of the field are active: they
# read from an active argument, and Enzyme accumulates the derivative of each
# element read through the loaded pointer into the field's shadow. This pass
# proves, before differentiation, that a field only ever holds constant data
# (`Const` arguments of the differentiated function, memory reached from them,
# and values loaded from other such fields), following the slot through the
# internal functions it is passed to by reference. It then tags every load of the
# field as inactive and every store into it with `enzymejl_constant_field`. No
# derivative flows through the loaded pointer, so the field's shadow is never read
# or written, and the mixed-activity handler stores the primal pointer into it,
# as it already does for data whose type cannot hold derivatives. A value written
# through a tagged load into constant memory is not differentiated, which is the
# documented behaviour for values stored into `Const` arguments.
#
# The tags hold only for the activities of the current compilation;
# `strip_activity_inactive_md!` removes them after `enzyme!`.
const CONSTANT_FIELD_STORE = "enzymejl_constant_field"

mutable struct ConstantFields
    primalf::LLVM.Function
    const_args::Set{LLVM.Value}
    # Union-find over slot bases: allocas and pointer parameters of internal functions.
    parent::Dict{LLVM.Value, LLVM.Value}
    escaped::Set{LLVM.Value}
    loads::Dict{LLVM.Value, Vector{Tuple{LLVM.LoadInst, Int}}}
    stores::Dict{LLVM.Value, Vector{Tuple{LLVM.StoreInst, Int}}}
    # (offset, length) of each zero `memset`, a store of null to the fields it covers.
    zeroed::Dict{LLVM.Value, Vector{Tuple{Int, Int}}}
    unreachable::Dict{LLVM.Function, Set{LLVM.BasicBlock}}
    callee::Dict{LLVM.Function, Bool}
    # (root base, byte offset) => whether the field is still assumed constant.
    constant::Dict{Tuple{LLVM.Value, Int}, Bool}
    # Phis of constant and active objects to split into phis of loaded values.
    unsink::Set{LLVM.PHIInst}
end

function ConstantFields(primalf::LLVM.Function, const_args::Set{LLVM.Value})
    return ConstantFields(
        primalf, const_args, Dict{LLVM.Value, LLVM.Value}(), Set{LLVM.Value}(),
        Dict{LLVM.Value, Vector{Tuple{LLVM.LoadInst, Int}}}(),
        Dict{LLVM.Value, Vector{Tuple{LLVM.StoreInst, Int}}}(),
        Dict{LLVM.Value, Vector{Tuple{Int, Int}}}(),
        Dict{LLVM.Function, Set{LLVM.BasicBlock}}(), Dict{LLVM.Function, Bool}(),
        Dict{Tuple{LLVM.Value, Int}, Bool}(), Set{LLVM.PHIInst}(),
    )
end

is_tracked_pointer(@nospecialize(ty::LLVM.LLVMType)) =
    isa(ty, LLVM.PointerType) && LLVM.addrspace(ty) == Tracked

function is_constant_field_load(@nospecialize(inst::LLVM.Value))::Bool
    isa(inst, LLVM.LoadInst) || return false
    md = metadata(inst)
    haskey(md, "enzyme_inactive") || return false
    return inactive_reason(md["enzyme_inactive"]) == INACTIVE_CONSTANT_FIELD
end

# Blocks from which every path ends in `unreachable`. Enzyme does not
# differentiate them (`getGuaranteedUnreachable`), so the throw paths that box a
# struct for an error message never see its shadow.
function guaranteed_unreachable(f::LLVM.Function)::Set{LLVM.BasicBlock}
    dead = Set{LLVM.BasicBlock}()
    changed = true
    while changed
        changed = false
        for bb in blocks(f)
            bb in dead && continue
            term = terminator(bb)
            term === nothing && continue
            isa(term, LLVM.RetInst) && continue
            if isa(term, LLVM.UnreachableInst)
                push!(dead, bb)
                changed = true
                continue
            end
            succs = successors(term)
            isempty(succs) && continue
            if all(in(dead), succs)
                push!(dead, bb)
                changed = true
            end
        end
    end
    return dead
end

function in_unreachable(cf::ConstantFields, inst::LLVM.Instruction)::Bool
    bb = LLVM.parent(inst)
    f = LLVM.parent(bb)
    dead = get(cf.unreachable, f, nothing)
    if dead === nothing
        dead = guaranteed_unreachable(f)
        cf.unreachable[f] = dead
    end
    return bb in dead
end

# Whether all call sites of `f` are visible: a defined internal function, used only
# as the callee of direct calls, without a special Enzyme handler (custom rules and
# known functions carry `enzyme_math` and external linkage, see `handleCustom`).
function constant_field_callee(cf::ConstantFields, f::LLVM.Function)::Bool
    known = get(cf.callee, f, nothing)
    known === nothing || return known
    ok = !isempty(blocks(f)) && f != cf.primalf &&
        linkage(f) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage) &&
        !LLVM.isvararg(LLVM.function_type(f))
    if ok
        for attr in collect(function_attributes(f))
            if isa(attr, LLVM.StringAttribute) && kind(attr) == "enzyme_math"
                ok = false
                break
            end
        end
    end
    if ok
        for u in LLVM.uses(f)
            call = LLVM.user(u)
            if !isa(call, LLVM.CallInst) || called_operand(call) != f || f in arguments(call)
                ok = false
                break
            end
        end
    end
    cf.callee[f] = ok
    return ok
end

function constant_field_root(cf::ConstantFields, @nospecialize(base::LLVM.Value))::LLVM.Value
    while true
        p = get(cf.parent, base, base)
        p == base && return base
        base = p
    end
    return
end

function add_constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value))
    haskey(cf.parent, base) && return false
    cf.parent[base] = base
    cf.loads[base] = Tuple{LLVM.LoadInst, Int}[]
    cf.stores[base] = Tuple{LLVM.StoreInst, Int}[]
    cf.zeroed[base] = Tuple{Int, Int}[]
    return true
end

# Register `base` (an alloca or a pointer parameter) and record its accesses. The
# memory behind a parameter whose call sites this pass cannot all see escapes.
function constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value))
    add_constant_field_base!(cf, base) || return
    if isa(base, LLVM.Argument)
        f = LLVM.Function(LLVM.API.LLVMGetParamParent(base))
        if f == cf.primalf || !constant_field_callee(cf, f)
            push!(cf.escaped, base)
            return
        end
    end
    walk_constant_field_base!(cf, base, base, 0)
    return
end

function join_constant_field_bases!(cf::ConstantFields, @nospecialize(a::LLVM.Value), @nospecialize(b::LLVM.Value))
    ra = constant_field_root(cf, a)
    rb = constant_field_root(cf, b)
    ra == rb || (cf.parent[rb] = ra)
    return
end

# The byte offset of a GEP with only constant indices, or `nothing`.
function constant_gep_offset(gep::LLVM.GetElementPtrInst)::Union{Nothing, Int}
    for idx in operands(gep)[2:end]
        isa(idx, LLVM.ConstantInt) || return nothing
    end
    b = LLVM.IRBuilder()
    position!(b, gep)
    off = API.EnzymeComputeByteOffsetOfGEP(b, gep, LLVM.IntType(8 * sizeof(Int)))
    dispose(b)
    return isa(off, LLVM.ConstantInt) ? convert(Int, off) : nothing
end

# Strip casts and constant GEPs from a pointer: its base and byte offset.
function constant_field_pointer(@nospecialize(ptr::LLVM.Value))::Tuple{LLVM.Value, Int}
    offset = 0
    while true
        if isa(ptr, LLVM.BitCastInst) || isa(ptr, LLVM.AddrSpaceCastInst)
            ptr = operands(ptr)[1]
        elseif isa(ptr, LLVM.GetElementPtrInst)
            off = constant_gep_offset(ptr)
            off === nothing && break
            offset += off
            ptr = operands(ptr)[1]
        else
            break
        end
    end
    return ptr, offset
end

# Record every access to the memory of `base` reached through `ptr`, which points
# `offset` bytes into it. Any use that could let other code read or write the
# memory without going through such an access makes the whole slot escape.
function walk_constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value), @nospecialize(ptr::LLVM.Value), offset::Int)
    for u in LLVM.uses(ptr)
        user = LLVM.user(u)
        if !isa(user, LLVM.Instruction)
            push!(cf.escaped, base)
            continue
        end
        in_unreachable(cf, user) && continue
        if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst)
            walk_constant_field_base!(cf, base, user, offset)
        elseif isa(user, LLVM.GetElementPtrInst) && operands(user)[1] == ptr
            off = constant_gep_offset(user)
            if off === nothing
                push!(cf.escaped, base)
            else
                walk_constant_field_base!(cf, base, user, offset + off)
            end
        elseif isa(user, LLVM.LoadInst)
            push!(cf.loads[base], (user, offset))
        elseif isa(user, LLVM.StoreInst) && operands(user)[2] == ptr && operands(user)[1] != ptr
            push!(cf.stores[base], (user, offset))
        elseif isa(user, LLVM.CallInst)
            walk_constant_field_call!(cf, base, user, ptr, offset)
        else
            push!(cf.escaped, base)
        end
    end
    return
end

function walk_constant_field_call!(cf::ConstantFields, @nospecialize(base::LLVM.Value), call::LLVM.CallInst, @nospecialize(ptr::LLVM.Value), offset::Int)
    callee = called_operand(call)
    if isa(callee, LLVM.Function)
        intr = LLVM.API.LLVMGetIntrinsicID(callee)
        if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
                intr == LLVM.Intrinsic("llvm.lifetime.end").id
            return
        end
        # Julia 1.12 zeroes the roots array of a by-reference argument first.
        if intr == LLVM.Intrinsic("llvm.memset").id
            ops = arguments(call)
            if ops[1] == ptr && ops[2] != ptr && isa(ops[2], LLVM.ConstantInt) &&
                    convert(Int, ops[2]) == 0 && isa(ops[3], LLVM.ConstantInt)
                push!(cf.zeroed[base], (offset, convert(Int, ops[3])))
                return
            end
        end
    end
    if !isa(callee, LLVM.Function) || callee == ptr || offset != 0 ||
            !constant_field_callee(cf, callee)
        push!(cf.escaped, base)
        return
    end
    params = parameters(callee)
    for (i, arg) in enumerate(arguments(call))
        arg == ptr || continue
        constant_field_base!(cf, params[i])
        join_constant_field_bases!(cf, base, params[i])
    end
    return
end

# Whether `val` is constant data: a `Const` argument of the differentiated
# function, a value read from memory reached from one, a value loaded from a
# field still assumed constant, or a merge of such values. `seen` breaks cycles
# optimistically, as befits the greatest fixpoint `mark_constant_fields!` computes.
function constant_field_data(cf::ConstantFields, @nospecialize(val::LLVM.Value), seen::Set{LLVM.Value})::Bool
    while isa(val, LLVM.BitCastInst) || isa(val, LLVM.AddrSpaceCastInst)
        val = operands(val)[1]
    end
    # A pointer field of a `Const` argument passed by value.
    if isa(val, LLVM.ExtractValueInst)
        return operands(val)[1] in cf.const_args
    end
    if isa(val, LLVM.PointerNull) || isa(val, LLVM.UndefValue) || isa(val, LLVM.PoisonValue)
        return true
    end
    val in seen && return true
    if isa(val, LLVM.Argument)
        val in cf.const_args && return true
        f = LLVM.Function(LLVM.API.LLVMGetParamParent(val))
        (f != cf.primalf && constant_field_callee(cf, f)) || return false
        push!(seen, val)
        idx = findfirst(==(val), collect(parameters(f)))::Int
        for u in LLVM.uses(f)
            call = LLVM.user(u)::LLVM.CallInst
            in_unreachable(cf, call) && continue
            constant_field_data(cf, arguments(call)[idx], seen) || return false
        end
        return true
    end
    if isa(val, LLVM.PHIInst)
        push!(seen, val)
        for (v, _) in LLVM.incoming(val)
            constant_field_data(cf, v, seen) || return false
        end
        return true
    end
    if isa(val, LLVM.SelectInst)
        push!(seen, val)
        return constant_field_data(cf, operands(val)[2], seen) &&
            constant_field_data(cf, operands(val)[3], seen)
    end
    isa(val, LLVM.LoadInst) || return false
    push!(seen, val)
    base, offset = constant_field_pointer(operands(val)[1])
    if isa(base, LLVM.AllocaInst) && haskey(metadata(base), "enzyme_inactive")
        return true
    end
    if haskey(cf.parent, base)
        root = constant_field_root(cf, base)
        root in cf.escaped || return get(cf.constant, (root, offset), false)
    end
    # Memory reached from constant data (a field of a `Const` struct argument).
    return constant_field_data(cf, base, seen)
end

# The instructions of the block of `phi` that compute loads through it: casts,
# GEPs, loads of pointers and `julia.gc_loaded`, ending in loads of values that are
# not pointers, in block order. Load sinking turns `c ? base[j] : active[s]` into a
# load through a phi of the two data pointers, which `nodecayed_phis!` makes a phi
# of the two arrays: one active and one constant pointer, which has no shadow.
# Repeating the loads in each predecessor (each must branch only to this block,
# and no write may precede the loads) leaves a phi of loaded values instead,
# whose constant operand simply has no derivative.
function constant_field_unsink_tree(phi::LLVM.PHIInst)
    bb = LLVM.parent(phi)
    for (_, pred) in LLVM.incoming(phi)
        length(collect(successors(terminator(pred)))) == 1 || return nothing
    end
    tree = Set{LLVM.Instruction}()
    leaves = LLVM.LoadInst[]
    work = LLVM.Value[phi]
    while !isempty(work)
        v = pop!(work)
        for u in LLVM.uses(v)
            user = LLVM.user(u)
            (isa(user, LLVM.Instruction) && LLVM.parent(user) == bb) || return nothing
            user in tree && continue
            if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                    isa(user, LLVM.GetElementPtrInst) ||
                    (isa(user, LLVM.CallInst) && called_function_name(user) == "julia.gc_loaded")
                push!(tree, user)
                push!(work, user)
            elseif isa(user, LLVM.LoadInst) && !Bool(LLVM.API.LLVMGetVolatile(user))
                push!(tree, user)
                if isa(value_type(user), LLVM.PointerType)
                    push!(work, user)
                else
                    push!(leaves, user)
                end
            else
                return nothing
            end
        end
    end
    isempty(leaves) && return nothing
    # Every other operand must be available in the predecessors.
    for inst in tree, op in operands(inst)
        op in tree && continue
        if isa(op, LLVM.Instruction) && LLVM.parent(op) == bb && !isa(op, LLVM.PHIInst)
            return nothing
        end
    end
    ordered = LLVM.Instruction[]
    written = false
    for inst in instructions(bb)
        isa(inst, LLVM.PHIInst) && continue
        if inst in tree
            written && return nothing
            push!(ordered, inst)
            length(ordered) == length(tree) && break
        elseif root_path_may_write(inst)
            written = true
        end
    end
    return ordered, leaves
end

function incoming_value(phi::LLVM.PHIInst, pred::LLVM.BasicBlock)
    for (v, b) in LLVM.incoming(phi)
        b == pred && return v
    end
    return nothing
end

function unsink_constant_field_phi!(phi::LLVM.PHIInst)::Bool
    found = constant_field_unsink_tree(phi)
    found === nothing && return false
    ordered, leaves = found
    bb = LLVM.parent(phi)
    builder = LLVM.IRBuilder()
    position!(builder, phi)
    merged = Dict{LLVM.LoadInst, LLVM.PHIInst}()
    for leaf in leaves
        merged[leaf] = phi!(builder, value_type(leaf), LLVM.name(leaf))
    end
    for (_, pred) in LLVM.incoming(phi)
        position!(builder, terminator(pred))
        copies = Dict{LLVM.Value, LLVM.Value}()
        for inst in ordered
            clone = LLVM.Instruction(LLVM.API.LLVMInstructionClone(inst))
            LLVM.API.LLVMInsertIntoBuilder(builder, clone)
            for (i, op) in enumerate(operands(clone))
                new = get(copies, op, nothing)
                if new === nothing && isa(op, LLVM.PHIInst) && LLVM.parent(op) == bb
                    new = incoming_value(op, pred)
                end
                new === nothing || LLVM.API.LLVMSetOperand(clone, i - 1, new)
            end
            copies[inst] = clone
        end
        for leaf in leaves
            push!(LLVM.incoming(merged[leaf]), (copies[leaf], pred))
        end
    end
    for leaf in leaves
        # Memory metadata such as `!tbaa` is not allowed on a phi.
        haskey(metadata(leaf), "enzyme_type") &&
            (metadata(merged[leaf])["enzyme_type"] = metadata(leaf)["enzyme_type"])
        replace_uses!(leaf, merged[leaf])
    end
    for inst in reverse(ordered)
        isempty(LLVM.uses(inst)) && LLVM.erase!(inst)
    end
    isempty(LLVM.uses(phi)) && LLVM.erase!(phi)
    dispose(builder)
    return true
end

# Whether every use of `val`, a pointer read from a constant field (or derived
# from one), stays away from active data: a load or store through it, a store
# into a constant field, a comparison, or an argument of an internal function
# whose parameter is used likewise. A phi that also merges active data must be one
# `unsink_constant_field_phi!` can split; it is recorded in `cf.unsink`. Anything
# else would merge the pointer with active data where it has no shadow, so the
# mixed-activity handler would copy the whole array there, possibly in a loop,
# while without the tag one copy is made where the struct is built.
function constant_field_uses_safe(cf::ConstantFields, @nospecialize(val::LLVM.Value), visited::Set{LLVM.Value})::Bool
    val in visited && return true
    push!(visited, val)
    for u in LLVM.uses(val)
        user = LLVM.user(u)
        isa(user, LLVM.Instruction) || return false
        in_unreachable(cf, user) && continue
        ok = if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                isa(user, LLVM.GetElementPtrInst)
            constant_field_uses_safe(cf, user, visited)
        elseif isa(user, LLVM.LoadInst)
            # A pointer read from constant memory is constant data too.
            !isa(value_type(user), LLVM.PointerType) || constant_field_uses_safe(cf, user, visited)
        elseif isa(user, LLVM.StoreInst)
            if operands(user)[1] != val
                true
            else
                base, offset = constant_field_pointer(operands(user)[2])
                haskey(cf.parent, base) &&
                    get(cf.constant, (constant_field_root(cf, base), offset), false)
            end
        elseif isa(user, LLVM.PHIInst) || isa(user, LLVM.SelectInst)
            if constant_field_data(cf, user, Set{LLVM.Value}())
                constant_field_uses_safe(cf, user, visited)
            elseif isa(user, LLVM.PHIInst) && constant_field_unsink_tree(user) !== nothing
                push!(cf.unsink, user)
                true
            else
                false
            end
        elseif isa(user, LLVM.ICmpInst)
            true
        elseif isa(user, LLVM.PtrToIntInst)
            # Julia 1.12 compares data addresses (`mightalias`); an integer has no
            # shadow unless it is turned back into a pointer.
            never_pointer_again(user, Set{LLVM.Value}())
        elseif isa(user, LLVM.CallInst)
            constant_field_call_safe(cf, user, val, visited)
        else
            false
        end
        ok || return false
    end
    return true
end

# Whether an integer, and everything computed from it, stays an integer.
function never_pointer_again(@nospecialize(val::LLVM.Value), seen::Set{LLVM.Value})::Bool
    val in seen && return true
    push!(seen, val)
    for u in LLVM.uses(val)
        user = LLVM.user(u)
        isa(user, LLVM.Instruction) || return false
        if isa(value_type(user), LLVM.IntegerType)
            never_pointer_again(user, seen) || return false
        elseif isa(user, LLVM.GetElementPtrInst)
            # An index: the GEP's activity is its pointer operand's (the byte offsets
            # `nodecayed_phis!` computes on Julia 1.12).
            operands(user)[1] == val && return false
        elseif !(isa(user, LLVM.BrInst) || isa(user, LLVM.SwitchInst))
            return false
        end
    end
    return true
end

function constant_field_call_safe(cf::ConstantFields, call::LLVM.CallInst, @nospecialize(val::LLVM.Value), visited::Set{LLVM.Value})::Bool
    callee = called_operand(call)
    (isa(callee, LLVM.Function) && callee != val) || return false
    intr = LLVM.API.LLVMGetIntrinsicID(callee)
    if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
            intr == LLVM.Intrinsic("llvm.lifetime.end").id
        return true
    end
    nm = LLVM.name(callee)
    nm == "julia.gc_loaded" && return constant_field_uses_safe(cf, call, visited)
    nm == "julia.write_barrier" && return true
    constant_field_callee(cf, callee) || return false
    params = parameters(callee)
    for (i, arg) in enumerate(arguments(call))
        arg == val || continue
        constant_field_uses_safe(cf, params[i], visited) || return false
    end
    return true
end

# Drop a field once one of its stores writes data that is not constant, or one of
# its loads is used unsafely, under the current assumptions.
function settle_constant_fields!(cf::ConstantFields)
    changed = true
    while changed
        changed = false
        for (base, ss) in cf.stores
            root = constant_field_root(cf, base)
            for (st, off) in ss
                get(cf.constant, (root, off), false) || continue
                if !constant_field_data(cf, operands(st)[1], Set{LLVM.Value}())
                    cf.constant[(root, off)] = false
                    changed = true
                end
            end
        end
        changed && continue
        empty!(cf.unsink)
        for (base, ls) in cf.loads
            root = constant_field_root(cf, base)
            for (ld, off) in ls
                get(cf.constant, (root, off), false) || continue
                if !constant_field_uses_safe(cf, ld, Set{LLVM.Value}())
                    cf.constant[(root, off)] = false
                    changed = true
                end
            end
        end
    end
    return
end

function mark_constant_fields!(mod::LLVM.Module, primalf::LLVM.Function, @nospecialize(args_activity::Vector{API.CDIFFE_TYPE}))
    # A recursive call could pass active data where the outer call passes `Const`.
    isempty(LLVM.uses(primalf)) || return 0
    const_args = Set{LLVM.Value}()
    for (p, act) in zip(parameters(primalf), args_activity)
        act == API.DFT_CONSTANT && push!(const_args, p)
    end
    isempty(const_args) && return 0
    cf = ConstantFields(primalf, const_args)

    for f in functions(mod)
        isempty(blocks(f)) && continue
        if f != primalf && constant_field_callee(cf, f)
            for (i, p) in enumerate(parameters(f))
                isa(value_type(p), LLVM.PointerType) || continue
                constant_field_base!(cf, p)
                # Every call site must pass the start of a slot this pass follows.
                for u in LLVM.uses(f)
                    call = LLVM.user(u)::LLVM.CallInst
                    in_unreachable(cf, call) && continue
                    actual, off = constant_field_pointer(arguments(call)[i])
                    if off == 0 && (isa(actual, LLVM.AllocaInst) || isa(actual, LLVM.Argument))
                        constant_field_base!(cf, actual)
                        join_constant_field_bases!(cf, p, actual)
                    else
                        push!(cf.escaped, p)
                    end
                end
            end
        end
        for bb in blocks(f), inst in instructions(bb)
            isa(inst, LLVM.AllocaInst) && constant_field_base!(cf, inst)
        end
    end
    for base in collect(cf.escaped)
        push!(cf.escaped, constant_field_root(cf, base))
    end

    # Candidate fields: pointer-sized tracked-pointer accesses at one offset, with
    # no other access overlapping it.
    accesses = Dict{LLVM.Value, Vector{Tuple{LLVM.Instruction, Int}}}()
    zeroed = Dict{LLVM.Value, Vector{Tuple{Int, Int}}}()
    for (base, ls) in cf.loads
        root = constant_field_root(cf, base)
        root in cf.escaped && continue
        acc = get!(accesses, root, Tuple{LLVM.Instruction, Int}[])
        append!(acc, ls)
        append!(acc, cf.stores[base])
        append!(get!(zeroed, root, Tuple{Int, Int}[]), cf.zeroed[base])
    end
    dl = datalayout(mod)
    for (root, acc) in accesses
        fields = Dict{Int, Bool}()
        for (inst, off) in acc
            ty = isa(inst, LLVM.LoadInst) ? value_type(inst) : value_type(operands(inst)[1])
            if is_tracked_pointer(ty)
                get!(fields, off, true)
            end
        end
        for (inst, off) in acc
            ty = isa(inst, LLVM.LoadInst) ? value_type(inst) : value_type(operands(inst)[1])
            sz = Int(LLVM.storage_size(dl, ty))
            for (foff, ok) in fields
                ok || continue
                overlaps = off < foff + sizeof(Int) && foff < off + sz
                if overlaps && !(off == foff && is_tracked_pointer(ty))
                    fields[foff] = false
                end
            end
        end
        for (off, len) in zeroed[root]
            for (foff, ok) in fields
                ok || continue
                overlaps = off < foff + sizeof(Int) && foff < off + len
                covers = off <= foff && foff + sizeof(Int) <= off + len
                if overlaps && !covers
                    fields[foff] = false
                end
            end
        end
        for (foff, ok) in fields
            ok && (cf.constant[(root, foff)] = true)
        end
    end
    isempty(cf.constant) && return 0

    # A greatest fixpoint.
    settle_constant_fields!(cf)
    for phi in cf.unsink
        unsink_constant_field_phi!(phi)
    end

    marked = 0
    for (base, ls) in cf.loads
        root = constant_field_root(cf, base)
        for (ld, off) in ls
            get(cf.constant, (root, off), false) || continue
            md = metadata(ld)
            haskey(md, "enzyme_inactive") && continue
            md["enzyme_inactive"] = inactive_md(INACTIVE_CONSTANT_FIELD)
            marked += 1
        end
    end
    for (base, ss) in cf.stores
        root = constant_field_root(cf, base)
        for (st, off) in ss
            get(cf.constant, (root, off), false) || continue
            metadata(st)[CONSTANT_FIELD_STORE] = MDNode(LLVM.Metadata[])
        end
    end
    return marked
end
