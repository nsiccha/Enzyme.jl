function restore_alloca_type!(f::LLVM.Function)
    replaceAndErase = Tuple{LLVM.AllocaInst, LLVMType}[]
    dl = f.parent.datalayout
    for bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.AllocaInst)
            if haskey(inst.metadata, "enzymejl_allocart") || haskey(inst.metadata, "enzymejl_gc_alloc_rt")
                mds = inst.metadata[haskey(inst.metadata, "enzymejl_allocart") ? "enzymejl_allocart" : "enzymejl_gc_alloc_rt"].operands[1]::MDString
                mds = Base.convert(String, mds)
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, mds))
                RT = Base.unsafe_pointer_to_objref(ptr)
                if RT isa Union
                    continue
                end
                at = inst.allocated_type
                lrt = struct_to_llvm(RT)
                if at == lrt
                    continue
                end
                cnt = inst.operands[1]
                if !isa(cnt, LLVM.ConstantInt) || convert(UInt, cnt) != 1
                    continue
                end
                if LLVM.storage_size(dl, at) == LLVM.storage_size(dl, lrt) && CountTrackedPointers(at).count == 0
                    push!(replaceAndErase, (inst, lrt))
                end
            end
        end
    end

    if length(replaceAndErase) == 0
        return false
    end

    for (al, lrt) in replaceAndErase
        at = al.allocated_type
        tracked_lrt = CountTrackedPointers(lrt).count
        tracked_at = CountTrackedPointers(at).count
        if tracked_lrt != 0 && tracked_at == 0
            lrt2 = strip_tracked_pointers(lrt)
            @assert LLVM.storage_size(dl, lrt2) == LLVM.storage_size(dl, lrt)
            lrt = lrt2
            tracked_lrt = CountTrackedPointers(lrt).count
            if tracked_lrt != 0
                ccall(:jl_, Cvoid, (Any,), ("BAD1", string(al), string(lrt), tracked_lrt))
                throw(AssertionError("tracked_lrt ($tracked_lrt) != 0, $(string(lrt))"))
            end
        end
        if tracked_lrt != tracked_at
            ccall(:jl_, Cvoid, (Any,), ("BAD2", string(al), string(lrt), tracked_lrt))
            throw(AssertionError("tracked_lrt ($tracked_lrt) != tracked_at ($tracked_at), at=$(string(at)), lrt=$(string(lrt)) al=$(string(al))"))
        end
        b = IRBuilder()
        position!(b, LLVM.before(al))
        al2 = alloca!(b, lrt)
        cst = al2
        if cst.value_type != al.value_type
            cst = bitcast!(b, cst, al.value_type)
        end
        API.EnzymeCopyMetadata(al2, al)
        al2.alignment = al.alignment
        take_name!(al2, al)
        LLVM.replace_uses!(al, cst)
        erase!(al)
    end
    return true
end

# Rewrite calls with "jl_roots" to only have the jl_value_t attached and not  { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [2 x i64] } %unbox110183_replacementA
function rewrite_ccalls!(mod::LLVM.Module)
    @dispose B = IRBuilder() begin
        rewrite_ccalls!(B, mod)
    end
    return
end
function rewrite_ccalls!(B::IRBuilder, mod::LLVM.Module)
    for f in collect(mod.functions)
        replaceAndErase = Tuple{Instruction, Instruction}[]
        for bb in f.blocks, inst in bb.instructions
            if isa(inst, LLVM.CallInst)
                fn = inst.called_operand
                changed = false
                position!(B, LLVM.before(inst))
                if isa(fn, LLVM.Function) && fn.name == "llvm.julia.gc_preserve_begin"
                    uservals = LLVM.Value[]
                    # the argument of the original call that each of `uservals` is, if any
                    userargs = Union{Int, Nothing}[]
                    for (i, lval) in enumerate(collect(inst.arguments))
                        llty = lval.value_type
                        if isa(llty, LLVM.PointerType)
                            push!(uservals, lval)
                            push!(userargs, i)
                            continue
                        end
                        vals = get_julia_inner_types(B, nothing, lval)
                        unchanged = length(vals) == 1 && vals[1] == lval
                        for v in vals
                            if isa(v, LLVM.PointerNull)
                                subchanged = true
                                continue
                            end
                            push!(uservals, v)
                            push!(userargs, unchanged ? i : nothing)
                        end
                        if unchanged
                            continue
                        end
                        changed = true
                    end
                    if changed
                        prevname = inst.name
                        inst.name = ""
                        newinst = call!(
                            B,
                            inst.called_type,
                            inst.called_operand,
                            uservals,
                            collect(inst.operand_bundles),
                            prevname,
                        )
                        append!(newinst.function_attributes, inst.function_attributes)
                        append!(newinst.return_attributes, inst.return_attributes)
                        # arguments that were split or dropped have no attributes to keep
                        for (j, i) in enumerate(userargs)
                            i === nothing && continue
                            append!(newinst.argument_attributes[j], inst.argument_attributes[i])
                        end
                        API.EnzymeCopyMetadata(newinst, inst)
                        newinst.callconv = inst.callconv
                        push!(replaceAndErase, (inst, newinst))
                    end
                    continue
                end
                newbundles = OperandBundle[]
                for bunduse in inst.operand_bundles
                    if bunduse.tag != "jl_roots"
                        push!(newbundles, bunduse)
                        continue
                    end
                    uservals = LLVM.Value[]
                    subchanged = false
                    for lval in bunduse.inputs
                        llty = lval.value_type
                        if isa(llty, LLVM.PointerType)
                            push!(uservals, lval)
                            continue
                        end
                        vals = get_julia_inner_types(B, nothing, lval)
                        for v in vals
                            if isa(v, LLVM.PointerNull)
                                subchanged = true
                                continue
                            end
                            push!(uservals, v)
                        end
                        if length(vals) == 1 && vals[1] == lval
                            continue
                        end
                        subchanged = true
                    end
                    if !subchanged
                        push!(newbundles, bunduse)
                        continue
                    end
                    changed = true
                    push!(newbundles, OperandBundle(bunduse.tag, uservals))
                end
                changed = false
                if changed
                    prevname = inst.name
                    inst.name = ""
                    newinst = call!(
                        B,
                        inst.called_type,
                        inst.called_operand,
                        collect(inst.arguments),
                        newbundles,
                        prevname,
                    )
                    append!(newinst.function_attributes, inst.function_attributes)
                    append!(newinst.return_attributes, inst.return_attributes)
                    for i in 1:(length(inst.arguments))
                        append!(newinst.argument_attributes[i], inst.argument_attributes[i])
                    end
                    API.EnzymeCopyMetadata(newinst, inst)
                    newinst.callconv = inst.callconv
                    push!(replaceAndErase, (inst, newinst))
                end
            end
        end
        for (inst, newinst) in replaceAndErase
            replace_uses!(inst, newinst)
            erase!(inst)
        end
    end
    return
end

"""
    fixup_1p12_sret!(f::LLVM.Function)

Rewrite the untyped store of a return value that needs both an `sret` buffer and a
`returnRoots` array into field-wise stores that skip the GC-tracked slots.

Julia 1.12 changed the convention for such a return: the callee writes the tracked
pointers only into `returnRoots` and leaves the corresponding slots of the `sret`
buffer undefined, so a caller has to recombine the two halves (which is what
[`recombine_value!`](@ref) does). Codegen spells the write of the remaining, inline
data as a single untyped `llvm.memcpy` out of an `[N x i64]` alloca, which also
drags the undefined bytes into the tracked slots -- and Enzyme would then take those
for live `jlvalue`s. Replacing the memcpy with stores of just the untracked fields
keeps the tracked slots alone, matching what the convention promises the caller.

The rewrite is keyed on the ABI actually present in the IR rather than on a version
bound: it only fires for a `memcpy` whose destination parameter carries an `sret`
attribute for exactly `RT`.
"""
function fixup_1p12_sret!(f::LLVM.Function)
    if VERSION < v"1.12"
        return
    end
    mi, RT = enzyme_custom_extract_mi(f, false)
    if mi === nothing
        return
    end

    _, sret, returnRoots = get_return_info(RT)

    if sret === nothing || returnRoots == nothing
        return
    end

    lltype = convert(LLVMType, RT)

    # Bail out if parameter 1 is not the `sret` buffer holding an `RT`, i.e. if the
    # calling convention is not the one this rewrite knows about.
    if sret_ty(f, 1, #=btval=# nothing, #=throw_error=# false) != lltype
        return
    end

    dl = f.parent.datalayout

    torep = LLVM.Instruction[]
    for u in f.parameters[1].uses
        ci = u.user
        if isa(ci, LLVM.CallInst)
            cf = ci.called_function
            if cf !== nothing && cf.intrinsic == LLVM.Intrinsic("llvm.memcpy")
                cst = ci.operands[3]
                if cst isa LLVM.ConstantInt
                    push!(torep, ci)
                end
            end
        end
    end

    if length(torep) > 0
        # The memcpy covers the full layout up to Julia 1.13.0 and only the data
        # half, without the trailing tracked slots, since 1.13.1.
        sz = LLVM.storage_size(dl, lltype)
        split_sz = split_value_size(dl, lltype)
        for ci in torep
            cst = ci.operands[3]::LLVM.ConstantInt
            if convert(UInt, cst) != sz && convert(UInt, cst) != split_sz
                continue
            end
            B = LLVM.IRBuilder()
            position!(B, LLVM.before(ci))
            copy_struct_into!(B, lltype, ci.operands[1], ci.operands[2], false)
            LLVM.erase!(ci)
        end
    end
    return
end

"""
    unfold_root_phi_loads!(f::LLVM.Function) -> Bool

Turn a load through a phi of root-array pointers back into a phi of loads.

Julia 1.13.1+ keeps the inline GC roots of a split value lazily, as a pointer
into whatever roots array already holds them (`jl_gc_roots_t`), so the roots a
phi merges may be loaded right in its predecessors: from the roots argument of the
function on one edge, from a local roots alloca on another. LLVM's instcombine
then folds that phi of loads into one load of a `phi ptr` of the arrays. Enzyme
promotes every alloca of the augmented primal to a heap allocation and cannot
push that address-space change through a phi whose other operands are not
allocas ("Illegal address space propagation"). Sinking the load back into the
predecessors gives the alloca only plain loads again.

Only rewrite what is certainly equivalent: a `phi ptr` in address space 0 with an
alloca among its incoming values, whose users are all non-atomic loads of tracked
pointers, in its own block or in a successor whose only predecessor it is, that no
memory write precedes. Each load is re-created before the terminator of every
predecessor. That load is speculative on an edge whose predecessor has other
successors, and on every edge when the original load is in the successor, so it is
then only done for a pointer that is always dereferenceable: an alloca, or an
argument whose `dereferenceable` bytes cover it.
"""
function unfold_root_phi_loads!(f::LLVM.Function)::Bool
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    changed = false
    for bb in f.blocks
        for phi in collect(bb.instructions)
            isa(phi, LLVM.PHIInst) || continue
            pty = phi.value_type
            (isa(pty, LLVM.PointerType) && pty.addrspace == 0) || continue

            incs = collect(phi.incoming)
            any(((v, _),) -> isa(first(get_base_and_offset(v)), LLVM.AllocaInst), incs) || continue

            # The users: loads of tracked pointers, directly or through a GEP
            # with constant indices, before any write. They may sit in this
            # block, or in a successor whose only predecessor it is (a split
            # critical edge).
            accesses = Tuple{LLVM.LoadInst, Union{Nothing, LLVM.GetElementPtrInst}}[]
            ok = true
            for u in phi.uses
                inst = u.user
                if isa(inst, LLVM.GetElementPtrInst) && root_block_after(inst, bb) &&
                        inst.operands[1] == phi && all(isa(op, LLVM.ConstantInt) for op in inst.operands[2:end])
                    for u2 in inst.uses
                        ld = u2.user
                        if !(root_load_in(ld, inst, inst.parent, T_prjlvalue))
                            ok = false
                            break
                        end
                        push!(accesses, (ld, inst))
                    end
                elseif root_block_after(inst, bb) && root_load_in(inst, phi, inst.parent, T_prjlvalue)
                    push!(accesses, (inst, nothing))
                else
                    ok = false
                end
                ok || break
            end
            (ok && !isempty(accesses)) || continue
            # A write before one of the loads could change what they read: none
            # in this block after the phi, and none in a load's own block before it.
            for inst in bb.instructions
                root_path_may_write(inst) || continue
                # The first write in this block: a load in a successor, or
                # after it in this block, may read what it wrote.
                for (ld, _) in accesses
                    if ld.parent != bb || comes_before(inst, ld)
                        ok = false
                        break
                    end
                end
                break
            end
            for (ld, _) in accesses
                ld.parent == bb && continue
                for inst in ld.parent.instructions
                    root_path_may_write(inst) || continue
                    if comes_before(inst, ld)
                        ok = false
                    end
                    break
                end
            end
            ok || continue

            # A speculative load needs an always dereferenceable pointer.
            hoisted = any(((ld, _),) -> ld.parent != bb, accesses)
            for (v, pred) in incs
                nsucc = length(collect(pred.terminator.successors))
                if (hoisted || nsucc != 1) && !dereferenceable_root_ptr(v)
                    ok = false
                    break
                end
            end
            ok || continue

            builder = LLVM.IRBuilder()
            for (ld, gep) in accesses
                newphi = let
                    position!(builder, LLVM.before(phi))
                    phi!(builder, T_prjlvalue, ld.name)
                end
                done = Dict{LLVM.BasicBlock, LLVM.LoadInst}()
                for (v, pred) in incs
                    nld = get!(done, pred) do
                        position!(builder, LLVM.before(pred.terminator))
                        ptr = v
                        if gep !== nothing
                            elty = gep.source_element_type
                            inds = LLVM.Value[op for op in gep.operands[2:end]]
                            ptr = if gep.inbounds
                                inbounds_gep!(builder, elty, v, inds)
                            else
                                gep!(builder, elty, v, inds)
                            end
                        end
                        nld = load!(builder, T_prjlvalue, ptr)
                        nld.alignment = ld.alignment
                        copy_metadata!(nld, ld)
                        nld
                    end
                    push!(newphi.incoming, (nld, pred))
                end
                # Memory metadata such as `!tbaa` is not allowed on a phi.
                for k in ("enzyme_type", "enzyme_inactive", "enzyme_active")
                    if haskey(ld.metadata, k)
                        newphi.metadata[k] = ld.metadata[k]
                    end
                end
                replace_uses!(ld, newphi)
                LLVM.erase!(ld)
            end
            for (_, gep) in accesses
                if gep !== nothing && isempty(collect(gep.uses))
                    LLVM.erase!(gep)
                end
            end
            @assert isempty(collect(phi.uses))
            LLVM.erase!(phi)
            dispose(builder)
            changed = true
        end
    end
    return changed
end

"""
    root_block_after(inst, bb)

Whether `inst` is in `bb`, or in a successor of `bb` that has no other
predecessor, so that `bb` runs right before it.
"""
function root_block_after(@nospecialize(inst::LLVM.Value), bb::LLVM.BasicBlock)::Bool
    isa(inst, LLVM.Instruction) || return false
    ib = inst.parent
    ib == bb && return true
    preds = ib.predecessors
    return length(preds) == 1 && first(preds) == bb
end

"""
    root_path_may_write(inst) -> Bool

Whether `inst` may write memory. Unlike `mayWriteToMemory`, which only reads the
attributes of the call site, a call is also known not to write when its callee is
read-only, as for intrinsics such as `llvm.smax` that LLVM leaves between a phi
and the loads it moved into a successor.
"""
function root_path_may_write(@nospecialize(inst::LLVM.Instruction))::Bool
    mayWriteToMemory(inst) || return false
    if isa(inst, LLVM.CallInst)
        callee = inst.called_operand
        if isa(callee, LLVM.Function) && is_readonly(callee)
            return false
        end
    end
    return true
end

"""
    dereferenceable_root_ptr(v) -> Bool

Whether a pointer-sized load from `v` is safe even where the program would not
have loaded from it: `v` points into an alloca, or at a constant offset into an
argument whose `dereferenceable` attribute covers the load.
"""
function dereferenceable_root_ptr(@nospecialize(v::LLVM.Value))::Bool
    base, offset = get_base_and_offset(v)
    isa(base, LLVM.AllocaInst) && return true
    isa(base, LLVM.Argument) || return false
    offset >= 0 || return false
    f = base.parent
    idx = findfirst(==(base), collect(f.parameters))
    idx === nothing && return false
    for attr in collect(f.parameter_attributes[idx])
        if isa(attr, LLVM.EnumAttribute) && attr.kind == :dereferenceable
            return offset + sizeof(Int) <= attr.value
        end
    end
    return false
end

"""
    root_load_in(inst, ptr, bb, T_prjlvalue)

Whether `inst` is a plain (non-volatile, non-atomic) load of a tracked pointer
from `ptr`, placed in the block `bb`.
"""
function root_load_in(@nospecialize(inst::LLVM.Value), @nospecialize(ptr::LLVM.Value), bb::LLVM.BasicBlock, T_prjlvalue::LLVM.LLVMType)::Bool
    return isa(inst, LLVM.LoadInst) && inst.parent == bb &&
        inst.operands[1] == ptr && inst.value_type == T_prjlvalue &&
        !inst.volatile && !LLVM.isatomic(inst)
end

"""
    rewrite_abi_converter_calls!(mod::LLVM.Module)

Julia 1.12+ lowers `@cfunction` to a world-age-guarded dispatch site that calls
`jl_get_abi_converter` to obtain a callable pointer for the target in the
current world. That runtime resolver picks a code instance from the native JIT
cache, which is the wrong one for GPUCompiler-emitted code: both its owner
(`ci->owner`) and its ABI (compiled with `gcstack_arg = true`, i.e. pgcstack in
the swiftself register) differ from this module (`gcstack_arg = false`), so the
raw specsig pointer handed back is called with a mismatched ABI and reads a
garbage GC stack out of the swiftself register (#3284).

Rewrite each such site to act like `jl_apply_generic` instead: Julia's codegen
already emitted an in-module `unspecialized` dispatcher thunk for every site
(stored in the cfuncdata global) that boxes the arguments and dispatches via
`jl_apply_generic`. That thunk was compiled with this module's own ABI and
resolves the callee from the correct cache in the current world, so replace
every `jl_get_abi_converter` call with it and let the world-age guard fold
away.
"""
function rewrite_abi_converter_calls!(mod::LLVM.Module)
    for fname in ("jl_get_abi_converter", "ijl_get_abi_converter")
        if !haskey(mod.functions, fname)
            continue
        end
        f = mod.functions[fname]
        for u in collect(f.uses)
            ci = u.user
            if !isa(ci, LLVM.CallInst) || ci.called_operand != f
                continue
            end
            # `jl_get_abi_converter(ct, data)` on 1.13+, `(ct, fptr, last_world, data)`
            # on 1.12; the last argument is the cfuncdata global in both.
            data = ci.arguments[end]
            if !isa(data, LLVM.GlobalVariable)
                error("Enzyme internal error: expected cfuncdata global as last argument of $fname in $(string(ci))")
            end
            init = data.initializer
            nslots = length(init.operands)
            # struct cfuncdata_t: the unspecialized thunk is the 3rd field on 1.12
            # ([plast_codeinst, last_codeinst, unspecialized, declrt, sigt, flags]) and
            # the 5th on 1.13+ ([fptr, last_world, plast_codeinst, last_codeinst,
            # unspecialized, declrt, sigt, flags]).
            slot = nslots == 6 ? 3 : nslots == 8 ? 5 : error("Enzyme internal error: unknown cfuncdata layout with $nslots slots in $(string(data))")
            unspec = init.operands[slot]
            if !isa(unspec, LLVM.Function)
                error("Enzyme internal error: cfuncdata of $fname has no unspecialized dispatcher in $(string(data))")
            end
            replace_uses!(ci, unspec)
            LLVM.erase!(ci)
        end
    end
    return nothing
end

"""
    memoryref_root_load(call::LLVM.CallInst)

The root operand of `julia.gc_loaded(root, data)` when both operands are loads of the two
fields of one `MemoryRef` (`ptr_or_offset`, then `mem`, the next pointer-sized word), and
`nothing` otherwise.

From Julia 1.11 an element read loads both fields of the array's `MemoryRef` and joins them
in `julia.gc_loaded`. `force_recompute!` recomputes the data pointer in the reverse pass, and
libEnzyme then recomputes `julia.gc_loaded` too, so it needs the root there. Left cacheable,
that root, or its shadow, went on the tape whenever a later write might overwrite the memory
holding the `MemoryRef`. The two fields are only ever written together, so reloading the
data pointer already assumes the `mem` field is unchanged as well, and the tape kept a root
for a pointer it no longer roots. A recursive function's tape is a heap object, so that one
cached root cost an allocation per recursive call: 48 bytes per node of Base's pairwise
`mapreduce_impl` over a `SubArray`, on every Reverse call.
"""
function memoryref_root_load(call::LLVM.CallInst)
    ops = call.operands
    root = ops[1]
    data = ops[2]
    if isa(data, LLVM.BitCastInst)
        data = data.operands[1]
    end
    isa(root, LLVM.LoadInst) || return nothing
    isa(data, LLVM.LoadInst) || return nothing
    root_base, root_offset = get_base_and_offset(root.operands[1])
    data_base, data_offset = get_base_and_offset(data.operands[1])
    root_base == data_base || return nothing
    root_offset == data_offset + sizeof(Ptr{Cvoid}) || return nothing
    return root
end

"""
    cache_memoryref_pointer_integers(value)

Keep integer addresses derived from a recomputed MemoryRef data pointer when the reverse
pass needs them. Julia's bounds checks compute offsets relative to the Memory's data
pointer. A later sizehint! can change the primal Array's MemoryRef, so recomputing only
one address in that subtraction gives an offset into a different allocation. This also
occurs when reversing the pointer arithmetic generated by an inner Forward pass.

Cache the integer uses, rather than the pointer or its GC root: ordinary recursive
reductions must still recompute their MemoryRef loads without allocating rooted tapes.
"""
function cache_memoryref_pointer_integers(value::LLVM.Value)
    seen = Set{LLVM.Value}()
    work = LLVM.Value[value]
    while !isempty(work)
        value = pop!(work)
        value in seen && continue
        push!(seen, value)
        for use in value.uses
            user = use.user
            if user isa LLVM.PtrToIntInst
                user.metadata["enzyme_mustcache"] = MDNode(LLVM.Metadata[])
            elseif user isa Union{LLVM.BitCastInst, LLVM.AddrSpaceCastInst, LLVM.GetElementPtrInst, LLVM.PHIInst, LLVM.SelectInst}
                push!(work, user)
            end
        end
    end
    return nothing
end

function force_recompute!(mod::LLVM.Module)
    for f in mod.functions, bb in f.blocks
        iter = LLVM.API.LLVMGetFirstInstruction(bb)
        while iter != C_NULL
            inst = LLVM.Instruction(iter)
            iter = LLVM.API.LLVMGetNextInstruction(iter)
            if isa(inst, LLVM.CallInst)
                cf = inst.called_operand
                if isa(cf, LLVM.Function) && cf.name == "julia.gc_loaded"
                    root = memoryref_root_load(inst)
                    if root !== nothing
                        root.metadata["enzyme_nocache"] = MDNode(LLVM.Metadata[])
                    end
                end
            end
            if isa(inst, LLVM.LoadInst)
                has_loaded = false
                for u in inst.uses
                    v = u.user
                    if isa(v, LLVM.CallInst)
                        cf = v.called_operand
                        if isa(cf, LLVM.Function) && cf.name == "julia.gc_loaded" && v.operands[2] == inst
                            has_loaded = true
                            break
                        end
                    end
                    if isa(v, LLVM.BitCastInst)
                        for u2 in v.uses
                            v2 = u2.user
                            if isa(v2, LLVM.CallInst)
                                cf = v2.called_operand
                                if isa(cf, LLVM.Function) && cf.name == "julia.gc_loaded" && v2.operands[2] == v
                                    has_loaded = true
                                    break
                                end
                            end
                        end
                    end
                end
                if has_loaded
                    cache_memoryref_pointer_integers(inst)
                    inst.metadata["enzyme_nocache"] = MDNode(LLVM.Metadata[])
                end
            end
            if isa(inst, LLVM.CallInst)
                cf = inst.called_operand
                if isa(cf, LLVM.Function)
                    if cf.name == "llvm.julia.gc_preserve_begin"
                        has_use = false
                        for u2 in inst.uses
                            has_use = true
                            break
                        end
                        if !has_use
                            eraseInst(bb, inst)
                        end
                    end
                end
            end
        end
    end
    return
end

function permit_inlining!(f::LLVM.Function)
    for bb in f.blocks, inst in bb.instructions
        # remove illegal invariant.load and jtbaa_const invariants
        if isa(inst, LLVM.LoadInst)
            md = inst.metadata
            if haskey(md, LLVM.MD_tbaa)
                modified = LLVM.Metadata(
                    ccall(
                        (:EnzymeMakeNonConstTBAA, API.libEnzyme),
                        LLVM.API.LLVMMetadataRef,
                        (LLVM.API.LLVMMetadataRef,),
                        md[LLVM.MD_tbaa],
                    ),
                )
                setindex!(md, modified, LLVM.MD_tbaa)
            end
            if haskey(md, LLVM.MD_invariant_load)
                delete!(md, LLVM.MD_invariant_load)
            end
        end
    end
    return
end

function addNA(@nospecialize(inst::LLVM.Instruction), @nospecialize(node::LLVM.Metadata), MD::LLVM.MDKind)
    md = inst.metadata
    next = nothing
    if haskey(md, MD)
        next = LLVM.MDNode(Metadata[node, md[MD].operands...])
    else
        next = LLVM.MDNode(Metadata[node])
    end
    return setindex!(md, next, MD)
end

function addr13NoAlias(mod::LLVM.Module)
    ctx = mod.context
    dom = API.EnzymeAnonymousAliasScopeDomain("addr13", ctx)
    scope = API.EnzymeAnonymousAliasScope(dom, "na_addr13")
    aliasscope = noalias = scope
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.StoreInst)
            addNA(inst, noalias, LLVM.MD_noalias)
        elseif isa(inst, LLVM.CallInst)
            fn = inst.called_operand
            if isa(fn, LLVM.Function)
                name = fn.name
                if startswith(name, "llvm.memcpy") || startswith(name, "llvm.memmove")
                    addNA(inst, noalias, LLVM.MD_noalias)
                end
            end
        elseif isa(inst, LLVM.LoadInst)
            ty = inst.value_type
            if isa(ty, LLVM.PointerType)
                if ty.addrspace == 13
                    addNA(inst, aliasscope, LLVM.MD_alias_scope)
                end
            end
        end
    end
    return true
end

## given code like
#  %a = alloca
#  ...                          (nothing stores to, sets, or copies into %a)
#  memcpy(%dst, %a + off, n)
# the copy reads only undefined bytes, so dropping it is a refinement of the
# program. Julia 1.13 emits exactly this for the tracked slots of an aggregate
# whose roots travel separately: SROA leaves the slice of the data half that
# would hold them unwritten and unread, but still copies it into the destination.
# There is no type to be found for such bytes, so type analysis cannot type the
# copy; see #3589.
function erase_memcpy_from_undef!(mod::LLVM.Module)
    memcpy = LLVM.Intrinsic("llvm.memcpy")
    memmove = LLVM.Intrinsic("llvm.memmove")
    lstart = LLVM.Intrinsic("llvm.lifetime.start")
    lend = LLVM.Intrinsic("llvm.lifetime.end")
    for f in mod.functions
        isempty(f.blocks) && continue
        todel = Set{LLVM.Instruction}()
        for alloca in first(f.blocks).instructions
            isa(alloca, LLVM.AllocaInst) || continue
            todo = LLVM.Value[alloca]
            copies = LLVM.Instruction[]
            written = false
            while !isempty(todo) && !written
                cur = pop!(todo)
                for u in cur.uses
                    user = u.user
                    if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                            isa(user, LLVM.GetElementPtrInst)
                        push!(todo, user)
                        continue
                    end
                    if isa(user, LLVM.LoadInst)
                        continue
                    end
                    if isa(user, LLVM.CallInst) && isa(user.called_operand, LLVM.Function)
                        intr = user.called_operand.intrinsic
                        if intr == lstart || intr == lend
                            continue
                        end
                        if (intr == memcpy || intr == memmove) &&
                                user.operands[2] == cur && user.operands[1] != cur
                            push!(copies, user)
                            continue
                        end
                    end
                    # a store, memset, copy into it, or an escape
                    written = true
                    break
                end
            end
            if !written
                union!(todel, copies)
            end
        end
        for inst in todel
            eraseInst(inst.parent, inst)
        end
    end
    return
end

## given code like
#  % a = alloca
#  ...
#  memref(cast(%a), %b, constant size == sizeof(a))
#
#  turn this into load/store, as this is more
#  amenable to caching analysis infrastructure
function memcpy_alloca_to_loadstore(mod::LLVM.Module, world::UInt, enzyme_ctx::EnzymeContext)
    dl = mod.datalayout
    ctx = mod.context
    seen = TypeTreeTable()
    for f in mod.functions
        if length(f.blocks) != 0
            bb = first(f.blocks)
            todel = Set{LLVM.Instruction}()
            for alloca in bb.instructions
                if !isa(alloca, LLVM.AllocaInst)
                    continue
                end
                todo = Tuple{LLVM.Instruction, LLVM.Value}[(alloca, alloca)]
                copy = nothing
                legal = true
                elty = alloca.allocated_type
                lifetimestarts = LLVM.Instruction[]
                while length(todo) > 0
                    cur, prev = pop!(todo)
                    if isa(cur, LLVM.AllocaInst) ||
                            isa(cur, LLVM.AddrSpaceCastInst) ||
                            isa(cur, LLVM.BitCastInst)
                        for u in cur.users
                            push!(todo, (u, cur))
                        end
                        continue
                    end
                    if isa(cur, LLVM.CallInst) &&
                            isa(cur.called_operand, LLVM.Function)
                        intr = cur.called_operand.intrinsic
                        if intr == LLVM.Intrinsic("llvm.lifetime.start")
                            push!(lifetimestarts, cur)
                            continue
                        end
                        if intr == LLVM.Intrinsic("llvm.lifetime.end")
                            continue
                        end
                        if intr == LLVM.Intrinsic("llvm.memcpy")
                            sz = cur.operands[3]
                            if cur.operands[1] == prev &&
                                    isa(sz, LLVM.ConstantInt) &&
                                    convert(Int, sz) == LLVM.storage_size(dl, elty)
                                if copy === nothing || copy == cur
                                    copy = cur
                                    continue
                                end
                            end
                        end
                    end

                    # read only insts of arg, don't matter
                    if isa(cur, LLVM.LoadInst)
                        continue
                    end
                    if isa(cur, LLVM.CallInst) &&
                            isa(cur.called_operand, LLVM.Function)
                        legalc = true
                        for (i, ci) in enumerate(cur.arguments)
                            if ci == prev
                                nocapture = false
                                readonly = false
                                # vararg arguments have no callee parameter attributes
                                callee_params = cur.called_operand.parameters
                                param_attrs = if i <= length(callee_params)
                                    collect(cur.called_operand.parameter_attributes[i])
                                else
                                    LLVM.Attribute[]
                                end
                                for a in param_attrs
                                    if a.kind == READONLY_ATTR_KIND
                                        readonly = true
                                    end
                                    if a.kind == READNONE_ATTR_KIND
                                        readonly = true
                                    end
                                    if a.kind == NOCAPTURE_ATTR_KIND
                                        nocapture = true
                                    end
                                end
                                if !nocapture || !readonly
                                    legalc = false
                                    break
                                end
                            end
                        end
                        if legalc
                            continue
                        end
                    end

                    legal = false
                    break
                end

                if legal && copy !== nothing
                    B = LLVM.IRBuilder()
                    position!(B, LLVM.before(copy))
                    dst = copy.operands[1]
                    src = copy.operands[2]
                    dst0 = bitcast!(
                        B,
                        dst,
                        LLVM.PointerType(LLVM.IntType(8), dst.value_type.addrspace),
                    )

                    dst =
                        bitcast!(B, dst, LLVM.PointerType(elty, dst.value_type.addrspace))
                    src =
                        bitcast!(B, src, LLVM.PointerType(elty, src.value_type.addrspace))

                    src = load!(B, elty, src)

                    T_jlvalue = LLVM.StructType(LLVMType[])
                    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

                    legal, source_typ, byref = abs_typeof(src, enzyme_ctx)
                    codegen_typ = src.value_type
                    if legal
                        if codegen_typ isa LLVM.PointerType || codegen_typ isa LLVM.IntegerType
                        else
                            @assert byref == GPUCompiler.BITS_VALUE
                            source_typ
                        end

                        ec = typetree_in_world(world, source_typ, ctx, string(dl), seen)
                        if byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF
                            ec = copy(ec)
                            merge!(ec, TypeTree(API.DT_Pointer, ctx))
                            only!(ec, -1)
                        end
                        src.metadata["enzyme_type"] = to_md(ec, ctx)
                        if EmitTypeNames[]
                            src.metadata["enzymejl_source_type_$(source_typ)"] = MDNode(LLVM.Metadata[])
                        end
                        src.metadata["enzymejl_byref_$(byref)"] = MDNode(LLVM.Metadata[])
                        mark_load_dereferenceable!(src, source_typ, byref)

                        @static if VERSION < v"1.11-"
                        else
                            legal2, obj = absint(src, enzyme_ctx)
                            if legal2 && is_memory_instance(unbind(obj))
                                src.metadata["nonnull"] = MDNode(LLVM.Metadata[])
                            end
                        end

                    elseif codegen_typ == T_prjlvalue
                        src.metadata["enzyme_type"] =
                            to_md(typetree_in_world(world, Ptr{Cvoid}, ctx, dl, seen), ctx)
                    end
                    FT = LLVM.FunctionType(
                        LLVM.VoidType(),
                        [LLVM.IntType(64), dst0.value_type],
                    )
                    lifetimestart, _ = get_function!(mod, LLVM.overloaded_name(LLVM.Intrinsic("llvm.lifetime.start"), [dst0.value_type]), FT)
                    call!(
                        B,
                        FT,
                        lifetimestart,
                        LLVM.Value[LLVM.ConstantInt(Int64(LLVM.storage_size(dl, elty))), dst0],
                    )
                    store!(B, src, dst)
                    push!(todel, copy)
                end
                for lt in lifetimestarts
                    push!(todel, lt)
                end
            end
            for inst in todel
                eraseInst(inst.parent, inst)
            end
        end
    end
    return
end

# Split a memcpy into an sret with jlvaluet into individual load/stores.
# By default only an sret that mixes GC-tracked pointers with other data is split: the
# post-AD calling-convention fixup needs exactly those. Before AD, `all_tracked = true`
# also splits an sret of only tracked pointers (a `NamedTuple` or `Tuple` of arrays), so
# that each copied pointer is a store the mixed-activity handler sees.
# That pre-AD split copies the pointer fields on every Julia version. Only the post-AD
# fixup omits them on Julia 1.12+, where their roots are handled separately.
function memcpy_sret_split!(mod::LLVM.Module; all_tracked::Bool = false)
    dl = mod.datalayout
    ctx = mod.context
    sretkind = (
        if LLVM.version().major >= 12
            LLVM.TypeAttribute(:sret, LLVM.Int32Type())
        else
            LLVM.EnumAttribute(:sret)
        end
    ).kind
    for f in mod.functions

        if length(f.blocks) == 0
            continue
        end
        if length(f.parameters) == 0
            continue
        end
        sty = nothing
        for attr in collect(f.parameter_attributes[1])
            if attr.kind == sretkind
                sty = attr.value
                break
            end
        end
        if sty === nothing
            continue
        end
        tracked = CountTrackedPointers(sty)
        if tracked.count == 0 || (tracked.all && !all_tracked)
            continue
        end
        todo = LLVM.CallInst[]
        for bb in f.blocks
            for cur in bb.instructions
                if isa(cur, LLVM.CallInst) &&
                        isa(cur.called_operand, LLVM.Function)
                    intr = cur.called_operand.intrinsic
                    if intr == LLVM.Intrinsic("llvm.memcpy")
                        dst, _ = get_base_and_offset(cur.operands[1]; offsetAllowed = false)
                        if isa(dst, LLVM.Argument) && f.parameters[1] == dst
                            if isa(cur.operands[3], LLVM.ConstantInt) && LLVM.storage_size(dl, sty) == convert(Int, cur.operands[3])
                                push!(todo, cur)
                            end
                        end
                    end
                end
            end
        end
        for cur in todo
            B = IRBuilder()
            position!(B, LLVM.before(cur))
            dst, _ = get_base_and_offset(cur.operands[1]; offsetAllowed = false)
            src, _ = get_base_and_offset(cur.operands[2]; offsetAllowed = false)
            if !LLVM.isopaque(dst.value_type) && dst.value_type.element_type != src.value_type.element_type
                src = pointercast!(B, src, LLVM.PointerType(dst.value_type.element_type, src.value_type.addrspace), "memcpy_sret_split_pointercast")
            end
            copy_struct_into!(B, sty, dst, src, all_tracked || VERSION < v"1.12")
            LLVM.API.LLVMInstructionEraseFromParent(cur)
        end
    end
    return
end

# Root object of a derived pointer for the purpose of deciding whether an addrspace(11) phi
# needs an addrspace(10) parent. Unlike `get_base_and_offset` this also looks through GEPs with
# non-constant indices: LLVM 20 (Julia 1.13) strength-reduces loop indices into pointer
# induction variables (`%invariant.gep = gep i8 %arg, -8` + `gep double %invariant.gep, %iv`),
# and the byte offset is irrelevant for the rooting question.
function nodecayed_root(@nospecialize(v::LLVM.Value))::LLVM.Value
    base, _ = get_base_and_offset(v)
    while isa(base, LLVM.GetElementPtrInst)
        base, _ = get_base_and_offset(base.operands[1])
    end
    return base
end

# True if every value flowing into the addrspace(11) phi `inst` (through phis and selects) is
# derived from an addrspace(11) argument, undef or poison. Such a phi has no addrspace(10)
# object to root it and is left untouched by `nodecayed_phis!`.
function nodecayed_all_args(inst::LLVM.PHIInst)::Bool
    addrtodo = LLVM.Value[inst]
    seen = Set{LLVM.Value}()
    while length(addrtodo) != 0
        v = pop!(addrtodo)
        base = nodecayed_root(v)
        if in(base, seen)
            continue
        end
        push!(seen, base)
        if isa(base, LLVM.Argument) && base.value_type.addrspace == 11
            continue
        end
        if isa(base, LLVM.PHIInst)
            for (iv, _) in base.incoming
                push!(addrtodo, iv)
            end
            continue
        end
        if isa(base, LLVM.SelectInst)
            push!(addrtodo, base.operands[2])
            push!(addrtodo, base.operands[3])
            continue
        end
        undeforpoison = isa(base, LLVM.UndefValue)
        undeforpoison |= isa(base, LLVM.PoisonValue)
        if undeforpoison
            # undef/poison incomings impose no GC constraint
            continue
        end
        return false
    end
    return true
end

"""
    unfold_argument_phi_loads!(f::LLVM.Function) -> Bool

Turn a load through an addrspace(11) phi that merges an addrspace(11) argument with a
pointer into an object back into a phi of loads.

`nodecayed_phis!` gives an addrspace(11) phi a phi of addrspace(10) parent objects and
a phi of offsets. An addrspace(11) argument has no parent in the function (its caller
roots it), so a phi that merges one with a pointer into an object cannot be rewritten:
`nodecayed_getparent` fails on the argument, and reaching the phi throws "Enzyme could
not determine how to keep a pointer phi in this function rooted". Unless every incoming
value is an argument (`nodecayed_all_args`), which leaves the phi alone.

LLVM's instcombine makes such phis out of a phi of loads: it folds
`phi [load (gep %obj, k)], [load (gep %arg, k)]` into `load (gep (phi [%obj, %arg]), k)`.
Enzyme's augmented primal of Base's `copyto!(dest, doffs, src::SubArray, soffs, n)` is
one: `unalias(dest, src)` is either `src`, the argument, or `unaliascopy(src)`, held in
a stack slot that Enzyme moved to the heap, and the loop reads one field of whichever it
is. First-order Reverse never runs `nodecayed_phis!` on an augmented primal, but a
nested derivative (Forward over Reverse of `append!(v, @view w[i:end])`) imports it and
runs the pre-AD pipeline on it again.

Only rewrite what is certainly equivalent: a phi in address space 11 with an argument
among the roots of its incoming values and another root besides, whose users are all
non-volatile, non-atomic or unordered loads from it, directly or through bitcasts and
GEPs with constant indices, in its own block, that no memory write there precedes. Each
load is re-created, with its address, before the terminator of every predecessor, and
the loaded values are merged by a phi. An edge whose predecessor also branches elsewhere
is split first, so no load becomes speculative. Any other such phi still throws when
reached.
"""
function unfold_argument_phi_loads!(f::LLVM.Function)::Bool
    # Splitting an edge rebuilds the phis of the block it leads to, so the candidates
    # are collected again afterwards.
    edges = Tuple{LLVM.BasicBlock, LLVM.BasicBlock}[]
    any_candidate = false
    for bb in f.blocks
        for phi in bb.instructions
            isa(phi, LLVM.PHIInst) || break
            argument_phi_loads(phi) === nothing && continue
            any_candidate = true
            for (_, pred) in phi.incoming
                length(collect(pred.terminator.successors)) == 1 && continue
                (pred, bb) in edges || push!(edges, (pred, bb))
            end
        end
    end
    any_candidate || return false
    changed = false
    builder = IRBuilder()
    for (pred, succ) in edges
        split_edge!(builder, pred, succ)
    end
    for bb in f.blocks
        phis = LLVM.PHIInst[]
        for phi in bb.instructions
            isa(phi, LLVM.PHIInst) || break
            push!(phis, phi)
        end
        for phi in phis
            loads = argument_phi_loads(phi)
            loads === nothing && continue
            single = true
            for (_, pred) in phi.incoming
                single &= length(collect(pred.terminator.successors)) == 1
            end
            single || continue
            unfold_argument_phi_loads!(builder, phi, loads)
            changed = true
        end
    end
    dispose(builder)
    return changed
end

# Whether the addrspace(11) phi `phi` merges a value derived from an addrspace(11)
# argument with one that is not, so that `nodecayed_phis!` has no parent to give it.
function nodecayed_mixed_argument_phi(phi::LLVM.PHIInst)::Bool
    ty = phi.value_type
    (isa(ty, LLVM.PointerType) && ty.addrspace == Derived) || return false
    for (v, _) in phi.incoming
        base = nodecayed_root(v)
        if isa(base, LLVM.Argument) && base.value_type.addrspace == Derived
            return !nodecayed_all_args(phi)
        end
    end
    return false
end

# Whether `ld` is a load that may be moved to the end of a predecessor of its block.
function unfoldable_phi_load(ld::LLVM.LoadInst)::Bool
    Bool(LLVM.API.LLVMGetVolatile(ld)) && return false
    ord = LLVM.API.LLVMGetOrdering(ld)
    return ord == LLVM.API.LLVMAtomicOrderingNotAtomic ||
        ord == LLVM.API.LLVMAtomicOrderingUnordered
end

# The loads that `unfold_argument_phi_loads!` moves out of the block of `phi`, each with
# the bitcasts and constant GEPs that compute its address from `phi` (innermost first),
# or `nothing`.
function argument_phi_loads(phi::LLVM.PHIInst)::Union{Nothing, Vector{Tuple{LLVM.LoadInst, Vector{LLVM.Instruction}}}}
    nodecayed_mixed_argument_phi(phi) || return nothing
    bb = phi.parent
    for (_, pred) in phi.incoming
        term = pred.terminator
        (isa(term, LLVM.BrInst) || isa(term, LLVM.SwitchInst)) || return nothing
        count(Base.Fix2(==, bb), term.successors) == 1 || return nothing
    end
    loads = Tuple{LLVM.LoadInst, Vector{LLVM.Instruction}}[]
    todo = Tuple{LLVM.Value, Vector{LLVM.Instruction}}[(phi, LLVM.Instruction[])]
    while !isempty(todo)
        addr, chain = pop!(todo)
        for u in addr.uses
            user = u.user
            (isa(user, LLVM.Instruction) && user.parent == bb) || return nothing
            if isa(user, LLVM.LoadInst)
                unfoldable_phi_load(user) || return nothing
                push!(loads, (user, chain))
            elseif isa(user, LLVM.BitCastInst)
                push!(todo, (user, LLVM.Instruction[chain..., user]))
            elseif isa(user, LLVM.GetElementPtrInst) && user.operands[1] == addr
                for op in user.operands[2:end]
                    isa(op, LLVM.ConstantInt) || return nothing
                end
                push!(todo, (user, LLVM.Instruction[chain..., user]))
            else
                return nothing
            end
        end
    end
    isempty(loads) && return nothing
    # A write before one of the loads could change what it reads.
    remaining = length(loads)
    for inst in bb.instructions
        isa(inst, LLVM.PHIInst) && continue
        if isa(inst, LLVM.LoadInst) && any(Base.Fix2(==, inst) ∘ first, loads)
            remaining -= 1
            remaining == 0 && break
        elseif root_path_may_write(inst)
            return nothing
        end
    end
    return loads
end

# Clone `inst` before the builder's position, with `ptr` as its pointer operand.
function clone_with_pointer!(builder::LLVM.IRBuilder, inst::LLVM.Instruction, @nospecialize(ptr::LLVM.Value))::LLVM.Instruction
    clone = LLVM.Instruction(LLVM.API.LLVMInstructionClone(inst))
    LLVM.API.LLVMSetOperand(clone, 0, ptr)
    LLVM.API.LLVMInsertIntoBuilderWithName(builder, clone, inst.name)
    return clone
end

function unfold_argument_phi_loads!(builder::LLVM.IRBuilder, phi::LLVM.PHIInst, loads::Vector{Tuple{LLVM.LoadInst, Vector{LLVM.Instruction}}})
    # The address computations shared by several loads are cloned once per predecessor.
    clones = Dict{Tuple{LLVM.Instruction, LLVM.BasicBlock}, LLVM.Instruction}()
    for (ld, chain) in loads
        position!(builder, LLVM.before(phi))
        merged = phi!(builder, ld.value_type, ld.name)
        for (v, pred) in phi.incoming
            position!(builder, LLVM.before(pred.terminator))
            addr = v
            for inst in chain
                key = (inst, pred)
                if !haskey(clones, key)
                    clones[key] = clone_with_pointer!(builder, inst, addr)
                end
                addr = clones[key]
            end
            push!(merged.incoming, (clone_with_pointer!(builder, ld, addr), pred))
        end
        # Memory metadata such as `!tbaa` is not allowed on a phi.
        for k in ("enzyme_type", "enzyme_inactive", "enzyme_active")
            haskey(ld.metadata, k) && (merged.metadata[k] = ld.metadata[k])
        end
        replace_uses!(ld, merged)
        LLVM.erase!(ld)
    end
    # The address computations, now unused, users before definitions.
    dead = LLVM.Instruction[]
    for (_, chain) in loads, inst in chain
        inst in dead || push!(dead, inst)
    end
    while !isempty(dead)
        remaining = LLVM.Instruction[]
        for inst in dead
            if isempty(inst.uses)
                LLVM.erase!(inst)
            else
                push!(remaining, inst)
            end
        end
        @assert length(remaining) < length(dead)
        dead = remaining
    end
    @assert isempty(phi.uses)
    LLVM.erase!(phi)
    return
end

# If there is a phi node of a decayed value, Enzyme may need to cache it
# Here we force all decayed pointer phis to first addrspace from 10
# State shared by every level of the `nodecayed_getparent` recursion below. It used to be
# captured by a closure defined inside `nodecayed_phis!`; as a closure its type embedded all of
# these and the ~200-line recursive body cost ~0.6 s to compile on the first `autodiff` of a
# session, outside of anything the precompile workload can cover.
struct NoDecayedPhiState
    addr::Int
    offty::LLVM.IntegerType
    ctx::LLVM.Context
    f::LLVM.Function
    inst::LLVM.PHIInst          # the phi whose incoming value is being analyzed (for error reporting)
    v0::LLVM.Value              # that incoming value, before any rewriting (for error reporting)
    nextvs::Dict{LLVM.PHIInst, LLVM.PHIInst}
    goffsets::Dict{LLVM.PHIInst, LLVM.PHIInst}
    phicache::Dict{LLVM.PHIInst, Tuple{LLVM.PHIInst, LLVM.PHIInst}}
end

# Walk `v` back to its addrspace-10 parent, returning (parent, byte offset, hasload).
function nodecayed_getparent(st::NoDecayedPhiState, b::LLVM.IRBuilder, @nospecialize(v::LLVM.Value), @nospecialize(offset::LLVM.Value), hasload::Bool)::Tuple{LLVM.Value, LLVM.Value, Bool}
    if st.addr == 11 && v.value_type.addrspace == 10
        return v, offset, hasload
    end
    if st.addr == 13 && hasload && v.value_type.addrspace == 10
        return v, offset, hasload
    end

    if st.addr == 13  && !hasload
        if isa(v, LLVM.LoadInst)
            v2, o2, hl2 = nodecayed_getparent(st, b, v.operands[1], LLVM.ConstantInt(st.offty, 0), true)
            @static if VERSION < v"1.11-"
            else
                @assert offset == LLVM.ConstantInt(st.offty, 0)
                return v2, o2, true
            end

            rhs = LLVM.ConstantInt(st.offty, 0)
            if o2 != rhs
                msg = sprint() do io::IO
                    println(
                        io,
                        "Enzyme internal error addr13 load doesn't keep offset 0",
                    )
                    println(io, "v=", string(v))
                    println(io, "v2=", string(v2))
                    println(io, "o2=", string(o2))
                    println(io, "hl2=", string(hl2))
                    println(io, "st.offty=", string(st.offty))
                    println(io, "rhs=", string(rhs))
                end
                throw(AssertionError(msg))
            end
            return v2, offset, true
        end
        if isa(v, LLVM.CallInst)
            cf = v.called_operand
            if isa(cf, LLVM.Function) && cf.name == "julia.gc_loaded"
                ld = v.operands[2]
                ld0, o0, ol0 = nodecayed_getparent(st, b, ld, LLVM.ConstantInt(st.offty, 0), hasload)
                v2 = ld0
                # v2, o2, hl2 = nodecayed_getparent(st, b, operands(ld)[1], LLVM.ConstantInt(st.offty, 0), true)

                rhs = LLVM.ConstantInt(st.offty, sizeof(Int))
                o2 = o0

                base_2, off_2 = get_base_and_offset(v2)
                base_1, off_1 = get_base_and_offset(v.operands[1])

                if o2 == rhs && base_1 == base_2 && off_1 == off_2
                    return v.operands[1], offset, true
                end

                pty = TypeTree(API.DT_Pointer, ld.context)
                only!(pty, -1)
                rhs = ptrtoint!(b, get_memory_data(b, v.operands[1]), st.offty)
                rhs.metadata["enzyme_type"] = to_md(pty, st.ctx)
                lhs = ptrtoint!(b, v.operands[2], st.offty)
                rhs.metadata["enzyme_type"] = to_md(pty, st.ctx)
                off2 = nuwsub!(b, lhs, rhs)
                ity = TypeTree(API.DT_Integer, ld.context)
                only!(ity, -1)
                off2.metadata["enzyme_type"] = to_md(ity, st.ctx)
                add = nuwadd!(b, offset, off2)
                add.metadata["enzyme_type"] = to_md(ity, st.ctx)
                return v.operands[1], add, true
            end
        end
    end

    if st.addr == 13 && isa(v, LLVM.ConstantExpr)
        if v.opcode == LLVM.Opcode.AddrSpaceCast
            v2 = v.operands[1]
            if v2.value_type.addrspace == 0
                if st.addr == 13 && isa(v, LLVM.ConstantExpr)
                    PT = if LLVM.isopaque(v.value_type)
                        LLVM.PointerType(10)
                    else
                        LLVM.PointerType(v.value_type.element_type, 10)
                    end
                    v2 = const_addrspacecast(
                        v.operands[1],
                        PT
                    )
                    return v2, offset, hasload
                end
            end
        end
    end

    if isa(v, LLVM.ConstantExpr)
        if v.opcode == LLVM.Opcode.AddrSpaceCast
            v2 = v.operands[1]
            if v2.value_type.addrspace == 10
                return v2, offset, hasload
            end
            if v2.value_type.addrspace == 0
                if st.addr == 11
                    PT = if LLVM.isopaque(v.value_type)
                        LLVM.PointerType(10)
                    else
                        LLVM.PointerType(v.value_type.element_type, 10)
                    end
                    v2 = const_addrspacecast(
                        v2,
                        PT
                    )
                    return v2, offset, hasload
                end
            end
            if LLVM.isnull(v2)
                PT = if LLVM.isopaque(v.value_type)
                    LLVM.PointerType(10)
                else
                    LLVM.PointerType(v.value_type.element_type, 10)
                end
                v2 = const_addrspacecast(
                    v2,
                    PT
                )
                return v2, offset, hasload
            end
        end
        if v.opcode == LLVM.Opcode.BitCast
            preop = v.operands[1]
            while isa(preop, LLVM.ConstantExpr) && preop.opcode == LLVM.Opcode.BitCast
                preop = preop.operands[1]
            end
            v2, offset, skipload =
                nodecayed_getparent(st, b, preop, offset, hasload)
            v2 = const_bitcast(
                v2,
                LLVM.PointerType(
                    v.value_type.element_type,
                    v2.value_type.addrspace,
                ),
            )
            @assert v2.value_type.element_type == v.value_type.element_type
            return v2, offset, skipload
        end

        if v.opcode == LLVM.Opcode.GetElementPtr
            v2, offset, skipload =
                nodecayed_getparent(st, b, v.operands[1], offset, hasload)
            offset = const_add(
                offset,
                API.EnzymeComputeByteOffsetOfGEP(b, v, st.offty),
            )
            if !LLVM.isopaque(v.value_type)
                v2 = const_bitcast(
                    v2,
                    LLVM.PointerType(
                        v.value_type.element_type,
                        v2.value_type.addrspace,
                    ),
                )
                @assert v2.value_type.element_type == v.value_type.element_type
            end
            return v2, offset, skipload
        end

    end

    if isa(v, LLVM.AddrSpaceCastInst)
        if v.operands[1].value_type.addrspace == 0
            PT = if LLVM.isopaque(v.value_type)
                LLVM.PointerType(10)
            else
                LLVM.PointerType(v.value_type.element_type, 10)
            end
            v2 = addrspacecast!(
                b,
                v.operands[1],
                PT
            )
            return v2, offset, hasload
        end
        nv, noffset, nhasload =
            nodecayed_getparent(st, b, v.operands[1], offset, hasload)
        if !isopaque(nv.value_type) && nv.value_type.element_type != v.value_type.element_type
            nv = bitcast!(
                b,
                nv,
                LLVM.PointerType(
                    v.value_type.element_type,
                    nv.value_type.addrspace,
                ),
            )
        end
        return nv, noffset, nhasload
    end

    if isa(v, LLVM.BitCastInst)
        preop = v.operands[1]
        while isa(preop, LLVM.BitCastInst)
            preop = preop.operands[1]
        end
        v2, offset, skipload =
            nodecayed_getparent(st, b, preop, offset, hasload)
        v2 = bitcast!(
            b,
            v2,
            LLVM.PointerType(
                v.value_type.element_type,
                v2.value_type.addrspace,
            ),
        )
        @assert v2.value_type.element_type == v.value_type.element_type
        return v2, offset, skipload
    end

    if isa(v, LLVM.GetElementPtrInst) && all(
            x -> (isa(x, LLVM.ConstantInt) && LLVM.isnull(x)),
            v.operands[2:end],
        )
        v2, offset, skipload =
            nodecayed_getparent(st, b, v.operands[1], offset, hasload)
        if !LLVM.isopaque(v.value_type)
            v2 = bitcast!(
                b,
                v2,
                LLVM.PointerType(
                    v.value_type.element_type,
                    v2.value_type.addrspace,
                ),
            )
        end
        @assert v2.value_type.element_type == v.value_type.element_type
        return v2, offset, skipload
    end

    if isa(v, LLVM.GetElementPtrInst)
        v2, offset, skipload =
            nodecayed_getparent(st, b, v.operands[1], offset, hasload)
        offset = nuwadd!(
            b,
            offset,
            API.EnzymeComputeByteOffsetOfGEP(b, v, st.offty),
        )
        if !LLVM.isopaque(v2.value_type)
            v2 = bitcast!(
                b,
                v2,
                LLVM.PointerType(
                    v.value_type.element_type,
                    v2.value_type.addrspace,
                ),
            )
            @assert v2.value_type.element_type == v.value_type.element_type
        end
        return v2, offset, skipload
    end

    undeforpoison = isa(v, LLVM.UndefValue)
    undeforpoison |= isa(v, LLVM.PoisonValue)
    if undeforpoison
        PT = if LLVM.isopaque(v.value_type)
            LLVM.PointerType(10)
        else
            LLVM.PointerType(v.value_type.element_type, 10)
        end
        return LLVM.UndefValue(PT), offset, st.addr == 13
    end

    # A null pointer derives from no object: its base is null. Such a phi arm appears where
    # the code tests a pointer it loaded from a `julia.constgv` slot, which LLVM cannot fold
    # while the slot is a declaration (see `make_slots_symbolic!`).
    if isa(v, LLVM.PointerNull)
        PT = if LLVM.isopaque(v.value_type)
            LLVM.PointerType(10)
        else
            LLVM.PointerType(v.value_type.element_type, 10)
        end
        return LLVM.null(PT), offset, st.addr == 13
    end

    if isa(v, LLVM.PHIInst) && !hasload && haskey(st.goffsets, v)
        offset = nuwadd!(b, offset, st.goffsets[v])
        nv = st.nextvs[v]
        return nv, offset, st.addr == 13
    end

    @static if VERSION < v"1.11-"
    else
        if st.addr == 13 && isa(v, LLVM.PHIInst)
            if haskey(st.phicache, v)
                return (st.phicache[v]..., hasload)
            end
            vs = Union{LLVM.Value, Nothing}[]
            offs = Union{LLVM.Value, Nothing}[]
            blks = LLVM.BasicBlock[]

            sPT = if !LLVM.isopaque(v.value_type)
                LLVM.PointerType(v.value_type.element_type, 10)
            else
                LLVM.PointerType(10)
            end
            vphi, ophi = @dispose B = LLVM.IRBuilder() begin
                position!(B, LLVM.before(v))
                (
                    phi!(B, sPT, "nondecay.vphi." * v.name),
                    phi!(B, offset.value_type, "nondecay.ophi" * v.name),
                )
            end
            st.phicache[v] = (vphi, ophi)

            bbcache = Dict{BasicBlock, Value}()
            for (vt, bb) in v.incoming
                v2, o2 = @dispose b2 = IRBuilder() begin
                    position!(b2, LLVM.before(bb.terminator))
                    v2, o2, hl2 = nodecayed_getparent(st, b2, vt, offset, hasload)
                    if v2.value_type != sPT
                        if haskey(bbcache, bb)
                            v2 = bbcache[bb]
                        else
                            v2 = bitcast!(b2, v2, sPT)
                            bbcache[bb] = v2
                        end
                    end
                    (v2, o2)
                end

                @assert sPT == v2.value_type
                push!(vs, v2)
                @assert offset.value_type == o2.value_type
                push!(offs, o2)
                push!(blks, bb)
            end

            append!(ophi.incoming, collect(zip(offs, blks)))

            append!(vphi.incoming, collect(zip(vs, blks)))

            return vphi, ophi, hasload
        end
    end

    if isa(v, LLVM.SelectInst)
        lhs_v, lhs_offset, lhs_skipload =
            nodecayed_getparent(st, b, v.operands[2], offset, hasload)
        rhs_v, rhs_offset, rhs_skipload =
            nodecayed_getparent(st, b, v.operands[3], offset, hasload)
        if lhs_v.value_type != rhs_v.value_type ||
                lhs_offset.value_type != rhs_offset.value_type ||
                lhs_skipload != rhs_skipload
            msg = sprint() do io
                println(
                    io,
                    "Could not analyze [select] garbage collection behavior of",
                )
                println(io, " st.v0: ", string(st.v0))
                println(io, " v: ", string(v))
                println(io, " offset: ", string(offset))
                println(io, " hasload: ", string(hasload))
                println(io, " lhs_v", lhs_v)
                println(io, " rhs_v", rhs_v)
                println(io, " lhs_offset", lhs_offset)
                println(io, " rhs_offset", rhs_offset)
                println(io, " lhs_skipload", lhs_skipload)
                println(io, " rhs_skipload", rhs_skipload)
            end
            bt = GPUCompiler.backtrace(st.inst)
            mi, _ = Compiler.enzyme_custom_extract_mi(st.f, false) #=error=#
            world = enzyme_world_if_active()
            if mi !== nothing && world !== nothing
                throw(EnzymeInternalError{Core.MethodInstance, UInt}(msg, string(st.f), bt, mi, world))
            else
                throw(EnzymeInternalError{Nothing, Nothing}(msg, string(st.f), bt, mi, nothing))
            end
        end
        return select!(b, v.operands[1], lhs_v, rhs_v),
            select!(b, v.operands[1], lhs_offset, rhs_offset),
            lhs_skipload
    end

    msg = sprint() do io
        println(io, "Could not analyze garbage collection behavior of")
        println(io, " st.inst: ", string(st.inst))
        println(io, " st.v0: ", string(st.v0))
        println(io, " v: ", string(v))
        println(io, " offset: ", string(offset))
        println(io, " hasload: ", string(hasload))
    end
    bt = GPUCompiler.backtrace(st.inst)
    mi, _ = Compiler.enzyme_custom_extract_mi(st.f, false) #=error=#
    world = enzyme_world_if_active()
    if mi !== nothing && world !== nothing
        throw(EnzymeInternalError{Core.MethodInstance, UInt}(msg, string(st.f), bt, mi, world))
    else
        throw(EnzymeInternalError{Nothing, Nothing}(msg, string(st.f), bt, mi, nothing))
    end
end

# Message of the runtime error thrown on reaching a phi `nodecayed_phis!` could
# not handle, carrying the compile-time diagnosis.
function nodecayed_runtime_message(err::EnzymeInternalError)::String
    return sprint() do io
        println(io, "Enzyme could not determine how to keep a pointer phi in this function rooted for")
        println(io, "the garbage collector, so reaching the phi throws this error. Please open an issue")
        println(io, "with the code to reproduce on github.com/EnzymeAD/Enzyme.jl.")
        println(io)
        print(io, err.msg)
        if err.bt !== nothing && !isempty(err.bt)
            println(io, "Location of the phi:")
            Base.show_backtrace(io, err.bt)
            println(io)
        end
        if VERBOSE_ERRORS[] && err.ir !== nothing
            println(io, "Function at the time of the failure:")
            print(io, err.ir)
        end
    end
end

# Attributes a function loses once it may throw, on itself and its calls: the
# throw unwinds, does not return, and allocates and writes the exception.
const THROWING_BODY_DROPPED_ATTRS = (
    :nounwind, :willreturn, :speculatable, :readnone, :readonly, :writeonly,
    :argmemonly, :inaccessiblememonly, :inaccessiblemem_or_argmemonly, :memory,
)

function drop_throwing_body_attrs!(attrs)
    for kind in THROWING_BODY_DROPPED_ATTRS
        delete!(attrs, kind)
    end
    return nothing
end

# Rebuild `phi` without its incoming values from blocks in `dropped`.
function drop_phi_incoming!(phi::LLVM.PHIInst, dropped::Set{LLVM.BasicBlock})
    kept = Tuple{LLVM.Value, LLVM.BasicBlock}[(v, bb) for (v, bb) in phi.incoming if !in(bb, dropped)]
    if length(kept) == length(phi.incoming)
        return nothing
    end
    B = LLVM.IRBuilder()
    position!(B, LLVM.before(phi))
    nphi = phi!(B, phi.value_type, phi.name)
    append!(nphi.incoming, kept)
    replace_uses!(phi, nphi)
    LLVM.API.LLVMInstructionEraseFromParent(phi)
    return nothing
end

# `nodecayed_phis!` could not give the phis in `failed` an addrspace(10)
# parent. Such a phi is only a problem if it is reached, so make reaching its
# block throw the compile-time diagnosis instead: keep the block's phis, then
# throw, and remove the code that this leaves unreachable, which holds every
# use of the phi.
function nodecayed_cut_failed!(f::LLVM.Function, failed::Vector{Tuple{LLVM.PHIInst, EnzymeInternalError}})
    # Placeholder phis whose rewrite failed have no incoming values yet.
    preds = Dict{LLVM.BasicBlock, Vector{LLVM.BasicBlock}}(bb => LLVM.BasicBlock[] for bb in f.blocks)
    for bb in f.blocks, succ in bb.successors
        push!(preds[succ], bb)
    end
    for bb in f.blocks, inst in collect(bb.instructions)
        isa(inst, LLVM.PHIInst) || break
        if isempty(inst.incoming)
            append!(inst.incoming, [(LLVM.UndefValue(inst.value_type), pb) for pb in preds[bb]])
        end
    end

    cut = Dict{LLVM.BasicBlock, EnzymeInternalError}()
    for (inst, err) in failed
        get!(cut, inst.parent, err)
    end

    dead = LLVM.Instruction[]
    for (bb, err) in cut
        rest = LLVM.Instruction[inst for inst in bb.instructions if !isa(inst, LLVM.PHIInst)]
        append!(dead, rest)
        B = LLVM.IRBuilder()
        position!(B, LLVM.before(first(rest)))
        msg = nodecayed_runtime_message(err)
        if err.mi !== nothing && err.world !== nothing
            emit_error(B, nothing, (msg, err.mi, err.world), EnzymeRuntimeExceptionMI)
        else
            emit_error(B, nothing, msg)
        end
        unreachable!(B)
    end

    # The cut blocks no longer branch anywhere.
    reachable = Set{LLVM.BasicBlock}()
    worklist = LLVM.BasicBlock[f.entry]
    while !isempty(worklist)
        bb = pop!(worklist)
        in(bb, reachable) && continue
        push!(reachable, bb)
        haskey(cut, bb) && continue
        append!(worklist, collect(bb.successors))
    end
    deadblocks = LLVM.BasicBlock[bb for bb in f.blocks if !in(bb, reachable)]
    dropped = Set{LLVM.BasicBlock}(deadblocks)
    union!(dropped, keys(cut))
    for bb in reachable
        haskey(cut, bb) && continue
        for inst in collect(bb.instructions)
            isa(inst, LLVM.PHIInst) || break
            drop_phi_incoming!(inst, dropped)
        end
    end

    # Erase the dead code, users before definitions. Its only cycles go through
    # phis, which are not of token type.
    for bb in deadblocks, inst in bb.instructions
        push!(dead, inst)
    end
    for inst in dead
        if isa(inst, LLVM.PHIInst)
            replace_uses!(inst, LLVM.UndefValue(inst.value_type))
        end
    end
    while !isempty(dead)
        remaining = LLVM.Instruction[]
        for inst in dead
            if isempty(inst.uses)
                LLVM.API.LLVMInstructionEraseFromParent(inst)
            else
                push!(remaining, inst)
            end
        end
        @assert length(remaining) < length(dead)
        dead = remaining
    end
    for bb in deadblocks
        LLVM.API.LLVMDeleteBasicBlock(bb)
    end

    # The failed phis and their placeholders are now unused.
    for bb in keys(cut)
        changed = true
        while changed
            changed = false
            for inst in collect(bb.instructions)
                isa(inst, LLVM.PHIInst) || break
                if isempty(inst.uses)
                    LLVM.API.LLVMInstructionEraseFromParent(inst)
                    changed = true
                end
            end
        end
    end

    drop_throwing_body_attrs!(f.function_attributes)
    for u in f.uses
        call = u.user
        if isa(call, LLVM.CallInst) && call.called_operand == f
            drop_throwing_body_attrs!(call.function_attributes)
        end
    end
    return nothing
end

# `v` with the steps that do not change its value stripped: pointer bitcasts, and the
# `add x, 0` / `mul x, 1` that `nodecayed_getparent` builds for zero offsets.
function nodecayed_strip(@nospecialize(v::LLVM.Value))::LLVM.Value
    if isa(v, LLVM.BitCastInst) || (isa(v, LLVM.ConstantExpr) && v.opcode == LLVM.Opcode.BitCast)
        return nodecayed_strip(v.operands[1])
    end
    if isa(v, LLVM.Instruction) && (v.opcode == LLVM.Opcode.Add || v.opcode == LLVM.Opcode.Mul)
        unit = v.opcode == LLVM.Opcode.Add ? 0 : 1
        lhs, rhs = v.operands
        if isa(rhs, LLVM.ConstantInt) && convert(Int, rhs) == unit
            return nodecayed_strip(lhs)
        end
        if isa(lhs, LLVM.ConstantInt) && convert(Int, lhs) == unit
            return nodecayed_strip(rhs)
        end
    end
    return v
end

function nodecayed_undef_or_poison(@nospecialize(v::LLVM.Value))::Bool
    undeforpoison = isa(v, LLVM.UndefValue)
    @static if LLVM.version() >= v"12"
        undeforpoison |= isa(v, LLVM.PoisonValue)
    end
    return undeforpoison
end

# Replace each set of the phis in `live` that take, besides each other, one and the same
# value by that value.
#
# GVN keeps a field it re-reads after a call that may write memory (say the data pointer of a
# `Const` array, loaded before a loop and again after a call inside it) as a web of phis.
# `nodecayed_phis!` turns that web into phis of the parent object and of the byte offset,
# which then only ever carry the one object and offset 0, but through a cycle of phis and
# bitcasts that it does not fold. Enzyme then cannot recompute the data pointer reloaded from
# such a phi in the reverse pass, so it caches that addrspace(13) pointer, and fails with
# "Illegal cached pointer".
#
# A set of phis whose operands are each other or one value `V` (undef or poison aside)
# can only ever hold `V`. Without undef or poison operands `V` also dominates every one of
# them: each path to one of the phis reaches it with `V`, or with another of them, which it
# reached before. An undef or poison operand breaks that argument, since its path need not
# pass `V`. Under `--check-bounds=yes`, for instance, Julia 1.10's `matmul3x3` reaches a
# block from seven bounds-checked paths, and only one of them carries the array whose data
# pointer the phi carries; the other six bring undef, and `V`, that load, does not dominate
# the phi. So only the phis that `V` dominates are replaced, and any other stays a phi,
# which then takes `V` instead of the replaced ones (an operand of a phi dominates the end of
# its incoming block, and `V` dominates the block of each replaced phi).
function fold_redundant_nodecayed_phis!(live::Set{LLVM.PHIInst})
    isempty(live) && return
    # `live` holds the phis created in one function.
    domtree = LLVM.DomTree(((first(live)).parent).parent)
    fold_redundant_nodecayed_phis!(domtree, live)
    dispose(domtree)
    return
end

nodecayed_dominates(domtree::LLVM.DomTree, @nospecialize(v::LLVM.Value), q::LLVM.PHIInst)::Bool =
    !isa(v, LLVM.Instruction) || LLVM.dominates(domtree, v, q)

function fold_redundant_nodecayed_phis!(domtree::LLVM.DomTree, live::Set{LLVM.PHIInst})
    changed = true
    while changed
        changed = false
        for p in collect(live)
            in(p, live) || continue
            group = LLVM.PHIInst[p]
            seen = Set{LLVM.PHIInst}(group)
            same = nothing
            ok = true
            i = 1
            while ok && i <= length(group)
                for (v, _) in (group[i]).incoming
                    sv = nodecayed_strip(v)
                    if isa(sv, LLVM.PHIInst) && in(sv, live)
                        if !in(sv, seen)
                            push!(seen, sv)
                            push!(group, sv)
                        end
                    elseif nodecayed_undef_or_poison(sv)
                        continue
                    elseif same === nothing
                        same = sv
                    elseif same != sv
                        ok = false
                        break
                    end
                end
                i += 1
            end
            (ok && same !== nothing) || continue
            castable = true
            for q in group
                castable &= nodecayed_castable(same.value_type, q.value_type)
            end
            castable || continue
            for q in group
                nodecayed_dominates(domtree, same, q) || continue
                repl = same
                if repl.value_type != q.value_type
                    b = IRBuilder()
                    position!(b, LLVM.before(first_non_phi(q.parent)))
                    repl = bitcast!(b, repl, q.value_type)
                end
                replace_uses!(q, repl)
                LLVM.API.LLVMInstructionEraseFromParent(q)
                delete!(live, q)
                changed = true
            end
        end
    end
    return
end

# Whether a value of type `from` can stand for one of type `to`, with at most a bitcast.
function nodecayed_castable(@nospecialize(from::LLVM.LLVMType), @nospecialize(to::LLVM.LLVMType))::Bool
    from == to && return true
    return isa(from, LLVM.PointerType) && isa(to, LLVM.PointerType) && from.addrspace == to.addrspace
end

function first_non_phi(bb::LLVM.BasicBlock)::LLVM.Instruction
    for inst in bb.instructions
        isa(inst, LLVM.PHIInst) || return inst
    end
    error("block without a terminator")
end

# Julia 1.11+ can keep both an Array phi and a separate phi of the raw data
# pointers stored in those Arrays. Recreate the latter from the former when each
# Array has exactly one write to its data field. This lets Enzyme use the Array's
# shadow for the data pointer as well.
function array_data_store(alloc::LLVM.CallInst, array_phi::LLVM.PHIInst)
    called_function_name(alloc) in GC_OBJECT_ALLOCATORS || return nothing
    data = nothing
    todo = LLVM.Value[alloc]
    seen = Set{LLVM.Value}()
    while !isempty(todo)
        cur = pop!(todo)
        cur in seen && continue
        push!(seen, cur)
        for use in cur.uses
            user = use.user
            if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                    isa(user, LLVM.GetElementPtrInst)
                push!(todo, user)
            elseif isa(user, LLVM.StoreInst)
                user.operands[1] == cur && return nothing
                base, off = get_base_and_offset(user.operands[2])
                base == alloc || return nothing
                if off == 0
                    data === nothing || return nothing
                    data = user.operands[1]
                end
            elseif isa(user, LLVM.CallInst)
                if readonly_noreturn_use(user, cur)
                    continue
                end
                memoff = sizeof(Ptr{Cvoid})
                if writes_beside_memory_field(user, cur, alloc, memoff)
                    _, off = get_base_and_offset(cur)
                    off >= memoff && continue
                end
                return nothing
            elseif user == array_phi || isa(user, LLVM.LoadInst)
                continue
            else
                return nothing
            end
        end
    end
    return data
end

function matching_array_data_phi(raw_phi::LLVM.PHIInst, phis::Vector{LLVM.PHIInst})
    raw_incoming = raw_phi.incoming
    for array_phi in phis
        ty = array_phi.value_type
        (isa(ty, LLVM.PointerType) && ty.addrspace == Tracked) || continue
        array_incoming = array_phi.incoming
        length(raw_incoming) == length(array_incoming) || continue
        matched = true
        for ((raw, raw_bb), (array, array_bb)) in zip(raw_incoming, array_incoming)
            if raw_bb != array_bb
                matched = false
                break
            end
            alloc, off = get_base_and_offset(array; offsetAllowed = false)
            if off != 0 || !(isa(alloc, LLVM.CallInst)) ||
                    array_data_store(alloc, array_phi) != raw
                matched = false
                break
            end
        end
        matched && return array_phi
    end
    return nothing
end

function fold_array_data_phis!(mod::LLVM.Module)
    @static if VERSION < v"1.11-"
        return nothing
    else
        for fn in mod.functions, bb in fn.blocks
            phis = LLVM.PHIInst[]
            for inst in bb.instructions
                isa(inst, LLVM.PHIInst) || break
                push!(phis, inst)
            end
            for raw_phi in phis
                ty = raw_phi.value_type
                (isa(ty, LLVM.PointerType) && ty.addrspace == 0) || continue
                # A load in this block cannot replace a phi incoming from a
                # different predecessor, since it would not dominate that edge.
                has_phi_user = false
                for use in raw_phi.uses
                    if isa(use.user, LLVM.PHIInst)
                        has_phi_user = true
                        break
                    end
                end
                has_phi_user && continue
                array_phi = matching_array_data_phi(raw_phi, phis)
                array_phi === nothing && continue
                B = IRBuilder()
                position!(B, LLVM.before(first_non_phi(bb)))
                ptrty = LLVM.PointerType(LLVM.Int8Type())
                data_field = pointercast!(B, array_phi, LLVM.PointerType(ptrty, 11))
                data = load!(B, ty, data_field)
                replace_uses!(raw_phi, data)
                LLVM.API.LLVMInstructionEraseFromParent(raw_phi)
            end
        end
        return nothing
    end
end

"""
    catches_exceptions(fn::LLVM.Function) -> Bool

Whether `fn` may catch an exception it raised: it enters a handler (`try`), which Julia
1.10 does with `jl_enter_handler` and a `setjmp`, a `returns_twice` call.
"""
function catches_exceptions(fn::LLVM.Function)
    handlers = (
        "julia.except_enter", "jl_enter_handler", "ijl_enter_handler", "jl_setjmp",
        "ijl_setjmp", "__sigsetjmp", "sigsetjmp", "_setjmp", "setjmp",
    )
    for bb in fn.blocks, inst in bb.instructions
        isa(inst, LLVM.CallInst) || continue
        callee = inst.called_operand
        isa(callee, LLVM.Function) || continue
        callee.name in handlers && return true
        haskey(callee.function_attributes, :returns_twice) && return true
    end
    return false
end

function never_returns(call::LLVM.CallInst)
    callee = call.called_operand
    isa(callee, LLVM.Function) && is_noreturn(callee) && return true
    noreturn = (EnumAttribute(:noreturn)).kind
    for attr in callsite_function_attributes(call)
        isa(attr, LLVM.EnumAttribute) && attr.kind == noreturn && return true
    end
    return false
end

"""
    fresh_array_length_loads(alloc::LLVM.CallInst, catches::Bool) -> Union{Vector{LLVM.LoadInst}, Nothing}

The loads of the length of the vector `alloc`, a call of `jl_alloc_array_1d(atype, n)`,
that are certain to read `n`: those of its `length` and `nrows` fields (byte offsets 8
and 24 of Julia 1.10's `jl_array_t`). Only the runtime's resize functions change those
fields, and they receive the array. So this answers `nothing` unless every use of the
array, through casts and constant-offset GEPs, is a load, a comparison, or a write
barrier or GC preserve, which do not resize it. A store of the array, passing it to any
other call, a merge with another value and a store into its header all answer `nothing`.
The loads of its data pointer, and what is stored through that pointer, are not uses of
the header. A call that never returns, such as the bounds error a checked read throws,
may also receive the array when the function cannot catch (`catches` is false): no load
of the function runs after it.
"""
function fresh_array_length_loads(alloc::LLVM.CallInst, catches::Bool)
    lengths = LLVM.LoadInst[]
    todo = LLVM.Value[alloc]
    seen = Set{LLVM.Value}()
    while !isempty(todo)
        cur = pop!(todo)
        cur in seen && continue
        push!(seen, cur)
        for use in cur.uses
            user = use.user
            if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst)
                push!(todo, user)
            elseif isa(user, LLVM.GetElementPtrInst)
                base, _ = get_base_and_offset(user)
                base == alloc || return nothing
                push!(todo, user)
            elseif isa(user, LLVM.LoadInst)
                (Bool(LLVM.API.LLVMGetVolatile(user)) || LLVM.isatomic(user)) && continue
                user.value_type == LLVM.IntType(8 * sizeof(Int)) || continue
                base, off = get_base_and_offset(cur)
                base == alloc || return nothing
                off in (8, 24) && push!(lengths, user)
            elseif isa(user, LLVM.ICmpInst)
                continue
            elseif isa(user, LLVM.CallInst)
                fn = user.called_operand
                isa(fn, LLVM.Function) || return nothing
                fn.name in ("julia.write_barrier", "llvm.julia.gc_preserve_begin") &&
                    continue
                !catches && never_returns(user) && continue
                return nothing
            else
                return nothing
            end
        end
    end
    return lengths
end

"""
    fold_fresh_array_lengths!(mod::LLVM.Module) -> Bool

Replace each load of the length of a vector that `jl_alloc_array_1d(atype, n)` made in
the same function by `n`, where `fresh_array_length_loads` shows that nothing can have
resized it (Julia 1.10; later versions build an array's header with stores that LLVM
already forwards).

On Julia 1.10 the allocator is opaque, so a loop over `eachindex(out)` after
`out = similar(r)` reads the bound from `out`. The reverse pass then needs `out` itself
for that bound. When the allocation is in a loop, Enzyme rebuilds it there (rematerializes
it, which allocates it again and replays every store into it) only to read its length.
After the fold the reverse pass reads `n` instead and needs no copy of `out`.
"""
function fold_fresh_array_lengths!(mod::LLVM.Module)
    @static if VERSION >= v"1.11-"
        return false
    else
        changed = false
        for fn in mod.functions
            allocs = LLVM.CallInst[]
            for bb in fn.blocks, inst in bb.instructions
                isa(inst, LLVM.CallInst) || continue
                callee = inst.called_operand
                isa(callee, LLVM.Function) || continue
                callee.name in ("jl_alloc_array_1d", "ijl_alloc_array_1d") || continue
                push!(allocs, inst)
            end
            isempty(allocs) && continue
            catches = catches_exceptions(fn)
            for alloc in allocs
                n = collect(alloc.arguments)[2]
                lengths = fresh_array_length_loads(alloc, catches)
                lengths === nothing && continue
                for ld in lengths
                    ld.value_type == n.value_type || continue
                    replace_uses!(ld, n)
                    LLVM.API.LLVMInstructionEraseFromParent(ld)
                    changed = true
                end
            end
        end
        return changed
    end
end

"""
    unfold_field_phi_loads!(f::LLVM.Function) -> Bool

Turn a load through a phi of field addresses at different offsets back into a phi
of loads.

LLVM's instcombine folds a phi of two loads into one load through a phi of their
addresses. Julia 1.13.1 (LLVM 20) does so in broadcast's `unalias`, for the data
pointer of a copied `Memory` (its field at offset 8) and that of the original
`Array` (its field at offset 0). `nodecayed_phis!` then rewrites that phi into a
phi of the two objects and a phi of the offsets, and Enzyme's type analysis merges
the layouts of a `Memory` and an `Array` (`IllegalTypeAnalysisException`). Loading
in the predecessors keeps the objects apart, as they were before the fold.

The same fold makes a phi of addresses into array data (address space 13) out of a
loop-carried value that is read from an array on every edge into the loop, such as
`prev` in `prev = x[first]; for i in first+1:last; ...; prev = x[i]; end`, the shape
of a loop whose first iteration is peeled. On Julia 1.10, `nodecayed_phis!` rewrites
that phi into a phi of the array objects and a phi of byte offsets, and reloads the
data pointer from the object on every iteration. The reverse pass then reloads the
object and its data pointer on every iteration of the loop's reverse: an `i8` GEP of
the offset makes instcombine retype those loads, which drops the `!invariant.group`
Enzyme gave them, so nothing hoists them past the stores into the shadow. Loading in
the predecessors leaves a phi of the values, as in the loop that is not peeled.
`UnfoldDataPhiLoads[] = false` before a derivative compiles leaves such phis to
`nodecayed_phis!`.

Only rewrite what is certainly equivalent: a phi in address space 11 whose incoming
values are not all at the same constant offset from their base, or on Julia 1.10 a phi
in address space 13, whose users are all non-volatile loads from it in its own block that no
memory write precedes (in address space 13 also non-atomic or unordered). Each load is
re-created before the terminator of every predecessor. An edge whose predecessor also
branches elsewhere is split first, so no load becomes speculative.
"""
function unfold_field_phi_loads!(f::LLVM.Function)::Bool
    # Splitting an edge rebuilds the phis of the block it leads to, so the
    # candidates are collected again afterwards.
    edges = Tuple{LLVM.BasicBlock, LLVM.BasicBlock}[]
    for bb in f.blocks
        for phi in bb.instructions
            isa(phi, LLVM.PHIInst) || break
            field_phi_loads(phi) === nothing && continue
            for (_, pred) in phi.incoming
                length(pred.terminator.successors) == 1 && continue
                (pred, bb) in edges || push!(edges, (pred, bb))
            end
        end
    end
    changed = false
    builder = IRBuilder()
    for (pred, succ) in edges
        split_edge!(builder, pred, succ)
    end
    for bb in f.blocks
        phis = LLVM.PHIInst[]
        for phi in bb.instructions
            isa(phi, LLVM.PHIInst) || break
            push!(phis, phi)
        end
        for phi in phis
            loads = field_phi_loads(phi)
            loads === nothing && continue
            all_single = true
            for (_, pred) in phi.incoming
                all_single &= length(pred.terminator.successors) == 1
            end
            all_single || continue
            unfold_field_phi_loads!(builder, phi, loads)
            changed = true
        end
    end
    dispose(builder)
    return changed
end

# Whether `unfold_field_phi_loads!` also turns loads through a phi of addresses into
# array data (address space 13) into a phi of loads (see its docstring).
const UnfoldDataPhiLoads = Ref(true)

# The loads `unfold_field_phi_loads!` moves out of `phi`'s block, or `nothing`.
function field_phi_loads(phi::LLVM.PHIInst)::Union{Nothing, Vector{LLVM.LoadInst}}
    ty = phi.value_type
    isa(ty, LLVM.PointerType) || return nothing
    # Later versions read the data pointer through `julia.gc_loaded`, and their phis of
    # data addresses are left to `nodecayed_phis!` as before.
    data = VERSION < v"1.11-" && ty.addrspace == 13
    if data
        UnfoldDataPhiLoads[] || return nothing
    elseif ty.addrspace != Derived
        return nothing
    end
    bb = phi.parent
    offset = nothing
    same = true
    for (v, pred) in phi.incoming
        count(Base.Fix2(==, bb), pred.terminator.successors) == 1 || return nothing
        data && continue
        _, off = get_base_and_offset(v)
        offset === nothing || offset == off || (same = false)
        offset = off
    end
    same && !data && return nothing
    loads = LLVM.LoadInst[]
    for u in phi.uses
        ld = u.user
        (isa(ld, LLVM.LoadInst) && ld.parent == bb) || return nothing
        Bool(LLVM.API.LLVMGetVolatile(ld)) && return nothing
        data && !unfoldable_phi_load(ld) && return nothing
        ld in loads || push!(loads, ld)
    end
    isempty(loads) && return nothing
    # A write before one of the loads could change what it reads.
    remaining = length(loads)
    for inst in bb.instructions
        isa(inst, LLVM.PHIInst) && continue
        if isa(inst, LLVM.LoadInst) && inst in loads
            remaining -= 1
            remaining == 0 && break
        elseif root_path_may_write(inst)
            return nothing
        end
    end
    return loads
end

function unfold_field_phi_loads!(builder::LLVM.IRBuilder, phi::LLVM.PHIInst, loads::Vector{LLVM.LoadInst})
    for ld in loads
        position!(builder, LLVM.before(phi))
        merged = phi!(builder, ld.value_type, ld.name)
        for (v, pred) in phi.incoming
            position!(builder, LLVM.before(pred.terminator))
            clone = LLVM.Instruction(LLVM.API.LLVMInstructionClone(ld))
            LLVM.API.LLVMSetOperand(clone, 0, v)
            LLVM.API.LLVMInsertIntoBuilderWithName(builder, clone, ld.name)
            push!(merged.incoming, (clone, pred))
        end
        # Memory metadata such as `!tbaa` is not allowed on a phi.
        for k in ("enzyme_type", "enzyme_inactive", "enzyme_active")
            haskey(ld.metadata, k) && (merged.metadata[k] = ld.metadata[k])
        end
        replace_uses!(ld, merged)
        LLVM.erase!(ld)
    end
    LLVM.erase!(phi)
    return
end

function nodecayed_phis!(mod::LLVM.Module)
    # Simple handler to fix addrspace 11
    #complex handler for addrspace 13, which itself comes from a load of an
    # addrspace 10
    ctx = mod.context
    for f in mod.functions

        guaranteedInactive = false

        for attr in collect(f.function_attributes)
            if !isa(attr, LLVM.StringAttribute)
                continue
            end
            if attr.kind == "enzyme_inactive"
                guaranteedInactive = true
                break
            end
        end

        if guaranteedInactive
            continue
        end


        entry_ft = f.function_type

        RT = entry_ft.return_type
        inactiveRet = RT == LLVM.VoidType()

        for attr in collect(f.return_attributes)
            if !isa(attr, LLVM.StringAttribute)
                continue
            end
            if attr.kind == "enzyme_inactive"
                inactiveRet = true
                break
            end
        end

        if inactiveRet
            for idx in 1:length(f.parameters)
                inactiveParm = false
                for attr in collect(f.parameter_attributes[idx])
                    if !isa(attr, LLVM.StringAttribute)
                        continue
                    end
                    if attr.kind == "enzyme_inactive"
                        inactiveParm = true
                        break
                    end
                end
                if !inactiveParm
                    inactiveRet = false
                    break
                end
            end
            if inactiveRet
                continue
            end
        end

        offty = LLVM.IntType(8 * sizeof(Int))
        i8 = LLVM.IntType(8)

        # A phi that merges an addrspace(11) argument with a pointer into an object has
        # no parent to give it below; load through it in its predecessors instead.
        unfold_argument_phi_loads!(f)

        for addr in (11, 13)

            nextvs = Dict{LLVM.PHIInst, LLVM.PHIInst}()
            failed = Tuple{LLVM.PHIInst, EnzymeInternalError}[]
            # The phis created below that are still in the function.
            created = Set{LLVM.PHIInst}()
            mtodo = Vector{LLVM.PHIInst}[]
            goffsets = Dict{LLVM.PHIInst, LLVM.PHIInst}()
            nonphis = LLVM.Instruction[]
            anyV = false
            for bb in f.blocks
                todo = LLVM.PHIInst[]
                nonphi = nothing
                for inst in bb.instructions
                    if !isa(inst, LLVM.PHIInst)
                        nonphi = inst
                        break
                    end
                    ty = inst.value_type
                    if !isa(ty, LLVM.PointerType)
                        continue
                    end
                    if ty.addrspace != addr
                        continue
                    end
                    if addr == 11 && nodecayed_all_args(inst)
                        continue
                    end

                    push!(todo, inst)
                    nb = IRBuilder()
                    position!(nb, LLVM.before(inst))
                    el_ty = if addr == 11 && !LLVM.isopaque(ty)
                        ty.element_type
                    else
                        LLVM.StructType(LLVM.LLVMType[])
                    end
                    nphi = phi!(
                        nb,
                        LLVM.PointerType(el_ty, 10),
                        "nodecayed." * inst.name,
                    )
                    nextvs[inst] = nphi
                    push!(created, nphi)
                    anyV = true

                    goffsets[inst] = phi!(nb, offty, "nodecayedoff." * inst.name)
                    push!(created, goffsets[inst])
                end
                push!(mtodo, todo)
                push!(nonphis, nonphi)
            end
            for (bb, todo, nonphi) in zip(f.blocks, mtodo, nonphis)

                for inst in todo
                    ty = inst.value_type
                    el_ty = if addr == 11 && !LLVM.isopaque(ty)
                        ty.element_type
                    else
                        LLVM.StructType(LLVM.LLVMType[])
                    end
                    nvs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
                    offsets = Tuple{LLVM.Value, LLVM.BasicBlock}[]
                    try
                        for (v, pb) in inst.incoming
                            done = false
                            for ((nv, pb0), (offset, pb1)) in zip(nvs, offsets)
                                if pb0 == pb
                                    push!(nvs, (nv, pb))
                                    push!(offsets, (offset, pb))
                                    done = true
                                    break
                                end
                            end
                            if done
                                continue
                            end

                            v0 = v

                            b = IRBuilder()
                            position!(b, LLVM.before(pb.terminator))

                            phicache = Dict{LLVM.PHIInst, Tuple{LLVM.PHIInst, LLVM.PHIInst}}()
                            st = NoDecayedPhiState(addr, offty, ctx, f, inst, v0, nextvs, goffsets, phicache)
                            v, offset, hadload = nodecayed_getparent(st, b, v, LLVM.ConstantInt(offty, 0), false)

                            if addr == 13
                                @assert hadload
                            end

                            if !LLVM.isopaque(v.value_type) && v.value_type.element_type != el_ty
                                v = bitcast!(
                                    b,
                                    v,
                                    LLVM.PointerType(el_ty, v.value_type.addrspace),
                                )
                            end
                            push!(nvs, (v, pb))
                            push!(offsets, (offset, pb))
                        end
                    catch err
                        err isa EnzymeInternalError || rethrow()
                        push!(failed, (inst, err))
                        continue
                    end

                    nb = IRBuilder()
                    position!(nb, LLVM.before(nonphi))

                    offset = goffsets[inst]
                    append!(offset.incoming, offsets)
                    if all(x -> x[1] == offsets[1][1], offsets)
                        offset = offsets[1][1]
                    end

                    nphi = nextvs[inst]

                    function ogbc(@nospecialize(x::LLVM.Value))
                        while isa(x, LLVM.BitCastInst)
                            x = x.operands[1]
                        end
                        return x
                    end

                    if all(x -> ogbc(x[1]) == ogbc(nvs[1][1]), nvs)
                        bc = ogbc(nvs[1][1])
                        if bc.value_type != nphi.value_type
                            bc = bitcast!(nb, bc, nphi.value_type)
                        end
                        replace_uses!(nphi, bc)
                        delete!(created, nphi)
                        LLVM.API.LLVMInstructionEraseFromParent(nphi)
                        nphi = bc
                    else
                        append!(nphi.incoming, nvs)
                    end

                    if addr == 13
                        @static if VERSION < v"1.11-"
                            nphi = bitcast!(nb, nphi, LLVM.PointerType(ty, 10))
                            nphi = addrspacecast!(nb, nphi, LLVM.PointerType(ty, 11))
                            nphi = load!(nb, ty, nphi)
                        else
                            base_obj = nphi

                            jlt = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), 10)
                            pjlt = LLVM.PointerType(jlt)

                            nphi = get_memory_data(nb, nphi)
                            nphi = bitcast!(nb, nphi, pjlt)

                            GTy = LLVM.FunctionType(LLVM.PointerType(jlt, 13), LLVM.LLVMType[jlt, pjlt])
                            gcloaded, _ = get_function!(
                                mod,
                                "julia.gc_loaded",
                                GTy
                            )
                            nphi = call!(nb, GTy, gcloaded, LLVM.Value[base_obj, nphi])
                            if nphi.value_type != ty
                                nphi = bitcast!(nb, nphi, ty)
                            end
                        end
                    else
                        nphi = addrspacecast!(nb, nphi, ty)
                    end
                    if !isa(offset, LLVM.ConstantInt) || !LLVM.isnull(offset)
                        nphi = bitcast!(nb, nphi, LLVM.PointerType(i8, ty.addrspace))
                        nphi = gep!(nb, i8, nphi, [offset])
                        nphi = bitcast!(nb, nphi, ty)
                    end
                    replace_uses!(inst, nphi)
                end
                for inst in todo
                    any(x -> x[1] == inst, failed) && continue
                    erase!(inst)
                end
            end
            # The placeholders of the phis that failed above never get incoming
            # values, so they are not foldable; `nodecayed_cut_failed!` removes them.
            # Fold first: the cut erases dead code, which can hold phis in `created`.
            for (inst, _) in failed
                delete!(created, nextvs[inst])
                delete!(created, goffsets[inst])
            end
            fold_redundant_nodecayed_phis!(created)
            if !isempty(failed)
                nodecayed_cut_failed!(f, failed)
            end
        end
    end
    return nothing
end

function is_sret_like_attr(attr::LLVM.Attribute)::Bool
    sretkind = :sret
    k = attr.kind
    return k == sretkind ||
        k == "enzyme_sret" ||
        k == "enzymejl_returnRoots" ||
        k == "enzymejl_rooted_typ"
end

"""
    decay_args_readonly(st, fop, args)

Whether the call `st` merely reads through each of the argument positions in
`args`. That is the case when the position itself is marked `readonly` /
`readnone`, or when the callee as a whole only reads memory -- which is how
plain libcalls such as `memcmp` are annotated. Positions that carry an
sret-like marker are never treated as read-only, since the callee writes the
object back through them.
"""
function decay_args_readonly(
        st::LLVM.CallInst,
        @nospecialize(fop::LLVM.Value),
        args::Vector{Int},
    )::Bool
    fn_readonly = false
    for attr in collect(st.function_attributes)
        if is_readonly(attr)
            fn_readonly = true
            break
        end
    end
    if !fn_readonly && isa(fop, LLVM.Function) && is_readonly(fop)
        fn_readonly = true
    end
    for i in args
        attrs = collect(st.argument_attributes[i])
        # vararg arguments only have call site attributes
        if isa(fop, LLVM.Function) && i <= length(fop.parameters)
            append!(attrs, collect(fop.parameter_attributes[i]))
        end
        arg_readonly = fn_readonly
        for attr in attrs
            if is_sret_like_attr(attr)
                return false
            end
            if is_readonly(attr)
                arg_readonly = true
            end
        end
        if !arg_readonly
            return false
        end
    end
    return true
end

"""
    legalize_readonly_decay!(st, inst, args)

Rewrite the argument positions `args` of the read-only call `st`, all of which
are the illegal `addrspace(10) -> addrspace(0)` cast `inst`, to use a legally
derived pointer instead: decay to addrspace 11 and go through
`julia.pointer_from_objref`, with the object gc-preserved across the call.
Julia's late GC lowering understands that form; the raw cast it does not.
"""
function legalize_readonly_decay!(
        st::LLVM.CallInst,
        inst::LLVM.Instruction,
        args::Vector{Int},
    )
    obj = inst.operands[1]
    nb = IRBuilder()
    position!(nb, LLVM.before(st))
    if obj.value_type.addrspace == Derived
        # Already derived: preserve the tracked object it points into.
        token = emit_gc_preserve_begin(nb, LLVM.Value[decay_tracked_base(obj)])
        derived = obj
    else
        token = emit_gc_preserve_begin(nb, LLVM.Value[obj])
        derived = if LLVM.isopaque(obj.value_type)
            addrspacecast!(nb, obj, LLVM.PointerType(Derived))
        else
            T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
            addrspacecast!(
                nb,
                bitcast!(nb, obj, LLVM.PointerType(T_jlvalue, Tracked)),
                LLVM.PointerType(T_jlvalue, Derived),
            )
        end
    end
    raw = emit_pointerfromobjref!(nb, derived)
    if raw.value_type != inst.value_type
        raw = bitcast!(nb, raw, inst.value_type)
    end
    for i in args
        st.operands[i] = raw
    end
    eb = IRBuilder()
    position!(eb, LLVM.after(st))
    emit_gc_preserve_end(eb, token)
    return nothing
end

# The tracked (addrspace 10) object a derived pointer points into, or
# `nothing` if it does not trace back to one through GEPs and casts.
function decay_tracked_base(@nospecialize(v::LLVM.Value))
    while true
        if v.value_type.addrspace == Tracked
            return v
        end
        if isa(v, LLVM.GetElementPtrInst) ||
                isa(v, LLVM.BitCastInst) ||
                isa(v, LLVM.AddrSpaceCastInst)
            v = v.operands[1]
            continue
        end
        return nothing
    end
    return
end

function fix_decayaddr!(mod::LLVM.Module)
    for f in mod.functions
        invalid = LLVM.Instruction[]
        for bb in f.blocks, inst in bb.instructions
            if !isa(inst, LLVM.AddrSpaceCastInst)
                continue
            end
            prety = inst.operands[1].value_type
            postty = inst.value_type
            if postty.addrspace != 0
                continue
            end
            # When Enzyme moves a stack allocation to the GC heap, it casts a
            # derived pointer into the new object back to addrspace 0 for call
            # operands. Since Julia 1.13.1 codegen passes a pointer into a
            # `returnRoots` buffer on as another call's roots argument, which
            # yields such a cast.
            if prety.addrspace == Derived
                if decay_tracked_base(inst.operands[1]) === nothing
                    continue
                end
            elseif prety.addrspace != Tracked
                continue
            end
            push!(invalid, inst)
        end

        for inst in invalid
            temp = nothing
            # A single user may consume `inst` in more than one operand, and the
            # handlers below rewrite all of them at once. Snapshot the distinct
            # users first so that detaching a use does not invalidate the
            # iteration, and so that no user is visited twice.
            seen = Set{LLVM.Value}()
            users = LLVM.Value[]
            for user in inst.users
                if !(user in seen)
                    push!(seen, user)
                    push!(users, user)
                end
            end
            for st in users
                # Storing _into_ the decay addr is okay
                # we just cannot store the decayed addr into
                # somewhere
                if isa(st, LLVM.StoreInst)
                    if st.operands[2] == inst
                        st.operands[2] = inst.operands[1]
                        nb = IRBuilder()
                        position!(nb, LLVM.after(st))
                        julia_post_cache_store(st.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
                        continue
                    end
                end
                if isa(st, LLVM.LoadInst)
                    st.operands[1] = inst.operands[1]
                    continue
                end

                if isa(st, LLVM.GetElementPtrInst)
                    legal = true
                    torem = LLVM.Instruction[]
                    for u in st.uses
                        st2 = u.user
                        # Storing _into_ the decay addr is okay
                        # we just cannot store the decayed addr into
                        # somewhere
                        if isa(st2, LLVM.StoreInst)
                            if st2.operands[2] == st
                                push!(torem, st2)
                                continue
                            end
                        end
                        if isa(st2, LLVM.LoadInst)
                            push!(torem, st2)
                            continue
                        end
                        legal = false
                    end
                    if legal
                        B = IRBuilder()
                        position!(B, LLVM.after(st))
                        op1 = inst.operands[1]
                        cst = addrspacecast!(B, op1, LLVM.isopaque(op1.value_type) ? LLVM.PointerType(Derived) : LLVM.PointerType(op1.value_type.element_type))
                        gep2 = gep!(B, st.source_element_type, cst, st.operands[2:end])
                        for st2 in torem
                            if isa(st2, LLVM.StoreInst)
                                st2.operands[2] = gep2
                                nb = IRBuilder()
                                position!(nb, LLVM.after(st2))
                                julia_post_cache_store(st2.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
                                continue
                            end
                            if isa(st2, LLVM.LoadInst)
                                st2.operands[1] = gep2
                                continue
                            end

                        end
                        erase!(st)
                        continue
                    end
                end

                # if isa(st, LLVM.InsertValueInst)
                #    if operands(st)[1] == inst
                #        push!(invalid, st)
                #        st.operands[1] = LLVM.UndefValue(value_type(inst))
                #        continue
                #    end
                #    if operands(st)[2] == inst
                #        push!(invalid, st)
                #        st.operands[2] = LLVM.UndefValue(value_type(inst))
                #        continue
                #    end
                # end
                if !isa(st, LLVM.CallInst)
                    bt = GPUCompiler.backtrace(st)
                    msg = sprint() do io::IO
                        println(io, string(f))
                        println(io, inst)
                        println(io, st)
                        print(io, "Illegal decay of nonnull\n")
                        if bt !== nothing
                            print(io, "\nCaused by:")
                            Base.show_backtrace(io, bt)
                            println(io)
                        end
                    end
                    throw(AssertionError(msg))
                end

                fop = st.operands[end]

                intr = fop isa LLVM.Function ? fop.intrinsic : nothing

                if intr == LLVM.Intrinsic("llvm.memcpy") ||
                        intr == LLVM.Intrinsic("llvm.memmove") ||
                        intr == LLVM.Intrinsic("llvm.memset")
                    newvs = LLVM.Value[]
                    for (i, v) in enumerate(st.arguments)
                        if v == inst
                            st.operands[i] = inst.operands[1]
                            push!(newvs, inst.operands[1])
                            continue
                        end
                        push!(newvs, v)
                    end

                    nb = IRBuilder()
                    position!(nb, LLVM.before(st))
                    if intr == LLVM.Intrinsic("llvm.memcpy")
                        newi = memcpy!(nb, newvs[1], 0, newvs[2], 0, newvs[3])
                    elseif intr == LLVM.Intrinsic("llvm.memmove")
                        newi = memmove!(nb, newvs[1], 0, newvs[2], 0, newvs[3])
                    else
                        newi = memset!(nb, newvs[1], newvs[2], newvs[3], 0)
                    end

                    append!(newi.function_attributes, st.function_attributes)
                    append!(newi.return_attributes, st.return_attributes)
                    for i in 1:length(st.arguments)
                        append!(newi.argument_attributes[i], st.argument_attributes[i])
                    end

                    API.EnzymeCopyMetadata(newi, st)

                    erase!(st)
                    continue
                end

                # An ordinary libcall such as `memcmp` is not an intrinsic that
                # can be re-created over addrspace 10, and it has no sret to
                # copy the object through either. Since it only reads through
                # the pointer, hand it a legally derived one.
                decay_args = Int[]
                for (i, v) in enumerate(st.arguments)
                    if v == inst
                        push!(decay_args, i)
                    end
                end
                if !isempty(decay_args) && decay_args_readonly(st, fop, decay_args)
                    legalize_readonly_decay!(st, inst, decay_args)
                    continue
                end

                mayread = false
                maywrite = false
                sret = true
                sret_elty = nothing
                sretkind = :sret
                for (i, v) in enumerate(st.arguments)
                    if v == inst
                        readnone = false
                        readonly = false
                        writeonly = false
                        t_sret = false
                        # vararg arguments have no callee parameter attributes
                        param_attrs = if i <= length(fop.parameters)
                            collect(fop.parameter_attributes[i])
                        else
                            LLVM.Attribute[]
                        end
                        for a in param_attrs
                            if a.kind == sretkind
                                sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if a.kind == "enzyme_sret"
                                sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if a.kind == "enzymejl_returnRoots"
                                sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if a.kind == "enzymejl_rooted_typ"
                                sret_elty = convert(LLVMType, AnyArray(Int(CountTrackedPointers(get_rooted_typ(fop, i)).count)))
                                t_sret = true
                            end
                            # if kind(a) == kind(StringAttribute("enzyme_sret_v"))
                            #     t_sret = true
                            # end
                            if a.kind == :readonly
                                readonly = true
                            end
                            if a.kind == :readnone
                                readnone = true
                            end
                            if a.kind == :writeonly
                                writeonly = true
                            end
                        end
                        if !t_sret
                            sret = false
                        end
                        if readnone
                            continue
                        end
                        if !readonly
                            maywrite = true
                        end
                        if !writeonly
                            mayread = true
                        end
                    end
                end
                if !sret
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: did not have sret when expected")
                        println(io, "f=", string(f))
                        println(io, "inst=", string(inst))
                        println(io, "st=", string(st))
                        println(io, "fop=", string(fop))
                    end
                    throw(AssertionError(msg))
                end

                @assert sret_elty !== nothing
                if temp === nothing
                    nb = IRBuilder()
                    position!(nb, LLVM.at_begin(first(f.blocks)))
                    temp = alloca!(nb, sret_elty)
                end
                if mayread
                    nb = IRBuilder()
                    position!(nb, LLVM.before(st))
                    ld = load!(nb, sret_elty, inst.operands[1])
                    store!(nb, ld, temp)
                end
                if maywrite
                    nb = IRBuilder()
                    position!(nb, LLVM.after(st))
                    ld = load!(nb, sret_elty, temp)
                    si = store!(nb, ld, inst.operands[1])
                    julia_post_cache_store(si.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
                end
            end

            if temp !== nothing
                replace_uses!(inst, temp)
            end
            erase!(inst)
        end
    end
    return nothing
end

function pre_attr!(mod::LLVM.Module, run_attr)
    if run_attr
        for fn in mod.functions
            if isempty(fn.blocks)
                continue
            end
            attrs = collect(fn.function_attributes)
            prevent = any(
                attr.kind == PRESERVEPRIMAL_ATTR_KIND for attr in attrs
            )
            if !prevent
                continue
            end

            if fn.linkage == LLVM.Linkage.Internal
                push!(fn.function_attributes, StringAttribute("restorelinkage_internal"))
                fn.linkage = LLVM.Linkage.External
            end

            if fn.linkage == LLVM.Linkage.Private
                push!(fn.function_attributes, StringAttribute("restorelinkage_private"))
                fn.linkage = LLVM.Linkage.External
            end
            continue

            if !haskey(fn.function_attributes, :noinline)
                push!(fn.function_attributes, EnumAttribute(:noinline))
                push!(fn.function_attributes, StringAttribute("remove_noinline"))
            end

            if !haskey(fn.function_attributes, :optnone)
                push!(fn.function_attributes, EnumAttribute(:optnone))
                push!(fn.function_attributes, StringAttribute("remove_optnone"))
            end
        end
    end
    return nothing

    for fn in collect(mod.functions)
        if isempty(fn.blocks)
            continue
        end
        if fn.linkage != LLVM.Linkage.Internal &&
                fn.linkage != LLVM.Linkage.Private
            continue
        end

        fty = LLVM.FunctionType(fn)
        nfn = LLVM.Function(mod, "enzyme_attr_prev_" * enzymefn.name, fty)
        LLVM.IRBuilder() do builder
            entry = BasicBlock(nfn, "entry")
            position!(builder, LLVM.at_end(entry))
            cv = call!(fn, [LLVM.UndefValue(ty) for ty in fty.parameters])
            push!(res.argument_attributes[1], attr)
            if fty.return_type == LLVM.VoidType()
                ret!(builder)
            else
                ret!(builder, cv)
            end
        end
    end
end

function post_attr!(mod::LLVM.Module, run_attr)
    if run_attr
        for fn in mod.functions
            if haskey(fn.function_attributes, "restorelinkage_internal")
                delete!(fn.function_attributes, "restorelinkage_internal")
                fn.linkage = LLVM.Linkage.Internal
            end

            if haskey(fn.function_attributes, "restorelinkage_private")
                delete!(fn.function_attributes, "restorelinkage_private")
                fn.linkage = LLVM.Linkage.Private
            end

            if haskey(fn.function_attributes, "remove_noinline")
                delete!(fn.function_attributes, :noinline)
                delete!(fn.function_attributes, "remove_noinline")
            end

            if haskey(fn.function_attributes, "remove_optnone")
                delete!(fn.function_attributes, :optnone)
                delete!(fn.function_attributes, "remove_optnone")
            end
        end
    end
    return nothing
end

"""
    constant_aggregate_element(ty, idx)

The type of element `idx` (zero-based) of a constant of type `ty`, or `nothing` when
the constant has no such element: `idx` is past the end, or `ty` is not an array, a
vector or a struct.
"""
function constant_aggregate_element(@nospecialize(ty::LLVM.LLVMType), idx::UInt64)
    idx <= typemax(Cuint) || return nothing
    if isa(ty, LLVM.ArrayType) || isa(ty, LLVM.VectorType)
        return idx < ty.length ? ty.element_type : nothing
    elseif isa(ty, LLVM.StructType)
        elems = collect(ty.elements)
        return idx < length(elems) ? elems[idx + 1] : nothing
    end
    return nothing
end

"""
    constant_gep_path(ty, gep)

The `extractvalue` indices that the `getelementptr` instruction `gep` appends to a path
ending at a constant of type `ty`, together with the type they select, or `nothing` when
`gep` does not address an element of that constant.

It addresses one when it indexes the type of the constant, every index is a constant,
the first is zero, and each of the others is in bounds for the aggregate it indexes.

- A `getelementptr` may step outside its object: to the address past the last element,
  a loop's bound, or to the address before the first one (index `-1`, which reads as
  `2^64 - 1`), from which one-based indexing counts. Such an address names no element.
- With opaque pointers a `getelementptr` names the type it indexes, and that need not
  be the type of the constant: a struct of two `i32` and a `double` can be read as an
  array of two `double`. Its indices then count the elements of another layout.

Either way a load through the address cannot be replaced by the part of the initializer
the indices select, and the address is left alone. The one address that is the same in
every layout is the constant's own: when all indices are zero, the path and the type
stay as they are, whatever type the instruction indexes.
"""
function constant_gep_path(@nospecialize(ty::LLVM.LLVMType), gep::LLVM.GetElementPtrInst)
    ops = gep.operands
    same_type = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep)) == ty
    path = Cuint[]
    for i in 2:length(ops)
        idx = ops[i]
        isa(idx, LLVM.ConstantInt) || return nothing
        idx.value_type.width <= 64 || return nothing
        val = LLVM.API.LLVMConstIntGetZExtValue(idx)
        if i == 2 || !same_type
            val == 0 || return nothing
            continue
        end
        ty = constant_aggregate_element(ty, val)
        ty === nothing && return nothing
        push!(path, val)
    end
    return path, ty
end

function prop_global!(g::LLVM.GlobalVariable)
    newfns = String[]
    changed = false
    # The indices from the initializer down to the constant a pointer addresses, the
    # type of that constant, and a user of the pointer.
    todo = Tuple{Vector{Cuint}, LLVM.LLVMType, LLVM.Value}[]
    for u in g.users
        push!(todo, (Cuint[], g.global_value_type, u))
    end
    while length(todo) > 0
        path, ty, var = pop!(todo)
        if isa(var, LLVM.LoadInst)
            B = IRBuilder()
            position!(B, LLVM.before(var))
            res = g.initializer
            for p in path
                res = extract_value!(B, res, p)
            end
            changed = true
            for u in var.users
                if isa(u, LLVM.CallInst)
                    f2 = u.called_operand
                    if isa(f2, LLVM.Function)
                        push!(newfns, f2.name)
                    end
                end
            end
            if var.value_type != res.value_type
                al = alloca!(B, res.value_type)
                store!(B, res, al)
                res = load!(B, var.value_type, al)
            end
            replace_uses!(var, res)
            eraseInst(var.parent, var)
            continue
        end
        if isa(var, LLVM.AddrSpaceCastInst)
            for u in var.users
                push!(todo, (path, ty, u))
            end
            continue
        end
        if isa(var, LLVM.ConstantExpr) && var.opcode == LLVM.Opcode.AddrSpaceCast
            for u in var.users
                push!(todo, (path, ty, u))
            end
            continue
        end
        if isa(var, LLVM.GetElementPtrInst)
            element = constant_gep_path(ty, var)
            element === nothing && continue
            subpath, subty = element
            subpath = vcat(path, subpath)
            for u in var.users
                push!(todo, (subpath, subty, u))
            end
            continue
        end
    end
    return changed, newfns
end

# From https://llvm.org/doxygen/IR_2Instruction_8cpp_source.html#l00959
function mayWriteToMemory(@nospecialize(inst::LLVM.Instruction); err_is_readonly::Bool = false)::Bool
    # we will ignore fense here
    if isa(inst, LLVM.StoreInst)
        return true
    end
    if isa(inst, LLVM.VAArgInst)
        return true
    end
    if isa(inst, LLVM.AtomicCmpXchgInst)
        return true
    end
    if isa(inst, LLVM.AtomicRMWInst)
        return true
    end
    if isa(inst, LLVM.CatchPadInst)
        return true
    end
    if isa(inst, LLVM.CatchRetInst)
        return true
    end
    if isa(inst, LLVM.CallInst) || isa(inst, LLVM.InvokeInst) || isa(inst, LLVM.CallBrInst)
        for attr in inst.function_attributes
            if attr.kind == READNONE_ATTR_KIND
                return false
            end
            if attr.kind == READONLY_ATTR_KIND
                return false
            end
            # Note out of spec, and only legal in context of removing unused calls
            if attr.kind == "enzyme_error" && err_is_readonly
                return false
            end
        end
        # the call site's own memory effects
        if is_readonly(LLVM.MemoryEffects(inst.memory_effects))
            return false
        end
        return true
    end
    # Ignoring load unordered case
    return false
end

function remove_readonly_unused_calls!(fn::LLVM.Function, next::Set{String})
    calls = LLVM.CallInst[]

    hasUser = false
    for u in fn.uses
        un = u.user

        # Only permit call users
        if !isa(un, LLVM.CallInst)
            return false
        end
        un = un::LLVM.CallInst

        # Passing the fn as an argument is not permitted
        for op in un.arguments
            if op == fn
                return false
            end
        end

        # Something with a user is not permitted
        for u2 in un.uses
            hasUser = true
            break
        end
        push!(calls, un)
    end

    done = Set{LLVM.Function}()
    todo = LLVM.Function[fn]

    while length(todo) != 0
        cur = pop!(todo)
        if cur in done
            continue
        end
        push!(done, cur)

        if is_readonly(cur)
            continue
        end

        if cur.name == "julia.safepoint"
            continue
        end

        if isempty(cur.blocks)
            return false
        end

        err_is_readonly = true

        for bb in cur.blocks
            for inst in bb.instructions
                if !mayWriteToMemory(inst; err_is_readonly)
                    continue
                end
                if isa(inst, LLVM.CallInst)

                    fn2 = inst.called_operand
                    if isa(fn2, LLVM.Function)
                        push!(todo, fn2)
                        continue
                    end
                end
                return false
            end
        end
    end

    changed = set_readonly!(fn)

    if length(calls) == 0 || hasUser || !is_nounwind(fn)
        return changed
    end

    for c in calls
        parentf = c.parent.parent
        push!(next, parentf.name)
        erase!(c)
    end
    push!(next, fn.name)
    return true
end

"""
    needs_return_roots(fn, i)

Whether parameter `i` of `fn` is the `returnRoots` array of an `sret` in the
first parameter.
"""
function needs_return_roots(fn::LLVM.Function, i::Integer)
    i == 2 || return false
    roots = (StringAttribute("enzymejl_returnRoots")).kind
    any(attr -> attr.kind == roots, collect(fn.parameter_attributes[2])) || return false
    sretkind = (LLVM.TypeAttribute(:sret, LLVM.Int32Type())).kind
    return any(attr -> attr.kind == sretkind, collect(fn.parameter_attributes[1]))
end

function propagate_returned!(mod::LLVM.Module; keep_return_roots::Bool = false)
    globs = LLVM.GlobalVariable[]
    for g in mod.globals
        if g.linkage == LLVM.Linkage.Internal ||
                g.linkage == LLVM.Linkage.Private
            if !g.constant
                continue
            end
            push!(globs, g)
        end
    end
    todo = collect(mod.functions)
    while true
        next = Set{String}()
        changed = false
        for g in globs
            tc, tn = prop_global!(g)
            changed |= tc
            for f in tn
                push!(next, f)
            end
        end
        tofinalize = Tuple{LLVM.Function, Bool, Vector{Int64}}[]
        for fn in mod.functions
            if isempty(fn.blocks)
                continue
            end
            if remove_readonly_unused_calls!(fn, next)
                changed = true
            end
            has_user = false
            for u in fn.uses
                has_user = true
                break
            end
            attrs = collect(fn.function_attributes)
            prevent = any(
                attr.kind == PRESERVEPRIMAL_ATTR_KIND for
                    attr in attrs
            )
            # if any(kind(attr) == kind(EnumAttribute(:noinline)) for attr in attrs)
            #     continue
            # end
            argn = nothing
            toremove = Int64[]
            # Don't bother with functions we're about to delete anyways
            if has_user
                for (i, arg) in enumerate(fn.parameters)
                    if any(
                            attr.kind == RETURNED_ATTR_KIND for
                                attr in collect(fn.parameter_attributes[i])
                        )
                        argn = i
                    end

                    # GC lowering takes the argument after an sret for its
                    # returnRoots, so once the calling convention is fixed up the
                    # roots stay as long as the sret does, even if nothing reads
                    # them. Before that the fixup rebuilds any it needs.
                    if keep_return_roots && needs_return_roots(fn, i) && !(0 in toremove)
                        continue
                    end

                    # remove unused sret-like
                    if !prevent &&
                            (
                            fn.linkage == LLVM.Linkage.Internal ||
                                fn.linkage == LLVM.Linkage.Private
                        ) &&
                            any(
                            attr.kind == NOCAPTURE_ATTR_KIND for
                                attr in collect(fn.parameter_attributes[i])
                        )
                        val = nothing
                        illegalUse = false
                        torem = LLVM.Instruction[]

                        for u in fn.uses
                            un = u.user
                            if !isa(un, LLVM.CallInst)
                                illegalUse = true
                                break
                            end
                            if un.called_type != fn.function_type
                                illegalUse = true
                                break
                            end
                            bad = false
                            for op in un.arguments
                                if op == fn
                                    bad = true
                                    break
                                end
                            end
                            if bad
                                illegalUse = true
                                break
                            end
                            op_i = un.operands[i]
                            if !isa(op_i, LLVM.AllocaInst) && !isa(op_i, LLVM.UndefValue) && !isa(op_i, LLVM.PoisonValue)
                                illegalUse = true
                                break
                            end
                            seenfn = false
                            todo = LLVM.Instruction[]
                            if isa(op_i, LLVM.AllocaInst)
                                for u2 in op_i.uses
                                    un2 = u2.user
                                    push!(todo, un2)
                                end
                            end
                            while length(todo) > 0
                                un2 = pop!(todo)
                                if isa(un2, LLVM.BitCastInst)
                                    push!(torem, un2)
                                    for u3 in un2.uses
                                        un3 = u3.user
                                        push!(todo, un3)
                                    end
                                    continue
                                end
                                if isa(un2, LLVM.GetElementPtrInst)
                                    push!(torem, un2)
                                    for u3 in un2.uses
                                        un3 = u3.user
                                        push!(todo, un3)
                                    end
                                    continue
                                end
                                if !isa(un2, LLVM.CallInst)
                                    illegalUse = true
                                    break
                                end
                                ff = un2.called_operand
                                if !isa(ff, LLVM.Function)
                                    illegalUse = true
                                    break
                                end
                                if un2 == un && !seenfn
                                    seenfn = true
                                    continue
                                end
                                intr = ff.intrinsic
                                if intr == LLVM.Intrinsic("llvm.lifetime.start")
                                    push!(torem, un2)
                                    continue
                                end
                                if intr == LLVM.Intrinsic("llvm.lifetime.end")
                                    push!(torem, un2)
                                    continue
                                end
                                if ff.name != "llvm.enzyme.sret_use"
                                    illegalUse = true
                                    break
                                end
                                push!(torem, un2)
                            end
                            if illegalUse
                                break
                            end
                        end
                        if !illegalUse
                            has_use = false
                            for _ in arg.uses
                                has_use = true
                                break
                            end

                            argeltype = if has_use
                                argeltype0 = sret_ty(fn, i, #=btval=# nothing, #=throw_error=# false)
                                if argeltype0 === nothing
                                    illegalUse = true
                                end
                                argeltype0
                            end
                            if !illegalUse
                                for c in reverse(torem)
                                    eraseInst(c.parent, c)
                                end
                                if has_use
                                    B = IRBuilder()
                                    position!(B, LLVM.at_begin(first(fn.blocks)))
                                    al = alloca!(B, argeltype)
                                    if al.value_type != arg.value_type
                                        al = addrspacecast!(B, al, arg.value_type)
                                    end
                                    LLVM.replace_uses!(arg, al)
                                end
                            end
                        end
                    end

                    # interprocedural const prop from callers of arg
                    if !prevent && (
                            fn.linkage == LLVM.Linkage.Internal ||
                                fn.linkage == LLVM.Linkage.Private
                        )
                        val = nothing
                        illegalUse = false
                        for u in fn.uses
                            un = u.user
                            if !isa(un, LLVM.CallInst)
                                illegalUse = true
                                break
                            end
                            if un.called_type != fn.function_type
                                illegalUse = true
                                break
                            end
                            bad = false
                            for op in un.arguments
                                if op == fn
                                    bad = true
                                    break
                                end
                            end
                            if bad
                                illegalUse = true
                                break
                            end
                            op_i = un.operands[i]
                            if isa(op_i, LLVM.UndefValue) || isa(op_i, LLVM.PoisonValue)
                                continue
                            end
                            if op_i == arg
                                continue
                            end
                            if isa(op_i, LLVM.Constant)
                                if val === nothing
                                    val = op_i
                                else
                                    if val != op_i
                                        illegalUse = true
                                        break
                                    end
                                end
                                continue
                            end
                            illegalUse = true
                            break
                        end
                        if !illegalUse
                            if val === nothing
                                val = LLVM.UndefValue(arg.value_type)
                            end
                            for u in arg.users
                                if isa(u, LLVM.CallInst)
                                    f2 = u.called_operand
                                    if isa(f2, LLVM.Function)
                                        push!(next, f2.name)
                                    end
                                end
                                changed = true
                            end
                            LLVM.replace_uses!(arg, val)
                        end
                    end

                    # see if there are no users of the value (excluding recursive/return)
                    if !prevent
                        baduse = false
                        for u in arg.users
                            if argn == i && u isa LLVM.RetInst
                                continue
                            end
                            if !isa(u, LLVM.CallInst)
                                baduse = true
                                break
                            end
                            if u.called_operand != fn
                                baduse = true
                                break
                            end
                            for (si, op) in enumerate(u.operands)
                                if si == i
                                    continue
                                end
                                if op == arg
                                    baduse = true
                                    break
                                end
                            end
                            if baduse
                                break
                            end
                        end
                        if !baduse
                            push!(toremove, i - 1)
                        end
                    end
                end
            end
            illegalUse = !(
                fn.linkage == LLVM.Linkage.Internal ||
                    fn.linkage == LLVM.Linkage.Private
            )
            hasAnyUse = false
            for u in fn.uses
                un = u.user
                if !isa(un, LLVM.CallInst)
                    illegalUse = true
                    continue
                end
                if un.called_type != fn.function_type
                    illegalUse = true
                    continue
                end
                bad = false
                for op in un.arguments
                    if op == fn
                        bad = true
                        break
                    end
                end
                if bad
                    illegalUse = true
                    continue
                end
                if argn !== nothing
                    hasUse = false
                    for u2 in un.uses
                        hasUse = true
                        break
                    end
                    if hasUse
                        changed = true
                        push!(next, un.parent.parent.name)
                        LLVM.replace_uses!(un, un.operands[argn])
                    end
                else
                    for u in un.users
                        if u isa LLVM.CallInst
                            op = u.called_operand
                            if op isa LLVM.Function && op.name == "llvm.enzymefakeread"
                                continue
                            end
                        end
                        hasAnyUse = true
                        break
                    end
                end
            end
            #if the function return has no users whatsoever, remove it
            if argn === nothing &&
                    !hasAnyUse &&
                    fn.function_type.return_type != LLVM.VoidType()
                argn = -1
            end
            if argn === nothing && length(toremove) == 0
                continue
            end
            if !illegalUse
                push!(tofinalize, (fn, argn === nothing, toremove))
            end
        end
        for (fn, keepret, toremove) in tofinalize
            todo = LLVM.CallInst[]
            for u in fn.uses
                un = u.user
                push!(next, un.parent.parent.name)
            end
            delete_writes_into_removed_args(fn, toremove, keepret)
            nm = fn.name
            #try
            nfn = LLVM.Function(
                API.EnzymeCloneFunctionWithoutReturnOrArgs(fn, keepret, toremove),
            )
            for u in fn.uses
                un = u.user
                push!(todo, un)
            end
            for un in todo
                md = un.metadata
                if !keepret && haskey(md, LLVM.MD_range)
                    delete!(md, LLVM.MD_range)
                end
                API.EnzymeSetCalledFunction(un, nfn, toremove)
            end
            eraseInst(mod, fn)
            changed = true
            # catch e
            #    break
            #end
        end
        if !changed
            break
        else
            todo = LLVM.Function[]
            for name in next
                fn = mod.functions[name]
                if fn.linkage == LLVM.Linkage.Internal ||
                        fn.linkage == LLVM.Linkage.Private
                    has_external_user = false
                    for u in fn.uses
                        user_inst = u.user
                        if isa(user_inst, LLVM.Instruction)
                            user_fn = user_inst.parent.parent
                            if user_fn != fn
                                has_external_user = true
                                break
                            end
                        else
                            has_external_user = true
                            break
                        end
                    end
                    if !has_external_user
                        erase!(fn)
                        continue
                    end
                end
                push!(todo, fn)
            end
        end
    end
    return
end

function delete_writes_into_removed_args(fn::LLVM.Function, toremove::Vector{Int64}, keepret::Bool)
    args = collect(fn.parameters)
    if !keepret
        for u in fn.users
            replace_uses!(u, LLVM.UndefValue(u.value_type))
        end
    end
    for tr in toremove
        tr = tr + 1
        todorep = Tuple{LLVM.Instruction, LLVM.Value}[]
        for opv in args[tr].uses
            u = opv.user
            push!(todorep, (u, args[tr]))
        end
        toerase = LLVM.Instruction[]
        while length(todorep) != 0
            cur, cval = pop!(todorep)
            if isa(cur, LLVM.StoreInst)
                if cur.operands[2] == cval
                    erase!(nphi)
                    continue
                end
            end
            if isa(cur, LLVM.GetElementPtrInst) ||
                    isa(cur, LLVM.BitCastInst) ||
                    isa(cur, LLVM.AddrSpaceCastInst)
                for opv in cur.uses
                    u = opv.user
                    push!(todorep, (u, cur))
                end
                continue
            end
            if isa(cur, LLVM.CallInst)
                cf = cur.called_operand
                if cf == fn
                    baduse = false
                    for (i, v) in enumerate(cur.operands)
                        if i - 1 in toremove
                            continue
                        end
                        if v == cval
                            baduse = true
                        end
                    end
                    if !baduse
                        continue
                    end
                end
            end
            if !keepret && cur isa LLVM.RetInst
                cur.operands[1] = LLVM.UndefValue(cval.value_type)
                continue
            end
            throw(AssertionError("Deleting argument with an unknown dependency, $(string(cur)) uses $(string(cval))"))
        end
    end
    return
end

function validate_return_roots!(mod::LLVM.Module)
    for f in mod.functions
        srets = []
        enzyme_srets = Int[]
        enzyme_srets_v = Int[]
        rroots = Int[]
        rroots_v = Int[]
        sretkind = :sret
        for (i, a) in enumerate(f.parameters)
            for attr in collect(f.parameter_attributes[i])
                if isa(attr, StringAttribute)
                    if attr.kind == "enzymejl_returnRoots"
                        push!(rroots, i)
                    end
                    if attr.kind == "enzymejl_returnRoots_v"
                        push!(rroots_v, i)
                    end
                    if attr.kind == "enzyme_sret"
                        push!(enzyme_srets, i)
                    end
                    if attr.kind == "enzyme_sret_v"
                        push!(enzyme_srets, i)
                    end
                end
                if attr.kind == sretkind
                    push!(srets, (i, attr))
                end
            end
        end
        if length(enzyme_srets) >= 1 && length(srets) == 0
            @assert enzyme_srets[1] == 1
            VT = LLVM.VoidType()
            if length(enzyme_srets) == 1 &&
                    f.function_type.return_type == VT &&
                    length(enzyme_srets_v) == 0
                # Upgrading to sret requires writeonly
                if !any(
                        attr.kind == :writeonly for
                            attr in collect(f.parameter_attributes[1])
                    )
                    msg = sprint() do io::IO
                        println(io, "Enzyme internal error (not writeonly sret)")
                        println(io, string(f))
                        println(
                            io,
                            "collect(parameter_attributes(f, 1))=",
                            collect(f.parameter_attributes[1]),
                        )
                    end
                    throw(AssertionError(msg))
                end

                alty = nothing
                for u in f.users
                    @assert isa(u, LLVM.CallInst)
                    @assert u.called_operand == f
                    alop = u.operands[1]
                    if !isa(alop, LLVM.AllocaInst)
                        msg = sprint() do io::IO
                            println(io, "Enzyme internal error (!isa(alop, LLVM.AllocaInst))")
                            println(io, "alop=", alop)
                            println(io, "u=", u)
                            println(io, "f=", string(f))
                        end
                        throw(AssertionError(msg))

                    end
                    @assert isa(alop, LLVM.AllocaInst)
                    nty = alop.allocated_type
                    if alty === nothing
                        alty = nty
                    else
                        @assert alty == nty
                    end
                    attr = TypeAttribute(:sret, alty)
                    push!(u.argument_attributes[1], attr)
                    delete!(u.argument_attributes[1], "enzyme_sret")
                end
                @assert alty !== nothing
                attr = TypeAttribute(:sret, alty)

                push!(f.parameter_attributes[1], attr)
                delete!(f.parameter_attributes[1], "enzyme_sret")
                srets = [(1, attr)]
                enzyme_srets = Int[]
            else

                enzyme_srets2 = Int[]
                for idx in enzyme_srets
                    alty = nothing
                    bad = false
                    for u in f.users
                        @assert isa(u, LLVM.CallInst)
                        @assert u.called_operand == f
                        alop = u.operands[1]
                        @assert isa(alop, LLVM.AllocaInst)
                        nty = alop.allocated_type
                        if any_jltypes(nty)
                            bad = true
                        end
                        delete!(u.argument_attributes[idx], "enzyme_sret")
                    end
                    if !bad
                        delete!(f.parameter_attributes[idx], "enzyme_sret")
                    else
                        push!(enzyme_srets2, idx)
                    end
                end
                enzyme_srets = enzyme_srets2

                if length(enzyme_srets) != 0
                    msg = sprint() do io::IO
                        println(io, "Enzyme internal error (length(enzyme_srets) != 0)")
                        println(io, "f=", string(f))
                        println(io, "enzyme_srets=", enzyme_srets)
                        println(io, "enzyme_srets_v=", enzyme_srets_v)
                        println(io, "srets=", srets)
                        println(io, "rroots=", rroots)
                        println(io, "rroots_v=", rroots_v)
                    end
                    throw(AssertionError(msg))
                end
            end
        end
        @assert length(enzyme_srets_v) == 0
        for (i, attr) in srets
            @assert i == 1
        end
        for i in rroots
            @assert length(srets) != 0
            @assert i == 2
        end
        # illegal
        for i in rroots_v
            @assert false
        end
    end
    return
end

function checkNoAssumeFalse(mod::LLVM.Module, shouldshow::Bool = false)
    for f in mod.functions
        for bb in f.blocks, inst in bb.instructions
            if !isa(inst, LLVM.CallInst)
                continue
            end
            cf = inst.called_function
            if cf === nothing || cf.intrinsic != LLVM.Intrinsic("llvm.assume")
                continue
            end
            op = inst.operands[1]
            if isa(op, LLVM.ConstantInt)
                op2 = convert(Bool, op)
                if !op2
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: non-constant assume condition")
                        println(io, "mod=", string(mod))
                        println(io, "f=", string(f))
                        println(io, "bb=", string(bb))
                        println(io, "op2=", string(op2))
                    end
                    throw(AssertionError(msg))
                end
            end
            if isa(op, LLVM.ICmpInst)
                if op.predicate == LLVM.IntPredicate.NE &&
                        op.operands[1] == op.operands[2]
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: non-icmp assume condition")
                        println(io, "mod=", string(mod))
                        println(io, "f=", string(f))
                        println(io, "bb=", string(bb))
                        println(io, "op=", string(op))
                    end
                    throw(AssertionError(msg))
                end
            end
        end
    end
    return
end

function removeDeadArgs!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, post_gc_fixup::Bool; attributor::Bool = RunAttributor[])
    # We need to run globalopt first. This is because remove dead args will otherwise
    # take internal functions and replace their args with undef. Then on LLVM up to
    # and including 12 (but fixed 13+), Attributor will incorrectly change functions that
    # call code with undef to become unreachable, even when there exist other valid
    # callsites. See: https://godbolt.org/z/9Y3Gv6q5M
    run!(GlobalDCEPass(), mod)

    # Prevent dead-arg-elimination of functions which we may require args for in the derivative
    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[], vararg = true)
    if LLVM.version().major <= 15
        # These fake calls must not write memory they are passed, yet each also
        # writes inaccessible memory. On LLVM 15 a call that writes nothing is
        # deleted once it is nounwind, which InstCombine marks every call in a
        # nounwind function: InstCombine assumes an `llvm.` call that only
        # reads memory will return, and the Attributor deletes a nounwind
        # read-only call to any other function.
        func, _ = get_function!(
            mod,
            "llvm.enzymefakeuse",
            funcT,
            LLVM.Attribute[EnumAttribute(:inaccessiblememonly), EnumAttribute(:nofree)],
        )
        # The read of the pointer argument is stated at the call site.
        rfunc, _ = get_function!(
            mod,
            "llvm.enzymefakeread",
            funcT,
            LLVM.Attribute[
                EnumAttribute(:nofree),
                EnumAttribute(:inaccessiblemem_or_argmemonly),
            ],
        )
        sfunc, _ = get_function!(
            mod,
            "llvm.enzyme.sret_use",
            funcT,
            LLVM.Attribute[
                EnumAttribute(:nofree),
                EnumAttribute(:inaccessiblemem_or_argmemonly),
            ],
        )
        wfunc, _ = get_function!(
            mod,
            "llvm.enzymefakewrite",
            funcT,
            LLVM.Attribute[
                EnumAttribute(:writeonly),
                EnumAttribute(:nofree),
                EnumAttribute(:argmemonly),
            ],
        )
        rwfunc, _ = get_function!(
            mod,
            "llvm.enzymefakereadwrite",
            funcT,
            LLVM.Attribute[
                EnumAttribute(:nofree),
                EnumAttribute(:inaccessiblemem_or_argmemonly),
            ],
        )
    else
        func, _ = get_function!(
            mod, "llvm.enzymefakeuse", funcT, LLVM.Attribute[EnumAttribute(:nofree)];
            memory_effects = LLVM.MemoryEffects(:none),
        )
        rfunc, _ = get_function!(
            mod, "llvm.enzymefakeread", funcT, LLVM.Attribute[EnumAttribute(:nofree)];
            memory_effects = LLVM.MemoryEffects(argmem = :read),
        )
        sfunc, _ = get_function!(
            mod, "llvm.enzyme.sret_use", funcT, LLVM.Attribute[EnumAttribute(:nofree)];
            memory_effects = LLVM.MemoryEffects(argmem = :read),
        )
        wfunc, _ = get_function!(
            mod, "llvm.enzymefakewrite", funcT, LLVM.Attribute[EnumAttribute(:nofree)];
            memory_effects = LLVM.MemoryEffects(argmem = :write),
        )
        rwfunc, _ = get_function!(
            mod, "llvm.enzymefakereadwrite", funcT, LLVM.Attribute[EnumAttribute(:nofree)];
            memory_effects = LLVM.MemoryEffects(argmem = :read, inaccessiblemem = :write),
        )
    end

    for fn in mod.functions
        if isempty(fn.blocks)
            continue
        end

        rt = fn.function_type.return_type
        if rt isa LLVM.PointerType && rt.addrspace == 10
            for u in fn.users
                if isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    position!(B, LLVM.after(u))
                    cl = call!(B, funcT, rfunc, LLVM.Value[u])
                    push!(cl.argument_attributes[1], EnumAttribute(:nocapture))
                    push!(cl.argument_attributes[1], EnumAttribute(:readonly))
                end
            end
        end

        sretkind = :sret

        # Ensure that interprocedural optimizations do not delete the use of gc sret or returnRoots, this will only occur on 2.
        if post_gc_fixup
            for idx in (1, 2)
                if length(collect(fn.parameters)) >= idx && any(
                        (

                            (attr.kind == sretkind && any_jltypes(attr.value)) ||
                                attr.kind == "enzymejl_returnRoots"
                        ) for attr in collect(fn.parameter_attributes[idx])
                    )
                    if !isempty(fn.blocks)
                        B = IRBuilder()
                        position!(B, LLVM.at_begin(fn.entry))
                        p = fn.parameters[idx]
                        cl = call!(B, funcT, wfunc, LLVM.Value[p])
                        if isa(p.value_type, LLVM.PointerType)
                            push!(cl.argument_attributes[1], EnumAttribute(:nocapture))
                        end
                    end
                    for u in fn.users
                        if !isa(u, LLVM.CallInst)
                            # TODO investigate if the inttoptr store that comes from reference caller poses an issue.
                            continue
                            msg = sprint() do io
                                println(io, "Unknown user of fn: ", string(u))
                                println(io, "fn: ", string(fn))
                                println(io, "mod: ", string(fn.parent))
                            end
                            throw(AssertionError(msg))
                        end
                        B = IRBuilder()
                        position!(B, LLVM.after(u))
                        inp = u.operands[idx]
                        cl = call!(B, funcT, rwfunc, LLVM.Value[inp])
                        if isa(inp.value_type, LLVM.PointerType)
                            push!(cl.argument_attributes[1], EnumAttribute(:nocapture))
                            push!(cl.argument_attributes[1], EnumAttribute(:readonly))
                        end
                    end
                end
            end
        end
        for idx in (1, 2)
            if length(collect(fn.parameters)) < idx
                continue
            end
            attrs = collect(fn.parameter_attributes[idx])
            if any(
                    (
                        attr.kind == sretkind ||
                            attr.kind == "enzyme_sret" ||
                            attr.kind == "enzyme_sret_v"
                    ) for attr in attrs
                ) && any_jltypes(sret_ty(fn, idx))
                for u in fn.users
                    if isa(u, LLVM.ConstantExpr)
                        for u in u.users
                            if !isa(u, LLVM.CallInst)
                                continue
                            end
                            @assert isa(u, LLVM.CallInst)
                            B = IRBuilder()
                            position!(B, LLVM.after(u))
                            inp = u.operands[idx]
                            cl = call!(B, funcT, sfunc, LLVM.Value[inp])
                            if isa(inp.value_type, LLVM.PointerType)
                                push!(cl.argument_attributes[1], EnumAttribute(:nocapture))
                                push!(cl.argument_attributes[1], EnumAttribute(:readonly))
                            end
                        end
                        continue
                    end
                    if !isa(u, LLVM.CallInst)
                        continue
                    end
                    @assert isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    position!(B, LLVM.after(u))
                    inp = u.operands[idx]
                    cl = call!(B, funcT, sfunc, LLVM.Value[inp])
                    if isa(inp.value_type, LLVM.PointerType)
                        push!(cl.argument_attributes[1], EnumAttribute(:nocapture))
                        push!(cl.argument_attributes[1], EnumAttribute(:readonly))
                    end
                end
            end
        end
        attrs = collect(fn.function_attributes)
        prevent = any(
            attr.kind == PRESERVEPRIMAL_ATTR_KIND for attr in attrs
        )
        # && any(kind(attr) == kind(StringAttribute("enzyme_math")) for attr in attrs)
        if prevent
            B = IRBuilder()
            position!(B, LLVM.at_begin(first(fn.blocks)))
            call!(B, funcT, func, LLVM.Value[p for p in fn.parameters])
        end
    end
    propagate_returned!(mod; keep_return_roots = post_gc_fixup)
    LLVM.@dispose pb = PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, RestoreAllocaType())
        add!(pb, ModulePassManager()) do mpm
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
                add!(fpm, SROAPass())
                add!(fpm, EarlyCSEPass())
            end
        end
        LLVM.run!(pb, mod)
    end
    propagate_returned!(mod; keep_return_roots = post_gc_fixup)
    pre_attr!(mod, attributor)
    if attributor
        API.EnzymeDetectReadonlyOrThrow(mod)
        LLVM.@dispose pb = PassBuilder() begin
            register!(pb, EnzymeAttributorPass())
            add!(pb, ModulePassManager()) do mpm
                add!(mpm, EnzymeAttributorPass())
            end
            LLVM.run!(pb, mod)
        end
    end
    propagate_returned!(mod; keep_return_roots = post_gc_fixup)
    LLVM.@dispose pb = PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, EnzymeAttributorPass())
        register!(pb, RestoreAllocaType())
        add!(pb, ModulePassManager()) do mpm
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
                add!(fpm, SROAPass())
            end
            if attributor
                add!(mpm, EnzymeAttributorPass())
            end
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, EarlyCSEPass())
            end
        end
        LLVM.run!(pb, mod)
    end
    API.EnzymeDetectReadonlyOrThrow(mod)
    post_attr!(mod, attributor)
    propagate_returned!(mod; keep_return_roots = post_gc_fixup)

    for u in rwfunc.users
        eraseInst(u.parent, u)
    end
    eraseInst(mod, rwfunc)
    for u in wfunc.users
        eraseInst(u.parent, u)
    end
    eraseInst(mod, wfunc)
    for u in rfunc.users
        eraseInst(u.parent, u)
    end
    eraseInst(mod, rfunc)
    for u in sfunc.users
        eraseInst(u.parent, u)
    end
    eraseInst(mod, sfunc)
    for u in func.users
        eraseInst(u.parent, u)
    end
    return eraseInst(mod, func)
end

function safe_atomic_to_regular_store!(f::LLVM.Function)
    changed = false
    for bb in f.blocks, inst in bb.instructions
        if isa(inst, LLVM.StoreInst)
            continue
        end
        if !haskey(inst.metadata, "enzymejl_atomicgc")
            continue
        end
        Base.delete!(inst.metadata, "enzymejl_atomicgc")
        inst.syncscope = LLVM.SyncScope("system")
        inst.ordering = LLVM.AtomicOrdering.NotAtomic
        changed = true
    end
    return changed
end

function replace_builtin_fptr!(mod::LLVM.Module, enzyme_ctx::EnzymeContext)
    if !haskey(mod.functions, "jl_get_builtin_fptr")
        return false
    end
    jl_get_builtin_fptr_fn = mod.functions["jl_get_builtin_fptr"]

    to_replace_fptr = Tuple{LLVM.CallInst, LLVM.Value}[]

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()
    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])

    for f in collect(mod.functions)
        if isempty(f.blocks)
            continue
        end
        for bb in f.blocks, inst in bb.instructions
            if isa(inst, LLVM.CallInst)
                if inst.called_operand == jl_get_builtin_fptr_fn
                    arg1 = inst.operands[1]
                    legal, obj = absint(arg1, enzyme_ctx)
                    if legal
                        if isa(obj, DataType) && isdefined(obj, :instance)
                            obj = obj.instance
                        end
                        if isa(obj, Core.Builtin)
                            builtin_name = string(nameof(obj))
                            builtin_c_name = "jl_f_" * builtin_name
                            if haskey(mod.functions, "ijl_f_" * builtin_name)
                                builtin_c_name = "ijl_f_" * builtin_name
                            end
                            # Declare / Get the function
                            builtin_fn, _ = get_function!(mod, builtin_c_name, generic_FT)

                            # Cast fptr itself for any remaining uses
                            casted_val = builtin_fn
                            if builtin_fn.value_type != inst.value_type
                                B_fptr = IRBuilder()
                                position!(B_fptr, LLVM.before(inst))
                                casted_val = LLVM.pointercast!(B_fptr, builtin_fn, inst.value_type)
                            end
                            push!(to_replace_fptr, (inst, casted_val))
                        end
                    end
                end
            end
        end
    end

    changed = false
    for (inst, casted_val) in to_replace_fptr
        replace_uses!(inst, casted_val)
        eraseInst(inst.parent, inst)
        changed = true
    end
    return changed
end

function get_callee(inst::LLVM.CallInst)
    fn = inst.called_operand
    while true
        if isa(fn, LLVM.Function)
            return fn
        elseif isa(fn, LLVM.ConstantExpr)
            opc = fn.opcode
            if opc == LLVM.Opcode.BitCast || opc == LLVM.Opcode.AddrSpaceCast
                fn = fn.operands[1]
                continue
            end
        end
        break
    end
    return nothing
end

# LLVM's Inliner pass refuses to inline functions if their call sites have "jl_roots" operand bundles.
# This pass finds all calls to alwaysinline functions carrying "jl_roots", extracts the roots,
# wraps the call site in gc_preserve_begin/gc_preserve_end block pairs, and removes the "jl_roots" bundle
# from the call instruction itself. This preserves GC rooting safety while enabling successful inlining.
# This prevents JIT cross-module calling convention mismatches (e.g. DAE optimizing functions to fastcc
# but leaving call declarations as ccc).
function remove_alwaysinline_roots!(mod::LLVM.Module)
    changed = false
    for f in collect(mod.functions)
        if isempty(f.blocks)
            continue
        end

        B = IRBuilder()
        to_rewrite = []

        for bb in f.blocks, inst in bb.instructions
            if isa(inst, LLVM.CallInst)
                callee = get_callee(inst)
                if callee !== nothing && isa(callee, LLVM.Function) && haskey(callee.function_attributes, :alwaysinline)
                    has_roots = false
                    roots = LLVM.Value[]

                    other_bundles = OperandBundle[]

                    for bunduse in inst.operand_bundles
                        if bunduse.tag == "jl_roots"
                            has_roots = true
                            for val in bunduse.inputs
                                push!(roots, val)
                            end
                        else
                            push!(other_bundles, bunduse)
                        end
                    end

                    if has_roots
                        push!(to_rewrite, (inst, roots, other_bundles))
                    end
                end
            end
        end

        for (inst, roots, other_bundles) in to_rewrite
            position!(B, LLVM.before(inst))
            token = emit_gc_preserve_begin(B, roots)

            prevname = inst.name
            inst.name = ""

            newinst = call!(
                B,
                inst.called_type,
                inst.called_operand,
                collect(inst.arguments),
                other_bundles,
                prevname,
            )

            append!(newinst.function_attributes, inst.function_attributes)
            append!(newinst.return_attributes, inst.return_attributes)
            for i in 1:(length(inst.arguments))
                append!(newinst.argument_attributes[i], inst.argument_attributes[i])
            end
            API.EnzymeCopyMetadata(newinst, inst)
            newinst.callconv = inst.callconv

            emit_gc_preserve_end(B, token)

            replace_uses!(inst, newinst)
            LLVM.erase!(inst)
            changed = true
        end
    end
    return changed
end


function is_cast(v::LLVM.Value)
    if isa(v, LLVM.AddrSpaceCastInst) || isa(v, LLVM.BitCastInst)
        return true
    end
    if isa(v, LLVM.ConstantExpr)
        op = v.opcode
        return op == LLVM.Opcode.AddrSpaceCast || op == LLVM.Opcode.BitCast
    end
    return false
end

function evaluates_to_nothing(inst::LLVM.Value)
    if isa(inst, LLVM.LoadInst)
        ptr = inst.operands[1]
        while is_cast(ptr)
            ptr = ptr.operands[1]
        end
        return isa(ptr, LLVM.GlobalVariable) && ptr.name == "jl_nothing"
    elseif is_cast(inst)
        return evaluates_to_nothing(inst.operands[1])
    end
    return false
end

function evaluates_to_nothing_addr(val::LLVM.Value)
    if isa(val, LLVM.ConstantExpr) && val.opcode == LLVM.Opcode.IntToPtr
        val = val.operands[1]
    end
    if isa(val, LLVM.ConstantInt) && val.value_type.width == sizeof(Int) * 8
        nothing_addr = unsafe_load(cglobal(:jl_nothing, Ptr{Cvoid}))
        return convert(UInt, val) == reinterpret(UInt, nothing_addr)
    end
    return false
end

function is_nothing_val(val::LLVM.Value)
    if isa(val, LLVM.GlobalVariable)
        return false
    end
    if evaluates_to_nothing(val)
        return true
    end

    ptr = val
    while is_cast(ptr)
        ptr = ptr.operands[1]
    end
    return evaluates_to_nothing_addr(ptr)
end

function replace_nothing_loads!(mod::LLVM.Module)
    ejl_nothing = unsafe_nothing_to_llvm(mod)
    for f in mod.functions
        if isempty(f.blocks)
            continue
        end

        to_replace = LLVM.Instruction[]
        for bb in f.blocks, inst in bb.instructions
            if is_nothing_val(inst)
                push!(to_replace, inst)
            end

            for (idx, op) in enumerate(inst.operands)
                if !isa(op, LLVM.Instruction) && is_nothing_val(op)
                    replacement = ejl_nothing
                    if ejl_nothing.value_type != op.value_type
                        if isa(op.value_type, LLVM.PointerType) && ejl_nothing.value_type.addrspace != op.value_type.addrspace
                            replacement = LLVM.const_addrspacecast(ejl_nothing, op.value_type)
                        else
                            replacement = LLVM.const_bitcast(ejl_nothing, op.value_type)
                        end
                    end
                    inst.operands[idx] = replacement
                end
            end
        end

        for inst in to_replace
            replacement = ejl_nothing
            if ejl_nothing.value_type != inst.value_type
                if isa(inst.value_type, LLVM.PointerType) && ejl_nothing.value_type.addrspace != inst.value_type.addrspace
                    replacement = LLVM.const_addrspacecast(ejl_nothing, inst.value_type)
                else
                    replacement = LLVM.const_bitcast(ejl_nothing, inst.value_type)
                end
            end

            LLVM.replace_uses!(inst, replacement)
        end

        for inst in to_replace
            if isempty(inst.uses)
                erase!(inst)
            end
        end
    end
    return
end
