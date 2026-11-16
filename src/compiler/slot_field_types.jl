# The Julia type of a pointer reloaded from a stack slot.
#
# A `Const` argument passed by value and stored into a stack slot, such as the
# tuple a vararg method builds of its arguments, is read back from the slot by
# the callee inlined into the calling-convention wrapper. The callee's safepoint
# and its fences sit between the stores and the loads, so GVN leaves the loads in
# place, and `abs_typeof` cannot type a value loaded from a slot that carries no
# Julia type. The mixed-activity handler then cannot see that such a value, for
# instance a `Vector{Int}` of row indices, holds no derivative. It raises
# `EnzymeRuntimeActivityError` where the value meets active data: stored into the
# by-reference copy of a `view` of an active array, or merged with Base's
# `unalias` copy of it.
#
# When every write into a pointer field of a slot stores values that `abs_typeof`
# gives one Julia type, or clears it (a zero `memset`, or a stored null pointer),
# and no code this pass does not see can write into the slot, each load of the
# field returns a value of that type or null. The loads are tagged
# with it (`SLOT_FIELD_TYPE`, read by `abs_typeof`). A call may read the slot
# through argument positions that are `readonly` and `nocapture`; any other use
# of the slot leaves its loads untyped.

# Record the loads and writes of the slot memory that `ptr` points `offset` bytes
# into. `writes` holds (offset, length, stored value), with `nothing` for a write
# of unknown values and `false` for a zero `memset` or a stored zero constant.
# Returns false if code this walk does not see could write into the slot.
function slot_field_accesses!(
        loads::Vector{Tuple{LLVM.LoadInst, Int}},
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}},
        @nospecialize(ptr::LLVM.Value), offset::Int, dl::LLVM.DataLayout,
    )::Bool
    for u in ptr.uses
        user = u.user
        if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst)
            slot_field_accesses!(loads, writes, user, offset, dl) || return false
        elseif isa(user, LLVM.GetElementPtrInst) && user.operands[1] == ptr
            off = constant_gep_offset(user)
            off === nothing && return false
            slot_field_accesses!(loads, writes, user, offset + off, dl) || return false
        elseif isa(user, LLVM.LoadInst)
            push!(loads, (user, offset))
        elseif isa(user, LLVM.StoreInst) && user.operands[2] == ptr && user.operands[1] != ptr
            val = user.operands[1]
            len = Int(LLVM.storage_size(dl, val.value_type))
            # A stored null pointer (or any all-zero constant) clears the field,
            # as a zero `memset` does.
            push!(writes, (offset, len, isa(val, LLVM.Constant) && LLVM.isnull(val) ? false : val))
        elseif isa(user, LLVM.CallInst)
            slot_field_call!(writes, user, ptr, offset) || return false
        else
            return false
        end
    end
    return true
end

function slot_field_call!(
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}},
        call::LLVM.CallInst, @nospecialize(ptr::LLVM.Value), offset::Int,
    )::Bool
    callee = call.called_operand
    callee == ptr && return false
    args = arguments(call)
    if isa(callee, LLVM.Function)
        intr = LLVM.API.LLVMGetIntrinsicID(callee)
        if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
                intr == LLVM.Intrinsic("llvm.lifetime.end").id
            return true
        end
        # What libEnzyme makes of a slot before differentiating, where
        # `constant_slot_load` reads it: renamed lifetime markers and, for a slot it moves
        # to the heap (`enzyme_fromstack`), a write barrier per pointer stored into it,
        # which writes nothing into it, and a call that zeroes it.
        nm = callee.name
        (nm == "llvm.enzyme.lifetime_start" || nm == "llvm.enzyme.lifetime_end") && return true
        (nm == "julia.write_barrier" && args[1] == ptr) && return true
        if haskey(call.metadata, "enzyme_zerostack")
            (args[1] == ptr && isa(args[3], LLVM.ConstantInt)) || return false
            push!(writes, (offset, convert(Int, args[3]), false))
            return true
        end
        if intr == LLVM.Intrinsic("llvm.memset").id
            (args[1] == ptr && args[2] != ptr && isa(args[3], LLVM.ConstantInt)) || return false
            zero = isa(args[2], LLVM.ConstantInt) && convert(Int, args[2]) == 0
            push!(writes, (offset, convert(Int, args[3]), zero ? false : nothing))
            return true
        end
        if intr == LLVM.Intrinsic("llvm.memcpy").id || intr == LLVM.Intrinsic("llvm.memmove").id
            # Copying out of the slot only reads it.
            if args[1] == ptr
                isa(args[3], LLVM.ConstantInt) || return false
                push!(writes, (offset, convert(Int, args[3]), nothing))
            end
            return true
        end
    end
    positions = Int[i for (i, a) in enumerate(args) if a == ptr]
    isempty(positions) && return false
    decay_args_readonly(call, callee, positions) || return false
    for i in positions
        attrs = collect(argument_attributes(call, i))
        if isa(callee, LLVM.Function)
            append!(attrs, collect(callee.parameter_attributes[i]))
        end
        nocapture = false
        for attr in attrs
            attr.kind == NOCAPTURE_ATTR_KIND && (nocapture = true)
        end
        nocapture || return false
    end
    return true
end

# The Julia type and representation of the values a pointer field `off` bytes
# into a slot holds, or `nothing`. `seen` holds the slots being typed
# (`slot_reload_type`).
function slot_field_type(
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}}, off::Int,
        enzyme_ctx::Union{EnzymeContext, Nothing},
        seen::Set{LLVM.Instruction} = Set{LLVM.Instruction}(),
    )
    found = nothing
    for (woff, len, val) in writes
        (woff < off + sizeof(Int) && off < woff + len) || continue
        if val === false
            # A null pointer, from a zero `memset` or a zero store that covers
            # the field.
            (woff <= off && off + sizeof(Int) <= woff + len) || return nothing
            continue
        end
        (val isa LLVM.Value && woff == off && len == sizeof(Int)) || return nothing
        legal, T, byref = abs_typeof(val, enzyme_ctx, false, seen)
        legal || return nothing
        (found === nothing || found == (T, byref)) || return nothing
        found = (T, byref)
    end
    return found
end

# The Julia type and representation of the values that `ld`, a load of a tracked
# pointer from a stack slot, returns by the rule above, read off the function as it
# stands; `nothing` if the rule gives none. `abs_typeof` asks this of a load that
# carries no `SLOT_FIELD_TYPE` tag. It answers only in a module whose reloads
# `type_slot_field_loads!` has typed (`SLOT_TYPED_MODULE`), so it gives a type exactly
# where that pass gives one: not under runtime activity, and not before the pass.
#
# libEnzyme differentiates a copy of the function that it first simplifies. SROA there
# splits a slot into one alloca per field and writes new loads, which carry no tag, and
# a slot the reverse pass may need moves to the heap (`enzyme_fromstack`), declared an
# array of `Any`. A slot that a zero store initializes cannot become a value: the store
# writes an integer and the loads read a tracked pointer, which no cast joins. On Julia
# 1.12 that keeps two kinds of roots slot on the stack: that of an argument an exception
# handler may read, which Julia zero-initializes, and that of the argument tuple of a
# vararg method, which is zeroed before the arguments are stored. What asks for the type
# of their reloads while libEnzyme differentiates (the mixed-activity handler, for one)
# then met them untyped, or typed `Any` from the declaration of the heap slot.
#
# A stored value that is itself a reload, or a merge of reloads, is typed the same way
# through `abs_typeof`, as the fixpoint of `type_slot_field_loads!` does. `seen` holds
# the phis and the slots being typed: a slot whose type would rest on its own reloads
# stays untyped.
function slot_reload_type(
        ld::LLVM.LoadInst, enzyme_ctx::Union{EnzymeContext, Nothing}, seen = Set{LLVM.Instruction}(),
    )
    is_tracked_pointer(ld.value_type) || return nothing
    mod = ld.parent.parent.parent
    haskey(mod.metadata, SLOT_TYPED_MODULE) || return nothing
    slot, _ = constant_field_pointer(ld.operands[1])
    is_stack_slot(slot) || return nothing
    slot in seen && return nothing
    loads = Tuple{LLVM.LoadInst, Int}[]
    writes = Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}[]
    slot_field_accesses!(loads, writes, slot, 0, mod.datalayout) || return nothing
    inner = Set{LLVM.Instruction}(seen)
    push!(inner, slot)
    for (other, off) in loads
        other == ld && return slot_field_type(writes, off, enzyme_ctx, inner)
    end
    return nothing
end

function type_slot_field_loads!(mod::LLVM.Module, enzyme_ctx::Union{EnzymeContext, Nothing})
    # From here on a reload without a tag is typed when asked (`slot_reload_type`).
    get!(mod.metadata, SLOT_TYPED_MODULE)
    dl = mod.datalayout
    slots = Tuple{Vector{Tuple{LLVM.LoadInst, Int}}, Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}}}[]
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        isa(inst, LLVM.AllocaInst) || continue
        loads = Tuple{LLVM.LoadInst, Int}[]
        writes = Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}[]
        slot_field_accesses!(loads, writes, inst, 0, dl) || continue
        pointers = Tuple{LLVM.LoadInst, Int}[]
        for (ld, off) in loads
            is_tracked_pointer(ld.value_type) && push!(pointers, (ld, off))
        end
        isempty(pointers) || push!(slots, (pointers, writes))
    end
    # A value stored into one slot may itself be loaded from another.
    typed = 0
    changed = true
    while changed
        changed = false
        for (loads, writes) in slots, (ld, off) in loads
            haskey(ld.metadata, SLOT_FIELD_TYPE) && continue
            found = slot_field_type(writes, off, enzyme_ctx)
            found === nothing && continue
            T, byref = found
            ld.metadata[SLOT_FIELD_TYPE] = MDNode(
                LLVM.Metadata[
                    MDString(string(convert(UInt, unsafe_to_pointer(T)))),
                    MDString(string(Int(byref))),
                ]
            )
            typed += 1
            changed = true
        end
    end
    return typed
end
