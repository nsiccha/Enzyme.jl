# Find an initialization of this array's Memory field with the shared empty
# Memory singleton. Follow only constant field addresses of this allocation;
# a PHI, a loaded pointer, or a variable index does not establish this fact.
function stores_empty_array_memory(
        @nospecialize(value::LLVM.Value),
        offset::Int,
        memory_offset::Int,
        enzyme_ctx::Union{EnzymeContext, Nothing},
        seen::Set{LLVM.Value},
    )
    value in seen && return false
    push!(seen, value)
    for use in LLVM.uses(value)
        user = LLVM.user(use)
        if user isa LLVM.AddrSpaceCastInst || user isa LLVM.BitCastInst
            stores_empty_array_memory(user, offset, memory_offset, enzyme_ctx, seen) && return true
        elseif user isa LLVM.GetElementPtrInst && operands(user)[1] == value
            field_offset = constant_gep_offset(user)
            field_offset === nothing && continue
            0 <= field_offset <= memory_offset - offset || continue
            next_offset = offset + field_offset
            stores_empty_array_memory(user, next_offset, memory_offset, enzyme_ctx, seen) && return true
        elseif user isa LLVM.StoreInst && operands(user)[2] == value && offset == memory_offset
            legal, object = absint(operands(user)[1], enzyme_ctx)
            legal && is_memory_instance(unbind(object)) && return true
        end
    end
    return false
end

# An empty allocation has no element storage to describe. In particular,
# _collect's empty Array{Any} must not contribute a pointer element layout to
# the same return PHI as a nonempty array of inline floating-point structs.
# Keep the array and Memory headers, but omit element facts from allocations
# initialized with the empty Memory singleton. Later stores and allocations
# still provide their own type information if an initially empty array grows.
function relax_empty_array_types!(mod::LLVM.Module, enzyme_ctx::Union{EnzymeContext, Nothing})
    @static if VERSION < v"1.11-"
        return 0
    else
        changed = 0
        ctx = context(mod)
        dl = string(LLVM.datalayout(mod))
        for f in functions(mod), bb in blocks(f), inst in instructions(bb)
            inst isa LLVM.CallInst || continue
            callee = called_operand(inst)
            callee isa LLVM.Function && name(callee) == "julia.gc_alloc_obj" || continue
            legal, array_type, _ = abs_typeof(inst, enzyme_ctx)
            legal && array_type <: Array && isconcretetype(array_type) || continue
            memory_offset = Int(fieldoffset(array_type, 1) + fieldoffset(fieldtype(array_type, 1), 2))
            stores_empty_array_memory(inst, 0, memory_offset, enzyme_ctx, Set{LLVM.Value}()) || continue

            # Union{} has no elements, but has the same array header and Memory
            # header layout. Copy the cached tree before wrapping the GC pointer.
            tt = copy(typetree_in_world(enzyme_world(), Array{Union{}, ndims(array_type)}, ctx, dl))
            merge!(tt, TypeTree(API.DT_Pointer, ctx))
            only!(tt, -1)
            LLVM.API.LLVMRemoveCallSiteStringAttribute(inst, LLVM.API.LLVMAttributeReturnIndex, "enzyme_type", 11)
            LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeReturnIndex, StringAttribute("enzyme_type", string(tt)))
            dispose(tt)
            changed += 1
        end
        return changed
    end
end
