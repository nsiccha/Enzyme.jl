# Julia 1.12 and later allocate a `Memory` whose length is not a constant, or that does not
# fit a pool cell, with `jl_alloc_genericmemory_unchecked`. It returns the object with its
# data pointer set and leaves the length to the calling code, which stores it right away
# and, where elements hold references, then zeroes the data: no safepoint in between.
#
# Forward mode allocates the shadow of an allocation right after it, and each further
# shadow of a batch after the one before. That allocation is a safepoint, at which the
# earlier object still has the length its cell held before. A collection that starts there
# scans that many elements where they hold references, and follows whatever is in them.
# For other elements it counts that many bytes as scanned.
#
# So wherever a call that can collect comes before the store of an allocation's length, the
# length is stored as zero right after the allocation, which is how an empty `Memory`
# reads, until the calling code stores the real one. The store is volatile: dead store
# elimination would remove a plain one that a later store overwrites, since it does not
# know that a collection in between reads the field. Reverse mode allocates the shadow
# first, and the shadow's allocation rule (`array_shadow_handler`) stores its length at
# once, so nothing changes there.

const UNCHECKED_MEMORY_ALLOCATORS =
    ("jl_alloc_genericmemory_unchecked", "ijl_alloc_genericmemory_unchecked")

# Whether the length of the `Memory` that `alloc` allocates is stored before any later call
# that can collect. A `Memory` holds its length at its own address.
function memory_length_stored_first(alloc::LLVM.Instruction)::Bool
    ref = LLVM.API.LLVMGetNextInstruction(alloc)
    while ref != C_NULL
        inst = LLVM.Instruction(ref)
        if inst isa LLVM.StoreInst
            base, offset = constant_field_pointer(inst.operands[2])
            base == alloc && offset == 0 && return true
        elseif inst isa LLVM.CallBase
            callee = inst.called_operand
            callee isa LLVM.Function || return false
            fname = callee.name
            startswith(fname, "llvm.") || fname == "julia.pointer_from_objref" || return false
        end
        ref = LLVM.API.LLVMGetNextInstruction(inst)
    end
    return false
end

function store_empty_memory_lengths!(mod::LLVM.Module)::Bool
    changed = false
    for fname in UNCHECKED_MEMORY_ALLOCATORS
        haskey(mod.functions, fname) || continue
        allocator = mod.functions[fname]
        for use in collect(allocator.uses)
            alloc = use.user
            alloc isa LLVM.CallInst && alloc.called_operand == allocator || continue
            memory_length_stored_first(alloc) && continue
            B = LLVM.IRBuilder()
            position!(B, LLVM.after(alloc))
            ST = get_memory_struct()
            lenptr = inbounds_gep!(
                B, ST, struct_ptr!(B, alloc, ST),
                LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(0))],
            )
            st = store!(B, LLVM.ConstantInt(LLVM.IntType(8 * sizeof(Csize_t)), 0), lenptr)
            st.volatile = true
            dispose(B)
            changed = true
        end
    end
    return changed
end
