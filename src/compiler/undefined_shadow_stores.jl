# libEnzyme creates the shadow of an allocation only in the reverse pass when nothing in
# the forward pass reads a reference back from it (`backwardsOnlyShadows` without
# `primalInitialize`). In the augmented forward pass it then replaces the shadow by its
# undefined value, which for a garbage-collected pointer is `jl_nothing`
# (`julia_undef_value_for_type`), and leaves out its own stores into the shadow.
#
# A handler that fills shadow memory itself cannot do the same when it emits its store:
# libEnzyme visits the instructions of a block last to first, so the handler of a call
# runs while the shadow of an allocation above it is still a placeholder. The store it
# emits becomes a store through `jl_nothing` once the placeholder is replaced, and writes
# past the `nothing` singleton in the system image.
#
# The rule handler does that for the shadow a custom rule returns through an `sret`
# buffer. From Julia 1.12 the buffer of a record of references and numbers, such as
# `Tuple{Matrix{Float64}, Float64}`, is such an allocation: the callee writes the
# references into a separate array of return roots, so only numbers are loaded from the
# buffer. Every such derivative stored a number eight bytes past `nothing`, where Julia
# 1.13 keeps the cached box of `Int32(-512)` and Julia 1.12 the empty `Symbol`.
#
# So once libEnzyme has returned, every store whose address is derived from `jl_nothing`
# is removed. Such memory does not exist in that pass, and `nothing` has no fields, so no
# other code stores through it. The reverse pass allocates the shadow and zeroes it.

# Whether `user` computes an address from its first operand by a cast or an offset.
function is_address_computation(@nospecialize(user::LLVM.Value))::Bool
    if user isa LLVM.ConstantExpr
        op = user.opcode
        return op == LLVM.Opcode.BitCast || op == LLVM.Opcode.AddrSpaceCast || op == LLVM.Opcode.GetElementPtr
    end
    return user isa LLVM.BitCastInst || user isa LLVM.AddrSpaceCastInst || user isa LLVM.GetElementPtrInst
end

function remove_undefined_shadow_stores!(mod::LLVM.Module)::Bool
    haskey(mod.globals, "ejl_jl_nothing") || return false
    # `nothing` and every address computed from it.
    addresses = LLVM.Value[mod.globals["ejl_jl_nothing"]]
    stores = LLVM.StoreInst[]
    while !isempty(addresses)
        address = pop!(addresses)
        for use in address.uses
            user = use.user
            if user isa LLVM.StoreInst
                # Storing `nothing` itself somewhere is an ordinary store.
                user.operands[2] == address && !(user in stores) && push!(stores, user)
            elseif is_address_computation(user) && user.operands[1] == address
                push!(addresses, user)
            end
        end
    end
    for store in stores
        LLVM.erase!(store)
    end
    return !isempty(stores)
end
