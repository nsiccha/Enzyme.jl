# An ordinary Julia getter can throw, rebind the global, or have its own
# derivative. Evaluate it where it appeared. The interpreter recognizes
# Base's getproperty of an existing array field of an immutable NamedTuple:
# the captured record will still hold precisely that array at the use site.
struct PrepareProperty{Name} end
@inline function (::PrepareProperty{Name})(owner) where {Name}
    if Interpreter.plain_array_property(owner, Val(Name)) && isdefined(owner, Name)
        return nothing
    end
    # Preserve the original dynamic getter and its derivative rule on this path.
    return Some(Base.inferencebarrier(getproperty)(owner, Name))
end

# In the conservative case the separately differentiated identity gives a
# Const array its own writable zero shadow, just as the original dynamic
# getter did. The pass below can erase that boundary in a read-only consumer.
projected_array(x::Array) = x
@noinline property_array_shadow(x::Array) = Base.inferencebarrier(projected_array)(x)::typeof(x)

@inline property_operand(::Val{Name}, owner, ::Nothing) where {Name} =
    property_array_shadow(getfield(owner, Name))
@inline property_operand(key, owner, payload::Some) = getfield(payload, :value)

struct PropertyCall{Name, Position} end

# Reconstruct exactly the user's call. Tuple insertion or indexing methods
# can themselves have user dispatch and differentiation rules.
@generated function (::PropertyCall{Name, Position})(f::F, payload, args::Vararg{Any, N}) where {Name, Position, F, N}
    operands = Vector{Expr}(undef, N)
    for i in 1:N
        operands[i] = :(getfield(args, $i))
    end
    owner = operands[Position]
    operands[Position] = :(property_operand(Val($(QuoteNode(Name))), $owner, payload))
    target = payload <: Some ? :(Base.inferencebarrier(f)) : :f
    return :($target($(operands...)))
end

function property_dispatch(call::LLVM.CallInst)
    dynamic_call_operands(call) == (2, false) || return false
    isempty(call.operand_bundles) || return false
    # Call-site metadata can name argument positions. The new calls have
    # different arguments, and only the debug location applies to both.
    for (kind, _) in call.metadata
        kind == LLVM.MD_dbg || return false
    end
    return true
end

# A load of the value of a Julia global binding. Keep the existing dynamic
# call ABI for receivers from active arguments or other storage.
function global_binding_load(@nospecialize(value::LLVM.Value))
    if value isa LLVM.CallInst
        called = value.called_operand
        return called isa LLVM.Function && called.name == "jl_get_binding_value_seqcst"
    end
    value isa LLVM.LoadInst && haskey(value.metadata, "tbaa") || return false
    operands = value.metadata["tbaa"].operands
    return length(operands) >= 2 && md_node_name(operands[2]) == "jtbaa_binding"
end

# Both calls stay at their original sites. In particular the receiver passed
# to the consumer is the SSA value captured by the getter, even when a later
# call has rebound its global. Across blocks, require a single predecessor:
# a loop consuming the same getter result several times must share its shadow.
function combine_property_calls!(entry::LLVM.Function, enzyme_ctx::EnzymeContext)
    haskey(entry.parent.functions, "julia.call") || return nothing
    getters = LLVM.CallInst[]
    for bb in entry.blocks, inst in bb.instructions
        inst isa LLVM.CallInst || continue
        ops = inst.arguments
        length(ops) == 4 && global_binding_load(ops[3]) && property_dispatch(inst) || continue
        known, fn = absint(ops[2], enzyme_ctx)
        known && fn === getproperty && push!(getters, inst)
    end
    for getter in Iterators.reverse(getters)
        ops = getter.arguments
        known, field = absint(ops[4], enzyme_ctx)
        known && field isa Symbol || continue
        use = LLVM.API.LLVMGetFirstUse(getter)
        (use != C_NULL && LLVM.API.LLVMGetNextUse(use) == C_NULL) || continue
        user = first(getter.users)
        user isa LLVM.CallInst && property_dispatch(user) || continue
        if user.parent != getter.parent
            preds = LLVM.predecessors(user.parent)
            length(preds) == 1 && only(preds) == getter.parent || continue
        end
        values = collect(user.arguments)
        pos = findfirst(isequal(getter), values)
        (pos !== nothing && pos >= 3) || continue
        owner = ops[3]
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            position!(builder, LLVM.before(getter))
            prepare = unsafe_to_llvm(builder, PrepareProperty{field}(); insert_name_if_not_exists = "prepare_property", enzyme_ctx)
            payload = emit_apply_generic!(builder, LLVM.Value[prepare, owner])
            payload.debug_location = getter.debug_location
            position!(builder, LLVM.before(user))
            # [jl_apply_generic, f, args...] becomes [helper, f, payload, args...].
            # The owner replaces the selected field in args.
            helper = PropertyCall{field, pos - 2}()
            values[1] = unsafe_to_llvm(builder, helper; insert_name_if_not_exists = "property_call", enzyme_ctx)
            values[pos] = owner
            insert!(values, 3, payload)
            combined = emit_apply_generic!(builder, values)
            combined.debug_location = user.debug_location
            LLVM.replace_uses!(user, combined)
        end
        LLVM.erase!(user)
        LLVM.erase!(getter)
    end
    return nothing
end

function elide_readonly_property_shadows!(mod::LLVM.Module, enzyme_ctx::EnzymeContext)
    for fn in mod.functions
        occursin("property_array_shadow", fn.name) || continue
        mi, _ = enzyme_custom_extract_mi(fn, false)
        mi === nothing && continue
        sig = mi.specTypes
        sig isa DataType && sig.parameters[1] === typeof(property_array_shadow) || continue
        for use in collect(fn.uses)
            call = use.user
            call isa LLVM.CallInst && call.called_operand == fn || continue
            args = call.arguments
            length(args) == 1 && args[1].value_type == call.value_type || continue
            root = call.parent.parent
            state = ReadonlyWalkState(mod)
            wrote, returned = readonly_walk!(
                LLVM.Value[call], root, enzyme_ctx.world, enzyme_ctx, state, false; defer_calls = false,
            )
            (wrote || returned || state.thrown[]) && continue
            LLVM.replace_uses!(call, args[1])
            LLVM.erase!(call)
        end
    end
    return nothing
end
