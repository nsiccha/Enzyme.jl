# From Julia 1.11, shrinking a vector changes its size and sizehint! can replace
# its MemoryRef. Earlier reverse instructions recompute element addresses through
# this header, so restore each shadow header before reversing those instructions.
# Keep the old storage alive too: adjoints can arrive through aliases to it.
@static if VERSION >= v"1.11-"
    array_storage_shadows(x::Union{Duplicated, DuplicatedNoNeed}) = (x.dval,)
    array_storage_shadows(x::Union{BatchDuplicated, BatchDuplicatedNoNeed}) = x.dval

    function restore_array_storage_shadow(tape)
        old, shadow, after = tape
        if old.ref.mem !== after.mem
            # sizehint! copied the retained primal elements to new storage. Move
            # their adjoints back additively, preserving contributions through
            # aliases to the old Memory, then consume the new storage's adjoints.
            # Read them where the call put them: later growth may have moved the
            # shadow's own header since (pushfirst! into storage reserved at the
            # front, or a reallocation that moved its adjoints back here).
            retained = Base.wrap(Array, after, size(old))
            for i in eachindex(old)
                @inbounds old[i] += retained[i]
                @inbounds retained[i] = zero(eltype(retained))
            end
        end
        # Deletion of isbits elements only changes the logical length. Its old
        # storage may already contain adjoints from other aliases; keep them.
        setfield!(shadow, :ref, old.ref)
        setfield!(shadow, :size, size(old))
        return nothing
    end
    restore_array_storage_shadow(::Nothing) = nothing

    struct ArrayStorageShadow{F, P, N, K, A}
        func::F
        primal::P
        n::N
        kwargs::K
        runtime_activity::Bool
    end

    function (r::ArrayStorageShadow{F, P, N, K, A})(shadow) where {F, P, N, K, A}
        r.runtime_activity && shadow === r.primal && return nothing
        old = A ? shadow_at_call(shadow) : nothing
        r.func(shadow, r.n; r.kwargs...)
        if A
            if old.ref.mem !== shadow.ref.mem
                # The shadow's copied values belong to the old storage. Keep
                # them there and collect only new contributions in this storage.
                # Zero all of the new Memory, not only the shadow's length: it
                # was allocated uninitialized, and later growth within its
                # capacity (push!, append!, resize!) makes those slots elements
                # whose adjoints the reverse pass accumulates into.
                fill!(shadow.ref.mem, zero(eltype(shadow)))
            end
            return (old, shadow, shadow.ref)
        end
        return nothing
    end

    function array_storage_lane(f, x, n, kwargs, rta, ::Val{A}) where {A}
        return ArrayStorageShadow{typeof(f), typeof(x), typeof(n), typeof(kwargs), A}(f, x, n, kwargs, rta)
    end

    function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfig,
            func::Const{F},
            ::Type{RT},
            x::Union{Duplicated{Vector{T}}, BatchDuplicated{Vector{T}}},
            n::Annotation{<:Integer}; kwargs...,
        ) where {F <: Union{typeof(Base._deleteend!), typeof(sizehint!)}, T <: Base.IEEEFloat, RT}
        value = func.val(x.val, n.val; kwargs...)
        lane = array_storage_lane(func.val, x.val, n.val, values(kwargs), EnzymeRules.runtime_activity(config), Val(true))
        tape = map(lane, array_storage_shadows(x))
        primal = EnzymeRules.needs_primal(config) ? value : nothing
        shadow = EnzymeRules.needs_shadow(config) ? x.dval : nothing
        return EnzymeRules.AugmentedReturn(primal, shadow, tape)
    end

    function EnzymeRules.reverse(
            ::EnzymeRules.RevConfig,
            ::Const{F},
            dret,
            tape,
            x::Union{Duplicated{Vector{T}}, BatchDuplicated{Vector{T}}},
            n::Annotation{<:Integer}; kwargs...,
        ) where {F <: Union{typeof(Base._deleteend!), typeof(sizehint!)}, T <: Base.IEEEFloat}
        foreach(restore_array_storage_shadow, tape)
        return (nothing, nothing)
    end

    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig,
            func::Const{F},
            ::Type{RT},
            x::Union{Duplicated{Vector{T}}, BatchDuplicated{Vector{T}}},
            n::Annotation{<:Integer}; kwargs...,
        ) where {F <: Union{typeof(Base._deleteend!), typeof(sizehint!)}, T <: Base.IEEEFloat, RT}
        value = func.val(x.val, n.val; kwargs...)
        lane = array_storage_lane(func.val, x.val, n.val, values(kwargs), EnzymeRules.runtime_activity(config), Val(false))
        foreach(lane, array_storage_shadows(x))
        if EnzymeRules.needs_shadow(config)
            return EnzymeRules.needs_primal(config) ? EnzymeRules.forward_rule_return_type(config, RT)(value, x.dval) : x.dval
        end
        return EnzymeRules.needs_primal(config) ? value : nothing
    end
end
