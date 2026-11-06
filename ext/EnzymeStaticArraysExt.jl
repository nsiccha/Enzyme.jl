module EnzymeStaticArraysExt

using StaticArrays
using Enzyme
using PrecompileTools: @setup_workload, @compile_workload

@inline function Base.convert(::Type{SArray}, tpa::Enzyme.TupleArray{T, L, N}) where {T, L, N}
    return SArray{Tuple{size(tpa)...}, T, N, L}(tpa.data)
end
@inline Base.convert(::Type{StaticArray}, tpa::Enzyme.TupleArray) = convert(SArray, tpa)

@inline function Enzyme.tupstack(rows::Tuple{Vararg{T}}, outshape::Tuple{Vararg{Int}}, inshape::Tuple{Vararg{Int}}) where {T<:StaticArrays.SArray}
    reshape(reduce(hcat, map(vec, rows)), Size(outshape..., inshape...))
end

# The input's type has the static shape, so the output can be built type-stably from it.
@inline function Enzyme.specialize_output(
        output::Enzyme.TupleArray{T, L, N}, ::StaticArray{S, <:Any, N}
    ) where {T, L, N, S}
    return SArray{S, T, N, L}(output.data)
end

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}) where {S, T, N, L}
    ntuple(Val(L)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(Enzyme.onehot(NTuple{L, T})[i])
    end
end

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}, start::Int, endl::Int) where {S, T, N, L}
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(
        ntuple(Val(L)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end)
    end
end

@inline function Enzyme.EnzymeCore.make_zero(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:SArray{S,T}}
    return Base.zero(prev)::FT
end
@inline function Enzyme.EnzymeCore.make_zero(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    return Base.zero(prev)::FT
end

@inline function Enzyme.EnzymeCore.make_zero(
    ::Type{FT}, seen::IdDict, prev::FT, ::Val{copy_if_inactive} = Val(false)
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:SArray{S,T},copy_if_inactive}
    return Base.zero(prev)::FT
end
@inline function Enzyme.EnzymeCore.make_zero(
    ::Type{FT}, seen::IdDict, prev::FT, ::Val{copy_if_inactive} = Val(false)
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T},copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]
    end
    new = Base.zero(prev)::FT
    seen[prev] = new
    return new
end

@inline function Enzyme.EnzymeCore.make_zero!(
    prev::FT, seen
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    fill!(prev, zero(T))
    return nothing
end
@inline function Enzyme.EnzymeCore.make_zero!(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    Enzyme.EnzymeCore.make_zero!(prev, nothing)
    return nothing
end

# StaticArrays adds `Base.cconvert(::Type{Ptr{T}}, ...)` methods for its arrays. They match
# every `cconvert` call whose argument Julia inferred as `Any`, and the compiler makes such
# calls into the LLVM and Enzyme C APIs: most through LLVM.jl 9's `LLVM.context()`, which
# reads the active context from task local storage untyped, the rest with values Enzyme passes
# untyped. Each one reaches the functions every derivative is compiled through, so loading
# StaticArrays, before or after Enzyme, discards the native code Enzyme's image holds for its
# compiler, and the first derivative of the session compiles it all again (a trivial one took
# 20 to 23 s instead of 2.8 s on Julia 1.10, 19 to 37 s instead of 5.4 to 6 s on 1.12). This
# image is written with both packages loaded, so Enzyme's own workload (`src/precompile.jl`),
# run again here, caches that compiler as it stands with StaticArrays. It does not give back
# everything Julia drops: about 6 s remain on Julia 1.10 when StaticArrays loads first, and
# about 9 s on 1.12 when it loads second.
@setup_workload begin
    precompile_module = @eval module $(gensym())
        f(x) = x^2
    end

    # As in `src/precompile.jl`: the workload is for the compiler, not for its derivatives.
    Enzyme.Compiler.RetainDerivatives[] = false
    @compile_workload begin
        Enzyme.autodiff(Reverse, precompile_module.f, Active(2.0))
    end
    Enzyme.Compiler.RetainDerivatives[] = true

    # As in `src/precompile.jl`: what the workload cached belongs to this process.
    Enzyme.Compiler.clear_caches!()
end

end
