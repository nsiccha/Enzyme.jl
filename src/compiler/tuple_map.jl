# Base's `map` over tuples of 32 or more elements (`map(f, t::Any32)` and its two- and
# many-tuple methods, `base/tuple.jl`) fills a `Vector{Any}` in a loop and splats it
# back into a tuple. The loop reads the tuples at a run-time index, so one load carries
# every element type, and Julia's union split then stores that one value into boxes
# laid out for each element type. Enzyme's type analysis assigns a value one layout, so
# differentiating such a `map` over a tuple of two element types raises
# `IllegalTypeAnalysisException`; its result type is unknown either way. Map such tuples
# field by field instead, in the same order and over the shortest tuple, as Base does.
# This keeps the call statically typed under differentiation, both its elements and its
# result. DifferentiationInterface's `Rewrap`, `make_context_shadows` and
# `translate_prepared!` reach these methods with more than 31 contexts.
# One element of the mapped tuples, so that `ntuple` unrolls the map without a closure.
struct MapElement{F, T}
    f::F
    ts::T
end
@inline map_element_args(::Tuple{}, i) = ()
@inline map_element_args(ts::Tuple, i) = (getfield(first(ts), i), map_element_args(Base.tail(ts), i)...)
@inline (m::MapElement)(i) = m.f(map_element_args(m.ts, i)...)
@inline map_shortest(ts::Tuple) = minimum(map(length, ts))

# `ntuple` with a `Val` length unrolls into one call per element, each reading its tuple
# fields at a literal index, so every element and the result keep their types.
@inline function override_map(f, ts::Vararg{Tuple, M}) where {M}
    return ntuple(MapElement(f, ts), Val(map_shortest(ts)))
end

const MAP_LONG_TUPLE_METHODS = (
    which(Base.map, Tuple{Any, Base.Any32}),
    which(Base.map, Tuple{Any, Base.Any32, Base.Any32}),
    which(Base.map, Tuple{Any, Base.Any32, Base.Any32, Base.Any32}),
)

map_rewrite(interp::EnzymeInterpreter, f, argtypes, sv::AbsIntState) = nothing

function map_rewrite(interp::EnzymeInterpreter, f::typeof(Base.map), argtypes, sv::AbsIntState)
    length(argtypes) >= 3 || return nothing
    sig = Core.Compiler.argtypes_to_type(argtypes)
    Base.isdispatchtuple(sig) || return nothing
    for T in sig.parameters[3:end]
        T <: Tuple && isconcretetype(T) || return nothing
    end
    method = MAP_LONG_TUPLE_METHODS[min(length(argtypes) - 2, 3)]
    return checked_method_rewrite(interp, f, sig, sv, (method, override_map))
end
