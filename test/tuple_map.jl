using Enzyme, Test

# Base's `map` over tuples of 32 or more elements (`map(f, t::Any32)`) fills a
# `Vector{Any}` at a run-time index and splats it into a tuple. Under differentiation
# that loop reads one value of every element type and stores it into boxes laid out for
# each type, which type analysis cannot reconcile, and the result type is unknown.
# Enzyme's interpreter maps such tuples field by field instead (`override_map`).

struct TupleMapConstant{T}
    data::T
end
struct TupleMapCache{T}
    data::T
end
tuple_map_constant(c) = TupleMapConstant(c)
tuple_map_cache(c) = TupleMapCache(c)
tuple_map_wrap(c::AbstractArray) = TupleMapConstant(c)
tuple_map_wrap(c::Base.RefValue) = TupleMapCache(c)

# DifferentiationInterface's `Rewrap`: a tuple of maker functions applied to a tuple of
# unwrapped contexts with the two-tuple `map`.
struct TupleMapRewrap{C, T}
    makers::T
    TupleMapRewrap(makers...) = new{length(makers), typeof(makers)}(makers)
end
(r::TupleMapRewrap{C})(cs::Vararg{Any, C}) where {C} = map((m, c) -> m(c), r.makers, cs)

@inline tuple_map_read(c::TupleMapCache) = c.data[]
@inline tuple_map_read(c::TupleMapConstant) = sum(c.data)
@inline tuple_map_sum(::Tuple{}) = 0.0
@inline tuple_map_sum(cs::Tuple) = tuple_map_read(first(cs)) + tuple_map_sum(Base.tail(cs))
@noinline tuple_map_total(x, cs::Tuple) = tuple_map_sum(cs) * x[1]

# `(sum(v₁) + ⋯ + sum(vₙ) + x₂) * x₁`, with the box stored into and read back through
# the mapped tuple: d/dx₁ = Σ + x₂, d/dx₂ = x₁. The box comes right after `x`, and the
# mapped 41-tuple is built by a generated constructor, so its type is concrete at the
# `map` call sites, as DifferentiationInterface's tuples are, and no splat of more than
# 32 elements is involved (a `Core._apply_iterate` call would hide the type, and on
# Julia 1.12 one with an active argument among 32 or more stops Enzyme: see `broken`).
@generated function tuple_map_with_box(box, vs::NTuple{N, Any}) where {N}
    return Expr(:tuple, (:(vs[$i]) for i in 1:N)..., :box)
end
function tuple_map_rewrap_loss(x, box, rewrap, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, rewrap(tuple_map_with_box(box, vs)...))
end
function tuple_map_single_loss(x, box, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, map(tuple_map_wrap, tuple_map_with_box(box, vs)))
end
function tuple_map_triple_loss(x, box, makers, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    cs = tuple_map_with_box(box, vs)
    return tuple_map_total(x, map((m, c, ::Int) -> m(c), makers, cs, ntuple(identity, Val(N + 1))))
end
# The mapped tuple built by a splat of more than 32 elements is untyped at the call site,
# so the `map` is dispatched at run time and Enzyme compiles the dispatched method with
# the run-time types.
function tuple_map_untyped_single_loss(x, box, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, map(tuple_map_wrap, (vs..., box)))
end
function tuple_map_untyped_double_loss(x, box, makers, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, map((m, c) -> m(c), makers, (vs..., box)))
end
function tuple_map_untyped_triple_loss(x, box, makers, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, map((m, c, ::Int) -> m(c), makers, (vs..., box), ntuple(identity, Val(N + 1))))
end

# `broken`: on Julia 1.12 a call with an active argument among 32 or more arguments, at
# Enzyme's own entry or dispatched at run time (`Core._apply_iterate`), stops Enzyme
# before and after the typed map, with `AssertionError: conv == 37` in Forward and
# `Unexpected calling conv, got 0` in Reverse; a separate defect (Enzyme todo filed with
# snag nested-di-hvp-pa-5ec056c4). Julia 1.10 differentiates every shape below.
const TUPLE_MAP_ACTIVE_PAST_32_BROKEN = VERSION >= v"1.12"
function tuple_map_check(loss, K, extra...; broken::Bool = false)
    vs = ntuple(j -> fill(Float64(j), 3), K)
    consts = map(Const, vs)
    x = [0.3, 0.7]
    S = sum(3j for j in 1:K)
    expected = [S + x[2], x[1]]
    if broken
        ran = try
            autodiff(Forward, loss, Duplicated(x, [1.0, 0.0]), Duplicated(Ref(0.0), Ref(0.0)), extra..., consts...)
            autodiff(ReverseWithPrimal, loss, Active, Duplicated(x, zeros(2)), Duplicated(Ref(0.0), Ref(0.0)), extra..., consts...)
            true
        catch e
            msg = sprint(showerror, e)
            occursin("calling conv", msg) || occursin("conv == 37", msg) || rethrow()
            false
        end
        @test_broken ran
        return nothing
    end
    box = Ref(0.0)
    dx = [1.0, 0.0]
    tangent = autodiff(Forward, loss, Duplicated(x, dx), Duplicated(box, Ref(0.0)), extra..., consts...)[1]
    @test tangent ≈ expected[1]
    @test box[] == x[2]
    dx = zeros(2)
    box = Ref(0.0)
    primal = autodiff(ReverseWithPrimal, loss, Active, Duplicated(x, dx), Duplicated(box, Ref(0.0)), extra..., consts...)[2]
    @test dx ≈ expected
    @test primal ≈ (S + x[2]) * x[1]
    @test x == [0.3, 0.7]
    @test box[] == x[2]
    return nothing
end

@testset "override_map matches Base.map" begin
    f1(a) = a + 1
    f2(a, b) = a * b
    f3(a, b, c) = (a, b, c)
    for n in (0, 1, 2, 31, 32, 33, 41)
        t = ntuple(identity, n)
        s = ntuple(i -> 2i, n)
        @test Enzyme.Compiler.Interpreter.override_map(f1, t) === map(f1, t)
        @test Enzyme.Compiler.Interpreter.override_map(f2, t, s) === map(f2, t, s)
        @test Enzyme.Compiler.Interpreter.override_map(f3, t, s, t) === map(f3, t, s, t)
    end
    # Over the shortest tuple, as Base does.
    @test Enzyme.Compiler.Interpreter.override_map(f2, ntuple(identity, 40), ntuple(identity, 33)) === map(f2, ntuple(identity, 40), ntuple(identity, 33))
    @test Enzyme.Compiler.Interpreter.override_map(f3, ntuple(identity, 40), ntuple(identity, 35), ntuple(identity, 33)) === map(f3, ntuple(identity, 40), ntuple(identity, 35), ntuple(identity, 33))
    # Elements are visited in order, each exactly once.
    seen = Int[]
    Enzyme.Compiler.Interpreter.override_map(i -> push!(seen, i), ntuple(identity, 41))
    @test seen == 1:41
    # The result is inferred for a tuple of two element types, which Base's method is not.
    mixed = (ntuple(j -> fill(Float64(j), 3), 40)..., Ref(0.5))
    @test Base.return_types(map, Tuple{typeof(tuple_map_wrap), typeof(mixed)}) == Any[Tuple]
    @test @inferred(Enzyme.Compiler.Interpreter.override_map(tuple_map_wrap, mixed)) == map(tuple_map_wrap, mixed)
    @test Enzyme.Compiler.Interpreter.override_map(f1, (1, 2.0)) === (2, 3.0)
end

@testset "Differentiating map over long mixed tuples" begin
    for K in (8, 40)
        makers = (ntuple(Returns(tuple_map_constant), K)..., tuple_map_cache)
        broken = K >= 32 && TUPLE_MAP_ACTIVE_PAST_32_BROKEN
        # `rewrap(cs...)` splats the 41 contexts into the callable, as DI's Rewrap call does.
        tuple_map_check(tuple_map_rewrap_loss, K, Const(TupleMapRewrap(makers...)); broken)
        tuple_map_check(tuple_map_single_loss, K; broken)
        tuple_map_check(tuple_map_triple_loss, K, Const(makers); broken)
        # Base's method reached as a run-time dispatch target, in the three arities.
        tuple_map_check(tuple_map_untyped_single_loss, K; broken)
        tuple_map_check(tuple_map_untyped_double_loss, K, Const(makers); broken)
        tuple_map_check(tuple_map_untyped_triple_loss, K, Const(makers); broken)
    end
end

# A more specific `map` method stays authoritative: the rewrite applies only when Base's
# long-tuple method is the one dispatch selects.
struct TupleMapUserMapper end
(::TupleMapUserMapper)(c) = tuple_map_wrap(c)
Base.map(::TupleMapUserMapper, cs::Base.Any32) = map(c -> TupleMapConstant(fill(0.0, 1)), cs)

function tuple_map_user_loss(x, box, vs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return tuple_map_total(x, map(TupleMapUserMapper(), tuple_map_with_box(box, vs)))
end

@testset "A user map method is not replaced" begin
    K = 32
    vs = ntuple(j -> fill(Float64(j), 3), K)
    x = [0.3, 0.7]
    if TUPLE_MAP_ACTIVE_PAST_32_BROKEN
        ran = try
            autodiff(Reverse, tuple_map_user_loss, Active, Duplicated(x, zeros(2)), Duplicated(Ref(0.0), Ref(0.0)), map(Const, vs)...)
            true
        catch e
            occursin("calling conv", sprint(showerror, e)) || rethrow()
            false
        end
        @test_broken ran
    else
        dx = zeros(2)
        autodiff(Reverse, tuple_map_user_loss, Active, Duplicated(x, dx), Duplicated(Ref(0.0), Ref(0.0)), map(Const, vs)...)
        @test dx == [0.0, 0.0]
        @test autodiff(Forward, tuple_map_user_loss, Duplicated(x, [1.0, 0.0]), Duplicated(Ref(0.0), Ref(0.0)), map(Const, vs)...)[1] == 0.0
    end
end
