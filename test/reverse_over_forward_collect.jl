using Enzyme, Test

# A typed nested comprehension over ranges, `T[f(i, j) for j in R for i in S]`,
# collects a `Flatten` of generators. Its iterator state is a tuple of the outer
# counter and two objects, and Julia keeps the next outer counter as `i + 1`
# inserted into that tuple. The Reverse derivative of the Forward derivative of
# such a `collect` (Reverse over Forward, as DifferentiationInterface's
# `SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse), AutoEnzyme(; mode = Enzyme.Forward))`
# runs it) differentiates the Forward derivative as a split callee. Its gradient
# pass dropped that counter although it builds the shadow of the state tuple
# from it, and failed with "Illegal replace ficticious phi for … add i64
# %value_phi21, 1". The use analysis had memoized the loop-exit phi of the
# tuple's shadow as never needed, provisionally, while deciding the loop-header
# phi that is its only user, and kept that answer once the header phi turned
# out needed.

@noinline function rof_diffs(ts, d)
    return Float64[ts[i] - d[j] for j in eachindex(d) for i in eachindex(ts)]
end

@noinline function rof_udts(ts, d)
    diffs = Float64[ts[i] - d[j] for j in eachindex(d) for i in eachindex(ts)]
    return sort!(filter!(x -> x >= 0.0, diffs))
end

rof_sumsq(ts, d) = sum(abs2, rof_diffs(ts, d))
rof_sumexp(ts, d) = sum(exp, rof_udts(ts, d))

# Reverse over Forward: the gradient of the directional derivative.
function rof_hvp(f, ts, v, d)
    dir(ts, v, d) = autodiff(Forward, Const(f), Duplicated(ts, v), Const(d))[1]
    hv = zero(ts)
    autodiff(Reverse, Const(dir), Active, Duplicated(ts, hv), Const(v), Const(d))
    return hv
end

# Reverse over Forward with both arrays active, returning the product with
# the first array's block.
function rof_hvp_both(f, ts, v, d, dd)
    dir(ts, v, d, dd) = autodiff(Forward, Const(f), Duplicated(ts, v), Duplicated(d, dd))[1]
    hv = zero(ts)
    hd = zero(d)
    autodiff(Reverse, Const(dir), Active, Duplicated(ts, hv), Const(v), Duplicated(d, hd), Const(dd))
    return hv, hd
end

# Forward over Reverse, for comparison.
function for_hvp(f, ts, v, d)
    grad(ts, d) = (g = zero(ts); autodiff(Reverse, Const(f), Active, Duplicated(ts, g), Const(d)); g)
    return autodiff(Forward, Const(grad), Duplicated(ts, v), Const(d))[1]
end

@testset "Reverse over Forward of a typed nested comprehension" begin
    ts = [0.5, 1.5, 2.0, 3.5, 0.25]
    d = [0.0, 1.0, 2.5]
    v = [cos(i) for i in eachindex(ts)]
    dd = [sin(j) for j in eachindex(d)]

    # sum of squares of every difference: H = 2 length(d) I on ts, -2 ones on
    # the ts-by-d block
    @test rof_hvp(rof_sumsq, ts, v, d) ≈ 2 * length(d) .* v
    hv, hd = rof_hvp_both(rof_sumsq, ts, v, d, dd)
    @test hv ≈ 2 * length(d) .* v .- 2 * sum(dd)
    @test hd ≈ -2 * sum(v) .+ 2 * length(ts) .* dd
    @test for_hvp(rof_sumsq, ts, v, d) ≈ 2 * length(d) .* v

    # sum of exp over the sorted nonnegative differences: a diagonal Hessian
    # on ts, Σ_j exp(ts_i - d_j) over the j with ts_i >= d_j. No value through
    # `sort!(filter!(…))` is compared: on Julia 1.10 the gradient and both
    # products read memory whose contents differ from process to process, so
    # they are right in most runs and garbage in some, on the tip and with this
    # repair alike (Enzyme todo 0uohlgh); on Julia 1.11 and later the gradient
    # through a `filter!` that drops an element is wrong (todo 063gmv7), and a
    # nested derivative built on it can crash, so none is taken there.
    g = zero(ts)
    autodiff(Reverse, Const(rof_sumexp), Active, Duplicated(ts, g), Const(d))
    @test length(g) == length(ts)
    if VERSION < v"1.11"
        hv = rof_hvp(rof_sumexp, ts, v, d)
        @test length(hv) == length(ts)
        @test length(for_hvp(rof_sumexp, ts, v, d)) == length(ts)
    end
end
