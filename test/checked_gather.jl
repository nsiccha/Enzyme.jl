using Enzyme, Test

@inline function checked_entry(xs, j, g)
    j == 1 && return xs[1][g]
    j == 2 && return xs[2][g]
    j == 3 && return xs[3][g]
    return xs[4][g]
end

function checked_gather(xs, groups, picks)
    s = 0.0
    for g in groups, j in picks
        v = checked_entry(xs, j, g)
        s += v * v
    end
    return s
end

@testset "Split reverse of four bounds-checked views" begin
    x = collect(1.0:16.0)
    dx = zero(x)
    ranges = (1:4, 5:8, 9:12, 13:16)
    xs = map(Base.Fix1(view, x), ranges)
    dxs = map(Base.Fix1(view, dx), ranges)
    groups = [2, 3, 2, 1]
    picks = [3, 1, 4, 2, 3]
    args = (Duplicated(xs, dxs), Const(groups), Const(picks))
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(checked_gather)}, Active, map(typeof, args)...)
    tape, value, _ = fwd(Const(checked_gather), args...)
    rev(Const(checked_gather), args..., 1.0, tape)

    expected = zero(x)
    for j in picks, g in groups
        k = ranges[j][g]
        expected[k] += 2 * x[k]
    end
    @test value == checked_gather(xs, groups, picks)
    @test dx == expected
end
