using Enzyme, Test

# Each loop iteration takes one of three paths: `x <= 0` skips `sqrt`, the
# other two are split by a condition on its result. The reverse pass needs the
# merged value `v` and recomputes it from `x`, which it may reload. It must not
# call `sqrt` on the path that skipped it, where `x` is negative and `sqrt`
# throws a `DomainError` the primal never raised.
function skipped_sqrt_times(ys, xs)
    s = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        v = if x <= 0
            1.0
        else
            r = sqrt(x)
            if r >= 2.0
                exp(r)
            else
                r
            end
        end
        s += v * ys[i]
    end
    return s
end

# The same, with the merged value computed from the active input. (A call in one
# arm keeps the two arms apart: LLVM would merge two arithmetic arms into a
# `select`, leaving two paths.)
function skipped_sqrt_squared(ys, xs)
    s = 0.0
    @inbounds for i in eachindex(xs)
        x = xs[i]
        y = ys[i]
        v = if x <= 0
            y
        else
            r = sqrt(x)
            if r >= 2.0
                y * exp(r)
            else
                y * r
            end
        end
        s += v * v
    end
    return s
end

# The shape of a closed-form propagator: the identity for `dt <= 0`, otherwise
# a `sqrt` whose result decides, through `min` and `max`, between two formulas.
function skipped_sqrt_propagator(ys, dts)
    s = 0.0
    @inbounds for i in eachindex(dts)
        dt = dts[i]
        y = ys[i]
        p = 1.0
        if dt > 0
            g = sqrt(dt)
            separation = min(g, abs(2.0 - g))
            scale = max(1.0, g)
            if separation > 1.0e-3 * scale
                p = exp(-g * y)
            else
                p = 1.0 - g * y
            end
        end
        s += p * y
    end
    return s
end

# Without a loop: three paths that return.
@noinline function skipped_sqrt_three_way(x::Float64, y::Float64)
    x <= 0 && return y * y
    s = sqrt(x)
    c = max(s, y)
    if 1.0e-3 * c >= min(s, 2.0)
        return y * exp(s)
    else
        return s * y
    end
end
skipped_sqrt_three_way_v(v) = skipped_sqrt_three_way(v[1], v[2])

function skipped_sqrt_reference(f, ys, xs)
    c = [x <= 0 ? 1.0 : (sqrt(x) >= 2 ? exp(sqrt(x)) : sqrt(x)) for x in xs]
    if f === skipped_sqrt_times
        return c
    elseif f === skipped_sqrt_squared
        return 2 .* c .^ 2 .* ys
    else
        return map(ys, xs) do y, dt
            dt <= 0 && return 1.0
            g = sqrt(dt)
            if min(g, abs(2.0 - g)) > 1.0e-3 * max(1.0, g)
                exp(-g * y) * (1 - g * y)
            else
                1.0 - 2 * g * y
            end
        end
    end
end

@testset "Reverse does not recompute a call on a path that skipped it" begin
    xs = [2.0, -1.0, 9.0, -3.0, 0.5, 0.0, 4.0]
    ys = [0.3, 0.7, -0.2, 1.1, 0.9, -0.4, 0.25]

    for f in (skipped_sqrt_times, skipped_sqrt_squared, skipped_sqrt_propagator)
        ref = skipped_sqrt_reference(f, ys, xs)
        @testset "$f" begin
            for mode in (Reverse, set_runtime_activity(Reverse))
                dys = zero(ys)
                autodiff(mode, f, Active, Duplicated(ys, dys), Const(xs))
                @test dys ≈ ref
            end

            dys = zero(ys)
            fwd, rev = autodiff_thunk(
                ReverseSplitWithPrimal, Const{typeof(f)}, Active,
                Duplicated{typeof(ys)}, Const{typeof(xs)},
            )
            tape, primal, _ = fwd(Const(f), Duplicated(ys, dys), Const(xs))
            rev(Const(f), Duplicated(ys, dys), Const(xs), 1.0, tape)
            @test primal ≈ f(ys, xs)
            @test dys ≈ ref
            @test xs == [2.0, -1.0, 9.0, -3.0, 0.5, 0.0, 4.0]
        end
    end

    @testset "three paths without a loop" begin
        for (v, ref) in (([-1.0, 0.5], [0.0, 1.0]), ([4.0, 0.5], [0.125, 2.0]))
            dv = zero(v)
            autodiff(Reverse, skipped_sqrt_three_way_v, Active, Duplicated(v, dv))
            @test dv ≈ ref
        end
    end
end
