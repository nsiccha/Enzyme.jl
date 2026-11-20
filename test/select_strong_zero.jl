using Enzyme, Test
import Libdl

# A guarded reciprocal `x > t ? 1 / x : 0` has derivative 0 where the guard is
# false. LLVM speculates the division into a select (`select c, 1 / x, 0`), so it
# runs on every path; every spelling of the guard below compiles to that same
# select. Where the guard is false the reverse pass gives the division a zero
# adjoint, and before the dogfood libEnzyme repair its rule then computed 0 / 0
# and 0 * inf at x = 0: the Reverse derivative was NaN, and so was everything an
# unused output of the same function reached. Forward mode was right.

double_guard(l, t) = l > t ? 1 / (l > t ? l : 1.0) : 0.0
single_guard(l, t) = l > t ? 1 / l : 0.0
safe_operand(l, t) = 1 / (l > t ? l : Inf)
ifelse_guard(l, t) = ifelse(l > t, 1 / l, 0.0)

const SPELLINGS = (double_guard, single_guard, safe_operand, ifelse_guard)

rev(f, x, t) = autodiff(Reverse, f, Active, Active(x), Const(t))[1][1]
fwd(f, x, t) = autodiff(Forward, f, Duplicated(x, 1.0), Const(t))[1]

# The shape of the report (an exact Kalman update): a pseudo-inverse of a
# covariance with an eigenvalue of 0 among outputs that the caller does not use.
@noinline function inverse_and_sum(l1, l2, t)
    il1 = l1 > t ? 1 / (l1 > t ? l1 : 1.0) : 0.0
    il2 = l2 > t ? 1 / (l2 > t ? l2 : 1.0) : 0.0
    return (il1 + il2, l1 + 2l2)
end
second_output(l) = inverse_and_sum(l[1], l[2], 1.0e-12)[2]
both_outputs(l) = sum(inverse_and_sum(l[1], l[2], 1.0e-12))

function guarded_inverse_sum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i] > 0.5 ? 1 / x[i] : 0.0
    end
    return s
end

function guarded_quotient_chain(w, x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i] > 0.5 ? w[1] / x[i] * w[2] : 0.0
    end
    return s
end

unguarded(x) = 1 / x

guarded_gradient(x) = (dx = zero(x); autodiff(Reverse, guarded_inverse_sum, Active, Duplicated(x, dx)); dx)

# Each is first differentiated under the setting of the testset that calls it.
off_guard(l, t) = l > t ? 1 / l : 0.0
on_guard(l, t) = l > t ? 1 / l : 0.0

function guarded_integer(x::T, n::Int) where {T <: AbstractFloat}
    if isfinite(n) && isinteger(n) && n >= 0 && isfinite(x) && 0 <= x && x <= 1
        return x * x
    else
        return T(-Inf)
    end
end

# Finite inputs can also overflow before a select discards the result.
function guarded_fourth(x)
    if -1 <= x <= 1
        y = x * x
        return y * y
    else
        return zero(x)
    end
end

function guarded_product(x, y)
    return isfinite(x) && isfinite(y) ? x * y : zero(x)
end

function guarded_square_sum(x)
    s = zero(eltype(x))
    for v in x
        s += guarded_integer(v, 5)
    end
    return s
end

function guarded_square_gradient(x)
    dx = zero(x)
    autodiff(Reverse, guarded_square_sum, Active, Duplicated(x, dx))
    return dx
end

unguarded_square(x) = x * x

function guarded_endpoint(b, m, s)
    b == -Inf && return -Inf
    z = (b - m) / s
    return exp(-z^2)
end

function endpoint_sum(x, y, lo)
    m, s = x[1], exp(x[2])
    total = 0.0
    @simd for i in eachindex(y)
        v = y[i]
        total += if v < lo || v > 2.0
            -Inf
        else
            upper = guarded_endpoint(2.0, m, s)
            mass = lo == -Inf ? upper : upper - guarded_endpoint(lo, m, s)
            -((v - m) / s)^2 / 2 - log(s) - mass
        end
    end
    return total
end

@testset "Endpoint arithmetic hoisted ahead of its only use" begin
    x, y = [0.7, log(0.6)], [1.2, 0.9, -0.5]
    for lo in (-Inf, -1.0)
        dx = zero(x)
        autodiff(Reverse, endpoint_sum, Active, Duplicated(x, dx), Const(y), Const(lo))
        for i in eachindex(x)
            h = zero(x)
            h[i] = 1.0e-6
            fd = (endpoint_sum(x + h, y, lo) - endpoint_sum(x - h, y, lo)) / (2 * h[i])
            @test dx[i] ≈ fd rtol = 1.0e-8
        end
    end
end

# The finite-endpoint arm compares a value computed from the endpoint, such as
# the interval width `b - a`, before it uses the endpoint. LLVM computes that
# value ahead of the loop together with `a`, and only the comparison reads it.
for (name, test) in (
        (:width, :(b - a >= 0.125)),
        (:sum, :(b + a >= 0.125)),
        (:scaled, :(2.0 * a >= 0.125)),
        (:all, :((a >= -8) & (b <= 8) & (b - a >= 0.125))),
    )
    @eval function $(Symbol(:compared_lower_, name))(x, y, lo)
        m, s = x[1], exp(x[2])
        total = 0.0
        @simd for i in eachindex(y)
            v = y[i]
            total += if v < lo || v > 2.0
                -Inf
            else
                b = (2.0 - m) / s
                mass = if lo == -Inf
                    exp(-b * b)
                else
                    a = (lo - m) / s
                    $test ? exp(-a * a) : 0.0
                end
                -((v - m) / s)^2 / 2 - log(s) - mass
            end
        end
        return total
    end
end

function compared_upper_width(x, y, hi)
    m, s = x[1], exp(x[2])
    total = 0.0
    @simd for i in eachindex(y)
        v = y[i]
        total += if v < 0.0 || v > hi
            -Inf
        else
            a = (0.0 - m) / s
            mass = if hi == Inf
                exp(-a * a)
            else
                b = (hi - m) / s
                b - a >= 0.125 ? exp(-b * b) : 0.0
            end
            -((v - m) / s)^2 / 2 - log(s) - mass
        end
    end
    return total
end

@testset "Endpoint arithmetic hoisted ahead of a comparison" begin
    x = [0.7, log(0.6)]
    cases = (
        (compared_lower_width, [1.2, 0.9, -0.5], (-Inf, -1.0)),
        (compared_lower_sum, [1.2, 0.9, -0.5], (-Inf, -1.0)),
        (compared_lower_scaled, [1.2, 0.9, -0.5], (-Inf, -1.0)),
        (compared_lower_all, [1.2, 0.9, -0.5], (-Inf, -1.0)),
        (compared_upper_width, [1.2, 0.9, 0.5], (Inf, 2.0)),
    )
    @testset "$f, bound $bound" for (f, y, bounds) in cases, bound in bounds
        dx = zero(x)
        autodiff(Reverse, f, Active, Duplicated(x, dx), Const(y), Const(bound))
        for i in eachindex(x)
            h = zero(x)
            h[i] = 1.0e-6
            fd = (f(x + h, y, bound) - f(x - h, y, bound)) / (2 * h[i])
            @test dx[i] ≈ fd rtol = 1.0e-8
        end
    end
end

@testset "Select-masked nonfinite products" begin
    @testset "$T" for T in (Float32, Float64)
        for x in T.((0.0, 0.4, 1.0, -0.2, 1.2, NaN, Inf, -Inf, floatmax(T)))
            valid = isfinite(x) && 0 <= x <= 1
            expected = valid ? 2x : zero(x)
            result = autodiff(ReverseWithPrimal, guarded_integer, Active, Active(x), Const(5))
            @test result[1][1] == expected
            @test result[2] == (valid ? x * x : T(-Inf))
            @test autodiff(Reverse, guarded_integer, Active, Active(x), Const(-1))[1][1] == 0
            @test autodiff(Reverse, guarded_fourth, Active, Active(x))[1][1] ≈ (isfinite(x) && -1 <= x <= 1 ? 4x^3 : zero(x))
        end
        for x in T.((NaN, Inf, -Inf)), y in T.((0.4, NaN, Inf, -Inf))
            dx, dy = autodiff(Reverse, guarded_product, Active, Active(x), Active(y))[1]
            @test dx == 0
            @test dy == 0
        end
        @test autodiff(Reverse, guarded_product, Active, Active(T(0.4)), Active(T(0.6)))[1] == (T(0.6), T(0.4))
        @test isnan(autodiff(Reverse, unguarded_square, Active, Active(T(NaN)))[1][1])
        @test autodiff(Reverse, unguarded_square, Active, Active(T(Inf)))[1][1] == Inf
    end

    x = [0.0, 0.4, 1.0, -0.2, 1.2, NaN, Inf, -Inf]
    expected = [0.0, 0.8, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test guarded_square_gradient(x) == expected
    dx1, dx2 = zero(x), zero(x)
    autodiff(Reverse, guarded_square_sum, Active, BatchDuplicated(x, (dx1, dx2)))
    @test dx1 == expected
    @test dx2 == expected

    fwdthunk, revthunk = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(guarded_square_sum)}, Active,
        Duplicated{Vector{Float64}},
    )
    dx = zero(x)
    tape, primal, _ = fwdthunk(Const(guarded_square_sum), Duplicated(x, dx))
    revthunk(Const(guarded_square_sum), Duplicated(x, dx), 1.0, tape)
    @test primal == -Inf
    @test dx == expected
    @test autodiff(Forward, guarded_square_gradient, Duplicated(x, ones(length(x))))[1] == [2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end

@testset "Guarded division speculated into a select" begin
    @testset "$(nameof(f))" for f in SPELLINGS
        @test f(0.0, 1.0e-12) == 0.0
        @test rev(f, 0.0, 1.0e-12) == 0.0
        @test fwd(f, 0.0, 1.0e-12) == 0.0
        @test rev(f, 2.0, 1.0e-12) == -0.25
        @test fwd(f, 2.0, 1.0e-12) == -0.25
        dx, = autodiff(set_runtime_activity(Reverse), f, Active, Active(0.0), Const(1.0e-12))[1]
        @test dx == 0.0
        res = autodiff(ReverseWithPrimal, f, Active, Active(0.0), Const(1.0e-12))
        @test res[1][1] == 0.0
        @test res[2] == 0.0
    end

    @testset "unused output of the same function" begin
        g = zeros(2)
        autodiff(Reverse, second_output, Active, Duplicated([0.0, 3.0], g))
        @test g == [1.0, 2.0]
        g = zeros(2)
        autodiff(Reverse, both_outputs, Active, Duplicated([0.0, 2.0], g))
        @test g == [1.0, 2.0 - 0.25]
        g = zeros(2)
        autodiff(Reverse, both_outputs, Active, Duplicated([4.0, 2.0], g))
        @test g == [1.0 - 1 / 16, 2.0 - 0.25]
    end

    @testset "in a loop" begin
        x = [0.0, 2.0, 0.0, 4.0, 0.25]
        expected = [0.0, -0.25, 0.0, -1 / 16, 0.0]
        dx = zero(x)
        autodiff(Reverse, guarded_inverse_sum, Active, Duplicated(copy(x), dx))
        @test dx == expected

        dx1, dx2 = zero(x), zero(x)
        autodiff(Reverse, guarded_inverse_sum, Active, BatchDuplicated(copy(x), (dx1, dx2)))
        @test dx1 == expected
        @test dx2 == expected

        w = [0.7, 1.3]
        dw, dxc = zero(w), zero(x)
        autodiff(Reverse, guarded_quotient_chain, Active, Duplicated(copy(w), dw), Duplicated(copy(x), dxc))
        live = x .> 0.5
        @test dw ≈ [w[2] * sum(1 ./ x[live]), w[1] * sum(1 ./ x[live])]
        @test dxc ≈ [l ? -w[1] * w[2] / v^2 : 0.0 for (l, v) in zip(live, x)]
        @test all(isfinite, dxc)
    end

    @testset "split mode" begin
        fwdthunk, revthunk = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(guarded_inverse_sum)}, Active,
            Duplicated{Vector{Float64}},
        )
        x = [0.0, 2.0, 0.0]
        dx = zero(x)
        tape, primal, _ = fwdthunk(Const(guarded_inverse_sum), Duplicated(x, dx))
        revthunk(Const(guarded_inverse_sum), Duplicated(x, dx), 1.0, tape)
        @test primal == 0.5
        @test dx == [0.0, -0.25, 0.0]
    end

    @testset "Forward over Reverse" begin
        x = [0.0, 2.0, 0.0, 4.0]
        hv = autodiff(Forward, guarded_gradient, Duplicated(x, [1.0, 1.0, 1.0, 1.0]))[1]
        @test hv == [0.0, 2 / 8, 0.0, 2 / 64]
    end

    @testset "a real pole keeps its derivative" begin
        @test autodiff(Reverse, unguarded, Active, Active(0.0))[1][1] == -Inf
        @test autodiff(Reverse, unguarded, Active, Active(2.0))[1][1] == -0.25
    end

    @testset "libEnzyme option -enzyme-select-strong-zero" begin
        sym = :EnzymeSelectStrongZero
        ptr = Libdl.dlsym(Libdl.dlopen(Enzyme.API.libEnzyme), sym)
        @test Enzyme.API.EnzymeGetCLBool(ptr) == 1
        # A derivative compiled while the option is off keeps the previous rules.
        Enzyme.API.EnzymeSetCLBool(sym, false)
        try
            @test isnan(rev(off_guard, 0.0, 1.0e-12))
        finally
            Enzyme.API.EnzymeSetCLBool(sym, true)
        end
        @test Enzyme.API.EnzymeGetCLBool(ptr) == 1
        @test rev(on_guard, 0.0, 1.0e-12) == 0.0
    end
end
