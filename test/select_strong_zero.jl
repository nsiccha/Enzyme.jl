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
