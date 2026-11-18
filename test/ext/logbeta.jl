module LogBetaTests

using Enzyme
using SpecialFunctions
using Test
using FiniteDifferences

logabsbeta_value(a, b) = first(logabsbeta(a, b))
logabsbeta_sign(a, b) = last(logabsbeta(a, b))
reverse_a(f, a, b) = autodiff(Reverse, f, Active, Active(a), Active(b))[1][1]
forward_a(f, a, b) = autodiff(Forward, f, Duplicated(a, one(a)), Const(b))[1]

function beta_sum(x, y)
    s = zero(eltype(x))
    for i in eachindex(x, y)
        s += logbeta(x[i], y[i])
    end
    return s
end

function beta_sum_gradient!(dx, x, y)
    fill!(dx, zero(eltype(dx)))
    autodiff(Reverse, beta_sum, Active, Duplicated(x, dx), Const(y))
    return nothing
end

@testset "log beta derivatives" begin
    for T in (Float32, Float64), (a0, b0) in (
                (0.3, 0.7), (2.3, 0.7), (0.7, 2.3), (2.5, 8.0),
                (8.0, 2.5), (2.5, 100.0), (100.0, 2.5), (1.0e16, 1.0), (1.0, 1.0e16),
                (-0.5, 1.25), (-2.5, 0.7), (-0.5, -0.25),
            )
        a, b = T(a0), T(b0)
        da = T(digamma(BigFloat(a)) - digamma(BigFloat(a) + BigFloat(b)))
        db = T(digamma(BigFloat(b)) - digamma(BigFloat(a) + BigFloat(b)))
        rtol = T === Float32 ? 2.0f-5 : 2.0e-12
        for f in (logbeta, logabsbeta_value)
            if f === logbeta && last(logabsbeta(a, b)) < 0
                @test_throws DomainError autodiff(Reverse, f, Active, Active(a), Active(b))
                @test_throws DomainError autodiff(Forward, f, Duplicated(a, one(a)), Const(b))
                continue
            end
            rev = autodiff(ReverseWithPrimal, f, Active, Active(a), Active(b))
            @test rev[2] == f(a, b)
            @test isapprox(rev[1][1], da; rtol)
            @test isapprox(rev[1][2], db; rtol)
            @test isapprox(autodiff(Reverse, f, Active, Active(a), Const(b))[1][1], da; rtol)
            @test isapprox(autodiff(Reverse, f, Active, Const(a), Active(b))[1][2], db; rtol)
            fwd = autodiff(ForwardWithPrimal, f, Duplicated(a, T(2)), Duplicated(b, T(-3)))
            @test fwd[2] == f(a, b)
            @test isapprox(fwd[1], 2da - 3db; rtol)
            batch = autodiff(Forward, f, BatchDuplicated(a, (one(a), zero(a))), BatchDuplicated(b, (zero(b), one(b))))[1]
            @test isapprox(batch[1], da; rtol)
            @test isapprox(batch[2], db; rtol)
        end
        @test autodiff(Forward, logabsbeta_sign, Duplicated(a, one(a)), Const(b)) == ()
    end

    for f in (logbeta, logabsbeta_value), (a, b) in ((2.3, 0.7), (100.0, 2.5))
        da, db = FiniteDifferences.grad(central_fdm(5, 1), f, a, b)
        actual = autodiff(Reverse, f, Active, Active(a), Active(b))[1]
        @test actual[1] ≈ da
        @test actual[2] ≈ db
        aa = trigamma(a) - trigamma(a + b)
        ab = -trigamma(a + b)
        for df in (reverse_a, forward_a)
            @test autodiff(Forward, df, Const(f), Duplicated(a, 1.0), Const(b))[1] ≈ aa
            @test autodiff(Forward, df, Const(f), Const(a), Duplicated(b, 1.0))[1] ≈ ab
        end
        second = autodiff(Reverse, forward_a, Active, Const(f), Active(a), Active(b))[1]
        @test second[2] ≈ aa
        @test second[3] ≈ ab
    end

    # SpecialFunctions continues the value at these coincident gamma poles
    # through logbeta(1-a-b, b). Preserve the derivatives of that branch.
    for (a, b, da, db) in ((-2.0, 1.0, 0.5, -1.0), (-3.0, 2.0, 5 / 6, 0.0))
        for (x, y, dx, dy) in ((a, b, da, db), (b, a, db, da))
            actual = autodiff(Reverse, logabsbeta_value, Active, Active(x), Active(y))[1]
            @test actual[1] ≈ dx
            @test actual[2] ≈ dy
            @test autodiff(Forward, logabsbeta_value, Duplicated(x, 1.0), Const(y))[1] ≈ dx
        end
    end

    # Mixed precision and constant integer arguments retain normal promotion.
    for (a, b) in ((0.5f0, 100.0), (100.0, 0.5f0), (1, 1.0e16), (1.0e16, 1))
        actual = autodiff(Forward, logbeta, Const(a), Duplicated(float(b), one(float(b))))[1]
        expected = Float64(digamma(BigFloat(b)) - digamma(BigFloat(a) + BigFloat(b)))
        @test isapprox(actual, expected; rtol = 2.0e-5)
    end

    for f in (logbeta, logabsbeta_value)
        fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Active{Float64}, Active{Float64})
        tape, primal, _ = fwd(Const(f), Active(2.3), Active(0.7))
        @test primal == f(2.3, 0.7)
        actual = rev(Const(f), Active(2.3), Active(0.7), 2.0, tape)[1]
        @test actual[1] ≈ 2 * (digamma(2.3) - digamma(3.0))
        @test actual[2] ≈ 2 * (digamma(0.7) - digamma(3.0))
    end
end

@testset "log beta reverse allocations" begin
    for n in (8, 256)
        x = [0.5 + mod(i, 17) / 4 for i in 1:n]
        y = [1.5 + mod(i, 23) / 2 for i in 1:n]
        dx = zero(x)
        beta_sum_gradient!(dx, x, y)
        expected = digamma.(x) .- digamma.(x .+ y)
        @test dx ≈ expected
        @test (@allocated beta_sum_gradient!(dx, x, y)) <= 256
        dx1, dx2 = zero(x), zero(x)
        autodiff(Reverse, beta_sum, Active, BatchDuplicated(x, (dx1, dx2)), Const(y))
        @test dx1 ≈ expected
        @test dx2 ≈ expected
    end
end

end
