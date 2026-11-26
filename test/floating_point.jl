using Enzyme, Test

double_exponential(x) = exp(-exp(x))
double_exp_forward(x, seed) = only(autodiff(Forward, double_exponential, Duplicated(x, seed)))
double_exp_first(x) = double_exp_forward(x, 1.0)
double_exp_second(x) = only(autodiff(Forward, double_exp_first, Duplicated(x, 1.0)))
double_exp_reverse(x) = only(only(autodiff(Reverse, double_exponential, Active(x))))

function double_exp_reference(x, seeds...)
    a = exp(BigFloat(x))
    p = (-a, a * (a - 1), a * (-a^2 + 3a - 1))
    return Float64(prod(BigFloat, seeds) * p[length(seeds)] * exp(-a))
end

observed_exp(x) = exp(x)
Enzyme.EnzymeRules.@easy_rule(observed_exp(x::Float64), (7.0,))

@testset "Derivatives of exp(-exp(x))" begin
    for x in (-750.0, -1.0, 0.0, 2.0, 6.5, log(750.0), 354.0, 355.0, 700.0, 710.0, 990.0, 1000.0)
        first, second, third = (double_exp_reference(x, ntuple(Returns(1.0), n)...) for n in 1:3)
        tangent, value = autodiff(ForwardWithPrimal, double_exponential, Duplicated(x, 1.0))
        @test isequal(value, double_exponential(x))
        for d in (tangent, double_exp_reverse(x))
            @test isapprox(d, first; rtol = 2.0e-13, atol = 0.0)
        end
        for d in (
                double_exp_second(x),
                only(autodiff(Forward, double_exp_reverse, Duplicated(x, 1.0))),
                only(only(autodiff(Reverse, double_exp_first, Active(x)))),
            )
            @test isapprox(d, second; rtol = 2.0e-13, atol = 0.0)
        end
        @test isapprox(only(only(autodiff(Reverse, double_exp_second, Active(x)))), third; rtol = 2.0e-13, atol = 0.0)
    end
    for (x, seed) in ((log(750.0), 2.0), (-750.0, 1.0e300), (7.0, -1.0e200))
        @test double_exp_forward(x, seed) ≈ double_exp_reference(x, seed) rtol = 2.0e-13
    end
    x = log(750.0)
    @test Tuple(only(autodiff(Forward, double_exponential, BatchDuplicated(x, (0.0, -3.0))))) ==
        (0.0, double_exp_reference(x, -3.0))
    # A later derivative can act on the seed as well as on x.
    dx, ds = only(autodiff(Reverse, double_exp_forward, Active(x), Active(0.0)))
    @test dx == 0.0
    @test ds == double_exp_reference(x, 1.0)
    @test only(autodiff(Forward, double_exp_forward, Duplicated(x, 2.0), Duplicated(-3.0, 0.0))) ==
        double_exp_reference(x, 2.0, -3.0)
    # The two seeds overflow if multiplied before the common exponential scale.
    @test only(autodiff(Forward, double_exp_forward, Duplicated(7.0, 1.0e200), Const(1.0e200))) ≈
        double_exp_reference(7.0, 1.0e200, 1.0e200) rtol = 2.0e-13
    scaled(t) = 1.0e200 * double_exponential(t)
    @test only(only(autodiff(Reverse, scaled, Active(7.0)))) ≈ double_exp_reference(7.0, 1.0e200) rtol = 2.0e-13

    # Both exponentials also occur in ordinary likelihoods or can have other uses.
    loglik(t) = -0.5 * ((0.4 - double_exponential(t)) / 0.2)^2
    score(t) = only(autodiff(Forward, loglik, Duplicated(t, 1.0)))
    curvature(t) = only(autodiff(Forward, score, Duplicated(t, 1.0)))
    @test only(only(autodiff(Reverse, curvature, Active(1000.0)))) == 0.0
    function population(t, offsets)
        means = @. exp(-exp(t + offsets))
        return -(t^2 + sum(abs2, (0.4 .- means) ./ 0.2)) / 2
    end
    @test only(autodiff(Reverse, population, Active(1000.0), Const([-1.0, 2.0])))[1] == -1000.0
    function shared(t)
        a = exp(t)
        return exp(-a) + a
    end
    @test only(only(autodiff(Reverse, shared, Active(0.0)))) ≈ 1 - exp(-1)
    @test only(autodiff(Forward, shared, Duplicated(1000.0, 1.0))) == Inf
    custom(t) = exp(-observed_exp(t))
    @test only(autodiff(Forward, custom, Duplicated(0.0, 1.0))) ≈ -7 * exp(-1)
    @test only(only(autodiff(Reverse, custom, Active(0.0)))) ≈ -7 * exp(-1)
    @test isnan(double_exp_first(NaN))
    @test isnan(only(only(autodiff(Reverse, double_exp_second, Active(NaN)))))
    singular(t) = t / t
    @test isnan(only(autodiff(Forward, singular, Duplicated(0.0, 1.0))))
end

struct ScaledExpLog{T}
    scale::T
    offset::T
end

(f::ScaledExpLog)(t) = exp(log(t * f.scale) + f.offset)

@testset "Representable derivatives through small intermediates" begin
    # Multiplying the exponential's adjoint by scale before dividing by t*scale
    # underflows for these inputs. A relative check also detects the precision
    # lost when that product is subnormal instead of zero.
    for scale in (1.0e-100, 1.0e-160, 1.0e-190), offset in (0.0, 0.5), t in (0.5, 1.0, 2.0)
        f = ScaledExpLog(scale, offset)
        expected = Float64(BigFloat(scale) * exp(BigFloat(offset)))
        @test gradient(Reverse, Const(f), t)[1] ≈ expected rtol = 2.0e-13 atol = 0.0
        @test autodiff(Forward, Const(f), Duplicated(t, 1.0))[1] ≈ expected rtol = 2.0e-13 atol = 0.0
    end
end

@testset "Forward derivatives of a division by a tiny or huge divisor" begin
    # The forward rule of x / y used to divide by y * y, which underflows to zero for
    # |y| < 1.5e-154 and overflows for |y| > 1.3e154, although the tangent is representable.
    @test autodiff(Forward, b -> 1.0 / b, Duplicated(1.0e-188, 1.0e-188))[1] ≈ -1.0e188 rtol = 1.0e-15
    @test autodiff(Forward, x -> x / 1.0e200, Duplicated(3.0, 1.0))[1] ≈ 1.0e-200 rtol = 1.0e-15
    @test autodiff(Forward, y -> 3.0 / y, Duplicated(1.0e160, 1.0e150))[1] ≈ -3.0e-170 rtol = 1.0e-15
    # Each nested derivative through a division squared the divisor once more.
    dlog(b) = autodiff(Reverse, log, Active, Active(b))[1][1]
    @test autodiff(Forward, dlog, Duplicated(1.0e-188, 1.0e-188))[1] ≈ -1.0e188 rtol = 1.0e-15
    # log(exp(-x) / sqrt(pi)) = -x - log(sqrt(pi)): Hessian 0 and third derivative 0.
    shifted(x) = log(exp(-x) / sqrt(pi))
    dshifted(x) = autodiff(Reverse, shifted, Active, Active(x))[1][1]
    third(x) = autodiff(Forward, y -> autodiff(Forward, z -> autodiff(Forward, shifted, Duplicated(z, 1.0))[1], Duplicated(y, 1.0))[1], Duplicated(x, 1.0))[1]
    for x in (200.0, 400.0)
        @test autodiff(Forward, dshifted, Duplicated(x, 1.0))[1] ≈ 0.0 atol = 1.0e-12
        @test third(x) ≈ 0.0 atol = 1.0e-9
    end
    # The rule goes through dy / y, not through x / y: a quotient outside Float64's range
    # does not disturb a representable tangent.
    quotient(x, y) = x / y
    @test autodiff(Forward, quotient, Duplicated(3.0, 2.0), Duplicated(7.0, 5.0))[1] ≈ -1 / 49 rtol = 1.0e-15
    # x / y overflows, and y has an exactly zero tangent, whose product with an infinite
    # quotient would be NaN. The second function has a finite primal past that quotient.
    @test autodiff(Forward, quotient, Duplicated(1.0e308, 1.0), Duplicated(1.0e-10, 0.0))[1] ≈ 1.0e10 rtol = 1.0e-15
    @test autodiff(set_strong_zero(Forward), quotient, Duplicated(1.0e308, 1.0), Duplicated(1.0e-10, 0.0))[1] ≈ 1.0e10 rtol = 1.0e-15
    saturating(x, y) = exp(-x / y)
    @test autodiff(Forward, saturating, Duplicated(1.0e308, 1.0), Duplicated(1.0e-10, 0.0))[1] == 0.0
    # x / y underflows to zero, or to a subnormal with few digits.
    @test autodiff(Forward, quotient, Duplicated(1.0e-300, 0.0), Duplicated(1.0e100, 1.0e300))[1] ≈ -1.0e-200 rtol = 1.0e-15
    @test autodiff(Forward, quotient, Duplicated(1.0e-300, 0.0), Duplicated(1.0e10, 1.0e160))[1] ≈ -1.0e-160 rtol = 1.0e-15
end

function sum_of_squares(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i] * x[i]
    end
    return s
end

# Each compilation below needs its own function: a derivative already compiled keeps
# the flags it was compiled with.
negzero_times_default(x) = -0.0 * x
negzero_times_strict(x) = -0.0 * x

@testset "Fast-math flags of derivative arithmetic" begin
    derivative_ir() = sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, sum_of_squares, Active, Tuple{Duplicated{Vector{Float64}}})
    end
    plain_zero_add = r"fadd double [^,]+, 0\.000000e\+00"
    x = [0.1 * i for i in 1:100]

    # By default derivative arithmetic may ignore the sign of a zero and fuse a multiply
    # and an add, but never regroups an expression or assumes away NaN or infinity.
    ir = derivative_ir()
    @test occursin("nsz contract", ir)
    @test !occursin("reassoc", ir)
    @test !occursin(" fast ", ir)
    # So an adjoint's `+ 0.0` folds away, and the reverse loop vectorizes without
    # reassociation.
    @test !occursin(plain_zero_add, ir)
    @test occursin("vector.body", ir)
    @test gradient(Reverse, sum_of_squares, x)[1] ≈ 2 .* x
    @test signbit(gradient(Reverse, negzero_times_default, 1.0)[1])

    try
        # Fully strict: `0.0 + -0.0` is `+0.0`, so the adjoint's `+ 0.0` stays.
        Enzyme.API.fast_math_flags!()
        ir = derivative_ir()
        @test !occursin("nsz", ir)
        @test !occursin(" contract ", ir)
        @test occursin(plain_zero_add, ir)
        @test !signbit(gradient(Reverse, negzero_times_strict, 1.0)[1])
    finally
        Enzyme.API.fast_math_flags!(nsz = true, contract = true)
    end
end
