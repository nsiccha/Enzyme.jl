using Enzyme, Test

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
