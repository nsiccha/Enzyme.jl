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
