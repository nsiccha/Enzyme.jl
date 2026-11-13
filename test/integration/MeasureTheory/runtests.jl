using Enzyme
using MeasureTheory
using Test

const observations = [0.5, 0.2, -0.1]

range_density(x) = logdensityof(For(i -> Normal(x[1], 1.0), 1:3), observations)
vector_density(x, offsets) = logdensityof(For(z -> Normal(x[1] + z, 1.0), offsets), observations)
active_vector_density(x) = logdensityof(For(m -> Normal(m, x[2]), [x[1], 0.3]), observations[1:2])

@testset "For measures with captured active values" begin
    x = [0.1]
    dx = zero(x)
    _, value = autodiff(ReverseWithPrimal, range_density, Active, Duplicated(x, dx))
    @test value ≈ sum(logdensityof(Normal(x[1], 1.0), y) for y in observations)
    @test dx ≈ [sum(observations .- x[1])]

    offsets = [0.0, 1.0, 2.0]
    original_offsets = copy(offsets)
    fill!(dx, 0)
    _, value = autodiff(ReverseWithPrimal, vector_density, Active, Duplicated(x, dx), Const(offsets))
    @test value ≈ sum(logdensityof(Normal(x[1] + z, 1.0), y) for (z, y) in zip(offsets, observations))
    @test dx ≈ [sum(observations .- x[1] .- offsets)]
    @test offsets == original_offsets

    x = [0.1, 1.0]
    dx = zero(x)
    _, value = autodiff(ReverseWithPrimal, active_vector_density, Active, Duplicated(x, dx))
    residuals = observations[1:2] .- [x[1], 0.3]
    @test value ≈ sum(logdensityof(Normal(m, x[2]), y) for (m, y) in zip([x[1], 0.3], observations))
    @test dx ≈ [residuals[1] / x[2]^2, -2 / x[2] + sum(abs2, residuals) / x[2]^3]
end
