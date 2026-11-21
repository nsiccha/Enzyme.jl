using Enzyme
using Test

broadcast_residual(u, x) = 0.2 - (u[1] .+ x .* u[2])

@testset "Materializing a rank-zero broadcast returns its scalar element" begin
    u = [0.3, -0.2]
    for x in (0.5, fill(0.5))
        du = zero(u)
        _, value = autodiff(ReverseWithPrimal, broadcast_residual, Active, Duplicated(u, du), Const(x))
        @test value === broadcast_residual(u, x)
        @test du == [-1.0, -0.5]
    end
end

@testset "Broadcasting a rank-zero array with a vector returns a vector" begin
    x = [1.0, 2.0]
    dx, value = autodiff(ForwardWithPrimal, broadcast, Duplicated, Const(*), Const(fill(0.5)), Duplicated(x, ones(2)))
    @test value == [0.5, 1.0]
    @test dx == [0.5, 0.5]
end
