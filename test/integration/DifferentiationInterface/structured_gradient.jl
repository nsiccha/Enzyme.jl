using DifferentiationInterface: DifferentiationInterface as DI, AutoEnzyme
using Enzyme: Enzyme
using Test

di_tuple_loss(x) = sum(abs2, x[1]) + x[2][]^2

@testset "Structured gradient destination without contexts" begin
    for backend in (
            AutoEnzyme(; mode = Enzyme.Reverse),
            AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const),
        )
        point = ([1.0, 2.0, 3.0], Ref(4.0))
        dest = (fill(NaN, 3), Ref(NaN))
        prep = DI.prepare_gradient(di_tuple_loss, backend, point)
        for scale in (1.0, 2.0)
            point[1] .= scale .* [1.0, 2.0, 3.0]
            point[2][] = scale * 4.0
            value, result = DI.value_and_gradient!(di_tuple_loss, dest, prep, backend, point)
            expected_value, expected = DI.value_and_gradient(di_tuple_loss, prep, backend, point)
            @test value == expected_value == di_tuple_loss(point)
            @test result === dest
            @test dest[1] == expected[1] == 2 .* point[1]
            @test dest[2][] == expected[2][] == 2 * point[2][]
            result = DI.gradient!(di_tuple_loss, dest, prep, backend, point)
            @test result === dest
            @test dest[1] == 2 .* point[1]
            @test dest[2][] == 2 * point[2][]
        end
    end
end
