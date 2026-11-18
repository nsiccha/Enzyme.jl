using Enzyme, Test

struct GradientStorage
    vector::Vector{Float64}
    scalar::Base.RefValue{Float64}
end

mutable struct MutableGradientStorage
    value::Float64
end

tuple_storage_loss(x) = sum(abs2, x[1]) + x[2][]^2
record_storage_loss(x) = sum(abs2, x.vector) + x.scalar[]^2
nested_storage_loss(x) = sum(abs2, x.vector.values) + x.scalar.value[]^2
ref_storage_loss(x) = x[]^2
mutable_storage_loss(x) = x.value^2
aliased_storage_loss(x) = sum(abs2, x[1]) + sum(abs2, x[2])

@testset "Reverse gradient! structured storage" begin
    for mode in (Reverse, ReverseWithPrimal)
        x = ([1.0, 2.0, 3.0], Ref(4.0))
        dx = (fill(NaN, 3), Ref(NaN))
        for scale in (1.0, 2.0)
            x[1] .= scale .* [1.0, 2.0, 3.0]
            x[2][] = scale * 4.0
            result = gradient!(mode, dx, tuple_storage_loss, x)
            derivs = mode === Reverse ? result : result.derivs
            @test derivs[1] === dx
            @test dx[1] == 2 .* x[1]
            @test dx[2][] == 2 * x[2][]
            @test x[1] == scale .* [1.0, 2.0, 3.0]
            @test x[2][] == scale * 4.0
            if mode === ReverseWithPrimal
                @test result.val == tuple_storage_loss(x)
            end
        end

        for point in (
                (vector = [1.0, 2.0], scalar = Ref(3.0)),
                GradientStorage([1.0, 2.0], Ref(3.0)),
            )
            dest = make_zero(point)
            dest.vector .= NaN
            dest.scalar[] = NaN
            result = gradient!(mode, dest, record_storage_loss, point)
            derivs = mode === Reverse ? result : result.derivs
            @test derivs[1] === dest
            @test dest.vector == [2.0, 4.0]
            @test dest.scalar[] == 6.0
        end

        point = (vector = (values = [1.0, 2.0],), scalar = (value = Ref(3.0),), tag = 7)
        dest = make_zero(point)
        gradient!(mode, dest, nested_storage_loss, point)
        @test dest.vector.values == [2.0, 4.0]
        @test dest.scalar.value[] == 6.0
        @test dest.tag == 7

        point = Ref(3.0)
        dest = Ref(NaN)
        gradient!(mode, dest, ref_storage_loss, point)
        @test dest[] == 6.0
        point = MutableGradientStorage(3.0)
        dest = MutableGradientStorage(NaN)
        gradient!(mode, dest, mutable_storage_loss, point)
        @test dest.value == 6.0

        point = [1.0, 2.0]
        dest = fill(NaN, 2)
        gradient!(mode, (dest, dest), aliased_storage_loss, (point, point))
        @test dest == [4.0, 8.0]

        # Existing array destinations still reset, rather than accumulate.
        gradient!(mode, dest, sum, point)
        @test dest == ones(2)
        gradient!(mode, dest, sum, point)
        @test dest == ones(2)
    end
end

@testset "Reverse gradient! requires writable derivatives" begin
    @test_throws ArgumentError gradient!(Reverse, 0.0, sin, 1.0)
    point = ([1.0, 2.0], 3.0)
    dest = ([5.0, 6.0], 7.0)
    @test_throws ArgumentError gradient!(Reverse, dest, first, point)
    @test dest[1] == [5.0, 6.0]
end
