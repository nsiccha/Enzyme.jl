using Enzyme, Test

@inline function storage_overlaps(a, b)
    isempty(a) && return false
    isempty(b) && return false
    return GC.@preserve a b begin
        pa, pb = UInt(pointer(a)), UInt(pointer(b))
        pa <= pb ? pb - pa < sizeof(a) : pa - pb < sizeof(b)
    end
end

function checked_broadcast!(dst, x, h, g)
    length(dst) == length(x) == length(g) || error("length mismatch")
    if storage_overlaps(dst, x) || storage_overlaps(dst, g)
        copyto!(dst, x .+ (h / 2) .* g)
    else
        dst .= x .+ (h / 2) .* g
    end
    return nothing
end

function reverse_checked!(dst, ddst, x, dx, g, dg)
    partials = only(
        autodiff(
            Reverse, checked_broadcast!, Const, Duplicated(dst, ddst),
            Duplicated(x, dx), Active(0.5), Duplicated(g, dg),
        ),
    )
    return partials[3]
end

@testset "Reverse through a GC-preserved storage overlap check" begin
    @testset "disjoint destination" begin
        x, g, dst = [2.0, 3.0], [5.0, 7.0], zeros(2)
        dx, dg, ddst = zeros(2), zeros(2), ones(2)
        @test reverse_checked!(dst, ddst, x, dx, g, dg) == 6.0
        @test dst == [3.25, 4.75]
        @test dx == [1.0, 1.0]
        @test dg == [0.25, 0.25]
        @test ddst == [0.0, 0.0]
    end

    @testset "destination overlaps the first input by one element" begin
        # Use the same dense-vector specialization as the disjoint call. A
        # direct left-to-right write would overwrite x[2] before reading it.
        data, shadow = [2.0, 3.0, 0.0], [0.0, 2.0, -1.0]
        g, dg = [5.0, 7.0], zeros(2)
        GC.@preserve data shadow begin
            x = unsafe_wrap(Array, pointer(data), 2)
            dst = unsafe_wrap(Array, pointer(data, 2), 2)
            dx = unsafe_wrap(Array, pointer(shadow), 2)
            ddst = unsafe_wrap(Array, pointer(shadow, 2), 2)
            @test reverse_checked!(dst, ddst, x, dx, g, dg) == 1.5
        end
        @test data == [2.0, 3.25, 4.75]
        @test shadow == [2.0, -1.0, 0.0]
        @test dg == [0.5, -0.25]
    end

    @testset "destination is the second input" begin
        x, g = [2.0, 3.0], [5.0, 7.0]
        dx, dg = zeros(2), [2.0, -1.0]
        @test reverse_checked!(g, dg, x, dx, g, dg) == 1.5
        @test g == [3.25, 4.75]
        @test dx == [2.0, -1.0]
        @test dg == [0.5, -0.25]
    end
end
