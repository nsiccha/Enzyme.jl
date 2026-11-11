using Enzyme, Test

@static if VERSION >= v"1.11-"
    function storage_objective!(x, count, hint, first, shrink)
        Base._deleteend!(x, count)
        sizehint!(x, hint; first, shrink)
        return sum(abs2, x)
    end

    function storage_mutation_only!(x, count, hint)
        Base._deleteend!(x, count)
        sizehint!(x, hint)
        return nothing
    end

    @testset "Storage survives collection between split passes" begin
        original = collect(1.0:8.0)
        x, dx = copy(original), zero(original)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(storage_objective!)}, Active,
            Duplicated{Vector{Float64}}, Const{Int}, Const{Int}, Const{Bool}, Const{Bool}
        )
        tape, primal, _ = fwd(Const(storage_objective!), Duplicated(x, dx), Const(3), Const(100), Const(false), Const(true))
        # Both vectors now refer to new storage. Reverse must retain what it
        # needs from before the resize even if a GC runs between the passes.
        GC.gc()
        rev(Const(storage_objective!), Duplicated(x, dx), Const(3), Const(100), Const(false), Const(true), 1.0, tape)
        @test primal == sum(abs2, original[1:5])
        @test x == original[1:5]
        @test dx == [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0]
    end

    @testset "Inactive storage mutation lanes" begin
        original = collect(1.0:8.0)
        for count in (0, 3, 8), hint in (0, 100)
            x = copy(original)
            autodiff(set_runtime_activity(Reverse), storage_mutation_only!, Const, Duplicated(x, x), Const(count), Const(hint))
            @test x == original[1:(8 - count)]
            for inactive_lane in (1, 2)
                x, dx = copy(original), zero(original)
                shadows = inactive_lane == 1 ? (x, dx) : (dx, x)
                autodiff(set_runtime_activity(Reverse), storage_mutation_only!, Const, BatchDuplicated(x, shadows), Const(count), Const(hint))
                @test x == original[1:(8 - count)]
                @test dx == zero(original)
            end
        end
    end

    @testset "Forward sizehint return aliases" begin
        x, dx, dy = collect(1.0:8.0), ones(8), fill(2.0, 8)
        result = autodiff(ForwardWithPrimal, sizehint!, Duplicated(x, dx), Const(100))
        @test result[1] === dx
        @test result[2] === x
        result = autodiff(ForwardWithPrimal, sizehint!, BatchDuplicated(x, (dx, dy)), Const(200))
        @test result[1][1] === dx
        @test result[1][2] === dy
        @test result[2] === x
        @test dx == ones(8)
        @test dy == fill(2.0, 8)
    end

    @testset "Forward storage changes" begin
        original = Float64[1, 2, 3, 4, 5, 6, 7, 8]
        tangent = cos.(eachindex(original))
        for count in (0, 1, 3, 8), hint in (0, 100), first in (false, true), shrink in (false, true)
            n = length(original) - count
            first && n == 0 && continue
            x, dx = copy(original), copy(tangent)
            derivative = only(autodiff(Forward, storage_objective!, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink)))
            @test derivative ≈ 2sum(original[1:n] .* tangent[1:n])
            @test dx == tangent[1:n]
            @test x == original[1:n]
        end
    end

    function storage_alias_objective(x, count, hint, first, shrink)
        y = copy(x)
        old = y.ref.mem
        Base._deleteend!(y, count)
        sizehint!(y, hint; first, shrink)
        return sum(abs2, y) + 3sum(old)
    end

    @testset "Array storage restoration" begin
        for T in (Float32, Float64), mode in (Reverse, set_runtime_activity(Reverse))
            original = T[1, 2, 3, 4, 5, 6, 7, 8]
            for count in (0, 1, 3, 8), first in (false, true), shrink in (false, true)
                n = length(original) - count
                # Base.sizehint! itself rejects some first=true calls on an empty vector.
                first && n == 0 && continue
                for hint in (0, n, 100)
                    x, dx = copy(original), zero(original)
                    autodiff(mode, storage_objective!, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                    expected = [i <= n ? 2original[i] : zero(T) for i in eachindex(original)]
                    @test x == original[1:n]
                    @test dx == expected

                    x, dx1, dx2 = copy(original), zero(original), zero(original)
                    autodiff(mode, storage_objective!, Active, BatchDuplicated(x, (dx1, dx2)), Const(count), Const(hint), Const(first), Const(shrink))
                    @test dx1 == expected
                    @test dx2 == expected

                    x, dx = copy(original), zero(original)
                    autodiff(mode, storage_alias_objective, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                    @test dx == expected .+ 3
                    @test x == original

                    if count == 0
                        x, dx = copy(original), fill(T(17), length(original))
                        autodiff(mode, storage_objective!, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                        @test dx == expected .+ 17
                    end
                end
            end
        end
    end
end
