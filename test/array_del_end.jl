using Enzyme, Test

function array_del_end_objective!(x, count)
    Base._deleteend!(x, count)
    return sum(abs2, x)
end

# Before Julia 1.11, shrinking a Vector calls the runtime's array_del_end.
# Its reverse regrows the shadow and must clear every restored element, in
# every batch lane. Nonzero output seeds make a stale tail deterministic.
@static if VERSION < v"1.11-"
    @testset "Reverse array_del_end clears all restored adjoints" begin
        for T in (Float32, Float64), mode in (Reverse, set_runtime_activity(Reverse))
            original = T[1, 2, 3, 4, 5, 6, 7, 8]
            for count in (0, 1, 3, 8)
                n = length(original) - count
                x = copy(original)
                dx = fill(T(17), length(x))
                autodiff(mode, array_del_end_objective!, Active, Duplicated(x, dx), Const(count))
                expected = [i <= n ? T(17) + 2 * original[i] : zero(T) for i in eachindex(original)]
                @test x == original[1:n]
                @test dx == expected

                x = copy(original)
                dx1 = fill(T(17), length(x))
                dx2 = fill(T(-3), length(x))
                autodiff(mode, array_del_end_objective!, Active, BatchDuplicated(x, (dx1, dx2)), Const(count))
                expected2 = [i <= n ? T(-3) + 2 * original[i] : zero(T) for i in eachindex(original)]
                @test x == original[1:n]
                @test dx1 == expected
                @test dx2 == expected2
            end
        end
    end
end
