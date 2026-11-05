using Enzyme, Test

@noinline returned_gather(x, indices) = x[indices]
@noinline returned_gather_outer(x, indices) = returned_gather(x, indices)
@noinline returned_gather_tuple(x, indices) = (returned_gather(x, indices), length(indices))
@noinline returned_gather_box(x, indices) = [returned_gather(x, indices)]
@noinline returned_gather_unbox(x, indices) = first(returned_gather_box(x, indices))

gather_energy(x, indices) = sum(abs2, returned_gather(x, indices))
outer_gather_energy(x, indices) = sum(abs2, returned_gather_outer(x, indices))
tuple_gather_energy(x, indices) = sum(abs2, first(returned_gather_tuple(x, indices)))
fresh_gather_energy(x, indices) = sum(abs2, returned_gather(2 .* x, indices))
unboxed_gather_energy(x, indices) = sum(abs2, returned_gather_unbox(x, indices))

@testset "Reverse preserves shadows for returned memory" begin
    for (f, scale) in (
            (gather_energy, 1),
            (outer_gather_energy, 1),
            (tuple_gather_energy, 1),
            (fresh_gather_energy, 4),
            (unboxed_gather_energy, 1),
        )
        for indices in (Int[], [3], [3, 2, 1], [2, 2, 1])
            x = [0.2, 0.3, 0.4]
            original = copy(x)
            dx = fill(0.7, length(x))
            expected = copy(dx)
            for i in indices
                expected[i] += 2 * scale * x[i]
            end
            _, primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(indices))
            @test primal ≈ scale * sum(abs2, x[indices])
            @test dx ≈ expected
            @test x == original
        end
    end
end
