using Enzyme, Test

function unalias_sum_squares(x, dest, indices)
    # Keep the mixed pointer/integer SubArray return across a call boundary.
    # On Julia 1.10 this uses an sret and a separate returnRoots argument.
    selected = @noinline Base.unalias(dest, indices)
    result = zero(eltype(x))
    for i in selected
        result += x[i]^2
    end
    return result
end

function unalias_gradient!(dx, x, dest, indices)
    autodiff(Reverse, unalias_sum_squares, Active, Duplicated(x, dx), Const(dest), Const(indices))
    return nothing
end

@testset "Nested differentiation through unalias" begin
    x = [0.7, 1.3, 2.1]
    direction = [0.4, -0.2, 0.7]
    indices = [3, 1, 2, 1, 3]
    for range in (2:4, 1:2:5), aliases in (false, true)
        selected = view(indices, range)
        dest = aliases ? indices : Int[]
        counts = zeros(Int, length(x))
        for i in selected
            counts[i] += 1
        end
        dx = zero(x)
        unalias_gradient!(dx, x, dest, selected)
        @test dx ≈ 2 .* counts .* x

        fill!(dx, 0)
        hvp = zero(x)
        autodiff(
            Forward, unalias_gradient!, Const,
            Duplicated(dx, hvp), Duplicated(x, direction),
            Const(dest), Const(selected),
        )
        @test dx ≈ 2 .* counts .* x
        @test hvp ≈ 2 .* counts .* direction
    end
end
