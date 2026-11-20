using Enzyme, LinearAlgebra, Test

indexed_product(q, D, rows) = sum(abs2, view(D, rows, :) * q)

function indexed_product_grad!(g, q, D, rows)
    autodiff(Reverse, indexed_product, Active, Duplicated(q, g), Const(D), Const(rows))
    return nothing
end

@testset "Boxing an indexed view in generic matrix multiplication" begin
    q0 = [0.3 0.6 -0.8; -0.5 -0.1 0.9]
    v0 = [0.6 -1.0 0.2; 0.7 0.3 -0.9]
    data = [0.6 -0.3; 0.9 -0.75; 1.2 0.5]
    for indices in ([3, 1], [1, 1])
        q, v, rows = copy(q0), copy(v0), copy(indices)
        A = data[indices, :]
        g = zero(q)
        hv = zero(q)
        D = copy(data)
        autodiff(Forward, indexed_product_grad!, Const, Duplicated(g, hv), Duplicated(q, v), Const(D), Const(rows))
        @test D == data
        @test (q, v, rows) == (q0, v0, indices)
        @test g ≈ 2 * A' * (A * q0)
        @test hv ≈ 2 * A' * (A * v0)

        # An active view must still accumulate into its parent's shadow, including
        # both contributions when two rows refer to the same storage.
        D = copy(data)
        dD = zero(D)
        autodiff(Reverse, indexed_product, Active, Const(q), Duplicated(D, dD), Const(rows))
        expected = zero(D)
        dA = 2 * (A * q0) * q0'
        for (i, row) in pairs(indices)
            expected[row, :] .+= dA[i, :]
        end
        @test D == data
        @test (q, rows) == (q0, indices)
        @test dD ≈ expected
    end
end
