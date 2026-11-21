using Enzyme
using LinearAlgebra
using Test

sum_packed_solve(x, A, r) = sum(A \ view(x, r))

@testset "Dense solve with a packed right-hand side" begin
    A = [2.0 -0.4; 0.3 1.5]
    x = [17.0, 0.35, -0.2, 23.0]
    r = 2:3
    dx = [2.0, 3.0, 5.0, 7.0]
    expected = copy(dx)
    expected[r] .+= transpose(A) \ ones(2)
    autodiff(Reverse, sum_packed_solve, Active, Duplicated(x, dx), Const(A), Const(r))
    @test dx[r] ≈ expected[r]
    @test dx[[1, 4]] == expected[[1, 4]]

    dA = zero(A)
    _, value = autodiff(ReverseWithPrimal, sum_packed_solve, Active, Duplicated(x, dx), Duplicated(A, dA), Const(r))
    y = A \ x[r]
    z = transpose(A) \ ones(2)
    @test value ≈ sum(y)
    @test dA ≈ -z * transpose(y)
    expected[r] .+= z
    @test dx ≈ expected

    # A rectangular matrix has the same Julia type. Check the least-squares
    # and minimum-norm pullbacks, including the dependence on the matrix.
    for A in ([2.0 -0.4; 0.3 1.5; 0.7 -0.1], [2.0 -0.4 0.7; 0.3 1.5 -0.1])
        r = 2:2:(2 * size(A, 1))
        x = collect(1.0:(last(r) + 1))
        dx = zero(x)
        dA = zero(A)
        autodiff(Reverse, sum_packed_solve, Active, Duplicated(x, dx), Duplicated(A, dA), Const(r))
        expected = zero(x)
        expected[r] = transpose(A) \ ones(size(A, 2))
        @test dx ≈ expected

        for i in eachindex(A)
            step = zero(A)
            step[i] = 1.0e-5
            fd = (sum_packed_solve(x, A + step, r) - sum_packed_solve(x, A - step, r)) / (2 * step[i])
            @test dA[i] ≈ fd atol = 1.0e-8
        end
    end
end
