using Enzyme, LinearAlgebra, Test

# A `Const` argument is merged with a fresh copy of itself, and the merged value goes
# with active data to a call that is not inlined, which activity analysis assumes
# may write active data into it. `getproperty(::Cholesky, :L)` returns
# `LowerTriangular(uplo == 'L' ? factors : copy(factors'))`, and on Julia 1.12 that
# copy goes through `Base.unalias` (`mightalias(dest, src) ? unaliascopy(src) : src`).
@noinline function scale_by_diagonal!(b, A)
    for i in eachindex(b)
        @inbounds b[i] *= A[i, i]
    end
    return b
end

copy_or_alias(A, e) = e ? A : copy(A')
copy_or_alias_scaled(x, A, e) = sum(abs2, scale_by_diagonal!(copy(x), copy_or_alias(A, e)))

cholesky_lower_solve(x, F, y) = sum(abs2, F.L \ (y .- x))
cholesky_upper_adjoint_solve(x, F, y) = sum(abs2, F.U' \ (y .- x))

@testset "Const argument merged with a fresh copy of itself" begin
    A = [2.0 1.0; 1.0 3.0]
    x0 = [1.0, 2.0]
    for e in (true, false)
        dx = zero(x0)
        autodiff(Reverse, copy_or_alias_scaled, Active, Duplicated(copy(x0), dx), Const(A), Const(e))
        @test dx ≈ 2 .* x0 .* [4.0, 9.0]
        @test A == [2.0 1.0; 1.0 3.0]
    end

    n = 4
    M = [sin(i + 2j) for i in 1:n, j in 1:n]
    S = M * M' + n * I
    y = collect(range(-1.0, 1.0; length = n))
    x1 = fill(0.1, n)
    expected = -2 .* (S \ (y .- x1))
    for F in (cholesky(S), cholesky(Hermitian(S, :L))),
            f in (cholesky_lower_solve, cholesky_upper_adjoint_solve)
        factors = copy(F.factors)
        dx = zero(x1)
        autodiff(Reverse, f, Active, Duplicated(copy(x1), dx), Const(F), Const(y))
        @test dx ≈ expected
        @test F.factors == factors

        d1 = zero(x1)
        d2 = zero(x1)
        autodiff(Reverse, f, Active, BatchDuplicated(copy(x1), (d1, d2)), Const(F), Const(y))
        @test d1 ≈ expected
        @test d2 ≈ expected

        dx = zero(x1)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            Duplicated{typeof(x1)}, Const{typeof(F)}, Const{typeof(y)},
        )
        tape, primal, _ = fwd(Const(f), Duplicated(copy(x1), dx), Const(F), Const(y))
        rev(Const(f), Duplicated(copy(x1), dx), Const(F), Const(y), 1.0, tape)
        @test primal ≈ f(x1, F, y)
        @test dx ≈ expected
    end
end

@testset "Const Cholesky split reverse batch seeds and shape checks" begin
    n = 4
    M = [sin(i + 2j) for i in 1:n, j in 1:n]
    S = M * M' + n * I
    y = collect(range(-1.0, 1.0; length = n))
    x = fill(0.1, n)
    expected = -2 .* (S \ (y .- x))
    for f in (cholesky_lower_solve, cholesky_upper_adjoint_solve)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            BatchDuplicated{typeof(x), 2}, Const{Cholesky{Float64, Matrix{Float64}}}, Const{typeof(y)},
        )
        for F in (cholesky(S), cholesky(Hermitian(S, :L)))
            d1, d2 = zero(x), zero(x)
            factors = copy(F.factors)
            ax = BatchDuplicated(copy(x), (d1, d2))
            tape, primal, _ = fwd(Const(f), ax, Const(F), Const(y))
            rev(Const(f), ax, Const(F), Const(y), (1.0, 2.0), tape)
            @test primal ≈ f(x, F, y)
            @test d1 ≈ expected
            @test d2 ≈ 2 .* expected
            @test ax.val == x
            @test F.factors == factors
        end
        for uplo in ('U', 'L')
            Fbad = Cholesky(ones(n, n + 1), uplo, 0)
            ax = BatchDuplicated(copy(x), (zero(x), zero(x)))
            @test_throws DimensionMismatch fwd(Const(f), ax, Const(Fbad), Const(y))
        end
    end
end
