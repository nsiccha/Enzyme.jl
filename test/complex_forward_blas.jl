using Enzyme, LinearAlgebra, Test

# Forward mode through complex BLAS and LAPACK calls (the c and z routines).
#
# libEnzyme's Forward rules add each tangent term after the first with
# `beta = 1`. A complex routine's scalar is the pair (re, im), and the rule
# generator used to emit every rule constant `v` as the splat (v, v), that is
# v + v * im. So with two or more active arguments every term but the first
# came out multiplied by 1 + im: Forward of `A * x` with `A` and `x` both
# active returned `dA * x + (1 + im) * A * dx`, with no error. The Forward rule
# of `potrf` was also the real symmetric formula: it transposed where a
# Hermitian factor needs the conjugate transpose and did not conjugate the
# triangle it copied.
#
# Every case below is a real scalar of complex data that depends on one active
# real `t`, the way the reporter met it, and its derivative is compared with the
# exact tangent (products are linear in each factor; the Cholesky tangent is
# `L * Phi(L \ dP / L')`). n = 5 keeps LinearAlgebra on BLAS: it multiplies 2×2
# and 3×3 matrices in Julia.

const CF_N = 5

cf_re(w, M) = real(dot(w, M))

# Tangent of `t -> F(t)` at `t`, by Forward mode with `t` active.
cf_fwd(F, t) = autodiff(Forward, Const(F), Duplicated(t, one(t)))[1]

function cf_data(T, n)
    R = real(T)
    c(i, j) = T(sin(1.3i + 0.7j), cos(0.4i - 1.1j))
    A0 = [c(i, j) for i in 1:n, j in 1:n]
    dA = [c(j + 3, i - 2) for i in 1:n, j in 1:n]
    B0 = [c(2i, j + 1) for i in 1:n, j in 1:n]
    dB = [c(i - 1, 3j) for i in 1:n, j in 1:n]
    x0 = [c(i, 2) for i in 1:n]
    dx = [c(3, i) for i in 1:n]
    W = [c(i + j, i * j) for i in 1:n, j in 1:n]
    wv = [c(5, i) for i in 1:n]
    P0 = A0 * A0' + R(n) * I
    dP = dA + dA'
    return (; A0, dA, B0, dB, x0, dx, W, wv, P0, dP)
end

const cf_libblas = LinearAlgebra.BLAS.libblastrampoline

for (fname, T) in ((:zpotrf_, ComplexF64), (:cpotrf_, ComplexF32))
    @eval function cf_potrf!(uplo::Char, A::Matrix{$T})
        n = size(A, 1)
        info = Ref{LinearAlgebra.BlasInt}(0)
        ccall(
            (LinearAlgebra.BLAS.@blasfunc($fname), cf_libblas), Cvoid,
            (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{$T}, Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Clong),
            uplo, n, A, max(1, stride(A, 2)), info, 1
        )
        info[] == 0 || throw(PosDefException(info[]))
        return A
    end
end

# The tangent of the lower Cholesky factor of P along dP (dP Hermitian).
function cf_chol_tangent(P, dP)
    L = cholesky(Hermitian(P, :L)).L
    X = L \ dP / L'
    Phi = tril(X)
    for i in axes(Phi, 1)
        Phi[i, i] = real(Phi[i, i]) / 2
    end
    return Matrix(L), L * Phi
end

@testset "Complex Forward BLAS and LAPACK" begin
    @testset "A * x with both factors active ($T)" for T in (ComplexF64, ComplexF32)
        d = cf_data(T, CF_N)
        f(A, x) = A * x
        dy = autodiff(Forward, f, Duplicated(d.A0, d.dA), Duplicated(d.x0, d.dx))[1]
        @test dy ≈ d.dA * d.x0 + d.A0 * d.dx rtol = sqrt(eps(real(T)))
        dy2 = autodiff(Forward, f, Const(d.A0), Duplicated(d.x0, d.dx))[1]
        @test dy2 ≈ d.A0 * d.dx rtol = sqrt(eps(real(T)))
    end

    @testset "$name ($T)" for T in (ComplexF64, ComplexF32), (name, F, dF) in (
                (
                    "gemv A*x", (d, t) -> cf_re(d.wv, (d.A0 + t * d.dA) * (d.x0 + t * d.dx)),
                    (d, t) -> cf_re(d.wv, d.dA * (d.x0 + t * d.dx) + (d.A0 + t * d.dA) * d.dx),
                ),
                (
                    "gemv A'*x", (d, t) -> cf_re(d.wv, (d.A0 + t * d.dA)' * (d.x0 + t * d.dx)),
                    (d, t) -> cf_re(d.wv, d.dA' * (d.x0 + t * d.dx) + (d.A0 + t * d.dA)' * d.dx),
                ),
                (
                    "gemv transpose(A)*x", (d, t) -> cf_re(d.wv, transpose(d.A0 + t * d.dA) * (d.x0 + t * d.dx)),
                    (d, t) -> cf_re(d.wv, transpose(d.dA) * (d.x0 + t * d.dx) + transpose(d.A0 + t * d.dA) * d.dx),
                ),
                (
                    "gemm A*B", (d, t) -> cf_re(d.W, (d.A0 + t * d.dA) * (d.B0 + t * d.dB)),
                    (d, t) -> cf_re(d.W, d.dA * (d.B0 + t * d.dB) + (d.A0 + t * d.dA) * d.dB),
                ),
                (
                    "gemm A'*B", (d, t) -> cf_re(d.W, (d.A0 + t * d.dA)' * (d.B0 + t * d.dB)),
                    (d, t) -> cf_re(d.W, d.dA' * (d.B0 + t * d.dB) + (d.A0 + t * d.dA)' * d.dB),
                ),
                (
                    "gemm A*B'", (d, t) -> cf_re(d.W, (d.A0 + t * d.dA) * (d.B0 + t * d.dB)'),
                    (d, t) -> cf_re(d.W, d.dA * (d.B0 + t * d.dB)' + (d.A0 + t * d.dA) * d.dB'),
                ),
                (
                    "gemm A*A' (two arrays)", (d, t) -> cf_re(d.W, (d.A0 + t * d.dA) * (d.A0 + t * d.dA)'),
                    (d, t) -> cf_re(d.W, d.dA * (d.A0 + t * d.dA)' + (d.A0 + t * d.dA) * d.dA'),
                ),
                (
                    "symv Symmetric(A)*x", (d, t) -> cf_re(d.wv, Symmetric(d.A0 + t * d.dA) * (d.x0 + t * d.dx)),
                    (d, t) -> cf_re(d.wv, Symmetric(d.dA) * (d.x0 + t * d.dx) + Symmetric(d.A0 + t * d.dA) * d.dx),
                ),
                (
                    "symm Symmetric(A)*B", (d, t) -> cf_re(d.W, Symmetric(d.A0 + t * d.dA) * (d.B0 + t * d.dB)),
                    (d, t) -> cf_re(d.W, Symmetric(d.dA) * (d.B0 + t * d.dB) + Symmetric(d.A0 + t * d.dA) * d.dB),
                ),
                (
                    "syrk BLAS.syrk", (d, t) -> cf_re(d.W, Symmetric(BLAS.syrk('U', 'N', one(eltype(d.A0)), d.A0 + t * d.dA))),
                    (d, t) -> cf_re(d.W, Symmetric(triu(d.dA * transpose(d.A0 + t * d.dA) + (d.A0 + t * d.dA) * transpose(d.dA)))),
                ),
                (
                    "gemv with alpha and beta", (d, t) -> cf_re(d.wv, BLAS.gemv!('N', eltype(d.A0)(0.3, 0.2), d.A0 + t * d.dA, d.x0 + t * d.dx, eltype(d.A0)(0.5, -0.1), d.x0 + t * d.dx)),
                    (d, t) -> cf_re(d.wv, eltype(d.A0)(0.3, 0.2) * (d.dA * (d.x0 + t * d.dx) + (d.A0 + t * d.dA) * d.dx) + eltype(d.A0)(0.5, -0.1) * d.dx),
                ),
            )
        d = cf_data(T, CF_N)
        t = real(T)(0.1)
        @test cf_fwd(t -> F(d, t), t) ≈ dF(d, t) rtol = sqrt(eps(real(T)))
    end

    @testset "potrf $(uplo) ($T)" for T in (ComplexF64, ComplexF32), uplo in ('L', 'U')
        d = cf_data(T, CF_N)
        t = real(T)(0.1)
        F(t) = cf_re(d.W, LAPACK.potrf!(uplo, d.P0 + t * d.dP)[1])
        L, dL = cf_chol_tangent(d.P0 + t * d.dP, d.dP)
        # potrf returns the factor in the `uplo` triangle and leaves the other
        # triangle as it found it, which here moves with t as P does.
        dM = uplo == 'L' ? dL + triu(d.dP, 1) : dL' + tril(d.dP, -1)
        @test cf_fwd(F, t) ≈ cf_re(d.W, dM) rtol = sqrt(eps(real(T)))
    end

    # The same factorization through a ccall of its own, so that libEnzyme's
    # rule for the routine is what differentiates it even where a Julia rule
    # for `LAPACK.potrf!` takes precedence.
    @testset "direct zpotrf/cpotrf ccall $(uplo) ($T)" for T in (ComplexF64, ComplexF32), uplo in ('L', 'U')
        d = cf_data(T, CF_N)
        t = real(T)(0.1)
        F(t) = cf_re(d.W, cf_potrf!(uplo, d.P0 + t * d.dP))
        L, dL = cf_chol_tangent(d.P0 + t * d.dP, d.dP)
        dM = uplo == 'L' ? dL + triu(d.dP, 1) : dL' + tril(d.dP, -1)
        @test cf_fwd(F, t) ≈ cf_re(d.W, dM) rtol = sqrt(eps(real(T)))
    end

    # ComplexF64 only: on Julia 1.10, Forward through the copy that
    # `cholesky(Hermitian(::Matrix{ComplexF32}))` makes raises
    # `EnzymeNoDerivativeError` in `copyto_unaliased!`, a separate defect
    # (snag `forward-complex-defedc65`). The ComplexF32 factorization itself is
    # checked above through `LAPACK.potrf!` and the direct ccall.
    @testset "cholesky(Hermitian(P)).L ($T)" for T in (ComplexF64,)
        d = cf_data(T, CF_N)
        t = real(T)(0.1)
        F(t) = cf_re(d.W, Matrix(cholesky(Hermitian(d.P0 + t * d.dP, :L)).L))
        L, dL = cf_chol_tangent(d.P0 + t * d.dP, d.dP)
        @test cf_fwd(F, t) ≈ cf_re(d.W, dL) rtol = sqrt(eps(real(T)))
    end

    @testset "potrf reads only the real part of the diagonal" begin
        d = cf_data(ComplexF64, CF_N)
        Dim = Diagonal([0.7, -0.3, 1.1, 0.2, -0.9]) * im
        # An imaginary diagonal in the tangent of the stored triangle changes
        # nothing zpotrf reads, so the derivative is that of the Hermitian part.
        F(t) = real(LAPACK.potrf!('L', d.P0 + t * (d.dP + Dim))[1][3, 2])
        G(t) = real(LAPACK.potrf!('L', d.P0 + t * d.dP)[1][3, 2])
        @test cf_fwd(F, 0.1) ≈ cf_fwd(G, 0.1) rtol = 1.0e-12
    end

    @testset "batched Forward of A * x" begin
        d = cf_data(ComplexF64, CF_N)
        f(A, x) = A * x
        dA2 = d.dA .* (0.5 - 0.2im)
        dx2 = d.dx .* (1.0 + 0.3im)
        res = autodiff(Forward, f, BatchDuplicated(d.A0, (d.dA, dA2)), BatchDuplicated(d.x0, (d.dx, dx2)))[1]
        @test res[1] ≈ d.dA * d.x0 + d.A0 * d.dx rtol = 1.0e-12
        @test res[2] ≈ dA2 * d.x0 + d.A0 * dx2 rtol = 1.0e-12
    end

    @testset "Float64 products are unchanged" begin
        A0 = [sin(i + 2j) for i in 1:CF_N, j in 1:CF_N]
        dA = [cos(i - j) for i in 1:CF_N, j in 1:CF_N]
        x0 = [sin(3i) for i in 1:CF_N]
        dx = [cos(2i) for i in 1:CF_N]
        f(A, x) = A * x
        dy = autodiff(Forward, f, Duplicated(A0, dA), Duplicated(x0, dx))[1]
        @test dy ≈ dA * x0 + A0 * dx rtol = 1.0e-12
        P0 = A0 * A0' + CF_N * I
        dP = dA + dA'
        F(t) = sum(LAPACK.potrf!('L', P0 + t * dP)[1][2:end, 1])
        L, dL = cf_chol_tangent(P0 + 0.1 * dP, dP)
        @test cf_fwd(F, 0.1) ≈ sum(dL[2:end, 1]) rtol = 1.0e-10
    end
end
