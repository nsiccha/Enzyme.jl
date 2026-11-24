using Enzyme
using LinearAlgebra
using Test

# BLAS routines libEnzyme has no derivative for in a mode (Forward `trmm`, `trmv`,
# `spmv`) are differentiated through their reference bitcode. On Julia 1.12 and later
# a BLAS call is named `ejlstr$<routine>$<library>`; that declaration must be inlined
# into the bitcode, as the plainly named one is on Julia 1.10, instead of reaching
# libEnzyme's native rule for the routine ("cannot handle blas argument within trmm").

const TRI_L0 = [2.0 0 0; 0.5 3 0; -1 0.25 1.5]
const TRI_M = [1.0 0 0; 2 1 0; -1 0.5 3]
const TRI_B0 = [1.0 2; 3 4; 5 6]

tri_lmul(::Type{W}, t) where {W} = sum(lmul!(W(TRI_L0 .+ t .* TRI_M), TRI_B0 .+ t))
tri_rmul(::Type{W}, t) where {W} = sum(rmul!(permutedims(TRI_B0) .+ t, W(TRI_L0 .+ t .* TRI_M)))
tri_mulvec(::Type{W}, t) where {W} = sum(W(TRI_L0 .+ t .* TRI_M) * (TRI_B0[:, 1] .+ t))

# The same products in Julia arithmetic: the tangent at `t` is `p1 + 2 * p2 * t`, and the
# second derivative is `2 * p2`.
function tri_poly(f, W)
    L0 = Matrix(W(TRI_L0))
    # A unit wrapper reads its diagonal as one, so `t` does not reach it.
    M = Matrix(W(TRI_M))
    if W <: Union{UnitLowerTriangular, UnitUpperTriangular}
        M = M - I
    end
    if f === tri_lmul
        return sum(M * TRI_B0 .+ L0 * ones(size(TRI_B0))), sum(M * ones(size(TRI_B0)))
    elseif f === tri_rmul
        B = permutedims(TRI_B0)
        return sum(B * M .+ ones(size(B)) * L0), sum(ones(size(B)) * M)
    else
        x = TRI_B0[:, 1]
        return sum(M * x .+ L0 * ones(length(x))), sum(M * ones(length(x)))
    end
end

const TRI_WRAPPERS = (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)

# The tangent of `f(W, t)` at `t`, and the second derivative by Forward over Forward.
tri_tangent(f, W, t) = autodiff(Forward, Base.Fix1(f, W), Duplicated(t, 1.0))[1]
tri_second(f, W, t) = autodiff(Forward, tri_tangent, Const(f), Const(W), Duplicated(t, 1.0))[1]

# Forward over Forward through libEnzyme's Forward rule for potrf, which calls trmm.
# `chol22_literal` is the reporter's form. Its matrix literal mixes Float64 and Int, and
# on Julia 1.10 Forward over Forward through it raises EnzymeRuntimeActivityError at
# `hvcat_fill!` before it reaches potrf, a separate defect (noted on Enzyme todo
# `1kiq87k`). `chol22` fills the same matrix entry by entry, so that Julia 1.10 checks
# potrf too.
chol22_literal(t) = LAPACK.potrf!('L', [4.0 0; 1 + t 9])[1][2, 2]
function chol22(t)
    A = Matrix{Float64}(undef, 2, 2)
    A[1, 1] = 4.0
    A[2, 1] = 1 + t
    A[1, 2] = 0.0
    A[2, 2] = 9.0
    return LAPACK.potrf!('L', A)[1][2, 2]
end
dchol22(s) = autodiff(Forward, chol22, Duplicated(s, 1.0))[1]
dchol22_literal(s) = autodiff(Forward, chol22_literal, Duplicated(s, 1.0))[1]

# Complex triangular products take the same path (`ztrmm`, `ctrmv`, ...). Only the real
# scalar `t` is active, and the result is a polynomial in it: the tangent at `t` is
# `real(sum(M * B + L * J)) + 2 * t * real(sum(M * J))` for `J` all ones, with `L` and
# `M` conjugated where the product takes the adjoint.
const TRI_LC = ComplexF64[2 0 0; 0.5 + 1im 3 0; -1 0.25 - 0.5im 1.5]
const TRI_MC = ComplexF64[1 0 0; 2 1 0; -1 0.5im 3]
const TRI_BC = ComplexF64[1 2im; 3 4; 5 - 1im 6]

ctri_lmul(L, M, B, t) = real(sum(lmul!(LowerTriangular(L .+ t .* M), B .+ t)))
ctri_adjmul(L, M, B, t) = real(sum(UpperTriangular(permutedims(L .+ t .* M))' * (B .+ t)))
ctri_mulvec(L, M, B, t) = real(sum(LowerTriangular(L .+ t .* M) * (B[:, 1] .+ t)))

function ctri_tangent(f, L, M, B, t)
    if f === ctri_adjmul
        L, M = conj(L), conj(M)
    elseif f === ctri_mulvec
        B = B[:, 1:1]
    end
    J = ones(eltype(B), size(B))
    return real(sum(M * B .+ L * J)) + 2 * t * real(sum(M * J))
end

# Packed symmetric matrix-vector product: Forward has no native rule; Reverse uses the
# bitcode too, as the routine is not among the native Reverse rules Enzyme selects.
const SPMV_AP0 = [4.0, 1.0, 5.0, 2.0, 0.5, 6.0]
const SPMV_DAP = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
const SPMV_X0 = [1.0, -2.0, 0.5]
function packed_symv(t)
    y = zeros(3)
    BLAS.spmv!('U', 1.0, SPMV_AP0 .+ t .* SPMV_DAP, SPMV_X0 .+ t, 0.0, y)
    return sum(abs2, y)
end
packed_dense(AP) = [AP[1] AP[2] AP[4]; AP[2] AP[3] AP[5]; AP[4] AP[5] AP[6]]
packed_symv_reference(t) = sum(abs2, packed_dense(SPMV_AP0 .+ t .* SPMV_DAP) * (SPMV_X0 .+ t))

@testset "BLAS routines differentiated through their reference implementation" begin
    @testset "Forward through triangular products (trmm, trmv): $f, $W" for f in (tri_lmul, tri_rmul, tri_mulvec), W in TRI_WRAPPERS
        p1, p2 = tri_poly(f, W)
        for t in (0.0, 0.1, -0.7)
            @test tri_tangent(f, W, t) ≈ p1 + 2 * p2 * t
            @test collect(autodiff(Forward, Base.Fix1(f, W), BatchDuplicated(t, (1.0, 2.0)))[1]) ≈
                [p1 + 2 * p2 * t, 2 * (p1 + 2 * p2 * t)]
        end
        @test tri_second(f, W, 0.1) ≈ 2 * p2
        @test autodiff(Reverse, Base.Fix1(f, W), Active, Active(0.1))[1][1] ≈ p1 + 2 * p2 * 0.1
    end

    @testset "Forward through trmm in Float32" begin
        L0, M, B0 = Float32.(TRI_L0), Float32.(TRI_M), Float32.(TRI_B0)
        f(t) = sum(lmul!(LowerTriangular(L0 .+ t .* M), B0 .+ t))
        p1, p2 = tri_poly(tri_lmul, LowerTriangular)
        @test autodiff(Forward, f, Duplicated(0.25f0, 1.0f0))[1] ≈ Float32(p1 + 2 * p2 * 0.25) rtol = 1.0f-5
    end

    @testset "Forward through complex trmm and trmv: $f, $E" for f in (ctri_lmul, ctri_adjmul, ctri_mulvec), E in (ComplexF64, ComplexF32)
        L, M, B = E.(TRI_LC), E.(TRI_MC), E.(TRI_BC)
        for t in real(E).((0.0, 0.1, -0.7))
            @test autodiff(Forward, f, Const(L), Const(M), Const(B), Duplicated(t, one(t)))[1] ≈ ctri_tangent(f, L, M, B, t)
        end
    end

    @testset "Forward over Forward through potrf" begin
        for t in (0.0, 0.35, -0.5)
            a = 1 + t
            l22 = sqrt(9 - a^2 / 4)
            second = -(1 / 4) / l22 - (a / 4)^2 / l22^3
            @test dchol22(t) ≈ -(a / 4) / l22
            @test autodiff(Forward, dchol22, Duplicated(t, 1.0))[1] ≈ second
            @test dchol22_literal(t) ≈ -(a / 4) / l22
            if VERSION >= v"1.12"
                @test autodiff(Forward, dchol22_literal, Duplicated(t, 1.0))[1] ≈ second
            else
                @test_broken autodiff(Forward, dchol22_literal, Duplicated(t, 1.0))[1] ≈ second
            end
        end
    end

    @testset "spmv in Forward and Reverse" begin
        for t in (0.0, 0.3)
            expected = autodiff(Forward, packed_symv_reference, Duplicated(t, 1.0))[1]
            @test packed_symv(t) ≈ packed_symv_reference(t)
            @test autodiff(Forward, packed_symv, Duplicated(t, 1.0))[1] ≈ expected
            @test autodiff(Reverse, packed_symv, Active, Active(t))[1][1] ≈ expected
        end
    end

    @testset "Repair registry" begin
        @test Enzyme.has_dogfood_repair(:blas_fallback_wrappers)
    end
end
