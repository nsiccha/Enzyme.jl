using Enzyme
using LinearAlgebra
using Test

function complex_cholesky_pivot(t, A, D)
    B = A .+ t .* D
    return real(cholesky(Hermitian(B, :L)).L[2, 2])
end

@testset "Hermitian Cholesky pivot" begin
    for T in (Float64, ComplexF64)
        A = T[4 0; 1 9]
        D = T[0 0; 1 0]
        T <: Complex && (A[2, 1] += 2im)
        t = 0.35
        z = A[2, 1] + t
        expected = -real(z) / (4 * sqrt(9 - abs2(z) / 4))
        (dt, _, _) = only(autodiff(Reverse, complex_cholesky_pivot, Active, Active(t), Const(A), Const(D)))
        @test dt ≈ expected
    end
end

potrf_result!(uplo, A) = LAPACK.potrf!(uplo, A)

@testset "Complex potrf! storage and batched pullbacks" begin
    for T in (ComplexF32, ComplexF64), uplo in ('L', 'U')
        # Both triangles, including the one passing through potrf! untouched,
        # have independent values. The imaginary diagonal is ignored by LAPACK.
        A = T[7 + 1im 1 + 2im 2 - im; -2 + 1im 9 - 1im 1 + 1im; 1 - im 1 + 2im 8 + 2im]
        seeds = (T[1 + im 2 1 - im; 1 + 2im 3im -1; im 2 + im 1], T[im -1 im; 2 1 + im 1; 2 - im -2im 3])
        input = BatchDuplicated(copy(A), (zero(A), zero(A)))
        fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(potrf_result!)}, BatchDuplicated, Const{Char}, typeof(input))
        tape, (factor, info), shadow = fwd(Const(potrf_result!), Const(uplo), input)
        @test iszero(info)
        @test factor === input.val
        for b in 1:2
            @test first(shadow[b]) === input.dval[b]
            copyto!(first(shadow[b]), seeds[b])
        end
        # The reverse pass needs the factor as it was at the end of potrf!.
        fill!(factor, T(NaN))
        rev(Const(potrf_result!), Const(uplo), input, tape)
        # One finite difference per real coordinate, including unused storage.
        h = T === ComplexF32 ? 0.01f0 : 1.0e-5
        for i in eachindex(A), direction in (1, im)
            perturbed = copy(A)
            perturbed[i] += h * direction
            plus = first(potrf_result!(uplo, perturbed))
            perturbed = copy(A)
            perturbed[i] -= h * direction
            minus = first(potrf_result!(uplo, perturbed))
            for b in 1:2
                expected = real(dot(seeds[b], plus - minus)) / (2h)
                got = real(conj(input.dval[b][i]) * direction)
                @test isapprox(got, expected; atol = T === ComplexF32 ? 2.0e-4 : 1.0e-9)
            end
        end
    end
end

# potrf! returns a positive status for an indefinite matrix. A caller can use
# that status and the untouched triangle without using the partial factor.
function failed_potrf_fallback(A)
    F, info = LAPACK.potrf!('L', A)
    return iszero(info) ? real(F[2, 2]) : imag(F[1, 2])
end

@testset "Complex potrf! failure status" begin
    A = ComplexF64[1 3im; 2 1]
    dA = zero(A)
    autodiff(Reverse, failed_potrf_fallback, Active, Duplicated(A, dA))
    @test dA == [0 im; 0 0]
end
