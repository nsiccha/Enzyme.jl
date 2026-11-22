using Enzyme, LinearAlgebra, Test

# Julia 1.10 copies each ComplexF32 element of copyto!(dest, Hermitian(B)) as
# one i64: the upper float of a stored element is unpacked with lshr 32 and
# trunc, the mirrored element negates its imaginary part as a float, the
# diagonal one has a zero imaginary part, and the merged halves are packed
# again with zext, shl 32 and or. Type analysis lost the float type of the
# shifted half, so Forward stopped with "cannot handle unknown binary
# operator: shl" and Reverse with "Cannot deduce adding type (cast)" (repair
# `packed_float_lanes`, in libEnzyme). The copy is real-linear in B, so each
# derivative has an exact expected value.

herm_copy!(dest, B, uplo) = (copyto!(dest, Hermitian(B, uplo)); nothing)

function weighted_copy(B, W, uplo)
    C = copyto!(similar(B), Hermitian(B, uplo))
    s = zero(real(eltype(B)))
    for i in eachindex(C, W)
        s += real(C[i]) * real(W[i]) + imag(C[i]) * imag(W[i])
    end
    return s
end

# The gradient of a real-linear function of a complex matrix, one element at a
# time: real part from the real unit, imaginary part from the imaginary unit.
function linear_gradient(f, B)
    G = similar(B)
    for k in eachindex(B)
        E = zero(B)
        E[k] = 1
        re = f(E)
        E[k] = im
        G[k] = complex(re, f(E))
    end
    return G
end

@testset "copyto! of a Hermitian matrix, $T, n = $n, :$uplo" for T in (ComplexF32, ComplexF64),
        n in (1, 2, 3, 5), uplo in (:L, :U)
    B = [T(sin(i + 2j), cos(3i - j)) for i in 1:n, j in 1:n]
    dB = [T(cos(i * j), sin(i + j)) for i in 1:n, j in 1:n]
    dB2 = [T(i - j, i + 2j) for i in 1:n, j in 1:n]
    W = [T(1 + i - j, 2i + j) for i in 1:n, j in 1:n]

    dest = zeros(T, n, n)
    ddest = zeros(T, n, n)
    autodiff(Forward, herm_copy!, Const, Duplicated(dest, ddest), Duplicated(copy(B), copy(dB)), Const(uplo))
    @test dest == Matrix(Hermitian(B, uplo))
    @test ddest == Matrix(Hermitian(dB, uplo))

    dest = zeros(T, n, n)
    dests = (zeros(T, n, n), zeros(T, n, n))
    autodiff(Forward, herm_copy!, Const, BatchDuplicated(dest, dests),
        BatchDuplicated(copy(B), (copy(dB), copy(dB2))), Const(uplo))
    @test dests[1] == Matrix(Hermitian(dB, uplo))
    @test dests[2] == Matrix(Hermitian(dB2, uplo))

    gB = zero(B)
    v = autodiff(ReverseWithPrimal, weighted_copy, Active, Duplicated(copy(B), gB), Const(W), Const(uplo))[2]
    @test v ≈ weighted_copy(B, W, uplo)
    @test gB ≈ linear_gradient(E -> weighted_copy(E, W, uplo), B)
end

@testset "the reported scalar case" begin
    A = ComplexF32[4 0; 1+2im 9]
    D = ComplexF32[0 0; 1 0]
    f(t, A, D) = (B = A .+ t .* D; real(copyto!(similar(B), Hermitian(B, :L))[1, 2]))
    g(t, A, D) = (B = A .+ t .* D; sum(abs2, copyto!(similar(B), Hermitian(B, :L))))
    @test autodiff(Forward, Const(f), Duplicated(0.35f0, 1.0f0), Const(A), Const(D))[1] == 1
    @test autodiff(Reverse, Const(f), Active, Active(0.35f0), Const(A), Const(D))[1][1] == 1
    # d/dt of |1.35 + 2im|^2 twice (stored and mirrored) is 4 * 1.35
    @test autodiff(Forward, Const(g), Duplicated(0.35f0, 1.0f0), Const(A), Const(D))[1] ≈ 5.4f0
    @test autodiff(Reverse, Const(g), Active, Active(0.35f0), Const(A), Const(D))[1][1] ≈ 5.4f0
end
