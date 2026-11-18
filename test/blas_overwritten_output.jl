using Enzyme, LinearAlgebra, Test

# A BLAS routine that computes `out := alpha * op(...) + beta * out` does not
# read `out` when `beta == 0`, and a copy never reads its destination. The
# adjoint of what `out` held before such a call is therefore exactly zero, as it
# is after an ordinary store, whatever the adjoint of the result holds.
#
# The reverse rules used to multiply the output's shadow by `beta` (and the rule
# of `copy` left it alone). `0 * NaN` is `NaN`, so one call whose output adjoint
# held a `NaN` or an infinity left it in the shadow. A caller that keeps one
# shadow for an output buffer across calls, as a prepared gradient with a reused
# work buffer does, then got `NaN` gradients from every later call. Through a
# copy the shadow kept growing, so later gradients were wrong with finite values
# too, and two copies into one buffer were wrong within a single reverse pass.

@noinline function bo_weighted(out, w)
    s = 0.0
    @inbounds for i in eachindex(out, w)
        s += w[i] * out[i]
    end
    return s
end

bo_loss(x, out, w, op) = (op(x, out); bo_weighted(out, w))

function bo_gradient(mode, op, x, out, dout, w)
    dx = zero(x)
    autodiff(mode, bo_loss, Active, Duplicated(x, dx), Duplicated(out, dout), Const(w), Const(op))
    return dx
end

const BO_A = [sin(i + 2j) for i in 1:4, j in 1:4]
const BO_S = BO_A' * BO_A
const BO_AP = [BO_S[i, j] for j in 1:4 for i in 1:j]   # upper triangle, packed by columns
const BO_VEC = [0.5, -0.25, 0.1, 0.3]
const BO_MAT = reshape(collect(0.1:0.1:0.8), 4, 2)

bo_gemv!(x, out) = mul!(out, BO_A, x)
bo_gemv_direct!(x, out) = BLAS.gemv!('N', 1.0, BO_A, x, 0.0, out)
bo_gemv_transposed!(x, out) = BLAS.gemv!('T', 1.0, BO_A, x, 0.0, out)
bo_symv!(x, out) = mul!(out, Symmetric(BO_S), x)
bo_spmv!(x, out) = BLAS.spmv!('U', 1.0, BO_AP, x, 0.0, out)
bo_gemm!(x, out) = mul!(out, BO_A, x)
bo_symm!(x, out) = mul!(out, Symmetric(BO_S), x)
bo_syrk!(x, out) = mul!(out, x', x)
bo_copy!(x, out) = BLAS.blascopy!(length(x), x, 1, out, 1)

# (name, operation, active input, an output buffer of the right shape)
const BO_OVERWRITING = (
    ("gemv through mul!", bo_gemv!, BO_VEC, zeros(4)),
    ("gemv called directly", bo_gemv_direct!, BO_VEC, zeros(4)),
    ("gemv, transposed", bo_gemv_transposed!, BO_VEC, zeros(4)),
    ("symv", bo_symv!, BO_VEC, zeros(4)),
    ("spmv", bo_spmv!, BO_VEC, zeros(4)),
    ("gemm", bo_gemm!, BO_MAT, zeros(4, 2)),
    ("symm", bo_symm!, BO_MAT, zeros(4, 2)),
    ("syrk", bo_syrk!, BO_MAT, zeros(2, 2)),
    ("copy", bo_copy!, BO_VEC, zeros(4)),
)

@testset "Reverse: the adjoint of an overwritten BLAS output is zero" begin
    @testset "$name" for (name, op, x, out0) in BO_OVERWRITING
        @testset "$label" for (label, mode) in (("static", Reverse), ("runtime activity", set_runtime_activity(Reverse)))
            n = length(out0)
            finite = ones(n)
            # One shadow for the output, used by every call, as a caller with a
            # reused work buffer has it.
            out = copy(out0)
            dout = zero(out0)
            reference = bo_gradient(mode, op, x, out, dout, finite)
            @test all(iszero, dout)
            @test reference == bo_gradient(mode, op, x, copy(out0), zero(out0), finite)

            @testset "after an output adjoint of $bad" for bad in (NaN, Inf, -Inf)
                poisoned = copy(finite)
                poisoned[2] = bad
                bo_gradient(mode, op, x, out, dout, poisoned)
                # 0 * NaN and 0 * Inf must not be what clears the shadow.
                @test all(iszero, dout)
                @test bo_gradient(mode, op, x, out, dout, finite) == reference
                @test all(iszero, dout)
                fill!(dout, 0.0)
            end

            # Finite adjoints only: a shadow used again gives what a fresh one gives.
            fill!(dout, 0.0)
            for w in (finite, 2 .* finite, collect(1.0:n))
                @test bo_gradient(mode, op, x, out, dout, w) == bo_gradient(mode, op, x, copy(out0), zero(out0), w)
                @test all(iszero, dout)
            end
        end
    end
end

# With beta != 0 the routine does read the output, and its adjoint is scaled.
bo_gemv_accumulate!(x, out) = mul!(out, BO_A, x, 1.0, 0.5)
bo_gemm_accumulate!(x, out) = mul!(out, BO_A, x, 1.0, 0.5)
bo_gemv_add!(x, out) = mul!(out, BO_A, x, 1.0, 1.0)

@testset "Reverse: beta != 0 still scales the output's adjoint" begin
    w = collect(1.0:4.0)
    dout = zeros(4)
    dx = bo_gradient(Reverse, bo_gemv_accumulate!, BO_VEC, ones(4), dout, w)
    @test dx ≈ BO_A' * w
    @test dout == 0.5 .* w
    dout = zeros(4)
    bo_gradient(Reverse, bo_gemv_add!, BO_VEC, ones(4), dout, w)
    @test dout == w
    W = collect(1.0:8.0)
    dC = zeros(4, 2)
    dX = bo_gradient(Reverse, bo_gemm_accumulate!, BO_MAT, ones(4, 2), dC, W)
    @test dX ≈ BO_A' * reshape(W, 4, 2)
    @test dC == 0.5 .* reshape(W, 4, 2)
end

# syrk writes one triangle of C. The other triangle keeps its values, so its
# adjoint is the adjoint of an input and must be left alone.
bo_syrk_upper!(x, out) = BLAS.syrk!('U', 'T', 1.0, x, 0.0, out)
bo_syrk_lower!(x, out) = BLAS.syrk!('L', 'T', 1.0, x, 0.0, out)
bo_syrk_upper_accumulate!(x, out) = BLAS.syrk!('U', 'T', 1.0, x, 0.5, out)

@testset "Reverse: syrk clears only the triangle it writes" begin
    W = [1.0 3.0; 2.0 4.0]
    @testset "$uplo" for (uplo, op) in (('U', bo_syrk_upper!), ('L', bo_syrk_lower!))
        written = uplo == 'U' ? triu(W) : tril(W)
        for bad in (1.0, NaN)
            w = copy(W)
            w[1, 1] = bad
            dC = zeros(2, 2)
            bo_gradient(Reverse, op, BO_MAT, zeros(2, 2), dC, vec(w))
            @test dC == W .- written
        end
        dC = zeros(2, 2)
        dX = bo_gradient(Reverse, op, BO_MAT, zeros(2, 2), dC, vec(W))
        # d/dX of sum(written .* (X' * X))
        @test dX ≈ BO_MAT * (written + written')
    end
    # With beta != 0 the adjoint of the written triangle is scaled, and only it.
    dC = zeros(2, 2)
    bo_gradient(Reverse, bo_syrk_upper_accumulate!, BO_MAT, ones(2, 2), dC, vec(W))
    @test dC == W .- 0.5 .* triu(W)
end

# lacpy copies a triangle, or all, of A into B. LinearAlgebra wraps it from
# Julia 1.11 on; before that this file calls it itself.
@static if isdefined(LinearAlgebra.LAPACK, :lacpy!)
    bo_lacpy!(B::Matrix{Float64}, A::Matrix{Float64}, uplo::Char) = LinearAlgebra.LAPACK.lacpy!(B, A, uplo)
else
    function bo_lacpy!(B::Matrix{Float64}, A::Matrix{Float64}, uplo::Char)
        m, n = size(A)
        ccall(
            (BLAS.@blasfunc(dlacpy_), BLAS.libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ptr{Float64}, Ref{BLAS.BlasInt}, Ptr{Float64}, Ref{BLAS.BlasInt}, Clong),
            uplo, m, n, A, max(1, stride(A, 2)), B, max(1, stride(B, 2)), 1,
        )
        return B
    end
end
bo_lacpy_const_source!(x, out, uplo) = (bo_lacpy!(out, BO_A, uplo); out[1, 1] += sum(x); out)
bo_lacpy_active_source!(x, out, uplo) = bo_lacpy!(out, x, uplo)

@testset "Reverse: lacpy clears the adjoint of what it overwrites" begin
    W = reshape(collect(1.0:16.0), 4, 4)
    @testset "uplo = $uplo" for uplo in ('U', 'L', 'A')
        written = uplo == 'U' ? triu(W) : uplo == 'L' ? tril(W) : W
        @testset "seed $bad" for bad in (1.0, NaN)
            w = copy(W)
            w[2, 2] = bad
            # The source is constant: nothing but the clearing is left to do.
            dB = zeros(4, 4)
            dx = bo_gradient(Reverse, (x, out) -> bo_lacpy_const_source!(x, out, uplo), BO_VEC, zeros(4, 4), dB, vec(w))
            @test dx == fill(W[1, 1], 4)
            @test dB == W .- written
            # The source is active: its adjoint takes what the destination's held.
            dB = zeros(4, 4)
            X = BO_A .+ 1.0
            dX = bo_gradient(Reverse, (x, out) -> bo_lacpy_active_source!(x, out, uplo), X, zeros(4, 4), dB, vec(W))
            @test dX == written
            @test dB == W .- written
        end
    end
end

# A strided copy writes every second entry of the destination. The entries in
# between keep their values, and with them the adjoints of inputs.
bo_copy_strided!(x, out) = BLAS.blascopy!(length(x), x, 1, out, 2)
bo_copy_reversed!(x, out) = BLAS.blascopy!(length(x), x, 1, out, -1)

function bo_two_copies(x, w, y)
    BLAS.blascopy!(length(x), x, 1, y, 1)
    a = 0.0
    @inbounds for i in eachindex(y)
        a += y[i]
    end
    BLAS.blascopy!(length(w), w, 1, y, 1)
    b = 0.0
    @inbounds for i in eachindex(y)
        b += 2 * y[i]
    end
    return a + b
end

function bo_copy_const_source(c, y, v)
    BLAS.blascopy!(length(c), c, 1, y, 1)
    return bo_weighted(y, v)
end

@testset "Reverse: copy" begin
    @testset "strided destination" begin
        w = collect(1.0:8.0)
        dout = zeros(8)
        dx = bo_gradient(Reverse, bo_copy_strided!, BO_VEC, zeros(8), dout, w)
        @test dx == w[1:2:7]
        @test dout == [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]
    end
    @testset "negative increment" begin
        w = collect(1.0:4.0)
        dout = zeros(4)
        dx = bo_gradient(Reverse, bo_copy_reversed!, BO_VEC, zeros(4), dout, w)
        @test dx == reverse(w)
        @test all(iszero, dout)
    end
    @testset "two copies into one buffer, one reverse pass" begin
        x = [1.0, 2.0, 3.0]
        w = [4.0, 5.0, 6.0]
        dx = zero(x)
        dw = zero(w)
        dy = zeros(3)
        autodiff(Reverse, bo_two_copies, Active, Duplicated(x, dx), Duplicated(w, dw), Duplicated(zeros(3), dy))
        @test dx == ones(3)
        @test dw == 2 .* ones(3)
        @test all(iszero, dy)
    end
    @testset "constant source" begin
        c = [1.0, 2.0, 3.0]
        dv = zeros(3)
        dy = zeros(3)
        for _ in 1:2
            autodiff(Reverse, bo_copy_const_source, Active, Const(c), Duplicated(zeros(3), dy), Duplicated([0.5, 1.5, 2.5], dv))
            @test all(iszero, dy)
        end
        @test dv == 2 .* c
    end
end

@testset "Reverse: batches and split mode" begin
    w = collect(1.0:4.0)
    @testset "a batch of two" begin
        dx = (zeros(4), zeros(4))
        dout = (fill(NaN, 4), [1.0, Inf, 2.0, -Inf])
        autodiff(Reverse, bo_loss, BatchDuplicated(copy(BO_VEC), dx), BatchDuplicated(zeros(4), dout), Const(w), Const(bo_gemv!))
        @test all(iszero, dout[1])
        @test all(iszero, dout[2])
    end
    @testset "split mode" begin
        forward, reverse = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(bo_loss)}, Active,
            Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}, Const{Vector{Float64}}, Const{typeof(bo_gemv!)},
        )
        x = copy(BO_VEC)
        dx = zero(x)
        out = zeros(4)
        dout = zeros(4)
        poisoned = [1.0, NaN, 1.0, 1.0]
        for weights in (poisoned, w)
            fill!(dx, 0.0)
            tape, value, _ = forward(Const(bo_loss), Duplicated(x, dx), Duplicated(out, dout), Const(weights), Const(bo_gemv!))
            reverse(Const(bo_loss), Duplicated(x, dx), Duplicated(out, dout), Const(weights), Const(bo_gemv!), 1.0, tape)
            @test all(iszero, dout)
        end
        @test dx ≈ BO_A' * w
    end
end

# Forward mode: a copy from a constant source writes a zero tangent, whatever
# the destination's tangent held (a fresh forward shadow is not initialized).
function bo_forward_copy!(y, x)
    BLAS.blascopy!(length(x), x, 1, y, 1)
    return nothing
end

@testset "Forward: copy from a constant source" begin
    for bad in (NaN, Inf)
        dy = fill(bad, 4)
        autodiff(Forward, bo_forward_copy!, Const, Duplicated(zeros(4), dy), Const(copy(BO_VEC)))
        @test all(iszero, dy)
    end
    dy = fill(NaN, 4)
    dx = collect(1.0:4.0)
    autodiff(Forward, bo_forward_copy!, Const, Duplicated(zeros(4), dy), Duplicated(copy(BO_VEC), dx))
    @test dy == dx
end

# Forward over Reverse: the outer pass differentiates the clearing of the
# output's shadow together with the rest of the reverse pass.
function bo_quadratic(x, out)
    mul!(out, reshape(view(x, 1:16), 4, 4), view(x, 17:20))
    return sum(abs2, out) / 2
end

function bo_matrix_quadratic(x, out)
    mul!(out, reshape(view(x, 1:16), 4, 4), reshape(view(x, 17:24), 4, 2))
    return sum(abs2, out) / 2
end

function bo_gradient!(g, f, x, out, dout)
    fill!(g, 0.0)
    autodiff(Reverse, Const(f), Active, Duplicated(x, g), Duplicated(out, dout))
    return nothing
end

function bo_hvp(f, x, v, out0)
    hv = zero(x)
    autodiff(
        Forward, Const(bo_gradient!), Const, Duplicated(zero(x), hv), Const(f), Duplicated(copy(x), copy(v)),
        Duplicated(copy(out0), zero(out0)), Duplicated(zero(out0), zero(out0)),
    )
    return hv
end

function bo_hvp_central(f, x, v, out0; h = 1.0e-6)
    gp = zero(x)
    gm = zero(x)
    bo_gradient!(gp, f, x .+ h .* v, copy(out0), zero(out0))
    bo_gradient!(gm, f, x .- h .* v, copy(out0), zero(out0))
    return (gp .- gm) ./ (2h)
end

@testset "Forward over Reverse through an overwriting product" begin
    @testset "$name" for (name, f, n, out0) in (
            ("gemv", bo_quadratic, 20, zeros(4)),
            ("gemm", bo_matrix_quadratic, 24, zeros(4, 2)),
        )
        x = [0.1 + 0.07i for i in 1:n]
        v = [cos(i) for i in 1:n]
        @test bo_hvp(f, x, v, out0) ≈ bo_hvp_central(f, x, v, out0) rtol = 1.0e-6
    end
end
