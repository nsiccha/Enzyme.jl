using Enzyme, LinearAlgebra, Test

# Forward over Reverse (a Hessian-vector product) through matrix products that
# LinearAlgebra sends to BLAS. The reverse rules of gemm, symm and syrk scale the
# output's shadow by beta with lascl, so the outer Forward pass differentiates a
# lascl call. Julia's `A * transpose(B)` also keeps a runtime `A === B` branch to
# syrk, whose reverse calls lascl too; before lascl had a Forward rule, the gemm
# call site then reached a stub that raised "The original primal code hits this
# error condition" instead of naming the missing derivative.

function blas_grad!(g, f, x)
    fill!(g, 0.0)
    autodiff(Reverse, Const(f), Active, Duplicated(x, g))
    return nothing
end

function blas_hvp_forward_over_reverse(f, x, v)
    g = zero(x)
    hv = zero(x)
    autodiff(Forward, Const(blas_grad!), Const, Duplicated(g, hv), Const(f), Duplicated(x, copy(v)))
    return hv
end

blas_dir(f, x, v) = only(autodiff(Forward, Const(f), Duplicated(x, v)))

function blas_hvp_reverse_over_forward(f, x, v)
    hv = zero(x)
    autodiff(Reverse, Const(blas_dir), Active, Const(f), Duplicated(x, hv), Const(v))
    return hv
end

function blas_hvp_central(f, x, v; h = 1.0e-6)
    gp = zero(x)
    gm = zero(x)
    blas_grad!(gp, f, x .+ h .* v)
    blas_grad!(gm, f, x .- h .* v)
    return (gp .- gm) ./ (2h)
end

const BLAS_CONST_LEFT = [1.0 2 3; 4 5 6; 7 8 9.5; 1.5 0.5 2.5]
const BLAS_CONST_RIGHT = [1.0 2 3; 4 5 6]

# The output of `mul!(C, A, B, 1.0, 0.5)` starts from active data, so the reverse
# pass scales C's shadow by beta = 0.5, not 0.
function blas_scaled_accumulate(x)
    C = reshape(x[1:6], 3, 2) .+ 0.0
    mul!(C, reshape(x[7:15], 3, 3), reshape(x[16:21], 3, 2), 1.0, 0.5)
    return sum(abs2, C)
end

function blas_active_beta(x)
    C = reshape(x[1:6], 3, 2) .+ 0.0
    mul!(C, reshape(x[7:18], 3, 4), reshape(x[19:26], 4, 2), 1.0, x[27])
    return sum(abs2, C)
end

const BLAS_FOR_CASES = (
    # The reported shapes: gemm('N','T') with a one-row left factor, and gemm('N','N').
    "gemm N,T 1x3 * transpose(3x3)" =>
        (x -> sum(abs2, reshape(x[1:3], 1, 3) * transpose(reshape(x[4:12], 3, 3))), 12),
    "gemm N,N 4x3 * 3x3" => (x -> sum(abs2, reshape(x[1:12], 4, 3) * reshape(x[13:21], 3, 3)), 21),
    "gemm N,N 4x3 * 3x2" => (x -> sum(abs2, reshape(x[1:12], 4, 3) * reshape(x[13:18], 3, 2)), 18),
    "gemm N,T 4x3 * transpose(2x3)" =>
        (x -> sum(abs2, reshape(x[1:12], 4, 3) * transpose(reshape(x[13:18], 2, 3))), 18),
    "gemm T,N transpose(3x4) * 3x2" =>
        (x -> sum(abs2, transpose(reshape(x[1:12], 3, 4)) * reshape(x[13:18], 3, 2)), 18),
    "gemm with beta = 0.5" => (blas_scaled_accumulate, 21),
    # `X' * X` and `X * X'` take the syrk branch, whose reverse also calls lascl.
    "syrk X' * X" => (x -> sum(abs2, reshape(x[1:12], 4, 3)' * reshape(x[1:12], 4, 3)), 12),
    "syrk X * X'" => (x -> (X = reshape(x[1:12], 3, 4); sum(abs2, X * X')), 12),
    # gemv's reverse scales with scal; it passed before and must keep passing.
    "gemv N 4x3 * 3" => (x -> sum(abs2, reshape(x[1:12], 4, 3) * x[13:15]), 15),
)

# Products for which Julia keeps a run-time `A === B` branch to syrk. On Julia 1.12
# and 1.13 Reverse over Forward through that branch raises an EnzymeInternalError
# (Enzyme todo `0j5bnf4`, also before lascl had rules), so the reference comparison
# is pinned there. Forward over Reverse itself passes.
const BLAS_SYRK_BRANCH = (
    "gemm N,T 1x3 * transpose(3x3)", "gemm N,T 4x3 * transpose(2x3)",
    "gemm T,N transpose(3x4) * 3x2", "syrk X' * X", "syrk X * X'",
)

@testset "Forward over Reverse through BLAS products" begin
    @testset "$name" for (name, (f, n)) in BLAS_FOR_CASES
        x = [0.1 + 0.07i for i in 1:n]
        v = [cos(i) for i in 1:n]
        hv = blas_hvp_forward_over_reverse(f, x, v)
        @test isapprox(hv, blas_hvp_central(f, x, v); rtol = 1.0e-6, atol = 1.0e-6)
        if VERSION >= v"1.12" && name in BLAS_SYRK_BRANCH
            @test_broken isapprox(hv, blas_hvp_reverse_over_forward(f, x, v); rtol = 1.0e-10, atol = 1.0e-10)
        else
            @test isapprox(hv, blas_hvp_reverse_over_forward(f, x, v); rtol = 1.0e-10, atol = 1.0e-10)
        end
        # The inputs are left alone.
        @test x == [0.1 + 0.07i for i in 1:n]
    end

    @testset "a batch of two directions" begin
        f = last(BLAS_FOR_CASES[1])[1]
        n = 12
        x = [0.1 + 0.07i for i in 1:n]
        v1 = [cos(i) for i in 1:n]
        v2 = [sin(2i) for i in 1:n]
        g = zero(x)
        h1 = zero(x)
        h2 = zero(x)
        autodiff(
            Forward, Const(blas_grad!), Const, BatchDuplicated(g, (h1, h2)), Const(f),
            BatchDuplicated(x, (copy(v1), copy(v2))),
        )
        @test isapprox(h1, blas_hvp_forward_over_reverse(f, x, v1); rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(h2, blas_hvp_forward_over_reverse(f, x, v2); rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(h1, blas_hvp_central(f, x, v1); rtol = 1.0e-6, atol = 1.0e-6)
    end

    @testset "symm" begin
        f(x) = sum(abs2, Symmetric(reshape(x[1:9], 3, 3)) * reshape(x[10:15], 3, 2))
        n = 15
        x = [0.1 + 0.07i for i in 1:n]
        v = [cos(i) for i in 1:n]
        hv = blas_hvp_forward_over_reverse(f, x, v)
        @test isapprox(hv, blas_hvp_central(f, x, v); rtol = 1.0e-6, atol = 1.0e-6)
    end

    # A constant operand: the inner reverse keeps the constant matrix's data pointer
    # on its tape, which is active in the outer Forward pass, and the mixed-activity
    # handler has no shadow for a bare `Ptr{Float64}` (EnzymeRuntimeActivityError).
    # Reverse over Forward differentiates these. Tracking: Enzyme todo `0mcxlsc`.
    @testset "constant operand (not yet)" begin
        for (f, n) in (
                (x -> sum(abs2, BLAS_CONST_LEFT * reshape(x[1:6], 3, 2)), 6),
                (x -> sum(abs2, reshape(x[1:8], 4, 2) * BLAS_CONST_RIGHT), 8),
            )
            x = [0.1 + 0.07i for i in 1:n]
            v = [cos(i) for i in 1:n]
            rof = blas_hvp_reverse_over_forward(f, x, v)
            @test isapprox(rof, blas_hvp_central(f, x, v); rtol = 1.0e-6, atol = 1.0e-6)
            @test_broken isapprox(blas_hvp_forward_over_reverse(f, x, v), rof; rtol = 1.0e-10, atol = 1.0e-10)
        end
    end

    # The scale factor itself carries a derivative: the rule refuses rather than
    # dropping that part of the Hessian.
    @testset "active beta is refused" begin
        n = 27
        x = [0.1 + 0.07i for i in 1:n]
        v = [cos(i) for i in 1:n]
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError blas_hvp_forward_over_reverse(blas_active_beta, x, v)
    end
end
