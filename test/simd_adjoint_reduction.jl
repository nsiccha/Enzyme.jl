using Enzyme, Test

function simd_regression(u, X, y)
    b1, b2, s = u
    ss = 0.0
    @inbounds @simd for i in eachindex(y)
        r = y[i] - (X[i, 1] * b1 + X[i, 2] * b2)
        ss += r * r
    end
    return -length(y) * s - 0.5 * exp(-2s) * ss + s
end

function simd_regression_reference(u, X, y)
    b1, b2, s = u
    ss = a1 = a2 = 0.0
    for i in eachindex(y)
        r = y[i] - (X[i, 1] * b1 + X[i, 2] * b2)
        ss += r * r
        a1 += r * X[i, 1]
        a2 += r * X[i, 2]
    end
    w = exp(-2s)
    return [w * a1, w * a2, -length(y) + w * ss + 1]
end

function simd_scalar_residual(x, data)
    total = 0.0
    @inbounds @simd for i in eachindex(data)
        r = data[i] - x
        total += r * r
    end
    return total
end

function strict_scalar_residual(x, data)
    total = 0.0
    @inbounds for i in eachindex(data)
        r = data[i] - x
        total += r * r
    end
    return total
end

function simd_scalar_gradient(x, data)
    return autodiff(Reverse, simd_scalar_residual, Active, Active(x), Const(data))[1][1]
end

function simd_tiny_sum(t, data)
    total = 0.0
    @inbounds @simd for i in eachindex(data)
        total += exp(log(t * data[i]))
    end
    return total
end

function simd_tiny_subtract(t, data)
    total = 0.0
    @inbounds @simd for i in eachindex(data)
        total -= exp(log(t * data[i]))
    end
    return total
end

function simd_tiny_negative_seed(t, data)
    return -simd_tiny_sum(t, data)
end

function simd_adjoint_ir(f, types; second_stage = false, returnPrimal = false, AtomicAdd = false)
    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, types; second_stage, returnPrimal, AtomicAdd,
    )
    return String(take!(io))
end

@testset "Independent adjoint reductions inherit SIMD permission" begin
    for n in (0, 1, 2, 3, 7, 8, 9, 17, 1192)
        X = [sin(i + j) for i in 1:n, j in 1:2]
        y = [cos(i / 3) for i in 1:n]
        u = [0.3, -0.7, 0.2]
        inputs = (copy(u), copy(X), copy(y))
        expected = simd_regression_reference(u, X, y)
        for mode in (ReverseWithPrimal, set_atomic_add(ReverseWithPrimal, true))
            # A nonzero incoming shadow must also participate correctly.
            du = fill(0.25, 3)
            result = autodiff(mode, simd_regression, Active, Duplicated(u, du), Const(X), Const(y))
            @test du ≈ expected .+ 0.25 rtol = 2.0e-12 atol = 2.0e-12
            @test result[2] ≈ simd_regression(u, X, y)
        end
        fw, rv = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(simd_regression)}, Active,
            Duplicated{Vector{Float64}}, Const{Matrix{Float64}}, Const{Vector{Float64}},
        )
        du = zeros(3)
        args = (Duplicated(u, du), Const(X), Const(y))
        tape, primal, _ = fw(Const(simd_regression), args...)
        rv(Const(simd_regression), args..., 1.0, tape)
        @test du ≈ expected rtol = 2.0e-12 atol = 2.0e-12
        @test primal ≈ simd_regression(u, X, y)
        @test (u, X, y) == inputs

        x = 0.3
        expected_scalar = 2 * sum(x .- y)
        @test simd_scalar_gradient(x, y) ≈ expected_scalar rtol = 2.0e-12 atol = 2.0e-12
        @test autodiff(Forward, simd_scalar_gradient, Duplicated(x, 1.0), Const(y))[1] ≈ 2n
        @test autodiff(Reverse, strict_scalar_residual, Active, Active(x), Const(y))[1][1] ≈ expected_scalar rtol = 2.0e-12 atol = 2.0e-12
    end

    types = Tuple{Active{Float64}, Const{Vector{Float64}}}
    annotated = simd_adjoint_ir(simd_scalar_residual, types)
    strict = simd_adjoint_ir(strict_scalar_residual, types)
    atomic = simd_adjoint_ir(simd_scalar_residual, types; AtomicAdd = true)
    # nsz comes from the derivative policy; the primal SIMD sum has only
    # reassoc+contract. Thus this checks the adjoint, not just the primal loop.
    @test occursin(r"f(?:add|sub) reassoc nsz contract double", annotated)
    @test !occursin(r"f(?:add|sub) reassoc nsz contract double", strict)
    @test !occursin(r"f(?:add|sub) reassoc nsz contract double", atomic)
    @test !occursin(r"f(?:mul|div) reassoc", annotated)

    row_types = Tuple{Duplicated{Vector{Float64}}, Const{Matrix{Float64}}, Const{Vector{Float64}}}
    rows_before = simd_adjoint_ir(simd_regression, row_types; returnPrimal = true)
    rows_after = simd_adjoint_ir(simd_regression, row_types; second_stage = true, returnPrimal = true)
    @test occursin(r"f(?:add|sub) reassoc nsz contract double", rows_before)
    @test !occursin(r"f(?:mul|div) reassoc", rows_before)
    # nsz distinguishes the vectorized reverse reductions from the primal sum.
    @test occursin(r"call reassoc nsz contract double @llvm.vector.reduce.fadd", rows_after)
end

@testset "SIMD preserves tiny per-iteration gradients" begin
    for n in (1, 33, 1192), t in (0.5, 1.0, 2.0)
        data = fill(1.0e-190, n)
        for (f, sign) in ((simd_tiny_sum, 1), (simd_tiny_subtract, -1), (simd_tiny_negative_seed, -1))
            actual = autodiff(Reverse, f, Active, Active(t), Const(data))[1][1]
            @test actual ≈ sign * n * 1.0e-190 rtol = 2.0e-12 atol = 0
        end
    end
end
