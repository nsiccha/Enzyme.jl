using Enzyme
using Enzyme: EnzymeRules
using LinearAlgebra
using Test

# Enzyme's rule for `A \ b` solves with the structure `\` picks for the matrix: a diagonal,
# lower or upper triangular solve, an LU factorization, the matrix itself for a real
# rectangular system, or a pivoted QR factorization. It keeps that factorization in fields
# of concrete types, so neither its solve nor its tape is dispatched at run time.

dsd_sum(A, b) = sum(A \ b)
dsd_view_sum(A, x, r) = sum(A \ view(x, r))

dsd_unstable(report) = [
    x for d in report.derivatives for x in something(d.findings, [])
        if !x.throw_only && x.kind in (:dynamic_dispatch, :enzyme_runtime_fallback, :box)
]

# A finding whose innermost frame is in Enzyme's linear algebra rules.
dsd_in_rule(x) = !isempty(x.frames) && endswith(string(first(x.frames).file), joinpath("internal_rules", "linalg.jl"))

function dsd_check_stable(report)
    @test only(report.derivatives).findings !== nothing
    @test isempty(filter(dsd_in_rule, dsd_unstable(report)))
    # Julia 1.10 compiles a call with a widened `Function` argument, which LinearAlgebra's
    # pivoted QR solve makes (`generic_trimatdiv!`), as `jl_invoke`. Repair
    # `specialized_invokes` covers Julia 1.12 and later, and `specialized_invokes_1_10`
    # Julia 1.10 (Enzyme snag `julia-1-10-widen-e9392d77`).
    if VERSION >= v"1.12-" || Enzyme.has_dogfood_repair(:specialized_invokes_1_10)
        @test isempty(dsd_unstable(report))
    else
        @test_broken isempty(dsd_unstable(report))
    end
    return
end

# The calls Julia dispatches at run time in the optimized code of `f` for argument types
# `types`: a `:call` of anything but a builtin or an intrinsic. `derivative_report` does
# not see into the reverse rule, which a derivative calls as separately compiled code.
function dsd_dynamic_calls(f, types)
    calls = String[]
    for (ci, _) in code_typed(f, types; optimize = true)
        for st in ci.code
            Meta.isexpr(st, :call) || continue
            callee = st.args[1]
            if callee isa GlobalRef && isdefined(callee.mod, callee.name)
                callee = getglobal(callee.mod, callee.name)
            end
            callee isa Core.Builtin && continue
            push!(calls, string(st))
        end
    end
    return calls
end

function dsd_gradient(A, b)
    dA = zero(A)
    db = zero(b)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, dsd_sum, Active, Duplicated(A, dA), Duplicated(b, db))
    end
    return report, dA, db
end

# A finite difference of `sum(A \ b)` in each entry of A.
function dsd_fd(A, b)
    return map(CartesianIndices(A)) do i
        step = zero(A)
        step[i] = 1.0e-6
        (dsd_sum(A + step, b) - dsd_sum(A - step, b)) / 2.0e-6
    end
end

@testset "Dense solve rule: concretely typed factorization" begin
    b = [0.35, -0.2]
    @testset "$name" for (name, A) in (
            ("diagonal", [2.0 0.0; 0.0 1.5]),
            ("lower triangular", [2.0 0.0; 0.3 1.5]),
            ("upper triangular", [2.0 -0.4; 0.0 1.5]),
            ("LU", [2.0 -0.4; 0.3 1.5]),
        )
        report, dA, db = dsd_gradient(A, b)
        y = A \ b
        z = transpose(A) \ ones(2)
        @test db ≈ z
        @test dA ≈ -z * transpose(y)
        dsd_check_stable(report)
    end

    @testset "real rectangular, $shape" for (shape, A, b) in (
            ("tall", [2.0 -0.4; 0.3 1.5; 0.7 -0.1], [0.35, -0.2, 0.9]),
            ("wide", [2.0 -0.4 0.7; 0.3 1.5 -0.1], [0.35, -0.2]),
        )
        report, dA, db = dsd_gradient(A, b)
        @test db ≈ transpose(A) \ ones(size(A, 2))
        @test dA ≈ dsd_fd(A, b) atol = 1.0e-7
        dsd_check_stable(report)
    end

    @testset "strided view right-hand side" begin
        A = [2.0 -0.4; 0.3 1.5]
        x = [4.0, 0.35, -0.2, 5.0]
        dx = zero(x)
        report = Enzyme.derivative_report() do
            autodiff(Reverse, dsd_view_sum, Active, Const(A), Duplicated(x, dx), Const(2:3))
        end
        @test dx ≈ [0.0; transpose(A) \ ones(2); 0.0]
        dsd_check_stable(report)
    end

    # A matrix and a right-hand side of different element types take pivoted QR.
    @testset "pivoted QR" begin
        A = Float32[2 -0.4; 0.3 1.5; 0.7 -0.1]
        b = [0.35, -0.2, 0.9]
        db = zero(b)
        autodiff(Reverse, dsd_sum, Active, Const(A), Duplicated(b, db))
        @test db ≈ transpose(Float64.(A)) \ ones(2) rtol = 1.0e-5
    end

    @testset "the tape and the reverse rule are concretely typed" begin
        config = EnzymeRules.RevConfig{true, true, 1, (false, false, false), false, false}()
        RT = Duplicated{Vector{Float64}}
        b = [0.35, -0.2]
        for A in ([2.0 0.0; 0.0 1.5], [2.0 -0.4; 0.3 1.5], [2.0 -0.4; 0.3 1.5; 0.7 -0.1])
            bb = size(A, 1) == 2 ? b : [b; 0.9]
            aug = EnzymeRules.augmented_primal(config, Const(\), RT, Const(A), Duplicated(bb, zero(bb)))
            TT = typeof(aug.tape)
            @test all(isconcretetype, fieldtypes(TT))
            types = (typeof(config), Const{typeof(\)}, Type{RT}, TT, Const{Matrix{Float64}}, Duplicated{Vector{Float64}})
            @test isempty(dsd_dynamic_calls(EnzymeRules.reverse, types))
        end
        types = (typeof(config), Type{RT}, Const{Matrix{Float64}}, Duplicated{Vector{Float64}})
        @test isempty(dsd_dynamic_calls(Enzyme._augmented_dense_solve, types))
    end

    @testset "batch of two" begin
        A = [2.0 -0.4; 0.3 1.5]
        b = [0.35, -0.2]
        dA = (zero(A), zero(A))
        db = (zero(b), zero(b))
        autodiff(Reverse, dsd_sum, Active, BatchDuplicated(A, dA), BatchDuplicated(b, db))
        z = transpose(A) \ ones(2)
        y = A \ b
        for i in 1:2
            @test db[i] ≈ z
            @test dA[i] ≈ -z * transpose(y)
        end
    end
end
