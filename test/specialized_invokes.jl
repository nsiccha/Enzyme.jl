using Enzyme, LinearAlgebra, Test
import Enzyme.EnzymeRules

@noinline invoke_select(op::Function, x) = op === identity ? x * x : x
invoke_square(x) = invoke_select(identity, x)

# Julia 1.11 has no toolchain on the hosts that test the dogfood line, so the
# repair is not applied (nor tested) there.
const SPECIALIZED_INVOKES = VERSION >= v"1.12-" || VERSION < v"1.11-"

returning_findings(report) = [
    finding for d in report.derivatives for finding in something(d.findings, [])
        if !finding.throw_only && finding.kind in (:dynamic_dispatch, :enzyme_runtime_fallback)
]

@static if SPECIALIZED_INVOKES
    @testset "A non-inlined call keeps its inferred function argument" begin
        report = Enzyme.derivative_report() do
            autodiff(Reverse, invoke_square, Active, Active(3.0))
        end
        @test report.value == ((6.0,),)
        @test all(f -> f.throw_only || f.kind !== :enzyme_runtime_fallback, only(report.derivatives).findings)
        @test isempty(returning_findings(report))
    end

    # The primal of a custom rule is compiled with the derivative, so its calls
    # that do not inline keep their inferred function argument too. LinearAlgebra's
    # `generic_trimatdiv!(C, uplo, isunit, tfun::Function, A, B)`, which the
    # pivoted QR solve calls, is one.
    rule_square(x) = x * x
    function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(rule_square)},
            ::Type{<:Active}, x::Active,
        )
        primal = EnzymeRules.needs_primal(config) ? invoke_select(identity, x.val) : nothing
        return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
    end
    function EnzymeRules.reverse(
            config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(rule_square)},
            dret::Active, tape, x::Active,
        )
        return (2 * x.val * dret.val,)
    end

    @testset "A custom rule's primal keeps an inferred function argument" begin
        report = Enzyme.derivative_report() do
            autodiff(ReverseWithPrimal, rule_square, Active, Active(3.0))
        end
        @test report.value == ((6.0,), 9.0)
        @test any(d -> d.func === typeof(rule_square), report.derivatives)
        @test isempty(returning_findings(report))
    end
end

triangular_product_sum(L, x) = sum(L * x)
triangular_product_sum!(L, x, dest) = sum(mul!(dest, L, x))

@testset "Reverse of a triangular product with a vector view" begin
    x = view(collect(1.0:6.0), 2:5)
    A = reshape(collect(1.0:16.0), 4, 4)
    for L in (LowerTriangular(A), LowerTriangular(A'))
        for (f, outputs) in (
                (triangular_product_sum, ()),
                (triangular_product_sum!, (Duplicated(similar(x), zero(x)),)),
            )
            dL, dx = Enzyme.make_zero(L), Enzyme.make_zero(x)
            report = Enzyme.derivative_report() do
                autodiff(Reverse, f, Active, Duplicated(L, dL), Duplicated(x, dx), outputs...)
            end
            @test Matrix(dL) ≈ tril(ones(length(x)) * x')
            @test dx ≈ L' * ones(length(x))
            @test any(d -> d.func === typeof(f), report.derivatives)
            @test all(
                finding.throw_only || finding.kind !== :enzyme_runtime_fallback for
                    d in report.derivatives for finding in something(d.findings)
            )
        end
    end
end
