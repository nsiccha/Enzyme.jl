using Enzyme, LinearAlgebra, Test

@noinline invoke_select(op::Function, x) = op === identity ? x * x : x
invoke_square(x) = invoke_select(identity, x)

@static if VERSION >= v"1.12-"
    @testset "A non-inlined call keeps its inferred function argument" begin
        report = Enzyme.derivative_report() do
            autodiff(Reverse, invoke_square, Active, Active(3.0))
        end
        @test report.value == ((6.0,),)
        @test all(f -> f.throw_only || f.kind !== :enzyme_runtime_fallback, only(report.derivatives).findings)
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
