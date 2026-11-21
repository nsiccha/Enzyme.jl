using Enzyme, Test

module CcallLibraries
    const warm_calls = Ref(0)
    const cold_calls = Ref(0)
    function warm_name()
        warm_calls[] += 1
        return Base.libm_name
    end
    function cold_name()
        cold_calls[] += 1
        return Base.libm_name
    end
end

@static if VERSION < v"1.13-"
    cold_cos_computed(x) = ccall((:cos, CcallLibraries.cold_name()), Float64, (Float64,), x)
    warm_cos_computed(x) = ccall((:cos, CcallLibraries.warm_name()), Float64, (Float64,), x)
    reverse_derivative(f, x) = only(only(autodiff(Reverse, f, Active, Active(x))))
end

@testset "Computed library address" begin
    x = 0.3
    @static if VERSION < v"1.13-"
        @testset "Imported derivatives share a computed library lookup" begin
            @test CcallLibraries.warm_calls[] == CcallLibraries.cold_calls[] == 0
            @test reverse_derivative(warm_cos_computed, x) ≈ -sin(x)
            @test CcallLibraries.warm_calls[] == 1
            for _ in 1:2
                @test only(autodiff(Forward, reverse_derivative, Const(warm_cos_computed), Duplicated(x, 1.0))) ≈ -cos(x)
                @test CcallLibraries.warm_calls[] == 1
            end

            # The outer derivative can also be the first user of the call site.
            @test only(autodiff(Forward, reverse_derivative, Const(cold_cos_computed), Duplicated(x, 1.0))) ≈ -cos(x)
            @test CcallLibraries.cold_calls[] == 1
            @test reverse_derivative(cold_cos_computed, x) ≈ -sin(x)
            @test CcallLibraries.cold_calls[] == 1
        end
    end
end
