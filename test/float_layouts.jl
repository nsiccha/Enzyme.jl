using Enzyme
using Test

const FloatLayoutLLVM = Enzyme.LLVM

struct StaticLayoutFloat{V} <: AbstractFloat end

struct SingletonFieldFloat <: AbstractFloat
    value::Nothing
end

struct NonemptyLayoutFloat <: AbstractFloat
    hi::Float64
    lo::Float64
end

primitive type UnknownLayoutFloat <: AbstractFloat 64 end

@noinline float_layout_pair(x) = (StaticLayoutFloat{2.0}(), x[1]^2)
float_layout_pair_objective(x) = float_layout_pair(x)[2]
@noinline float_layout_record(x) = (value = x[1]^2, singleton = SingletonFieldFloat(nothing))
float_layout_record_objective(x) = float_layout_record(x).value

@testset "Singleton floating-point layouts" begin
    for T in (StaticLayoutFloat{0.0}, StaticLayoutFloat{1.0f0}, SingletonFieldFloat)
        @test Base.issingletontype(T)
        @test isempty(Enzyme.get_offsets(T))
        @test Enzyme.get_offsets(Tuple{T, Float64}) == [(Enzyme.API.DT_Double, 0)]
        FloatLayoutLLVM.Context() do ctx
            dl = string(FloatLayoutLLVM.DataLayout(FloatLayoutLLVM.JITTargetMachine()))
            @test string(Enzyme.typetree(T, ctx, dl)) == "{}"
            @test string(Enzyme.typetree(Tuple{T, Float64}, ctx, dl)) == "{[-1]:Float@double}"
        end
    end

    # Nonempty float formats still need explicit layouts and arithmetic rules.
    for T in (UnknownLayoutFloat, NonemptyLayoutFloat)
        @test_throws AssertionError Enzyme.get_offsets(T)
        FloatLayoutLLVM.Context() do ctx
            dl = string(FloatLayoutLLVM.DataLayout(FloatLayoutLLVM.JITTargetMachine()))
            @test_throws AssertionError Enzyme.typetree(T, ctx, dl)
        end
    end

    for T in (Float32, Float64), f in (float_layout_pair_objective, float_layout_record_objective)
        x = T(0.3)
        values = [x]
        gradient = zero(values)
        _, primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(values, gradient))
        @test primal ≈ x^2
        @test gradient ≈ [2x]
        @test only(autodiff(Forward, f, Duplicated(values, ones(T, 1)))) ≈ 2x
    end
end
