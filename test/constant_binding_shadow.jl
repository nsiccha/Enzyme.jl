using Enzyme, Test

module ConstantStringProvider
    name() = Base.libm_name
    const selected = Ref{Any}(name)
end

by_length(x) = length(ConstantStringProvider.selected[]()) * x^2
df(f, x) = only(only(autodiff(Reverse, f, Active, Active(x))))

@testset "Constant binding returned by dynamic dispatch" begin
    d2 = only(autodiff(Forward, df, Const(by_length), Duplicated(0.7, 1.0)))
    @test d2 ≈ 2 * length(Base.libm_name)
end
