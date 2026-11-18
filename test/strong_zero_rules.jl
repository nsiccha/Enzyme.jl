using Enzyme, Test

# Under strong zero (`Enzyme.set_strong_zero`) a derivative that is zero
# contributes zero, whatever it is multiplied by or divided into. The rules of
# math functions scale the incoming derivative by a partial derivative, such as
# `log`'s `dx / x`, which is infinite where the function has a pole or an
# infinite slope. Before the dogfood libEnzyme repair `strong_zero_rules` only the
# rules of `*`, `/`, `^` and `rem` kept strong zero, so a zero derivative
# reaching `log` at 0 or `exp` at infinity still gave 0 / 0 or 0 * Inf = NaN.

# `0.0 * x[1]` has a zero derivative whatever the seed, so these differentiate
# `log` at 0 and `exp` at Inf (overflowed) with a zero derivative.
zero_tangent_log(x) = log(0.0 * x[1]) + x[2]
zero_tangent_exp(x) = inv(exp(1000.0 + 0.0 * x[1])) + x[2]

# A mixture with a constant zero-mass component, the shape of the report: the
# constant component sits in one vector with the active one, so it has a zero
# shadow, and `log(0)` of its zero probability is differentiated with a zero
# derivative. The first component's weight carries the whole density.
struct StrongZeroComponent
    p::Vector{Float64}
end

function component_logpdf(c::StrongZeroComponent, y)
    s = 0.0
    for i in eachindex(y)
        s += log(y[i] == 1 ? c.p[i] : 1 - c.p[i])
    end
    return s
end

function zero_mass_mixture(q, y)
    components = [StrongZeroComponent(q[2:end]), StrongZeroComponent(zeros(length(q) - 1))]
    lp1 = log(q[1]) + component_logpdf(components[1], y)
    lp2 = log(1 - q[1]) + component_logpdf(components[2], y)
    m = max(lp1, lp2)
    return m + log(exp(lp1 - m) + exp(lp2 - m))
end

# A call that reaches the result only through a select arm: Reverse gives it a
# zero adjoint where the select picks the other arm (dogfood repair
# `select_strong_zero`), which now holds for calls as it did for `/` and `*`.
masked_log(x, t) = ifelse(x > t, log(x), 0.0)

@testset "Strong zero in every rule that scales the incoming derivative" begin
    x = [0.3, 0.7]
    strong = Enzyme.set_strong_zero(Forward)

    @testset "log at 0 and exp at Inf with a zero derivative" begin
        for f in (zero_tangent_log, zero_tangent_exp)
            @test autodiff(strong, f, Duplicated(x, [1.0, 0.0]))[1] == 0.0
            @test autodiff(strong, f, Duplicated(x, [0.0, 1.0]))[1] == 1.0
            batched = autodiff(strong, f, BatchDuplicated(x, ([1.0, 0.0], [0.0, 1.0])))[1]
            @test collect(Tuple(batched)) == [0.0, 1.0]
        end
    end

    @testset "zero-mass mixture component" begin
        q = [0.4, 0.2, 0.4, 0.6]
        y = [1, 0, 1]
        expected = [1 / 0.4, 1 / 0.2, -1 / 0.6, 1 / 0.6]
        reverse = zeros(4)
        autodiff(Reverse, zero_mass_mixture, Active, Duplicated(q, reverse), Const(y))
        @test reverse ≈ expected
        basis = ntuple(i -> [Float64(j == i) for j in 1:4], 4)
        single = [autodiff(strong, zero_mass_mixture, Duplicated(q, basis[i]), Const(y))[1] for i in 1:4]
        @test single ≈ expected
        for mode in (strong, Enzyme.set_abi(strong, InlineABI))
            batched = autodiff(mode, zero_mass_mixture, BatchDuplicated(q, basis), Const(y))[1]
            @test collect(Tuple(batched)) ≈ expected
        end
    end

    @testset "a call masked by a select, in Reverse" begin
        @test autodiff(Reverse, masked_log, Active, Active(0.0), Const(1.0e-12))[1][1] == 0.0
        @test autodiff(Reverse, masked_log, Active, Active(2.0), Const(1.0e-12))[1][1] == 0.5
    end
end
