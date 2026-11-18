module GenIntegrationTests

using Enzyme, Gen, Test

@gen function normal_model(y::Vector{Float64})
    mu ~ normal(0, 1)
    for i in eachindex(y)
        {(:y, i)} ~ normal(mu, 1)
    end
end

const Y = [0.3, -1.2, 0.8, 2.1, -0.4]
const OLD_TRACE = let choices = choicemap((:mu, 0.1))
    for (i, y) in enumerate(Y)
        choices[(:y, i)] = y
    end
    first(generate(normal_model, (Y,), choices))
end

function updated_score(x)
    trace = first(update(OLD_TRACE, (Y,), (NoChange(),), choicemap((:mu, x[1]))))
    return get_score(trace)
end

@gen (static) function observation(log_rate::Float64)
    count ~ poisson(exp(log_rate))
end

const observation_map = Map(observation)
rates(a, t) = a .* t

@gen (static) function poisson_model(t::Vector{Float64})
    a ~ normal(0, 1)
    counts ~ observation_map(rates(a, t))
end

struct MapScore{C}
    t::Vector{Float64}
    counts::C
end

function (f::MapScore)(x)
    choices = Gen.StaticChoiceMap((a = x[1],), (counts = f.counts,))
    return first(assess(poisson_model, (f.t,), choices))
end

@testset "Reverse through a dynamic trace update" begin
    x = [0.3]
    dx = zero(x)
    _, score = autodiff(ReverseWithPrimal, Const(updated_score), Active, Duplicated(x, dx))
    @test score ≈ updated_score(x)
    @test dx[1] ≈ sum(Y) - (1 + length(Y)) * x[1]
    @test OLD_TRACE[:mu] == 0.1
end

@testset "Reverse through a Map of static observations" begin
    # Four observations fit in the first leaf, forty need another leaf.
    for n in (4, 40)
        t = repeat([0.1, 0.5, 0.9, 1.3], n ÷ 4)
        counts = repeat([1, 0, 3, 2], n ÷ 4)
        submaps = [Gen.StaticChoiceMap((count = k,), NamedTuple()) for k in counts]
        f = MapScore(t, Gen.vectorize_internal(submaps))
        x = [0.3]
        dx = zero(x)
        _, score = autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(x, dx))
        @test score ≈ f(x)
        @test dx[1] ≈ -x[1] + sum(t .* (counts .- exp.(x[1] .* t)))
    end
end

end # module
