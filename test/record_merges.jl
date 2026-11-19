using Enzyme, Test

const SampleRecord = Tuple{@NamedTuple{data::Matrix{Float64}}, @NamedTuple{score::Float64}}

Base.@noinline function fill_samples!(samples, x, weights, refresh)
    data = x .* weights
    local sample
    for ij in CartesianIndices(samples)
        if ij == first(CartesianIndices(samples)) || refresh[ij]
            sample = ((; data), (; score = x * ij[1] + ij[2]))
        end
        @inbounds samples[ij] = sample
    end
    return nothing
end

Base.@noinline function fill_history!(history, data, x, refresh)
    local entry
    for i in eachindex(history)
        if i == 1 || refresh[i]
            entry = (data, i * x)
        end
        @inbounds history[i] = entry
    end
    return nothing
end

function history_sum(x, weights, refresh)
    history = Vector{Tuple{typeof(weights), Float64}}(undef, length(refresh))
    fill_history!(history, x .* weights, x, refresh)
    return sum(r -> sum(r[1]) + r[2], history; init = 0.0)
end

function sample_sum(x, weights, refresh)
    samples = Matrix{SampleRecord}(undef, size(refresh))
    fill_samples!(samples, x, weights, refresh)
    return sum(s -> sum(s[1].data) + s[2].score, samples; init = 0.0)
end

@testset "Reverse of a conditionally reused pointer and scalar" begin
    for (refresh, dx) in (([false, false, true], 11.0), (trues(3), 12.0))
        weights = fill(2.0, 1, 1)
        dw = zero(weights)
        result = autodiff(Reverse, history_sum, Active, Active(0.4), Duplicated(weights, dw), Const(refresh))
        @test result[1][1] ≈ dx
        @test only(dw) ≈ 1.2
    end
end

@testset "Reverse of loop-carried records with pointer and floating-point fields" begin
    weights = [0.2 0.3; 0.5 0.7]
    for (refresh, row_sum) in (
                (trues(2, 3), 9),
                (falses(2, 3), 6),
                (Bool[0 0 1; 1 0 0], 9),
                (falses(0, 3), 0),
                (falses(2, 0), 0),
            ), w in (weights, zeros(0, 3), zeros(2, 0), zeros(0, 0))
        expected = length(refresh) * sum(w) + row_sum
        dx = autodiff(Reverse, sample_sum, Active, Active(0.4), Const(w), Const(refresh))[1][1]
        @test dx ≈ expected
    end
end
