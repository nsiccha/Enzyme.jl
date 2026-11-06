using Enzyme, Test

# The generator is differentiated as a separate call. Storing each result in
# an output array must not make that call's input arrays appear overwritten.
function owned_output_sum(field, weights, starts)
    out = Vector{Float64}(undef, length(starts))
    for i in eachindex(starts)
        s = starts[i]
        out[i] = sum((weights[j] * field[s + j] for j in eachindex(weights)); init = 0.0)
    end
    return sum(abs2, out)
end

function owned_generated_output_sum(field, weights, starts)
    return owned_output_sum(sin.(field), exp.(weights), starts)
end

@noinline function owned_output_reader(v, x)
    return sum(((v[j] - x)^2 for j in eachindex(v)); init = 0.0)
end

# Here the reader receives the new array itself, so writes on the loop
# backedge really do overwrite its inputs. Its earlier values must be taped.
function owned_output_operand(x)
    v = [1.0, 2.0, 3.0]
    total = 0.0
    for i in eachindex(v)
        total += owned_output_reader(v, x)
        v[i] = 0.0
    end
    return total
end

@testset "Reader followed by writes to owned output storage" begin
    field = [sin(0.3i) for i in 1:64]
    weights = [cos(0.7j) for j in 1:8]
    starts = [3, 17, 40, 5, 22]
    for f in (owned_output_sum, owned_generated_output_sum)
        df, dw = zero(field), zero(weights)
        autodiff(Reverse, Const(f), Active, Duplicated(field, df), Duplicated(weights, dw), Const(starts))
        generated = f === owned_generated_output_sum
        values = generated ? sin.(field) : field
        coeffs = generated ? exp.(weights) : weights
        rf, rw = zero(field), zero(weights)
        for s in starts
            value = sum(coeffs[j] * values[s + j] for j in eachindex(coeffs))
            for j in eachindex(coeffs)
                rf[s + j] += 2value * coeffs[j]
                rw[j] += 2value * values[s + j]
            end
        end
        if generated
            rf .*= cos.(field)
            rw .*= exp.(weights)
        end
        @test df ≈ rf
        @test dw ≈ rw
    end
    for x in (0.3, -1.2, 2.5)
        dx = autodiff(Reverse, owned_output_operand, Active(x))[1][1]
        @test dx ≈ 18x - 28
    end
end
