module ConditionalRuleReturn

using Enzyme, Test

# The reshape rule returns the primal array through a cached load. The early
# return means that this load does not dominate the whole reverse pass.
function reshape_loss(x, observations)
    size(observations, 2) == 1 || return zero(eltype(x))
    scales = reshape(x, 1, length(x))
    return sum(abs2, observations ./ scales)
end

function reshape_gradient(x, observations)
    dx = zero(x)
    autodiff(Reverse, reshape_loss, Active, Duplicated(x, dx), Const(observations))
    return dx
end

@testset "Cached rule returns after a conditional branch" begin
    for n in (0, 1, 3, 17), columns in (1, 2)
        x = (-1.0) .^ (1:n) .* (0.8 .+ collect(1:n) ./ 5)
        observations = reshape(collect(1:(5 * columns)) ./ 10, 5, columns)
        saved = copy(x)
        expected = columns == 1 ? -2sum(abs2, observations) ./ x .^ 3 : zero(x)
        for mode in (Reverse, ReverseWithPrimal)
            dx = zero(x)
            result = autodiff(
                mode, reshape_loss, Active, Duplicated(x, dx), Const(observations)
            )
            @test dx ≈ expected
            if mode === ReverseWithPrimal
                @test result[2] ≈ reshape_loss(x, observations)
            end
        end

        dx1, dx2 = zero(x), zero(x)
        autodiff(
            Reverse, reshape_loss, Active, BatchDuplicated(x, (dx1, dx2)),
            Const(observations)
        )
        @test dx1 ≈ expected
        @test dx2 ≈ expected

        args = (Duplicated(x, zero(x)), Const(observations))
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(reshape_loss)}, Active, map(typeof, args)...
        )
        tape, primal, _ = fwd(Const(reshape_loss), args...)
        rev(Const(reshape_loss), args..., 1.0, tape)
        @test args[1].dval ≈ expected
        @test primal ≈ reshape_loss(x, observations)
        @test x == saved
    end

    for columns in (1, 2)
        x = [-0.8, 1.2, 2.3]
        direction = [0.2, -0.7, 1.0]
        observations = reshape(collect(1:(5 * columns)) ./ 10, 5, columns)
        expected = columns == 1 ? 6sum(abs2, observations) .* direction ./ x .^ 4 : zero(x)
        hvp = only(
            autodiff(
                Forward, reshape_gradient, Duplicated(x, direction), Const(observations)
            )
        )
        @test hvp ≈ expected
    end
end

end
