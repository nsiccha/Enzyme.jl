using Enzyme, Test

struct ReshapedArrayFrame{T}
    values::Vector{Vector{T}}
end

@noinline reshaped_array_frame(a, c) = ReshapedArrayFrame(collect(map(vec, (a, c))))

function reshaped_frame_loss(u, data)
    frame = reshaped_array_frame(u[1] .+ data.offsets, data.log_scale)
    result = zero(eltype(u))
    for i in eachindex(frame.values[1])
        y = (u[i + 1] - frame.values[1][i]) * exp(-frame.values[2][i])
        result += y * y / 2
    end
    return result
end

function reshaped_frame_gradient!(du, u, data)
    fill!(du, zero(eltype(du)))
    autodiff(Reverse, reshaped_frame_loss, Active, Duplicated(u, du), Const(data))
    return nothing
end

function reshaped_frame_product(a, c)
    frame = reshaped_array_frame(a, c)
    return sum(frame.values[1] .* frame.values[2])
end

@testset "Constant reshapes stored beside active arrays" begin
    for T in (Float32, Float64), dims in ((2, 3), (1, 4), (0, 2))
        n = prod(dims)
        data = (
            offsets = reshape(T.(1:n) ./ 10, dims),
            log_scale = reshape(log.(T.(1:n) ./ 5 .+ 1), dims),
        )
        u = T.(cos.(1:(n + 1))) ./ 10
        saved_u, saved_data = copy(u), deepcopy(data)
        weights = exp.(-2 .* vec(data.log_scale))
        residual = u[2:end] .- u[1] .- vec(data.offsets)
        expected = [-sum(residual .* weights); residual .* weights]
        direction = T.(1:(n + 1)) ./ (n + 1)

        du = zero(u)
        result = autodiff(ReverseWithPrimal, reshaped_frame_loss, Active, Duplicated(u, du), Const(data))
        @test result[2] ≈ reshaped_frame_loss(u, data)
        @test du ≈ expected

        d1, d2 = zero(u), zero(u)
        autodiff(Reverse, reshaped_frame_loss, Active, BatchDuplicated(u, (d1, d2)), Const(data))
        @test d1 ≈ expected
        @test d2 ≈ expected

        tangent = autodiff(Forward, reshaped_frame_loss, Duplicated(u, direction), Const(data))[1]
        @test tangent ≈ sum(expected .* direction)
        tangents = autodiff(Forward, reshaped_frame_loss, BatchDuplicated(u, (direction, 2 .* direction)), Const(data))[1]
        @test tangents[1] ≈ tangent
        @test tangents[2] ≈ 2 * tangent

        forward, reverse = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(reshaped_frame_loss)}, Active,
            Duplicated{typeof(u)}, Const{typeof(data)},
        )
        fill!(du, zero(T))
        tape, primal, _ = forward(Const(reshaped_frame_loss), Duplicated(u, du), Const(data))
        reverse(Const(reshaped_frame_loss), Duplicated(u, du), Const(data), one(T), tape)
        @test primal ≈ reshaped_frame_loss(u, data)
        @test du ≈ expected

        hv = zero(u)
        autodiff(
            Forward, reshaped_frame_gradient!, Const,
            Duplicated(du, hv), Duplicated(u, direction), Const(data),
        )
        hresidual = (direction[2:end] .- direction[1]) .* weights
        @test hv ≈ [-sum(hresidual); hresidual]
        @test u == saved_u
        @test data == saved_data
    end
end

@testset "Reshapes preserve both active operands" begin
    a = [0.2 0.5; 0.4 0.8]
    c = [1.0 2.0; 3.0 4.0]
    saved_a, saved_c = copy(a), copy(c)
    da, dc = zero(a), zero(c)
    autodiff(Reverse, reshaped_frame_product, Active, Duplicated(a, da), Duplicated(c, dc))
    @test da ≈ c
    @test dc ≈ a
    tangent = autodiff(Forward, reshaped_frame_product, Duplicated(a, ones(size(a))), Duplicated(c, ones(size(c))))[1]
    @test tangent ≈ sum(a) + sum(c)
    @test a == saved_a
    @test c == saved_c
end
function reshaped_constant_row(u, c)
    C = reshape(c, 2, 3)
    B = zeros(2, 3)
    copyto!(view(B, 1, :), u[[1, 3, 5]])
    copyto!(view(B, 2, :), view(C, 2, :))
    return sum(B .* [1.0 2.0 3.0; 4.0 5.0 6.0])
end

@noinline function reshaped_constant_distance(x, m)
    t = zero(m)
    for i in eachindex(x)
        t += (x[i] - m)^2
    end
    return t
end
@noinline reshaped_constant_mix(x) = vec(fill(0.5, length(x), length(x)) * reshape(x, :, 1))
reshaped_constant_loss(x, m) = reshaped_constant_distance(x, m) + sum(reshaped_constant_mix(x))

@testset "Constant reshapes read through views and matrix products" begin
    u, c = collect(0.1:0.1:0.6), collect(0.3:0.3:1.8)
    saved_u, saved_c = copy(u), copy(c)
    expected = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
    du = zero(u)
    autodiff(Reverse, reshaped_constant_row, Active, Duplicated(u, du), Const(c))
    @test du ≈ expected
    @test autodiff(Forward, reshaped_constant_row, Duplicated(u, ones(6)), Const(c))[1] ≈ 6.0
    d1, d2 = zero(u), zero(u)
    autodiff(Reverse, reshaped_constant_row, Active, BatchDuplicated(u, (d1, d2)), Const(c))
    @test d1 ≈ expected
    @test d2 ≈ expected
    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(reshaped_constant_row)}, Active,
        Duplicated{typeof(u)}, Const{typeof(c)},
    )
    fill!(du, 0)
    tape, primal, _ = forward(Const(reshaped_constant_row), Duplicated(u, du), Const(c))
    reverse(Const(reshaped_constant_row), Duplicated(u, du), Const(c), 1.0, tape)
    @test du ≈ expected
    @test primal ≈ reshaped_constant_row(u, c)
    @test u == saved_u && c == saved_c

    x = [0.4]
    @test autodiff(Reverse, reshaped_constant_loss, Active, Const(x), Active(0.2))[1][2] ≈ -0.4
    @test x == [0.4]
end
