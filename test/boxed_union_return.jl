using Enzyme, LinearAlgebra, Test

# A call whose return type is a union of arrays with one element type, such as
# `getproperty(nt, s)` of `nt::@NamedTuple{y::Vector{Float64}, A::Matrix{Float64}}` with
# `s::Symbol`, returns a boxed pointer. On Julia 1.10 that union is a subtype of
# `Array{Float64}`, so its type tree is that of an array object; the return value needed the
# pointer level a boxed argument gets. Without it, compiling the derivative of such a call
# aborted the process with "Canonicalization failed".

# A non-constant global: every read of it dispatches at run time, so Enzyme compiles the
# derivative of `getproperty(::typeof(data), ::Symbol)` on its own.
data = (; y = [0.6, -0.7], A = [2.0 -0.4; 0.3 1.5])

const boxed_union_y = [0.6, -0.7]
const boxed_union_A = [2.0 -0.4; 0.3 1.5]

global_product_loss(b) = -2log(2pi) - (dot(b, b) + sum(abs2, data.y - data.A * b)) / 2
global_solve_loss(b) = -2log(2pi) - (dot(b, b) + sum(abs2, data.y - data.A \ b)) / 2

@noinline pick_field(nt, s) = getproperty(nt, s)
dynamic_pick(nt, s) = Base.inferencebarrier(pick_field)(nt, s)

# The union return carries derivatives: both fields alias the active input.
function built_loss(x)
    nt = (; y = x, A = reshape(x, 1, :))
    return sum(abs2, dynamic_pick(nt, :y)) + sum(dynamic_pick(nt, :A))
end

# A union of bits types is returned split (payload and selector), not boxed.
@noinline select_integer(b) = b ? Int32(2) : Int64(3)
split_union_loss(x) = x[1] * Base.inferencebarrier(select_integer)(x[2] > 0)

@testset "Boxed union of arrays returned by a dynamically dispatched call" begin
    q = [0.35, -0.2]
    product = transpose(boxed_union_A) * (boxed_union_y - boxed_union_A * q) - q
    solve = transpose(boxed_union_A) \ (boxed_union_y - boxed_union_A \ q) - q
    @test only(Enzyme.gradient(Reverse, global_product_loss, q)) ≈ product
    @test only(Enzyme.gradient(Reverse, global_solve_loss, q)) ≈ solve
    @test collect(only(Enzyme.gradient(Forward, global_product_loss, q))) ≈ product
    @test data.y == boxed_union_y && data.A == boxed_union_A

    for x in ([0.3, -1.2, 2.0], [1.5])
        @test only(Enzyme.gradient(Reverse, built_loss, x)) ≈ 2 .* x .+ 1
        @test collect(only(Enzyme.gradient(Forward, built_loss, x))) ≈ 2 .* x .+ 1
    end
    @test only(Enzyme.gradient(Reverse, built_loss, Float64[])) == Float64[]
end

@testset "Union of arrays as the differentiated function's return" begin
    r = autodiff(ReverseWithPrimal, Const(pick_field), Const, Const(data), Const(:A))
    @test r[2] === data.A
    r = autodiff(ForwardWithPrimal, Const(pick_field), Const, Const(data), Const(:y))
    @test only(r) === data.y
end

@testset "Split union of integers returned by a dynamically dispatched call" begin
    @test only(Enzyme.gradient(Reverse, split_union_loss, [0.7, 1.5])) == [2.0, 0.0]
    @test only(Enzyme.gradient(Reverse, split_union_loss, [0.7, -1.5])) == [3.0, 0.0]
end
