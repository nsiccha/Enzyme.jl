using Enzyme, Test

@inline wide_active_read(r::Base.RefValue) = r[]
@inline wide_active_read(v::AbstractVector) = sum(v)
@inline wide_active_sum(::Tuple{}) = 0.0
@inline wide_active_sum(cs::Tuple) = wide_active_read(first(cs)) + wide_active_sum(Base.tail(cs))
@noinline wide_active_total(x, cs::Vararg{Any, N}) where {N} = wide_active_sum(cs) * x[1]

function wide_active_last(x, cs::Vararg{Any, N}) where {N}
    last(cs)[] = x[2]
    return wide_active_total(x, cs...)
end
function wide_active_first(x, box, cs::Vararg{Any, N}) where {N}
    box[] = x[2]
    return wide_active_total(x, box, cs...)
end
wide_active_vector(j) = fill(Float64(j), 3)

# Exercise both the differentiated entry and a splat inside it, with the active
# box on either side of the constant vectors. On Julia 1.12 the old splat limit
# left C-ABI ijl_apply_generic calls that the derivative rules could not handle.
@testset "Wide calls with an active Ref" begin
    for K in (31, 32, 40), first_box in (false, true)
        vs = ntuple(wide_active_vector, K)
        consts = map(Const, vs)
        f = first_box ? wide_active_first : wide_active_last
        S = 3K * (K + 1) / 2
        x = [0.3, 0.7]
        expected = [S + x[2], x[1]]
        @testset "K=$K first_box=$first_box" begin
            box, dbox = Ref(0.0), Ref(0.0)
            args = first_box ? (Duplicated(box, dbox), consts...) : (consts..., Duplicated(box, dbox))
            tangent, primal = autodiff(ForwardWithPrimal, f, Duplicated(x, [1.0, 0.0]), args...)
            @test tangent ≈ expected[1]
            @test primal ≈ (S + x[2]) * x[1]
            @test box[] == x[2]

            box, dbox = Ref(0.0), Ref(0.0)
            dx = zeros(2)
            args = first_box ? (Duplicated(box, dbox), consts...) : (consts..., Duplicated(box, dbox))
            _, primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), args...)
            @test dx ≈ expected
            @test primal ≈ (S + x[2]) * x[1]
            @test box[] == x[2]
            @test x == [0.3, 0.7]
            @test vs == ntuple(wide_active_vector, K)
        end
    end
end
