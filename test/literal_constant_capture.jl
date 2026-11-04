using Enzyme, Test

struct LiteralMetadata{N}
    columns::NTuple{N, Vector{Float64}}
    index::Dict{Tuple{Float64, Float64}, Int32}
    rows::Vector{Int}
end

struct LiteralGeometry{M}
    metadata::M
    pointers::Vector{Int}
    nodes::Vector{Int}
end

const LITERAL_INDEX = Dict((1.0, 0.0) => Int32(2), (2.0, 0.0) => Int32(1))
literal_domain(g, r) = g.pointers[r]:(g.pointers[r + 1] - 1)
@noinline literal_coordinate(g, r, j) = g.nodes[j]
@noinline literal_reduce(gen) = sum(gen; init = 0.0)
@noinline literal_drive(step, rows) = sum(step(r) for r in rows; init = 0.0)

function literal_objective(x, columns, rows, ptr, nodes, ::Val{N}) where {N}
    g = LiteralGeometry(LiteralMetadata{N}(columns, LITERAL_INDEX, rows), ptr, nodes)
    field, weights = view(x, 1:2), view(x, 3:5)
    return literal_drive(rows) do r
        literal_reduce(weights[j] * field[literal_coordinate(g, r, j)] for j in literal_domain(g, r))
    end
end

@testset "Literal dictionary captured beside active views" begin
    want = [0.2, 0.4, 1.2, 0.7, 1.2]
    for n in (1, 8, 16, 32)
        x = [0.7, 1.2, 0.1, 0.2, 0.3]
        columns = ntuple(i -> [0.1i, 0.2i], n)
        rows, ptr, nodes = [1], [1, 4], [2, 1, 2]
        before = deepcopy((x, columns, rows, ptr, nodes, LITERAL_INDEX))
        args = (Const(columns), Const(rows), Const(ptr), Const(nodes), Const(Val(n)))

        dx = zero(x)
        res = autodiff(ReverseWithPrimal, literal_objective, Active, Duplicated(x, dx), args...)
        @test res[2] ≈ 0.62
        @test dx ≈ want

        d1, d2 = zero(x), zero(x)
        autodiff(Reverse, literal_objective, Active, BatchDuplicated(x, (d1, d2)), args...)
        @test d1 ≈ want
        @test d2 ≈ want

        tangent = collect(1.0:5.0)
        forward = autodiff(Forward, literal_objective, Duplicated(x, tangent), args...)
        @test forward[1] ≈ sum(want .* tangent)
        @test (x, columns, rows, ptr, nodes, LITERAL_INDEX) == before
    end
end

const LITERAL_GROUPS = [[1.5, -2.0], [0.5]]
const LITERAL_VALUES = [2.0, 3.0]

struct LiteralCapture{T}
    data::T
    active::Vector{Float64}
end

@noinline literal_read(c::LiteralCapture, i) = c.data[i] * c.active[i]
@noinline literal_nested_read(c::LiteralCapture, i) = sum(c.data[i]) * c.active[i]
literal_array_loss(x) = literal_drive(i -> literal_read(LiteralCapture(LITERAL_VALUES, x), i), 1:2)
literal_nested_loss(x) = literal_drive(i -> literal_nested_read(LiteralCapture(LITERAL_GROUPS, x), i), 1:2)

@testset "Literal constant data is shadowed without changing the primal" begin
    x = [0.7, 1.2]
    before = deepcopy((LITERAL_GROUPS, LITERAL_VALUES))
    for (f, want) in ((literal_array_loss, [2.0, 3.0]), (literal_nested_loss, [-0.5, 0.5]))
        dx = zero(x)
        autodiff(Reverse, f, Active, Duplicated(x, dx))
        @test dx ≈ want
        @test autodiff(Forward, f, Duplicated(x, ones(2)))[1] ≈ sum(want)
        @test (LITERAL_GROUPS, LITERAL_VALUES) == before
    end

    # Read the current contents on every execution of the same derivative.
    LITERAL_VALUES[1] = 4.0
    try
        dx = zero(x)
        autodiff(Reverse, literal_array_loss, Active, Duplicated(x, dx))
        @test dx ≈ [4.0, 3.0]
        @test LITERAL_VALUES == [4.0, 3.0]
    finally
        LITERAL_VALUES[1] = before[2][1]
    end
end
