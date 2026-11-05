using Enzyme, Test
struct Metadata{N}
    columns::NTuple{N, Vector{Float64}}
    index::Dict{Tuple{Float64, Float64}, Int32}
    rows::Vector{NamedTuple{(:position, :time), Tuple{Int32, Int32}}}
end
struct Geometry{M}
    metadata::M
    pointers::Vector{Int}
    nodes::Vector{Int}
end
const INDEX = Dict((1.0, 0.0) => Int32(2), (2.0, 0.0) => Int32(1))
domain(g, r) = g.pointers[r.position]:(g.pointers[r.position + 1] - 1)
function coordinate(g, r, j)
    t = r.time
    w, bit = divrem(t - 1, 64)
    lo, hi = 1, length(g.nodes)
    while lo <= hi
        mid = (lo + hi) ÷ 2
        candidate = g.nodes[mid]
        if candidate == j
            mask = UInt64(1) << bit
            UInt64(7) & mask == 0 && return 0
            return mod(candidate + count_ones(UInt64(7) & (mask - UInt64(1))), 2) + 1
        elseif candidate < j
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 1
end
@noinline reduce_generator(gen) = sum(gen; init = 0.0)
@noinline drive(step, rows) = sum(step(r) for r in rows; init = 0.0)
@noinline function cell(x, r, columns, rows, ptr, nodes, ::Val{N}) where {N}
    g = Geometry(Metadata{N}(columns, INDEX, rows), ptr, nodes)
    field, weights = x[1:2], x[3:5]
    return reduce_generator(weights[j] * field[coordinate(g, r, j)] for j in domain(g, r))
end
function objective(x, columns, rows, ptr, nodes, v)
    return drive(rows) do r
        cell(x, r, columns, rows, ptr, nodes, v)
    end
end
function measure(nkeys, nreads, ::Val{N}) where {N}
    empty!(INDEX)
    sizehint!(INDEX, nkeys)
    for i in 1:nkeys
        INDEX[(Float64(i), 0.0)] = Int32(i)
    end
    original = copy(INDEX)
    x = [0.7, 1.2, 0.1, 0.2, 0.3]
    dx = zero(x)
    columns = ntuple(i -> [0.1i, 0.2i], Val(N))
    rows = fill((position = Int32(1), time = Int32(1)), nreads)
    ptr, nodes = [1, 4], [1, 2, 3]
    run() = autodiff(Reverse, objective, Active, Duplicated(x, dx), Const(columns), Const(rows), Const(ptr), Const(nodes), Const(Val(N)))
    run()
    @test dx ≈ nreads .* [0.2, 0.4, 1.2, 0.7, 1.2]
    fill!(dx, 0)
    GC.gc()
    bytes = @allocated run()
    @test dx ≈ nreads .* [0.2, 0.4, 1.2, 0.7, 1.2]
    @test INDEX == original
    return bytes
end
@testset "Unused dictionary forwarded through reader records" begin
    # Only the numeric reader work may scale with the number of readers. The
    # dictionary is never read, even though its pointer travels through several
    # non-inlined reduction records before the binary-search coordinate helper.
    for width in (Val(0), Val(32))
        bytes = [measure(nkeys, nreads, width) for nkeys in (2, 128, 2048), nreads in (1, 16, 256)]
        for reads in axes(bytes, 2)
            @test maximum(bytes[:, reads]) <= minimum(bytes[:, reads]) + 1024
        end
    end
end

struct ForwardedRecord{T}
    payload::T
    work::Vector{Float64}
end

@noinline forwarded_read(r) = sum(r.payload) * sum(r.work)
@noinline forwarded_copy(r) = forwarded_read(ForwardedRecord(r.payload, r.work))
@noinline forwarded_branch(r, observe) = observe ? forwarded_copy(r) : sum(r.work)

function forwarded_objective(x, observe)
    return forwarded_branch(ForwardedRecord(x, x), observe)
end

@testset "Forwarded fields with observable contents remain active" begin
    x = [0.7, 1.2, -0.3]
    before = copy(x)
    for observe in (false, true)
        want = fill(observe ? 2sum(x) : 1.0, length(x))
        dx = zero(x)
        autodiff(Reverse, forwarded_objective, Active, Duplicated(x, dx), Const(observe))
        @test dx ≈ want
        @test autodiff(Forward, forwarded_objective, Duplicated(x, ones(length(x))), Const(observe))[1] ≈ sum(want)
        @test x == before
    end
end
