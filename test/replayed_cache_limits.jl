using Enzyme, Test

# For each group, `strip_walk` gathers at most `S` values into a buffer, runs a
# recurrence over them and stores every step, then refills the same buffer for
# each later strip in a `while` loop and continues the recurrence, and finally
# sums the squares of the stored steps. The first strip runs ahead of the
# `while` loop.
#
# On Julia 1.10 the reverse pass allocates the array of stored steps again for
# each group and replays the loops that fill it. The replay has loop counters of
# its own. The stored steps of the later strips come from a cache with one chunk
# per `while` iteration, because the extent of a strip is only known inside that
# loop, and reading such a cache computes the extent again. Before the dogfood
# libEnzyme repair that computation used the reverse pass's own counter of the
# `while` loop, which is not set while the replay runs, and left the strip bound
# it read there for the rest of the block. The exit test of the replayed step
# loop took that bound, so from the second `while` iteration on the replay never
# stopped at the end of its strip and stored past the end of the array: a
# segmentation fault or a corrupted heap at the first reverse gradient, as soon
# as one group took three strips.

@inline function strip_walk(theta::Vector{Float64}, idx::Vector{Int}, ::Val{S}) where {S}
    n = length(idx)
    lo = 1
    hi = min(lo + S - 1, n)
    amount = Vector{Float64}(undef, hi - lo + 1)
    @inbounds for i in 1:(hi - lo + 1)
        amount[i] = theta[idx[lo + i - 1]]
    end
    walk = Vector{Float64}(undef, n)
    if n > 0
        carry = 0.0
        @inbounds for j in 1:hi
            next = 0.5 * carry + amount[(j - lo) + 1]
            carry = next
            walk[j] = next
        end
        lo = hi + 1
        while lo <= n
            hi = min(lo + S - 1, n)
            @inbounds for i in 1:(hi - lo + 1)
                amount[i] = theta[idx[lo + i - 1]]
            end
            @inbounds for j in lo:hi
                next = 0.5 * carry + amount[(j - lo) + 1]
                carry = next
                walk[j] = next
            end
            lo = hi + 1
        end
    end
    s = 0.0
    @inbounds for j in 1:n
        s += abs2(walk[j])
    end
    return s
end

function strip_walks(theta::Vector{Float64}, index_by::Vector{Vector{Int}}, strip::Val)
    total = 0.0
    for g in eachindex(index_by)
        total += strip_walk(theta, index_by[g], strip)
    end
    return total
end

strip_walks_128(theta, index_by) = strip_walks(theta, index_by, Val(128))
strip_walks_4(theta, index_by) = strip_walks(theta, index_by, Val(4))

# The same body without the loop over groups: nothing is replayed.
single_walk_128(theta, index_by) = strip_walk(theta, index_by[1], Val(128))

# The same recurrence with its derivative carried along.
function strip_walks_reference(theta, index_by)
    value = 0.0
    gradient = zero(theta)
    for idx in index_by
        carry = 0.0
        dcarry = zero(theta)
        for q in idx
            carry = 0.5 * carry + theta[q]
            dcarry .*= 0.5
            dcarry[q] += 1.0
            value += abs2(carry)
            gradient .+= (2 * carry) .* dcarry
        end
    end
    return value, gradient
end

groups(lengths) = [[1 + mod(i, 3) for i in 1:n] for n in lengths]

const THETA = [0.3, -0.2, 0.7]

function check_reverse(f, lengths)
    index_by = groups(lengths)
    value, expected = strip_walks_reference(THETA, index_by)
    @test f(THETA, index_by) ≈ value

    theta = copy(THETA)
    gradient = zero(theta)
    autodiff(Reverse, f, Active, Duplicated(theta, gradient), Const(index_by))
    @test gradient ≈ expected rtol = 1.0e-12
    @test theta == THETA

    # A second call reuses the compiled derivative and must not depend on what
    # the first one left behind.
    fill!(gradient, 0.0)
    result = autodiff(ReverseWithPrimal, f, Active, Duplicated(theta, gradient), Const(index_by))
    @test result[2] ≈ value
    @test gradient ≈ expected rtol = 1.0e-12

    fill!(gradient, 0.0)
    autodiff(set_runtime_activity(Reverse), f, Active, Duplicated(theta, gradient), Const(index_by))
    @test gradient ≈ expected rtol = 1.0e-12
    return nothing
end

@testset "Stored steps of later strips, replayed per group" begin
    # 257 steps: strips of 128, 128 and 1. 387: 128, 128, 128 and 3.
    @testset "strips of 128, lengths $lengths" for lengths in (
            [257], [387], [3000], [5, 17, 129, 387, 64, 200, 1, 0, 513],
        )
        check_reverse(strip_walks_128, lengths)
    end
    @testset "strips of 4, lengths $lengths" for lengths in (
            [9], [13], [9, 2, 0, 13, 4, 6], [1, 5, 9, 4, 8, 12, 3, 11],
        )
        check_reverse(strip_walks_4, lengths)
    end
end

@testset "At most two strips, or no loop over groups" begin
    # These never read a later strip's bound at a stale index.
    @testset "lengths $lengths" for lengths in ([256], [128], [1], [0], Int[], [100, 256, 3])
        check_reverse(strip_walks_128, lengths)
    end
    check_reverse(single_walk_128, [387])
    check_reverse(strip_walks_4, [8, 4, 1])
end

@testset "Split mode and a batch of two" begin
    index_by = groups([9, 2, 0, 13, 4, 6])
    value, expected = strip_walks_reference(THETA, index_by)

    theta = copy(THETA)
    gradient = zero(theta)
    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(strip_walks_4)}, Active,
        Duplicated{Vector{Float64}}, Const{Vector{Vector{Int}}},
    )
    tape, primal, _ = forward(Const(strip_walks_4), Duplicated(theta, gradient), Const(index_by))
    @test primal ≈ value
    reverse(Const(strip_walks_4), Duplicated(theta, gradient), Const(index_by), 1.0, tape)
    @test gradient ≈ expected rtol = 1.0e-12

    first_gradient = zero(theta)
    second_gradient = zero(theta)
    autodiff(
        Reverse, strip_walks_4, Active,
        BatchDuplicated(theta, (first_gradient, second_gradient)), Const(index_by),
    )
    @test first_gradient ≈ expected rtol = 1.0e-12
    @test second_gradient ≈ expected rtol = 1.0e-12
end

# The objective is quadratic in theta, so its Hessian-vector product is the
# gradient of the same objective at the direction.
strip_walks_4_gradient!(gradient, theta, index_by) = begin
    fill!(gradient, 0.0)
    autodiff(Reverse, strip_walks_4, Active, Duplicated(theta, gradient), Const(index_by))
    return nothing
end

@testset "Forward over Reverse" begin
    index_by = groups([9, 2, 0, 13, 4, 6])
    direction = [1.0, -0.5, 0.25]
    _, expected = strip_walks_reference(direction, index_by)
    theta = copy(THETA)
    gradient = zero(theta)
    product = zero(theta)
    autodiff(
        Forward, strip_walks_4_gradient!, Const,
        Duplicated(gradient, product), Duplicated(theta, copy(direction)), Const(index_by),
    )
    @test gradient ≈ strip_walks_reference(THETA, index_by)[2] rtol = 1.0e-12
    @test product ≈ expected rtol = 1.0e-12
end
