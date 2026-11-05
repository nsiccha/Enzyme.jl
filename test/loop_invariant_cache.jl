using Enzyme, Test

# A binary search Julia does not inline. LLVM cannot tell that the call writes
# no memory, so it keeps the caller's loads of loop-invariant data inside the
# loop; Enzyme knows that the call only reads or throws.
@noinline function loop_invariant_slot(keys::Vector{Int}, k::Int)
    lo, hi = 1, length(keys)
    while lo <= hi
        mid = (lo + hi) >>> 1
        c = @inbounds keys[mid]
        c == k && return mid
        if c < k
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 0
end

# Base's `sum(generator; init)` differentiates `_foldl_impl` as a call with a
# tape. Its fold loop reloads the arrays the generator captures from the
# generator, which is passed by reference and which the caller's next
# iteration overwrites, so the reverse pass cannot reload them: they are taped.
function loop_invariant_gen(field, weights, keys, starts, K)
    t = 0.0
    for s in starts
        t += sum((weights[j] * field[loop_invariant_slot(keys, s + j)] for j in 1:K); init = 0.0)
    end
    return t
end

function loop_invariant_loop(field, weights, keys, starts, K)
    t = 0.0
    for s in starts
        for j in 1:K
            t += weights[j] * field[loop_invariant_slot(keys, s + j)]
        end
    end
    return t
end

# `buf[1]` is written by the outer loop and read by the inner loop: the same in
# every inner iteration, not in every outer one. The write after both loops
# makes Enzyme cache it.
function loop_invariant_nested(buf, x, w, keys)
    s = 0.0
    for i in eachindex(x)
        buf[1] = x[i]
        for j in eachindex(w)
            s += buf[1] * w[j] * loop_invariant_slot(keys, j)
        end
    end
    buf[1] = 0.0
    return s
end

# The loop reads `buf[1]` and then writes it: a different value every
# iteration.
function loop_variant_store(buf, x, keys)
    s = 0.0
    for i in eachindex(x)
        s += buf[1] * x[i] * loop_invariant_slot(keys, i)
        buf[1] = x[i]
    end
    buf[1] = 0.0
    return s
end

@noinline loop_variant_set!(buf, v) = (buf[1] = v; nothing)

# The same, with the write in a call Julia does not inline.
function loop_variant_call(buf, x, keys)
    s = 0.0
    for i in eachindex(x)
        s += buf[1] * x[i] * loop_invariant_slot(keys, i)
        loop_variant_set!(buf, x[i])
    end
    buf[1] = 0.0
    return s
end

@testset "Reverse caches of values that do not change in a loop" begin
    keys = collect(1:64)

    @testset "generator sum: gradient" begin
        field = [sin(0.3i) for i in 1:64]
        weights = [cos(0.7j) for j in 1:8]
        starts = [3, 17, 40, 5, 22]
        df, dw = zero(field), zero(weights)
        autodiff(
            Reverse, Const(loop_invariant_gen), Active,
            Duplicated(field, df), Duplicated(weights, dw), Const(keys), Const(starts), Const(8),
        )
        rf, rw = zero(field), zero(weights)
        for s in starts, j in 1:8
            rf[s + j] += weights[j]
            rw[j] += field[s + j]
        end
        @test df ≈ rf
        @test dw ≈ rw
        df2, dw2 = zero(field), zero(weights)
        autodiff(
            Reverse, Const(loop_invariant_loop), Active,
            Duplicated(field, df2), Duplicated(weights, dw2), Const(keys), Const(starts), Const(8),
        )
        @test df == df2
        @test dw == dw2
    end

    @testset "generator sum: tape memory" begin
        K = 64
        nkeys = 4096
        bigkeys = collect(1:nkeys)
        field = [sin(0.01i) for i in 1:nkeys]
        weights = [cos(0.1j) for j in 1:K]
        df, dw = zero(field), zero(weights)
        grad(st) = (
            autodiff(
                Reverse, Const(loop_invariant_gen), Active,
                Duplicated(field, df), Duplicated(weights, dw), Const(bigkeys), Const(st), Const(K),
            ); nothing
        )
        small = [1, 2]
        grad(small)
        grad(small)
        starts = collect(1:2:(nkeys - K))
        terms = K * length(starts)
        grad(starts)
        bytes = @allocated grad(starts)
        # One tape entry per call of `_foldl_impl` for each captured array and
        # its shadow, not one per term: about 2 bytes per term here, against
        # about 150 when every iteration's reload is taped.
        @test bytes < 16 * terms
    end

    @testset "invariant in the inner loop only" begin
        x = [0.3, -1.1, 2.0]
        w = [0.5, 1.5, -0.25, 0.75]
        c = Float64.(1:4)
        buf, dbuf = [7.0], [0.0]
        dx, dw = zero(x), zero(w)
        autodiff(
            Reverse, loop_invariant_nested, Active,
            Duplicated(buf, dbuf), Duplicated(x, dx), Duplicated(w, dw), Const(keys),
        )
        @test dx ≈ fill(sum(w .* c), 3)
        @test dw ≈ sum(x) .* c
        @test dbuf == [0.0]
    end

    for f in (loop_variant_store, loop_variant_call)
        @testset "written in the loop: $f" begin
            x = [0.3, -1.1, 2.0, 0.9]
            b0 = 7.0
            buf, dbuf = [b0], [0.0]
            dx = zero(x)
            autodiff(Reverse, f, Active, Duplicated(buf, dbuf), Duplicated(x, dx), Const(keys))
            # s = b0 x1 c1 + x1 x2 c2 + x2 x3 c3 + x3 x4 c4, with c = 1:4
            @test dx ≈ [b0 * 1 + x[2] * 2, x[1] * 2 + x[3] * 3, x[2] * 3 + x[4] * 4, x[3] * 4]
            @test dbuf ≈ [x[1] * 1]
        end
    end
end
