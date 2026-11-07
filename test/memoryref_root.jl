using Enzyme, Test

# From Julia 1.11 an element read loads both fields of the array's `MemoryRef`, the data
# pointer and the `Memory` that roots it. The reverse pass recomputes the data pointer, so it
# must recompute that root too rather than tape it: a recursive callee's tape is a heap object,
# so one taped root allocated on every Reverse call, once per recursion node.

# Base's `sum` of a view of length >= 16 calls the recursive, non-inlined `mapreduce_impl`;
# the broadcast after it is a later write that might overwrite the view.
function sum_reused(u, buf)
    c = view(u, 1:length(buf))
    buf .= c .- sum(c)
    return sum(abs2, buf)
end

@noinline function pairwise_sum(A, lo, hi)
    if hi - lo < 8
        s = 0.0
        for i in lo:hi
            s += A[i]
        end
        return s
    end
    mid = (lo + hi) >>> 1
    return pairwise_sum(A, lo, mid) + pairwise_sum(A, mid + 1, hi)
end

function pairwise_reused(u, buf)
    c = view(u, 2:(length(buf) + 1))
    s = pairwise_sum(c, 1, length(c))
    buf .= c .* s
    return sum(buf)
end

function pairwise_vector_reused(v, buf)
    s = pairwise_sum(v, 1, length(v))
    buf .= v .* s
    return sum(buf)
end

# sum_reused: with b = c .- sum(c), the gradient is 2b - 2sum(b) on the view and zero
# elsewhere, and the Hessian on the view is 2I + 2(n - 2) * ones(n, n).
function sum_reused_gradient(u, n)
    c = u[1:n]
    b = c .- sum(c)
    g = zero(u)
    g[1:n] .= 2 .* b .- 2 * sum(b)
    return g
end

function sum_reused_hvp(u, v, n)
    h = zero(u)
    h[1:n] .= 2 .* v[1:n] .+ 2 * (n - 2) * sum(v[1:n])
    return h
end

# pairwise_reused and pairwise_vector_reused are sum(c)^2, so the gradient is 2sum(c) on c.
function pairwise_gradient(u, n, first)
    g = zero(u)
    g[first:(first + n - 1)] .= 2 * sum(u[first:(first + n - 1)])
    return g
end

function reverse_gradient(f, x, buf)
    dx = zero(x)
    dbuf = zero(buf)
    autodiff(Reverse, f, Active, Duplicated(x, dx), Duplicated(buf, dbuf))
    return dx, dbuf
end

function warm_reverse_bytes(f, x, buf)
    dx = zero(x)
    dbuf = zero(buf)
    call() = (fill!(dx, 0.0); fill!(dbuf, 0.0); autodiff(Reverse, f, Active, Duplicated(x, dx), Duplicated(buf, dbuf)); nothing)
    call()
    call()
    return @allocated call()
end

# `buf` holds active values, so the outer pass must see it `Duplicated` too: values stored
# into `Const` storage are not differentiated.
function hvp_gradient(u, buf)
    du = zero(u)
    autodiff(Reverse, sum_reused, Active, Duplicated(u, du), Duplicated(buf, zero(buf)))
    return du
end

@testset "MemoryRef roots recomputed with their data pointer" begin
    # n = 16 and 21 take one sequential node, 1025 and 4096 recurse.
    @testset "sum of a view reused in a broadcast, n = $n" for n in (16, 21, 1025, 4096)
        u = [sin(0.37i) + 1.5 for i in 1:(n + 5)]
        buf = zeros(n)
        du, dbuf = reverse_gradient(sum_reused, u, buf)
        @test du ≈ sum_reused_gradient(u, n) rtol = 1.0e-10
        @test all(iszero, dbuf)
        @test buf ≈ u[1:n] .- sum(u[1:n])
        @test warm_reverse_bytes(sum_reused, u, zeros(n)) == 0
    end

    @testset "hand-written recursive sum, n = $n" for n in (21, 100)
        u = [cos(0.21i) for i in 1:(n + 3)]
        du, _ = reverse_gradient(pairwise_reused, u, zeros(n))
        @test du ≈ pairwise_gradient(u, n, 2) rtol = 1.0e-10
        @test warm_reverse_bytes(pairwise_reused, u, zeros(n)) == 0
        v = u[1:n]
        dv, _ = reverse_gradient(pairwise_vector_reused, v, zeros(n))
        @test dv ≈ pairwise_gradient(v, n, 1) rtol = 1.0e-10
        @test warm_reverse_bytes(pairwise_vector_reused, v, zeros(n)) == 0
    end

    @testset "split mode, n = $n" for n in (21, 1025)
        u = [sin(0.37i) + 1.5 for i in 1:(n + 5)]
        buf = zeros(n)
        du = zero(u)
        dbuf = zero(buf)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(sum_reused)}, Active,
            Duplicated{typeof(u)}, Duplicated{typeof(buf)}
        )
        tape, primal, _ = fwd(Const(sum_reused), Duplicated(u, du), Duplicated(buf, dbuf))
        rev(Const(sum_reused), Duplicated(u, du), Duplicated(buf, dbuf), 1.0, tape)
        c = u[1:n]
        @test primal ≈ sum(abs2, c .- sum(c)) rtol = 1.0e-12
        @test du ≈ sum_reused_gradient(u, n) rtol = 1.0e-10
    end

    @testset "Forward over Reverse, n = $n" for n in (21, 1025)
        u = [sin(0.37i) + 1.5 for i in 1:(n + 5)]
        v = [cos(0.11i) for i in 1:(n + 5)]
        hv = autodiff(Forward, hvp_gradient, Duplicated(u, v), Duplicated(zeros(n), zeros(n)))[1]
        @test hv ≈ sum_reused_hvp(u, v, n) rtol = 1.0e-9
    end
end
