using Enzyme, Test

# Forward over Reverse of an outer loop whose inner loop has a data-dependent
# exit, so the inner Reverse keeps per-iteration caches in exponentially grown
# allocations, and is skipped in some outer iterations, so each cache pointer
# the outer reverse reads merges the skip path with the loop exit. Two of those
# caches hold GC-tracked Julia objects (an element of a `Vector{Vector{Float64}}`
# and of a `Vector{Vector{Int}}`), and the Julia frontend seeds every such cache
# slot with its undefined tracked pointer, `jl_nothing`, cast to the slot's
# type. LLVM uniques that constant expression, and type analysis learned the
# float cache's element layout and the integer cache's on that one shared
# value: `IllegalTypeAnalysisException`, "Illegal updateAnalysis", on a
# `bitcast (@ejl_jl_nothing …)` whose origin is the cache phi.

function cache_seed_loss(u, ww::Vector{Vector{Float64}}, jj::Vector{Vector{Int}}, counts::Vector{Int}, stop)
    s = 0.0
    for t in eachindex(counts)
        j = 1
        while j <= counts[t]
            q = ww[j]
            m = jj[j]
            s += q[1] * u[m[1]] * u[m[1]] + q[2] * u[m[2]]
            ww[j] = q           # pointer stores after the reads: the elements are cached
            jj[j] = m
            q[1] == stop && break
            j += 1
        end
    end
    return s
end

# The same iteration, accumulating the gradient and the Hessian-vector product by hand.
function cache_seed_reference(u, v, ww, jj, counts, stop)
    g = zero(u)
    hv = zero(u)
    for t in eachindex(counts)
        j = 1
        while j <= counts[t]
            q = ww[j]
            m = jj[j]
            g[m[1]] += 2 * q[1] * u[m[1]]
            g[m[2]] += q[2]
            hv[m[1]] += 2 * q[1] * v[m[1]]
            q[1] == stop && break
            j += 1
        end
    end
    return g, hv
end

function cache_seed_grad(u, ww, jj, counts, stop)
    g = zero(u)
    autodiff(Reverse, Const(cache_seed_loss), Active, Duplicated(u, g), Const(ww), Const(jj), Const(counts), Const(stop))
    return g
end

@testset "Forward over Reverse with jl_nothing-seeded caches of two element types" begin
    u = [0.3, -1.2, 2.0, 0.7]
    v = [1.0, 0.5, -1.0, 2.0]
    mk() = ([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], [[1, 2], [3, 4], [2, 3]])
    for counts in ([2, 0, 3], [3, 3, 3], [0, 0, 0], [1]), stop in (-7.0, 0.5)
        ww, jj = mk()
        g_ref, hv_ref = cache_seed_reference(u, v, ww, jj, counts, stop)
        ww, jj = mk()
        @test cache_seed_grad(u, ww, jj, counts, stop) ≈ g_ref
        ww, jj = mk()
        hv = autodiff(Forward, Const(cache_seed_grad), Duplicated(u, v), Const(ww), Const(jj), Const(counts), Const(stop))[1]
        @test hv ≈ hv_ref
    end
end
