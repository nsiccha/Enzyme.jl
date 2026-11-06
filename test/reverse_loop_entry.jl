using Enzyme, Test

# In the reverse of an inner loop, a value of the outer loop's body that the
# reverse pass recomputes (`s[j]`, `idx[j]` and the data pointers below it) is the
# same in every inner iteration. libEnzyme computes it once on entry to that
# reverse, rather than again in each arm on each iteration. These tests check the
# derivatives stay right in the shapes where that move must, or must not, happen.
# libEnzyme's lit test ReverseMode/loopentryrecompute.ll checks where the
# recomputation goes; the last testset here checks it reaches Julia's code.

function central_differences(f, x; h = 1.0e-6)
    g = zero(x)
    for i in eachindex(x)
        up = copy(x)
        down = copy(x)
        up[i] += h
        down[i] -= h
        g[i] = (f(up) - f(down)) / (2h)
    end
    return g
end

function reverse_gradient(mode, f, x, args...)
    dx = zero(x)
    autodiff(mode, Const(f), Active, Duplicated(x, dx), map(Const, args)...)
    return dx
end

@inline function three_arms(t, s, c)
    if t < -0.5
        return s * t * t
    elseif t < 0.5
        return exp(s * (t - c)) + t
    else
        return s * t / (1 + c * c)
    end
end

# A ragged nest: the inner trip count differs per outer iteration, and may be 0.
function ragged(x, idx, s, c)
    total = 0.0
    for j in eachindex(idx)
        ij, sj, cj = idx[j], s[j], c[j]
        inner = 0.0
        for k in eachindex(ij)
            inner += three_arms(x[ij[k]], sj, cj)
        end
        total += inner
    end
    return total
end

const IDX = [[1, 2, 3], Int[], [4], [2, 4, 1, 3], [3, 3]]
const S = [0.5, 9.0, 1.5, -0.25, 2.0]
const C = [0.1, 9.0, -0.3, 0.7, 0.2]
const X = [-0.9, -0.1, 0.3, 0.8]

@testset "Values of the outer iteration in a ragged nest" begin
    fd = central_differences(x -> ragged(x, IDX, S, C), X)
    @test reverse_gradient(Reverse, ragged, X, IDX, S, C) ≈ fd atol = 1.0e-7
    @test reverse_gradient(ReverseWithPrimal, ragged, X, IDX, S, C) ≈ fd atol = 1.0e-7

    # Each outer iteration reads its own values: permuting them must move the
    # derivatives with them.
    p = [4, 1, 5, 3, 2]
    fdp = central_differences(x -> ragged(x, IDX[p], S[p], C[p]), X)
    @test reverse_gradient(Reverse, ragged, X, IDX[p], S[p], C[p]) ≈ fdp atol = 1.0e-7

    # Every inner loop empty, and no outer iterations at all.
    empties = [Int[], Int[]]
    @test reverse_gradient(Reverse, ragged, X, empties, S[1:2], C[1:2]) == zero(X)
    @test reverse_gradient(Reverse, ragged, X, Vector{Int}[], Float64[], Float64[]) == zero(X)
end

# A value used only on an arm that does not run in some outer iterations.
function rarely_used(x, idx, s)
    total = 0.0
    for j in eachindex(idx)
        sj = s[j]
        for i in idx[j]
            total += x[i] > 0 ? sj * x[i] : x[i] * x[i]
        end
    end
    return total
end

@testset "A value used on some arms only" begin
    x = [-1.0, 2.0, -3.0, 4.0]
    idx = [[1, 3], [2, 4], [1, 2]]
    s = [100.0, 3.0, 0.5]
    @test reverse_gradient(Reverse, rarely_used, x, idx, s) ≈
        central_differences(y -> rarely_used(y, idx, s), x) atol = 1.0e-6
end

# The memory a value was read from is overwritten after the inner loop, so the
# value cannot be recomputed in the reverse pass at all.
function overwritten_after(x, s, n)
    total = 0.0
    for j in 1:length(s)
        sj = s[j]
        for k in 1:n
            total += sj * x[k] * x[k]
        end
        s[j] = 0.0
    end
    return total
end

@testset "Memory overwritten after the inner loop" begin
    x = [1.0, 2.0, 3.0]
    primal_s = [2.0, 3.0]
    s = copy(primal_s)
    ds = zero(s)
    dx = zero(x)
    autodiff(Reverse, Const(overwritten_after), Active, Duplicated(x, dx), Duplicated(s, ds), Const(3))
    @test dx ≈ central_differences(y -> overwritten_after(y, copy(primal_s), 3), x) atol = 1.0e-6
    @test ds ≈ central_differences(t -> overwritten_after(x, copy(t), 3), primal_s) atol = 1.0e-6
    @test s == [0.0, 0.0]
end

# The inner loop may leave through two exits that both return to the outer loop's
# body (with the first a `break`), so that no reverse block dominates its reverse.
function two_exits(x, idx, s)
    total = 0.0
    for j in eachindex(idx)
        sj = s[j]
        running = 0.0
        hit = false
        for i in idx[j]
            running += sj * x[i]
            if running > 3.0
                hit = true
                break
            end
        end
        total += hit ? 2.0 * running : running
    end
    return total
end

@testset "An inner loop with two exits" begin
    x = [1.1, 2.3, 3.7, 4.2]
    idx = [[1, 2, 3], [4], [2, 1], [3, 4, 1]]
    s = [1.0, 2.0, 0.5, 3.0]
    @test reverse_gradient(Reverse, two_exits, x, idx, s) ≈
        central_differences(y -> two_exits(y, idx, s), x) atol = 1.0e-6
end

# An array of the outer iteration that the reverse pass rematerializes rather than
# caches, read in the inner loop.
function rematerialized(x, m)
    total = 0.0
    for j in eachindex(x)
        a = Vector{Float64}(undef, 2)
        a[1] = x[j]
        a[2] = 2 * x[j]
        for k in 1:m
            total += a[1] * a[2] * x[k]
        end
    end
    return total
end

@testset "An array the reverse pass rematerializes" begin
    x = [1.0, -2.0, 3.0]
    @test reverse_gradient(Reverse, rematerialized, x, 2) ≈
        central_differences(y -> rematerialized(y, 2), x) atol = 1.0e-6
end

# A sibling loop's last value read in a later inner loop.
function sibling(x, idx)
    total = 0.0
    for j in eachindex(idx)
        last_value = 0.0
        for i in idx[j]
            last_value = x[i]
        end
        for i in idx[j]
            total += last_value * x[i]
        end
    end
    return total
end

@testset "The last value of a sibling loop" begin
    x = [1.0, 2.0, 3.0]
    idx = [[1, 2], [3], [2, 3, 1]]
    @test reverse_gradient(Reverse, sibling, x, idx) ≈
        central_differences(y -> sibling(y, idx), x) atol = 1.0e-6
end

# Split mode: the reverse pass is a function of its own that receives the tape.
@testset "Split mode" begin
    args = (Duplicated(copy(X), zero(X)), Const(IDX), Const(S), Const(C))
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(ragged)}, Active, map(typeof, args)...,
    )
    tape, primal, _ = fwd(Const(ragged), args...)
    rev(Const(ragged), args..., 1.0, tape)
    @test primal ≈ ragged(X, IDX, S, C)
    @test args[1].dval ≈ central_differences(x -> ragged(x, IDX, S, C), X) atol = 1.0e-7
end

@testset "Second derivatives" begin
    v = [1.0, -0.5, 0.25, 2.0]
    gradient_of(x) = reverse_gradient(Reverse, ragged, x, IDX, S, C)
    expected = central_differences(x -> sum(gradient_of(x) .* v), X)
    forward_over_reverse = only(autodiff(Forward, Const(gradient_of), Duplicated(X, v)))
    @test forward_over_reverse ≈ expected atol = 1.0e-6

    directional(x) = only(autodiff(Forward, Const(ragged), Duplicated(x, v), map(Const, (IDX, S, C))...))
    reverse_over_forward = zero(X)
    autodiff(Reverse, Const(directional), Active, Duplicated(copy(X), reverse_over_forward))
    @test reverse_over_forward ≈ expected atol = 1.0e-6
end

function ragged!(out, x, idx, s, c)
    out[] = ragged(x, idx, s, c)
    return nothing
end

@testset "A batch of directions" begin
    dx1 = zero(X)
    dx2 = zero(X)
    autodiff(
        Reverse, Const(ragged!), Const,
        BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0))),
        BatchDuplicated(copy(X), (dx1, dx2)),
        Const(IDX), Const(S), Const(C),
    )
    fd = central_differences(x -> ragged(x, IDX, S, C), X)
    @test dx1 ≈ fd atol = 1.0e-7
    @test dx2 ≈ 2 .* fd atol = 1.0e-7
end

# Collections in the inner loop's reverse: what was computed on entry to it must
# stay valid across every safepoint of the loop. `exp` holds a safepoint.
@noinline collect_sometimes(i) = (i % 3 == 0 && GC.gc(false); nothing)
Enzyme.EnzymeRules.inactive(::typeof(collect_sometimes), args...) = nothing

function with_collections(x, n)
    # Allocated here and read in the loop, so that the reverse pass depends on
    # objects that only the function itself holds.
    idx = [collect(1:min(j, length(x))) for j in 1:n]
    s = [0.5 * j for j in 1:n]
    total = 0.0
    for j in 1:n
        ij, sj = idx[j], s[j]
        for k in eachindex(ij)
            collect_sometimes(k)
            total += exp(sj * x[ij[k]])
        end
    end
    return total
end

@testset "Collections during the loop" begin
    x = [0.1, -0.2, 0.3]
    expected = central_differences(y -> with_collections(y, 4), x)
    for _ in 1:20
        @test reverse_gradient(Reverse, with_collections, x, 4) ≈ expected atol = 1.0e-6
    end
end

# The ordinal log density of snag reverse-of-an-in-a042e462 (PosteriorDB Bones):
# five arms in each of three arms, in a ragged nest, reading four outer values.
@inline function log1pexp_arms(x)
    if x < -745.1332191019412
        return 0.0
    elseif x < -36.7368005696771
        return exp(x)
    elseif x < 18.021826694558577
        return log1p(exp(x))
    elseif x < 33.23111882352963
        return x + exp(-x)
    else
        return x
    end
end

@inline function ordinal_cell(g, t, c, s)
    K = length(c) + 1
    if g == 1
        return -log1pexp_arms(s * (t - c[1]))
    elseif g == K
        return -log1pexp_arms(-s * (t - c[K - 1]))
    else
        p = s * (t - c[g - 1])
        q = s * (t - c[g])
        return -log1pexp_arms(-p) - log1pexp_arms(q) + log(-expm1(q - p))
    end
end

function ordinal(theta, grade, child, cuts, delta)
    total = 0.0
    for j in eachindex(grade)
        gj, cj, c, s = grade[j], child[j], cuts[j], delta[j]
        inner = 0.0
        for k in eachindex(gj)
            inner += ordinal_cell(gj[k], theta[cj[k]], c, s)
        end
        total += inner
    end
    return total
end

@testset "Ordinal log density" begin
    grade = [[1, 2, 3], [2, 1], Int[], [3, 3, 1, 2]]
    child = [[1, 2, 4], [3, 1], Int[], [4, 2, 3, 1]]
    cuts = [[-0.5, 0.5], [0.0, 1.0], [-1.0, 1.0], [-1.5, 0.25]]
    delta = [1.0, 2.5, 7.0, 0.75]
    theta = [-1.0, 0.25, 0.5, 1.5]
    fd = central_differences(y -> ordinal(y, grade, child, cuts, delta), theta)
    @test reverse_gradient(Reverse, ordinal, theta, grade, child, cuts, delta) ≈ fd atol = 1.0e-6
    @test reverse_gradient(ReverseWithPrimal, ordinal, theta, grade, child, cuts, delta) ≈ fd atol = 1.0e-6
end

derivative_ir(f, argtypes) = sprint() do io
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, argtypes;
        dump_module = true, second_stage = false, debuginfo = :none
    )
end

function with_recompute_option(f, enabled)
    Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryRecompute, enabled)
    try
        return f()
    finally
        Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryRecompute, true)
    end
end

@testset "Recomputed once per entry of the inner loop's reverse" begin
    argtypes = Tuple{Duplicated{Vector{Float64}}, Const{Vector{Vector{Int}}}, Const{Vector{Float64}}, Const{Vector{Float64}}}
    loads = count(r"= load ", derivative_ir(ragged, argtypes))
    in_each_arm = with_recompute_option(false) do
        count(r"= load ", derivative_ir(ragged, argtypes))
    end
    @test loads < in_each_arm

    # Only where the recomputation runs differs, not what it computes. A new
    # method, since the option acts when a derivative is compiled.
    moved_back(x, idx, s, c) = ragged(x, idx, s, c)
    @test with_recompute_option(false) do
        reverse_gradient(Reverse, moved_back, X, IDX, S, C)
    end == reverse_gradient(Reverse, ragged, X, IDX, S, C)
end
