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

# A value of the outer iteration computed through a branch, read in the inner loop.
# The reverse pass rebuilds it as a branch on the same condition. One arm reads the
# result of a call the reverse pass caches, the other recomputes a call.
@noinline cached_arm(l) = l < 0.1 ? 0.5 - l / 6 : (1 + expm1(-l) / l) / l
@inline branch_weight(l) = l < 0.1 ? 1 - l * cached_arm(l) : -expm1(-l) / l

function through_branch(x, lambda, idx)
    total = 0.0
    for j in eachindex(idx)
        w = branch_weight(lambda[j])
        m = cached_arm(lambda[j])
        ij = idx[j]
        for k in eachindex(ij)
            t = x[ij[k]]
            # arms of the inner loop, each using the merge
            total += t < 0 ? w * t * t + m : w * exp(t) - m * t
        end
    end
    return total
end

const LAMBDA = [0.02, 0.5, 0.05, 0.3, 0.2]

function both_gradients(mode, f, x, l, args...)
    dx = zero(x)
    dl = zero(l)
    autodiff(mode, Const(f), Active, Duplicated(x, dx), Duplicated(l, dl), map(Const, args)...)
    return dx, dl
end

function check_both(f, x, l, args...; atol = 1.0e-6)
    fdx = central_differences(y -> f(y, l, args...), x)
    fdl = central_differences(y -> f(x, y, args...), l)
    for mode in (Reverse, ReverseWithPrimal)
        dx, dl = both_gradients(mode, f, x, l, args...)
        @test dx ≈ fdx atol = atol
        @test dl ≈ fdl atol = atol
    end
    return
end

@testset "A value through a branch: both arms, empty inner loops" begin
    check_both(through_branch, X, LAMBDA, IDX)
    p = [4, 1, 5, 3, 2]
    check_both(through_branch, X, LAMBDA[p], IDX[p])
    # Every outer iteration on one arm, then the other.
    check_both(through_branch, X, fill(0.05, 5), IDX)
    check_both(through_branch, X, fill(0.5, 5), IDX)
    # No inner iterations: the merge is not needed, and its adjoint is zero.
    dx, dl = both_gradients(Reverse, through_branch, X, LAMBDA[1:2], [Int[], Int[]])
    @test dx == zero(X)
    @test dl == zero(LAMBDA[1:2])
end

# An arm that throws when evaluated on the path the primal skipped: the rebuilt
# merge may only run the arm the forward pass ran.
function guarded_arms(x, lambda, idx)
    total = 0.0
    for j in eachindex(idx)
        l = lambda[j]
        w = l > 0 ? sqrt(l) + log(l) : l * l
        for i in idx[j]
            total += w * x[i] * x[i]
        end
    end
    return total
end

@testset "An arm that would throw on the path not taken" begin
    check_both(guarded_arms, [1.0, 2.0, 3.0], [-1.0, 4.0, -0.5], [[1, 2], [2, 3], [3]])
end

# Three paths into one merge, and a merge inside an arm of another.
function three_way_merge(x, lambda, idx)
    total = 0.0
    for j in eachindex(idx)
        l = lambda[j]
        w = l < 0.1 ? exp(l) : l < 0.3 ? l * l : sin(l) / l
        for i in idx[j]
            total += w * x[i] * x[i]
        end
    end
    return total
end

function nested_merge(x, lambda, idx)
    total = 0.0
    for j in eachindex(idx)
        l = lambda[j]
        w = if l < 0.3
            inner = l < 0.1 ? exp(l) : log1p(l)
            inner * cached_arm(l)
        else
            cos(l) / l
        end
        for i in idx[j]
            total += x[i] > 0 ? w * x[i] : x[i] * x[i] / w
        end
    end
    return total
end

@testset "Three paths, and a merge inside an arm" begin
    x = [1.1, -2.3, 3.7, 0.4]
    idx = [[1, 2], [3], Int[], [4, 1, 2], [2, 3]]
    l = [0.05, 0.2, 0.5, 0.25, 0.08]
    check_both(three_way_merge, x, l, idx)
    check_both(nested_merge, x, l, idx)
end

# Memory an arm reads is overwritten after the inner loop, so the reverse pass
# cannot recompute it there, and has to read what it held.
function arm_memory_overwritten(x, s, n)
    total = 0.0
    for j in eachindex(s)
        w = s[j] > 1.0 ? s[j] * s[j] : exp(s[j])
        for k in 1:n
            total += w * x[k] * x[k]
        end
        s[j] = 0.0
    end
    return total
end

@testset "Memory an arm reads overwritten after the inner loop" begin
    x = [1.0, 2.0, 3.0]
    primal_s = [2.0, 0.5, 3.0]
    s = copy(primal_s)
    ds = zero(s)
    dx = zero(x)
    autodiff(Reverse, Const(arm_memory_overwritten), Active, Duplicated(x, dx), Duplicated(s, ds), Const(3))
    @test dx ≈ central_differences(y -> arm_memory_overwritten(y, copy(primal_s), 3), x) atol = 1.0e-5
    @test ds ≈ central_differences(t -> arm_memory_overwritten(x, copy(t), 3), primal_s) atol = 1.0e-5
    @test s == zeros(3)
end

# The merge reads an array of the outer iteration the reverse pass rematerializes.
function arm_rematerialized(x, m)
    total = 0.0
    for j in eachindex(x)
        a = Vector{Float64}(undef, 2)
        a[1] = x[j]
        a[2] = 2 * x[j]
        w = x[j] > 0 ? a[1] * a[2] : a[2]
        for k in 1:m
            total += w * x[k]
        end
    end
    return total
end

@testset "An arm that reads an array the reverse pass rematerializes" begin
    x = [1.0, -2.0, 3.0]
    @test reverse_gradient(Reverse, arm_rematerialized, x, 2) ≈
        central_differences(y -> arm_rematerialized(y, 2), x) atol = 1.0e-6
end

# A merge of arrays, read element by element in the inner loop.
function merged_arrays(x, a, b, idx)
    total = 0.0
    for j in eachindex(idx)
        v = isodd(j) ? a : x
        ij = idx[j]
        for k in eachindex(ij)
            total += v[ij[k]] * x[ij[k]] + b[j]
        end
    end
    return total
end

@testset "A merge of arrays" begin
    x = [1.5, -0.5, 2.0, 0.25]
    a = [0.1, 0.2, 0.3, 0.4]
    b = [1.0, 2.0, 3.0]
    idx = [[1, 2], [3, 4], [2, 2, 1]]
    @test reverse_gradient(Reverse, merged_arrays, x, a, b, idx) ≈
        central_differences(y -> merged_arrays(y, a, b, idx), x) atol = 1.0e-6
end

# A merge of the last value of a sibling loop, read in a later inner loop.
function merged_sibling(x, flags, idx)
    total = 0.0
    for j in eachindex(idx)
        last_value = 1.0
        for i in idx[j]
            last_value = x[i]
        end
        w = flags[j] ? last_value * last_value : exp(x[1])
        for i in idx[j]
            total += w * x[i]
        end
    end
    return total
end

@testset "A merge of a sibling loop's last value" begin
    x = [1.0, 2.0, 3.0]
    flags = [true, false, true]
    idx = [[1, 2], [3], [2, 3, 1]]
    @test reverse_gradient(Reverse, merged_sibling, x, flags, idx) ≈
        central_differences(y -> merged_sibling(y, flags, idx), x) atol = 1.0e-6
end

# A running value is carried from one outer iteration to the next, so it is not
# rebuilt from its arms, and a merge of it reads that iteration's value.
function merged_running(x, lambda, idx)
    total = 0.0
    r = 1.0
    for j in eachindex(idx)
        w = lambda[j] < 0.1 ? r * lambda[j] : r + exp(lambda[j])
        for i in idx[j]
            total += w * x[i] * x[i]
        end
        r *= lambda[j]
    end
    return total
end

@testset "A merge of a value carried across outer iterations" begin
    check_both(merged_running, X, LAMBDA, IDX)
end

@testset "A value through a branch: split mode" begin
    x = copy(X)
    l = copy(LAMBDA)
    args = (Duplicated(x, zero(x)), Duplicated(l, zero(l)), Const(IDX))
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(through_branch)}, Active, map(typeof, args)...,
    )
    tape, primal, _ = fwd(Const(through_branch), args...)
    rev(Const(through_branch), args..., 1.0, tape)
    @test primal ≈ through_branch(X, LAMBDA, IDX)
    @test args[1].dval ≈ central_differences(y -> through_branch(y, LAMBDA, IDX), X) atol = 1.0e-6
    @test args[2].dval ≈ central_differences(y -> through_branch(X, y, IDX), LAMBDA) atol = 1.0e-6
end

@testset "A value through a branch: second derivatives" begin
    v = [1.0, -0.5, 0.25, 2.0]
    gradient_of(x) = first(both_gradients(Reverse, through_branch, x, LAMBDA, IDX))
    expected = central_differences(x -> sum(gradient_of(x) .* v), X)
    forward_over_reverse = only(autodiff(Forward, Const(gradient_of), Duplicated(X, v)))
    @test forward_over_reverse ≈ expected atol = 1.0e-6

    directional(x) = only(autodiff(Forward, Const(through_branch), Duplicated(x, v), Const(LAMBDA), Const(IDX)))
    reverse_over_forward = zero(X)
    autodiff(Reverse, Const(directional), Active, Duplicated(copy(X), reverse_over_forward))
    @test reverse_over_forward ≈ expected atol = 1.0e-6
end

function through_branch!(out, x, lambda, idx)
    out[] = through_branch(x, lambda, idx)
    return nothing
end

@testset "A value through a branch: a batch of directions" begin
    dx1 = zero(X)
    dx2 = zero(X)
    autodiff(
        Reverse, Const(through_branch!), Const,
        BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0))),
        BatchDuplicated(copy(X), (dx1, dx2)),
        Const(LAMBDA), Const(IDX),
    )
    fd = central_differences(y -> through_branch(y, LAMBDA, IDX), X)
    @test dx1 ≈ fd atol = 1.0e-6
    @test dx2 ≈ 2 .* fd atol = 1.0e-6
end

function branch_with_collections(x, lambda, n)
    idx = [collect(1:min(j, length(x))) for j in 1:n]
    total = 0.0
    for j in 1:n
        w = branch_weight(lambda[j])
        ij = idx[j]
        for k in eachindex(ij)
            collect_sometimes(k)
            total += w * exp(x[ij[k]])
        end
    end
    return total
end

@testset "A value through a branch: collections during the loop" begin
    x = [0.1, -0.2, 0.3]
    l = [0.05, 0.5, 0.02, 0.7]
    fdx = central_differences(y -> branch_with_collections(y, l, 4), x)
    fdl = central_differences(y -> branch_with_collections(x, y, 4), l)
    for _ in 1:20
        dx, dl = both_gradients(Reverse, branch_with_collections, x, l, 4)
        @test dx ≈ fdx atol = 1.0e-6
        @test dl ≈ fdl atol = 1.0e-6
    end
end

function with_merges_option(f, enabled)
    Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryMerges, enabled)
    try
        return f()
    finally
        Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryMerges, true)
    end
end

@testset "A merge rebuilt once per entry of the inner loop's reverse" begin
    argtypes = Tuple{Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}, Const{Vector{Vector{Int}}}}
    rebuilds(ir) = count(r"_phirc[\w.]*\"?:", ir)
    once = rebuilds(derivative_ir(through_branch, argtypes))
    in_each_arm = with_merges_option(false) do
        rebuilds(derivative_ir(through_branch, argtypes))
    end
    @test once < in_each_arm

    # Only where the merge is rebuilt differs, not what it computes. A new method,
    # since the option acts when a derivative is compiled.
    moved_back(x, l, idx) = through_branch(x, l, idx)
    @test with_merges_option(false) do
        both_gradients(Reverse, moved_back, X, LAMBDA, IDX)
    end == both_gradients(Reverse, through_branch, X, LAMBDA, IDX)
end
