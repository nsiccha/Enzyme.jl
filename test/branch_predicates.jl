using Enzyme, Test
using LogExpFunctions: log1pexp

# The reverse of a block reached from one of several predecessors must branch
# to the reverse of the one the forward pass came from. With four or more
# predecessors that choice used to be cached once per loop iteration; in a
# nested loop whose inner trip count varies, each outer iteration then
# allocated (and later freed) its own cache buffer. When the branch conditions
# can be recomputed, the reverse pass now replays them instead.

# LogExpFunctions' log1pexp for Float64, spelled out so that the shape (five
# arms meeting in one block) does not depend on that package.
@inline function fiveway_log1pexp(x::Float64)
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

function fiveway_derivative(x::Float64)
    if x < -745.1332191019412
        return 0.0
    elseif x < -36.7368005696771
        return exp(x)
    elseif x < 18.021826694558577
        return exp(x) / (1 + exp(x))
    elseif x < 33.23111882352963
        return 1 - exp(-x)
    else
        return 1.0
    end
end

# A ragged nest: the inner trip count differs between outer iterations.
function ragged_fiveway(idx, s, c, theta)
    total = 0.0
    for j in eachindex(idx)
        ij, sj, cj = idx[j], s[j], c[j]
        inner = 0.0
        for k in eachindex(ij)
            inner += fiveway_log1pexp(sj * (theta[ij[k]] - cj))
        end
        total += inner
    end
    return total
end

function ragged_log1pexp(idx, s, c, theta)
    total = 0.0
    for j in eachindex(idx)
        ij, sj, cj = idx[j], s[j], c[j]
        inner = 0.0
        for k in eachindex(ij)
            inner -= log1pexp(sj * (theta[ij[k]] - cj))
        end
        total += inner
    end
    return total
end

function ragged_expected(idx, s, c, theta)
    g = zero(theta)
    for j in eachindex(idx), i in idx[j]
        g[i] += s[j] * fiveway_derivative(s[j] * (theta[i] - c[j]))
    end
    return g
end

# An ordinal-logit cell (PosteriorDB's Bones): three arms chosen by integer
# data, each a log1pexp with its own five arms.
@inline function ordinal_cell(g, t, cuts, s)
    K = length(cuts) + 1
    if g == 1
        return -log1pexp(s * (t - cuts[1]))
    elseif g == K
        return -log1pexp(-s * (t - cuts[K - 1]))
    else
        p = s * (t - cuts[g - 1])
        q = s * (t - cuts[g])
        return -log1pexp(-p) - log1pexp(q) + log(-expm1(q - p))
    end
end

function ordinal_ragged(grade, child, cuts, delta, theta)
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

@noinline function ordinal_item(gj, cj, c, s, theta)
    inner = 0.0
    for k in eachindex(gj)
        inner += ordinal_cell(gj[k], theta[cj[k]], c, s)
    end
    return inner
end

function ordinal_outlined(grade, child, cuts, delta, theta)
    total = 0.0
    for j in eachindex(grade)
        total += ordinal_item(grade[j], child[j], cuts[j], delta[j], theta)
    end
    return total
end

# Five arms chosen by an integer read from data in the loop.
@inline function coded_term(code::Int, t::Float64)
    if code == 1
        return t * t
    elseif code == 2
        return sin(t)
    elseif code == 3
        return exp(t)
    elseif code == 4
        return t
    else
        return 2t
    end
end

coded_derivative(code, t) = code == 1 ? 2t : code == 2 ? cos(t) : code == 3 ? exp(t) : code == 4 ? 1.0 : 2.0

function coded_ragged(codes, idx, theta)
    total = 0.0
    for j in eachindex(idx)
        cj, ij = codes[j], idx[j]
        for k in eachindex(ij)
            total += coded_term(cj[k], theta[ij[k]])
        end
    end
    return total
end

# An arm chosen by a read that only some paths make: that condition cannot be
# evaluated on the other paths, so the choice is still cached. The gradient
# must be the same either way.
@inline function flagged_term(t::Float64, flags, i)
    if t < -1
        return exp(t)
    elseif @inbounds flags[i]
        return t * t
    elseif t < 1
        return sin(t)
    else
        return t
    end
end

flagged_derivative(t, flag) = t < -1 ? exp(t) : flag ? 2t : t < 1 ? cos(t) : 1.0

function flagged_ragged(flags, idx, theta)
    total = 0.0
    for j in eachindex(idx)
        ij = idx[j]
        for k in eachindex(ij)
            total += flagged_term(theta[ij[k]], flags, ij[k])
        end
    end
    return total
end

reverse_gradient(f, args, theta) =
    (g = zero(theta); autodiff(Reverse, Const(f), Active, map(Const, args)..., Duplicated(copy(theta), g)); g)

derivative_ir(f, argtypes) = sprint() do io
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, argtypes;
        dump_module = true, second_stage = false, debuginfo = :none
    )
end

# Every arm of fiveway_log1pexp is reached: s * (theta - c) spans all of them.
const RAGGED_IDX = [[1, 2, 3], Int[], [4, 5, 6, 7, 8], [2], [1, 3, 5, 7, 8, 6]]
const RAGGED_S = [1.0, 2.0, 1.5, 0.5, 30.0]
const RAGGED_C = [0.0, 1.0, -0.5, 2.0, 0.1]
const RAGGED_THETA = [-800.0, -40.0, -3.0, 0.2, 5.0, 20.0, 25.0, 40.0]

@testset "Multi-way merges in a ragged nest" begin
    idx, s, c, theta = RAGGED_IDX, RAGGED_S, RAGGED_C, RAGGED_THETA
    expected = ragged_expected(idx, s, c, theta)
    @test reverse_gradient(ragged_fiveway, (idx, s, c), theta) ≈ expected rtol = 1.0e-12
    @test reverse_gradient(ragged_log1pexp, (idx, s, c), theta) ≈ -expected rtol = 1.0e-12

    # Static Reverse and runtime activity agree, and so does a split derivative.
    g = zero(theta)
    autodiff(set_runtime_activity(Reverse), Const(ragged_fiveway), Active, Const(idx), Const(s), Const(c), Duplicated(copy(theta), g))
    @test g ≈ expected rtol = 1.0e-12
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(ragged_fiveway)}, Active,
        Const{typeof(idx)}, Const{typeof(s)}, Const{typeof(c)}, Duplicated{typeof(theta)}
    )
    g = zero(theta)
    args = (Const(idx), Const(s), Const(c), Duplicated(copy(theta), g))
    tape, primal, _ = fwd(Const(ragged_fiveway), args...)
    rev(Const(ragged_fiveway), args..., 1.0, tape)
    @test primal ≈ ragged_fiveway(idx, s, c, theta)
    @test g ≈ expected rtol = 1.0e-12

    # Inner trip counts of zero only, and no outer iterations.
    for empty in ([Int[], Int[]], Vector{Int}[])
        n = length(empty)
        @test iszero(reverse_gradient(ragged_fiveway, (empty, ones(n), zeros(n)), theta))
    end

    # The reverse pass decides which arm each iteration took from the
    # recomputed comparisons, so it allocates no buffer per outer iteration.
    # Julia 1.10's LLVM peels the inner loop's first iteration, which merges
    # three ways after it (empty, one iteration, the loop's exit); a path
    # through the loop keeps that choice cached, in two buffers with one entry
    # per outer iteration, allocated once. Before, it also allocated two
    # buffers inside every outer iteration (eight call sites on Julia 1.10,
    # four on 1.12).
    ir = derivative_ir(
        ragged_fiveway,
        Tuple{Const{Vector{Vector{Int}}}, Const{Vector{Float64}}, Const{Vector{Float64}}, Duplicated{Vector{Float64}}}
    )
    @test count(r"call[^\n]*@malloc\(", ir) <= (VERSION < v"1.11-" ? 2 : 0)
end

@testset "Integer data selecting among merged arms" begin
    grade = [[1, 2, 2], [3, 1], Int[], [2, 4, 1, 3]]
    child = [[1, 2, 3], [2, 4], Int[], [1, 3, 4, 2]]
    cuts = [[0.0], [-1.0, 1.0], [0.5], [-2.0, 0.0, 2.5]]
    delta = [1.3, 0.7, 2.0, 40.0]
    theta = [-1.2, 0.3, 0.9, 2.2]
    g = reverse_gradient(ordinal_ragged, (grade, child, cuts, delta), theta)
    @test g ≈ reverse_gradient(ordinal_outlined, (grade, child, cuts, delta), theta) rtol = 1.0e-12
    h = 1.0e-6
    for i in eachindex(theta)
        tp = copy(theta); tp[i] += h
        tm = copy(theta); tm[i] -= h
        fd = (ordinal_ragged(grade, child, cuts, delta, tp) - ordinal_ragged(grade, child, cuts, delta, tm)) / 2h
        @test g[i] ≈ fd rtol = 1.0e-6 atol = 1.0e-8
    end

    codes = [[1, 2, 3], [4, 5], Int[], [5, 4, 3, 2, 1]]
    idx = [[1, 2, 3], [1, 4], Int[], [4, 3, 2, 1, 2]]
    theta = [0.3, -1.1, 0.8, 2.0]
    expected = zero(theta)
    for j in eachindex(idx), k in eachindex(idx[j])
        expected[idx[j][k]] += coded_derivative(codes[j][k], theta[idx[j][k]])
    end
    @test reverse_gradient(coded_ragged, (codes, idx), theta) ≈ expected rtol = 1.0e-12
end

@testset "A condition read on some paths only" begin
    flags = [true, false, true, false, false, true]
    idx = [[1, 2], [3, 4, 5, 6], Int[], [6, 1]]
    theta = [-2.0, -0.5, 0.4, 0.7, 3.0, 2.5]
    expected = zero(theta)
    for j in eachindex(idx), i in idx[j]
        expected[i] += flagged_derivative(theta[i], flags[i])
    end
    @test reverse_gradient(flagged_ragged, (flags, idx), theta) ≈ expected rtol = 1.0e-12
end
