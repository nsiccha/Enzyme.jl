using Enzyme, Test

# Reads that pick each element from data the function made itself or from active data,
# `iszero(s) ? c[i] : active[s]`, with `c` computed from a `Const` argument.
#
# LLVM folds the two loads into one load through a merge of the two pointers. Activity
# analysis finds `c` constant and `active` active, and a pointer into `c` has no shadow,
# so reaching that merge raised `EnzymeRuntimeActivityError`, only on the path that takes
# `c`. On Julia 1.10 the load also sits behind the bounds check of `result[i] = …`, which
# LLVM sinks it past. See `split_constant_merges!` and `duplicate_merge_tails!` in
# src/compiler/constant_fields.jl. The merge of `Const` data itself is tested in
# test/const_element_store.jl.

@inline function fdm_mix(c, active, slots)
    result = Vector{Float64}(undef, length(slots))
    for i in eachindex(c, slots)
        s = slots[i]
        result[i] = iszero(s) ? c[i] : active[s]
    end
    return result
end

# Julia does not inline it, so `c` reaches it as a parameter.
@noinline function fdm_mix_outlined(c, active, slots)
    result = Vector{Float64}(undef, length(slots))
    for i in eachindex(c, slots)
        s = slots[i]
        result[i] = iszero(s) ? c[i] : active[s]
    end
    return result
end

# Without the check the load stays in the block of the merge.
@inline function fdm_mix_inbounds(c, active, slots)
    result = Vector{Float64}(undef, length(slots))
    @inbounds for i in eachindex(c, slots)
        s = slots[i]
        result[i] = iszero(s) ? c[i] : active[s]
    end
    return result
end

# Two checks of the destination.
@inline function fdm_mix_matrix(c, active, slots)
    result = Matrix{Float64}(undef, length(slots), 2)
    for i in eachindex(c, slots)
        s = slots[i]
        result[i, 2] = iszero(s) ? c[i] : active[s]
        result[i, 1] = 1.0
    end
    return result
end

@inline function fdm_mix_two(c, d, active, slots)
    result = Vector{Float64}(undef, length(slots))
    other = Vector{Float64}(undef, length(slots))
    for i in eachindex(c, slots)
        s = slots[i]
        result[i] = iszero(s) ? c[i] : active[s]
        other[i] = iszero(s) ? d[i] : active[s]
    end
    return result .+ other
end

# `c` is written after each read: the read must see what `c` holds at that point.
@inline function fdm_mix_write(c, active, slots)
    result = Vector{Float64}(undef, length(slots))
    for i in eachindex(c, slots)
        s = slots[i]
        x = iszero(s) ? c[i] : active[s]
        c[i] = 2.0
        result[i] = x
    end
    return result .+ c
end

# The merge is of whole arrays, read through. Its only other use is the error of the
# bounds check, on a path that never returns.
@inline function fdm_pick(c, active, slots)
    t = 0.0
    for i in eachindex(slots)
        v = iszero(slots[i]) ? c : active
        t += v[i] * v[i]
    end
    return t
end

# The merged array reaches a call, so there is nothing to split and it needs a shadow.
@inline fdm_pick_sum(c, active, slots) = sum(abs2, iszero(slots[1]) ? c : active)

fdm_sumsq(x) = (t = 0.0; for v in x; t += v * v; end; t)
@noinline fdm_gather(base, rows) = base[rows]
@noinline fdm_same(x) = x
@noinline function fdm_filled(base)
    c = similar(base)
    for i in eachindex(c)
        c[i] = 2 * base[i]
    end
    return c
end

# Each takes the active vector, the `Const` base, the slots and the rows.
fdm_gathered(a, b, s, r) = fdm_sumsq(fdm_mix(b[r], a, s))
fdm_copied(a, b, s, r) = fdm_sumsq(fdm_mix(copy(b), a, s))
fdm_broadcast(a, b, s, r) = fdm_sumsq(fdm_mix(2 .* b, a, s))
fdm_zeros(a, b, s, r) = fdm_sumsq(fdm_mix(zeros(length(s)), a, s))
fdm_helper(a, b, s, r) = fdm_sumsq(fdm_mix(fdm_gather(b, r), a, s))
fdm_filled_loop(a, b, s, r) = fdm_sumsq(fdm_mix(fdm_filled(b), a, s))
fdm_outlined(a, b, s, r) = fdm_sumsq(fdm_mix_outlined(copy(b), a, s))
fdm_inbounds(a, b, s, r) = fdm_sumsq(fdm_mix_inbounds(b[r], a, s))
fdm_matrix(a, b, s, r) = fdm_sumsq(fdm_mix_matrix(b[r], a, s))
fdm_two(a, b, s, r) = fdm_sumsq(fdm_mix_two(b[r], copy(b), a, s))
fdm_written_late(a, b, s, r) = fdm_sumsq(fdm_mix_write(copy(b), a, s))
fdm_pick_copy(a, b, s, r) = fdm_pick(copy(b), a, s)
fdm_pick_gather(a, b, s, r) = fdm_pick(b[r], a, s)
fdm_pick_sum_copy(a, b, s, r) = fdm_pick_sum(copy(b), a, s)
fdm_pick_sum_gather(a, b, s, r) = fdm_pick_sum(b[r], a, s)

# What the function made holds derivatives: a side that was split must keep them.
fdm_active_fresh(a, b, s, r) = fdm_sumsq(fdm_mix(2 .* a, a, s))
fdm_active_copy(a, b, s, r) = fdm_sumsq(fdm_mix(copy(a), a, s))
fdm_active_other_side(a, b, s, r) = fdm_sumsq(fdm_mix(b[r], 3 .* a, s))
fdm_active_call(a, b, s, r) = fdm_sumsq(fdm_mix(fdm_same(a), a, s))
fdm_active_stored(a, b, s, r) = (c = copy(b); c[1] = a[1]; fdm_sumsq(fdm_mix(c, a, s)))
fdm_active_stored_late(a, b, s, r) = (c = copy(b); x = fdm_mix(c, a, s); c[2] = a[2]; fdm_sumsq(x) + c[2])
fdm_active_pick(a, b, s, r) = fdm_pick(2 .* a, a, s)
fdm_active_pick_sum(a, b, s, r) = fdm_pick_sum(2 .* a, a, s)

const FDM_SHAPES = (
    fdm_gathered, fdm_copied, fdm_broadcast, fdm_zeros, fdm_helper, fdm_filled_loop,
    fdm_outlined, fdm_inbounds, fdm_matrix, fdm_two, fdm_written_late, fdm_pick_copy,
    fdm_pick_gather, fdm_pick_sum_copy, fdm_pick_sum_gather, fdm_active_fresh,
    fdm_active_copy, fdm_active_other_side, fdm_active_call, fdm_active_stored,
    fdm_active_stored_late, fdm_active_pick, fdm_active_pick_sum,
)

function fdm_gradient(mode, f, a, b, s, r)
    d = zero(a)
    autodiff(mode, f, Active, Duplicated(copy(a), d), Const(b), Const(s), Const(r))
    return d
end

function fdm_forward(f, a, b, s, r)
    return [
        only(autodiff(Forward, f, Duplicated(copy(a), Float64.(eachindex(a) .== i)), Const(b), Const(s), Const(r)))
            for i in eachindex(a)
    ]
end

function fdm_batch(f, a, b, s, r)
    d1, d2 = zero(a), zero(a)
    autodiff(Reverse, f, Active, BatchDuplicated(copy(a), (d1, d2)), Const(b), Const(s), Const(r))
    @test d1 == d2
    return d1
end

function fdm_split(f, a, b, s, r)
    args = (Duplicated(copy(a), zero(a)), Const(b), Const(s), Const(r))
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, map(typeof, args)...)
    tape, primal, _ = fwd(Const(f), args...)
    @test primal ≈ f(a, b, s, r)
    rev(Const(f), args..., 1.0, tape)
    return args[1].dval
end

fdm_static_gradient(a, f, b, s, r) = fdm_gradient(Reverse, f, a, b, s, r)
fdm_fwd_rev(f, a, v, b, s, r) =
    only(autodiff(Forward, Const(fdm_static_gradient), Duplicated(copy(a), v), Const(f), Const(b), Const(s), Const(r)))

fdm_directional(a, v, f, b, s, r) =
    only(autodiff(Forward, f, Duplicated(a, v), Const(b), Const(s), Const(r)))
function fdm_rev_fwd(f, a, v, b, s, r)
    d = zero(a)
    autodiff(Reverse, fdm_directional, Active, Duplicated(copy(a), d), Const(v), Const(f), Const(b), Const(s), Const(r))
    return d
end

fdm_central(g, a, h) = (g(a .+ h) .- g(a .- h)) ./ 2
fdm_runtime_gradient(a, fargs) = fdm_gradient(set_runtime_activity(Reverse), fargs[1], a, fargs[2]...)

@testset "Reads that pick data the function made or active data" begin
    base = [1.0, 2.0, 3.0, 4.0]
    rows = [4, 3, 2, 1]
    a = [10.0, 20.0, 30.0, 40.0]
    v = [1.0, -0.5, 2.0, 0.25]
    h = 1.0e-6
    RA = set_runtime_activity(Reverse)
    @testset "$(nameof(f)), slots $(Int.(slots))" for f in FDM_SHAPES, slots in (Int32[1, 2, 3, 4], Int32[1, 0, 2, 0], Int32[0, 0, 0, 0])
        args = (base, slots, rows)
        want = [
            (f(a .+ h .* (eachindex(a) .== i), args...) - f(a .- h .* (eachindex(a) .== i), args...)) / (2h)
                for i in eachindex(a)
        ]
        got = fdm_gradient(Reverse, f, a, args...)
        @test got ≈ want rtol = 1.0e-6 atol = 1.0e-5
        @test got == fdm_gradient(RA, f, a, args...)
        @test fdm_forward(f, a, args...) ≈ got
        @test fdm_batch(f, a, args...) == got
        @test fdm_split(f, a, args...) ≈ got
        hv = fdm_central(Base.Fix2(fdm_runtime_gradient, (f, args)), a, h .* v) ./ h
        @test fdm_fwd_rev(f, a, v, args...) ≈ hv rtol = 1.0e-5 atol = 1.0e-4
        @test fdm_rev_fwd(f, a, v, args...) ≈ hv rtol = 1.0e-5 atol = 1.0e-4
        @test base == [1.0, 2.0, 3.0, 4.0]
    end
end

# The load is not moved above a branch that follows the merge.
@inline function fdm_short_result(c, active, slots)
    result = Vector{Float64}(undef, 2)
    for i in eachindex(c, slots)
        s = slots[i]
        result[i] = iszero(s) ? c[i] : active[s]
    end
    return result
end
fdm_short(a, b, s, r) = fdm_sumsq(fdm_short_result(copy(b), a, s))

# `c` is shorter than `active`, so reading it at `i` is valid only behind the test.
@inline function fdm_guarded(c, active, slots)
    t = 0.0
    for i in eachindex(slots)
        v = iszero(slots[i]) ? c : active
        if i <= length(v)
            t += v[i] * v[i]
        end
    end
    return t
end
fdm_guarded_short(a, b, s, r) = fdm_guarded(b[r], a, s)

@testset "Merges behind branches keep the order of the program" begin
    base = [1.0, 2.0, 3.0, 4.0]
    a = [10.0, 20.0, 30.0, 40.0]
    slots = Int32[1, 0, 2, 0]
    d = zero(a)
    @test_throws BoundsError autodiff(Reverse, fdm_short, Active, Duplicated(copy(a), d), Const(base), Const(slots), Const([1, 2, 3, 4]))
    @test_throws BoundsError autodiff(Forward, fdm_short, Duplicated(copy(a), ones(4)), Const(base), Const(slots), Const([1, 2, 3, 4]))

    # 1 and 2 read `c`, 3 and 4 read past its end through the test, or read `active`.
    for slots in (Int32[0, 0, 0, 0], Int32[1, 0, 3, 0], Int32[0, 2, 3, 4])
        got = fdm_gradient(Reverse, fdm_guarded_short, a, base, slots, [1, 2])
        @test got == fdm_gradient(set_runtime_activity(Reverse), fdm_guarded_short, a, base, slots, [1, 2])
        @test got == [iszero(slots[i]) ? 0.0 : 2 * a[i] for i in 1:4]
    end
end

# These must cost what runtime activity costs: no copy of the data per read.
@testset "No copy of the data per read" begin
    n = 10_000
    base = collect(1.0:n)
    rows = collect(1:n)
    slots = Int32[isodd(i) ? i : 0 for i in 1:n]
    a = 10.0 .* base
    da = zero(a)
    RA = set_runtime_activity(Reverse)
    @testset "$(nameof(f))" for f in (fdm_gathered, fdm_copied, fdm_pick_copy, fdm_pick_gather)
        static() = (fill!(da, 0.0); autodiff(Reverse, f, Active, Duplicated(a, da), Const(base), Const(slots), Const(rows)))
        runtime() = (fill!(da, 0.0); autodiff(RA, f, Active, Duplicated(a, da), Const(base), Const(slots), Const(rows)))
        static()
        runtime()
        @test (@allocated static()) <= (@allocated runtime()) + 512
    end
end
