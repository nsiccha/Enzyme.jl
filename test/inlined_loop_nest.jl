using Enzyme, Libdl, Test
using Enzyme: EnzymeRules

# An inner recurrence over a gathered slice writes a fresh output vector, and the outer
# loop over paths then reads that output. Inlined into the outer loop, Reverse used to
# tape more than the same inner loop in a function Julia does not inline:
#
# - each element of the gathered slice and of the reader's `Const` weights, one native
#   tape entry per step. The next path's gather writes only the array it allocates, and
#   the per-load cacheability check did not know that such a call (an
#   `enzyme_LocalReadOnlyOrThrow` callee) cannot overwrite what an earlier load read.
# - the gather's own index values, a GC-allocated tape per path. The store into the
#   output counted as a write the gather might see, because the reader's numbers read
#   back from the output made the output look exposed to the next gather.
# - on Julia 1.10, a second copy of the output per path: the reverse pass rebuilt it
#   (rematerialized it, replaying every store and so recomputing the recurrence) only to
#   read its length, the reader loop's bound. `fold_fresh_array_lengths!` gives that
#   bound as the size the vector was allocated with.

for (name, inl) in ((:ilnest_ar_noinl, false), (:ilnest_ar_inl, true))
    f = :(
        function $name(r, a, s)
            out = similar(r)
            c = a
            for t in eachindex(r)
                c = c + (1 - c) * s * r[t]
                @inbounds out[t] = c
            end
            return out
        end
    )
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end

for (name, inl) in ((:ilnest_wash_noinl, false), (:ilnest_wash_inl, true))
    f = :(
        function $name(r, recover, remaining, s, a0, g0)
            traj = similar(r, NTuple{3, Float64})
            air, ground = a0, g0
            for t in eachindex(r)
                air1 = air + (1 - air) * recover
                deposit = air1 * (-expm1(-s * r[t]))
                air = air1 - deposit
                o1 = 0.9 * ground + 0.5 * deposit
                ground = remaining * ground + 0.9 * deposit
                @inbounds traj[t] = (o1, air, ground)
            end
            return traj
        end
    )
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end

# The live native bytes, read by a custom reverse rule on the objective's result: its
# reverse runs first, while every tape is still allocated. Julia's allocation counter does
# not see native tapes; glibc's mallinfo2 does.
struct ILNestMallInfo
    arena::Csize_t
    ordblks::Csize_t
    smblks::Csize_t
    hblks::Csize_t
    hblkhd::Csize_t
    usmblks::Csize_t
    fsmblks::Csize_t
    uordblks::Csize_t
    fordblks::Csize_t
    keepcost::Csize_t
end

const ILNEST_MALLINFO2 = Ref{Ptr{Cvoid}}(C_NULL)
const ILNEST_LIVE_IN_REVERSE = Ref(0)

function ilnest_live_bytes()
    info = ccall(ILNEST_MALLINFO2[], ILNestMallInfo, ())
    return Int(info.uordblks + info.hblkhd)
end

# `x + 0.0` is x, but LLVM cannot fold it (signed zeros).
@noinline ilnest_probe(x::Float64) = x + 0.0

function EnzymeRules.augmented_primal(
        config, ::Const{typeof(ilnest_probe)}, ::Type{RT}, x::Active
    ) where {RT}
    primal = EnzymeRules.needs_primal(config) ? x.val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function ilnest_record_live_bytes()
    if ILNEST_MALLINFO2[] != C_NULL
        ILNEST_LIVE_IN_REVERSE[] = ilnest_live_bytes()
    end
    return nothing
end

function EnzymeRules.reverse(
        config, ::Const{typeof(ilnest_probe)}, dret::Active, tape, x::Active
    )
    ilnest_record_live_bytes()
    return (dret.val,)
end

for path in (:ilnest_ar_noinl, :ilnest_ar_inl)
    @eval function $(Symbol(:obj_, path))(rain, idx, a, s, w)
        acc = 0.0
        for g in eachindex(idx)
            p = $path(rain[idx[g]], a[g], s[g])
            wg = w[g]
            for t in eachindex(p)
                acc += wg[t] * p[t]
            end
        end
        return ilnest_probe(acc)
    end
end

for path in (:ilnest_wash_noinl, :ilnest_wash_inl)
    @eval function $(Symbol(:obj_, path))(rain, idx, rec, scav, dec, a0, g0, w)
        acc = 0.0
        for g in eachindex(idx)
            traj = $path(rain[idx[g]], -expm1(-rec[g]), exp(-dec[g]), scav[g], a0[g], g0[g])
            wg = w[g]
            for t in eachindex(traj)
                acc += wg[t, 1] * traj[t][1] + wg[t, 2] * traj[t][2] + wg[t, 3] * traj[t][3]
            end
        end
        return ilnest_probe(acc)
    end
end

function ilnest_problem(G, T)
    rain = [0.5 + 0.4 * sin(1.3 * i) for i in 1:(G * T)]
    idx = [collect(((g - 1) * T + 1):(g * T)) for g in 1:G]
    a = [0.2 + 0.1 * cos(g) for g in 1:G]
    s = [0.05 + 0.02 * sin(2g) for g in 1:G]
    rec = [0.1 + 0.05 * cos(3g) for g in 1:G]
    dec = [0.2 + 0.05 * sin(g) for g in 1:G]
    a0 = [0.3 + 0.1 * sin(5g) for g in 1:G]
    g0 = [0.4 + 0.1 * cos(7g) for g in 1:G]
    w1 = [[cos(0.7t + g) for t in 1:T] for g in 1:G]
    w3 = [[sin(0.3t + 0.5k + g) for t in 1:T, k in 1:3] for g in 1:G]
    return (; rain, idx, a, s, rec, dec, a0, g0, w1, w3)
end

# The active arguments of each objective, and the call with them.
ilnest_ar_args(P) = (P.rain, P.a, P.s)
ilnest_ar_call(f, P, (rain, a, s)) = f(rain, P.idx, a, s, P.w1)
ilnest_wash_args(P) = (P.rain, P.rec, P.s, P.dec, P.a0, P.g0)
ilnest_wash_call(f, P, (rain, rec, s, dec, a0, g0)) = f(rain, P.idx, rec, s, dec, a0, g0, P.w3)

function ilnest_ar_grad(f, P, shadows)
    foreach(x -> fill!(x, 0.0), shadows)
    rain, a, s = ilnest_ar_args(P)
    autodiff(
        Reverse, Const(f), Active, Duplicated(rain, shadows[1]), Const(P.idx),
        Duplicated(a, shadows[2]), Duplicated(s, shadows[3]), Const(P.w1)
    )
    return nothing
end

function ilnest_wash_grad(f, P, shadows)
    foreach(x -> fill!(x, 0.0), shadows)
    rain, rec, s, dec, a0, g0 = ilnest_wash_args(P)
    autodiff(
        Reverse, Const(f), Active, Duplicated(rain, shadows[1]), Const(P.idx),
        Duplicated(rec, shadows[2]), Duplicated(s, shadows[3]), Duplicated(dec, shadows[4]),
        Duplicated(a0, shadows[5]), Duplicated(g0, shadows[6]), Const(P.w3)
    )
    return nothing
end

# The directional derivative along v by central differences.
function ilnest_central(call, f, P, args, v; h = 1.0e-6)
    plus = map((x, d) -> x .+ h .* d, args, v)
    minus = map((x, d) -> x .- h .* d, args, v)
    return (call(f, P, plus) - call(f, P, minus)) / (2h)
end

const ILNEST_CASES = (
    (
        "scalar recurrence", obj_ilnest_ar_noinl, obj_ilnest_ar_inl, ilnest_ar_args,
        ilnest_ar_call, ilnest_ar_grad, 1,
    ),
    (
        "two-state recurrence", obj_ilnest_wash_noinl, obj_ilnest_wash_inl, ilnest_wash_args,
        ilnest_wash_call, ilnest_wash_grad, 2,
    ),
)

@testset "Inlined loop nests tape no more than a non-inlined inner loop" begin
    libc = Sys.islinux() ? Libdl.dlopen_e("libc.so.6") : C_NULL
    ILNEST_MALLINFO2[] = libc == C_NULL ? C_NULL : Libdl.dlsym_e(libc, :mallinfo2)

    @testset "$name" for (name, f_noinl, f_inl, args_of, call, grad!, carried) in ILNEST_CASES
        @testset "gradient, G = $G, T = $T" for (G, T) in ((1, 1), (3, 7), (5, 40))
            P = ilnest_problem(G, T)
            args = args_of(P)
            d_noinl = map(zero, args)
            d_inl = map(zero, args)
            grad!(f_noinl, P, d_noinl)
            grad!(f_inl, P, d_inl)
            for (x, y) in zip(d_inl, d_noinl)
                @test x ≈ y rtol = 1.0e-12 atol = 1.0e-14
            end
            v = map(x -> [cos(0.37i) for i in eachindex(x)], args)
            fd = ilnest_central(call, f_inl, P, args, v)
            ad = sum(sum(d .* dv) for (d, dv) in zip(d_inl, v))
            @test ad ≈ fd rtol = 1.0e-6 atol = 1.0e-9
            @test call(f_inl, P, args) ≈ call(f_noinl, P, args) rtol = 1.0e-14
        end

        # Small enough that every array stays in Julia's pools (an object over 2032
        # bytes is malloc'd, which the native measurement below would count).
        G, T = 40, 60
        P = ilnest_problem(G, T)
        shadows = map(zero, args_of(P))
        warm(f) = (grad!(f, P, shadows); grad!(f, P, shadows); @allocated grad!(f, P, shadows))

        # From Julia 1.11 the gather is inlined too, and the reverse pass rebuilds the
        # gathered slice and the output of each path instead of keeping them, while the
        # callee form tapes its fresh input at every step (Enzyme todo `1nw1luu`).
        later_julia = VERSION >= v"1.11-"

        # The gather's index tape and the rebuilt output were Julia allocations, one of
        # each per path (Julia 1.10).
        @testset "Julia allocations" begin
            bytes_noinl = warm(f_noinl)
            bytes_inl = warm(f_inl)
            @test bytes_inl <= bytes_noinl broken = later_julia
        end

        # The inner loop's carried state is one native tape entry per step per carried
        # value; before, the inlined loop also taped the gathered element and the weights
        # (and the callee form the weights) at every step.
        if ILNEST_MALLINFO2[] != C_NULL
            @testset "native tape" begin
                live(f) = begin
                    grad!(f, P, shadows)
                    GC.gc()
                    before = ilnest_live_bytes()
                    grad!(f, P, shadows)
                    ILNEST_LIVE_IN_REVERSE[] - before
                end
                step_bytes = G * T * sizeof(Float64)
                live_noinl = live(f_noinl)
                live_inl = live(f_inl)
                if get(ENV, "ILNEST_REPORT", "") == "1"
                    println(stderr, "ILNEST native $name: noinl=$(live_noinl / step_bytes) inl=$(live_inl / step_bytes) entries per step")
                end
                @test live_noinl < (carried + 0.5) * step_bytes broken = later_julia
                @test live_inl < (carried + 0.5) * step_bytes
            end
        else
            @test_skip ILNEST_MALLINFO2[] != C_NULL
        end
    end
end

# The length of a vector made in the function is its allocated size only while nothing
# can resize it. Each of these resizes or hands it on, so the fold must leave its length
# loads alone; a wrong length reads out of bounds or drops terms.
function ilnest_pushed(x, w)
    out = similar(x)
    for t in eachindex(x)
        @inbounds out[t] = x[t] * x[t]
    end
    push!(out, x[1])
    acc = 0.0
    for t in eachindex(out)
        acc += w[t] * out[t]
    end
    return acc
end

function ilnest_shrunk(x, w)
    out = similar(x)
    for t in eachindex(x)
        @inbounds out[t] = x[t] * x[t]
    end
    resize!(out, length(x) - 1)
    acc = 0.0
    for t in eachindex(out)
        acc += w[t] * out[t]
    end
    return acc
end

@noinline ilnest_grow!(v) = (push!(v, 2.0); nothing)

function ilnest_handed_on(x, w)
    out = similar(x)
    for t in eachindex(x)
        @inbounds out[t] = x[t] * x[t]
    end
    ilnest_grow!(out)
    acc = 0.0
    for t in eachindex(out)
        acc += w[t] * out[t]
    end
    return acc
end

function ilnest_stored(x, w, box)
    out = similar(x)
    for t in eachindex(x)
        @inbounds out[t] = x[t] * x[t]
    end
    box[] = out
    ilnest_grow!(box[])
    acc = 0.0
    for t in eachindex(out)
        acc += w[t] * out[t]
    end
    return acc
end

@testset "Lengths of vectors that are resized or handed on" begin
    x = [0.3, -1.1, 0.8, 1.7, -0.4]
    w = [cos(0.9i) for i in 1:(length(x) + 1)]
    n = length(x)
    # On Julia 1.10, Reverse loses the gradient of the elements written before a `push!`
    # with every build, without the fold too (Enzyme todo `0mtt8gj`). The primal checks
    # the fold.
    pushed_broken = VERSION < v"1.11-"
    for (f, value, grad, broken) in (
            (ilnest_pushed, sum(w[1:n] .* x .^ 2) + w[n + 1] * x[1], 2 .* w[1:n] .* x .+ [w[n + 1]; zeros(n - 1)], pushed_broken),
            (ilnest_shrunk, sum(w[1:(n - 1)] .* x[1:(n - 1)] .^ 2), [2 .* w[1:(n - 1)] .* x[1:(n - 1)]; 0.0], false),
            (ilnest_handed_on, sum(w[1:n] .* x .^ 2) + 2 * w[n + 1], 2 .* w[1:n] .* x, false),
        )
        dx = zero(x)
        _, primal = autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(x, dx), Const(w))
        @test primal ≈ value
        @test dx ≈ grad broken = broken
    end
    box = Ref{Vector{Float64}}()
    dx = zero(x)
    _, primal = autodiff(
        ReverseWithPrimal, Const(ilnest_stored), Active, Duplicated(x, dx), Const(w),
        Duplicated(box, Ref{Vector{Float64}}(zeros(n + 1)))
    )
    @test primal ≈ sum(w[1:n] .* x .^ 2) + 2 * w[n + 1]
    @test dx ≈ 2 .* w[1:n] .* x
end
