using Enzyme, StaticArrays, Test, Libdl
using Enzyme: EnzymeRules

# A combined Reverse derivative reserves the loop caches of the non-inlined
# calls it makes in loops, and its own bounded caches of numbers, from one tape
# arena per call, released when the derivative returns (repair `tape_arena`).
# Without it, each such cache is one malloc and one free per call or loop
# entry. Each case compares the derivative with the arena and the same code
# compiled with both switches off, and with central differences.

const ARENA = Enzyme.has_dogfood_repair(:tape_arena)

function with_arena(f, on::Bool)
    Enzyme.API.EnzymeSetCLBool(:EnzymeCallerOwnedTapes, on)
    Enzyme.API.EnzymeSetCLBool(:EnzymeArenaCombinedCaches, on)
    try
        return f()
    finally
        Enzyme.API.EnzymeSetCLBool(:EnzymeCallerOwnedTapes, true)
        Enzyme.API.EnzymeSetCLBool(:EnzymeArenaCombinedCaches, true)
    end
end

# A read-only loop over a subject's events, called once per observation and
# not inlined: its reverse needs the event times and amplitudes it read
# (numbers) and which events preceded the observation (Bools), and the caller
# writes a buffer that may alias them, so they are cached on every call.
@inline function arena_response(dt, weights, rates)
    acc = zero(dt)
    for q in 1:length(weights)
        acc += weights[q] * exp(-rates[q] * dt)
    end
    return acc
end
@noinline function arena_superpose(t, taus, lo, hi, amplitudes, weights, rates)
    acc = zero(t)
    for j in lo:hi
        dt = t - taus[j]
        if dt > 0
            acc += amplitudes[j] * arena_response(dt, weights, rates)
        end
    end
    return acc
end
function arena_model(p, buf, times, subjects, taus, offsets, amounts, ys, ::Val)
    w = (p[1], p[2]); r = (exp(p[3]), exp(p[4]))
    amplitudes = amounts .* p[5]
    @inbounds for i in eachindex(times)
        s = subjects[i]
        buf[i] = arena_superpose(times[i], taus, offsets[s] + 1, offsets[s + 1], amplitudes, w, r)
    end
    total = zero(eltype(p))
    @inbounds for i in eachindex(buf)
        total -= (ys[i] - buf[i])^2
    end
    return total
end

# Subjects with no, one and several events, so the callee's ranges are empty,
# singleton and varying.
function arena_data(T, events)
    taus = T[]; offsets = [0]
    for (s, n) in enumerate(events)
        append!(taus, T[0.37 * s + 1.1 * k for k in 1:n]); push!(offsets, offsets[end] + n)
    end
    times = T[]; subjects = Int[]
    for s in eachindex(events), k in 1:3
        push!(times, T(2.5 * k + 0.1 * s)); push!(subjects, s)
    end
    amounts = T[1 + 0.1 * j for j in eachindex(taus)]
    ys = T[0.2 * sin(i) for i in eachindex(times)]
    return (times, subjects, taus, offsets, amounts, ys)
end

function arena_gradient(p, data, tag)
    dp = zero(p); buf = zeros(eltype(p), length(data[1]))
    autodiff(Reverse, arena_model, Active, Duplicated(copy(p), dp), Duplicated(buf, zero(buf)), map(Const, data)..., Const(tag))
    return dp
end
function arena_split_gradient(p, data, tag)
    buf = zeros(eltype(p), length(data[1]))
    args = (Duplicated(copy(p), zero(p)), Duplicated(buf, zero(buf)), map(Const, data)..., Const(tag))
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(arena_model)}, Active, map(typeof, args)...)
    tape, _, _ = fwd(Const(arena_model), args...)
    rev(Const(arena_model), args..., one(eltype(p)), tape)
    return args[1].dval
end
function arena_gradient!(dp, p, data, tag)
    fill!(dp, 0)
    buf = zeros(eltype(p), length(data[1]))
    autodiff(Reverse, arena_model, Active, Duplicated(p, dp), Duplicated(buf, zero(buf)), map(Const, data)..., Const(tag))
    return nothing
end
function arena_hvp(p, v, data, tag)
    dp = zero(p); hv = zero(p)
    autodiff(Forward, arena_gradient!, Const, Duplicated(dp, hv), Duplicated(copy(p), copy(v)), Const(data), Const(tag))
    return hv
end
arena_gradient_sum(p, data, tag) = sum(arena_gradient(p, data, tag))
function arena_reverse_reverse(p, data, tag)
    dp = zero(p)
    autodiff(Reverse, arena_gradient_sum, Active, Duplicated(copy(p), dp), Const(data), Const(tag))
    return dp
end

central(f, p, h) = [(f(p .+ h .* (eachindex(p) .== i)) - f(p .- h .* (eachindex(p) .== i))) / 2h for i in eachindex(p)]

# StanBlocks' scaling-and-squaring matrix exponential: a Taylor loop of fixed
# length and a squaring loop whose trip count depends on the input, each
# caching the nine entries of its matrix per iteration.
function arena_expm(x::SMatrix{3, 3, T}) where {T}
    scale = zero(T)
    for i in 1:3
        row = zero(T)
        for j in 1:3
            row += abs(x[i, j])
        end
        scale = max(scale, row)
    end
    scaled = x; squarings = 0
    while scale > 0.5
        scaled = scaled / 2; scale /= 2; squarings += 1
    end
    result = SMatrix{3, 3, T}(I3); term = result
    for k in 1:24
        term = term * scaled / k
        result = result + term
    end
    for _ in 1:squarings
        result = result * result
    end
    return result
end
const I3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
arena_matrix(p) = SMatrix{3, 3}(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
arena_expm_once(p, ::Val) = sum(arena_expm(arena_matrix(p)))
function arena_expm_many(p, scales, ::Val)
    total = zero(eltype(p))
    for c in scales
        total += sum(arena_expm(c * arena_matrix(p)))
    end
    return total
end

# Keep both call boundaries, as in a propagator that uses the matrix
# exponential on one branch. The intermediate function has no loop of its own.
@noinline function arena_expm_step(x, scale)
    m = Base.@noinline arena_expm(scale * x)
    return m[1] + m[5] + m[9]
end

function arena_nested_expm(p, scales, ::Val)
    total = zero(eltype(p))
    for scale in scales
        total += arena_expm_step(arena_matrix(p), scale)
    end
    return total
end
arena_expm_gradient(p, tag) = (dp = zero(p); autodiff(Reverse, arena_expm_once, Active, Duplicated(copy(p), dp), Const(tag)); dp)
arena_expm_many_gradient(p, scales, tag) = (dp = zero(p); autodiff(Reverse, arena_expm_many, Active, Duplicated(copy(p), dp), Const(scales), Const(tag)); dp)
arena_expm_many_gradient!(dp, p, scales, tag) = (fill!(dp, 0); autodiff(Reverse, arena_expm_many, Active, Duplicated(p, dp), Const(scales), Const(tag)); nothing)
function arena_expm_hvp(p, v, scales, tag)
    dp = zero(p); hv = zero(p)
    autodiff(Forward, arena_expm_many_gradient!, Const, Duplicated(dp, hv), Duplicated(copy(p), copy(v)), Const(scales), Const(tag))
    return hv
end

# The native bytes live when the reverse pass starts, every tape still allocated: a custom
# rule on the objective's result reads glibc's mallinfo2 in its reverse, which runs first.
struct ArenaMallInfo
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
const ARENA_MALLINFO2 = Ref{Ptr{Cvoid}}(C_NULL)
const ARENA_LIVE_IN_REVERSE = Ref(0)
arena_live_bytes() = (info = ccall(ARENA_MALLINFO2[], ArenaMallInfo, ()); Int(info.uordblks + info.hblkhd))
@noinline arena_probe(x::Float64) = x + 0.0
function EnzymeRules.augmented_primal(config, ::Const{typeof(arena_probe)}, ::Type{RT}, x::Active) where {RT}
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? x.val : nothing, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(arena_probe)}, dret::Active, tape, x::Active)
    ARENA_LIVE_IN_REVERSE[] = arena_live_bytes()
    return (dret.val,)
end
arena_probed(p, buf, times, subjects, taus, offsets, amounts, ys, tag::Val) =
    arena_probe(arena_model(p, buf, times, subjects, taus, offsets, amounts, ys, tag))
# Buffers are allocated once, outside the measured gradient.
function arena_tape_bytes(data, tag)
    p = [0.7, 0.3, log(0.5), log(2.0), 1.2]; dp = zero(p)
    buf = zeros(length(data[1])); dbuf = zero(buf)
    run() = (fill!(dp, 0); fill!(dbuf, 0); autodiff(Reverse, arena_probed, Active, Duplicated(p, dp), Duplicated(buf, dbuf), map(Const, data)..., Const(tag)))
    run(); run(); GC.gc()
    before = arena_live_bytes()
    run()
    return ARENA_LIVE_IN_REVERSE[] - before
end

if ARENA
    @testset "Tape arena: a non-inlined callee in a loop" begin
        for (T, tol) in ((Float64, 1.0e-12), (Float32, 1.0f-5)), events in ([0, 1, 4, 0, 7, 2], Int[], fill(3, 20))
            data = arena_data(T, events)
            p = T[0.7, 0.3, log(0.5), log(2.0), 1.2]
            v = T[0.3, -0.2, 0.5, 0.1, -0.4]
            name = Symbol(T, :_, length(events), :_, sum(events))
            tag = Val(name)
            on = arena_gradient(p, data, tag)
            off = with_arena(() -> arena_gradient(p, data, Val(Symbol(:off_, name))), false)
            @test on ≈ off rtol = tol atol = tol
            @test arena_split_gradient(p, data, Val(Symbol(:split_, name))) ≈ on rtol = tol atol = tol
            if T == Float64
                fd = central(x -> arena_model(x, zeros(length(data[1])), data..., Val(:primal)), p, 1.0e-6)
                @test on ≈ fd rtol = 1.0e-6 atol = 1.0e-8
                # Forward over Reverse and Reverse over Reverse differentiate the
                # arena's reservations again.
                hv = arena_hvp(p, v, data, Val(Symbol(:hvp_, name)))
                hv_off = with_arena(() -> arena_hvp(p, v, data, Val(Symbol(:hvp_off_, name))), false)
                @test hv ≈ hv_off rtol = 1.0e-10 atol = 1.0e-12
                hv_fd = (arena_gradient(p .+ 1.0e-6 .* v, data, tag) .- arena_gradient(p .- 1.0e-6 .* v, data, tag)) ./ 2.0e-6
                @test hv ≈ hv_fd rtol = 1.0e-5 atol = 1.0e-7
                rr = arena_reverse_reverse(p, data, Val(Symbol(:rr_, name)))
                rr_off = with_arena(() -> arena_reverse_reverse(p, data, Val(Symbol(:rr_off_, name))), false)
                @test rr ≈ rr_off rtol = 1.0e-10 atol = 1.0e-12
            end
        end
    end

    @testset "Tape arena: a matrix exponential" begin
        p = collect(0.1:0.1:0.9)
        v = collect(range(-0.4, 0.4, length = 9))
        for scales in ([0.5, 1.0, 1.5], [0.1], Float64[])
            once = arena_expm_gradient(p, Val(Symbol(:once_, length(scales))))
            once_off = with_arena(() -> arena_expm_gradient(p, Val(Symbol(:once_off_, length(scales)))), false)
            @test once ≈ once_off rtol = 1.0e-12 atol = 1.0e-12
            @test once ≈ central(x -> arena_expm_once(x, Val(:primal)), p, 1.0e-6) rtol = 1.0e-6 atol = 1.0e-7
            many = arena_expm_many_gradient(p, scales, Val(Symbol(:many_, length(scales))))
            many_off = with_arena(() -> arena_expm_many_gradient(p, scales, Val(Symbol(:many_off_, length(scales)))), false)
            @test many ≈ many_off rtol = 1.0e-12 atol = 1.0e-12
            @test many ≈ central(x -> arena_expm_many(x, scales, Val(:primal)), p, 1.0e-6) rtol = 1.0e-6 atol = 1.0e-7
            hv = arena_expm_hvp(p, v, scales, Val(Symbol(:mhvp_, length(scales))))
            hv_off = with_arena(() -> arena_expm_hvp(p, v, scales, Val(Symbol(:mhvp_off_, length(scales)))), false)
            @test hv ≈ hv_off rtol = 1.0e-10 atol = 1.0e-10
        end
    end

    function arena_derivative_ir(f, argtypes)
        io = IOBuffer()
        Enzyme.Compiler.enzyme_code_llvm(io, f, Active, argtypes; raw = true, dump_module = true, debuginfo = :none)
        return String(take!(io))
    end

    function allocation_sites(f, argtypes)
        ir = arena_derivative_ir(f, argtypes)
        # Calls of malloc outside the arena's own chunk allocation.
        body = replace(ir, r"define [^\n]*@__enzyme_caller_tape_reserve\(.*?\n}\n"s => "")
        return (malloc = count(r"call[^\n]*@malloc\(", body), reserve = count(r"call[^\n]*@__enzyme_caller_tape_reserve\(", ir))
    end

    libc = Sys.islinux() ? Libdl.dlopen_e("libc.so.6") : C_NULL
    ARENA_MALLINFO2[] = libc == C_NULL ? C_NULL : Libdl.dlsym_e(libc, :mallinfo2)
    if ARENA_MALLINFO2[] != C_NULL
        # Chunks double from 256 bytes to 64 KiB, and a request above 4 KiB gets a chunk of
        # its own size, so the arena holds at most twice what a small tape stores and at most
        # one partly used 64 KiB chunk, plus the unused ends of full chunks, beyond a large one.
        @testset "Tape arena: native bytes" begin
            for (events, bound) in ((fill(3, 20), off -> 2off + 64 * 1024), (fill(3, 2000), off -> off + 64 * 1024 + off ÷ 8))
                data = arena_data(Float64, events)
                on = arena_tape_bytes(data, Val(Symbol(:bytes_on_, length(events))))
                off = with_arena(() -> arena_tape_bytes(data, Val(Symbol(:bytes_off_, length(events)))), false)
                @test off > 0
                @test on <= bound(off)
            end
        end
    end

    @testset "Tape arena: allocation sites" begin
        data = arena_data(Float64, [0, 1, 4])
        argtypes = Tuple{Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}, map(d -> Const{typeof(d)}, data)..., Const{Val{:sites}}}
        on = allocation_sites(arena_model, argtypes)
        off = with_arena(() -> allocation_sites(arena_model, argtypes), false)
        @test on.reserve > 0
        @test off.reserve == 0
        @test on.malloc < off.malloc
        expm_types = Tuple{Duplicated{Vector{Float64}}, Const{Val{:expm_sites}}}
        expm_on = allocation_sites(arena_expm_once, expm_types)
        expm_off = with_arena(() -> allocation_sites(arena_expm_once, expm_types), false)
        # Eighteen caches of the Taylor and squaring loops, each one malloc
        # without the arena.
        @test expm_on.malloc <= expm_off.malloc - 17
    end

    @testset "Tape arena through an augmented caller" begin
        p = [0.75, 0.14, -0.22, -0.1, 0.83, 0.03, 0.08, -0.29, 0.58]
        scales = [0.2, 0.8, 2.0]
        tag = Val(:nested_caller)
        dp = zero(p)
        autodiff(Reverse, arena_nested_expm, Active, Duplicated(p, dp), Const(scales), Const(tag))
        @test dp ≈ central(x -> arena_nested_expm(x, scales, tag), p, 1.0e-6) rtol = 1.0e-6 atol = 1.0e-7

        types = Tuple{Duplicated{typeof(p)}, Const{typeof(scales)}, Const{typeof(tag)}}
        leaf = r"^define [^\n]*@augmented_julia_arena_expm_\d+[^\n]*\n(.*?)^}"ms
        on = match(leaf, arena_derivative_ir(arena_nested_expm, types))
        off = with_arena(() -> match(leaf, arena_derivative_ir(arena_nested_expm, types)), false)
        @test on !== nothing && off !== nothing
        if on !== nothing && off !== nothing
            # Count inside the matrix exponential, independently of the outer
            # derivative's cache allocations and the arena's own chunk mallocs.
            alloc = r"call[^\n]*@malloc\("
            @test count(alloc, off.captures[1]) >= 9
            @test count(alloc, on.captures[1]) < count(alloc, off.captures[1])
            @test occursin("@__enzyme_caller_tape_reserve(", on.captures[1])
        end
    end
end
