using Enzyme, Test

struct BufferedMap!{X}
    x_buffer::Vector{X}
end

function (bc::BufferedMap!)()
    return @inbounds bc.x_buffer[1][1]
end

@testset "Absint struct vector of vector" begin
    f = BufferedMap!([[2.7]])
    df = BufferedMap!([[3.1]])

    @test autodiff(Forward, Duplicated(f, df))[1] ≈ 3.1
end

@testset "Absint sum vector of vector" begin
    a = [[2.7]]
    da = [[3.1]]
    @test autodiff(Forward, sum, Duplicated(a, da))[1] ≈ [3.1]
end

struct MyStruct
    a::Float64
    b::Int
    c::Float64
    d::Int
end

function f_absint_memcpy!(dest, src)
    if length(src) > 0
        dest[1] = src[1]
        for i in 2:length(src)
            dest[i] = src[i]
        end
    end
    nothing
end

@testset "Absint Ptr/GEP memcpy translation" begin
    dest = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]
    ddest = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]
    src = [MyStruct(1.0, 2, 3.0, 4) for _ in 1:3]
    dsrc = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]

    autodiff(Reverse, f_absint_memcpy!, Duplicated(dest, ddest), Duplicated(src, dsrc))
    @test ddest[1].a == 0.0 # Just verifying it runs without EnzymeNoTypeError
end

struct PeriodicTorsion{N, T}
    phases::NTuple{N, T}
    proper::Bool
end
function inject_interaction(inter::PeriodicTorsion{N, T}, params_dic) where {N, T}
    return PeriodicTorsion{N, T}(
        Base.inferencebarrier(ntuple(Returns(params_dic[]), N)),
        inter.proper,
    )
end
function loss(params_dic, inters)
    # Broadcast inject_interaction
    new_inters = inject_interaction.(inters, (params_dic,))
    inter = first(new_inters)
    # Use phases and ks
    return first(inter.phases)
end
@testset "Absint Ptr/GEP of select" begin
    T = Float64
    params_dic = Ref(1.5)
    
    inters = [
        PeriodicTorsion{2, Float64}(
            (2.7, 3.1),
            true
        )
    ]
    types = ["type1"]
    grads_enzyme = make_zero(params_dic)
    
    autodiff(
        set_runtime_activity(Reverse), loss, Active,
        Duplicated(params_dic, grads_enzyme), Const(inters),
    )
    @test grads_enzyme[] ≈ 1.0
end

@testset "Absint load of constantexpr gep HVP" begin
    @inline function mydouble(x)
        y = similar(x)
        for i in eachindex(x, y)
            y[i] = 2 * x[i]
        end
        return y
    end

    @inline function myouterproduct(x, y)
        z = similar(x, length(x), length(y))
        for i in eachindex(x)
            for j in eachindex(y)
                z[i, j] = x[i] * y[j]
            end
        end
        return z
    end

    function arr_to_num(x::AbstractArray)
        a = mydouble(x)
        b = myouterproduct(a, x)
        return b[1]
    end

    function f(x, c)
        c[1] = arr_to_num(x)
        return c[1]
    end

    function g(x, c)
        dx = zero(x)
        dc = zero(c)
        autodiff(Reverse, f, Duplicated(x, dx), Duplicated(c, dc))
        return dx
    end

    function h(x, c, dx_batch)
        dc_batch = map(dx -> zero(c), dx_batch)
        result = autodiff(Forward, g, BatchDuplicated(x, dx_batch), BatchDuplicated(c, dc_batch))
        return result
    end

    x = [3.0, 5.0]
    dx_batch = ([1.0, 0.0], [0.0, 1.0])
    c = [0.0]
    res = h(x, c, dx_batch)
    @test res[1][1] ≈ [4.0, 0.0]
    @test res[1][2] ≈ [0.0, 0.0]
end

struct AbsintNode
    neighbors::Vector{AbsintNode}
end

mutable struct AbsintArchetype
    tables::Vector{UInt32}
    node::AbsintNode
end

struct AbsintTable
    entities::Vector{Int64}
    id::UInt32
end

struct AbsintQuery{S <: Tuple}
    archetypes::Vector{AbsintArchetype}
    tables::Vector{AbsintTable}
    storages::S
end

function absint_iterate(q::AbsintQuery, state::Tuple{Int, Int})
    arch, tab = state
    while arch <= length(q.archetypes)
        @inbounds archetype = q.archetypes[arch]
        if tab == 0
            if isempty(archetype.tables)
                arch += 1
                continue
            end
            tab = 1
        end
        tables = archetype.tables
        while tab <= length(tables)
            @inbounds table = q.tables[Int(tables[tab])]
            @inbounds positions = q.storages[1][table.id]
            return (table.entities, positions), (arch, tab + 1)
        end
        arch += 1
        tab = 0
    end
    return nothing
end

Base.iterate(q::AbsintQuery, state::Tuple{Int, Int}) = absint_iterate(q, state)
Base.iterate(q::AbsintQuery) = Base.iterate(q, (1, 0))

function absint_run_world(args::Vector{Float64})
    @inbounds alpha = args[1]
    archetypes = AbsintArchetype[AbsintArchetype(UInt32[1], AbsintNode(Vector{AbsintNode}(undef, 1)))]
    tables = AbsintTable[AbsintTable(Int64[1], UInt32(1))]
    push!(archetypes, AbsintArchetype(UInt32[2], AbsintNode(Vector{AbsintNode}(undef, 1))))
    push!(tables, AbsintTable(Int64[101], UInt32(2)))
    columns = Vector{Float64}[[alpha], [2 * alpha]]
    q = AbsintQuery(archetypes, tables, (columns,))
    total = zero(alpha)
    for (entities, positions) in q
        @inbounds for pos in positions
            total += pos
        end
    end
    return total
end

# The `node` field makes `AbsintNode` a recursive type, whose typetree is
# necessarily incomplete. Without absint deducing the type of the dynamically
# indexed `q.archetypes[arch]` load, the `UInt32` table index loaded from
# `archetype.tables` is conservatively assumed active, and Enzyme errors out
# trying to differentiate the `shl` computing the byte offset into `q.tables`.
@testset "Absint dynamic index of vector of mutable structs" begin
    @test absint_run_world([1.0]) ≈ 3.0
    # On 1.10 this hits an "undef value upon lcssa" in lookupM, which reproduces
    # without the absint change this test was added alongside. Fixed by
    # https://github.com/EnzymeAD/Enzyme/pull/3114; unskip once that is in a
    # released Enzyme_jll.
    @static if VERSION < v"1.11-"
        @test_skip Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    else
        @test Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    end
end

struct PaddedMixed
    data::Vector{Float64}
    sp::Tuple{Int, Bool}
end

@inline function absint_copy_zip(A)
    dst = similar(A)
    @inbounds for (i, a) in zip(eachindex(dst), A)
        dst[i] = a
    end
    return dst
end

function absint_padded_copies!(out, v)
    src = [PaddedMixed(v .* i, (i, isodd(i))) for i in 1:3]
    out[1] = absint_copy_zip(src)
    out[2] = absint_copy_zip(src)
    s = 0.0
    for m in out[1]
        s += sum(m.data)
    end
    for m in out[2]
        s += sum(m.data)
    end
    return s
end

@testset "Absint negative offset memcpy into array of padded structs" begin
    v = [1.0, 2.0, 3.0]
    dv = zero(v)
    out = Vector{Vector{PaddedMixed}}(undef, 2)
    dout = Vector{Vector{PaddedMixed}}(undef, 2)
    autodiff(Reverse, absint_padded_copies!, Active, Duplicated(out, dout), Duplicated(v, dv))
    @test dv ≈ [12.0, 12.0, 12.0]
    for k in 1:2
        @test [m.sp for m in dout[k]] == [m.sp for m in out[k]]
    end
end

struct AbsintInlineField
    data::Vector{Float64}
    pad::NTuple{22, Float64}
end

mutable struct AbsintWideModel
    pad::NTuple{70, Float64}
    vel::@NamedTuple{a::AbsintInlineField, b::AbsintInlineField, c::AbsintInlineField, d::AbsintInlineField}
    x::Float64
end

@noinline function absint_consume_wide(vel, n)
    s = 0.0
    for i in 1:n
        s += vel.d.pad[1] * vel.a.data[i] * vel.d.data[i]
    end
    return s
end

@noinline function absint_wide_tendencies(m::AbsintWideModel, callbacks)
    vel = m.vel
    s = absint_consume_wide(vel, length(vel.a.data))
    for cb in callbacks
        cb(m)
    end
    return s
end

absint_wide_loss(m) = absint_wide_tendencies(m, Any[])

absint_wide_field(v, p) = AbsintInlineField(v, ntuple(Returns(p), 22))

@testset "Absint memcpy out of an object past the type analysis offset limit" begin
    # `m.vel` is 736 bytes at offset 560, copied out piecewise: each tracked
    # pointer is loaded on its own and the runs of floats between them are
    # memcpy'd. Both the source offset and the fresh stack slot it lands in lie
    # beyond what type analysis keeps for either object, so the copy must be
    # typed from the Julia layout of the source.
    m = AbsintWideModel(
        ntuple(Returns(0.0), 70),
        (
            a = absint_wide_field([1.0, 2.0], 0.0), b = absint_wide_field([0.0, 0.0], 0.0),
            c = absint_wide_field([0.0, 0.0], 0.0), d = absint_wide_field([3.0, 4.0], 2.0),
        ),
        0.0,
    )
    dm = Enzyme.make_zero(m)
    autodiff(Reverse, absint_wide_loss, Active, Duplicated(m, dm))
    @test dm.vel.a.data ≈ [6.0, 8.0]
    @test dm.vel.d.data ≈ [2.0, 4.0]
    @test dm.vel.d.pad[1] ≈ 11.0
end

Base.@noinline absint_undef_barrier(f, args...; kwargs...) = f(args...; kwargs...)

struct AbsintAnyBox
    fn::Any
end
(box::AbsintAnyBox)(a, b) = box.fn(a, b)

struct AbsintTwoPtrs{T, C}
    tunable::T
    caches::C
end

struct AbsintPtrAndInt{U}
    u::U
    i::Int
end

function absint_undef_loss(p, box, s)
    absint_undef_barrier(box, p, (s,))
    return sum(p.tunable)
end

@testset "Absint memcpy from a stack slot that is never written" begin
    # On 1.13 the tracked slots of `p` travel in a separate roots object, so SROA
    # leaves the slice of the argument tuple's data half that would hold them
    # unwritten and unread -- but still memcpys it into the varargs tuple. That
    # copy reads only undefined bytes and has no type to find.
    p = AbsintTwoPtrs([2.0], [0.0])
    dp = Enzyme.make_zero(p)
    s = AbsintPtrAndInt([1.0, 1.0], 0)
    box = AbsintAnyBox((a, b) -> nothing)
    autodiff(
        set_runtime_activity(Reverse), Const(absint_undef_loss), Active,
        Duplicated(p, dp), Const(box), Const(s),
    )
    @test dp.tunable == [1.0]
    @test dp.caches == [0.0]
end

# A closure passed to `autodiff` unannotated must be proven not to write what it
# captures. On Julia 1.11+ a loop reads `xs[i]` at byte offset `8i - 8`, or at
# an `or`/phi offset once the loop is unrolled; unless that load is typed as an
# element of `xs`, the proof follows the loaded float into the result it is
# stored in and rejects the closure.
@inline function absint_closure_scale(xs, g)
    r = similar(xs)
    for i in eachindex(xs)
        r[i] = xs[i] * g
    end
    return r
end

@inline function absint_closure_scan(xs, g)
    r = similar(xs)
    isempty(xs) && return r
    c = xs[1] * g
    r[1] = c
    for i in 2:length(xs)
        c = muladd(xs[i], g, c)
        r[i] = c
    end
    return r
end

@inline function absint_closure_fixed(xs, g)
    r = Vector{Float64}(undef, 8)
    fill!(r, 0.0)
    @inbounds for i in eachindex(xs)
        r[i] = xs[i] * g
    end
    return r
end

@inline function absint_closure_simd(xs, g)
    r = similar(xs)
    @inbounds @simd for i in eachindex(xs)
        r[i] = xs[i] * g
    end
    return r
end

absint_closure_broadcast(xs, g) = xs .* g

@inline function absint_closure_nested(vs, g)
    r = zeros(length(vs))
    for i in eachindex(vs)
        r[i] = vs[i][1] * g
    end
    return r
end

@inline function absint_closure_shift!(ys, g)
    for i in 2:length(ys)
        ys[i] = ys[i - 1] * g
    end
    return sum(ys)
end

struct AbsintClosureHolder
    v::Vector{Float64}
end

absint_closure_field(h, g) = absint_closure_scale(h.v, g)

absint_closure_reverse(kernel, data) =
    autodiff(ReverseWithPrimal, g -> sum(kernel(data, g)), Active, Active(0.7))

absint_closure_forward(kernel, data) =
    autodiff(Forward, g -> sum(kernel(data, g)), Duplicated(0.7, 1.0))[1]

@testset "Absint loop loads from an array captured by a read-only closure" begin
    for n in (1, 5, 64)
        xs = sin.(1:n)
        for (kernel, expected) in (
                (absint_closure_scale, sum(xs)),
                (absint_closure_scan, sum(cumsum(xs))),
                (absint_closure_simd, sum(xs)),
                (absint_closure_broadcast, sum(xs)),
            )
            ((d,), value) = absint_closure_reverse(kernel, xs)
            @test d ≈ expected
            @test value ≈ sum(kernel(xs, 0.7))
            @test absint_closure_forward(kernel, xs) ≈ expected
        end
    end
    # Previously no error was raised here, but the primal and both derivatives
    # were silently 0.0.
    xs = sin.(1:5)
    ((d,), value) = absint_closure_reverse(absint_closure_fixed, xs)
    @test d ≈ sum(xs)
    @test value ≈ 0.7 * sum(xs)
    @test absint_closure_forward(absint_closure_fixed, xs) ≈ sum(xs)

    vs = [sin.(1:3) .* i for i in 1:5]
    @test absint_closure_reverse(absint_closure_nested, vs)[1][1] ≈ sum(v[1] for v in vs)
    h = AbsintClosureHolder(sin.(1:5))
    @test absint_closure_reverse(absint_closure_field, h)[1][1] ≈ sum(h.v)

    # Refused: the closure writes the array it captures, so it must be passed
    # as `Duplicated` or `Const` (Enzyme FAQ, "Activity of temporary storage").
    ys = sin.(1:5)
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(
        Reverse, g -> absint_closure_shift!(ys, g), Active, Active(0.7)
    )
end
