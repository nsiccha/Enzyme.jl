using Enzyme, Test
import LLVM

# A vararg callable stores its `Const` arguments in a tuple on the stack and reads them
# back past its safepoint, whose fences keep GVN from forwarding the stores. Whether the
# tuple survives depends on the size of the function: a wide one does, as a compiled
# model's operands do, and a narrow one is optimized away. An overlay of a `Const` array
# and an active one is then built from such a reload, not from the argument itself.
#
# Two things went wrong there, and `dataids` is the control for the first of them:
#
# - `view(overlay, rows)` checks `rows` for aliasing through `unalias`, `mightalias` and
#   Base's default `dataids(A) = (UInt(objectid(A)),)`. The `objectid` runtime call took
#   the overlay's slot by reference, which the constant-field proof read as an escape.
# - A reload that reaches active memory was not recognized as `Const`-argument data, so
#   a read the proof does not cover raised `EnzymeRuntimeActivityError` in a large
#   function where a small one makes a zero-filled copy.
struct SlotReloadOverlay{B, A, S} <: AbstractVector{Float64}
    base::B
    active::A
    slots::S
end
Base.size(p::SlotReloadOverlay) = size(p.base)
Base.IndexStyle(::Type{<:SlotReloadOverlay}) = IndexLinear()
@inline function Base.getindex(p::SlotReloadOverlay, i::Int)
    @boundscheck checkbounds(p.base, i)
    s = @inbounds p.slots[i]
    return iszero(s) ? @inbounds(p.base[i]) : @inbounds(p.active[s])
end

# A mutable `Const` wrapper: its `dataids` hashes a pointer the overlay holds.
mutable struct SlotReloadBox <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(m::SlotReloadBox) = size(m.data)
Base.IndexStyle(::Type{SlotReloadBox}) = IndexLinear()
@inline Base.getindex(m::SlotReloadBox, i::Int) = m.data[i]

const SLOT_RELOAD_FIELDS = 96
@eval struct SlotReloadPlan
    op::SubArray{Int, 1, Vector{Int}, Tuple{Vector{Int}}, false}
    slots::Vector{Int}
    small::Vector{Int}
    pick::Vector{Int}
    $([:($(Symbol(:r, i))::Vector{Int}) for i in 1:SLOT_RELOAD_FIELDS]...)
end

@noinline slot_reload_total(v) = (
    s = 0.0; for x in v
        s += x * x
    end; s
)
@noinline slot_reload_reads(I, rows) = I == rows
@noinline slot_reload_base(p) = p.base

struct SlotReloadCall{K} end
macro slot_reload(K, body)
    return esc(
        quote
            @noinline function (::SlotReloadCall{$(QuoteNode(K))})(u, contexts::Vararg{Any, N}) where {N}
                base = getfield(contexts, 1)
                plan = getfield(contexts, 2)
                slot_reload_reads(plan.op, plan.r1) && return 0.0
                p = SlotReloadOverlay(base, u, plan.slots)
                $body
            end
        end
    )
end

# Reads the proof covers: no error and no copy of `base`.
@slot_reload view_call slot_reload_total(view(p, plan.r1))
@slot_reload view_inline begin
    v = view(p, plan.r1)
    t = 0.0
    @inbounds for i in eachindex(v)
        t += v[i] * v[i]
    end
    t
end
# `dataids` of a view reaches its parent's.
@slot_reload view_of_view slot_reload_total(view(view(p, 1:length(p)), plan.r1))
# The hash is an integer to its uses, whatever they do with it.
@slot_reload hash_branch begin
    t = slot_reload_total(view(p, plan.r1))
    iseven(objectid(p)) ? t : t + 0.0
end
@slot_reload hash_stored begin
    ids = Vector{UInt}(undef, 1)
    ids[1] = objectid(p)
    t = slot_reload_total(view(p, plan.r1))
    iseven(ids[1]) ? t : t + 0.0
end

# Reads the proof does not cover: a zero-filled copy of `base`, as in a small function.
@slot_reload dynamic Base.inferencebarrier(slot_reload_total)(view(p, plan.r1))
@slot_reload returned slot_reload_total(view(p, plan.r1)) + 0.0 * sum(slot_reload_base(p))
@slot_reload matmul sum(abs2, reshape(view(p, plan.r1), :, 1) * ones(1, 1))
@noinline function (::SlotReloadCall{:boxed})(u, contexts::Vararg{Any, N}) where {N}
    base = getfield(contexts, 1)
    plan = getfield(contexts, 2)
    slot_reload_reads(plan.op, plan.r1) && return 0.0
    p = SlotReloadOverlay(SlotReloadBox(base), u, plan.slots)
    return slot_reload_total(view(p, plan.r1)) + 0.0 * slot_reload_total(view(p.base, plan.r2))
end
# A reload merged with active data: which one the field holds is known only at run time.
@noinline function (::SlotReloadCall{:merged})(u, contexts::Vararg{Any, N}) where {N}
    base = getfield(contexts, 1)
    plan = getfield(contexts, 2)
    slot_reload_reads(plan.op, plan.r1) && return 0.0
    p = SlotReloadOverlay(plan.pick[1] > 0 ? u : base, u, plan.slots)
    return slot_reload_total(view(p, plan.small))
end

function slot_reload_data(n, pick = 1)
    base = collect(0.25 .* (1:n))
    slots = zeros(Int, n)
    for (k, i) in enumerate(5:5:40)
        slots[i] = k
    end
    rows(i) = [mod1(3i + j, 8) * 5 - (j % 5) for j in 1:6]
    plan = SlotReloadPlan(
        view([7, 8, 9], [1, 3]), slots, [1, 2, 5], [pick], (rows(i) for i in 1:SLOT_RELOAD_FIELDS)...,
    )
    return base, plan
end

function slot_reload_differences(c, u, base, plan)
    h = 1.0e-6
    return map(eachindex(u)) do j
        hi, lo = copy(u), copy(u)
        hi[j] += h
        lo[j] -= h
        (c(hi, base, plan) - c(lo, base, plan)) / (2h)
    end
end

function slot_reload_reverse!(du, c, u, base, plan)
    fill!(du, 0)
    autodiff(Reverse, Const(c), Active, Duplicated(u, du), Const(base), Const(plan))
    return du
end

# The gradient and the primal value from split mode.
function slot_reload_split(c, u, base, plan)
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(c)}, Active, Duplicated{typeof(u)},
        Const{typeof(base)}, Const{typeof(plan)},
    )
    ds = zero(u)
    tape, prim, _ = fwd(Const(c), Duplicated(u, ds), Const(base), Const(plan))
    rev(Const(c), Duplicated(u, ds), Const(base), Const(plan), 1.0, tape)
    return ds, prim
end

# Julia 1.12 keeps the tuple's pointers in a separate roots slot. Where an augmented
# pass (split mode, or the one a dynamic call compiles for its callee) meets Base's
# `unalias` merging a reloaded row vector with its copy, the mixed-activity handler
# cannot read the phi's type and raises EnzymeRuntimeActivityError (Enzyme todo
# 0qtmgmj). Julia 1.10 and 1.13 differentiate every shape here, and the reads the
# proof covers with no copy.
const SLOT_RELOAD_ROOTS_SLOT = v"1.12" <= VERSION < v"1.13"

@testset "Const data read back from a stack tuple beside active data" begin
    u = collect(0.5:0.5:4.0)
    @testset "the proof covers an overlay `view` hashes: $K" for K in (:view_call, :view_inline, :view_of_view, :hash_branch, :hash_stored)
        c = SlotReloadCall{K}()
        base, plan = slot_reload_data(40)
        want = zero(u)
        autodiff(set_runtime_activity(Reverse), Const(c), Active, Duplicated(u, want), Const(base), Const(plan))
        @test any(!iszero, want)
        du = zero(u)
        @test slot_reload_reverse!(du, c, u, base, plan) == want
        @test du ≈ slot_reload_differences(c, u, base, plan) atol = 1.0e-7
        v = collect(range(0.5, 1.5; length = length(u)))
        @test autodiff(Forward, Const(c), Duplicated(u, v), Const(base), Const(plan))[1] ≈ sum(want .* v)
        d1, d2 = zero(u), zero(u)
        autodiff(Reverse, Const(c), Active, BatchDuplicated(u, (d1, d2)), Const(base), Const(plan))
        @test d1 == want && d2 == want
        if SLOT_RELOAD_ROOTS_SLOT && K in (:view_call, :hash_branch, :hash_stored)
            @test_broken slot_reload_split(c, u, base, plan) == (want, c(u, base, plan))
        else
            @test slot_reload_split(c, u, base, plan) == (want, c(u, base, plan))
        end
        # A compact read costs what its selection costs, whatever the size of `base`.
        big, bigplan = slot_reload_data(40_000)
        g() = slot_reload_reverse!(du, c, u, big, bigplan)
        g()
        if SLOT_RELOAD_ROOTS_SLOT && K !== :view_inline
            # Julia 1.12: the constant-field proof does not reach the roots slot, so the
            # read gets a zero-filled copy of `base`, once per call.
            @test_broken (@allocated g()) < sizeof(big) ÷ 10
        else
            @test (@allocated g()) < sizeof(big) ÷ 10
        end
        @test du == want
        @test base == 0.25 .* (1:40) && big == 0.25 .* (1:40_000)
    end
    @testset "other reads get a zero-filled copy, not an error: $K" for K in (:dynamic, :returned, :matmul, :boxed, :merged)
        c = SlotReloadCall{K}()
        base, plan = slot_reload_data(40)
        want = zero(u)
        autodiff(set_runtime_activity(Reverse), Const(c), Active, Duplicated(u, want), Const(base), Const(plan))
        @test any(!iszero, want)
        du = zero(u)
        if SLOT_RELOAD_ROOTS_SLOT && K === :dynamic
            @test_broken slot_reload_reverse!(du, c, u, base, plan) == want
        else
            @test slot_reload_reverse!(du, c, u, base, plan) == want
            @test du ≈ slot_reload_differences(c, u, base, plan) atol = 1.0e-7
            big, bigplan = slot_reload_data(40_000)
            g() = slot_reload_reverse!(du, c, u, big, bigplan)
            g()
            # A copy per overlay built, not one per read.
            @test (@allocated g()) < 3 * sizeof(big)
            @test big == 0.25 .* (1:40_000)
        end
        @test base == 0.25 .* (1:40)
    end
    @testset "a merge on its constant side leaves the `Const` array alone" begin
        c = SlotReloadCall{:merged}()
        base, plan = slot_reload_data(40, 0)
        want = zero(u)
        autodiff(set_runtime_activity(Reverse), Const(c), Active, Duplicated(u, want), Const(base), Const(plan))
        du = zero(u)
        @test slot_reload_reverse!(du, c, u, base, plan) == want
        @test du ≈ slot_reload_differences(c, u, base, plan) atol = 1.0e-7
        @test base == 0.25 .* (1:40)
    end
end

# `objectid` of a struct held on the stack reads the struct through its pointer. When it
# is the last thing that reads the slot, a call declared to read nothing lost the stores
# into the slot, and the runtime followed the pointers it found there: a segfault.
struct SlotReloadPair{A, B}
    a::A
    b::B
end
function hash_after_call(u, c)
    p = SlotReloadPair(c, u)
    t = slot_reload_total(p.b)
    return iseven(objectid(p)) ? t : t + 0.0
end
# The reverse pass needs the hash here: it picks the derivative.
function hash_picks_derivative(u, c)
    p = SlotReloadPair(c, u)
    t = slot_reload_total(p.b)
    return iseven(objectid(p)) ? 2.0 * t : 3.0 * t
end
# Each iteration hashes another struct, and the hash decides the derivative: a reverse
# pass that recomputes it sees the slot as the last iteration left it.
slot_reload_factor(h) = (k = h % 3; k == 0 ? 2.0 : k == 1 ? 3.0 : 5.0)
function hash_per_iteration(u, cs)
    t = 0.0
    for i in eachindex(u)
        p = SlotReloadPair(cs[i], u)
        t += slot_reload_factor(objectid(p)) * u[i] * u[i]
    end
    return t
end
function hash_kept(u, c, seen)
    p = SlotReloadPair(c, u)
    h = objectid(p)
    push!(seen, h)
    return slot_reload_total(p.b)
end

@testset "`objectid` of a struct on the stack under Reverse" begin
    u = [1.0, 2.0, 3.0]
    c = [4.0, 5.0]
    expected = objectid(SlotReloadPair(c, u))
    @testset "$(nameof(f))" for (f, scale) in (
            (hash_after_call, 1.0),
            (hash_picks_derivative, iseven(expected) ? 2.0 : 3.0),
        )
        du = zero(u)
        autodiff(Reverse, Const(f), Active, Duplicated(u, du), Const(c))
        @test du == 2 * scale .* u
        du = zero(u)
        autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(u, du), Const(c))
        @test du == 2 * scale .* u
        @test only(autodiff(Forward, Const(f), Duplicated(u, [1.0, 0.0, 0.0]), Const(c))) == 2 * scale * u[1]
    end
    @testset "the reverse pass sees each iteration's hash" begin
        v = collect(1.0:6.0)
        # Different arrays hash differently, so the factors differ.
        cs = [fill(Float64(i), 2) for i in 1:12]
        factors = [slot_reload_factor(objectid(SlotReloadPair(cs[i], v))) for i in eachindex(v)]
        for mode in (Reverse, set_runtime_activity(Reverse))
            dv = zero(v)
            autodiff(mode, Const(hash_per_iteration), Active, Duplicated(v, dv), Const(cs))
            @test dv == 2 .* factors .* v
        end
    end
    @testset "the primal sees the hash Julia computes" begin
        seen = UInt[]
        du = zero(u)
        autodiff(Reverse, Const(hash_kept), Active, Duplicated(u, du), Const(c), Const(seen))
        @test du == 2 .* u
        @test !isempty(seen) && all(==(expected), seen)
    end
end

const SLOT_RELOAD_IR_STRUCT = "{ {} addrspace(10)*, {} addrspace(10)* }"
function slot_reload_ir(uses)
    return """
    source_filename = "start"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
    target triple = "x86_64-linux-gnu"

    declare i64 @ijl_object_id_({} addrspace(10)*, i8 addrspace(11)*)
    declare i64 @opaque_reader(i8 addrspace(11)*)

    define double @f({} addrspace(10)* %base, {} addrspace(10)* %active) {
    top:
      %slot = alloca $SLOT_RELOAD_IR_STRUCT, align 8
      %f0 = getelementptr inbounds $SLOT_RELOAD_IR_STRUCT, $SLOT_RELOAD_IR_STRUCT* %slot, i64 0, i32 0
      %f1 = getelementptr inbounds $SLOT_RELOAD_IR_STRUCT, $SLOT_RELOAD_IR_STRUCT* %slot, i64 0, i32 1
      store {} addrspace(10)* %base, {} addrspace(10)** %f0, align 8
      store {} addrspace(10)* %active, {} addrspace(10)** %f1, align 8
      %c = addrspacecast $SLOT_RELOAD_IR_STRUCT* %slot to i8 addrspace(11)*
      %reload = load {} addrspace(10)*, {} addrspace(10)** %f0, align 8
      %other = load {} addrspace(10)*, {} addrspace(10)** %f1, align 8
      %data = addrspacecast {} addrspace(10)* %reload to double addrspace(11)*
      %bytes = addrspacecast {} addrspace(10)* %reload to i8 addrspace(11)*
      %activedata = addrspacecast {} addrspace(10)* %other to double addrspace(11)*
      $uses
      %v = load double, double addrspace(11)* %data, align 8
      %w = load double, double addrspace(11)* %activedata, align 8
      %r = fadd double %v, %w
      ret double %r
    }
    """
end

# Whether the constant-field proof tags the field of `%base` and leaves `%active`'s alone.
function slot_reload_proof(uses)
    return LLVM.Context() do ctx
        mod = parse(LLVM.Module, slot_reload_ir(uses))
        f = LLVM.functions(mod)["f"]
        activity = Enzyme.API.CDIFFE_TYPE[Enzyme.API.DFT_CONSTANT, Enzyme.API.DFT_DUP_ARG]
        Enzyme.Compiler.mark_constant_fields!(mod, f, activity)
        tags = Dict{String, Bool}()
        for inst in LLVM.instructions(first(LLVM.blocks(f)))
            nm = LLVM.name(inst)
            if nm in ("reload", "other")
                tags[nm] = Enzyme.Compiler.is_constant_field_load(inst)
            elseif inst isa LLVM.StoreInst
                stored = LLVM.name(LLVM.operands(inst)[1])
                tags[stored] = haskey(LLVM.metadata(inst), Enzyme.Compiler.CONSTANT_FIELD_STORE)
            end
        end
        tags
    end
end

@testset "The constant-field proof and `objectid`" begin
    constant = Dict("reload" => true, "base" => true, "other" => false, "active" => false)
    unproven = Dict("reload" => false, "base" => false, "other" => false, "active" => false)
    @testset "nothing else reads the slot" begin
        @test slot_reload_proof("") == constant
    end
    @testset "`objectid` of the slot only reads it" begin
        @test slot_reload_proof("%id = call i64 @ijl_object_id_({} addrspace(10)* null, i8 addrspace(11)* %c)") == constant
    end
    @testset "`objectid` of the array the field holds only reads it" begin
        @test slot_reload_proof("%id = call i64 @ijl_object_id_({} addrspace(10)* null, i8 addrspace(11)* %bytes)") == constant
    end
    @testset "any other call on the slot keeps its escape" begin
        @test slot_reload_proof("%id = call i64 @opaque_reader(i8 addrspace(11)* %c)") == unproven
    end
    @testset "any other call on the array keeps the field out of the proof" begin
        @test slot_reload_proof("%id = call i64 @opaque_reader(i8 addrspace(11)* %bytes)") == unproven
    end
end
