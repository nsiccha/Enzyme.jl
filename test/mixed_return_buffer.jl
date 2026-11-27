using Enzyme, Test
using Enzyme: EnzymeRules

# From Julia 1.12 a function that returns a record of references and numbers, such as
# `Tuple{Matrix{Float64}, Float64}`, writes the numbers into a buffer its caller passes
# (`sret`) and the references into a separate array of return roots. Enzyme calls the
# method a dynamic call reaches through a wrapper that owns both, and libEnzyme moves
# the wrapper's stack slots to the garbage-collected heap, because the reverse pass may
# need them. Nothing in the forward pass reads a reference back from the buffer, so
# libEnzyme creates the buffer's shadow in the reverse pass only. Two things went wrong
# there (snag mixed-activity-t-3d3418f9):
#
# - libEnzyme dropped the zeroing of the buffer itself together with the zeroing of its
#   shadow. The buffer's reference slot, which Julia never writes, then held whatever the
#   allocator left there, and the tape keeps the buffer reachable. The next collection
#   followed that word: a segmentation fault in `gc_mark_obj8`, or "GC error (probable
#   corruption)". Repair `heap_slot_zero_init`, in libEnzyme.
# - In the forward pass the buffer's shadow is libEnzyme's undefined value, `nothing`.
#   The code that calls a custom rule stored the numbers of the rule's shadow through it,
#   eight bytes past `nothing` in the system image, where Julia 1.10 and 1.13 keep the
#   cached box of `Int32(-512)` and Julia 1.12 the empty `Symbol`. Repair
#   `undefined_shadow_stores`, in Enzyme.jl.
#
# The first needs a collection while the call runs and shows as a crash, so it runs in a
# child process. The second is silent and is checked here by reading the word back.

# The word that follows `nothing` in the system image.
word_after_nothing() =
    unsafe_load(Ptr{UInt}(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), nothing)) + sizeof(UInt))

struct Scaled
    basis::Matrix{Float64}
    scale::Float64
end

# A function with a rule of its own. Its shadow carries a number that is not zero, so
# that a store of it in the wrong place shows.
@noinline passthrough(t) = t
marked(a::Array) = zero(a)
marked(::T) where {T <: AbstractFloat} = T(12345)
marked(t::Tuple) = map(marked, t)
marked(t::NamedTuple) = map(marked, t)
marked(s::Scaled) = Scaled(marked(s.basis), marked(s.scale))

const COLLECT = Ref(false)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(passthrough)}, ::Type{RT}, t::Annotation,
    ) where {RT}
    COLLECT[] && GC.gc(true)
    primal = EnzymeRules.needs_primal(config) ? t.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? marked(t.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end
EnzymeRules.reverse(::EnzymeRules.RevConfig, ::Const{typeof(passthrough)}, ::Type, tape, t::Annotation) =
    (nothing,)

basis(t::Tuple) = t[1]
basis(t::NamedTuple) = t.basis
basis(t::Scaled) = t.basis
scale(t::Tuple{Any, Any}) = t[2]
scale(t::Tuple{Any, Any, Any}) = t[2] * sum(t[3])
scale(t::NamedTuple) = t.scale
scale(t::Scaled) = t.scale
scale(t::Tuple{Matrix{Float64}, Matrix{Float64}}) = sum(t[2])

use(x, t) = sum(abs2, basis(t) * x) * scale(t)
# The call is dispatched at run time, so Enzyme differentiates `passthrough` on its own.
dynamic(x, data) = use(x, Base.inferencebarrier(passthrough)(data))
direct(x, data) = use(x, passthrough(data))

const A = [0.5 -1.0 2.0; 1.5 0.25 -0.75; -2.0 1.0 0.5; 0.125 3.0 -1.5]
const X = [0.3, -0.7, 1.1]
expected(data) = 2 .* scale(data) .* (basis(data)' * (basis(data) * X))

const SHAPES = (
    ("tuple", (A, 0.3)),
    ("tuple with a vector", (A, 0.3, [1.0, 2.0])),
    ("named tuple", (; basis = A, scale = 0.3)),
    ("struct", Scaled(A, 0.3)),
    ("arrays only", (A, [0.1 0.2])),
)
const MODES = (
    ("Reverse", Reverse),
    ("ReverseWithPrimal", ReverseWithPrimal),
)

function gradient_of(f::F, mode, data) where {F}
    dx = zero(X)
    autodiff(mode, Const(f), Active, Duplicated(copy(X), dx), Const(data))
    return dx
end

@testset "repairs" begin
    @test Enzyme.has_dogfood_repair(:undefined_shadow_stores)
end

@testset "a rule's shadow of a returned record stays out of the system image" begin
    COLLECT[] = false
    before = word_after_nothing()
    @testset "$fname, $sname, $mname" for (fname, f) in (("dynamic call", dynamic), ("direct call", direct)),
            (sname, data) in SHAPES, (mname, mode) in MODES

        @test gradient_of(f, mode, data) ≈ expected(data)
        @test word_after_nothing() == before
    end
    @test gradient_of(dynamic, set_runtime_activity(Reverse), (A, 0.3)) ≈ expected((A, 0.3))
    @test word_after_nothing() == before
    # What Julia keeps there on 1.10 and 1.13.
    @test Base.inferencebarrier(Int32(-512)) + 512 == 0
end

# The child fills freed cells of the buffer's size with words that are not addresses,
# so that a buffer whose reference slot was never written fails at the first
# collection, not only in some runs.
const COLLECTING_CHILD = raw"""
using Enzyme
using Enzyme: EnzymeRules

struct Scaled
    basis::Matrix{Float64}
    scale::Float64
end
@noinline passthrough(t) = t
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(passthrough)}, ::Type{RT}, t::Annotation,
    ) where {RT}
    GC.gc(true)
    primal = EnzymeRules.needs_primal(config) ? t.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Enzyme.make_zero(t.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end
EnzymeRules.reverse(::EnzymeRules.RevConfig, ::Const{typeof(passthrough)}, ::Type, tape, t::Annotation) =
    (nothing,)

basis(t::Tuple) = t[1]
basis(t::NamedTuple) = t.basis
basis(t::Scaled) = t.basis
scale(t::Tuple{Any, Any}) = t[2]
scale(t::Tuple{Any, Any, Any}) = t[2] * sum(t[3])
scale(t::NamedTuple) = t.scale
scale(t::Scaled) = t.scale
use(x, t) = sum(abs2, basis(t) * x) * scale(t)
dynamic(x, data) = use(x, Base.inferencebarrier(passthrough)(data))
direct(x, data) = use(x, passthrough(data))
# The same without a rule: libEnzyme differentiates the callee itself.
@noinline rebuilt(t::Tuple{Matrix{Float64}, Float64}) = (GC.gc(true); (copy(t[1]), t[2]))
plain(x, data) = use(x, Base.inferencebarrier(rebuilt)(data))

mutable struct Dirt16
    a::UInt
    b::UInt
end
mutable struct Dirt24
    a::UInt
    b::UInt
    c::UInt
end
# Not an address on any supported platform.
const JUNK = 0x0f0f0f0f0f0f0f01
function dirty!(keep::Vector{Any})
    for i in eachindex(keep)
        keep[i] = isodd(i) ? Dirt16(JUNK, JUNK) : Dirt24(JUNK, JUNK, JUNK)
    end
    fill!(keep, nothing)
    GC.gc(true)
    return nothing
end

const A = [0.5 -1.0 2.0; 1.5 0.25 -0.75; -2.0 1.0 0.5; 0.125 3.0 -1.5]
const X = [0.3, -0.7, 1.1]
expected(data) = 2 .* scale(data) .* (basis(data)' * (basis(data) * X))
function gradient_of(f::F, mode, data) where {F}
    dx = zero(X)
    autodiff(mode, Const(f), Active, Duplicated(copy(X), dx), Const(data))
    return dx
end

const keep = Vector{Any}(undef, 20_000)
const RECORDS = (("tuple", (A, 0.3)), ("tuple3", (A, 0.3, [1.0, 2.0])), ("named", (; basis = A, scale = 0.3)), ("struct", Scaled(A, 0.3)))
const CASES = Any[(fname, f, sname, data) for (fname, f) in (("dynamic", dynamic), ("direct", direct)) for (sname, data) in RECORDS]
push!(CASES, ("plain", plain, "tuple", (A, 0.3)))
for (fname, f, sname, data) in CASES, (mname, mode) in (("Reverse", Reverse), ("ReverseWithPrimal", ReverseWithPrimal))
    ok = true
    for round in 1:4
        dirty!(keep)
        ok &= gradient_of(f, mode, data) ≈ expected(data)
        GC.gc(true)
    end
    println("DONE ", fname, " ", sname, " ", mname, " ", ok)
end
"""

@testset "a collection while the call runs finds an initialised return buffer" begin
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $COLLECTING_CHILD`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
    timer = Timer(900) do _
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
    close(timer)
    output = String(take!(out))
    done = filter(startswith("DONE "), split(output, '\n'))
    @test success(proc)
    @test length(done) == 18
    @test all(endswith(" true"), done)
    if !success(proc) || length(done) != 18
        println(stderr, "Child output:\n", first(output, 6000))
    end
end
