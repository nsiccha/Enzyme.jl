# A custom reverse rule may reread its argument in the reverse pass when
# `overwritten(config)` reports that the argument was not modified since the
# augmented call. An argument can hold a reference to its data: an immutable
# struct, a tuple or a view is passed by reference to a copy of itself, and that
# copy holds the array it wraps. The flag has to cover the array too, whichever
# way a later write reaches it.
# https://github.com/EnzymeAD/Enzyme.jl/issues/3955
module OverwrittenReferences

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using LinearAlgebra
using Test

struct Wrapped
    values::Vector{Float64}
end
mutable struct MutableWrapped
    values::Vector{Float64}
end
struct Outer
    inner::Wrapped
    count::Int
end
struct Labelled
    values::Vector{Float64}
    name::Symbol
    label::String
end

# The array behind each container.
data(a::Vector{Float64}) = a
data(g::Wrapped) = g.values
data(g::MutableWrapped) = g.values
data(g::Outer) = g.inner.values
data(g::Labelled) = g.values
data(g::Tuple{Vector{Float64}, Int}) = g[1]
data(g::NamedTuple) = g.values
data(g::Base.RefValue{Vector{Float64}}) = g[]
data(g::Vector{Vector{Float64}}) = g[1]
data(g::SubArray) = parent(g)
data(g::Adjoint) = parent(g)
data(g::Symmetric) = parent(g)

weight(g) = @inbounds data(g)[1]

plain(g, x) = weight(g) * x
scaled(g, x) = weight(g) * x

# The flags the rule saw at its most recent augmented call.
const FLAGS = Ref{Any}(nothing)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(scaled)}, ::Type{<:Active}, g::Annotation, x::Active
    )
    FLAGS[] = EnzymeRules.overwritten(config)
    # Save the weight only if it may change before the reverse pass.
    saved = EnzymeRules.overwritten(config)[2] ? weight(g.val) : nothing
    primal = EnzymeRules.needs_primal(config) ? scaled(g.val, x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, saved)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(scaled)}, dy::Active, saved, g::Annotation, x::Active
    )
    w = EnzymeRules.overwritten(config)[2] ? saved : weight(g.val)
    if !(g isa Const)
        data(g.dval)[1] += dy.val * x.val
    end
    return (nothing, dy.val * w)
end

# A callable that holds its data: the rule reads `overwritten(config)[1]`.
struct Scale
    values::Vector{Float64}
end
(s::Scale)(x) = @inbounds s.values[1] * x

const FUNCTION_FLAGS = Ref{Any}(nothing)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        s::Const{Scale}, ::Type{<:Active}, x::Active
    )
    FUNCTION_FLAGS[] = EnzymeRules.overwritten(config)
    saved = EnzymeRules.overwritten(config)[1] ? s.val.values[1] : nothing
    primal = EnzymeRules.needs_primal(config) ? s.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, saved)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        s::Const{Scale}, dy::Active, saved, x::Active
    )
    w = EnzymeRules.overwritten(config)[1] ? saved : s.val.values[1]
    return (dy.val * w,)
end

const shared = [3.0]

# What happens around the rule call. `a` is the array behind `g`, passed again.
unchanged(f, g, a, x) = f(g, x)
function write_before(f, g, a, x)
    a[1] = 3.0
    return f(g, x)
end
function write_argument(f, g, a, x)
    y = f(g, x)
    a[1] = 42.0
    return y
end
function write_global(f, g, a, x)
    y = f(g, x)
    shared[1] = 42.0
    return y
end
function write_through(f, g, a, x)
    y = f(g, x)
    data(g)[1] = 42.0
    return y
end
@noinline call_in_callee(f, g, x) = f(g, x)
function write_after_callee(f, g, a, x)
    y = call_in_callee(f, g, x)
    a[1] = 42.0
    return y
end
function fresh_after(f, g, a, x)
    y = f(g, x)
    out = Vector{Float64}(undef, 2)
    out[1] = y
    out[2] = 2 * y
    return out[1] + 0 * out[2]
end
function write_in_loop(f, g, a, x)
    total = 0.0
    for _ in 1:3
        total += f(g, x)
        a[1] += 1.0
    end
    return total
end

function run(f::F, caller::C, g, a, x) where {F, C}
    return caller(f, g, a, x)
end

function derivative(f, caller, g)
    FLAGS[] = nothing
    result = Enzyme.autodiff(
        Reverse, Const(run), Active, Const(f), Const(caller), Const(g), Const(shared), Active(2.0)
    )
    return result[1][end]
end

apply(s, x) = s(x)

function function_derivative(caller)
    FUNCTION_FLAGS[] = nothing
    result = Enzyme.autodiff(
        Reverse, Const(run), Active, Const(apply), Const(caller), Const(Scale(shared)), Const(shared), Active(2.0)
    )
    return result[1][end]
end

const containers = (
    "Vector" => identity,
    "immutable struct" => Wrapped,
    "mutable struct" => MutableWrapped,
    "struct in a struct" => v -> Outer(Wrapped(v), 1),
    "struct with a Symbol and a String" => v -> Labelled(v, :weights, "weights"),
    "tuple" => v -> (v, 1),
    "named tuple" => v -> (; values = v, count = 1),
    "Ref" => v -> Ref(v),
    "vector of vectors" => v -> [v],
    "view" => v -> view(v, 1:1),
    "adjoint" => v -> v',
    "Symmetric" => v -> Symmetric(reshape(v, 1, 1)),
)

# The callers after which the weight differs from the one the rule call read,
# with the derivative that the weight at each call gives.
const writers = (
    write_argument => 3.0,
    write_global => 3.0,
    write_through => 3.0,
    write_after_callee => 3.0,
    write_in_loop => 3.0 + 4.0 + 5.0,
)
# The callers that leave it alone.
const readers = (unchanged, write_before, fresh_after)

# Whether the loaded libEnzyme reports the writes that follow a call. Without that,
# an argument that holds a reference always counts as overwritten.
const precise = isdefined(Enzyme.API, :reports_subsequent_writes) &&
    Enzyme.API.reports_subsequent_writes()

# A mutable object the function creates, stored into an argument's object before the
# rule call and written directly afterwards. No pointer the rule received points to
# it, yet the rule reads it through the argument.
mutable struct Holder
    box::Base.RefValue{Float64}
end
data(h::Holder) = h.box
weight(h::Holder) = h.box[]
function write_local_box(f, h, a, x)
    box = Ref(3.0)
    h.box = box
    y = f(h, x)
    box[] = 42.0
    return y
end

# An immutable type that refers to itself: nothing behind it can change.
struct Chain
    head::Float64
    tail::Union{Chain, Nothing}
end
# A field that holds a type.
struct Tagged{T}
    kind::Type{T}
    count::Int
end

# A rule that keeps its output by reference unless the flag says the output may be
# overwritten: the output is a preallocated array, passed directly or inside a
# struct or a tuple, and the caller clears part of it afterwards.
exp_into!(A, out) = (data(out) .= exp.(A); out)
data(g::Tuple{Wrapped}) = g[1].values

const OUTPUT_FLAGS = Ref{Any}(nothing)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(exp_into!)}, ::Type{RT}, A::Annotation, out::Annotation
    ) where {RT}
    result = exp_into!(A.val, out.val)
    OUTPUT_FLAGS[] = EnzymeRules.overwritten(config)
    kept = EnzymeRules.overwritten(config)[3] ? copy(data(result)) : data(result)
    primal = EnzymeRules.needs_primal(config) ? result : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (kept, out.dval))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(exp_into!)}, ::Type{RT}, tape, A::Annotation, out::Annotation
    ) where {RT}
    kept, dout = tape
    if !(A isa Const)
        A.dval .+= data(dout) .* kept
    end
    data(dout) .= 0
    return (nothing, nothing)
end

function sum_then_clear(x, make_output)
    out = exp_into!(copy(x), make_output())
    kept = data(out)[1:2]
    data(out)[1:2] .= 0
    return sum(kept)
end

make_vector() = zeros(3)
make_wrapped() = Wrapped(zeros(3))
make_tuple() = (Wrapped(zeros(3)),)

# Enzyme's own rule for `det` keeps its argument unless the flag tells it to copy.
function det_then_write(M)
    d = det(view(M, 1:2, 1:2))
    M[1, 1] = 100.0
    return d
end

@testset "overwritten flags of arguments that hold references" begin

    @testset "A later write to the array behind $name" for (name, wrap) in containers
        @testset "$(nameof(caller))" for (caller, expected) in writers
            shared[1] = 3.0
            @test derivative(plain, caller, wrap(shared)) == expected
            shared[1] = 3.0
            @test derivative(scaled, caller, wrap(shared)) == expected
            @test FLAGS[][2]
        end
        @testset "$(nameof(caller))" for caller in readers
            shared[1] = 3.0
            @test derivative(scaled, caller, wrap(shared)) == 3.0
            if precise
                @test !FLAGS[][2]
            end
        end
    end

    @testset "Data held by the function itself" begin
        for (caller, expected) in (write_argument => 3.0, write_global => 3.0, write_in_loop => 12.0)
            shared[1] = 3.0
            @test function_derivative(caller) == expected
            @test FUNCTION_FLAGS[][1]
        end
        shared[1] = 3.0
        @test function_derivative(unchanged) == 3.0
        if precise
            @test !FUNCTION_FLAGS[][1]
        end
    end

    @testset "A Duplicated container" begin
        for wrap in (Wrapped, v -> (v, 1), v -> view(v, 1:1))
            values = [3.0]
            g = wrap(values)
            dg = Enzyme.make_zero(g)
            result = Enzyme.autodiff(
                Reverse, Const(run), Active, Const(scaled), Const(write_through),
                Duplicated(g, dg), Const([0.0]), Active(2.0)
            )
            @test result[1][end] == 3.0
            @test values == [42.0]
            @test data(dg)[1] == 2.0
            @test FLAGS[][2]
        end
    end

    @testset "A local object reachable through an argument" begin
        for f in (plain, scaled)
            @test derivative(f, write_local_box, Holder(Ref(0.0))) == 3.0
        end
        @test FLAGS[][2]
    end

    @testset "Split mode: the caller may write between the two passes" begin
        for wrap in (identity, Wrapped, v -> (v, 1))
            values = [3.0]
            g = wrap(values)
            forward, pullback = Enzyme.autodiff_thunk(
                ReverseSplitWithPrimal, Const{typeof(scaled)}, Active, Const{typeof(g)}, Active{Float64}
            )
            tape, primal, _ = forward(Const(scaled), Const(g), Active(2.0))
            @test primal == 6.0
            values[1] = 42.0
            @test pullback(Const(scaled), Const(g), Active(2.0), 1.0, tape)[1][end] == 3.0
        end
    end

    if isdefined(Enzyme.Compiler, :reaches_mutable_memory)
        @testset "Which argument types lead to other memory that can change" begin
            reaches = Enzyme.Compiler.reaches_mutable_memory
            for T in (
                    Float64, Vector{Float64}, Matrix{Int}, Tuple{Float64, Int},
                    @NamedTuple{name::Symbol, label::String, count::Int},
                    Base.RefValue{Float64}, UnitRange{Int}, Vector{Union{Missing, Float64}},
                    Vector{String}, Vector{Tuple{Float64, Int}}, Chain, Tagged{Float64},
                )
                @test !reaches(T)
            end
            for T in (
                    Wrapped, MutableWrapped, Outer, Labelled, Scale, Holder,
                    Tuple{Vector{Float64}, Int}, @NamedTuple{values::Vector{Float64}},
                    Base.RefValue{Vector{Float64}}, Vector{Vector{Float64}}, Vector{Any},
                    Vector{Wrapped}, typeof(view([1.0], 1:1)), typeof([1.0]'),
                    Symmetric{Float64, Matrix{Float64}}, Dict{Symbol, Float64},
                    Ptr{Float64}, Tuple{Ptr{Float64}, Int}, Any, AbstractVector{Float64},
                )
                @test reaches(T)
            end
        end
    end

    @testset "An output cleared after the call, passed as $name" for (name, make_output) in (
            "a vector" => make_vector,
            "a struct that holds a vector" => make_wrapped,
            "a tuple of such structs" => make_tuple,
        )
        x = [0.1, 0.2, 0.3]
        dx = zero(x)
        OUTPUT_FLAGS[] = nothing
        Enzyme.autodiff(Reverse, sum_then_clear, Active, Duplicated(x, dx), Const(make_output))
        @test OUTPUT_FLAGS[][3]
        @test dx ≈ [exp(0.1), exp(0.2), 0.0]
    end

    @testset "det of a view whose parent is written afterwards" begin
        M = [2.0 1.0; 1.0 3.0]
        dM = zero(M)
        Enzyme.autodiff(Reverse, det_then_write, Active, Duplicated(M, dM))
        # The gradient of det at the original M is its cofactor matrix. The shadow of
        # the overwritten entry restarts at zero and then receives its cofactor.
        @test dM ≈ [3.0 -1.0; -1.0 2.0]
    end

end

end # module
