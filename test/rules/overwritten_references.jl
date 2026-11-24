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

# A rule that returns a new vector, which the caller stores into an array of its
# own, as a plate of cells does. Julia follows each such store of a reference with
# a write barrier on the array.
doubled(g, x) = [weight(g) * x, 2 * weight(g) * x]

const ROW_FLAGS = Ref{Any}(nothing)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(doubled)}, ::Type{RT}, g::Annotation, x::Active
    ) where {RT}
    ROW_FLAGS[] = EnzymeRules.overwritten(config)
    saved = EnzymeRules.overwritten(config)[2] ? weight(g.val) : nothing
    y = doubled(g.val, x.val)
    dy = zero(y)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dy : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (dy, saved))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(doubled)}, ::Type{RT}, tape, g::Annotation, x::Active
    ) where {RT}
    dy, saved = tape
    w = EnzymeRules.overwritten(config)[2] ? saved : weight(g.val)
    return (nothing, w * dy[1] + 2 * w * dy[2])
end

# Each result goes into an array allocated before the loop, which the rule never
# sees, so nothing written after a call can change what the rule read. The array
# has a length known when the function compiles, or one read at run time.
function stored_rows(g, a, x)
    cells = Vector{Vector{Float64}}(undef, 3)
    for k in 1:3
        cells[k] = doubled(g, x)
    end
    return sum(sum, cells)
end
function stored_rows_runtime_length(g, a, x)
    cells = Vector{Vector{Float64}}(undef, length(a) + 2)
    for k in eachindex(cells)
        cells[k] = doubled(g, x)
    end
    return sum(sum, cells)
end
# Cells that hold a number computed from each result, as a plate whose cells hold
# a sum does. On Julia 1.12 and 1.13 such an array of length 3 lives on the stack.
function summed_rows(g, a, x)
    term = Vector{Float64}(undef, 3)
    for k in 1:3
        term[k] = sum(doubled(g, x))
    end
    return sum(term)
end
function summed_rows_runtime_length(g, a, x)
    term = Vector{Float64}(undef, length(a) + 2)
    for k in eachindex(term)
        term[k] = sum(doubled(g, x))
    end
    return sum(term)
end
# One result stored once, outside a loop.
function stored_row(g, a, x)
    cell = Vector{Vector{Float64}}(undef, length(a))
    cell[1] = doubled(g, x)
    return sum(cell[1])
end
# The same, also writing the array behind `g` after each call.
function stored_rows_written(g, a, x)
    cells = Vector{Vector{Float64}}(undef, length(a) + 2)
    for k in eachindex(cells)
        cells[k] = doubled(g, x)
        a[1] += 1.0
    end
    return sum(sum, cells)
end
# The rule's argument read from a tuple the function receives, inside the loop.
# Julia passes the struct to the rule by reference, through a stack slot that it
# fills again before each call, with the same value.
function ctx_summed_rows(ctx, a, x)
    s = 0.0
    for k in 1:(length(a) + 2)
        s += sum(doubled(ctx[1], x))
    end
    return s
end
function ctx_stored_rows(ctx, a, x)
    cells = Vector{Vector{Float64}}(undef, length(a) + 2)
    for k in eachindex(cells)
        cells[k] = doubled(ctx[1], x)
    end
    return sum(sum, cells)
end
# The same, also writing the array behind the rule's argument after each call.
function ctx_written_rows(ctx, a, x)
    s = 0.0
    for k in 1:(length(a) + 2)
        s += sum(doubled(ctx[1], x))
        a[1] += 1.0
    end
    return s
end
# A number stored into the array the rule read, after the call.
function summed_into_argument(g, a, x)
    y = sum(doubled(g, x))
    a[1] = 4.0
    return y
end
# Here the array the result goes into is the rule's argument: the store replaces
# the vector the rule read.
function replaced_row(g, a, x)
    y = doubled(g, x)
    g[1] = [42.0]
    return sum(y)
end

function row_derivative(caller, g)
    ROW_FLAGS[] = nothing
    result = Enzyme.autodiff(
        Reverse, Const(caller), Active, Const(g), Const(shared), Active(2.0)
    )
    return result[1][end]
end

# The result goes into a `Ref` the caller passes, as a prepared pullback keeps its
# outputs. That store of a pointer cannot change the numbers the rule read.
function stored_in_box(box, g, x)
    y = doubled(g, x)
    box[] = y
    return sum(y)
end

function box_derivative(caller, g)
    ROW_FLAGS[] = nothing
    box = Ref([0.0, 0.0])
    dbox = Ref([0.0, 0.0])
    result = Enzyme.autodiff(
        Reverse, Const(caller), Active, Duplicated(box, dbox), Const(g), Active(2.0)
    )
    return result[1][end]
end

# Containers whose own storage or reachable memory holds a pointer that a store
# of a pointer replaces after the call, so that the rule would read another
# vector in its reverse pass.
struct Boxed
    box::Base.RefValue{Vector{Float64}}
end
struct IndexedWeights
    values::Vector{Float64}
    index::Dict{Tuple{Float64, Float64}, Int32}
end
struct Rows
    rows::Vector{Vector{Float64}}
end
mutable struct Name
    name::Symbol
end
struct Named
    name::Name
end
data(g::IndexedWeights) = g.values
data(g::Boxed) = g.box[]
data(g::Rows) = g.rows[1]

function write_boxed_number(box, g, x)
    y = scaled(g, x)
    box[] = 42.0
    return y
end

function replaced_in_ref(g, a, x)
    y = doubled(g, x)
    g[] = [42.0]
    return sum(y)
end
function replaced_field(g, a, x)
    y = doubled(g, x)
    g.values = [42.0]
    return sum(y)
end
function replaced_in_box(g, a, x)
    y = doubled(g, x)
    g.box[] = [42.0]
    return sum(y)
end
function replaced_row_of(g, a, x)
    y = doubled(g, x)
    g.rows[1] = [42.0]
    return sum(y)
end

# Whether the loaded libEnzyme lets a write barrier leave such an array unexposed.
# Julia 1.11 and later keep an array's elements in a `Memory`, whose data pointer
# libEnzyme traces back to the allocation, also when the memory is allocated
# inline, moved to the stack or merged with the shared empty memory, only with
# repair `unexposed_memory_storage`; without it the flag stays `true` there.
const barrier_precise = precise && isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:unexposed_storage_barriers) &&
    (VERSION < v"1.11-" || Enzyme.has_dogfood_repair(:unexposed_memory_storage))

# Whether the loaded libEnzyme sees that the store Julia makes into the stack
# slot of a struct it passes by reference, before each call in a loop, writes
# back the value the slot already holds.
const slot_precise = precise && isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:unchanged_slot_stores)

# Whether stores of pointers alone after the call leave the flag `false` for an
# argument whose reachable mutable memory holds no pointers. From Julia 1.11 an
# array's object holds a reference to its memory that Julia code replaces, so
# there an argument that holds an array keeps the flag `true`.
const pointer_precise = precise && VERSION < v"1.11-" &&
    isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:later_pointer_stores)
# The containers whose own storage is immutable and holds the array by reference.
const pointer_free_containers = (
    "immutable struct", "struct in a struct", "struct with a Symbol and a String",
    "tuple", "named tuple", "view", "adjoint", "Symmetric",
)

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

    @testset "Results stored into an array allocated before a loop, $name" for (name, wrap) in (
            "Vector" => identity,
            "immutable struct" => Wrapped,
            "tuple" => v -> (v, 1),
        )
        @testset "$(nameof(caller))" for (caller, expected) in (
                stored_rows => 27.0, stored_rows_runtime_length => 27.0,
                summed_rows => 27.0, summed_rows_runtime_length => 27.0,
                stored_row => 9.0,
            )
            shared[1] = 3.0
            @test row_derivative(caller, wrap(shared)) == expected
            if barrier_precise
                @test !ROW_FLAGS[][2]
            end
        end
        shared[1] = 3.0
        @test row_derivative(stored_rows_written, wrap(shared)) == 3 * (3.0 + 4.0 + 5.0)
        @test ROW_FLAGS[][2]
        shared[1] = 3.0
        @test row_derivative(summed_into_argument, wrap(shared)) == 9.0
        @test ROW_FLAGS[][2]
        shared[1] = 3.0
    end

    @testset "The rule's argument read from a tuple in the loop, $name" for (name, wrap) in (
            "Vector" => identity,
            "immutable struct" => Wrapped,
            "tuple" => v -> (v, 1),
        )
        @testset "$(nameof(caller))" for caller in (ctx_summed_rows, ctx_stored_rows)
            shared[1] = 3.0
            @test row_derivative(caller, (wrap(shared), 1)) == 27.0
            if slot_precise && barrier_precise
                @test !ROW_FLAGS[][2]
            end
        end
        shared[1] = 3.0
        @test row_derivative(ctx_written_rows, (wrap(shared), 1)) == 3 * (3.0 + 4.0 + 5.0)
        @test ROW_FLAGS[][2]
        shared[1] = 3.0
    end

    @testset "A result stored into the array the rule read" begin
        shared[1] = 3.0
        @test row_derivative(replaced_row, [shared]) == 9.0
        @test ROW_FLAGS[][2]
    end

    @testset "A result stored into a Ref the caller passes, $name" for (name, wrap) in containers
        shared[1] = 3.0
        @test box_derivative(stored_in_box, wrap(shared)) == 9.0
        # The store writes a pointer into the box, and this argument leads to no
        # mutable memory that holds a pointer. The other containers hold one in
        # their own storage, where libEnzyme's flag of the operand counts the box
        # as possibly the same object.
        if pointer_precise && name in pointer_free_containers
            @test !ROW_FLAGS[][2]
        end
    end

    @testset "A later store to a Ref outside a dictionary-backed argument" begin
        g = IndexedWeights([3.0], Dict((1.0, 2.0) => Int32(1)))
        @test box_derivative(stored_in_box, g) == 9.0
        if isdefined(Enzyme, :has_dogfood_repair) && Enzyme.has_dogfood_repair(:later_store_targets)
            @test !ROW_FLAGS[][2]
            @test box_derivative(stored_in_box, Wrapped([3.0])) == 9.0
            @test !ROW_FLAGS[][2]
        end
    end

    @testset "The same Ref reaches the rule and the later store" begin
        box = Ref([3.0])
        ROW_FLAGS[] = nothing
        result = Enzyme.autodiff(
            Reverse, Const(stored_in_box), Active, Duplicated(box, Ref([0.0])),
            Const(Boxed(box)), Active(2.0),
        )
        @test box[][1] == 6.0
        @test result[1][end] == 9.0
        @test ROW_FLAGS[][2]
    end

    @testset "An array wraps the later store's scalar field" begin
        box = Ref(3.0)
        GC.@preserve box begin
            values = unsafe_wrap(Vector{Float64}, Base.unsafe_convert(Ptr{Float64}, box), 1)
            FLAGS[] = nothing
            result = Enzyme.autodiff(
                Reverse, Const(write_boxed_number), Active, Duplicated(box, Ref(0.0)),
                Const(Wrapped(values)), Active(2.0),
            )
            @test box[] == 42.0
            @test FLAGS[][2]
            @test result[1][end] == 3.0
        end
    end

    @testset "A pointer stored after the call replaces what the rule read, $(nameof(caller))" for (caller, wrap) in (
            replaced_in_ref => v -> Ref(v),
            replaced_field => MutableWrapped,
            replaced_row => v -> [v],
            replaced_in_box => v -> Boxed(Ref(v)),
            replaced_row_of => v -> Rows([v]),
        )
        shared[1] = 3.0
        @test row_derivative(caller, wrap(shared)) == 9.0
        @test ROW_FLAGS[][2]
    end

    if isdefined(Enzyme.Compiler, :reaches_mutable_pointers)
        @testset "Which argument types lead to mutable memory that holds a pointer" begin
            reaches = Enzyme.Compiler.reaches_mutable_memory
            pointers = Enzyme.Compiler.reaches_mutable_pointers
            never = (
                Float64, Vector{Float64}, Matrix{Int}, Tuple{Float64, Int},
                Base.RefValue{Float64}, UnitRange{Int}, Vector{Union{Missing, Float64}},
                Vector{String}, Chain, Tagged{Float64}, Holder,
            )
            # The array behind the argument holds numbers. Before Julia 1.11 only
            # runtime calls write its header; from 1.11 its `ref` field is a pointer.
            arrays = (
                Wrapped, MutableWrapped, Outer, Labelled, Scale,
                Tuple{Vector{Float64}, Int}, @NamedTuple{values::Vector{Float64}},
                Base.RefValue{Vector{Float64}}, Vector{Vector{Float64}}, Vector{Wrapped},
                typeof(view([1.0], 1:1)), typeof([1.0]'), Symmetric{Float64, Matrix{Float64}},
            )
            always = (
                Boxed, Rows, Named, Vector{Any}, Vector{Vector{String}},
                Dict{Symbol, Float64}, Ptr{Float64}, Tuple{Ptr{Float64}, Int}, Any,
                AbstractVector{Float64},
            )
            for T in never
                @test !pointers(T)
            end
            for T in arrays
                @test pointers(T) == (VERSION >= v"1.11-")
            end
            for T in always
                @test pointers(T)
            end
            # What stores of pointers can change, a write of any kind can change.
            for T in (never..., arrays..., always...)
                @test !pointers(T) || reaches(T)
            end
        end
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
