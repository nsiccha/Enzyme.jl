using Enzyme, Test, LinearAlgebra
using Enzyme: EnzymeRules

# Reverse mode through `deepcopy`, by what the copied value is and how the call reaches
# Enzyme.
#
# `deepcopy` has one method, `deepcopy(@nospecialize x)`, so Julia compiles a call to it
# either as a call dispatched at run time or as a call of that one method with the value
# boxed. The rule is then compiled for the value's own type, or for `Any`:
#
#   * for its own type, a value that holds differentiable numbers directly beside mutable
#     data (a tuple of an array and a float) arrives as `MixedDuplicated`. The rule would
#     return the copy as an immutable value and never see the derivative of the numbers
#     the copy holds. Enzyme compiles such a call as a deep copy of a `Ref` box instead;
#   * for `Any`, such a value and its copy have `Ref` boxes as shadows, and the rule
#     accumulates box to box;
#   * a `Const` value can still be asked for a copy with a shadow. That shadow is a zero
#     copy of its own, and nothing goes back into the constant value. Where the call is
#     dispatched at run time, Enzyme makes that copy without compiling the rule.
#
# Before the repair most cases below stopped with an exception from the rule's reverse
# method or from the code that calls the rule: `MethodError` in `accumulate_into` for a
# `Ref` box, `FieldError` for the `dval` of a `Const`, `setfield!` of an immutable, and on
# Julia 1.10 `MixedReturnException` or `AssertionError: !(x isa Active)`. Copies of an
# array passed, and so did a few forms of a tuple of constant arrays and of plain floats.
# On Julia 1.12 and later the rule compiled for a value's own type also returned a value
# holding arrays and floats through the caller's buffers, which could end the process in
# the garbage collector before any of those errors (Enzyme snag
# `mixed-activity-t-3d3418f9`).

struct DeepcopyRecord
    a::Vector{Float64}
    s::Float64
end

const DC_A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
const DC_B = [0.5 -1.0; 2.0 0.25]
const DC_X = [0.3, -0.7]

# How the call is written. `v` is the expression of the value to copy.
const DC_FORMS = (
    :static => v -> :(deepcopy($v)),
    :hidden_function => v -> :(Base.inferencebarrier(deepcopy)($v)),
    :hidden_argument => v -> :(deepcopy(Base.inferencebarrier($v))),
    :splatted => v -> :(deepcopy(($v,)...)),
    :hidden_splat => v -> :(deepcopy(Base.inferencebarrier(($v,))...)),
    :invokelatest => v -> :(Base.invokelatest(deepcopy, $v)),
    :mapped => v -> :(map(deepcopy, ($v,))[1]),
    :any_element => v -> :(deepcopy(Any[$v][1])),
)

# name => (the value built from `x` and the constant argument `d`, the copy's type,
#          the result computed from the copy `t`)
const DC_VALUES = (
    # Constant data, read beside the active `x`.
    :const_arrays_float => (:d, :(Tuple{Matrix{Float64}, Matrix{Float64}, Float64}), :(sum(abs2, t[1] * (t[2] * x)) * t[3])),
    :const_arrays => (:d, :(Tuple{Matrix{Float64}, Matrix{Float64}}), :(sum(abs2, t[1] * (t[2] * x)))),
    :const_array_float => (:d, :(Tuple{Matrix{Float64}, Float64}), :(sum(abs2, t[1] * x) * t[2])),
    :const_field => (:(d.basis), :(Tuple{Matrix{Float64}, Matrix{Float64}, Float64}), :(sum(abs2, t[1] * (t[2] * x)) * t[3])),
    :const_record => (:d, :DeepcopyRecord, :(dot(t.a, x) * t.s)),
    # Active data.
    :active_float => (:((d, x[1])), :(Tuple{Matrix{Float64}, Float64}), :(sum(t[1]) * t[2])),
    :active_array => (:((x, d)), :(Tuple{Vector{Float64}, Float64}), :(sum(abs2, t[1]) * t[2])),
    :active_both => (:((d * x, x[2])), :(Tuple{Vector{Float64}, Float64}), :(sum(t[1]) * t[2])),
    :active_record => (:(DeepcopyRecord(d * x, x[2])), :DeepcopyRecord, :(sum(t.a) * t.s)),
    :active_named => (:((; a = d * x, s = x[2])), :(@NamedTuple{a::Vector{Float64}, s::Float64}), :(sum(t.a) * t.s)),
    :active_nested => (:(((d * x, x[1]), x[2])), :(Tuple{Tuple{Vector{Float64}, Float64}, Float64}), :(sum(t[1][1]) * t[1][2] * t[2])),
    :active_number => (:(x[1] * x[2] * d), :Float64, :(t * 1.0)),
    :active_numbers => (:((x[1], d * x[2])), :(Tuple{Float64, Float64}), :(t[1] * t[2])),
    :active_vector => (:(d .* x), :(Vector{Float64}), :(sum(abs2, t))),
)

for (value, (make, T, use)) in DC_VALUES, (form, call) in DC_FORMS
    name = Symbol(:dc_, value, :_, form)
    @eval function $name(x, d)
        t = $(call(make))::$T
        return $use
    end
end

# The constant argument and the gradient with respect to `x` of each value's result.
dc_both_gradient(x) = vec(sum(DC_A; dims = 1)) .* x[2] .+ [0.0, sum(DC_A * x)]
const DC_CASES = (
    :const_arrays_float => ((DC_A, DC_B, 0.3), x -> 0.6 .* (DC_B' * (DC_A' * (DC_A * (DC_B * x))))),
    :const_arrays => ((DC_A, DC_B), x -> 2 .* (DC_B' * (DC_A' * (DC_A * (DC_B * x))))),
    :const_array_float => ((DC_A, 0.3), x -> 0.6 .* (DC_A' * (DC_A * x))),
    :const_field => ((; basis = (DC_A, DC_B, 0.3)), x -> 0.6 .* (DC_B' * (DC_A' * (DC_A * (DC_B * x))))),
    :const_record => (DeepcopyRecord([1.5, -2.0], 0.3), x -> 0.3 .* [1.5, -2.0]),
    :active_float => (DC_A, x -> [sum(DC_A), 0.0]),
    :active_array => (0.3, x -> 0.6 .* x),
    :active_both => (DC_A, dc_both_gradient),
    :active_record => (DC_A, dc_both_gradient),
    :active_named => (DC_A, dc_both_gradient),
    :active_nested => (
        DC_A,
        x -> vec(sum(DC_A; dims = 1)) .* (x[1] * x[2]) .+ [sum(DC_A * x) * x[2], sum(DC_A * x) * x[1]],
    ),
    :active_number => (2.0, x -> [2 * x[2], 2 * x[1]]),
    :active_numbers => (2.0, x -> [2 * x[2], 2 * x[1]]),
    :active_vector => (2.0, x -> 8 .* x),
)

dc_unchanged(a::DeepcopyRecord, b::DeepcopyRecord) = a.a == b.a && a.s == b.s
dc_unchanged(a::NamedTuple, b::NamedTuple) = dc_unchanged(a.basis, b.basis)
dc_unchanged(a, b) = a == b

const DC_MODES = (Reverse, ReverseWithPrimal, set_runtime_activity(ReverseWithPrimal))

# On Julia 1.10 the caller, not the copy, needs runtime activity when it splats a boxed
# tuple around its constant argument into a call dispatched at run time: the argument's
# arrays are stored into a box the callee could write to. Any callee does this, with or
# without a rule, and it does not concern `deepcopy`.
function dc_caller_needs_runtime_activity(value::Symbol, form::Symbol, mode)
    VERSION < v"1.11-" || return false
    Enzyme.runtime_activity(mode) && return false
    return form === :hidden_splat && value in (:const_arrays_float, :const_arrays, :const_array_float, :const_record)
end

@testset "deepcopy by value and call form" begin
    seed = [0.5, -0.25]
    for (value, (d, gradient)) in DC_CASES, (form, _) in DC_FORMS
        f = getfield(@__MODULE__, Symbol(:dc_, value, :_, form))
        @testset "$value, $form" begin
            for mode in DC_MODES
                x = copy(DC_X)
                dx = copy(seed)
                d0 = deepcopy(d)
                if dc_caller_needs_runtime_activity(value, form, mode)
                    @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError autodiff(
                        mode, Const(f), Active, Duplicated(x, dx), Const(d),
                    )
                    continue
                end
                res = autodiff(mode, Const(f), Active, Duplicated(x, dx), Const(d))
                # The derivative is added to what the shadow held.
                @test dx ≈ seed .+ gradient(DC_X) atol = 1.0e-12
                @test x == DC_X
                # Nothing is written into the constant argument.
                @test dc_unchanged(d, d0)
                if mode isa ReverseMode{true}
                    @test res[2] ≈ f(DC_X, d)
                end
            end
        end
    end
end

# The same array twice in the copied value: the copy holds one array, and so does its
# shadow. A write through one of them is read through the other.
function dc_aliased(x, c)
    v = 2 .* x
    t = Base.inferencebarrier(deepcopy)((v, v, c))::Tuple{Vector{Float64}, Vector{Float64}, Float64}
    t[1][1] = 3 * t[2][2]
    return sum(abs2, t[2]) * t[3]
end

# A copy of a copy, with the copied value used again afterwards.
function dc_chain(x, A)
    p = (A * x, x[2])
    t = deepcopy(p)::Tuple{Vector{Float64}, Float64}
    u = Base.inferencebarrier(deepcopy)(t)::Tuple{Vector{Float64}, Float64}
    return sum(t[1]) * t[2] + sum(u[1]) * u[2] + sum(p[1]) * p[2]
end

@testset "aliases and repeated copies" begin
    for mode in DC_MODES
        dx = zero(DC_X)
        autodiff(mode, Const(dc_aliased), Active, Duplicated(copy(DC_X), dx), Const(0.3))
        # After the write, t[2] is [6 x[2], 2 x[2]].
        @test dx ≈ [0.0, 0.3 * 80 * DC_X[2]] atol = 1.0e-12

        dx = zero(DC_X)
        autodiff(mode, Const(dc_chain), Active, Duplicated(copy(DC_X), dx), Const(DC_A))
        @test dx ≈ 3 .* dc_both_gradient(DC_X) atol = 1.0e-12
    end
end

dc_case(value::Symbol) = last(only(filter(c -> first(c) === value, collect(DC_CASES))))

@testset "split mode" begin
    for value in (:const_field, :const_array_float, :active_float, :active_both, :active_record),
            form in (:static, :hidden_function, :hidden_argument)

        f = getfield(@__MODULE__, Symbol(:dc_, value, :_, form))
        d, gradient = dc_case(value)
        forward, reverse = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{Vector{Float64}}, Const{typeof(d)},
        )
        dx = zero(DC_X)
        x = copy(DC_X)
        tape, value_out, _ = forward(Const(f), Duplicated(x, dx), Const(d))
        @test value_out ≈ f(DC_X, d)
        reverse(Const(f), Duplicated(x, dx), Const(d), 1.0, tape)
        @test dx ≈ gradient(DC_X) atol = 1.0e-12
    end
end

# Two derivative lanes at once. Constant data copied through a dynamically dispatched
# call is copied once, with a zero shadow per lane. A value that holds its own shadow
# per lane does not differentiate in a batch yet, before or with this repair: the rule
# receives only the first lane's shadow for a boxed argument.
@testset "batches of constant data" begin
    for value in (:const_arrays_float, :const_arrays, :const_field),
            form in (:hidden_function, :hidden_splat, :invokelatest)

        dc_caller_needs_runtime_activity(value, form, Reverse) && continue
        f = getfield(@__MODULE__, Symbol(:dc_, value, :_, form))
        d, gradient = dc_case(value)
        dx1 = zero(DC_X)
        dx2 = zero(DC_X)
        d0 = deepcopy(d)
        autodiff(Reverse, Const(f), Active, BatchDuplicated(copy(DC_X), (dx1, dx2)), Const(d))
        @test dx1 ≈ gradient(DC_X) atol = 1.0e-12
        @test dx2 ≈ gradient(DC_X) atol = 1.0e-12
        @test dc_unchanged(d, d0)
    end
end

@testset "the rule and the dispatch hook" begin
    C = Enzyme.Compiler
    mixed = Tuple{Vector{Float64}, Float64}
    hook(f, T, Annotation) = C.reverse_runtime_function(f, Tuple{T}, Tuple{Annotation{T}})
    # A constant value is copied by the runtime rule itself, whatever its type. An
    # active one is copied inside a box only when it holds numbers beside mutable data.
    @test hook(deepcopy, mixed, Const) isa C.ConstantDeepcopy
    @test hook(deepcopy, Vector{Float64}, Const) isa C.ConstantDeepcopy
    @test hook(deepcopy, Tuple{Matrix{Float64}, Matrix{Float64}}, Const) isa C.ConstantDeepcopy
    @test hook(deepcopy, mixed, MixedDuplicated) isa C.BoxedDeepcopy
    @test hook(deepcopy, DeepcopyRecord, MixedDuplicated) isa C.BoxedDeepcopy
    @test C.reverse_runtime_function(deepcopy, Tuple{mixed}, Tuple{BatchMixedDuplicated{mixed, 2}}) isa C.BoxedDeepcopy
    @test hook(deepcopy, Vector{Float64}, Duplicated) === deepcopy
    @test hook(deepcopy, Tuple{Float64, Float64}, Active) === deepcopy
    @test hook(deepcopy, Base.RefValue{mixed}, Duplicated) === deepcopy
    @test hook(copy, mixed, MixedDuplicated) === copy
    @test hook(copy, mixed, Const) === copy
    @test C.runtime_function(deepcopy, Tuple{mixed}) === deepcopy
    value = ([1.0, 2.0], 3.0)
    for copied in (C.BoxedDeepcopy()(value), C.ConstantDeepcopy()(value))
        @test copied == value
        @test copied[1] !== value[1]
    end

    config = EnzymeRules.RevConfig{true, true, 1, (false, false), false, false}()
    func = Const(deepcopy)

    # A `Const` argument: a copy with a zero shadow of its own, and no accumulation.
    aug = EnzymeRules.augmented_primal(config, func, Duplicated{mixed}, Const(value))
    @test aug.primal == value
    @test aug.primal[1] !== value[1]
    @test aug.shadow == ([0.0, 0.0], 0.0)
    aug.shadow[1] .= 7.0
    @test EnzymeRules.reverse(config, func, Duplicated{mixed}, aug.tape, Const(value)) === (nothing,)
    @test value == ([1.0, 2.0], 3.0)

    # Passed boxed, the value and its copy have `Ref` boxes as shadows.
    dvalue = Ref(([0.5, 0.5], 0.25))
    aug = EnzymeRules.augmented_primal(config, func, Duplicated{Any}, Duplicated{Any}(value, dvalue, false))
    @test aug.shadow isa Base.RefValue{mixed}
    @test aug.shadow[] == ([0.0, 0.0], 0.0)
    aug.shadow[][1] .= [1.0, 2.0]
    aug.shadow[] = (aug.shadow[][1], 4.0)
    @test EnzymeRules.reverse(config, func, Duplicated{Any}, aug.tape, Duplicated{Any}(value, dvalue, false)) === (nothing,)
    @test dvalue[] == ([1.5, 2.5], 4.25)
    @test aug.shadow[] == ([0.0, 0.0], 0.0)

    # A boxed array keeps the shadow it always had.
    array = [1.0, 2.0]
    aug = EnzymeRules.augmented_primal(config, func, Duplicated{Any}, Duplicated{Any}(array, [0.0, 0.0], false))
    @test aug.shadow isa Vector{Float64}

    # Compiled for the value's own type, the rule cannot return the derivative of the
    # numbers the copy holds, and says so instead of dropping it.
    @test_throws Enzyme.DeepcopyMixedArgument{mixed} EnzymeRules.augmented_primal(
        config, func, Duplicated{mixed}, MixedDuplicated(value, Ref(([0.0, 0.0], 0.0))),
    )
    message = sprint(showerror, Enzyme.DeepcopyMixedArgument{mixed}())
    @test occursin("Tuple{Vector{Float64}, Float64}", message)
    @test occursin("Ref", message)

    # A number whose copy is differentiated takes the copy's derivative. One whose copy
    # is not differentiated takes none, in each lane.
    @test EnzymeRules.reverse(config, func, Active(1.5), nothing, Active(2.0)) === (1.5,)
    unused = EnzymeRules.RevConfig{false, false, 1, (false, false), false, false}()
    @test EnzymeRules.reverse(unused, func, Const{Float64}, nothing, Active(2.0)) === (0.0,)
    unused2 = EnzymeRules.RevConfig{false, false, 2, (false, false), false, false}()
    @test EnzymeRules.reverse(unused2, func, Const{Float64}, nothing, Active(2.0)) === ((0.0, 0.0),)

    @test Enzyme.has_dogfood_repair(:deepcopy_reverse_activities)
end
