using Enzyme, LinearAlgebra, Test
using Enzyme: EnzymeRules

# An unannotated function argument whose type can hold derivative data must be
# proven readonly (`err_if_func_written`). These callables only read the data
# they hold, so they must differentiate exactly as `Const(f)` does.
#
# `Matrix + UniformScaling` takes the `UniformScaling` by reference, so the
# copy that the calling-convention wrapper makes of a by-value argument
# (`.innerparm`) survives optimization. LLVM splits the store that fills that
# copy into one store per field; the check used to report those stores as
# writes to the argument. It also used to treat the fresh matrix such a call
# returns as data of the argument, and so rejected passing it to LAPACK, and
# where a load of a held array was untyped (Julia 1.10), it followed the float
# read from it into the fresh result.

struct ROShifted{T}
    data::T
end
function (c::ROShifted)(x)
    A, s = c.data
    return sum(x^2 .* A + s)
end

struct ROSwapped{T}
    data::T
end
function (c::ROSwapped)(x)
    s, A = c.data
    return sum(x^2 .* A + s)
end

struct ROFields{M, S}
    A::M
    s::S
end
(c::ROFields)(x) = sum(x^2 .* c.A + c.s)

struct ROThree{T}
    data::T
end
function (c::ROThree)(x)
    A, B, s = c.data
    return sum((x^2 .* A + s) .* B)
end

struct ROCholesky{T}
    data::T
end
function (c::ROCholesky)(rho)
    D, s = c.data
    K = exp.(-0.5 .* D ./ rho^2) + s
    return logdet(cholesky(Symmetric(K)))
end

@noinline ro_tuple_sum(t) = sum(t[1]) * t[2]

# The tuple passed by reference is a stack slot holding the captured matrix and
# active data.
struct ROLocalTuple{M}
    A::M
end
(c::ROLocalTuple)(x) = ro_tuple_sum((c.A, x * x))

@noinline ro_first(t) = t[1]

# Controls: each writes data the callable holds, so the check must still raise.
# `ROWriteReturned` writes into data of the argument that a callee returned.
struct ROWriteMatrix{T}
    data::T
end
function (c::ROWriteMatrix)(x)
    A, s = c.data
    A[1] = x
    return sum(x .* A + s)
end

struct ROWriteRef{T}
    data::T
end
function (c::ROWriteRef)(x)
    A, s, r = c.data
    r[] = x
    return sum(x .* A + s) + r[]
end

struct ROWriteReturned{T}
    data::T
end
function (c::ROWriteReturned)(x)
    A = ro_first(c.data)
    A[1] = x
    return sum(A + c.data[2])
end

ro_A() = [1.0 0.5; 0.5 1.0]
ro_B() = [2.0 0.0; 1.0 3.0]
ro_D() = [0.0 1.0 4.0; 1.0 0.0 1.0; 4.0 1.0 0.0]

@testset "Readonly check: by-value argument copies and fresh call results" begin
    for f in (
            ROShifted((ro_A(), 1.0e-10I)), ROSwapped((1.0e-10I, ro_A())),
            ROFields(ro_A(), 1.0e-10I), ROThree((ro_A(), ro_B(), 1.0e-10I)),
            ROLocalTuple(ro_A()),
        )
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(1.5))
        @test autodiff(ReverseWithPrimal, f, Active, Active(1.5)) == expected
        dexpected = autodiff(ForwardWithPrimal, Const(f), Duplicated(1.5, 1.0))
        @test autodiff(ForwardWithPrimal, f, Duplicated(1.5, 1.0)) == dexpected
    end
    @test autodiff(Reverse, ROShifted((ro_A(), 1.0e-10I)), Active, Active(1.5))[1][1] ≈ 9.0

    f = ROCholesky((ro_D(), 1.0e-10I))
    @test autodiff(ReverseWithPrimal, f, Active, Active(1.3)) ==
        autodiff(ReverseWithPrimal, Const(f), Active, Active(1.3))

    for f in (
            ROWriteMatrix((ro_A(), 1.0e-10I)), ROWriteRef((ro_A(), 1.0e-10I, Ref(0.0))),
            ROWriteReturned((ro_A(), 1.0e-10I)),
        )
        @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(1.5))
        @test f.data[1] == ro_A()
    end
end

# The check walks a callee's body only when the callee has a single call site.
# A `@noinline` function with a custom rule, called from a second site, is
# then not walked, and the call that passes it a field of the callable by
# reference used to be reported as a write, although the parameter's type
# cannot hold derivative data (a ReactiveKernels derivative rule for
# `logbeta`, hoisted out of a loop). The parameter type now decides. The rule
# here calls `lgamma_r`, which writes the sign of Γ through a pointer, so LLVM
# cannot infer it read-only: Julia 1.12 infers a rule without such a call
# read-only, and `is_readonly(called)` accepted it before this check existed.
struct RORule
    a::Bool
    b::NTuple{3, Bool}
end
@noinline function (r::RORule)(x, y)
    sign = Ref{Int32}(0)
    ccall((:lgamma_r, Base.libm_name), Float64, (Float64, Ref{Int32}), y, sign)
    return r.a ? x * y + r.b[2] : x + y
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{RORule}, ::Type{RT},
        x::EnzymeRules.Annotation, y::EnzymeRules.Annotation
    ) where {RT}
    primal = f.val(x.val, y.val)
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? primal : nothing, nothing, (x.val, y.val)
    )
end
function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, f::Const{RORule}, dret::Active, tape,
        x::EnzymeRules.Annotation, y::EnzymeRules.Annotation
    )
    xv, yv = tape
    dx, dy = f.val.a ? (yv, xv) : (1.0, 1.0)
    return (x isa Active ? dret.val * dx : nothing, y isa Active ? dret.val * dy : nothing)
end

struct RORuleHolder{M}
    data::M
    rule::RORule
end
(c::RORuleHolder)(x) = c.rule(x, 2.0) + RORule(false, (true, false, true))(x, 3.0) + sum(c.data)

struct RORulePair{M}
    data::M
    rules::Tuple{RORule, RORule}
end
(c::RORulePair)(x) = c.rules[1](x, 2.0) * c.rules[2](x, 3.0) + sum(c.data)

struct ROBox{M}
    v::M
end

# Not yet accepted: a parameter that holds derivative data, passed to a callee
# that only reads it but is neither walked (it has a second call site) nor
# read-only (it allocates; a custom rule alone does not stop LLVM from
# inferring that on Julia 1.12 without forced bounds checks).
@noinline ro_box_scaled(b::ROBox, x) = sum(b.v .* x)
struct ROAllocBox{M}
    box::ROBox{M}
end
(c::ROAllocBox)(x) = ro_box_scaled(c.box, x) + ro_box_scaled(ROBox([1.0]), x)

@noinline ro_box_rule(b::ROBox, x) = sum(b.v .* x)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(ro_box_rule)}, ::Type{RT},
        b::EnzymeRules.Annotation, x::EnzymeRules.Annotation
    ) where {RT}
    primal = ro_box_rule(b.val, x.val)
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? primal : nothing, nothing, sum(b.val.v)
    )
end
function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(ro_box_rule)}, dret::Active, tape,
        b::EnzymeRules.Annotation, x::EnzymeRules.Annotation
    )
    return (nothing, x isa Active ? dret.val * tape : nothing)
end
struct ROCustomBox{M}
    box::ROBox{M}
end
(c::ROCustomBox)(x) = ro_box_rule(c.box, x) + ro_box_rule(ROBox([1.0]), x)

# Controls: callees with a second call site that write data the callable holds.
@noinline ro_box_set!(b::ROBox, x) = (b.v[1] = x; x)
struct ROWriteBox{M}
    box::ROBox{M}
end
(c::ROWriteBox)(x) = ro_box_set!(c.box, x) + ro_box_set!(ROBox([0.0]), x) + sum(c.box.v)

@noinline ro_ref_set!(r, x) = (r[] = x; nothing)
struct ROWriteFloatRef{M}
    data::M
    r::Base.RefValue{Float64}
end
(c::ROWriteFloatRef)(x) = (ro_ref_set!(c.r, x); ro_ref_set!(Ref(0.0), x); c.r[] * sum(c.data))

@testset "Readonly check: calls whose parameter cannot hold derivative data" begin
    for f in (
            RORuleHolder([1.0, 2.0], RORule(true, (true, false, true))),
            RORulePair([1.0, 2.0], (RORule(true, (true, true, true)), RORule(false, (true, false, true)))),
        )
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(1.5))
        @test autodiff(ReverseWithPrimal, f, Active, Active(1.5)) == expected
    end
    @test autodiff(Reverse, RORuleHolder([1.0, 2.0], RORule(true, (true, false, true))), Active, Active(1.5))[1][1] == 3.0

    for f in (ROAllocBox(ROBox([1.0, 2.0])), ROCustomBox(ROBox([1.0, 2.0])))
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(1.5))
        @test_broken autodiff(ReverseWithPrimal, f, Active, Active(1.5)) == expected
    end

    f = ROWriteBox(ROBox([1.0, 2.0]))
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(1.5))
    @test f.box.v == [1.0, 2.0]
    f = ROWriteFloatRef([1.0, 2.0], Ref(0.0))
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(1.5))
    @test f.r[] == 0.0
end

# Dynamic calls (`Base.inferencebarrier`, or any call whose result Julia cannot
# infer) reach a method known only at run time. When data the callable holds
# reaches one, as the called function (a field Julia boxes to pass it) or as an
# argument, the check is left to the rules of that call: they compile the
# method's derivative with the check on those operands. What such a call
# returns is followed like any other value.
struct ROScaler{M}
    v::M
end
(s::ROScaler)(x) = sum(s.v .* x)

struct RODynCallee{M}
    s::ROScaler{M}
    t::Float64
end
(c::RODynCallee)(x) = Base.inferencebarrier(c.s)(x) * c.t

ro_dyn_scaled(v, x) = sum(v .* x)
struct RODynArg{M}
    v::M
end
(c::RODynArg)(x) = Base.inferencebarrier(ro_dyn_scaled)(c.v, x)

struct RODynLatest{M}
    v::M
end
(c::RODynLatest)(x) = Base.invokelatest(ro_dyn_scaled, c.v, x)

ro_dyn_step(P, s, a) = (P[1] * s[1] + a, P[2] * s[2], s[3])
ro_dyn_stepv(props, j, s, a) = ro_dyn_step(props[j], s, a)
struct RODynLoop{T}
    t::T
end
function (h::RODynLoop)(k)
    seed, props, amounts = h.t
    s = seed
    for j in eachindex(amounts)
        s = amounts[j] > 1 ? Base.inferencebarrier(ro_dyn_stepv)(props, j, s, amounts[j] * k) :
            (props[j][1] * s[1], s[2] * k, s[3])
    end
    return s[1] + s[2] + s[3]
end

# Julia 1.12 unboxes the tuple the dynamic call returns with a `memcpy` into a
# stack slot.
struct RODynTuple{M}
    v::M
end
(c::RODynTuple)(x) = sum(Base.inferencebarrier(getindex)(c.v, 2)::NTuple{3, Float64}) * x

# A copy of held float data reads it: the bytes cannot refer to memory.
struct ROCopyOut{M}
    v::M
end
(c::ROCopyOut)(x) = sum(copy(c.v)) * x

ro_dyn_first(t) = t[1]
struct RODynReturned{T}
    data::T
end
(c::RODynReturned)(x) = sum(Base.inferencebarrier(ro_dyn_first)(c.data)) * x

# Controls: each writes data the callable holds through a dynamic call, as the
# called function, as an argument, or into what one returned.
struct ROSetter{M}
    v::M
end
(s::ROSetter)(x) = (s.v[1] = x; x)
struct RODynWriteCallee{M}
    s::ROSetter{M}
    t::Float64
end
(c::RODynWriteCallee)(x) = Base.inferencebarrier(c.s)(x) * c.t + sum(c.s.v)

ro_dyn_set!(v, x) = (v[1] = x; x)
struct RODynWriteArg{M}
    v::M
end
(c::RODynWriteArg)(x) = Base.inferencebarrier(ro_dyn_set!)(c.v, x) + sum(c.v)

struct ROCopyInto{M}
    v::M
end
(c::ROCopyInto)(x) = (copyto!(c.v, [x, 2x]); sum(c.v))

struct RODynWriteReturned{T}
    data::T
end
function (c::RODynWriteReturned)(x)
    v = Base.inferencebarrier(ro_dyn_first)(c.data)::Vector{Float64}
    v[1] = x
    return sum(v)
end

@testset "Readonly check: data passed to dynamic calls" begin
    for f in (
            RODynCallee(ROScaler([1.0, 2.0]), 3.0), RODynArg([1.0, 2.0]), RODynLatest([1.0, 2.0]),
            RODynLoop(((1.0, 2.0, 3.0), [(0.5, 0.25), (0.9, 0.8), (0.7, 0.6)], [2.0, 0.0, 3.0])),
            RODynReturned(([1.0, 2.0], 3.0)), RODynTuple([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]),
            ROCopyOut([1.0, 2.0]),
        )
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(0.7))
        @test autodiff(ReverseWithPrimal, f, Active, Active(0.7)) == expected
        rmode = set_runtime_activity(ReverseWithPrimal)
        @test autodiff(rmode, f, Active, Active(0.7)) == autodiff(rmode, Const(f), Active, Active(0.7))
        dexpected = autodiff(ForwardWithPrimal, Const(f), Duplicated(0.7, 1.0))
        @test autodiff(ForwardWithPrimal, f, Duplicated(0.7, 1.0)) == dexpected
        bexpected = autodiff(Forward, Const(f), BatchDuplicated(0.7, (1.0, 2.0)))
        @test autodiff(Forward, f, BatchDuplicated(0.7, (1.0, 2.0))) == bexpected
    end
    @test autodiff(Reverse, RODynCallee(ROScaler([1.0, 2.0]), 3.0), Active, Active(0.7))[1][1] == 9.0
    @test autodiff(
        Reverse,
        RODynLoop(((1.0, 2.0, 3.0), [(0.5, 0.25), (0.9, 0.8), (0.7, 0.6)], [2.0, 0.0, 3.0])),
        Active, Active(0.7),
    )[1][1] ≈ 4.56

    f = RODynWriteCallee(ROSetter([1.0, 2.0]), 3.0)
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
    @test f.s.v == [1.0, 2.0]
    f = RODynWriteArg([1.0, 2.0])
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
    @test f.v == [1.0, 2.0]
    f = RODynWriteReturned(([1.0, 2.0], 3.0))
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
    @test f.data[1] == [1.0, 2.0]
    f = RODynWriteArg([1.0, 2.0])
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Forward, f, Duplicated(0.7, 1.0))
    @test f.v == [1.0, 2.0]
    f = ROCopyInto([1.0, 2.0])
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
    @test f.v == [1.0, 2.0]
end

# A copy of held data whose elements hold no references is fresh memory: writing
# it, or passing it to a function that writes it, does not write the data the
# callable holds. Before Julia 1.11 `copy` of an `Array` is `jl_array_copy`, whose
# result the check used to follow as data of the argument, so each store into
# the copy was reported (an ODL likelihood that adds terms to a copy of held log
# weights). Julia 1.11+ copies such an array with a `memmove`, or, for a union
# such as `Union{Missing, Float64}`, with `jl_genericmemory_copyto`, which the
# check used to report as a write. (On Julia 1.12 the derivative of
# `ROCopyUnion` is wrong with `Const(f)` as well, Enzyme todo
# `2026-11-01T05-40-00-100-0pg3umn`; the check is tested against `Const(f)`.)
struct ROCopyTerms{V}
    log_weights::V
end
function (c::ROCopyTerms)(means)
    terms = copy(c.log_weights)
    for k in eachindex(c.log_weights), r in axes(means, 1)
        terms[k] += -(means[r, k]^2) / 2
    end
    return sum(terms)
end

struct ROCopyStore{V}
    v::V
end
(c::ROCopyStore)(x) = (t = copy(c.v); t[1] = x * x; sum(t))

struct ROCopyMatrix{M}
    m::M
end
(c::ROCopyMatrix)(x) = (t = copy(c.m); t .*= x; sum(abs2, t))

struct ROCopyPush{V}
    v::V
end
(c::ROCopyPush)(x) = (t = copy(c.v); push!(t, x); sum(abs2, t))

struct ROCopyUnion{V}
    u::V
end
function (c::ROCopyUnion)(x)
    t = copy(c.u)
    t[1] = x
    s = 0.0
    for v in t
        v === missing || (s += v * v)
    end
    return s
end

# Controls: each writes the data the callable holds after copying it.
struct ROCopyThenWrite{V}
    v::V
end
(c::ROCopyThenWrite)(x) = (t = copy(c.v); c.v[1] = t[2] * x; sum(c.v))

@noinline ro_pick(a, b, x) = x > 0 ? a : b
struct ROCopyOrHeld{V}
    v::V
end
(c::ROCopyOrHeld)(x) = (t = ro_pick(copy(c.v), c.v, -x); t[1] = x; sum(t))

@testset "Readonly check: writes into a copy of held data" begin
    log_weights = [log(0.3), log(0.7)]
    f = ROCopyTerms(copy(log_weights))
    means = [0.0 3.0; 0.1 2.7]
    dmeans = zero(means)
    autodiff(Reverse, f, Active, Duplicated(copy(means), dmeans))
    @test dmeans == -means
    dconst = zero(means)
    autodiff(Reverse, Const(f), Active, Duplicated(copy(means), dconst))
    @test dmeans == dconst
    @test f.log_weights == log_weights
    @test autodiff(ForwardWithPrimal, f, Duplicated(copy(means), ones(2, 2))) ==
        autodiff(ForwardWithPrimal, Const(f), Duplicated(copy(means), ones(2, 2)))

    for f in (
            ROCopyStore([0.5, 1.5, 2.5]), ROCopyMatrix([1.0 2.0; 3.0 4.0]),
            ROCopyPush([0.5, 1.5]), ROCopyUnion(Union{Missing, Float64}[1.0, missing, 2.0]),
        )
        data = deepcopy(getfield(f, 1))
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(0.7))
        @test autodiff(ReverseWithPrimal, f, Active, Active(0.7)) == expected
        rmode = set_runtime_activity(ReverseWithPrimal)
        @test autodiff(rmode, f, Active, Active(0.7)) == autodiff(rmode, Const(f), Active, Active(0.7))
        dexpected = autodiff(ForwardWithPrimal, Const(f), Duplicated(0.7, 1.0))
        @test autodiff(ForwardWithPrimal, f, Duplicated(0.7, 1.0)) == dexpected
        bexpected = autodiff(Forward, Const(f), BatchDuplicated(0.7, (1.0, 2.0)))
        @test autodiff(Forward, f, BatchDuplicated(0.7, (1.0, 2.0))) == bexpected
        @test isequal(getfield(f, 1), data)
    end
    @test autodiff(Reverse, ROCopyStore([0.5, 1.5, 2.5]), Active, Active(0.7))[1][1] ≈ 1.4

    for f in (ROCopyThenWrite([1.0, 2.0]), ROCopyOrHeld([1.0, 2.0]))
        @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
        @test f.v == [1.0, 2.0]
    end
end

# Nested differentiation through a dynamic call. The inner call's rules pass
# `readonly_args` to `thunk`, and the outer differentiation of a rule's body can
# use the inner thunk's code only where the thunk it calls is a constant: it must
# infer as one `Val`, also when no operand is checked. `ro_va_sq` ignores its
# trailing varargs, so Julia calls it through its boxed-argument ABI, which Enzyme
# handles as a dynamic call.
@noinline ro_va_sq(u, s, rest...) = s .* u .* u
ro_va_sum(u) = sum(ro_va_sq(u, 2.0, u, 3.0))
ro_dyn_sq(v, x) = sum(v .* x .* x)
struct RODynSq{M}
    v::M
end
(c::RODynSq)(x) = Base.inferencebarrier(ro_dyn_sq)(c.v, x)

ro_dir(f, u, v) = only(autodiff(Forward, f, Duplicated(u, v)))
ro_grad!(f, du, u) = (autodiff(Reverse, f, Active, Duplicated(u, du)); nothing)
ro_dx(f, x) = only(autodiff(Forward, f, Duplicated(x, 1.0)))

@testset "Readonly check: nested differentiation through a dynamic call" begin
    @test (@inferred Enzyme.Compiler.readonly_args((false, true, false, false, true, true))) === Val(false)
    @test (@inferred Enzyme.Compiler.readonly_args((false, Enzyme.Compiler.ReadonlyArg{true}(), true))) ===
        Val((false, true, false))

    u = [0.3, -1.2, 2.0]
    v = [1.0, 0.5, -2.0]
    # Reverse over Forward: a Hessian-vector product, 4v
    hv = zero(u)
    autodiff(Reverse, ro_dir, Active, Const(ro_va_sum), Duplicated(copy(u), hv), Const(v))
    @test hv ≈ 4 .* v
    # Forward over Reverse. On Julia 1.12 its Hessian-vector product through a
    # dynamic call comes back zero, with registered Enzyme as well (Enzyme todo
    # `2026-10-31T18-49-30-104-0x51ygt`); the gradient is right.
    du = zero(u)
    hv = zero(u)
    autodiff(Forward, ro_grad!, Const(ro_va_sum), Duplicated(du, hv), Duplicated(copy(u), copy(v)))
    @test du ≈ 4 .* u
    if VERSION >= v"1.12"
        @test_broken hv ≈ 4 .* v
    else
        @test hv ≈ 4 .* v
    end
    # Forward over Forward: v' H v
    @test only(autodiff(Forward, ro_dir, Const(ro_va_sum), Duplicated(copy(u), copy(v)), Const(v))) ≈
        4 * dot(v, v)

    # The same with an operand the readonly check passes on (`ReadonlyArg`)
    f = RODynSq([1.0, 2.0])
    @test autodiff(Reverse, ro_dx, Active, Const(f), Active(0.7))[1][2] ≈ 6.0
    @test only(autodiff(Forward, ro_dx, Const(f), Duplicated(0.7, 1.0))) ≈ 6.0
    @test f.v == [1.0, 2.0]
end
