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
# neither walked nor read-only, so the call that passes it a field of the
# callable by reference used to be reported as a write, although the
# parameter's type cannot hold derivative data (a ReactiveKernels derivative
# rule hoisted out of a loop). The parameter type now decides.
struct RORule
    a::Bool
    b::NTuple{3, Bool}
end
@noinline (r::RORule)(x, y) = r.a ? x * y + r.b[2] : x + y

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
# that only reads it but is neither walked nor read-only (it allocates, or has
# a custom rule) because it has a second call site.
@noinline ro_box_scaled(b::ROBox, x) = sum(b.v .* x)
struct ROAllocBox{M}
    box::ROBox{M}
end
(c::ROAllocBox)(x) = ro_box_scaled(c.box, x) + ro_box_scaled(ROBox([1.0]), x)

@noinline ro_box_rule(b::ROBox, x) = sum(b.v) * x
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
