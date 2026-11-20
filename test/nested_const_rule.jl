using Enzyme, Test
using Enzyme: EnzymeRules

# Forward over reverse through a custom rule that takes a `Const` array. The inner
# reverse pass keeps the array on its tape for the rule's reverse; when the outer
# forward pass differentiates the code that builds that tape, the array is constant
# there too. Each Hessian-vector product below is exact under static activity.

@noinline nested_scalemul!(y, c, x) = (y .= c .* x; y)

# The tangent of c .* x, from c's tangent when c is active.
nested_add_c_tangent!(dy, c::Const, x) = dy
nested_add_c_tangent!(dy, c::Duplicated, x) = (dy .+= c.dval .* x; dy)

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, func::Const{typeof(nested_scalemul!)}, RT::Type,
        y::Duplicated, c::Union{Const, Duplicated}, x::Duplicated
    )
    func.val(y.val, c.val, x.val)
    y.dval .= c.val .* x.dval
    nested_add_c_tangent!(y.dval, c, x.val)
    np, ns = EnzymeRules.needs_primal(config), EnzymeRules.needs_shadow(config)
    return np && ns ? Duplicated(y.val, y.dval) : ns ? y.dval : np ? y.val : nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(nested_scalemul!)}, ::Type{RT},
        y::Duplicated, c::Const, x::Duplicated
    ) where {RT}
    func.val(y.val, c.val, x.val)
    primal = EnzymeRules.needs_primal(config) ? y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? y.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(nested_scalemul!)}, ::Type{RT},
        tape, y::Duplicated, c::Const, x::Duplicated
    ) where {RT}
    x.dval .+= c.val .* y.dval
    fill!(y.dval, 0)
    return (nothing, nothing, nothing)
end

struct NestedConstWrap{T}
    c::T
end

struct NestedConstSwitch{T}
    c::T
    on::Bool
end

# Each objective is -sum(abs2, c .* x) / 2: gradient -(c .^ 2) .* x, Hessian diag(-c .^ 2).
nested_inlined(x, c) = -sum(abs2, nested_scalemul!(similar(x), c, x)) / 2
@noinline nested_callee(c, x) = nested_scalemul!(similar(x), c, x)
nested_noinline(x, c) = -sum(abs2, nested_callee(c, x)) / 2
@noinline nested_callee_branch(c, x) = sum(c) > 0 ? nested_scalemul!(similar(x), c, x) : copy(x)
nested_branch(x, c) = -sum(abs2, nested_callee_branch(c, x)) / 2
@noinline nested_callee_field(w, x) = nested_scalemul!(similar(x), w.c, x)
nested_field(x, w) = -sum(abs2, nested_callee_field(w, x)) / 2
# The field reaches the tape on one branch only.
@noinline nested_callee_switch(w, x) = w.on ? nested_scalemul!(similar(x), w.c, x) : x .* w.c
nested_switch(x, w) = -sum(abs2, nested_callee_switch(w, x)) / 2

function nested_grad!(dx, f, x, c)
    fill!(dx, 0)
    autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, dx), Const(c))
    return nothing
end

function nested_hvp(mode, f, x, v, c)
    dx, hv = zero(x), zero(x)
    autodiff(mode, Const(nested_grad!), Const, Duplicated(dx, hv), Const(f), Duplicated(x, copy(v)), Const(c))
    return dx, hv
end

@testset "Forward over reverse through a custom rule taking a Const array" begin
    c = [2.0, 3.0, 0.5]
    x = [0.3, -0.7, 1.1]
    v = [1.0, 0.5, -0.25]
    cases = (
        (nested_inlined, c),
        (nested_noinline, c),
        (nested_branch, c),
        (nested_field, NestedConstWrap(c)),
        (nested_switch, NestedConstSwitch(c, true)),
        (nested_switch, NestedConstSwitch(c, false)),
    )
    @testset "$f" for (f, arg) in cases
        for mode in (Forward, set_runtime_activity(Forward))
            dx, hv = nested_hvp(mode, f, x, v, arg)
            @test dx ≈ -(c .^ 2) .* x
            @test hv ≈ -(c .^ 2) .* v
        end
        @test c == [2.0, 3.0, 0.5]
    end

    # Active at the outer level, the array's own tangent reaches the gradient.
    dc = [0.5, -1.0, 2.0]
    dx, hv = zero(x), zero(x)
    autodiff(
        Forward, Const(nested_grad!), Const, Duplicated(dx, hv), Const(nested_noinline),
        Duplicated(x, zero(x)), Duplicated(c, dc)
    )
    @test dx ≈ -(c .^ 2) .* x
    @test hv ≈ -2 .* c .* dc .* x
end
