_expnegexp(x::Float64) = exp(-exp(x))

# With a = exp(x), the nth derivative is exp(-a) P_n(a), where
# P_1 = -a and P_{n+1} = a (P_n' - P_n). The differentiation order is
# known at compilation. Omit the constant coefficient, which is always zero.
function _expnegexp_next(::Val{C}) where {C}
    degrees = ntuple(identity, Val(length(C)))
    differentiated = map(*, degrees, C)
    return Val(map(-, (differentiated..., 0.0), (0.0, C...)))
end

function _expnegexp_polynomial(x, a, ::Val{C}) where {C}
    # For the second derivative, preserve the zero at x = 0 and its slope
    # even when exp(x) rounds to one.
    if length(C) == 2
        return x > 0 ? -expm1(-x) : expm1(x)
    end
    return x > 0 ? evalpoly(inv(a), reverse(C)) : evalpoly(a, C)
end

function _expnegexp_seeded(x, polynomial::Val{C}, seeds::Tuple) where {C}
    a = exp(x)
    # Factor out the highest power of a on the right and a on the left. Both Horner
    # evaluations then have an argument in [0, 1].
    z = isinf(a) ? -Inf : (x > 0 ? length(C) * x : x) - a
    m, e = frexp(_expnegexp_polynomial(x, a, polynomial))
    for seed in seeds
        s, k = frexp(seed)
        m, carry = frexp(m * s)
        e += k + carry
    end
    isnan(z) && return z
    # Even after applying all the seeds, this is below half a subnormal.
    # The loose bound also keeps the integer range reduction below in range.
    z < -2.0 * (abs(e) + 1075) && return 0.0 * m
    # In this interval, multiplying exp(z) by a normalized significand
    # stays in the normal range. The final ldexp applies the seeds' scale.
    if abs(z) < 500
        return ldexp(m * exp(z), e)
    end
    k = round(Int, z * log2(MathConstants.e))
    # Split log(2) to retain the residual after removing k powers of two.
    r = fma(-Float64(k), 0.6931471803691238, z)
    r = fma(-Float64(k), 1.9082149292705877e-10, r)
    return ldexp(m * exp(r), k + e)
end

# A further derivative in x appends its seed. A derivative in an earlier
# seed replaces just that factor, including when the earlier seed is zero.
struct ExpNegExpPartial{C, S}
    x::Float64
    polynomial::Val{C}
    seeds::S
end
_expnegexp_add(::Type, v) = v
_expnegexp_add(prev, v) = prev + v

function EnzymeRules.multiply_fwd_into(prev, p::ExpNegExpPartial, d::Float64)
    return _expnegexp_add(prev, _expnegexp_seeded(p.x, p.polynomial, (p.seeds..., d)))
end
function EnzymeRules.multiply_rev_into(prev, p::ExpNegExpPartial, d::Float64)
    return EnzymeRules.multiply_fwd_into(prev, p, d)
end

struct ExpNegExpSeeds{P}
    partial::P
end
function EnzymeRules.multiply_fwd_into(prev, ds::ExpNegExpSeeds, d::Tuple)
    p = ds.partial
    for i in eachindex(d)
        prev = _expnegexp_add(prev, _expnegexp_seeded(p.x, p.polynomial, Base.setindex(p.seeds, d[i], i)))
    end
    return prev
end
struct ExpNegExpSeedPullback{P}
    partial::P
    seed::Float64
end
function (pb::ExpNegExpSeedPullback)(i)
    p = pb.partial
    return _expnegexp_seeded(p.x, p.polynomial, Base.setindex(p.seeds, pb.seed, i))
end
function EnzymeRules.multiply_rev_into(prev, ds::ExpNegExpSeeds, d::Float64)
    p = ds.partial
    v = ntuple(ExpNegExpSeedPullback(p, d), Val(length(p.seeds)))
    return prev isa Type ? v : map(+, prev, v)
end

EnzymeRules.@easy_rule(_expnegexp(x::Float64), (ExpNegExpPartial(x, Val((-1.0,)), ()),))
EnzymeRules.@easy_rule(
    _expnegexp_seeded(x::Float64, polynomial::Val, seeds::Tuple),
    (
        ExpNegExpPartial(x, _expnegexp_next(polynomial), seeds),
        @Constant,
        ExpNegExpSeeds(ExpNegExpPartial(x, polynomial, seeds)),
    ),
)

_hypotvalue(x) = x.val

# Change the evaluation order when x * dx leaves the normal range. Always
# normalizing first can instead lose a tiny x / h that a large seed rescales.
function _hypotmul(x, dx, h)
    product = x * dx
    if isfinite(product) && abs(product) >= floatmin(real(product))
        return product / h
    else
        return (x / h) * dx
    end
end

_hypotforward(x::Const, h, i) = zero(real(x.val))
function _hypotforward(x, h, i)
    dx = i === nothing ? x.dval : x.dval[i]
    return real(_hypotmul(conj(x.val), dx, h))
end

struct HypotForward{A, H}
    args::A
    h::H
end
# Convert to the primal's type, as upstream #3981 does: mixed-precision arguments
# give a tangent of another type than `hypot`'s result.
(f::HypotForward)(i) = convert(typeof(f.h), sum(_hypotforward.(f.args, f.h, i)))

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.hypot)},
        RT,
        x::Annotation,
        y::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    args = (x, y, xs...)
    if EnzymeRules.needs_shadow(config)
        h = func.val(map(_hypotvalue, args)...)
        n = iszero(h) ? one(h) : h
        tangent = HypotForward(args, n)
        if EnzymeRules.width(config) == 1
            dh = tangent(nothing)
            if EnzymeRules.needs_primal(config)
                return Duplicated(h, dh)
            else
                return dh
            end
        else
            dh = ntuple(tangent, Val(EnzymeRules.width(config)))
            if EnzymeRules.needs_primal(config)
                return BatchDuplicated(h, dh)
            else
                return dh
            end
        end
    elseif EnzymeRules.needs_primal(config)
        return func.val(map(_hypotvalue, args)...)
    else
        return nothing
    end
end

_hypotreverse(x::Const, ::Val{W}, ::Type{<:Const}, h) where {W} = nothing
_hypotreverse(x::Const, ::Val{W}, dret, h) where {W} = nothing
function _hypotreverse(x, w::Val{W}, ::Type{<:Const}, h) where {W}
    if W == 1
        return zero(x.val)
    else
        return ntuple(Returns(zero(x.val)), w)
    end
end
function _hypotreverse(x, w::Val{W}, dret, h) where {W}
    if W == 1
        return _hypotreverse_seed((x.val, h), dret)
    else
        return map(Base.Fix1(_hypotreverse_seed, (x.val, h)), dret)
    end
end

function _hypotreverse_seed(x_h, dret)
    x, h = x_h
    return oftype(x, _hypotmul(x, dret.val, h))
end

struct HypotReverse{W, D, H}
    width::Val{W}
    dret::D
    h::H
end
(r::HypotReverse)(x) = _hypotreverse(x, r.width, r.dret, r.h)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.hypot)},
        ::Type,
        x::Annotation,
        y::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    h = hypot(map(_hypotvalue, (x, y, xs...))...)
    primal = needs_primal(config) ? h : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.hypot)},
        dret,
        tape,
        x::Annotation,
        y::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    args = (x, y, xs...)
    h = hypot(map(_hypotvalue, args)...)
    n = iszero(h) ? one(h) : h
    w = Val(EnzymeRules.width(config))
    return map(HypotReverse(w, dret, n), args)
end
