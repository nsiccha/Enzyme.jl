module CustomRuleReturnCache

using Enzyme, Test
using Enzyme.EnzymeRules

# A scalar rule shares its exponential between the primal and the partial.
# Only the partial belongs on the tape of a linear reduction, even though
# Julia returns the primal and tape together through one sret buffer.
softplus(x) = ifelse(x >= 0, x, zero(x)) + log1p(exp(-abs(x)))

function EnzymeRules.augmented_primal(
        config::RevConfig, ::Const{typeof(softplus)}, ::Type{<:Active}, x::Active
    )
    positive = x.val >= 0
    q = exp(ifelse(positive, -x.val, x.val))
    primal = ifelse(positive, x.val, zero(x.val)) + log1p(q)
    partial = ifelse(positive, one(q), q) / (1 + q)
    return AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, (partial,))
end

scale_partial(dy::Active, partial) = dy.val * partial

function EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(softplus)}, dy::Active, tape, x::Active
    )
    return (scale_partial(dy, tape[1]),)
end

function EnzymeRules.reverse(
        ::RevConfig, ::Const{typeof(softplus)}, dy::Tuple, tape, x::Active
    )
    return (map(Base.Fix2(scale_partial, tape[1]), dy),)
end

function objective(x, y, ::Val{Squared}) where {Squared}
    value = 0.0
    @inbounds @simd for i in eachindex(x, y)
        s = softplus(x[i])
        value += y[i] * x[i] - (Squared ? s * s : s)
    end
    return value
end

function gradient(x, y)
    dx = zero(x)
    autodiff(Reverse, objective, Active, Duplicated(x, dx), Const(y), Const(Val(false)))
    return dx
end

# A fixed trip count exposes the native allocation sizes in LLVM IR, avoiding
# Julia's allocation counters, which do not count Enzyme's malloc-backed tape.
function fixed_objective(x, y)
    value = 0.0
    @inbounds @simd for i in 1:128
        value += y[i] * x[i] - softplus(x[i])
    end
    return value
end

@testset "Custom-rule return fields" begin
    for n in (0, 1, 17, 257), squared in (false, true)
        x = sin.(collect(1:n)) .* 40
        y = Float64.(isodd.(1:n))
        input = copy(x)
        partials = 1 ./ (1 .+ exp.(-x))
        expected = squared ? y .- 2 .* softplus.(x) .* partials : y .- partials
        for mode in (Reverse, ReverseWithPrimal)
            dx = zero(x)
            result = autodiff(
                mode, objective, Active, Duplicated(x, dx), Const(y), Const(Val(squared))
            )
            @test dx ≈ expected
            if mode === ReverseWithPrimal
                @test result[2] ≈ objective(x, y, Val(squared))
            end
        end
        dx1, dx2 = zero(x), zero(x)
        autodiff(
            Reverse, objective, Active, BatchDuplicated(x, (dx1, dx2)),
            Const(y), Const(Val(squared))
        )
        @test dx1 ≈ expected
        @test dx2 ≈ expected

        args = (Duplicated(x, zero(x)), Const(y), Const(Val(squared)))
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(objective)}, Active, map(typeof, args)...
        )
        tape, primal, _ = fwd(Const(objective), args...)
        rev(Const(objective), args..., 1.0, tape)
        @test primal ≈ objective(x, y, Val(squared))
        @test args[1].dval ≈ expected
        @test x == input
    end

    x = [-2.0, -0.2, 0.3, 2.0]
    y = [0.0, 1.0, 0.0, 1.0]
    direction = [0.2, -0.7, 1.0, 0.5]
    partials = 1 ./ (1 .+ exp.(-x))
    hvp = only(autodiff(Forward, gradient, Duplicated(x, direction), Const(y)))
    @test hvp ≈ -partials .* (1 .- partials) .* direction

    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(
        io, fixed_objective, Active,
        Tuple{Duplicated{Vector{Float64}}, Const{Vector{Float64}}};
        returnPrimal = true, dump_module = true, debuginfo = :none
    )
    ir = String(take!(io))
    allocations = collect(eachmatch(r"\bcall\b[^\n]*@malloc\(i(?:32|64) (\d+)\)", ir))
    # Account for every malloc call, including a possible dynamically sized
    # allocation. Only one Float64 partial per iteration needs to survive.
    @test length(allocations) == count(r"\bcall\b[^\n]*@malloc\(", ir)
    @test sum(parse(Int, m.captures[1]) for m in allocations; init = 0) <= 128 * sizeof(Float64)
end

end
