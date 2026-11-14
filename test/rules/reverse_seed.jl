using Enzyme
using Enzyme: EnzymeRules
using Test

seed_square(x) = x * x

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{typeof(seed_square)},
        ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

@noinline function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(seed_square)}, dy::Active,
        ::Nothing, x::Active
    )
    return (2 * x.val * dy.val,)
end

scale_seed(scale, dy) = scale * dy.val

@noinline function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(seed_square)}, dy::NTuple{W, Active},
        ::Nothing, x::Active
    ) where {W}
    return (map(Base.Fix1(scale_seed, 2 * x.val), dy),)
end

weighted_square(x, weight) = weight * seed_square(x)
seed_gradient(x, weight) = autodiff(ReverseWithPrimal, weighted_square, Active(x), Const(weight))
seed_derivative(x, weight) = first(first(seed_gradient(x, weight)))

function seed_square!(out, x)
    out[] = seed_square(x[])
    return nothing
end

function seed_batch!(out, x)
    autodiff(Reverse, seed_square!, out, x)
    return nothing
end

# A tuple result takes the seed through the aggregate-return path.
seed_pair(x) = (x * x, x * x * x)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{typeof(seed_pair)},
        ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

@noinline function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(seed_pair)}, dy::Active,
        ::Nothing, x::Active
    )
    return (2 * x.val * dy.val[1] + 3 * x.val^2 * dy.val[2],)
end

function weighted_pair(x, weight)
    a, b = seed_pair(x)
    return weight * a + 2 * weight * b
end

pair_gradient(x, weight) = autodiff(ReverseWithPrimal, weighted_pair, Active(x), Const(weight))

# Saving an immutable seed must save its value, not the caller's temporary slot.
saved_square(x, saved) = x * x

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{typeof(saved_square)},
        ::Type{<:Active}, x::Active, saved::Const
    )
    primal = EnzymeRules.needs_primal(config) ? f.val(x.val, saved.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

@noinline function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(saved_square)}, dy::Active,
        ::Nothing, x::Active, saved::Const
    )
    saved.val[] = dy
    return (2 * x.val * dy.val, nothing)
end

weighted_saved(x, weight, saved) = weight * saved_square(x, saved)
saved_gradient(x, weight, saved) = autodiff(Reverse, weighted_saved, Active(x), Const(weight), Const(saved))

function check_seed_allocations(x, weight)
    seed_gradient(x, weight)
    pair_gradient(x, weight)
    @test (@allocated seed_gradient(x, weight)) == 0
    @test (@allocated pair_gradient(x, weight)) == 0
    return nothing
end

function check_batch_seed_allocations(out, x)
    seed_batch!(out, x)
    @test (@allocated seed_batch!(out, x)) == 0
    return nothing
end

@testset "Active seeds at a retained reverse-rule call" begin
    for x in (0.4, -1.3), weight in (0.7, -2.0)
        @test seed_gradient(x, weight) == ((2 * x * weight, nothing), weight * x^2)
        gradient, primal = pair_gradient(x, weight)
        @test gradient[1] ≈ 2 * x * weight + 6 * x^2 * weight
        @test primal ≈ weight * x^2 + 2 * weight * x^3
        @test autodiff(Forward, seed_derivative, Duplicated(x, 1.0), Const(weight))[1] ≈ 2 * weight
        check_seed_allocations(x, weight)
    end

    out = BatchDuplicated(Ref(0.0), (Ref(0.7), Ref(-2.0)))
    x = BatchDuplicated(Ref(1.3), (Ref(0.0), Ref(0.0)))
    seed_batch!(out, x)
    @test out.val[] ≈ 1.3^2
    @test x.dval[1][] ≈ 2 * 1.3 * 0.7
    @test x.dval[2][] ≈ 2 * 1.3 * -2.0
    check_batch_seed_allocations(out, x)

    saved = Ref{Any}()
    saved_gradient(0.4, 0.7, saved)
    first_seed = saved[]
    saved_gradient(-1.3, -2.0, saved)
    GC.gc()
    @test first_seed isa Active{Float64}
    @test first_seed.val == 0.7
    @test saved[].val == -2.0

    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(
        io, weighted_square, Active, Tuple{Active{Float64}, Const{Float64}};
        returnPrimal = true, raw = true, dump_module = true
    )
    ir = String(Base.take!(io))
    @test occursin(r"\bcall\b[^\n]*@ejl_enzyme_reverse_", ir)
    @test !occursin(r"\bcall\b[^\n]*@(?:ijl_gc_small_alloc|ijl_gc_alloc_typed|julia\.gc_alloc_obj)", ir)
end
