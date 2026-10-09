using Enzyme, Test

concat_hcat(c, a) = hcat(c, a)
concat_typed_hcat(c, a) = Float64[c a]
concat_vcat(c, a) = vcat(c, a)
concat_typed_vcat(c, a) = Float64[c; a]
concat_hvcat(c, a) = [c a; a c]
concat_stack(c, a) = stack((c, a))
concat_stack_first(c, a) = stack((c, a); dims = 1)
concat_stack_second(c, a) = stack((c, a); dims = 2)
concat_cat_first(c, a) = cat(c, a; dims = 1)
concat_cat_second(c, a) = cat(c, a; dims = 2)
concat_cat_diagonal(c, a) = cat(c, a; dims = (1, 2))
concat_cat_third(c, a) = cat(c, a; dims = 3)
concat_value(f, u, c) = f(c, exp.(u))
concat_weights(a) = reshape(collect(1.0:length(a)), size(a))
function concat_loss(f, u, c)
    value = concat_value(f, u, c)
    return sum(value .* concat_weights(value))
end

function concat_expected(f, u, c)
    weights = concat_weights(concat_value(f, u, c))
    expected = zero(u)
    for i in eachindex(u)
        basis = zero(u)
        basis[i] = exp(u[i])
        expected[i] = sum(f(zero(c), basis) .* weights)
    end
    return expected
end

@testset "Dense stack and cat with scalar and higher dimensional arrays" begin
    for shape in ((), (2, 1, 2), (0, 2, 3))
        u = reshape([0.1i for i in 1:prod(shape)], shape)
        c = reshape([0.3i for i in 1:prod(shape)], shape)
        for f in (concat_stack, concat_stack_first, concat_cat_first, concat_cat_second, concat_cat_third, concat_cat_diagonal)
            du = zero(u)
            result = autodiff(ReverseWithPrimal, concat_loss, Active, Const(f), Duplicated(u, du), Const(c))
            @test result[2] ≈ concat_loss(f, u, c)
            @test du ≈ concat_expected(f, u, c)
            result = autodiff(ForwardWithPrimal, concat_value, Duplicated, Const(f), Duplicated(u, ones(size(u))), Const(c))
            @test result[2] == concat_value(f, u, c)
            @test result[1] ≈ f(zero(c), exp.(u))
        end
    end
end

@testset "Dense concatenation keeps operand activity separate" begin
    for T in (Float32, Float64), shape in ((0,), (3,), (0, 2), (2, 0), (2, 3))
        u = reshape(T[0.1i for i in 1:prod(shape)], shape)
        c = reshape(T[0.3i for i in 1:prod(shape)], shape)
        before = (copy(u), copy(c))
        for f in (
                concat_hcat, concat_typed_hcat, concat_vcat, concat_typed_vcat,
                concat_hvcat, concat_stack, concat_stack_first, concat_stack_second,
                concat_cat_first, concat_cat_second, concat_cat_diagonal,
            )
            @testset "$f / $T / $shape" begin
                expected = concat_expected(f, u, c)
                du = ones(T, shape)
                result = autodiff(ReverseWithPrimal, concat_loss, Active, Const(f), Duplicated(u, du), Const(c))
                @test result[2] ≈ concat_loss(f, u, c)
                @test du ≈ ones(T, shape) + expected

                tangent = ones(T, shape)
                result = autodiff(ForwardWithPrimal, concat_value, Duplicated, Const(f), Duplicated(u, tangent), Const(c))
                @test result[2] == concat_value(f, u, c)
                @test result[1] ≈ f(zero(c), exp.(u))
                @test tangent == ones(T, shape)
                @test (u, c) == before
            end
        end
    end
end

concat_mixed_matrix(a, c) = hcat(c, a)
concat_ragged_blocks(a, c) = hvcat((2, 1), c, a, hcat(a, c))
concat_many_hcat(a, c) = hcat(c, a, c, a, c, a, c, a)
concat_alias_hcat(a, c) = hcat(c, a, a, c)
concat_alias_stack(a, c) = stack((c, a, a, c); dims = 1)
concat_alias_cat(a, c) = cat(c, a, a, c; dims = (1, 2))
concat_alias_loss(f, a, c) = sum(abs2, f(a, c))

@testset "Dense concatenation shapes, aliases, and batching" begin
    a, c = [0.2, 0.7], [3.0 4.0; 5.0 6.0]
    da = zero(a)
    result = autodiff(ForwardWithPrimal, concat_mixed_matrix, Duplicated, Duplicated(a, ones(2)), Const(c))
    @test result[2] == hcat(c, a)
    @test result[1] == hcat(zero(c), ones(2))
    for f in (concat_many_hcat, concat_ragged_blocks)
        c = [2.0, 3.0]
        expected = f(ones(2), zero(c))
        result = autodiff(Forward, f, BatchDuplicated, BatchDuplicated(a, (ones(2), fill(2.0, 2))), Const(c))
        @test result[1][1] == expected
        @test result[1][2] == 2 .* expected
    end
    c = [2.0, 3.0]
    for f in (concat_alias_hcat, concat_alias_stack, concat_alias_cat)
        da1, da2 = zero(a), ones(2)
        autodiff(Reverse, concat_alias_loss, Active, Const(f), BatchDuplicated(a, (da1, da2)), Const(c))
        @test da1 ≈ 4 .* a
        @test da2 ≈ 1 .+ 4 .* a
        result = autodiff(Forward, f, BatchDuplicated, BatchDuplicated(a, (ones(2), fill(2.0, 2))), Const(c))
        @test result[1][1] == f(ones(2), zero(c))
        @test result[1][2] == f(fill(2.0, 2), zero(c))
    end
    @test c == [2.0, 3.0]
end

concat_fold_vcat(xs) = reduce(vcat, xs; init = Float64[])
concat_fold_mapped_vcat(xs) = mapfoldl(reverse, vcat, xs; init = Float64[])
concat_fold_hcat(xs) = reduce(hcat, xs; init = zeros(2, 0))
concat_typed_vcat_pair(acc, x) = Float64[acc; x]
concat_fold_typed_vcat(xs) = reduce(concat_typed_vcat_pair, xs; init = Float64[])
concat_row_vcat(acc, x) = vcat(acc, permutedims(x))
concat_fold_row_vcat(xs) = reduce(concat_row_vcat, xs; init = zeros(0, 2))
concat_block_hvcat(acc, x) = [acc; x x]
concat_fold_hvcat(xs) = reduce(concat_block_hvcat, xs; init = zeros(0, 2))
concat_fold_loss(f, xs, w) = sum(f(xs) .* w)
concat_fold_tuple(a, c) = foldl(vcat, (c, a, c))
concat_fold_tuple_loss(a, c, w) = sum(concat_fold_tuple(a, c) .* w)

# An interpreter with an empty cache and no rules (so creating it leaves the shared
# rule tracking alone). Earlier inference results, e.g. of Base's tuple `sum`,
# would otherwise hide the widening below.
function concat_fresh_interp()
    world = Base.get_world_counter()
    cache = @static if VERSION >= v"1.11.0-DEV.1552"
        gensym(:concat_fold)
    else
        Enzyme.Compiler.GPUCompiler.CodeCache()
    end
    return Enzyme.Compiler.Interpreter.EnzymeInterpreter(cache, nothing, world, false, false, false)
end

# The overrides are inferred inside the reduction's own `mapfoldl_impl` frame.
# Summing operand sizes with Base's `sum` re-entered that method, and Julia 1.10
# then widened the reduction's result to `Any`.
@testset "Dense concatenation inside a reduction keeps its result type" begin
    interp = concat_fresh_interp()
    xs = [[0.5, 1.5], [2.5, 3.5]]
    for f in (
            concat_fold_vcat, concat_fold_mapped_vcat, concat_fold_hcat,
            concat_fold_typed_vcat, concat_fold_row_vcat, concat_fold_hvcat,
        )
        @testset "$f" begin
            value = f(xs)
            @test only(Base.code_typed(f, (typeof(xs),); interp))[2] === typeof(value)
            w = concat_weights(value)
            dxs = [zero(x) for x in xs]
            autodiff(Reverse, concat_fold_loss, Active, Const(f), Duplicated(xs, dxs), Const(w))
            for i in eachindex(xs), j in eachindex(xs[i])
                basis = [zero(x) for x in xs]
                basis[i][j] = 1
                @test dxs[i][j] == sum(f(basis) .* w)
            end
            @test xs == [[0.5, 1.5], [2.5, 3.5]]
        end
    end

    a, c = [0.5, 1.5], [7.0, 9.0]
    @test only(Base.code_typed(concat_fold_tuple, (Vector{Float64}, Vector{Float64}); interp))[2] === Vector{Float64}
    w = concat_weights(concat_fold_tuple(a, c))
    da = zero(a)
    autodiff(Reverse, concat_fold_tuple_loss, Active, Duplicated(a, da), Const(c), Const(w))
    @test da == w[3:4]
    @test (a, c) == ([0.5, 1.5], [7.0, 9.0])
end

function concat_error(f, args...)
    try
        f(args...)
    catch err
        return err
    end
    return nothing
end

@testset "Dense concatenation preserves shape errors" begin
    interp = Enzyme.Compiler.Interpreter
    for (original, replacement, args) in (
            (Base._typed_hcat, interp.override_typed_hcat, (Float64, ([1.0], [1.0, 2.0]))),
            (Base._typed_vcat, interp.override_typed_vcat, (Float64, ([1.0 2.0], [1.0]))),
            (Base.typed_hvcat, interp.override_typed_hvcat, (Float64, (2,), [1.0], [1.0, 2.0])),
            (Base.typed_hvcat, interp.override_typed_hvcat, (Float64, (1, 1), [1.0 2.0], [1.0])),
            (Base._stack, interp.override_stack, (:, ([1.0], [1.0, 2.0]))),
            (Base._stack, interp.override_stack, (0, ([1.0], [1.0]))),
            (Base._stack, interp.override_stack, (3, ([1.0], [1.0]))),
            (Base._cat_t, interp.override_cat_t, (0, Float64, [1.0], [1.0])),
            (Base._cat_t, interp.override_cat_t, (2, Float64, [1.0], [1.0, 2.0])),
        )
        before = deepcopy(args)
        expected = concat_error(original, args...)
        actual = concat_error(replacement, args...)
        @test expected isa Exception
        @test typeof(actual) === typeof(expected)
        @test args == before
    end
    @static if VERSION < v"1.12-"
        @test interp.override_typed_hvcat(Float64, (-1,), [1.0]) == Base.typed_hvcat(Float64, (-1,), [1.0])
        @test interp.override_typed_hvcat(Float64, (1,), [1.0], [2.0]) == Base.typed_hvcat(Float64, (1,), [1.0], [2.0])
    else
        @test_throws ArgumentError interp.override_typed_hvcat(Float64, (1,), [1.0], [2.0])
    end
end

struct HcatDispatchElement
    value::Float64
end

Base.hcat(a::Vector{HcatDispatchElement}, b::Vector{HcatDispatchElement}) =
    reshape([HcatDispatchElement(a[1].value + 2b[1].value)], 1, 1)
concat_dispatch_loss(a, b) = only(hcat(a, b)).value

@testset "Dense hcat keeps custom dispatch" begin
    a, b = [HcatDispatchElement(3.0)], [HcatDispatchElement(5.0)]
    da, db = [HcatDispatchElement(0.0)], [HcatDispatchElement(0.0)]
    result = autodiff(ReverseWithPrimal, concat_dispatch_loss, Active, Duplicated(a, da), Duplicated(b, db))
    @test result[2] == 13.0
    @test da[1].value == 1.0
    @test db[1].value == 2.0
end

struct HcatRuleElement
    value::Float64
end

concat_rule_loss(a, b) = hcat(a, b)[1].value

@testset "Dense hcat before adding a rule" begin
    a, b = [HcatRuleElement(3.0)], [HcatRuleElement(5.0)]
    da = [HcatRuleElement(1.0)]
    result = autodiff(ForwardWithPrimal, concat_rule_loss, Duplicated, Duplicated(a, da), Const(b))
    @test result[1] == 1.0
    @test result[2] == 3.0
end

function Enzyme.EnzymeRules.forward(
        config::Enzyme.EnzymeRules.FwdConfig,
        func::Const{typeof(hcat)},
        ::Type{<:Duplicated},
        a::Duplicated{Vector{HcatRuleElement}},
        b::Const{Vector{HcatRuleElement}},
    )
    primal = hcat(a.val, b.val)
    shadow = reshape([HcatRuleElement(10a.dval[1].value), HcatRuleElement(0.0)], 1, 2)
    return Duplicated(primal, shadow)
end

@testset "Dense hcat keeps a rule added after compilation" begin
    a, b = [HcatRuleElement(3.0)], [HcatRuleElement(5.0)]
    da = [HcatRuleElement(1.0)]
    result = autodiff(ForwardWithPrimal, concat_rule_loss, Duplicated, Duplicated(a, da), Const(b))
    @test result[1] == 10.0
    @test result[2] == 3.0
end
