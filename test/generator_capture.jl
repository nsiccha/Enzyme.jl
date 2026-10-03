using Enzyme, Test

struct GeneratorPrepared{S}
    state::S
end

generator_reader(b, x, idx) =
    [b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx)]
range_generator_reader(b, x, idx) =
    [b[idx[i], 1] + b[idx[i], 2] * x[i] for i in 1:length(idx)]
zip_generator_reader(b, x, idx) =
    [b[j, 1] + b[j, 2] * xi for (j, xi) in zip(idx, x)]
lazy_generator_reader(b, x, idx, keep) =
    [keep[i] ? b[idx[i], 1] + b[idx[i], 2] * x[i] : b[idx[i], 1]
     for i in eachindex(idx)]
function prepared_generator_reader(b, prepared)
    data = prepared.state
    generator_reader(b, data.x, data.idx)
end
generator_sum(reader, b, args...) = sum(reader(b, args...))

@testset "Generator capture regression on Julia 1.10" begin
# Newer Julia uses a different array allocation path. Its remaining activity
# boundary is unchanged by this policy; this regression covers the 1.10 fix.
if VERSION < v"1.11-"
@testset "Generator captures with fixed arrays" begin
    @test !Enzyme.runtime_activity(Enzyme.Reverse)
    for T in (Float32, Float64), X in (Float32, Float64), n in (0, 1, 4, 17)
        b = reshape(T[-0.2, 0.1, 0.3, -0.1], 2, 2)
        x = X[(i - 2) / 4 for i in 1:n]
        idx = Int[mod1(i, 2) for i in 1:n]
        prepared = GeneratorPrepared((; x, idx))
        expected = zero(b)
        reference = promote_type(T, X)[]
        for i in eachindex(idx)
            expected[idx[i], 1] += one(T)
            expected[idx[i], 2] += x[i]
            push!(reference, b[idx[i], 1] + b[idx[i], 2] * x[i])
        end
        snapshot = deepcopy((b, x, idx))
        for (reader, args) in ((generator_reader, (x, idx)),
                               (range_generator_reader, (x, idx)),
                               (zip_generator_reader, (x, idx)),
                               (prepared_generator_reader, (prepared,)))
            @test reader(b, args...) == reference
            @test eltype(reader(b, args...)) == promote_type(T, X)
            for _ in 1:2
                db = zero(b)
                Enzyme.autodiff(Enzyme.Reverse, generator_sum, Enzyme.Active,
                    Enzyme.Const(reader), Enzyme.Duplicated(b, db),
                    map(Enzyme.Const, args)...)
                @test db ≈ expected
                @test isequal((b, x, idx), snapshot)
            end
        end
    end
end

@testset "Generator lazy branches" begin
    for n in (0, 1, 4, 17)
        b = reshape([-0.2, 0.1, 0.3, -0.1], 2, 2)
        idx = Int[mod1(i, 2) for i in 1:n]
        # Only the first element may read x. Later x accesses are out of bounds
        # and must stay inactive in both ordinary and differentiated execution.
        short_x = n == 0 ? Float64[] : [0.25]
        first_only = Bool[i == 1 for i in 1:n]
        snapshot = deepcopy((b, short_x, idx, first_only))
        expected_lazy = zero(b)
        for i in eachindex(idx)
            expected_lazy[idx[i], 1] += 1
            if first_only[i]
                expected_lazy[idx[i], 2] += short_x[i]
            end
        end
        @test length(lazy_generator_reader(b, short_x, idx, first_only)) == n
        for _ in 1:2
            db = zero(b)
            Enzyme.autodiff(Enzyme.Reverse, generator_sum, Enzyme.Active,
                Enzyme.Const(lazy_generator_reader), Enzyme.Duplicated(b, db),
                Enzyme.Const(short_x), Enzyme.Const(idx), Enzyme.Const(first_only))
            @test db ≈ expected_lazy
            @test isequal((b, short_x, idx, first_only), snapshot)
        end
    end
end

@testset "Both generator array captures active" begin
    b = reshape([-0.2, 0.1, 0.3, -0.1], 2, 2)
    x = [-0.5, 0.0, 0.5, 1.0]
    idx = [1, 1, 2, 2]
    db = zero(b)
    dx = zero(x)
    Enzyme.autodiff(Enzyme.Reverse, generator_sum, Enzyme.Active,
        Enzyme.Const(generator_reader), Enzyme.Duplicated(b, db),
        Enzyme.Duplicated(x, dx), Enzyme.Const(idx))
    @test db ≈ [2 -0.5; 2 1.5]
    @test dx ≈ [0.3, 0.3, -0.1, -0.1]
end

@testset "Generator forward mode" begin
    b = reshape([-0.2, 0.1, 0.3, -0.1], 2, 2)
    x = [-0.5, 0.0, 0.5, 1.0]
    idx = [1, 1, 2, 2]
    snapshot = deepcopy((b, x, idx))
    result = Enzyme.autodiff(Enzyme.ForwardWithPrimal, generator_sum,
        Enzyme.Const(generator_reader), Enzyme.Duplicated(b, ones(size(b))),
        Enzyme.Const(x), Enzyme.Const(idx))
    @test result[1] ≈ 5.0
    @test result[2] ≈ -0.5
    @test isequal((b, x, idx), snapshot)
end
else
    @test_skip false
end
end

# Marker derivatives distinguish rule dispatch from differentiation of the body.
# These test-only rules are unrelated to the default-mode compiler repair.
struct GeneratorForwardMap end
(::GeneratorForwardMap)(x) = 2 * x
forward_generator_sum(x) = sum(collect(Base.Generator(GeneratorForwardMap(), x)))

function Enzyme.EnzymeRules.forward(
    config::Enzyme.EnzymeRules.FwdConfigWidth{1}, ::Enzyme.Const{typeof(Base.collect)},
    ::Type, g::Enzyme.Duplicated{<:Base.Generator{I,GeneratorForwardMap}}
) where {I}
    shadow = 3 .* g.dval.iter
    return Enzyme.EnzymeRules.needs_primal(config) ?
        Enzyme.Duplicated(collect(g.val), shadow) : shadow
end

struct GeneratorReverseMap end
(::GeneratorReverseMap)(x) = 2 * x
reverse_generator_sum(x) = sum(collect(Base.Generator(GeneratorReverseMap(), x)))

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfigWidth{1}, ::Enzyme.Const{typeof(Base.collect)},
    ::Type, g::Enzyme.Duplicated{<:Base.Generator{I,GeneratorReverseMap}}
) where {I}
    ret = collect(g.val)
    shadow = zero(ret)
    return Enzyme.EnzymeRules.AugmentedReturn(
        Enzyme.EnzymeRules.needs_primal(config) ? ret : nothing,
        Enzyme.EnzymeRules.needs_shadow(config) ? shadow : nothing, shadow)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfigWidth{1}, ::Enzyme.Const{typeof(Base.collect)},
    ::Type, tape, g::Enzyme.Duplicated{<:Base.Generator{I,GeneratorReverseMap}}
) where {I}
    g.dval.iter .+= 5 .* tape
    return (nothing,)
end

@testset "Generator collection preserves explicit differentiation rules" begin
    x = [1.0, 2.0]
    snapshot = copy(x)
    @test forward_generator_sum(x) == 6.0
    @test Enzyme.autodiff(Enzyme.Forward, forward_generator_sum,
        Enzyme.Duplicated(x, ones(2)))[1] == 6.0
    @test Enzyme.autodiff(Enzyme.ForwardWithPrimal, forward_generator_sum,
        Enzyme.Duplicated(x, ones(2))) == (6.0, 6.0)
    @test reverse_generator_sum(x) == 6.0
    for _ in 1:2
        dx = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, reverse_generator_sum, Enzyme.Active,
            Enzyme.Duplicated(x, dx))
        @test dx == fill(5.0, 2)
        @test isequal(x, snapshot)
    end
end

# These private callable types isolate rule lookup to this test's generators.
# The compiler repair must preserve an explicitly registered rule.
struct GeneratorInactiveMap end
(::GeneratorInactiveMap)(x) = 2 * x
Enzyme.EnzymeRules.inactive(
    ::typeof(Base.collect), ::Base.Generator{I,GeneratorInactiveMap}
) where {I} = nothing
inactive_generator_sum(x) = sum(collect(Base.Generator(GeneratorInactiveMap(), x)))

@testset "Generator collection preserves explicit inactivity" begin
    x = [1.0, 2.0]
    snapshot = copy(x)
    @test inactive_generator_sum(x) == 6.0
    for _ in 1:2
        dx = zero(x)
        Enzyme.autodiff(Enzyme.Reverse, inactive_generator_sum, Enzyme.Active,
            Enzyme.Duplicated(x, dx))
        @test dx == zero(x)
        @test isequal(x, snapshot)
    end
end
