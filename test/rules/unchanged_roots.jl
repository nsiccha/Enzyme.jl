module UnchangedRootsRules

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

struct MetadataRule{Large} end

@noinline function (::MetadataRule{Large})(x, y, a, ax, metadata, b) where {Large}
    return (Large ? 2 : 1) * a * sum(metadata[1]) + sum(abs2, x) + sum(y) + b
end

@noinline function metadata_reverse(x, y, a, ax, metadata, b, seed, shadow)
    shadow .+= (2seed) .* x
    return seed * sum(metadata[1])
end

@generated function argument_values(args::Tuple)
    return Expr(:tuple, [:(getfield(args, $i).val) for i in 1:length(args.parameters)]...)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, fn::Const{MetadataRule{Large}},
        ::Type{RT}, args::Vararg{Annotation, N}
    ) where {Large, RT, N}
    primal = fn.val(argument_values(args)...)
    # The tuple and its arrays are unchanged, so the reverse rule can read
    # them from its arguments without retaining another reference on tape.
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? primal : nothing, nothing, nothing
    )
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{MetadataRule{Large}},
        dret::Active, tape, args::Vararg{Annotation, N}
    ) where {Large, N}
    partial = metadata_reverse(argument_values(args)..., dret.val, args[1].dval)
    return (nothing, nothing, (Large ? 2 : 1) * partial, nothing, nothing, nothing)
end

@noinline function metadata_objective(x, y, a, v, w, n)
    vx = view(x, v)
    ax = axes(y)
    metadata = (v, w, n, true)
    # Both calls must remain: forwarding the tuple through a straight-line
    # rule call did not expose the conflicting alias metadata on its roots.
    if n > 100
        return MetadataRule{true}()(vx, y, a, ax, metadata, 0.0)
    else
        return MetadataRule{false}()(vx, y, a, ax, metadata, 0.0)
    end
end

@testset "Unchanged custom-rule tuple roots" begin
    for mode in (ReverseWithPrimal, Reverse), n in (3, 101)
        x = Float64[1, 2, 3, 4, 5, 6]
        y = Float64[2, 3, 4]
        v = [1, 3, 5]
        w = [2, 4, 6]
        dx = zeros(6)
        result = autodiff(
            mode, metadata_objective, Active,
            Duplicated(x, dx), Const(y), Active(0.7), Const(v), Const(w), Const(n)
        )
        @test result[1][3] ≈ (n > 100 ? 2 : 1) * 9
        @test dx ≈ [2, 0, 6, 0, 10, 0]
        if mode === ReverseWithPrimal
            @test result[2] ≈ (n > 100 ? 2 : 1) * 6.3 + 35 + 9
        end
        @test x == [1, 2, 3, 4, 5, 6]
        @test y == [2, 3, 4]
        @test v == [1, 3, 5]
        @test w == [2, 4, 6]
    end
end

end # module
