using Enzyme, Test

gc_bundle_nested(x) = sum([x][1])

function gc_bundle_push(x)
    values = Vector{Float64}[]
    push!(values, x)
    return sum(values[1])
end

function gc_bundle_array(x)
    values = copy(x)
    return sum(abs2, values)
end

@testset "Julia GC allocation initialization bundles" begin
    # Julia's allocation bundles describe pointer initialization performed by
    # late GC lowering. Shadow allocations need the same initialization; the
    # bundles must survive both forward and reverse differentiation.
    x = [1.0, 2.0]
    for f in (gc_bundle_nested, gc_bundle_push)
        @test Enzyme.gradient(Forward, Const(f), x)[1] ≈ ones(2)
        @test Enzyme.gradient(Reverse, Const(f), x)[1] ≈ ones(2)
    end
    @test Enzyme.gradient(Forward, Const(gc_bundle_array), x)[1] ≈ 2x
    @test Enzyme.gradient(Reverse, Const(gc_bundle_array), x)[1] ≈ 2x
end
