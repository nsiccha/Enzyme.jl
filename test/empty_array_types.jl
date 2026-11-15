using Enzyme, Test

struct CollectionValue{T}
    value::T
end

# The constructor map has an EltypeUnknown empty branch. Its empty Array{Any}
# and nonempty inline-struct array meet inside _collect, before this dispatch.
Base.@noinline function collection_logsum(values::AbstractArray{<:CollectionValue})
    result = 0.0
    for value in values
        result += log(value.value)
    end
    return result
end

constructor_map_logsum(x) = collection_logsum(map(CollectionValue, x))
constructor_map_allow_empty(x) = isempty(x) ? 0.0 : constructor_map_logsum(x)

function initially_empty_grown(x)
    values = eltype(x)[]
    push!(values, x[1])
    resize!(values, 2)
    values[2] = x[2]
    return sum(abs2, values)
end

function initially_empty_references(x)
    values = typeof(x)[]
    push!(values, x)
    return sum(abs2, values[1])
end

# This repair relies on the Memory-backed array representation introduced in 1.11.
@static if VERSION >= v"1.11-"
    @testset "Empty constructor-map element layout" begin
        for T in (Float32, Float64), dims in ((4,), (2, 2))
            x = reshape(T[0.2, 0.3, 0.4, 0.5], dims)
            original = copy(x)
            dx = fill(T(0.5), dims)
            _, primal = autodiff(ReverseWithPrimal, constructor_map_logsum, Active, Duplicated(x, dx))
            @test primal ≈ sum(log, x)
            @test dx ≈ T(0.5) .+ inv.(x)
            @test x == original
            tangent = autodiff(Forward, constructor_map_logsum, Duplicated(x, ones(T, dims)))[1]
            @test tangent ≈ sum(inv, x)
        end
        for dims in ((0,), (0, 2))
            x = zeros(dims)
            dx = zero(x)
            _, primal = autodiff(ReverseWithPrimal, constructor_map_allow_empty, Active, Duplicated(x, dx))
            @test primal == 0.0
            @test dx == x
        end
    end
end

@testset "Initially empty arrays can acquire element storage" begin
    for f in (initially_empty_grown, initially_empty_references)
        x = [0.2, 0.3]
        original = copy(x)
        dx = zero(x)
        _, primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx))
        @test primal ≈ sum(abs2, x)
        @test dx ≈ 2 .* x
        @test x == original
        @test autodiff(Forward, f, Duplicated(x, ones(2)))[1] ≈ 2 * sum(x)
    end
end
