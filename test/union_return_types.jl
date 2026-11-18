using Enzyme, Test

struct UnionReturnPayload
    a::Float64
    b::Float64
    c::Float64
end

# Intern this representation first: Ptr{Union{Tuple{Nothing, T}, Tuple{T, T}}}
# can then reuse Ptr{Tuple{Union{Nothing, T}, T}}. Its element type is abstract,
# although the union return buffer holds the unboxed Float64 fields.
const CachedUnionReturnPointer = Ptr{Tuple{Union{Nothing, UnionReturnPayload}, UnionReturnPayload}}

@noinline function union_return_payload(x)
    value = UnionReturnPayload(x[1]^2, x[2]^2, x[3]^2)
    if x[1] < 0
        return nothing, value
    end
    return value, value
end

function union_return_sum(x)
    value = last(union_return_payload(x))
    return value.a + value.b + value.c
end

@testset "Union return buffer after equivalent pointer type" begin
    for x in ([1.0, 2.0, 3.0], [-1.0, 2.0, 3.0])
        @test only(Enzyme.gradient(Reverse, union_return_sum, x)) ≈ 2 .* x
    end
end
