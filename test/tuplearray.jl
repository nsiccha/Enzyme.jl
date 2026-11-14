using Enzyme, Test

@testset "TupleArray array interface" begin
    for shape in ((), (4,), (2, 2), (2, 1, 2), (0,), (2, 0))
        n = prod(shape)
        data = ntuple(Float64, n)
        a = Enzyme.TupleArray{Float64, n, length(shape)}(data, shape)
        expected = reshape(collect(Float64, data), shape)

        @test size(a) == shape
        @test ndims(a) == length(shape)
        @test ndims(typeof(a)) == length(shape)
        @test eltype(a) === Float64
        @test eltype(typeof(a)) === Float64
        @test collect(a) == expected
        @test a == expected
        @test convert(Array{Float64, length(shape)}, a) == expected
        @test convert(Array{Int, length(shape)}, a) == expected
        @test convert(Array, a) == expected

        # Converting an immutable tuple-backed array must give independent storage.
        b = convert(Array{Float64, length(shape)}, a)
        if n > 0
            b[1] = -1
            @test a[1] == data[1]
        end
    end

    a = Enzyme.TupleArray{Any, 2, 1}((1, "two"), (2,))
    @test eltype(a) === Any
    @test collect(a) == Any[1, "two"]
    @test convert(Vector{Any}, a) == Any[1, "two"]
end

@testset "Loading Enzyme preserves abstract array callers" begin
    # The test runner has already loaded Enzyme. Start a fresh process so the
    # callers below are compiled before Enzyme adds any methods to Base.
    script = raw"""
    using Test, LLVM

    # Load LLVM (and CEnum) before warming the callers: this measures Enzyme's
    # methods, independently of invalidations caused by loading its dependencies.

    module ArrayConsumer
        dims(a) = ndims(Base.inferencebarrier(a)::AbstractArray)
        element(a) = eltype(typeof(Base.inferencebarrier(a)::AbstractArray))
        convert_array(a) = convert(Vector{Float64}, Base.inferencebarrier(a)::AbstractVector{Float64})
    end

    function invalidated_callers(invalidations)
        callers = Symbol[]
        for entry in invalidations
            if entry isa Core.MethodInstance && entry.def isa Method && entry.def.module === ArrayConsumer
                push!(callers, entry.def.name)
            end
        end
        return callers
    end

    ArrayConsumer.dims([1.0])
    ArrayConsumer.element([1.0])
    ArrayConsumer.convert_array([1.0])
    invalidations = ccall(:jl_debug_method_invalidation, Any, (Cint,), 1)
    try
        @eval using Enzyme
    finally
        ccall(:jl_debug_method_invalidation, Any, (Cint,), 0)
    end
    @test isempty(invalidated_callers(invalidations))

    # Positive control: a redundant ndims method must invalidate the warmed
    # caller, even though it returns exactly what AbstractArray already does.
    ArrayConsumer.dims([1.0])
    struct ControlArray <: AbstractVector{Float64} end
    invalidations = ccall(:jl_debug_method_invalidation, Any, (Cint,), 1)
    try
        @eval Base.ndims(::ControlArray) = 1
    finally
        ccall(:jl_debug_method_invalidation, Any, (Cint,), 0)
    end
    @test :dims in invalidated_callers(invalidations)
    """
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`
    @test success(pipeline(cmd; stdout, stderr))
end
