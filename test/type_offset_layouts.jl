using Enzyme, LLVM, Test

function type_offset_limit()
    ptr = cglobal((:MaxTypeOffset, Enzyme.API.libEnzyme))
    return ccall((:EnzymeGetCLInteger, Enzyme.API.libEnzyme), Int64, (Ptr{Cvoid},), ptr)
end

@testset "Explicit layouts beyond the type-inference offset limit" begin
    limit = type_offset_limit()
    @test limit == 500
    LLVM.Context() do ctx
        dl = string(LLVM.DataLayout(LLVM.JITTargetMachine()))
        # A homogeneous layout can only collapse to -1 when every element is
        # present, including those past the native 500-byte inference limit.
        floats = Enzyme.typetree(NTuple{128, Float64}, ctx, dl)
        @test string(floats) == "{[-1]:Float@double}"
        LLVM.dispose(floats)

        # Alternating floats and pointers cannot collapse to one repeated type.
        # Check every floating field, rather than just the first field past 500.
        T = NTuple{80, Tuple{Float64, Ptr{Float64}}}
        mixed = Enzyme.typetree(T, ctx, dl)
        description = string(mixed)
        for i in 1:fieldcount(T)
            offset = fieldoffset(T, i)
            @test occursin("[$offset]:Float@double", description)
        end
        LLVM.dispose(mixed)
    end
    @test type_offset_limit() == limit
end

@generated function offset_operands(q, x, ::Val{N}) where {N}
    return Expr(:tuple, 0.3, [:(Base.Broadcast.broadcasted(*, q[$i], x)) for i in 1:N]...)
end

@noinline function offset_argtail(t)
    Base.@nospecialize t
    return Core._apply_iterate(Base.iterate, Base.argtail, t)
end

function offset_tuple_loss(q, x, n)
    operands = offset_argtail(offset_operands(q, x, n))
    total = 0.0
    for i in eachindex(operands)
        operand = operands[i]
        total += operand.args[1] * operand.args[2][1]
    end
    return total
end

@noinline function offset_broadcast_copy(t)
    Base.@nospecialize t
    broadcast = Core._apply_iterate(Base.iterate, Base.Broadcast.broadcasted, (+,), t)
    return copy(Base.Broadcast.instantiate(broadcast))
end

function offset_broadcast_loss(q, x, n)
    values = offset_broadcast_copy(offset_operands(q, x, n))
    total = 0.0
    for i in eachindex(x)
        total += values[i]
    end
    return total
end

@testset "Reverse through a wide dynamically dispatched tuple tail" begin
    for n in (30, 33)
        q = collect(1.0:n)
        dq = zero(q)
        autodiff(Reverse, offset_tuple_loss, Active, Duplicated(q, dq), Const([2.0, 3.0]), Const(Val(n)))
        @test dq == fill(2.0, n)
    end

    # Materializing the broadcast used to hide the missing type information:
    # differentiation succeeded but silently returned zero for q[32:33].
    q = collect(1.0:33)
    dq = zero(q)
    autodiff(Reverse, offset_broadcast_loss, Active, Duplicated(q, dq), Const([2.0, 3.0]), Const(Val(33)))
    @test dq == fill(5.0, 33)
    @test type_offset_limit() == 500
end
