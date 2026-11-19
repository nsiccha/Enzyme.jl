using Enzyme, Test
using Enzyme.Compiler: LLVM

# The first subtraction also has an independent use. Only the sum used to
# index the rooted memory should change, even when the addends are reversed.
@testset "Reassociate a rooted byte offset" begin
    LLVM.Context(; opaque_pointers = true) do ctx
        mod = parse(
            LLVM.Module, """
            target datalayout = "e-p:64:64"
            declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr)
            define ptr addrspace(13) @offset(ptr addrspace(10) %mem, ptr %base, ptr %data, ptr %out, i64 %i) {
              %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %base)
              %p = ptrtoint ptr %data to i64
              %q = ptrtoint ptr %base to i64
              %delta = sub i64 %i, %q
              store i64 %delta, ptr %out
              %sum = add i64 %p, %delta
              %addr = getelementptr i8, ptr addrspace(13) %rooted, i64 %sum
              ret ptr addrspace(13) %addr
            }
            """
        )
        Enzyme.Compiler.force_recompute!(mod)
        @test LLVM.verify(mod) === nothing
        ir = string(mod.functions["offset"])
        @test occursin("sub i64 %p, %q", ir)
        @test occursin("store i64 %delta", ir)
        @test !occursin("add i64 %p, %delta", ir)
    end
end

@static if VERSION >= v"1.12-"
    # Model the lowered read of a Float64 array backed by an interior MemoryRef.
    # LLVM can split the offset between two GEPs before differentiating it.
    const rooted_read_ir = """
    declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr)
    define double @read(ptr addrspace(10) %array, i$(Sys.WORD_SIZE) %i) {
      %data = load ptr, ptr addrspace(10) %array
      %memfield = getelementptr i8, ptr addrspace(10) %array, i32 $(sizeof(Ptr{Cvoid}))
      %mem = load ptr addrspace(10), ptr addrspace(10) %memfield
      %basefield = getelementptr i8, ptr addrspace(10) %mem, i32 $(sizeof(Ptr{Cvoid}))
      %base = load ptr, ptr addrspace(10) %basefield
      %rooted = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %base)
      %p = ptrtoint ptr %data to i$(Sys.WORD_SIZE)
      %q = ptrtoint ptr %base to i$(Sys.WORD_SIZE)
      %delta = sub i$(Sys.WORD_SIZE) %i, %q
      %offset = add i$(Sys.WORD_SIZE) %delta, %p
      %addr = getelementptr i8, ptr addrspace(13) %rooted, i$(Sys.WORD_SIZE) %offset
      %v = load double, ptr addrspace(13) %addr
      ret double %v
    }
    """
    function rooted_cube(x, i)
        return Base.llvmcall((rooted_read_ir, "read"), Float64, Tuple{Any, Int}, x, i)^3
    end
    function rooted_directional(x, dx, i)
        return only(autodiff(Forward, rooted_cube, Duplicated(x, dx), Const(i)))
    end
    interior_array(x) = Base.wrap(Array, memoryref(x.ref.mem, 2), length(x) - 1)

    @testset "Differentiate an interior MemoryRef with a dynamic byte offset" begin
        x = interior_array([3.0, 7.0, 11.0, 17.0])
        dx = interior_array([1.0, 2.0, 3.0, 4.0])
        g = interior_array(zeros(4))
        i = sizeof(Float64)
        @test rooted_cube(x, i) == 11.0^3
        autodiff(Reverse, rooted_cube, Active, Duplicated(x, g), Const(i))
        @test g == [0.0, 3 * 11.0^2, 0.0]
        fill!(g, 0)
        autodiff(Reverse, rooted_directional, Active, Duplicated(x, g), Const(dx), Const(i))
        @test g == [0.0, 6 * 11.0 * 3.0, 0.0]
    end
end
