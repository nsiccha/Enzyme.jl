using Enzyme, Test
import LLVM

# `propagate_returned!` replaces a load from an internal constant global by the part of
# the initializer it reads. To find that part it follows `getelementptr` instructions
# with constant indices from the global to the load and uses their indices as
# `extractvalue` indices.
#
# Not every such instruction addresses an element. The address before the first element
# (index `-1`), from which one-based indexing counts, and the address past the last one,
# a loop's bound, are both valid results of `getelementptr`, and either can have a
# constant global as its base, for instance after the pass has put a caller's constant
# argument into a callee. An `i64 -1` reads as `2^64 - 1`, which does not fit the 32-bit
# index of `extractvalue`, so compiling threw
# `InexactError: trunc(UInt32, 18446744073709551615)`. A smaller index past the end fit,
# and a load through it asked for an element the initializer does not have, which
# crashed the process.

const GEP_ARRAY = "@g = private unnamed_addr constant [3 x double] [double 1.0, double 2.0, double 3.0]"
const GEP_STRUCT_TYPE = "{ i64, [2 x double] }"
const GEP_STRUCT = "@s = private unnamed_addr constant $GEP_STRUCT_TYPE { i64 7, [2 x double] [double 4.0, double 5.0] }"
const GEP_VECTOR = "@v = private unnamed_addr constant <2 x double> <double 8.0, double 9.0>"

# The pointer type to `pointee`: typed on LLVM 15, opaque later.
gep_pointer(typed::Bool, pointee::String) = typed ? pointee * "*" : "ptr"

function gep_function(definitions::String, body::String)
    return """
    $definitions

    define double @f(i64 %i) {
    entry:
    $body
      ret double %v
    }
    """
end

# The address before the first element, indexed from one.
function gep_before_first(typed::Bool)
    arr = gep_pointer(typed, "[3 x double]")
    elt = gep_pointer(typed, "double")
    return gep_function(
        GEP_ARRAY, """
          %base = getelementptr [3 x double], $arr @g, i64 0, i64 -1
          %p = getelementptr double, $elt %base, i64 %i
          %v = load double, $elt %p
        """
    )
end

# A load through one address into the array, with the index written as `index`.
function gep_array_load(typed::Bool, index::String)
    arr = gep_pointer(typed, "[3 x double]")
    elt = gep_pointer(typed, "double")
    return gep_function(
        GEP_ARRAY, """
          %p = getelementptr [3 x double], $arr @g, i64 0, $index
          %v = load double, $elt %p
        """
    )
end

# A load through an address space cast of the array.
function gep_cast_load(typed::Bool, index::String)
    arr = gep_pointer(typed, "[3 x double]")
    cast = typed ? "[3 x double] addrspace(11)*" : "ptr addrspace(11)"
    elt = typed ? "double addrspace(11)*" : "ptr addrspace(11)"
    return gep_function(
        GEP_ARRAY, """
          %c = addrspacecast $arr @g to $cast
          %p = getelementptr [3 x double], $cast %c, i64 0, $index
          %v = load double, $elt %p
        """
    )
end

# A load through one address into the array a struct holds.
function gep_struct_load(typed::Bool, index::String)
    str = gep_pointer(typed, GEP_STRUCT_TYPE)
    elt = gep_pointer(typed, "double")
    return gep_function(
        GEP_STRUCT, """
          %p = getelementptr $GEP_STRUCT_TYPE, $str @s, i64 0, i32 1, $index
          %v = load double, $elt %p
        """
    )
end

# The same element through two addresses: the field, then the element of the field.
function gep_chained_load(typed::Bool, index::String)
    str = gep_pointer(typed, GEP_STRUCT_TYPE)
    arr = gep_pointer(typed, "[2 x double]")
    elt = gep_pointer(typed, "double")
    return gep_function(
        GEP_STRUCT, """
          %a = getelementptr $GEP_STRUCT_TYPE, $str @s, i64 0, i32 1
          %p = getelementptr [2 x double], $arr %a, i64 0, $index
          %v = load double, $elt %p
        """
    )
end

# A load of one element of a vector.
function gep_vector_load(typed::Bool, index::String)
    vec = gep_pointer(typed, "<2 x double>")
    elt = gep_pointer(typed, "double")
    return gep_function(
        GEP_VECTOR, """
          %p = getelementptr <2 x double>, $vec @v, i64 0, $index
          %v = load double, $elt %p
        """
    )
end

# Runs the pass on `ir(typed, args...)` and reports the constant `f` then returns, if its
# load was replaced by one, and the number of loads left in `f`.
function gep_fold(ir, args...)
    return LLVM.Context() do ctx
        mod = parse(LLVM.Module, ir(LLVM.supports_typed_pointers(ctx), args...))
        Enzyme.Compiler.propagate_returned!(mod)
        LLVM.verify(mod)
        returned = nothing
        loads = 0
        for bb in LLVM.blocks(LLVM.functions(mod)["f"]), inst in LLVM.instructions(bb)
            if inst isa LLVM.LoadInst
                loads += 1
            elseif inst isa LLVM.RetInst
                val = LLVM.operands(inst)[1]
                val isa LLVM.ConstantFP && (returned = convert(Float64, val))
            end
        end
        (; returned, loads)
    end
end

const GEP_KEPT = (; returned = nothing, loads = 1)
gep_folded(x::Float64) = (; returned = x, loads = 0)

@testset "Addresses outside a constant global are left alone" begin
    @testset "the address before the first element" begin
        @test gep_fold(gep_before_first) == GEP_KEPT
        @test gep_fold(gep_array_load, "i64 -1") == GEP_KEPT
        @test gep_fold(gep_array_load, "i32 -1") == GEP_KEPT
    end
    @testset "addresses past the last element" begin
        @test gep_fold(gep_array_load, "i64 3") == GEP_KEPT
        @test gep_fold(gep_array_load, "i64 4294967296") == GEP_KEPT
        @test gep_fold(gep_vector_load, "i64 2") == GEP_KEPT
    end
    @testset "inside a struct, through a cast and through two addresses" begin
        @test gep_fold(gep_struct_load, "i64 -1") == GEP_KEPT
        @test gep_fold(gep_struct_load, "i64 2") == GEP_KEPT
        @test gep_fold(gep_cast_load, "i64 -1") == GEP_KEPT
        @test gep_fold(gep_chained_load, "i64 -1") == GEP_KEPT
        @test gep_fold(gep_chained_load, "i64 2") == GEP_KEPT
    end
end

@testset "Loads of an element of a constant global are still replaced" begin
    @test gep_fold(gep_array_load, "i64 0") == gep_folded(1.0)
    @test gep_fold(gep_array_load, "i64 2") == gep_folded(3.0)
    @test gep_fold(gep_array_load, "i32 1") == gep_folded(2.0)
    @test gep_fold(gep_cast_load, "i64 1") == gep_folded(2.0)
    @test gep_fold(gep_struct_load, "i64 0") == gep_folded(4.0)
    @test gep_fold(gep_struct_load, "i64 1") == gep_folded(5.0)
    @test gep_fold(gep_chained_load, "i64 0") == gep_folded(4.0)
    @test gep_fold(gep_chained_load, "i64 1") == gep_folded(5.0)
    @test gep_fold(gep_vector_load, "i64 1") == gep_folded(9.0)
end
