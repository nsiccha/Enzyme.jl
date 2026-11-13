using Enzyme, Test

const MemsetLLVM = Enzyme.LLVM

function readonly_memset_ir(typed, destination)
    ptr = typed ? "i8*" : "ptr"
    slotptr = typed ? "i8**" : "ptr"
    intrinsic = typed ? "llvm.memset.p0i8.i64" : "llvm.memset.p0.i64"
    cast = typed ? "bitcast i8** %slot to i8*" : "getelementptr i8, ptr %slot, i64 0"
    return """
    declare void @$intrinsic($ptr, i8, i64, i1)
    define void @test($ptr %held) {
        %slot = alloca $ptr
        %slotbytes = $cast
        store $ptr %held, $slotptr %slot
        %alias = load $ptr, $slotptr %slot
        call void @$intrinsic($ptr %$destination, i8 0, i64 8, i1 false)
        ret void
    }
    """
end

@testset "Readonly check: memset of local slots and held data" begin
    MemsetLLVM.Context() do ctx
        for (destination, writes) in (("slotbytes", false), ("held", true), ("alias", true))
            mod = parse(MemsetLLVM.Module, readonly_memset_ir(MemsetLLVM.supports_typed_pointers(ctx), destination))
            fn = MemsetLLVM.functions(mod)["test"]
            world = Base.get_world_counter()
            result = Enzyme.Compiler.readonly_walk!(
                MemsetLLVM.Value[only(MemsetLLVM.parameters(fn))], fn, world,
                Enzyme.EnzymeContext(world), Enzyme.Compiler.ReadonlyWalkState(mod), false,
            )
            @test result == (writes, false)
        end
    end
end

struct ROZeroHeld{T}
    data::T
end
function (f::ROZeroHeld)(x)
    fill!(f.data, 0.0)
    return sum(f.data) * x
end

@testset "Readonly check: zeroing a captured array still raises" begin
    f = ROZeroHeld([1.0, 2.0, 3.0])
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(0.7))
    @test f.data == [1.0, 2.0, 3.0]
    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Forward, f, Duplicated(0.7, 1.0))
    @test f.data == [1.0, 2.0, 3.0]
end
