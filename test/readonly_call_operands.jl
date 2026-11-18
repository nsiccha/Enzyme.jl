using Enzyme, LLVM, Test
import GPUCompiler

# Foreign callees may read a header, then write through a pointer loaded from
# it. Attribute-only reasoning would accept those writes. Build calls without
# bodies so the proof must distinguish raw scalar storage from those headers.
function readonly_operand_case(;
        T = Ptr{Float64}, byref = GPUCompiler.BITS_VALUE,
        first_attrs = "nocapture readonly", second_attrs = "",
        first_value = "data", second_value = "other", bundle = "",
        callsite = false, form = :direct, U = T,
    )
    return LLVM.Context() do ctx
        ptr = LLVM.supports_typed_pointers(ctx) ? "i8*" : "ptr"
        typeattrs = "\"enzymejl_parmtype\"=\"$(UInt(pointer_from_objref(T)))\" " *
            "\"enzymejl_parmtype_ref\"=\"$(UInt(byref))\""
        otherattrs = "\"enzymejl_parmtype\"=\"$(UInt(pointer_from_objref(U)))\" " *
            "\"enzymejl_parmtype_ref\"=\"$(UInt(byref))\""
        bundle_ir = isempty(bundle) ? "" : "[ \"$bundle\"($ptr %data) ]"
        decl1, decl2 = callsite ? ("", "") : (first_attrs, second_attrs)
        arg1, arg2 = callsite ? (first_attrs, second_attrs) : ("", "")
        prolog = ""
        epilog = "ret void"
        if form == :phi
            prolog = """
            entry: br i1 %cond, label %left, label %right
            left:
                %offset = getelementptr i8, $ptr %data, i64 %n
                br label %merge
            right: br label %merge
            merge: %merged = phi $ptr [ %offset, %left ], [ %other, %right ]
            """
        elseif form == :select
            prolog = """
            %offset = getelementptr i8, $ptr %data, i64 %n
            %merged = select i1 %cond, $ptr %offset, $ptr %other
            """
        elseif form == :loop
            prolog = """
            entry: br label %loop
            loop:
                %merged = phi $ptr [ %data, %entry ], [ %offset, %loop ]
                %offset = getelementptr i8, $ptr %merged, i64 %n
            """
            epilog = "br i1 %cond, label %exit, label %loop\nexit: ret void"
        end
        form == :direct || (first_value = "merged")
        mod = parse(
            LLVM.Module, """
            declare void @sink($ptr $decl1, $ptr $decl2)
            define void @caller($ptr $typeattrs %data, $ptr $otherattrs %other, i64 %n, i1 %cond) {
                $prolog
                call void @sink($ptr $arg1 %$first_value, $ptr $arg2 %$second_value) $bundle_ir
                $epilog
            }
            """
        )
        try
            LLVM.verify(mod)
            caller = mod.functions["caller"]
            call = only(inst for bb in caller.blocks for inst in bb.instructions if inst isa LLVM.CallInst)
            val = form == :direct ? first(caller.parameters) : call.operands[1]
            world = Base.get_world_counter()
            return Enzyme.Compiler.is_readonly_call_operand(
                call, mod.functions["sink"], val,
                world, Enzyme.Compiler.EnzymeContext(world),
            )
        finally
            LLVM.dispose(mod)
        end
    end
end

@testset "Readonly call operands: references and escapes" begin
    @test readonly_operand_case()
    @test readonly_operand_case(T = Ptr{ComplexF64})
    @test readonly_operand_case(T = Core.LLVMPtr{Float64, 0})
    @test readonly_operand_case(callsite = true)
    @test readonly_operand_case(first_attrs = "nocapture readnone")
    @test !readonly_operand_case(first_attrs = "readonly")
    @test !readonly_operand_case(first_attrs = "nocapture")
    @test !readonly_operand_case(first_attrs = "")
    @test !readonly_operand_case(T = Ptr{Ptr{Float64}})
    @test !readonly_operand_case(T = Ptr{Vector{Float64}})
    @test !readonly_operand_case(T = Vector{Float64}, byref = GPUCompiler.MUT_REF)
    @test !readonly_operand_case(T = Float64, byref = GPUCompiler.BITS_REF)

    # The same pointer in a writable position must not be hidden by a readonly
    # position or a GC root. All argument occurrences matter.
    @test !readonly_operand_case(second_value = "data")
    @test !readonly_operand_case(second_value = "data", bundle = "jl_roots")
    @test readonly_operand_case(second_value = "data", second_attrs = "nocapture readonly")
    @test readonly_operand_case(bundle = "jl_roots")
    @test readonly_operand_case(first_value = "other", bundle = "jl_roots")
    @test !readonly_operand_case(first_attrs = "", bundle = "jl_roots")
    @test !readonly_operand_case(bundle = "unknown")
    @test !readonly_operand_case(first_value = "other", bundle = "unknown")
    @test !readonly_operand_case(first_value = "other")

    # A merged or loop-carried byte offset remains scalar storage only when
    # every base does. One branch with nested pointers must still be rejected.
    for form in (:phi, :select)
        @test readonly_operand_case(; form)
        @test readonly_operand_case(; form, U = Ptr{ComplexF64})
        @test !readonly_operand_case(; form, U = Ptr{Ptr{Float64}})
        @test !readonly_operand_case(; form, T = Ptr{Ptr{Float64}})
    end
    @test readonly_operand_case(form = :loop)
    @test !readonly_operand_case(form = :loop, T = Ptr{Ptr{Float64}})
end
