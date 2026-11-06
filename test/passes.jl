using Enzyme, LLVM, Test
using FileCheck
import Libdl
import GPUCompiler


@testset "Partial return preservation" begin
    @test @filecheck begin
        # Both stores into the freshly allocated struct must be preserved.
        # Match without spelling out the pointer type so this works under both
        # opaque (`ptr addrspace(10)`) and typed (`{} addrspace(10)*`) pointers.
        @check_label "@inner"
        @check "store atomic"
        @check_same "%v1"
        @check "store atomic"
        @check_same "%v2"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}**, i64, {} addrspace(10)*) local_unnamed_addr #5

                define internal fastcc nonnull {} addrspace(10)* @inner({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %newstruct = call noalias nonnull dereferenceable(16) {} addrspace(10)* @julia.gc_alloc_obj({}** null, i64 16, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 129778359735376 to {}*) to {} addrspace(10)*)) #30
                  %a31 = addrspacecast {} addrspace(10)* %newstruct to {} addrspace(10)* addrspace(11)*
                  %a32 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %a31, i64 1
                  store atomic {} addrspace(10)* %v1, {} addrspace(10)* addrspace(11)* %a31 release, align 8
                  %a33 = addrspacecast {} addrspace(10)* %newstruct to i8 addrspace(11)*
                  %a34 = getelementptr inbounds i8, i8 addrspace(11)* %a33, i64 8
                  %a35 = bitcast i8 addrspace(11)* %a34 to {} addrspace(10)* addrspace(11)*
                  store atomic {} addrspace(10)* %v2, {} addrspace(10)* addrspace(11)* %a35 release, align 8
                  ret {} addrspace(10)* %newstruct
                }

                define {} addrspace(10)* @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %ac = call fastcc nonnull {} addrspace(10)* @inner({} addrspace(10)* %v1, {} addrspace(10)* %v2)
                  %b = addrspacecast {} addrspace(10)* %ac to {} addrspace(10)* addrspace(11)*
                  %c = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %b unordered, align 8
                  ret {} addrspace(10)* %c
                }

                attributes #5 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_no_escaping_allocation" "enzymejl_world"="31504" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end


@testset "Dead return removal" begin
    @test @filecheck begin
        # The call to the (now dead) callee is removed, leaving only `ret void`,
        # and the callee itself is deleted from the module.
        @check_label "define void @caller"
        @check_next "top:"
        @check_next "ret void"
        @check_not "@julia_MyPrognosticVars_161"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}**, i64, {} addrspace(10)*) local_unnamed_addr #5

                define internal fastcc nonnull {} addrspace(10)* @julia_MyPrognosticVars_161({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %newstruct = call noalias nonnull dereferenceable(16) {} addrspace(10)* @julia.gc_alloc_obj({}** null, i64 16, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 129778359735376 to {}*) to {} addrspace(10)*)) #30
                  %a31 = addrspacecast {} addrspace(10)* %newstruct to {} addrspace(10)* addrspace(11)*
                  %a32 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %a31, i64 1
                  store atomic {} addrspace(10)* %v1, {} addrspace(10)* addrspace(11)* %a31 release, align 8
                  %a33 = addrspacecast {} addrspace(10)* %newstruct to i8 addrspace(11)*
                  %a34 = getelementptr inbounds i8, i8 addrspace(11)* %a33, i64 8
                  %a35 = bitcast i8 addrspace(11)* %a34 to {} addrspace(10)* addrspace(11)*
                  store atomic {} addrspace(10)* %v2, {} addrspace(10)* addrspace(11)* %a35 release, align 8
                  ret {} addrspace(10)* %newstruct
                }

                define void @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %ac = call fastcc nonnull {} addrspace(10)* @julia_MyPrognosticVars_161({} addrspace(10)* %v1, {} addrspace(10)* %v2)
                  ret void
                }

                attributes #5 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_no_escaping_allocation" "enzymejl_world"="31504" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "Return roots preservation" begin
    @test @filecheck begin
        # The dead `%data` argument is dropped from the call, but the `sret` and
        # `enzymejl_returnRoots` arguments must be preserved (roots last).
        @check_label "define void @caller"
        @check "call void @julia_dims_4189"
        @check_same "sret"
        @check_same "enzymejl_returnRoots"
        @check_same "%roots)"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                define private void @julia_dims_4189({ double, {} addrspace(10)*, {} addrspace(10)* }* sret({ double, {} addrspace(10)*, {} addrspace(10)* }) %res, [2 x {} addrspace(10)*]* "enzymejl_returnRoots"="2", double addrspace(11)* %data) #0 {
                top:
                  %val = load double, double addrspace(11)* %data, align 8
                  store { double, {} addrspace(10)*, {} addrspace(10)* } zeroinitializer, { double, {} addrspace(10)*, {} addrspace(10)* }* %res
                  ret void
                }

                define void @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2, double addrspace(11)* %data) {
                top:
                  %sret = alloca { double, {} addrspace(10)*, {} addrspace(10)* }
                  %roots = alloca [2 x {} addrspace(10)*]
                  call void @julia_dims_4189({ double, {} addrspace(10)*, {} addrspace(10)* }* sret({ double, {} addrspace(10)*, {} addrspace(10)* }) %sret, [2 x {} addrspace(10)*]* "enzymejl_returnRoots"="2" %roots, double addrspace(11)* %data)
                  ret void
                }

                attributes #0 = { nofree nosync nounwind willreturn noinline "enzyme_inactive" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), true)
            string(mod)
        end
    end
end

@testset "Recursively dead function removal" begin
    @test @filecheck begin
        # Both the recursive function and its callee are dead and must be removed.
        @check_not "@dead_recursive_fn"
        @check_not "@dead_callee"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define internal fastcc void @dead_callee(i32* nocapture %arg) {
                top:
                  %val = load i32, i32* %arg, align 4
                  ret void
                }

                define internal fastcc void @dead_recursive_fn(i32* %arg) {
                top:
                  call fastcc void @dead_recursive_fn(i32* %arg)
                  call fastcc void @dead_callee(i32* %arg)
                  ret void
                }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "Mismatched calling convention/function type DAE safety" begin
    @test @filecheck begin
        # The call site bitcasts the callee to a mismatched function type, so DAE
        # must leave the callee untouched: both of its arguments are preserved.
        @check "define internal fastcc void @callee("
        @check_same "%arg1"
        @check_same "%arg2"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define internal fastcc void @callee(i32* %arg1, i32* %arg2) {
                top:
                  store i32 42, i32* %arg1, align 4
                  ret void
                }

                define void @caller(i32 addrspace(10)* %arg1, i32 addrspace(10)* %arg2) {
                top:
                  call fastcc void (i32 addrspace(10)*, i32 addrspace(10)*) bitcast (void (i32*, i32*)* @callee to void (i32 addrspace(10)*, i32 addrspace(10)*)*)(i32 addrspace(10)* %arg1, i32 addrspace(10)* %arg2)
                  ret void
                }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "import_cached_autodiff!" begin
    LLVM.Context() do ctx
        # A cached thunk's bitcode carries the Julia functions it calls. Two
        # modules importing it would both define `@julia_shared`, and
        # `compile_unhooked` links every deferred module into one (#2788), so
        # the blob must be imported once and declared everywhere else.
        blob = parse(
            LLVM.Module,
            """
            define i64 @thunk(i64 %x) {
              %r = call i64 @julia_shared(i64 %x)
              ret i64 %r
            }
            define i64 @julia_shared(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            """,
        )
        buf = convert(LLVM.MemoryBuffer, blob)
        bitcode = String(convert(Vector{UInt8}, buf))
        dispose(buf)

        ptr = reinterpret(Ptr{Cvoid}, UInt(0x2788))
        Enzyme.Compiler.autodiff_cache[ptr] = Enzyme.Compiler.CachedThunk("thunk", bitcode, Enzyme.Compiler.JuliaValueTable())
        try
            Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT =>
                    Enzyme.Compiler.EnzymeContext(GPUCompiler.tls_world_age()) begin
                FT = LLVM.FunctionType(LLVM.Int64Type(), [LLVM.Int64Type()])

                first_mod = LLVM.Module("first")
                imported = Enzyme.Compiler.import_cached_autodiff!(first_mod, ptr, FT)
                @test !LLVM.isdeclaration(imported)
                # Externally visible, so the modules that only declare the entry
                # bind to this definition when everything is linked together.
                @test LLVM.linkage(imported) == LLVM.API.LLVMExternalLinkage
                @test haskey(LLVM.functions(first_mod), "julia_shared")

                second_mod = LLVM.Module("second")
                declared = Enzyme.Compiler.import_cached_autodiff!(second_mod, ptr, FT)
                @test LLVM.isdeclaration(declared)
                # The blob is not imported a second time, so nothing it carries
                # is defined twice.
                @test !haskey(LLVM.functions(second_mod), "julia_shared")

                LLVM.link!(first_mod, second_mod)
                Enzyme.Compiler.internalize_imported_thunks!(first_mod)
                @test LLVM.linkage(LLVM.functions(first_mod)["thunk"]) ==
                    LLVM.API.LLVMInternalLinkage
            end
        finally
            delete!(Enzyme.Compiler.autodiff_cache, ptr)
        end
    end
end

@testset "link_split_existing!" begin
    LLVM.Context() do ctx
        # Test 1: Differing definitions are preserved as distinct internal specializations
        dst = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            define i64 @only_in_dst(i64 %x) {
              ret i64 %x
            }
            """,
        )
        src = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 2
              ret i64 %r
            }
            define i64 @uses_dup(i64 %x) {
              %r = call i64 @julia___dup(i64 %x)
              ret i64 %r
            }
            """,
        )

        Enzyme.Compiler.link_split_existing!(dst, src)

        fns = LLVM.functions(dst)
        @test haskey(fns, "julia___dup")
        @test !LLVM.isdeclaration(fns["julia___dup"])
        @test LLVM.linkage(fns["julia___dup"]) == LLVM.API.LLVMExternalLinkage
        @test haskey(fns, "only_in_dst")
        @test haskey(fns, "uses_dup")
        usesfn = fns["uses_dup"]
        callinst = first(
            filter(
                Base.Fix2(isa, LLVM.CallInst),
                collect(instructions(first(blocks(usesfn)))),
            ),
        )
        called_fn = last(collect(operands(callinst)))
        @test LLVM.linkage(called_fn) == LLVM.API.LLVMInternalLinkage

        # Test 2: Identical definitions are folded by MergeFunctionsPass
        dst2 = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            """,
        )
        src2 = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            define i64 @uses_dup2(i64 %x) {
              %r = call i64 @julia___dup(i64 %x)
              ret i64 %r
            }
            """,
        )
        Enzyme.Compiler.link_split_existing!(dst2, src2)
        fns2 = LLVM.functions(dst2)
        @test haskey(fns2, "julia___dup")
        @test haskey(fns2, "uses_dup2")
        usesfn2 = fns2["uses_dup2"]
        callinst2 = first(
            filter(
                Base.Fix2(isa, LLVM.CallInst),
                collect(instructions(first(blocks(usesfn2)))),
            ),
        )
        @test LLVM.name(last(collect(operands(callinst2)))) == "julia___dup"
    end
end

@testset "Literal-pointer symbol resolution" begin
    # `jl_lookup_code_address` can attribute two distinct pointers to the same
    # (nearest) symbol name on platforms with export-only symbol info. The
    # helpers below back the guards in `check_ir!` that keep such call sites
    # from being merged onto one restored address.
    libmpfr = Libdl.dlpath(Base.MPFR.libmpfr)
    hnd = Libdl.dlopen(libmpfr)
    p_add = Libdl.dlsym(hnd, :mpfr_add)
    p_sub = Libdl.dlsym(hnd, :mpfr_sub)
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", libmpfr, p_add) == :match
    # A nearest-symbol misattribution must be detected, not trusted.
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", libmpfr, p_sub) == :mismatch
    # The containing module is found via the loader, so an unloadable `file` (as
    # reported on Linux/macOS, where it is a source path) does not degrade the answer.
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "/not/a/library/add.c", p_add) ==
        :match
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "/not/a/library/add.c", p_sub) ==
        :mismatch
    @test Enzyme.Compiler.resolve_symbol_name("not_a_real_symbol_abcxyz", libmpfr, p_add) ==
        :unknown
    # A pointer outside any loaded module cannot be attributed at all.
    heapptr = Libc.malloc(8)
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "", heapptr) == :unknown
    Libc.free(heapptr)

    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module,
            """
            declare void @foo() #0
            declare void @bar()
            attributes #0 = { "enzymejl_needs_restoration"="12345" }
            """,
        )
        @test Enzyme.Compiler.restoration_ptr(functions(mod)["foo"]) == UInt(12345)
        @test Enzyme.Compiler.restoration_ptr(functions(mod)["bar"]) === nothing
    end
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3284
# The `jl_get_abi_converter` lowering of `@cfunction` only exists on 1.12+, and
# older Julia's LLVM cannot parse the opaque-pointer IR below.
@static if VERSION >= v"1.12"

    @testset "Rewrite abi converter calls (1.12 pattern)" begin
        LLVM.Context() do ctx
            # Shape of the `@cfunction` dispatch site emitted by Julia 1.12's
            # codegen: `jl_get_abi_converter(ct, fptr, last_world, cfuncdata)` with
            # a six-slot cfuncdata whose third slot holds the in-module
            # `unspecialized` apply-generic thunk.
            mod = parse(
                LLVM.Module,
                """
                @jl_world_counter = external global i64
                @fptr = private global ptr @gfthunk
                @last_world = private global i64 0
                @declrt = private global ptr null
                @sigt = private global ptr null
                @cfuncdata = private global [6 x ptr] [ptr null, ptr null, ptr @gfthunk, ptr @declrt, ptr @sigt, ptr inttoptr (i64 1 to ptr)]

                declare ptr @jl_get_abi_converter(ptr, ptr, ptr, ptr)

                define internal double @gfthunk(double %0) {
                top:
                  ret double %0
                }

                define double @trampoline(ptr %ct, double %x) {
                top:
                  %last_world = load atomic i64, ptr @last_world acquire, align 8
                  %fptr = load atomic ptr, ptr @fptr monotonic, align 8
                  %world = load atomic i64, ptr @jl_world_counter acquire, align 8
                  %stale = icmp ne i64 %last_world, %world
                  br i1 %stale, label %guard_pass, label %guard_exit

                guard_pass:
                  %cw = call ptr @jl_get_abi_converter(ptr %ct, ptr @fptr, ptr @last_world, ptr @cfuncdata)
                  br label %guard_exit

                guard_exit:
                  %target = phi ptr [ %fptr, %top ], [ %cw, %guard_pass ]
                  %res = call double %target(double %x)
                  ret double %res
                }
                """,
            )
            Enzyme.Compiler.rewrite_abi_converter_calls!(mod)
            @test @filecheck begin
                @check_label "define double @trampoline"
                @check_not "call ptr @jl_get_abi_converter"
                @check "phi ptr [ %fptr, %top ], [ @gfthunk, %guard_pass ]"
                string(mod)
            end
        end
    end

    @testset "Rewrite abi converter calls (1.13 pattern)" begin
        LLVM.Context() do ctx
            # Shape of the dispatch site emitted by Julia 1.13+:
            # `jl_get_abi_converter(ct, cfuncdata)` with an eight-slot cfuncdata
            # ([fptr, last_world, plast_codeinst, last_codeinst, unspecialized,
            # declrt, sigt, flags]) whose fifth slot holds the thunk.
            mod = parse(
                LLVM.Module,
                """
                @jl_world_counter = external global i64
                @declrt = private global ptr null
                @sigt = private global ptr null
                @cfuncdata = private global [8 x ptr] [ptr @gfthunk, ptr null, ptr null, ptr null, ptr @gfthunk, ptr @declrt, ptr @sigt, ptr inttoptr (i64 1 to ptr)]

                declare ptr @ijl_get_abi_converter(ptr, ptr)

                define internal double @gfthunk(double %0) {
                top:
                  ret double %0
                }

                define double @trampoline(ptr %ct, double %x) {
                top:
                  %last_world_p = getelementptr inbounds i64, ptr @cfuncdata, i32 1
                  %last_world = load atomic i64, ptr %last_world_p acquire, align 8
                  %fptr = load atomic ptr, ptr @cfuncdata monotonic, align 8
                  %world = load atomic i64, ptr @jl_world_counter acquire, align 8
                  %stale = icmp ne i64 %last_world, %world
                  br i1 %stale, label %guard_pass, label %guard_exit

                guard_pass:
                  %cw = call ptr @ijl_get_abi_converter(ptr %ct, ptr @cfuncdata)
                  br label %guard_exit

                guard_exit:
                  %target = phi ptr [ %fptr, %top ], [ %cw, %guard_pass ]
                  %res = call double %target(double %x)
                  ret double %res
                }
                """,
            )
            Enzyme.Compiler.rewrite_abi_converter_calls!(mod)
            @test @filecheck begin
                @check_label "define double @trampoline"
                @check_not "call ptr @ijl_get_abi_converter"
                @check "phi ptr [ %fptr, %top ], [ @gfthunk, %guard_pass ]"
                string(mod)
            end
        end
    end

end # VERSION >= v"1.12"

# An addrspace(11) phi with argument-derived and poison incomings must be skipped by
# the `all_args` prefilter; the rewriter cannot handle a bare addrspace(11) argument.
@testset "nodecayed_phis! addrspace(11) phi with poison incoming" begin
    @test @filecheck begin
        # incomings: gep(arg, 8) and poison
        @check_label "@kernel_gep"
        @check_not "nodecayed"
        @check "phi"
        @check_same "addrspace(11)"
        @check_same "%gep"
        @check_same "poison"
        # zero-offset variant: the argument itself and poison
        @check_label "@kernel_direct"
        @check_not "nodecayed"
        @check "phi"
        @check_same "addrspace(11)"
        @check_same "%q"
        @check_same "poison"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define i8 @kernel_gep(i1 %cond, i8 addrspace(11)* %g) #0 {
                top:
                  br i1 %cond, label %ok, label %guard

                ok:
                  %gep = getelementptr inbounds i8, i8 addrspace(11)* %g, i64 8
                  br label %merge

                guard:
                  br label %merge

                merge:
                  %p = phi i8 addrspace(11)* [ %gep, %ok ], [ poison, %guard ]
                  %ld = load i8, i8 addrspace(11)* %p, align 1
                  ret i8 %ld
                }

                define i8 @kernel_direct(i1 %cond, i8 addrspace(11)* %q) #0 {
                top:
                  br i1 %cond, label %ok2, label %guard2

                ok2:
                  br label %merge2

                guard2:
                  br label %merge2

                merge2:
                  %p2 = phi i8 addrspace(11)* [ %q, %ok2 ], [ poison, %guard2 ]
                  %ld2 = load i8, i8 addrspace(11)* %p2, align 1
                  ret i8 %ld2
                }

                attributes #0 = { "enzymejl_world"="1" }
                """
            )

            Enzyme.Compiler.nodecayed_phis!(mod)
            string(mod)
        end
    end
end

# Julia 1.10 reloads an addrspace(13) pointer phi from a phi of its parent objects. When
# those only ever hold the one argument, the parent phis are folded away, so that the
# reload comes straight from the argument (see test/data_pointer_phi.jl).
@static if VERSION < v"1.11-"
    @testset "nodecayed_phis! folds parent phis that only hold one object" begin
        @test @filecheck begin
            # The header and latch reload the data pointer straight from %y.
            @check_label "@kernel_reload"
            @check_not "nodecayed"
            @check "loop:"
            @check_not "nodecayed"
            @check_not "phi {} addrspace(10)*"
            @check "bitcast {} addrspace(10)* %y"
            @check_not "nodecayed"
            @check "load i8 addrspace(13)*"
            @check_not "nodecayed"
            @check "latch:"
            @check_not "nodecayed"
            @check_not "phi {} addrspace(10)*"
            @check "bitcast {} addrspace(10)* %y"
            @check_not "nodecayed"
            @check "load i8 addrspace(13)*"
            @check_not "nodecayed"
            @check "ret double"
            LLVM.Context() do ctx
                mod = parse(
                    LLVM.Module, """
                    source_filename = "start"
                    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                    target triple = "x86_64-linux-gnu"

                    declare double @g(double)

                    define double @kernel_reload({} addrspace(10)* nocapture readonly %y, i64 %n) #0 {
                    top:
                      %hdr = addrspacecast {} addrspace(10)* %y to { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)*
                      %fld = getelementptr inbounds { i8 addrspace(13)*, i64, i16, i16, i32 }, { i8 addrspace(13)*, i64, i16, i16, i32 } addrspace(11)* %hdr, i64 0, i32 0
                      %data.pre = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %fld, align 8
                      br label %loop

                    loop:
                      %iv = phi i64 [ 0, %top ], [ %iv.next, %latch ]
                      %acc = phi double [ 0.0, %top ], [ %acc.next, %latch ]
                      %data = phi i8 addrspace(13)* [ %data.pre, %top ], [ %data.latch, %latch ]
                      %p = bitcast i8 addrspace(13)* %data to double addrspace(13)*
                      %e = getelementptr inbounds double, double addrspace(13)* %p, i64 %iv
                      %x = load double, double addrspace(13)* %e, align 8
                      %c = fcmp ogt double %x, 0.0
                      br i1 %c, label %call, label %latch

                    call:
                      %r = call double @g(double %x)
                      %data.re = load i8 addrspace(13)*, i8 addrspace(13)* addrspace(11)* %fld, align 8
                      br label %latch

                    latch:
                      %data.latch = phi i8 addrspace(13)* [ %data.re, %call ], [ %data, %loop ]
                      %v = phi double [ %r, %call ], [ 0.0, %loop ]
                      %p2 = bitcast i8 addrspace(13)* %data.latch to double addrspace(13)*
                      %e2 = getelementptr inbounds double, double addrspace(13)* %p2, i64 %iv
                      %x2 = load double, double addrspace(13)* %e2, align 8
                      %w = fadd double %v, %x2
                      %acc.next = fadd double %acc, %w
                      %iv.next = add i64 %iv, 1
                      %done = icmp eq i64 %iv.next, %n
                      br i1 %done, label %exit, label %loop

                    exit:
                      ret double %acc.next
                    }

                    attributes #0 = { "enzymejl_world"="1" }
                    """
                )

                Enzyme.Compiler.nodecayed_phis!(mod)
                LLVM.verify(mod)
                string(mod)
            end
        end
    end

    # An undef incoming reaches the phi on a path that does not pass the one object, so that
    # object does not dominate the phi: the parent phi stays, and the module still verifies.
    @testset "nodecayed_phis! keeps parent phis whose one object does not dominate them" begin
        @test @filecheck begin
            @check_label "@kernel_undef"
            @check "merge:"
            @check "phi {} addrspace(10)*"
            @check_same "[ undef, %skip ]"
            @check "load double addrspace(13)*"
            @check "ret double"
            LLVM.Context() do ctx
                mod = parse(
                    LLVM.Module, """
                    source_filename = "start"
                    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                    target triple = "x86_64-linux-gnu"

                    define double @kernel_undef({} addrspace(10)* addrspace(11)* nocapture readonly %field, i1 %c, i64 %i) #0 {
                    top:
                      br i1 %c, label %have, label %skip

                    have:
                      %obj = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %field unordered, align 8
                      %hdr = bitcast {} addrspace(10)* %obj to double addrspace(13)* addrspace(10)*
                      %hdr11 = addrspacecast double addrspace(13)* addrspace(10)* %hdr to double addrspace(13)* addrspace(11)*
                      %data.have = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %hdr11, align 8
                      br label %merge

                    skip:
                      br label %merge

                    merge:
                      %data = phi double addrspace(13)* [ %data.have, %have ], [ undef, %skip ]
                      %e = getelementptr inbounds double, double addrspace(13)* %data, i64 %i
                      %x = load double, double addrspace(13)* %e, align 8
                      ret double %x
                    }

                    attributes #0 = { "enzymejl_world"="1" }
                    """
                )

                Enzyme.Compiler.nodecayed_phis!(mod)
                LLVM.verify(mod)
                string(mod)
            end
        end
    end
end


# --- fix_decayaddr! -----------------------------------------------------------

struct DecayBig
    x::NTuple{100, Float64}
end

# `===` on a padding-free immutable this large lowers to `emit_bits_compare`,
# which decays both operands through `julia.pointer_from_objref` and compares
# them with `memcmp`; LLVM's `LibCallSimplifier::optimizeMemCmp` then rewrites
# that to `bcmp`, since the result feeds nothing but an `icmp eq ..., 0`.
# `@nospecialize` keeps the arguments boxed, so the operands are `addrspace(10)`
# and the decay is the one `fix_decayaddr!` has to deal with.
@noinline function decay_egal(@nospecialize(a), @nospecialize(b))
    return (a::DecayBig) === (b::DecayBig)
end

"""
    decay_egal_module()

The module Julia emits for [`decay_egal`](@ref), run through Enzyme's own
pre-AD optimization pipeline. Everything the test relies on -- the libcall, its
attributes, the `jl_roots` operand bundle -- comes from that emission rather
than from hand-written IR.

The module is generated straight into the active context, the same way
Enzyme's compile pipeline does it, rather than being round-tripped through
`code_llvm` text: Julia's own codegen context need not agree with a fresh
one on typed vs. opaque pointers (it does not on 1.11), and the text form
of the one cannot be parsed in the other.
"""
function decay_egal_module()
    target = Enzyme.Compiler.DefaultCompilerTarget()
    params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode)
    mi = Enzyme.Compiler.my_methodinstance(nothing, typeof(decay_egal), Tuple{Any, Any})
    job = GPUCompiler.CompilerJob(
        mi,
        GPUCompiler.CompilerConfig(
            target, params;
            kernel = false, libraries = true, toplevel = true, optimize = false,
            cleanup = false, only_entry = false, validate = false,
        ),
    )
    GPUCompiler.prepare_job!(job)
    mod, _ = GPUCompiler.emit_llvm(job)
    Enzyme.Compiler.optimize!(mod, Enzyme.Compiler.JIT.get_tm(), #=enzyme_ctx=# nothing)
    return mod
end

"The `memcmp` / `bcmp` call in `mod`, or `nothing` if there is none."
function find_bits_compare(mod::LLVM.Module)
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        isa(inst, LLVM.CallInst) || continue
        callee = LLVM.called_operand(inst)
        isa(callee, LLVM.Function) || continue
        if LLVM.name(callee) in ("bcmp", "memcmp")
            return inst
        end
    end
    return nothing
end

"""
    collapse_decay!(call)

Rewrite each `julia.pointer_from_objref(addrspacecast p10 -> p11)` feeding
`call` into the direct `addrspacecast p10 -> p0` it stands for, and return how
many were rewritten. Neither Julia nor Enzyme's pre-AD pipeline forms that cast
here -- it is what a later simplification of the two-step derivation leaves
behind, and it is the input `fix_decayaddr!` has to repair.
"""
function collapse_decay!(call::LLVM.CallInst)
    n = 0
    for (i, arg) in enumerate(Enzyme.Compiler.arg_operands_view(call))
        # With typed pointers the `{}*` result is bitcast to `i8*` first; look
        # through that to the derivation underneath.
        pfo = isa(arg, LLVM.BitCastInst) ? operands(arg)[1] : arg
        isa(pfo, LLVM.CallInst) || continue
        callee = LLVM.called_operand(pfo)
        (isa(callee, LLVM.Function) && LLVM.name(callee) == "julia.pointer_from_objref") ||
            continue
        src = operands(pfo)[1]
        isa(src, LLVM.AddrSpaceCastInst) || continue
        obj = operands(src)[1]
        LLVM.addrspace(value_type(obj)) == 10 || continue
        b = LLVM.IRBuilder()
        LLVM.position!(b, call)
        LLVM.API.LLVMSetOperand(
            call, i - 1, LLVM.addrspacecast!(b, obj, value_type(arg))
        )
        if arg != pfo && isempty(LLVM.uses(arg))
            LLVM.erase!(arg)
        end
        isempty(LLVM.uses(pfo)) && LLVM.erase!(pfo)
        n += 1
    end
    return n
end

"Strip every read-only marker from `call` and from the function it calls."
function drop_readonly!(call::LLVM.CallInst)
    callee = LLVM.called_operand(call)::LLVM.Function
    for attrs in (LLVM.function_attributes(callee), LLVM.function_attributes(call))
        for attr in collect(attrs)
            if Enzyme.Compiler.is_readonly(attr)
                delete!(attrs, attr)
            end
        end
    end
    return nothing
end

@testset "fix_decayaddr! read-only libcall" begin
    GPUCompiler.JuliaContext() do ctx
        mod = decay_egal_module()
        cmp = find_bits_compare(mod)
        @test cmp !== nothing
        # The callee has to be read-only for the rewrite below to apply at all;
        # that is how Julia and LLVM annotate `memcmp` / `bcmp`.
        @test Enzyme.Compiler.is_readonly(LLVM.called_operand(cmp)::LLVM.Function)
        @test collapse_decay!(cmp) == 2

        @test @filecheck begin
            # Each decayed argument becomes a gc-preserved
            # `julia.pointer_from_objref`, which late GC lowering turns back
            # into the cast that was there. The operand bundle is untouched --
            # only the argument operands get rewritten.
            @check_label "@julia_decay_egal"
            @check "gc_preserve_begin"
            @check "julia.pointer_from_objref"
            @check "gc_preserve_begin"
            @check "julia.pointer_from_objref"
            @check "@{{(bcmp|memcmp)}}("
            @check_same "jl_roots"
            @check "gc_preserve_end"
            @check "gc_preserve_end"
            Enzyme.Compiler.fix_decayaddr!(mod)
            string(mod)
        end

        # Nothing decays straight out of the tracked address space any more.
        for f in functions(mod), bb in blocks(f), inst in instructions(bb)
            if isa(inst, LLVM.AddrSpaceCastInst)
                @test !(
                    LLVM.addrspace(value_type(operands(inst)[1])) == 10 &&
                        LLVM.addrspace(value_type(inst)) == 0
                )
            end
        end
    end
end

@testset "fix_decayaddr! non-read-only libcall still rejected" begin
    # Guard against the read-only path swallowing calls that write through the
    # decayed pointer: those still need an sret to copy the object back.
    GPUCompiler.JuliaContext() do ctx
        mod = decay_egal_module()
        cmp = find_bits_compare(mod)
        @test cmp !== nothing
        @test collapse_decay!(cmp) == 2
        drop_readonly!(cmp)
        @test !Enzyme.Compiler.is_readonly(LLVM.called_operand(cmp)::LLVM.Function)

        @test_throws AssertionError Enzyme.Compiler.fix_decayaddr!(mod)
    end
end

@testset "fix_decayaddr! derived pointer into a heap-promoted roots buffer" begin
    # Since Julia 1.13.1, codegen passes a pointer into a `returnRoots` buffer
    # on as another call's roots argument. When Enzyme moves that buffer to the
    # GC heap, the argument becomes a derived pointer into the new object, cast
    # back to addrspace 0. The roots must be copied into a stack slot instead.
    rooted = string(convert(UInt, pointer_from_objref(Tuple{Vector{Float64}, Int})))
    LLVM.Context() do ctx
        if LLVM.supports_typed_pointers(ctx)
            return
        end
        mod = parse(
            LLVM.Module, """
            source_filename = "start"
            target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
            target triple = "x86_64-linux-gnu"

            declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

            declare void @use_roots(ptr nocapture readonly "enzymejl_rooted_typ"="$rooted")

            define void @promoted() {
            top:
              %obj = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 32, ptr addrspace(10) null)
              %derived = addrspacecast ptr addrspace(10) %obj to ptr addrspace(11)
              %slot = getelementptr i8, ptr addrspace(11) %derived, i64 8
              %decayed = addrspacecast ptr addrspace(11) %slot to ptr
              call void @use_roots(ptr nocapture readonly %decayed)
              ret void
            }

            define void @untracked(ptr addrspace(11) %arg) {
            top:
              %decayed = addrspacecast ptr addrspace(11) %arg to ptr
              call void @use_roots(ptr nocapture readonly %decayed)
              ret void
            }
            """
        )
        @test @filecheck begin
            @check_label "@promoted"
            @check "%[[TMP:[0-9]+]] = alloca [1 x ptr addrspace(10)]"
            @check "%[[SLOT:.+]] = getelementptr i8, ptr addrspace(11)"
            @check "load [1 x ptr addrspace(10)], ptr addrspace(11) %[[SLOT]]"
            @check "store [1 x ptr addrspace(10)] {{.*}}, ptr %[[TMP]]"
            @check "call void @use_roots(ptr nocapture readonly %[[TMP]])"
            # A derived pointer that does not trace back to a tracked object
            # is left alone.
            @check_label "@untracked"
            @check "addrspacecast ptr addrspace(11) %arg to ptr"
            Enzyme.Compiler.fix_decayaddr!(mod)
            string(mod)
        end
    end
end

# --- unfold_root_phi_loads! ---------------------------------------------------

function root_phi_addrspaces(f::LLVM.Function)
    spaces = Int[]
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.PHIInst)
            push!(spaces, Int(LLVM.addrspace(value_type(inst))))
        end
    end
    return spaces
end

@testset "unfold_root_phi_loads!" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            source_filename = "start"
            target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
            target triple = "x86_64-linux-gnu"

            define {} addrspace(10)* @diamond(i1 %cond, {} addrspace(10)** %arg) {
            top:
              %roots = alloca [2 x {} addrspace(10)*], align 8
              %r = bitcast [2 x {} addrspace(10)*]* %roots to {} addrspace(10)**
              br i1 %cond, label %a, label %b

            a:
              br label %merge

            b:
              br label %merge

            merge:
              %p = phi {} addrspace(10)** [ %r, %a ], [ %arg, %b ]
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              ret {} addrspace(10)* %ld
            }

            define {} addrspace(10)* @selfref(i1 %cond) {
            top:
              %roots = alloca [2 x {} addrspace(10)*], align 8
              %r = bitcast [2 x {} addrspace(10)*]* %roots to {} addrspace(10)**
              br label %header

            header:
              %p = phi {} addrspace(10)** [ %r, %top ], [ %p, %latch ]
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              br i1 %cond, label %latch, label %exit

            latch:
              br label %header

            exit:
              ret {} addrspace(10)* %ld
            }

            define {} addrspace(10)* @selfref_gep(i1 %cond) {
            top:
              %roots = alloca [2 x {} addrspace(10)*], align 8
              %r = bitcast [2 x {} addrspace(10)*]* %roots to {} addrspace(10)**
              br label %header

            header:
              %p = phi {} addrspace(10)** [ %r, %top ], [ %g, %latch ]
              %g = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)** %p, i64 1
              %ld = load {} addrspace(10)*, {} addrspace(10)** %g, align 8
              br i1 %cond, label %latch, label %exit

            latch:
              br label %header

            exit:
              ret {} addrspace(10)* %ld
            }
            """
        )

        # The pass replaces the phi of root-array pointers with a phi of loads.
        diamond = functions(mod)["diamond"]
        @test root_phi_addrspaces(diamond) == [0]
        @test Enzyme.Compiler.unfold_root_phi_loads!(diamond)
        @test root_phi_addrspaces(diamond) == [10]

        # The phi gets a value from its own block, and that value is the phi.
        # The pass must not change this function.
        selfref = functions(mod)["selfref"]
        @test !Enzyme.Compiler.unfold_root_phi_loads!(selfref)
        @test root_phi_addrspaces(selfref) == [0]

        # The same, but the value from its own block is a GEP of the phi.
        selfref_gep = functions(mod)["selfref_gep"]
        @test !Enzyme.Compiler.unfold_root_phi_loads!(selfref_gep)
        @test root_phi_addrspaces(selfref_gep) == [0]

        @test LLVM.verify(mod) === nothing
    end
end

@testset "unfold_root_phi_loads! with the load hoisted past the phi's block" begin
    # LLVM hoists the load of the merged root into a later block, such as the
    # preheader of the loop that uses it (`copyto!` from a row view on 1.13.1).
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            source_filename = "start"
            target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
            target triple = "x86_64-linux-gnu"

            declare i64 @llvm.smax.i64(i64, i64)
            declare void @fill({} addrspace(10)**)

            define {} addrspace(10)* @preheader(i1 %c, i64 %n, {} addrspace(10)** nonnull dereferenceable(8) %arg) {
            top:
              %roots = alloca {} addrspace(10)*, align 8
              br i1 %c, label %a, label %b

            a:
              call void @fill({} addrspace(10)** %roots)
              br label %merge

            b:
              br label %merge

            merge:
              %p = phi {} addrspace(10)** [ %roots, %a ], [ %arg, %b ]
              %m = call i64 @llvm.smax.i64(i64 %n, i64 0)
              %cc = icmp ult i64 %m, 4
              br i1 %cc, label %ph, label %exit

            ph:
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              ret {} addrspace(10)* %ld

            exit:
              ret {} addrspace(10)* null
            }

            define {} addrspace(10)* @store_between(i1 %c, i64 %n, {} addrspace(10)** nonnull dereferenceable(8) %arg, i64* %out) {
            top:
              %roots = alloca {} addrspace(10)*, align 8
              br i1 %c, label %a, label %b

            a:
              call void @fill({} addrspace(10)** %roots)
              br label %merge

            b:
              br label %merge

            merge:
              %p = phi {} addrspace(10)** [ %roots, %a ], [ %arg, %b ]
              %cc = icmp ult i64 %n, 4
              br i1 %cc, label %ph, label %exit

            ph:
              store i64 %n, i64* %out, align 8
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              ret {} addrspace(10)* %ld

            exit:
              ret {} addrspace(10)* null
            }

            define {} addrspace(10)* @not_dereferenceable(i1 %c, i64 %n, {} addrspace(10)** %arg) {
            top:
              %roots = alloca {} addrspace(10)*, align 8
              br i1 %c, label %a, label %b

            a:
              call void @fill({} addrspace(10)** %roots)
              br label %merge

            b:
              br label %merge

            merge:
              %p = phi {} addrspace(10)** [ %roots, %a ], [ %arg, %b ]
              %cc = icmp ult i64 %n, 4
              br i1 %cc, label %ph, label %exit

            ph:
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              ret {} addrspace(10)* %ld

            exit:
              ret {} addrspace(10)* null
            }

            define {} addrspace(10)* @two_preds(i1 %c, i1 %d, {} addrspace(10)** nonnull dereferenceable(8) %arg) {
            top:
              %roots = alloca {} addrspace(10)*, align 8
              br i1 %c, label %a, label %b

            a:
              call void @fill({} addrspace(10)** %roots)
              br label %merge

            b:
              br label %merge

            merge:
              %p = phi {} addrspace(10)** [ %roots, %a ], [ %arg, %b ]
              br i1 %d, label %ph, label %other

            other:
              br label %ph

            ph:
              %ld = load {} addrspace(10)*, {} addrspace(10)** %p, align 8
              ret {} addrspace(10)* %ld
            }
            """
        )

        # A read-only intrinsic between the phi and the load does not block the
        # rewrite, and both incoming pointers may be loaded speculatively.
        preheader = functions(mod)["preheader"]
        @test Enzyme.Compiler.unfold_root_phi_loads!(preheader)
        @test root_phi_addrspaces(preheader) == [10]

        # A store before the load could change what it reads.
        store_between = functions(mod)["store_between"]
        @test !Enzyme.Compiler.unfold_root_phi_loads!(store_between)
        @test root_phi_addrspaces(store_between) == [0]

        # Loading the argument where the program did not is only safe when it is
        # known to be dereferenceable.
        not_dereferenceable = functions(mod)["not_dereferenceable"]
        @test !Enzyme.Compiler.unfold_root_phi_loads!(not_dereferenceable)
        @test root_phi_addrspaces(not_dereferenceable) == [0]

        # The load's block has two predecessors, so the phi's block does not
        # lead straight to it.
        two_preds = functions(mod)["two_preds"]
        @test !Enzyme.Compiler.unfold_root_phi_loads!(two_preds)
        @test root_phi_addrspaces(two_preds) == [0]

        @test LLVM.verify(mod) === nothing
    end
end

@testset "addrspace(11) argument phis are left alone" begin
    # LLVM 20 strength-reduces loop indices into pointer induction variables, so a
    # loop over an addrspace(11) argument (an SVector passed by reference) becomes
    # `phi [ %arg, %top ], [ gep(%inv, %iv), %latch ]` with a non-constant GEP index.
    # There is no addrspace(10) object to root such a phi with; `nodecayed_phis!`
    # must skip it instead of failing in `nodecayed_getparent`.
    @test @filecheck begin
        @check_label "define double @loop"
        @check_not "nodecayed."
        @check "%p = phi"
        @check_not "nodecayed."
        @check_label "define double @sel"
        @check_not "nodecayed."
        @check "%p = phi"
        @check_not "nodecayed."
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                define double @loop(double addrspace(11)* %arg, i64 %n) {
                top:
                  %inv = getelementptr double, double addrspace(11)* %arg, i64 -1
                  br label %loop

                loop:
                  %p = phi double addrspace(11)* [ %arg, %top ], [ %gep, %latch ]
                  %iv = phi i64 [ 1, %top ], [ %ivnext, %latch ]
                  %acc = phi double [ 0.0, %top ], [ %accnext, %latch ]
                  %v = load double, double addrspace(11)* %p, align 8
                  %accnext = fadd double %acc, %v
                  %done = icmp eq i64 %iv, %n
                  br i1 %done, label %exit, label %latch

                latch:
                  %ivnext = add i64 %iv, 1
                  %gep = getelementptr double, double addrspace(11)* %inv, i64 %ivnext
                  br label %loop

                exit:
                  ret double %accnext
                }

                define double @sel(double addrspace(11)* %a, double addrspace(11)* %b, i1 %c, i64 %n) {
                top:
                  %b1 = getelementptr double, double addrspace(11)* %b, i64 1
                  %s = select i1 %c, double addrspace(11)* %a, double addrspace(11)* %b1
                  br label %loop

                loop:
                  %p = phi double addrspace(11)* [ %s, %top ], [ %gep, %latch ]
                  %iv = phi i64 [ 1, %top ], [ %ivnext, %latch ]
                  %acc = phi double [ 0.0, %top ], [ %accnext, %latch ]
                  %v = load double, double addrspace(11)* %p, align 8
                  %accnext = fadd double %acc, %v
                  %done = icmp eq i64 %iv, %n
                  br i1 %done, label %exit, label %latch

                latch:
                  %ivnext = add i64 %iv, 1
                  %gep = getelementptr double, double addrspace(11)* %p, i64 %ivnext
                  br label %loop

                exit:
                  ret double %accnext
                }
                """
            )

            Enzyme.Compiler.nodecayed_phis!(mod)
            string(mod)
        end
    end
end

# The address spaces of the pointer phis in `f`.
pointer_phi_addrspaces(f) = sort!(
    [
        LLVM.addrspace(value_type(inst)) for bb in blocks(f) for inst in instructions(bb)
            if isa(inst, LLVM.PHIInst) && isa(value_type(inst), LLVM.PointerType)
    ]
)

const ARGUMENT_PHI_HEADER = """
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"
"""

# Loads through a phi of an argument and a pointer into an object.
const ARGUMENT_PHI_UNFOLDED = """
; The argument comes straight from `top`, which also branches to `copy`, so that
; edge needs a block of its own for the load.
define i64 @field(i1 %c, {} addrspace(10)* %obj, { {} addrspace(10)*, i64, i64 } addrspace(11)* %arg) #0 {
top:
  br i1 %c, label %copy, label %merge

copy:
  %o = bitcast {} addrspace(10)* %obj to { {} addrspace(10)*, i64, i64 } addrspace(10)*
  %d = addrspacecast { {} addrspace(10)*, i64, i64 } addrspace(10)* %o to { {} addrspace(10)*, i64, i64 } addrspace(11)*
  br label %merge

merge:
  %p = phi { {} addrspace(10)*, i64, i64 } addrspace(11)* [ %d, %copy ], [ %arg, %top ]
  %f = getelementptr inbounds { {} addrspace(10)*, i64, i64 }, { {} addrspace(10)*, i64, i64 } addrspace(11)* %p, i64 0, i32 2
  %v = load i64, i64 addrspace(11)* %f, align 8
  ret i64 %v
}

; Two fields, one through a bitcast of a GEP, the tracked one atomic unordered.
define i64 @fields(i1 %c, {} addrspace(10)* %obj, { {} addrspace(10)*, i64, i64 } addrspace(11)* %arg) #0 {
top:
  br i1 %c, label %copy, label %orig

copy:
  %o = bitcast {} addrspace(10)* %obj to { {} addrspace(10)*, i64, i64 } addrspace(10)*
  %d = addrspacecast { {} addrspace(10)*, i64, i64 } addrspace(10)* %o to { {} addrspace(10)*, i64, i64 } addrspace(11)*
  br label %merge

orig:
  br label %merge

merge:
  %p = phi { {} addrspace(10)*, i64, i64 } addrspace(11)* [ %d, %copy ], [ %arg, %orig ]
  %f0 = getelementptr inbounds { {} addrspace(10)*, i64, i64 }, { {} addrspace(10)*, i64, i64 } addrspace(11)* %p, i64 0, i32 0
  %parent = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %f0 unordered, align 8
  %f1 = getelementptr inbounds { {} addrspace(10)*, i64, i64 }, { {} addrspace(10)*, i64, i64 } addrspace(11)* %p, i64 0, i32 1
  %f1a = bitcast i64 addrspace(11)* %f1 to [1 x i64] addrspace(11)*
  %a = load [1 x i64], [1 x i64] addrspace(11)* %f1a, align 8
  %ai = extractvalue [1 x i64] %a, 0
  %parenti = ptrtoint {} addrspace(10)* %parent to i64
  %s = add i64 %ai, %parenti
  ret i64 %s
}

; Every incoming value is an argument: left alone, as before.
define i64 @args(i1 %c, { {} addrspace(10)*, i64, i64 } addrspace(11)* %a, { {} addrspace(10)*, i64, i64 } addrspace(11)* %b) #0 {
top:
  br i1 %c, label %l, label %r

l:
  br label %merge

r:
  br label %merge

merge:
  %p = phi { {} addrspace(10)*, i64, i64 } addrspace(11)* [ %a, %l ], [ %b, %r ]
  %f = getelementptr inbounds { {} addrspace(10)*, i64, i64 }, { {} addrspace(10)*, i64, i64 } addrspace(11)* %p, i64 0, i32 2
  %v = load i64, i64 addrspace(11)* %f, align 8
  ret i64 %v
}
"""

const ARGUMENT_PHI_ATTRIBUTES = """
attributes #0 = { "enzymejl_world"="1" }
"""

# Phis of an argument and an object that stay: a write before the load, and the phi
# reaching a call.
const ARGUMENT_PHI_KEPT = """
define i64 @written(i1 %c, {} addrspace(10)* %obj, { {} addrspace(10)*, i64, i64 } addrspace(11)* %arg, i64* %out) #0 {
top:
  br i1 %c, label %copy, label %orig

copy:
  %o = bitcast {} addrspace(10)* %obj to { {} addrspace(10)*, i64, i64 } addrspace(10)*
  %d = addrspacecast { {} addrspace(10)*, i64, i64 } addrspace(10)* %o to { {} addrspace(10)*, i64, i64 } addrspace(11)*
  br label %merge

orig:
  br label %merge

merge:
  %p = phi { {} addrspace(10)*, i64, i64 } addrspace(11)* [ %d, %copy ], [ %arg, %orig ]
  store i64 0, i64* %out, align 8
  %f = getelementptr inbounds { {} addrspace(10)*, i64, i64 }, { {} addrspace(10)*, i64, i64 } addrspace(11)* %p, i64 0, i32 2
  %v = load i64, i64 addrspace(11)* %f, align 8
  ret i64 %v
}

declare i64 @use({ {} addrspace(10)*, i64, i64 } addrspace(11)*)

define i64 @escapes(i1 %c, {} addrspace(10)* %obj, { {} addrspace(10)*, i64, i64 } addrspace(11)* %arg) #0 {
top:
  br i1 %c, label %copy, label %orig

copy:
  %o = bitcast {} addrspace(10)* %obj to { {} addrspace(10)*, i64, i64 } addrspace(10)*
  %d = addrspacecast { {} addrspace(10)*, i64, i64 } addrspace(10)* %o to { {} addrspace(10)*, i64, i64 } addrspace(11)*
  br label %merge

orig:
  br label %merge

merge:
  %p = phi { {} addrspace(10)*, i64, i64 } addrspace(11)* [ %d, %copy ], [ %arg, %orig ]
  %v = call i64 @use({ {} addrspace(10)*, i64, i64 } addrspace(11)* %p)
  ret i64 %v
}
"""

@testset "nodecayed_phis! loads through a phi of an argument and an object in its predecessors" begin
    # Instcombine folds `phi [load (gep %obj, k)], [load (gep %arg, k)]` into a load
    # through `phi [%obj, %arg]`. The addrspace(11) argument has no addrspace(10)
    # parent, so `nodecayed_phis!` cannot rewrite that phi. Enzyme's augmented primal
    # of `copyto!` from a `SubArray` has one, and a nested derivative (Forward over
    # Reverse of `append!(v, @view w[i:end])`) runs `nodecayed_phis!` on it
    # (test/nested_unalias_view.jl).
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, ARGUMENT_PHI_HEADER * ARGUMENT_PHI_UNFOLDED * ARGUMENT_PHI_KEPT * ARGUMENT_PHI_ATTRIBUTES)
        field = functions(mod)["field"]
        fields = functions(mod)["fields"]
        @test pointer_phi_addrspaces(field) == [11]
        @test pointer_phi_addrspaces(fields) == [11]

        # Not rewritten: a write before the load, a phi that reaches a call, and a
        # phi of arguments only (which `nodecayed_phis!` leaves alone).
        for name in ("written", "escapes", "args")
            @test !Enzyme.Compiler.unfold_argument_phi_loads!(functions(mod)[name])
            @test pointer_phi_addrspaces(functions(mod)[name]) == [11]
        end

        @test Enzyme.Compiler.unfold_argument_phi_loads!(field)
        @test Enzyme.Compiler.unfold_argument_phi_loads!(fields)
        @test LLVM.verify(mod) === nothing

        # The loads moved to the predecessors: the addrspace(11) phi is gone, and the
        # merge block takes a phi of each loaded value.
        T_prjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), 10)
        for (f, merged) in ((field, [LLVM.IntType(64)]), (fields, [LLVM.ArrayType(LLVM.IntType(64), 1), T_prjlvalue]))
            merge = only(bb for bb in blocks(f) if LLVM.name(bb) == "merge")
            phis = [inst for inst in instructions(merge) if isa(inst, LLVM.PHIInst)]
            @test sort!(string.(value_type.(phis))) == sort!(string.(merged))
            for phi in phis, (v, pred) in LLVM.incoming(phi)
                # Each value is loaded at the end of its own predecessor, which
                # branches only to the merge block.
                @test isa(v, LLVM.LoadInst)
                @test LLVM.parent(v) == pred
                @test collect(successors(terminator(pred))) == [merge]
            end
            @test !any(isa(inst, LLVM.LoadInst) for inst in instructions(merge))
        end
        @test pointer_phi_addrspaces(field) == Int[]
        @test pointer_phi_addrspaces(fields) == [10]
        # The tracked field stays an atomic unordered load.
        fields_merge = only(bb for bb in blocks(fields) if LLVM.name(bb) == "merge")
        tracked = only(inst for inst in instructions(fields_merge) if isa(inst, LLVM.PHIInst) && isa(value_type(inst), LLVM.PointerType))
        for (v, _) in LLVM.incoming(tracked)
            @test LLVM.API.LLVMGetOrdering(v) == LLVM.API.LLVMAtomicOrderingUnordered
        end
        # `field`'s argument came straight from `top`, so that edge got a block.
        @test count(bb -> LLVM.name(bb) == "unsink.edge", blocks(field)) == 1
    end

    # `nodecayed_phis!` itself does this before it looks for parents. The phis it
    # still cannot root make their block throw when reached, as before.
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, ARGUMENT_PHI_HEADER * ARGUMENT_PHI_UNFOLDED * ARGUMENT_PHI_KEPT * ARGUMENT_PHI_ATTRIBUTES)
        Enzyme.Compiler.nodecayed_phis!(mod)
        @test LLVM.verify(mod) === nothing
        @test pointer_phi_addrspaces(functions(mod)["field"]) == Int[]
        @test pointer_phi_addrspaces(functions(mod)["fields"]) == [10]
        @test pointer_phi_addrspaces(functions(mod)["args"]) == [11]
        for name in ("field", "fields", "args")
            @test !occursin("nodecayed.", string(functions(mod)[name]))
            @test !occursin("jl_throw", string(functions(mod)[name]))
        end
        for name in ("written", "escapes")
            f = functions(mod)[name]
            merge = only(bb for bb in blocks(f) if LLVM.name(bb) == "merge")
            @test isa(terminator(merge), LLVM.UnreachableInst)
            @test any(inst -> isa(inst, LLVM.CallInst) && occursin("jl_throw", LLVM.name(LLVM.called_operand(inst))), instructions(merge))
            @test pointer_phi_addrspaces(f) == Int[]
        end
    end
end

@testset "get_function! doesn't rename a clashing declaration" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            @abort = external global i8

            declare void @existing(i64)
            """
        )
        FT = LLVM.FunctionType(LLVM.VoidType())
        @test_throws ArgumentError Enzyme.Compiler.get_function!(mod, "abort", FT)
        @test !haskey(LLVM.functions(mod), "abort.1")

        existing = LLVM.functions(mod)["existing"]
        F, _ = Enzyme.Compiler.get_function!(mod, "existing", LLVM.FunctionType(LLVM.VoidType(), [LLVM.Int64Type()]))
        @test F == existing

        F, _ = Enzyme.Compiler.get_function!(mod, "fresh", FT)
        @test LLVM.name(F) == "fresh"
    end
end

# A load through a merge of what a call returned and an argument, behind a branch.
function merge_tail_ir(typed::Bool)
    obj = typed ? "{} addrspace(10)*" : "ptr addrspace(10)"
    data = typed ? "double addrspace(13)*" : "ptr addrspace(13)"
    slot = typed ? "double addrspace(13)* addrspace(11)*" : "ptr addrspace(11)"
    lenp = typed ? "i64*" : "ptr"
    sides = """
      %fs = addrspacecast $obj %f to $slot
      br i1 %c, label %left, label %right

    left:
      %fd = load $data, $slot %fs, align 8
      %pl = getelementptr inbounds double, $data %fd, i64 %i
      br label %merge

    right:
      %as = addrspacecast $obj %a to $slot
      %ad = load $data, $slot %as, align 8
      %pr = getelementptr inbounds double, $data %ad, i64 %i
      br label %merge
    """
    return """
    source_filename = "start"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
    target triple = "x86_64-linux-gnu"

    declare noalias nonnull $obj @fresh()
    declare void @throw(i64) noreturn
    declare void @throw_array($obj) noreturn
    declare void @use($data)

    ; The load sits behind a check that does not depend on the merge. `%len` is used
    ; in both blocks behind it.
    define double @behind(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      %v = load double, $data %p, align 8
      %s = sitofp i64 %len to double
      %r = fmul double %v, %s
      ret double %r
    }

    ; Two branches between the merge and the load.
    define double @twice(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %mid, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    mid:
      %ok2 = icmp ne i64 %i, 3
      br i1 %ok2, label %in, label %oob2

    oob2:
      call void @throw(i64 %i)
      unreachable

    in:
      %v = load double, $data %p, align 8
      ret double %v
    }

    ; A merge of whole arrays, as a select, whose other use is the bounds error.
    define double @picked(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
      %m = select i1 %c, $obj %f, $obj %a
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw_array($obj %m)
      unreachable

    in:
      %ms = addrspacecast $obj %m to $slot
      %md = load $data, $slot %ms, align 8
      %p = getelementptr inbounds double, $data %md, i64 %i
      %v = load double, $data %p, align 8
      ret double %v
    }

    ; Left alone. The pointer reaches a call that returns.
    define double @escapes(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      call void @use($data %p)
      %v = load double, $data %p, align 8
      ret double %v
    }

    ; Left alone. The load's block has another predecessor.
    define double @joined(i1 %c, i1 %d, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      br i1 %d, label %in, label %other

    other:
      br label %in

    in:
      %v = load double, $data %p, align 8
      ret double %v
    }

    ; A store before the load, which it could write: the load stays behind it.
    define double @written(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      store i64 0, $lenp %lenp, align 8
      %v = load double, $data %p, align 8
      br label %done

    done:
      ret double %v
    }

    ; Left alone. As @written, but cloning the load's block would clone the return.
    define double @written_returns(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      store i64 0, $lenp %lenp, align 8
      %v = load double, $data %p, align 8
      ret double %v
    }

    ; A value of the load's block is also used by a block that never returns.
    define double @dead_behind(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
      %m = select i1 %c, $obj %f, $obj %a
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      %ms = addrspacecast $obj %m to $slot
      %md = load $data, $slot %ms, align 8
      %p = getelementptr inbounds double, $data %md, i64 %i
      %v = load double, $data %p, align 8
      %big = fcmp ogt double %v, 1.0
      br i1 %big, label %fine, label %oob2

    oob2:
      call void @use($data %p)
      unreachable

    fine:
      ret double %v
    }

    ; Left alone. Another pointer into an object leaves the block to be cloned.
    define double @other_pointer(i1 %c, i64 %i, $lenp %lenp, $obj %a) {
    top:
      %f = call $obj @fresh()
      %as2 = addrspacecast $obj %a to $slot
      %pl2 = load $data, $slot %as2, align 8
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %q = getelementptr inbounds double, $data %pl2, i64 %i
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      %v = load double, $data %p, align 8
      %w = load double, $data %q, align 8
      %r = fadd double %v, %w
      ret double %r
    }

    ; Left alone. Neither side is data the function made.
    define double @arguments(i1 %c, i64 %i, $lenp %lenp, $obj %a, $obj %f) {
    top:
    $sides
    merge:
      %p = phi $data [ %pl, %left ], [ %pr, %right ]
      %len = load i64, $lenp %lenp, align 8
      %ok = icmp ult i64 %i, %len
      br i1 %ok, label %in, label %oob

    oob:
      call void @throw(i64 %len)
      unreachable

    in:
      %v = load double, $data %p, align 8
      ret double %v
    }
    """
end

function merge_tail_split!(mod, name)
    f = functions(mod)[name]
    Enzyme.Compiler.split_constant_merges!(mod, Enzyme.Compiler.ConstantFields(f, Set{LLVM.Value}()))
    return f
end

@testset "Merges of data a function made, behind branches" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, merge_tail_ir(LLVM.supports_typed_pointers(ctx)))

        behind = merge_tail_split!(mod, "behind")
        @test LLVM.verify(mod) === nothing
        @test pointer_phi_addrspaces(behind) == Int[]
        in_block = only(bb for bb in blocks(behind) if LLVM.name(bb) == "in")
        loaded = only(inst for inst in instructions(in_block) if isa(inst, LLVM.PHIInst) && value_type(inst) == LLVM.DoubleType())
        for (v, pred) in LLVM.incoming(loaded)
            # Each load runs on its own edge into `in`, behind its copy of the check.
            @test isa(v, LLVM.LoadInst)
            @test LLVM.parent(v) == pred
            @test collect(successors(terminator(pred))) == [in_block]
            check = only(LLVM.predecessors(pred))
            @test any(isa(inst, LLVM.ICmpInst) for inst in instructions(check))
        end
        # `%len`, used behind the check, is a phi on both sides of it.
        oob = only(bb for bb in blocks(behind) if LLVM.name(bb) == "oob")
        @test count(inst -> isa(inst, LLVM.PHIInst) && value_type(inst) == LLVM.IntType(64), collect(instructions(oob))) == 1
        @test count(inst -> isa(inst, LLVM.PHIInst) && value_type(inst) == LLVM.IntType(64), collect(instructions(in_block))) == 1
        @test length(LLVM.predecessors(oob)) == 2

        twice = merge_tail_split!(mod, "twice")
        @test LLVM.verify(mod) === nothing
        @test pointer_phi_addrspaces(twice) == Int[]
        in_block = only(bb for bb in blocks(twice) if LLVM.name(bb) == "in")
        @test count(inst -> isa(inst, LLVM.PHIInst) && value_type(inst) == LLVM.DoubleType(), collect(instructions(in_block))) == 1
        @test !any(isa(inst, LLVM.LoadInst) for inst in instructions(in_block))

        picked = merge_tail_split!(mod, "picked")
        @test LLVM.verify(mod) === nothing
        @test !any(isa(inst, LLVM.SelectInst) for bb in blocks(picked) for inst in instructions(bb))
        in_block = only(bb for bb in blocks(picked) if LLVM.name(bb) == "in")
        @test pointer_phi_addrspaces(picked) == [10]
        oob = only(bb for bb in blocks(picked) if LLVM.name(bb) == "oob")
        @test isa(first(instructions(oob)), LLVM.PHIInst)
        @test !any(isa(inst, LLVM.PHIInst) && isa(value_type(inst), LLVM.PointerType) for inst in instructions(in_block))

        written = merge_tail_split!(mod, "written")
        @test LLVM.verify(mod) === nothing
        @test pointer_phi_addrspaces(written) == Int[]
        for bb in blocks(written)
            insts = collect(instructions(bb))
            at = findfirst(inst -> isa(inst, LLVM.LoadInst) && value_type(inst) == LLVM.DoubleType(), insts)
            at === nothing && continue
            @test any(isa(inst, LLVM.StoreInst) for inst in insts[1:at])
        end

        dead_behind = merge_tail_split!(mod, "dead_behind")
        @test LLVM.verify(mod) === nothing
        @test !any(isa(inst, LLVM.SelectInst) for bb in blocks(dead_behind) for inst in instructions(bb))
        live_phis = [
            inst for bb in blocks(dead_behind) for inst in instructions(bb)
                if isa(inst, LLVM.PHIInst) && isa(value_type(inst), LLVM.PointerType) && LLVM.name(bb) != "oob2" && LLVM.name(bb) != "oob"
        ]
        @test isempty(live_phis)

        for (name, kept) in (("escapes", 13), ("joined", 13), ("written_returns", 13), ("other_pointer", 13), ("arguments", 13))
            f = merge_tail_split!(mod, name)
            @test LLVM.verify(mod) === nothing
            @test pointer_phi_addrspaces(f) == [kept]
            @test any(LLVM.name(bb) == "merge" for bb in blocks(f))
        end
    end
end

# Two functions whose merges each need a block of their own on an edge into them.
function merge_edges_debug_ir(typed::Bool)
    obj = typed ? "{} addrspace(10)*" : "ptr addrspace(10)"
    data = typed ? "double addrspace(13)*" : "ptr addrspace(13)"
    slot = typed ? "double addrspace(13)* addrspace(11)*" : "ptr addrspace(11)"
    body(dbg) = """
    top:
      %f = call $obj @fresh(), !dbg $dbg
      %fs = addrspacecast $obj %f to $slot, !dbg $dbg
      %fd = load $data, $slot %fs, align 8, !dbg $dbg
      %as = addrspacecast $obj %a to $slot, !dbg $dbg
      %ad = load $data, $slot %as, align 8, !dbg $dbg
      br i1 %c, label %merge, label %right, !dbg $dbg

    right:
      br i1 %d, label %merge, label %other, !dbg $dbg

    other:
      ret double 0.0, !dbg $dbg

    merge:
      %p = phi $data [ %fd, %top ], [ %ad, %right ], !dbg $dbg
      %v = load double, $data %p, align 8, !dbg $dbg
      ret double %v, !dbg $dbg
    """
    return """
    source_filename = "start"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
    target triple = "x86_64-linux-gnu"

    declare noalias nonnull $obj @fresh()

    define double @first(i1 %c, i1 %d, $obj %a) !dbg !5 {
    $(body("!7"))
    }

    define double @second(i1 %c, i1 %d, $obj %a) !dbg !6 {
    $(body("!8"))
    }

    define double @both(i1 %c, i1 %d, $obj %a) {
    top:
      %x = call double @first(i1 %c, i1 %d, $obj %a)
      %y = call double @second(i1 %c, i1 %d, $obj %a)
      %r = fadd double %x, %y
      ret double %r
    }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!1}
    !0 = distinct !DICompileUnit(language: DW_LANG_Julia, file: !2, producer: "julia", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
    !1 = !{i32 2, !"Debug Info Version", i32 3}
    !2 = !DIFile(filename: "merges.jl", directory: ".")
    !3 = !DISubroutineType(types: !4)
    !4 = !{}
    !5 = distinct !DISubprogram(name: "first", linkageName: "first", scope: null, file: !2, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !4)
    !6 = distinct !DISubprogram(name: "second", linkageName: "second", scope: null, file: !2, line: 2, type: !3, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !4)
    !7 = !DILocation(line: 1, scope: !5)
    !8 = !DILocation(line: 2, scope: !6)
    """
end

@testset "Splitting edges keeps each function's debug locations" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, merge_edges_debug_ir(LLVM.supports_typed_pointers(ctx)))
        @test LLVM.verify(mod) === nothing
        both = functions(mod)["both"]
        Enzyme.Compiler.split_constant_merges!(mod, Enzyme.Compiler.ConstantFields(both, Set{LLVM.Value}()))
        @test LLVM.verify(mod) === nothing
        for name in ("first", "second")
            f = functions(mod)[name]
            @test pointer_phi_addrspaces(f) == Int[]
            @test count(bb -> startswith(LLVM.name(bb), "unsink.edge"), blocks(f)) == 2
        end
    end
end
