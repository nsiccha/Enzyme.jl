using Enzyme, Test
using LLVM

# `autodiff_deferred` called outside of a differentiated function jumps through a lazy JIT
# trampoline, whose materializer compiles the derivative and emits its module against a
# materialization responsibility that covers the trampoline's target alone. Julia's object
# layer does not claim the other global symbols such an object defines, yet resolves them
# through that responsibility, and a release build of LLVM then reads and writes past the
# end of the dylib's symbol table: the process segfaults (`JIT session error: Failed to
# materialize symbols`, then `unknown function (ip: (nil))`), or its heap is corrupted and it
# aborts later (`double free or corruption`). Julia 1.10 gives a derivative module such a
# symbol for every `ccall` whose library is a run-time value (`libname_<function>_<n>`), so
# `JIT.export_only!` gives every definition but the target internal linkage.

const DT_JIT = Enzyme.Compiler.JIT

function dt_return_constant!(fn::LLVM.Function, value::Integer)
    @dispose builder = IRBuilder() begin
        position!(builder, BasicBlock(fn, "top"))
        ret!(builder, ConstantInt(LLVM.Int64Type(), value))
    end
    return fn
end

function dt_defined_function!(mod::LLVM.Module, name::String, linkage, value::Integer)
    fn = dt_return_constant!(LLVM.Function(mod, name, LLVM.FunctionType(LLVM.Int64Type())), value)
    linkage!(fn, linkage)
    return fn
end

function dt_defined_global!(mod::LLVM.Module, name::String, linkage, value::Integer)
    gv = GlobalVariable(mod, LLVM.Int64Type(), name)
    initializer!(gv, ConstantInt(LLVM.Int64Type(), value))
    linkage!(gv, linkage)
    return gv
end

@testset "A trampoline's module exports only its target" begin
    @dispose ctx = Context() mod = LLVM.Module("trampoline") begin
        entry = dt_defined_function!(mod, "entry", LLVM.API.LLVMExternalLinkage, 1)
        other = dt_defined_function!(mod, "other", LLVM.API.LLVMExternalLinkage, 2)
        odr = dt_defined_function!(mod, "odr", LLVM.API.LLVMLinkOnceODRLinkage, 3)
        weak = dt_defined_function!(mod, "weak", LLVM.API.LLVMWeakAnyLinkage, 4)
        internal = dt_defined_function!(mod, "internal", LLVM.API.LLVMInternalLinkage, 5)
        declared = LLVM.Function(mod, "declared", LLVM.FunctionType(LLVM.Int64Type()))
        cache = dt_defined_global!(mod, "libname_cos_1", LLVM.API.LLVMExternalLinkage, 0)
        weak_odr = dt_defined_global!(mod, "weak_odr_global", LLVM.API.LLVMWeakODRLinkage, 1)
        common = dt_defined_global!(mod, "common_global", LLVM.API.LLVMCommonLinkage, 0)
        private = dt_defined_global!(mod, "private_global", LLVM.API.LLVMPrivateLinkage, 2)
        external_global = GlobalVariable(mod, LLVM.Int64Type(), "external_global")
        alias = LLVM.API.LLVMAddAlias2(mod, LLVM.FunctionType(LLVM.Int64Type()), 0, other, "other_alias")

        DT_JIT.export_only!(mod, "entry")

        @test linkage(entry) == LLVM.API.LLVMExternalLinkage
        for value in (other, odr, weak, internal, cache, weak_odr, common)
            @test linkage(value) == LLVM.API.LLVMInternalLinkage
        end
        @test LLVM.API.LLVMGetLinkage(alias) == LLVM.API.LLVMInternalLinkage
        @test linkage(private) == LLVM.API.LLVMPrivateLinkage
        @test isdeclaration(declared) && linkage(declared) == LLVM.API.LLVMExternalLinkage
        @test isdeclaration(external_global) && linkage(external_global) == LLVM.API.LLVMExternalLinkage
        @test (LLVM.verify(mod); true)
    end
end

# Top-level deferred gradients of functions whose derivative module carries such symbols: a
# custom rule calls libm through a library name held in a non-constant global. Each child
# compiles several trampolines after an ordinary gradient of the same function, exits through
# Julia's normal exit (which is where a corrupted heap made the reported process abort), and
# must print every gradient right.
const DEFERRED_TRAMPOLINE_CHILD = raw"""
using Enzyme
using Enzyme: EnzymeRules
dt_libm = Base.libm_name  # not constant: the ccalls below look their library up at run time
dt_cos(x::Float64) = ccall((:cos, dt_libm), Float64, (Float64,), x)
function EnzymeRules.augmented_primal(config, func::Const{typeof(dt_cos)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, x.val)
end
function EnzymeRules.reverse(config, func::Const{typeof(dt_cos)}, dret::Active, tape, x::Active)
    return (-ccall((:sin, dt_libm), Float64, (Float64,), tape) * dret.val,)
end
const NF = 8
for k in 1:NF
    @eval $(Symbol(:dt_f, k))(v) = dt_cos(v[1] * $k) * v[2] + $k * v[2]
end
function dt_deferred!(f, dv, v)
    fill!(dv, 0.0)
    Enzyme.autodiff_deferred(Reverse, Const(f), Active, Duplicated(v, dv))
    return nothing
end
v = [0.3, 1.7]
right = 0
for k in 1:NF
    f = getfield(Main, Symbol(:dt_f, k))
    expected = [-sin(v[1] * k) * k * v[2], cos(v[1] * k) + k]
    dv = zeros(2)
    Enzyme.autodiff(Reverse, Const(f), Active, Duplicated(v, dv))
    plain = isapprox(dv, expected; rtol = 1.0e-12)
    dv = zeros(2)
    dt_deferred!(f, dv, v)
    global right += plain && isapprox(dv, expected; rtol = 1.0e-12)
end
GC.gc(true)
println("DONE ", right, " ", NF)
"""

@testset "Top-level autodiff_deferred through a run-time library ccall" begin
    for round in 1:2
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $DEFERRED_TRAMPOLINE_CHILD`
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
        timer = Timer(900) do _
            kill(proc, Base.SIGKILL)
        end
        wait(proc)
        close(timer)
        output = String(take!(out))
        done = filter(startswith("DONE "), split(output, '\n'))
        @test success(proc)
        @test done == ["DONE 8 8"]
        if !success(proc) || done != ["DONE 8 8"]
            println(stderr, "Child output (round ", round, "):\n", output)
        end
    end
end
