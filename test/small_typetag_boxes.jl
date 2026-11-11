using Enzyme, Test
using Enzyme.EnzymeRules

# From Julia 1.10 some types have a small type tag (`enum jl_small_typeof_tags` in
# julia.h): `Bool`, `Char`, the `Int` and `UInt` types. An object of such a type carries
# `tag << 4` in its header word, not a pointer to its datatype, and the garbage collector
# aborts the process ("GC error (probable corruption)", printing the type) when it marks a
# pointer-tagged object of a small-tag type. Enzyme boxes such values itself: the primal a
# derivative returns when its entry returns a small `Union` of isbits members and the
# primal is requested (every call Enzyme dispatches at run time returns its primal), and a
# custom rule's union tape. Those boxes carried the datatype pointer, and a `Bool` got a
# box of its own where Julia only ever uses `jl_true` and `jl_false`, which it compares by
# pointer (snag
# ordinary-enzyme-050cf58b: a ReactiveKernels plate cell dispatching
# `_cell_array_operand`, which returns `Union{Missing, Bool}`, at run time under ordinary
# Reverse aborted the process at the next collection).

# The header word of the box an `Any` holds, with the GC bits masked off.
header(@nospecialize(obj)) = unsafe_load(Ptr{UInt}(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)), 0) & ~UInt(0x0f)

@testset "small type tags" begin
    @test Enzyme.Compiler.small_typetag(Bool) == header(true)
    @test Enzyme.Compiler.small_typetag(Int64) == header(Int64(1) << 40)
    @test Enzyme.Compiler.small_typetag(UInt8) == header(UInt8(7))
    @test Enzyme.Compiler.small_typetag(Char) == header('p')
    @test Enzyme.Compiler.small_typetag(Float64) === nothing
    @test header(2.5) == UInt(pointer_from_objref(Float64))
    @test Enzyme.Compiler.small_typetag(Base.RefValue{Float64}) === nothing
    @test Enzyme.has_dogfood_repair(:small_typetag_boxes)
end

@noinline missing_or_positive(x::Float64, flag::Bool) = flag ? missing : x > 0.0
@noinline nothing_or_rounded(x::Float64, flag::Bool) = flag ? nothing : Int(round(x))
@noinline nothing_or_sign(x::Float64, flag::Bool) = flag ? nothing : (x > 0.0 ? 'p' : 'n')
@noinline missing_or_double(x::Float64, flag::Bool) = flag ? missing : 2.0 * x

# Keeps every primal and tape alive across the collections below: a wrongly tagged box
# aborts the process only when a collection marks it.
const kept = Any[]
function collect_all()
    for _ in 1:3
        GC.gc(true)
    end
    return nothing
end

@testset "the primal of a small union return is tagged as Julia tags it" begin
    for (f, x, flag, want) in (
            (missing_or_positive, 2.0, false, true),
            (missing_or_positive, 2.0, true, missing),
            (nothing_or_rounded, 2.0, false, 2),
            (nothing_or_rounded, 2.0, true, nothing),
            (nothing_or_sign, 2.0, false, 'p'),
            (nothing_or_sign, -2.0, false, 'n'),
            (missing_or_double, 2.0, false, 4.0),
        )
        fw = autodiff(ForwardWithPrimal, Const(f), Const, Duplicated(x, 1.0), Const(flag))
        @test fw[1] === want
        @test header(fw[1]) == header(want)
        rv = autodiff(ReverseWithPrimal, Const(f), Const, Active(x), Const(flag))
        @test rv[2] === want
        @test header(rv[2]) == header(want)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Const, Active{Float64}, Const{Bool}
        )
        tape, prim, shad = fwd(Const(f), Active(x), Const(flag))
        @test prim === want
        @test header(prim) == header(want)
        push!(kept, fw, rv, (tape, prim, shad))
        collect_all()
    end
end

use_bool(r, x) = r === missing ? 0.0 : (r ? x^2 : -x^2)
use_int(r, x) = r === nothing ? 0.0 : r * x^2
use_char(r, x) = r === nothing ? 0.0 : (r == 'p' ? x^2 : -x^2)
use_float(r, x) = r === missing ? 0.0 : r * x
dispatched_bool(x, flag) = use_bool(Base.inferencebarrier(missing_or_positive)(x[1], flag), x[1])
dispatched_int(x, flag) = use_int(Base.inferencebarrier(nothing_or_rounded)(x[1], flag), x[1])
dispatched_char(x, flag) = use_char(Base.inferencebarrier(nothing_or_sign)(x[1], flag), x[1])
dispatched_float(x, flag) = use_float(Base.inferencebarrier(missing_or_double)(x[1], flag), x[1])

@testset "a call dispatched at run time returning a small union" begin
    for (f, flag, want) in (
            (dispatched_bool, false, 4.0),
            (dispatched_bool, true, 0.0),
            (dispatched_int, false, 8.0),
            (dispatched_int, true, 0.0),
            (dispatched_char, false, 4.0),
            (dispatched_char, true, 0.0),
            (dispatched_float, false, 8.0),
            (dispatched_float, true, 0.0),
        )
        for mode in (Reverse, set_runtime_activity(Reverse))
            x = [2.0]
            dx = zeros(1)
            r = autodiff(mode, f, Active, Duplicated(x, dx), Const(flag))
            @test dx[1] ≈ want
            push!(kept, r)
            collect_all()
        end
        x = [2.0]
        r = autodiff(ForwardWithPrimal, f, Duplicated(x, [1.0]), Const(flag))
        @test r[1] ≈ want
        push!(kept, r)
        collect_all()
    end
end

# A custom rule whose tape is a small union held inline in a concrete `AugmentedReturn`:
# the tape member is boxed when the tape is stored.
@noinline scaled_inline(x::Float64, flag::Bool) = flag ? 3.0 * x : 2.0 * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(scaled_inline)}, ::Type{<:Active},
        x::Active{Float64}, flag::Const{Bool}
    )
    tape = flag.val ? nothing : Int64(2)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val, flag.val) : nothing
    return EnzymeRules.AugmentedReturn{typeof(primal), Nothing, Union{Nothing, Int64}}(primal, nothing, tape)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(scaled_inline)}, dret::Active, tape,
        x::Active{Float64}, flag::Const{Bool}
    )
    return ((tape === nothing ? 3.0 : Float64(tape)) * dret.val, nothing)
end

# A custom rule whose tape type depends on the branch taken: the rule returns a small
# union of `AugmentedReturn`s, and the tape member is boxed under a selected tag.
@noinline scaled_split(x::Float64, flag::Bool) = flag ? 3.0 * x : 2.0 * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(scaled_split)}, ::Type{<:Active},
        x::Active{Float64}, flag::Const{Bool}
    )
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val, flag.val) : nothing
    if flag.val
        return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, Int64(2))
    end
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(scaled_split)}, dret::Active, tape,
        x::Active{Float64}, flag::Const{Bool}
    )
    return ((tape === nothing ? 3.0 : Float64(tape)) * dret.val, nothing)
end

rule_inline_loss(x, flag) = scaled_inline(x[1], flag) * x[1]
rule_split_loss(x, flag) = scaled_split(x[1], flag) * x[1]

@testset "a custom rule's small union tape" begin
    for (f, flag, want) in (
            (rule_inline_loss, false, 8.0),
            (rule_inline_loss, true, 12.0),
            (rule_split_loss, false, 8.0),
            (rule_split_loss, true, 12.0),
        )
        x = [2.0]
        dx = zeros(1)
        r = autodiff(Reverse, f, Active, Duplicated(x, dx), Const(flag))
        @test dx[1] ≈ want
        push!(kept, r)
        collect_all()
    end
end
