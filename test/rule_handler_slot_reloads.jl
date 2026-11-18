module RuleHandlerSlotReloads

using Enzyme, LLVM, Test
using Enzyme.EnzymeRules

# A custom rule whose methods hold an exception handler, differentiated again in Forward
# mode (a Hessian-vector product). Julia zero-initializes the roots slot of an argument
# that a handler may read. Once SROA has split the slot, that is `store i64 0` into a
# slot the rule reloads as a tracked pointer, which no cast joins, so the slot stays on
# the stack. libEnzyme's preprocessing rewrites its loads without the slot-type tag, and
# `abs_typeof` then had no type for them. The mixed-activity handler could not see that
# the reloaded index vector of a `view`, a `Vector{Int}`, holds no derivative: on Julia
# 1.12 the product raised `EnzymeRuntimeActivityError` where the same rule without the
# handler differentiated.
struct HandlerRule{A, R} end

function (::HandlerRule)(cache, scale, values)
    total = 0.0
    for value in values
        total += sin(scale * value)
    end
    return total
end

function handler_sweep!(tape, scale, values)
    total = 0.0
    for i in eachindex(values)
        value = values[i]
        total += sin(scale * value)
        tape[values.indices[1][i]] += scale * cos(scale * value)
    end
    return total
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1}, ::Const{HandlerRule{A, R}}, ::Type{<:Active},
        cache::Annotation, scale::Const, values::Duplicated
    ) where {A, R}
    tape = zeros(length(parent(values.val)))
    primal = if A
        try
            handler_sweep!(tape, scale.val, values.val)
        catch
            rethrow()
        end
    else
        handler_sweep!(tape, scale.val, values.val)
    end
    return AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, tape)
end

function handler_accumulate!(values, tape, seed)
    shadow = parent(values.dval)
    for i in eachindex(shadow)
        shadow[i] += seed * tape[i]
    end
    return nothing
end

function EnzymeRules.reverse(
        ::RevConfigWidth{1}, ::Const{HandlerRule{A, R}}, dret::Active, tape,
        cache::Annotation, scale::Const, values::Duplicated
    ) where {A, R}
    if R
        try
            handler_accumulate!(values, tape, dret.val)
        finally
            tape === first(cache.val) && (last(cache.val)[] = false)
        end
    else
        handler_accumulate!(values, tape, dret.val)
        tape === first(cache.val) && (last(cache.val)[] = false)
    end
    return (nothing, nothing, nothing)
end

handler_total(x, groups, cache, rule) = rule(cache, 0.6, view(x, groups))

function handler_gradient(x, groups, cache, rule)
    dx = zero(x)
    autodiff(
        Reverse, handler_total, Active, Duplicated(x, dx), Const(groups),
        Duplicated(cache, Enzyme.make_zero(cache)), Const(rule)
    )
    return dx
end

@testset "Hessian-vector product through a rule with an exception handler" begin
    @testset "augmented `try`: $A, reverse `try`: $R" for (A, R) in
        ((false, false), (true, false), (false, true), (true, true))
        rule = HandlerRule{A, R}()
        x = [0.3cos(i) for i in 1:30]
        groups = [mod1(3i, 29) for i in 1:200]
        saved = copy(groups)
        cache = (zero(x), Ref(false))
        direction = fill(0.3, length(x))
        gradient = zero(x)
        product = zero(x)
        for i in groups
            gradient[i] += 0.6cos(0.6x[i])
            product[i] += -0.36sin(0.6x[i]) * direction[i]
        end
        @test handler_gradient(x, groups, cache, rule) ≈ gradient
        @test autodiff(
            Forward, handler_gradient, Duplicated, Duplicated(x, direction), Const(groups),
            Const(cache), Const(rule)
        )[1] ≈ product
        @test groups == saved
        @test !cache[2][]
    end
end

# The type `abs_typeof` reads off a slot for a reload that carries no tag, on functions
# written out here: each keeps the slot the way the zero store keeps it in the rule above.
function slot_reload_ir(typed::Bool)
    obj = typed ? "{} addrspace(10)*" : "ptr addrspace(10)"
    slot = typed ? "{} addrspace(10)**" : "ptr"
    int = typed ? "i64*" : "ptr"
    nocapture = LLVM.version() >= v"21" ? "captures(none)" : "nocapture"
    # The zero store, through the cast that typed pointers need.
    clear(name) = typed ?
        "%$(name)z = bitcast $slot %$name to $int\n  store i64 0, $int %$(name)z, align 8" :
        "store i64 0, ptr %$name, align 8"
    return """
    declare $obj @ijl_box_int64(i64)
    declare $obj @ijl_box_float32(float)
    declare void @opaque($slot)
    declare void @inspect($slot readonly $nocapture)

    define $obj @typed(i64 %x) {
    top:
      %a = alloca $obj, align 8
      $(clear("a"))
      %box = call $obj @ijl_box_int64(i64 %x)
      store $obj %box, $slot %a, align 8
      call void @inspect($slot %a)
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }

    define $obj @chained(i64 %x) {
    top:
      %a = alloca $obj, align 8
      %b = alloca $obj, align 8
      $(clear("a"))
      $(clear("b"))
      %box = call $obj @ijl_box_int64(i64 %x)
      store $obj %box, $slot %a, align 8
      %first = load $obj, $slot %a, align 8
      store $obj %first, $slot %b, align 8
      %reload = load $obj, $slot %b, align 8
      ret $obj %reload
    }

    define $obj @circular(i1 %again) {
    top:
      %a = alloca $obj, align 8
      $(clear("a"))
      br label %loop
    loop:
      %kept = load $obj, $slot %a, align 8
      store $obj %kept, $slot %a, align 8
      br i1 %again, label %loop, label %done
    done:
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }

    define $obj @merged(i64 %x, i1 %which) {
    top:
      %a = alloca $obj, align 8
      $(clear("a"))
      %box = call $obj @ijl_box_int64(i64 %x)
      store $obj %box, $slot %a, align 8
      br i1 %which, label %left, label %right
    left:
      %kept = load $obj, $slot %a, align 8
      br label %done
    right:
      %other = call $obj @ijl_box_int64(i64 %x)
      br label %done
    done:
      %reload = phi $obj [ %kept, %left ], [ %other, %right ]
      ret $obj %reload
    }

    define $obj @mutual(i1 %again) {
    top:
      %a = alloca $obj, align 8
      %b = alloca $obj, align 8
      $(clear("a"))
      $(clear("b"))
      br label %loop
    loop:
      %fromb = load $obj, $slot %b, align 8
      store $obj %fromb, $slot %a, align 8
      %froma = load $obj, $slot %a, align 8
      store $obj %froma, $slot %b, align 8
      br i1 %again, label %loop, label %done
    done:
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }

    define $obj @declared(i64 %x) {
    top:
      %a = alloca $obj, align 8
      $(clear("a"))
      %box = call $obj @ijl_box_int64(i64 %x)
      store $obj %box, $slot %a, align 8
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }

    define $obj @mixed(i64 %x, float %y, i1 %which) {
    top:
      %a = alloca $obj, align 8
      %int = call $obj @ijl_box_int64(i64 %x)
      %float = call $obj @ijl_box_float32(float %y)
      br i1 %which, label %left, label %right
    left:
      store $obj %int, $slot %a, align 8
      br label %done
    right:
      store $obj %float, $slot %a, align 8
      br label %done
    done:
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }

    define $obj @escaped(i64 %x) {
    top:
      %a = alloca $obj, align 8
      %box = call $obj @ijl_box_int64(i64 %x)
      store $obj %box, $slot %a, align 8
      call void @opaque($slot %a)
      %reload = load $obj, $slot %a, align 8
      ret $obj %reload
    }
    """
end

# `typed` runs the pass that tags the reloads of a module and then drops every tag, as
# libEnzyme's rewrite of the loads does. Returns what `abs_typeof` makes of each
# function's `%reload`, and how many loads the pass had tagged.
function slot_reload_types(typed::Bool)
    return LLVM.Context() do ctx
        mod = parse(LLVM.Module, slot_reload_ir(LLVM.supports_typed_pointers(ctx)))
        # The slot of `declared` is declared to hold a `Ref{Any}`, as a slot that libEnzyme
        # moves to the heap is declared an array of `Any`.
        declared = first((first((mod.functions["declared"]).blocks)).instructions)
        any_ref = string(convert(UInt, Enzyme.Compiler.unsafe_to_pointer(Base.RefValue{Any})))
        declared.metadata["enzymejl_allocart"] = LLVM.MDNode(LLVM.Metadata[LLVM.MDString(any_ref)])
        tagged = typed ? Enzyme.Compiler.type_slot_field_loads!(mod, nothing) : 0
        for f in mod.functions, bb in f.blocks, inst in bb.instructions
            isa(inst, LLVM.LoadInst) || continue
            delete!(inst.metadata, Enzyme.Compiler.SLOT_FIELD_TYPE)
        end
        found = Dict{String, Any}()
        for f in mod.functions, bb in f.blocks, inst in bb.instructions
            inst.name == "reload" || continue
            legal, T, byref = Enzyme.Compiler.abs_typeof(inst, nothing)
            found[f.name] = legal ? (T, byref) : nothing
        end
        found, tagged
    end
end

@testset "The type of an untagged reload, read off its stack slot" begin
    found, tagged = slot_reload_types(true)
    boxed = (Int64, Enzyme.Compiler.GPUCompiler.MUT_REF)
    @testset "the pass tags the loads it can type" begin
        # `%reload` of `typed`, of `declared`, both loads of `chained`, `%kept` of `merged`.
        @test tagged == 5
    end
    @testset "a zero store beside the stored value" begin
        @test found["typed"] == boxed
    end
    @testset "a slot that holds the reload of another slot" begin
        @test found["chained"] == boxed
    end
    @testset "a slot declared to hold any object" begin
        @test found["declared"] == boxed
    end
    @testset "a reload merged with another value of its type" begin
        @test found["merged"] == boxed
    end
    @testset "a slot that only holds its own reload" begin
        @test found["circular"] === nothing
    end
    @testset "two slots that only hold each other's reloads" begin
        @test found["mutual"] === nothing
    end
    @testset "stored values of two types" begin
        @test found["mixed"] === nothing
    end
    @testset "a call that may write the slot" begin
        @test found["escaped"] === nothing
    end
    @testset "a module whose reloads the pass has not typed" begin
        untyped, _ = slot_reload_types(false)
        # For the slot of `declared` only its declared type is left, not the stored one.
        @test untyped["declared"] != boxed
        @test all(isnothing, (v for (k, v) in untyped if k != "declared"))
    end
end

end
