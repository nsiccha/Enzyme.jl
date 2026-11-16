"""
    EmitTypeNames[] = true

Also write the printed Julia type next to each type Enzyme records in the IR (the
`enzymejl_parmtype_str` attribute, and the `enzymejl_source_type_<T>` and
`enzymejl_allocart_name` metadata). They help reading IR dumps.
"""
const EmitTypeNames = Ref(false)

@enum(
    AllocFnKindEnum,
    AFKE_Unknown = 0,
    AFKE_Alloc = 1,
    AFKE_Realloc = 2,
    AFKE_Free = 4,
    AFKE_Uninitialized = 8,
    AFKE_Zeroed = 16,
    AFKE_Aligned = 32,
)

struct AllocFnKind
    data::UInt32
    AllocFnKind() = new(0)
    AllocFnKind(x::UInt32) = new(x)
    AllocFnKind(x::AllocFnKindEnum) = new(UInt32(x))
end

function Base.:|(lhs::AllocFnKind, rhs::AllocFnKind)
    return AllocFnKind(UInt32(lhs.data) | UInt32(rhs.data))
end

# Memory effects, as encoded in the `memory` attribute on LLVM 16+, are handled with
# LLVM.jl's `MemoryEffects`, which knows the memory locations of each LLVM version.
set_readonly(effects::LLVM.MemoryEffects) = effects & LLVM.MemoryEffects(:read)

is_readonly(effects::LLVM.MemoryEffects) = effects.access in (:none, :read)
is_readnone(effects::LLVM.MemoryEffects) = effects.access == :none
is_writeonly(effects::LLVM.MemoryEffects) = effects.access in (:none, :write)

Base.@assume_effects :removable :foldable :nothrow is_noreturn(f::LLVM.Function)::Bool =
    haskey(f.function_attributes, :noreturn)

Base.@assume_effects :removable :foldable :nothrow is_nounwind(f::LLVM.Function)::Bool =
    haskey(f.function_attributes, :nounwind)

"""
    clear_nounwind!(f::LLVM.Function)

Remove `nounwind` from `f`, from every call of `f`, and from each calling
function in turn, so that a throw inserted into `f` may unwind through its
callers.

Enzyme's preprocessing marks every call it differentiates `willreturn`, and
inlining a `nounwind` function marks the calls it inlines `nounwind`. A
`noreturn` throw that is also `nounwind` and `willreturn` can neither return,
unwind nor run forever, so LLVM treats it as unreachable and deletes it,
together with every path that leads to it.
"""
function clear_nounwind!(f::LLVM.Function)
    todo = LLVM.Function[f]
    done = Set{LLVM.Function}()
    while !isempty(todo)
        cur = pop!(todo)
        if cur in done
            continue
        end
        push!(done, cur)
        delete!(cur.function_attributes, :nounwind)
        for u in cur.uses
            call = u.user
            if !isa(call, LLVM.CallBase) || call.called_operand != cur
                continue
            end
            delete!(call.function_attributes, :nounwind)
            push!(todo, call.parent.parent)
        end
    end
    return nothing
end

"""
    is_float_data(ty::LLVM.LLVMType)::Bool

Whether a value of type `ty` is floating-point data (a float or a vector of
floats), which cannot refer to memory.
"""
is_float_data(::LLVM.LLVMType) = false
is_float_data(::LLVM.FloatingPointType) = true
is_float_data(ty::LLVM.VectorType) = is_float_data(ty.element_type)

"""
    is_inactive_operand(call, fn, val, world)::Bool

Whether `call` passes `val` only by reference (`BITS_REF` or `MUT_REF`) to
parameters of the Julia function `fn` whose declared type (`enzymejl_parmtype`)
cannot hold derivative data that a write could change (`active_reg` is
`ActiveState` or `AnyState`). Nothing `fn` does through `val` then writes
derivative data, whether or not its body can be inspected.
"""
function is_inactive_operand(call::LLVM.CallInst, fn::LLVM.Function, @nospecialize(val::LLVM.Value), world::UInt)::Bool
    found = false
    nparms = length(fn.parameters)
    for (i, op) in enumerate(call.arguments)
        op == val || continue
        i <= nparms || return false
        typ, byref = enzyme_extract_parm_type(fn, i, false)
        (typ === nothing || byref === nothing) && return false
        (byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF) || return false
        reg = active_reg(typ, world)
        (reg == ActiveState || reg == AnyState) || return false
        found = true
    end
    return found
end

"""
    readonly_checked_params(primalf, TT, err_if_written, world)::Vector{LLVM.Value}

The parameters of `primalf` whose data the readonly check follows. `err_if_written`
is `true` for the function argument alone (`err_if_func_written`), or a tuple with
one entry per element of `TT.parameters`, the function first, as the rules of a
dynamic call request it (`src/rules/jitrules.jl`). Only arguments whose type can
hold derivative data (`DupState` / `MixedState`) are followed.
"""
function readonly_checked_params(primalf::LLVM.Function, @nospecialize(TT::Type), err_if_written::Union{Bool, Tuple}, world::UInt)::Vector{LLVM.Value}
    checked = LLVM.Value[]
    parms = collect(primalf.parameters)
    # Lowered parameters: `swiftself`, then one per argument that is not a ghost.
    idx = Int(has_swiftself(primalf))
    for (i, T) in enumerate(TT.parameters)
        source_typ = eltype(T)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        idx += 1
        check = err_if_written isa Bool ? (i == 1 && err_if_written) : err_if_written[i]
        check || continue
        reg = active_reg(source_typ, world)
        if reg == DupState || reg == MixedState
            push!(checked, parms[idx])
        end
    end
    return checked
end

"""
    is_fresh_object(val::LLVM.Value)::Bool

Whether `val` is an object this function allocates (`julia.gc_alloc_obj`), such as
the box Julia makes of a value it passes to a dynamic call.
"""
function is_fresh_object(@nospecialize(val::LLVM.Value))::Bool
    isa(val, LLVM.CallInst) || return false
    fn = val.called_operand
    return isa(fn, LLVM.Function) && fn.name == "julia.gc_alloc_obj"
end

"""
    is_nonwriting_runtime_call(call::LLVM.CallInst)::Bool

Whether `call` reads its arguments, or builds a new object from them, but writes
no memory they refer to: a `julia.call` of a Julia builtin such as `tuple`,
`getfield` or `isa` (builtins take and return boxed values, as the code around a
dynamic call does), or `jl_array_copy` (`copy` of an `Array` before Julia 1.11).
"""
function is_nonwriting_runtime_call(call::LLVM.CallInst)::Bool
    fn = call.called_operand
    isa(fn, LLVM.Function) || return false
    builtin = fn.name == "julia.call"
    if builtin
        fn = call.operands[1]
        isa(fn, LLVM.Function) || return false
    end
    nm = fn.name
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    builtin || return nm == "jl_array_copy"
    return nm in (
        "jl_f_tuple",
        "jl_f_getfield",
        "jl_f__svec_ref",
        "jl_f_typeassert",
        "jl_f_isa",
        "jl_f_is",
        "jl_f_typeof",
        "jl_f_sizeof",
        "jl_f_nfields",
        "jl_f_isdefined",
        "jl_f_ifelse",
    )
end

"""
    local_memory_base(ptr::LLVM.Value)

The memory `ptr` points into, through GEPs and casts, when it belongs to the
function: a stack slot (`alloca`) or an object it allocates (`is_fresh_object`).
`nothing` otherwise.
"""
function local_memory_base(@nospecialize(ptr::LLVM.Value))
    base = ptr
    while isa(base, LLVM.BitCastInst) ||
            isa(base, LLVM.AddrSpaceCastInst) ||
            isa(base, LLVM.GetElementPtrInst)
        base = base.operands[1]
    end
    return (isa(base, LLVM.AllocaInst) || is_fresh_object(base)) ? base : nothing
end

"""
    is_pointerfree_copy(call::LLVM.CallInst, enzyme_ctx)::Bool

Whether a `memcpy` / `memmove` copies bytes of a Julia type that holds no
references, known from the type either pointer points to. Such bytes cannot
refer to memory.
"""
function is_pointerfree_copy(call::LLVM.CallInst, enzyme_ctx::EnzymeContext)::Bool
    for ptr in (call.operands[1], call.operands[2])
        base, _ = get_base_and_offset(ptr)
        legal, jTy, byref = abs_typeof(base, enzyme_ctx)
        legal || continue
        if byref == GPUCompiler.BITS_VALUE && jTy <: Ptr
            jTy = eltype(jTy)
        end
        if isa(jTy, DataType) && Base.isconcretetype(jTy) && Base.datatype_pointerfree(jTy)
            return true
        end
    end
    return false
end

"""
    is_pointerfree_runtime_copy(call::LLVM.CallInst, val, enzyme_ctx)::Bool

Whether `call` is a Julia runtime copy of elements that hold no references
(`isbitstype`, or a union of such types), which reads `val` and fills fresh memory
with bytes: `jl_array_copy` (`copy` of an `Array` before Julia 1.11), or
`jl_genericmemory_copyto` (`unsafe_copyto!` of `Memory` on Julia 1.11+, for an
element type Julia does not copy with a `memmove`, such as
`Union{Missing, Float64}`) with `val` among its source operands only. The element
type comes from the Julia type of the array or memory. Such bytes cannot refer to
memory, so nothing that writes the copy, or passes it on, reaches memory of the
argument; `is_pointerfree_copy` says the same of a `memcpy` / `memmove`.
"""
function is_pointerfree_runtime_copy(call::LLVM.CallInst, @nospecialize(val::LLVM.Value), enzyme_ctx::EnzymeContext)::Bool
    fn = call.called_operand
    isa(fn, LLVM.Function) || return false
    nm = fn.name
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    if nm == "jl_array_copy"
        legal, jTy, _ = abs_typeof(call, enzyme_ctx)
        (legal && isa(jTy, DataType) && jTy <: Array) || return false
    elseif nm == "jl_genericmemory_copyto"
        # (dest memory, dest data, src memory, src data, length)
        ops = call.arguments
        length(ops) == 5 || return false
        (val == ops[1] || val == ops[2]) && return false
        (val == ops[3] || val == ops[4]) || return false
        legal, jTy, _ = abs_typeof(ops[3], enzyme_ctx)
        @static if VERSION >= v"1.11-"
            (legal && isa(jTy, DataType) && jTy <: GenericMemory) || return false
        else
            return false
        end
    else
        return false
    end
    T = eltype(jTy)
    return isbitstype(T) || Base.isbitsunion(T)
end

"""
    is_throw_runtime_call(call::LLVM.CallInst)::Bool

Whether `call` is a Julia runtime function that throws what it receives (`jl_throw`,
`jl_rethrow_other`) or a `BoundsError` holding it (`jl_bounds_error_int`, …). It
writes nothing it receives, but a `catch` can read what it throws.
"""
function is_throw_runtime_call(call::LLVM.CallInst)::Bool
    fn = call.called_operand
    isa(fn, LLVM.Function) || return false
    nm = fn.name
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    return nm in (
        "jl_throw",
        "jl_rethrow_other",
        "jl_bounds_error",
        "jl_bounds_error_v",
        "jl_bounds_error_int",
        "jl_bounds_error_ints",
        "jl_bounds_error_tuple_int",
        "jl_bounds_error_unboxed_int",
    )
end

"""
    caught_exception_reads(mod::LLVM.Module)::Vector{LLVM.Value}

The calls in `mod` that read a caught exception: `jl_current_exception` (`catch e`)
and `jl_get_excstack` (`current_exceptions()`).
"""
function caught_exception_reads(mod::LLVM.Module)::Vector{LLVM.Value}
    reads = LLVM.Value[]
    fns = mod.functions
    for nm in ("jl_current_exception", "ijl_current_exception", "jl_get_excstack", "ijl_get_excstack")
        haskey(fns, nm) || continue
        for u in (fns[nm]).uses
            call = u.user
            if isa(call, LLVM.CallInst)
                push!(reads, call)
            end
        end
    end
    return reads
end

"""
    ReadonlyWalkState

What the walks of one readonly check share (`readonly_walk!`).

- `shared`: per callee called from several sites and parameter positions that
  receive the data, whether a walk of its body found a possible write and whether
  the data reached its `ret`.
- `caught`: the calls that read a caught exception, not yet followed.
- `thrown`: whether data may have been thrown, by a runtime call or by the method a
  dynamic call reaches. A `catch` can then read it, so `caught` is followed too.
"""
struct ReadonlyWalkState
    shared::Dict{Tuple{LLVM.Function, Vector{Int}}, Tuple{Bool, Bool}}
    caught::Vector{LLVM.Value}
    thrown::Base.RefValue{Bool}
end
ReadonlyWalkState(mod::LLVM.Module) = ReadonlyWalkState(
    Dict{Tuple{LLVM.Function, Vector{Int}}, Tuple{Bool, Bool}}(), caught_exception_reads(mod), Ref(false),
)

const READONLY_OPERANDS_MD = "enzymejl_readonly_operands"

"""
    dynamic_call_operands(call::LLVM.CallInst)

For a `julia.call` / `julia.call2` that dispatches at run time (`jl_apply_generic`,
`jl_invoke`, `invokelatest`, `invoke_in_world`, `_apply_iterate`), the index in
`arg_operands_view(call)` of the function it calls, the first of the operands the
dynamic-call rules (`generic_setup`, `src/rules/jitrules.jl`) pass on, and whether
the arguments after it are splatted (`_apply_iterate`). `nothing` for any other
call. The indices follow the handlers in `src/rules/llvmrules.jl`.
"""
function dynamic_call_operands(call::LLVM.CallInst)
    jlcall = call.called_operand
    isa(jlcall, LLVM.Function) || return nothing
    jlname = jlcall.name
    (jlname == "julia.call" || jlname == "julia.call2") || return nothing
    fn = call.operands[1]
    isa(fn, LLVM.Function) || return nothing
    nm = fn.name
    if jlname == "julia.call2"
        return nm in ("ijl_invoke", "jl_invoke") ? (3, false) : nothing
    end
    if nm in ("ijl_apply_generic", "jl_apply_generic")
        return (2, false)
    elseif nm in (
            "ijl_f__apply_latest",
            "ijl_f__call_latest",
            "ijl_f_invokelatest",
            "jl_f__apply_latest",
            "jl_f__call_latest",
            "jl_f_invokelatest",
        )
        return (3, false)
    elseif nm in ("ijl_f_invoke_in_world", "jl_f_invoke_in_world", "ijl_f__call_in_world", "jl_f__call_in_world")
        return (4, false)
    elseif nm in ("ijl_f__apply_iterate", "jl_f__apply_iterate")
        return (4, true)
    end
    return nothing
end

"""
    readonly_operands(call::LLVM.Instruction)::Vector{Int}

The operands of a dynamic call whose data the readonly check left to the method
the call reaches (`defer_readonly_operand!`), numbered as the dynamic-call rules
see them: 1 is the called function, 2 its first argument.
"""
function readonly_operands(call::LLVM.Instruction)::Vector{Int}
    haskey(call.metadata, READONLY_OPERANDS_MD) || return Int[]
    mds = (call.metadata[READONLY_OPERANDS_MD]).operands[1]::MDString
    return Int[parse(Int, s) for s in split(Base.convert(String, mds), ",")]
end

"""
    defer_readonly_operand!(call::LLVM.CallInst, val::LLVM.Value)::Bool

Whether the readonly check can leave `val`, an operand of `call`, to the method a
dynamic call reaches (`dynamic_call_operands`). It can when `val` is the called
function or, unless the arguments are splatted, one of its arguments. The
positions are then recorded on `call` (`readonly_operands`), and the rules of
`src/rules/jitrules.jl` compile the derivative of that method with the check on
them. What the call returns is followed like the other values the walk meets.
"""
function defer_readonly_operand!(call::LLVM.CallInst, @nospecialize(val::LLVM.Value))::Bool
    dyn = dynamic_call_operands(call)
    dyn === nothing && return false
    first, splat = dyn
    positions = readonly_operands(call)
    for (i, op) in enumerate(call.arguments)
        op == val || continue
        (i < first || (splat && i != first)) && return false
        push!(positions, i - first + 1)
    end
    call.metadata[READONLY_OPERANDS_MD] =
        MDNode(LLVM.Metadata[MDString(join(sort!(unique!(positions)), ","))])
    return true
end

"""
    is_readonly(attr::LLVM.Attribute)::Bool

Whether `attr` on its own establishes that the function or argument position it
is attached to is only read from. That is `readonly` / `readnone`, and on LLVM
16+ a `memory` effect whose modref is read-only.
"""
Base.@assume_effects :removable :foldable :nothrow function is_readonly(attr::LLVM.Attribute)::Bool
    if attr.kind == :readonly
        return true
    end
    if attr.kind == :readnone
        return true
    end
    if attr.kind == :memory
        if is_readonly(LLVM.MemoryEffects(attr))
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_readonly(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(f.function_attributes)
        if is_readonly(attr)
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_readnone(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    return is_readnone(LLVM.MemoryEffects(f.memory_effects))
end

Base.@assume_effects :removable :foldable :nothrow function is_writeonly(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    return is_writeonly(LLVM.MemoryEffects(f.memory_effects))
end

function set_readonly!(fn::LLVM.Function)
    # without a `memory` attribute (or on LLVM 15, one of the attributes it replaced),
    # a function may access any memory
    old = LLVM.MemoryEffects(fn.memory_effects)
    eff = set_readonly(old)
    fn.memory_effects = eff
    return old != eff
end

# declares a function for `get_function!`
struct FunctionDeclaration <: Function
    mod::LLVM.Module
    name::String
    FT::LLVM.FunctionType
    attrs::Vector{LLVM.Attribute}
    memory_effects::Union{Nothing, LLVM.MemoryEffects}
end

function (decl::FunctionDeclaration)()
    F = LLVM.Function(decl.mod, decl.name, decl.FT)
    append!(F.function_attributes, decl.attrs)
    if decl.memory_effects !== nothing
        F.memory_effects = decl.memory_effects
    end
    return F
end

function get_function!(
        mod::LLVM.Module,
        name::String,
        FT::LLVM.FunctionType,
        attrs::Vector{LLVM.Attribute} = LLVM.Attribute[];
        memory_effects::Union{Nothing, LLVM.MemoryEffects} = nothing,
    )
    # `get!` throws if another kind of global value already uses the name, where creating
    # the function would give it a different name
    F = get!(FunctionDeclaration(mod, name, FT, attrs, memory_effects), mod.functions, name)
    PT = LLVM.PointerType(FT)
    if F.value_type != PT
        F = LLVM.const_pointercast(F, PT)
    end
    return F, FT
end

function get_function!(@nospecialize(builderF), mod::LLVM.Module, name::String, attrs::Vector{LLVM.Attribute} = LLVM.Attribute[]; memory_effects::Union{Nothing, LLVM.MemoryEffects} = nothing)
    return get_function!(mod, name, builderF(), attrs; memory_effects)
end

T_ppjlvalue() = LLVM.PointerType(LLVM.PointerType(LLVM.StructType(LLVMType[])))

function declare_pgcstack!(mod::LLVM.Module)
    return get_function!(
        mod,
        "julia.get_pgcstack",
        LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue())),
        LLVM.Attribute[StringAttribute("enzyme_inactive"), StringAttribute("enzyme_no_escaping_allocation")]
    )
end

function emit_pgcstack(B::LLVM.IRBuilder, name::String = "")
    curent_bb = B.insert_block
    fn = curent_bb.parent
    mod = fn.parent
    func, fty = declare_pgcstack!(mod)
    return call!(B, fty, func, LLVM.Value[], name)
end

function get_pgcstack(func::LLVM.Function)
    entry_bb = first(func.blocks)
    mod = func.parent
    pgcstack_func, _ = declare_pgcstack!(mod)

    # A @cfunction wrapper fetches its pgcstack with julia.get_pgcstack_or_new,
    # which adopts the calling thread when it is not a Julia thread yet. It has
    # to stay the first getter of the entry block: FinalLowerGC roots the GC
    # frame in whichever getter comes first, and a plain getter placed ahead of
    # the adopting one dereferences a NULL pgcstack on a foreign thread.
    if haskey(mod.functions, "julia.get_pgcstack_or_new")
        or_new = mod.functions["julia.get_pgcstack_or_new"]
        for I in entry_bb.instructions
            if I isa LLVM.CallInst && I.called_operand == or_new
                return I
            end
        end
    end

    for I in entry_bb.instructions
        if I isa LLVM.CallInst && I.called_operand == pgcstack_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func::LLVM.Function, @nospecialize(PB::Union{Nothing, LLVM.IRBuilder}) = nothing)
    for i in 1:length(func.parameters)
        for attr in collect(func.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return func.parameters[i]
                end
            end
        end
    end

    pgs = get_pgcstack(func)
    return if pgs isa Nothing
        entry_bb = first(func.blocks)
        if PB !== nothing &&
                (PB.insert_block.name == "allocsForInversion" || PB.insert_block == entry_bb)
            # emit it with the caller's builder, which we don't own
            emit_pgcstack(PB, "newly_emitted_pgc_stack")
        else
            @dispose B = IRBuilder() begin
                position!(B, LLVM.at_begin(entry_bb))
                emit_pgcstack(B, "newly_emitted_pgc_stack")
            end
        end
    else
        entry_bb = first(func.blocks)
        fst = first(entry_bb.instructions)
        if fst != pgs
            API.moveBefore(pgs, fst, PB isa Nothing ? C_NULL : PB.ref)
        end
        pgs
    end
end

# `offsetof(jl_task_t, gcstack)`, which is fixed for a build of Julia. Declaring it foldable
# lets inference turn the load into a constant.
Base.@assume_effects :foldable :nothrow task_gcstack_offset() = Int(unsafe_load(cglobal(:jl_task_gcstack_offset, Cint)))

"""
    current_pgcstack() -> Ptr{Cvoid}

Give the pgcstack of the running task, for an llvmcall that takes it as an argument (see
[`use_gcstack_arg!`](@ref)).

Julia computes `current_task()` from the pgcstack of the function that this is inlined into
(on 1.13 that is the `"gcstack"` argument of the function). So the function gets no
`julia.get_pgcstack` call from this. An llvmcall of `julia.get_pgcstack` would not do: Julia
inlines it where it is called, which is the problem that [`use_gcstack_arg!`](@ref) avoids.
"""
@inline current_pgcstack() = pointer_from_objref(current_task()) + task_gcstack_offset()

"""
    use_gcstack_arg!(f::LLVM.Function, arg::LLVM.Argument)

Make the llvmcall `f` use its argument `arg` as the pgcstack of the code it inlines. The
caller gives this argument with [`current_pgcstack`](@ref), as a `Ptr{Cvoid}`.

Julia inlines an llvmcall into its caller, together with the `alwaysinline` functions that
the llvmcall calls. A `julia.get_pgcstack` call in that code thus goes into the middle of
the caller. On 1.13 the caller takes its pgcstack as its own `"gcstack"` argument. But the
GC lowering prefers a `julia.get_pgcstack` call in the entry block to that argument, and
pushes the GC frame of the caller only after the call. Then no safepoint before the call
has the roots of the caller.

Thus inline the `alwaysinline` functions into `f` here, and replace every
`julia.get_pgcstack` call in `f` with `arg`, as `LowerPTLS` does for a function that takes
its pgcstack as an argument.
"""
function use_gcstack_arg!(f::LLVM.Function, arg::LLVM.Argument)
    mod = f.parent
    run!(AlwaysInlinerPass(), mod)
    if !haskey(mod.functions, "julia.get_pgcstack")
        return
    end
    getter = mod.functions["julia.get_pgcstack"]
    calls = LLVM.CallInst[]
    for use in getter.uses
        call = use.user
        if call isa LLVM.CallInst && call.parent.parent == f
            push!(calls, call)
        end
    end
    if isempty(calls)
        return
    end
    T_pgcstack = getter.function_type.return_type
    pgcstack = @dispose B = IRBuilder() begin
        position!(B, LLVM.at_begin(f.entry))
        # Before 1.12, a `Ptr{Cvoid}` is an integer in Julia's IR.
        if arg.value_type isa LLVM.IntegerType
            inttoptr!(B, arg, T_pgcstack)
        else
            bitcast!(B, arg, T_pgcstack)
        end
    end
    for call in calls
        replace_uses!(call, pgcstack)
        erase!(call)
    end
    return
end

const swiftself_kind = :swiftself

Base.@assume_effects :removable :foldable :nothrow function has_swiftself(fn::LLVM.Function)::Bool
    for i in 1:length(fn.parameters)
        for attr in collect(fn.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return true
                end
            end
        end
    end
    return false
end

"""
    gcstack_arg_index(fn::LLVM.Function) -> Int

Give the index of the `pgcstack` parameter of `fn`, or `0` when `fn` has none.

Julia's codegen marks that parameter `swiftself` where the target supports the
swift calling convention, and turns the convention off where it does not
(`jl_codegen_output_t::use_swiftcc`, false on RISC-V). Julia 1.13 also gives
the parameter a `gcstack` string attribute, which `specsig` writes on every
version. Hence recognize either mark.
"""
Base.@assume_effects :removable :foldable :nothrow function gcstack_arg_index(fn::LLVM.Function)::Int
    for i in 1:length(fn.parameters)
        for attr in collect(fn.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return i
                end
            elseif attr isa LLVM.StringAttribute
                if attr.kind == "gcstack"
                    return i
                end
            end
        end
    end
    return 0
end

"""
    has_gcstack_arg(fn::LLVM.Function) -> Bool

Say if `fn` takes `pgcstack` as a parameter (see [`gcstack_arg_index`](@ref)).
"""
Base.@assume_effects :removable :foldable :nothrow has_gcstack_arg(fn::LLVM.Function)::Bool = gcstack_arg_index(fn) != 0

"""
    copy_metadata!(dst, src)

Attach every metadata node of the instruction `src`, except its debug location,
to `dst`.
"""
function copy_metadata!(dst::LLVM.Instruction, src::LLVM.Instruction)
    for (kind, md) in src.metadata
        kind == LLVM.MD_dbg && continue
        dst.metadata[kind] = md
    end
    return nothing
end

eraseInst(bb::LLVM.BasicBlock, @nospecialize(inst::LLVM.Instruction)) = erase!(inst)
eraseInst(bb::LLVM.Module, inst::LLVM.Function) = erase!(inst)
eraseInst(bb::LLVM.Module, inst::LLVM.GlobalVariable) = erase!(inst)

function unique_gcmarker!(func::LLVM.Function)
    entry_bb = first(func.blocks)
    pgcstack_func, _ = declare_pgcstack!(func.parent)

    found = LLVM.CallInst[]
    for I in entry_bb.instructions
        if I isa LLVM.CallInst && I.called_operand == pgcstack_func
            push!(found, I)
        end
    end
    if length(found) > 1
        for i in 2:length(found)
            LLVM.replace_uses!(found[i], found[1])
            eraseInst(entry_bb, found[i])
        end
    end
    return nothing
end

@inline AnonymousStruct(::Type{U}) where {U <: Tuple} =
    NamedTuple{ntuple(Symbol, Val(length(U.parameters))), U}

# recursively compute the eltype type indexed by idx[0], idx[1], ...
Base.@assume_effects :removable :foldable :nothrow function recursive_eltype(@nospecialize(val::LLVM.Value), idxs::Vector{Cuint})::LLVM.LLVMType
    ty = val.value_type::LLVM.LLVMType
    for i in idxs
        if isa(ty, LLVM.ArrayType)
            ty = ty.element_type::LLVM.LLVMType
        else
            @assert isa(ty, LLVM.StructType)
            ty = ty.elements[i + 1]::LLVM.LLVMType
        end
    end
    return ty
end

# Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
# and that Bool -> i8, not i1
function calling_conv_fixup(
        builder::LLVM.IRBuilder,
        @nospecialize(val::LLVM.Value),
        @nospecialize(tape::LLVM.LLVMType),
        @nospecialize(prev::LLVM.Value) = LLVM.UndefValue(tape),
        lidxs::Vector{Cuint} = Cuint[],
        ridxs::Vector{Cuint} = Cuint[],
        emesg = nothing,
    )::LLVM.Value
    ctype = recursive_eltype(val, lidxs)
    if ctype == tape
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        if length(ridxs) == 0
            return val
        else
            return insert_value!(builder, prev, val, ridxs)
        end
    end

    if isa(tape, LLVM.StructType)
        if isa(ctype, LLVM.ArrayType)
            @assert ctype.length == length(tape.elements)
            for (i, ty) in enumerate(tape.elements)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(ctype.elements) == length(tape.elements)
            for (i, ty) in enumerate(tape.elements)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
    elseif isa(tape, LLVM.ArrayType)
        if isa(ctype, LLVM.ArrayType)
            @assert ctype.length == tape.length
            for i in 1:tape.length
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, tape.element_type, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(ctype.elements) == tape.length
            for i in 1:tape.length
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, tape.element_type, prev, ln, rn, emesg)
            end
            return prev
        end
    end

    if isa(tape, LLVM.IntegerType) &&
            tape.width == 1 &&
            ctype.width != tape.width
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        val = trunc!(builder, val, tape)
        return if length(ridxs) != 0
            insert_value!(builder, prev, val, ridxs)
        else
            val
        end
    end
    if isa(tape, LLVM.PointerType) &&
            isa(ctype, LLVM.PointerType) &&
            tape.addrspace == ctype.addrspace
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        val = pointercast!(builder, val, tape)
        return if length(ridxs) != 0
            insert_value!(builder, prev, val, ridxs)
        else
            val
        end
    end
    if isa(ctype, LLVM.ArrayType) && ctype.length == 1 && ctype.element_type == tape
        lhs_n = copy(lidxs)
        push!(lhs_n, 0)
        return calling_conv_fixup(builder, val, tape, prev, lhs_n, ridxs, emesg)
    end


    msg2 = sprint() do io
        println(io, "Enzyme Internal Error: Illegal calling convention fixup")
        if emesg !== nothing
            emesg(io)
        end
        println(io, "ctype = ", ctype)
        println(io, "tape = ", tape)
        println(io, "val = ", string(val))
        println(io, "prev = ", string(prev))
        println(io, "lidxs = ", lidxs)
        println(io, "ridxs = ", ridxs)
        println(io, "tape_type(tape) = ", tape_type(tape))
        println(
            io,
            "convert(LLVMType, tape_type(tape)) = ",
            convert(LLVM.LLVMType, tape_type(tape); allow_boxed = true),
        )
    end
    throw(AssertionError(msg2))
end
