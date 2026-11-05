# Zero shadows of constant data stored into write-only output arguments.
#
# When a function stores constant data, such as a `Const` array held in a struct
# beside active data, into a `Duplicated` argument, the mixed-activity handler in
# `src/errors.jl` gives that data a zero-filled copy as its shadow. Derivatives
# written into that copy are discarded. A warm call that stores the same shape
# into the same output would allocate a new copy every time, as long as the
# constant array. Here the copy an earlier derivative stored at the same output
# location is reused instead, when all of the following hold:
#
# - an earlier call of this helper allocated it (`RETAINED_SINKS` records each
#   copy it allocates by object identity, without keeping it alive). Only such a
#   buffer is read in full: a shadow the caller supplied is never read beyond
#   what the primal accesses (docs/src/faq.md, "Identical types in `Duplicated`
#   / Memory Layout");
# - it has the type and size of the constant array;
# - it overlaps neither that array nor any argument or shadow of the call
#   (arrays passed directly or held in inline fields);
# - every element is a positive zero.
#
# Only combined reverse mode at batch width 1 without runtime activity qualifies,
# for a store by the differentiated function itself into one of its own
# `Duplicated` arguments, which it only writes and does not capture, when that
# function touches no global memory and it and its callees call nothing with
# unknown memory effects. Otherwise the handler allocates a new copy as before. A reference kept
# from an earlier call's output shadow may therefore be the same buffer as a
# later call's (docs/src/faq.md, "Reused zero shadows of constant data").
# Checking the buffer takes time linear in its length; this saves the
# allocation, not that time.

const RetainedSinkElement = Union{Base.IEEEFloat, Complex{<:Base.IEEEFloat}}

# The zero copies `reuse_retained_shadow` allocated, by address. A collected copy
# leaves an entry whose `WeakRef` is cleared, which the next sweep removes.
mutable struct RetainedSinks
    const lock::ReentrantLock
    const entries::Dict{UInt, WeakRef}
    limit::Int
end

const RETAINED_SINKS = RetainedSinks(ReentrantLock(), Dict{UInt, WeakRef}(), 64)

retained_sink_key(sink::Array)::UInt = UInt(pointer_from_objref(sink))

is_live_retained_sink(entry::Pair{UInt, WeakRef})::Bool = entry.second.value !== nothing

function register_retained_sink!(sink::Array)
    sinks = RETAINED_SINKS
    key = retained_sink_key(sink)
    lock(sinks.lock)
    try
        if length(sinks.entries) >= sinks.limit
            filter!(is_live_retained_sink, sinks.entries)
            sinks.limit = max(64, 2 * length(sinks.entries))
        end
        sinks.entries[key] = WeakRef(sink)
    finally
        unlock(sinks.lock)
    end
    return sink
end

function is_retained_sink(sink::Array)::Bool
    sinks = RETAINED_SINKS
    key = retained_sink_key(sink)
    lock(sinks.lock)
    found = try
        entry = get(sinks.entries, key, nothing)
        entry !== nothing && entry.value === sink
    finally
        unlock(sinks.lock)
    end
    return found
end

retained_sink_bits(x::Base.IEEEFloat) = reinterpret(Base.uinttype(typeof(x)), x)
retained_sink_bits(x::Complex) = retained_sink_bits(real(x)) | retained_sink_bits(imag(x))

# Whether every element is a positive zero, as `make_zero` writes them.
function retained_sink_is_zero(sink::Array{T})::Bool where {T <: RetainedSinkElement}
    acc = retained_sink_bits(zero(T))
    @inbounds @simd for i in eachindex(sink)
        acc |= retained_sink_bits(sink[i])
    end
    return iszero(acc)
end

# The field at `path` of `x`, or `nothing` where one on the way is undefined.
@generated function retained_shadow_field(x, ::Val{path}) where {path}
    body = Expr(:block, Expr(:meta, :inline))
    ex = :x
    for i in path
        push!(body.args, :(isdefined($ex, $i) || return nothing))
        ex = :(getfield($ex, $i))
    end
    push!(body.args, :(return $ex))
    return body
end

# Whether the storage of `candidate` and of the array `x` provably share no
# memory. An array whose elements hold references might hold `candidate`.
function retained_shadow_disjoint_array(candidate::Array, x::Array)::Bool
    isbitstype(eltype(x)) || return false
    isempty(candidate) && return true
    isempty(x) && return true
    GC.@preserve candidate x begin
        a = UInt(pointer(candidate))
        b = UInt(pointer(x))
        return a <= b ? b - a >= sizeof(candidate) : a - b >= sizeof(x)
    end
end

# Code that returns `false` unless `candidate` shares no memory with the value of
# expression `x` of type `T` or with the arrays it holds directly or in inline
# fields, leaving out the field at the path `skip`. Any other reference counts
# as overlapping. The checks are generated as straight-line code: a recursive walk
# exceeded inference's limits and allocated on every call.
function retained_shadow_disjoint_code(@nospecialize(T), x, skip::Tuple)
    T <: Array && return :(retained_shadow_disjoint_array(candidate, $x) || return false)
    (T <: Union{String, Symbol} || isprimitivetype(T)) && return nothing
    (T <: Ptr || !isconcretetype(T) || !isstructtype(T)) && return :(return false)
    body = Expr(:block)
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        field = :(getfield($x, $i))
        inner = if !isempty(skip) && i == first(skip)
            length(skip) > 1 || continue
            retained_shadow_disjoint_code(F, field, Base.tail(skip))
        elseif Base.allocatedinline(F) || F <: Array
            retained_shadow_disjoint_code(F, field, ())
        else
            :(return false)
        end
        inner === nothing && continue
        push!(
            body.args, :(
                if isdefined($x, $i)
                    $inner
                end
            )
        )
    end
    return body
end

# Whether `candidate` shares no memory with `prim`, with the output's `shadow`
# apart from the field at `path`, or with any of the values in `protected`.
@generated function retained_shadow_disjoint(
        candidate::Array, prim::Array, shadow::S, ::Val{path}, protected::P
    ) where {S, path, P <: Tuple}
    body = Expr(:block, Expr(:meta, :inline))
    push!(body.args, :(retained_shadow_disjoint_array(candidate, prim) || return false))
    code = retained_shadow_disjoint_code(S, :shadow, path)
    code === nothing || push!(body.args, code)
    for i in 1:fieldcount(P)
        code = retained_shadow_disjoint_code(fieldtype(P, i), :(getfield(protected, $i)), ())
        code === nothing || push!(body.args, code)
    end
    push!(body.args, :(return true))
    return body
end

# The shadow of the constant array `prim` stored into the output whose shadow is
# `shadow`, at the field `path` of that output. `protected` are the call's other
# arguments and shadows, the output's own primal among them.
function reuse_retained_shadow(
        prim::Array{T, N}, shadow, path::Val, protected::Vararg{Any, M}
    ) where {T <: RetainedSinkElement, N, M}
    candidate = retained_shadow_field(shadow, path)
    if candidate isa Array{T, N} &&
            candidate !== prim &&
            size(candidate) == size(prim) &&
            is_retained_sink(candidate) &&
            retained_shadow_disjoint(candidate, prim, shadow, path, protected) &&
            retained_sink_is_zero(candidate)
        return candidate
    end
    return register_retained_sink!(make_zero(prim))
end

# The path of field indices to the pointer field at byte `offset` of `T`, through
# inline fields, or `nothing`.
function retained_shadow_path(@nospecialize(T), offset::Int)
    Base.isconcretetype(T) || return nothing
    isstructtype(T) || return nothing
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        start = Int(fieldoffset(T, i))
        if !Base.allocatedinline(F)
            start == offset && return (i,)
        elseif start <= offset < start + sizeof(F)
            tail = retained_shadow_path(F, offset - start)
            tail === nothing || return (i, tail...)
        end
    end
    return nothing
end

# `enzyme!` marks the function it differentiates in combined reverse mode while
# libEnzyme runs. Only a store into one of its own arguments qualifies: the checks
# below see that function's arguments and code, which for a callee would miss the
# caller's other arguments and memory.
const RETAINED_SHADOW_ENTRY = "enzymejl_retained_shadow_entry"

function mark_retained_shadow_entry!(f::LLVM.Function)
    push!(function_attributes(f), StringAttribute(RETAINED_SHADOW_ENTRY, ""))
    return nothing
end

function is_retained_shadow_entry(f::LLVM.Function)::Bool
    for attr in collect(function_attributes(f))
        attr isa LLVM.StringAttribute && kind(attr) == RETAINED_SHADOW_ENTRY && return true
    end
    return false
end

function unmark_retained_shadow_entry!(f::LLVM.Function)
    is_retained_shadow_entry(f) || return nothing
    delete!(function_attributes(f), StringAttribute(RETAINED_SHADOW_ENTRY, ""))
    return nothing
end

function retained_shadow_arg_attr(p::LLVM.Argument, name::String)::Bool
    f = LLVM.Function(LLVM.API.LLVMGetParamParent(p))
    k = kind(LLVM.EnumAttribute(name))
    for (i, arg) in enumerate(parameters(f))
        arg == p || continue
        for attr in collect(parameter_attributes(f, i))
            attr isa LLVM.EnumAttribute && kind(attr) == k && return true
        end
    end
    return false
end

# The derivative reads `p`, which the primal only wrote.
function retained_shadow_readable!(p::LLVM.Argument)
    f = LLVM.Function(LLVM.API.LLVMGetParamParent(p))
    for (i, arg) in enumerate(parameters(f))
        arg == p || continue
        delete!(parameter_attributes(f, i), LLVM.EnumAttribute("writeonly"))
    end
    return nothing
end

function retained_shadow_external_readonly(attrs)::Bool
    for attr in attrs
        attr isa LLVM.EnumAttribute || continue
        k = kind(attr)
        if LLVM.version().major <= 15
            if k in (
                    kind(EnumAttribute("readnone")), kind(EnumAttribute("readonly")),
                    kind(EnumAttribute("argmemonly")), kind(EnumAttribute("inaccessiblememonly")),
                    kind(EnumAttribute("inaccessiblemem_or_argmemonly")),
                )
                return true
            end
        elseif k == kind(EnumAttribute("memory"))
            is_readonly(getModRef(MemoryEffect(value(attr)), Other)) && return true
        end
    end
    return false
end

function retained_shadow_no_argument_memory(attrs)::Bool
    for attr in attrs
        attr isa LLVM.EnumAttribute || continue
        k = kind(attr)
        if LLVM.version().major <= 15
            k in (kind(EnumAttribute("readnone")), kind(EnumAttribute("inaccessiblememonly"))) && return true
        elseif k == kind(EnumAttribute("memory"))
            effect = MemoryEffect(value(attr))
            is_readnone(getModRef(effect, ArgMem)) && is_readnone(getModRef(effect, Other)) && return true
        end
    end
    return false
end

# Whether values of the LLVM type `ty` hold no pointers.
function retained_shadow_pointer_free(@nospecialize(ty::LLVM.LLVMType))::Bool
    ty isa LLVM.PointerType && return false
    if ty isa LLVM.StructType
        for el in elements(ty)
            retained_shadow_pointer_free(el) || return false
        end
    elseif ty isa LLVM.ArrayType || ty isa LLVM.VectorType
        return retained_shadow_pointer_free(eltype(ty))
    end
    return true
end

function retained_shadow_global_pointer(@nospecialize(v::LLVM.Value))::Bool
    # Constant data without pointers cannot reach a buffer. Julia 1.12 copies the
    # literal fields of a view, such as its `1:2` indices, from such a global.
    v isa LLVM.GlobalVariable &&
        return !(isconstant(v) && retained_shadow_pointer_free(global_value_type(v)))
    if v isa LLVM.ConstantExpr || v isa LLVM.ConstantAggregate
        value_type(v) isa LLVM.PointerType && return true
        for op in operands(v)
            retained_shadow_global_pointer(op) && return true
        end
    end
    return false
end

# Whether `obj`, a Julia object a function loads from a global slot, can hold
# no buffer and receive no write: a bits value, a name, a type, or the empty
# `Memory` that Julia 1.11 and later share between empty arrays.
function retained_shadow_inert(@nospecialize(obj))::Bool
    isbits(obj) && return true
    obj isa Union{Symbol, String, Type} && return true
    isdefined(Core, :GenericMemory) && obj isa getfield(Core, :GenericMemory) && return isempty(obj)
    return false
end

# Whether the load `inst` reads the slot of a Julia object literal that
# `retained_shadow_inert` accepts.
function retained_shadow_inert_literal(inst::LLVM.LoadInst, enzyme_ctx)::Bool
    enzyme_ctx isa EnzymeContext || return false
    slot = operands(inst)[1]
    slot isa LLVM.GlobalVariable || return false
    val = slot_value(enzyme_ctx, LLVM.name(slot))
    val === nothing && return false
    return retained_shadow_inert(unbind(val.value))
end

# Whether `f` and everything it calls access only memory reached through their
# arguments or allocated by them, so a buffer that is zero at the store stays
# zero until it reaches the output.
function retained_shadow_local_memory(
        f::LLVM.Function, enzyme_ctx, seen::Set{LLVM.Function} = Set{LLVM.Function}()
    )::Bool
    f in seen && return true
    push!(seen, f)
    dead = guaranteed_unreachable(f)
    for bb in blocks(f)
        bb in dead && continue
        for inst in instructions(bb)
            inst isa LLVM.IntToPtrInst && return false
            if inst isa LLVM.CallBase
                callee = LLVM.called_operand(inst)
                callee isa LLVM.Function || return false
                attrs = vcat(collect(function_attributes(callee)), callsite_function_attributes(inst))
                if isempty(blocks(callee))
                    retained_shadow_external_readonly(attrs) || return false
                else
                    retained_shadow_local_memory(callee, enzyme_ctx, seen) || return false
                end
                retained_shadow_no_argument_memory(attrs) && continue
            end
            for op in operands(inst)
                retained_shadow_global_pointer(op) || continue
                inst isa LLVM.LoadInst && op == operands(inst)[1] &&
                    retained_shadow_inert_literal(inst, enzyme_ctx) && continue
                return false
            end
        end
    end
    return true
end

# The shadow of the constant array `cur` that the store `val` writes into an
# output argument, reusing that output's previous zero copy where
# `reuse_retained_shadow` can; `nothing` where this does not apply.
function emit_reused_retained_shadow!(
        B::LLVM.IRBuilder,
        gutils::GradientUtils,
        @nospecialize(val::LLVM.Value),
        @nospecialize(cur::LLVM.Value),
        width::Integer,
        created::Vector{LLVM.Instruction},
        enzyme_ctx,
    )::Union{Nothing, LLVM.Value}
    get_mode(gutils) == API.DEM_ReverseModeCombined || return nothing
    width == 1 || return nothing
    get_runtime_activity(gutils) && return nothing
    val isa LLVM.StoreInst || return nothing
    operands(val)[1] == cur || return nothing
    root, offset = constant_field_pointer(operands(val)[2])
    root isa LLVM.Argument || return nothing
    is_retained_shadow_entry(parent_scope(root)) || return nothing
    API.EnzymeGradientUtilsGetDiffeType(gutils, root, false) in
        (API.DFT_DUP_ARG, API.DFT_DUP_NONEED) || return nothing
    is_tracked_pointer(value_type(root)) || return nothing
    retained_shadow_arg_attr(root, "writeonly") || return nothing
    retained_shadow_arg_attr(root, "nocapture") || return nothing
    legal, T, byref = abs_typeof(root, enzyme_ctx, true)
    legal && byref == GPUCompiler.MUT_REF && ismutabletype(T) || return nothing
    path = retained_shadow_path(T, offset)
    path === nothing && return nothing
    legal, C, _ = abs_typeof(cur, enzyme_ctx, true)
    legal && C <: Array{<:RetainedSinkElement} || return nothing
    f = parent_scope(root)
    retained_shadow_local_memory(f, enzyme_ctx) || return nothing
    for p in parameters(f)
        ty = value_type(p)
        ty isa LLVM.PointerType || continue
        is_tracked_pointer(ty) || return nothing
        API.EnzymeGradientUtilsGetDiffeType(gutils, p, false) in
            (API.DFT_CONSTANT, API.DFT_DUP_ARG, API.DFT_DUP_NONEED) || return nothing
    end
    args = LLVM.Value[
        unsafe_to_llvm(B, reuse_retained_shadow),
        new_from_original(gutils, cur),
        invert_pointer(gutils, root, B),
        unsafe_to_llvm(B, Val(path)),
    ]
    for p in parameters(f)
        is_tracked_pointer(value_type(p)) || continue
        push!(args, new_from_original(gutils, p))
        p == root && continue
        if API.EnzymeGradientUtilsGetDiffeType(gutils, p, false) in (API.DFT_DUP_ARG, API.DFT_DUP_NONEED)
            push!(args, invert_pointer(gutils, p, B))
        end
    end
    for p in args
        p isa LLVM.Argument && retained_shadow_readable!(p)
    end
    return emit_zero_shadows!(B, args, value_type(cur), width, created)
end

# Differentiating a derivative that contains this call: both results are fresh
# zero arrays, and nothing flows back to the arguments. Treating the reuse as an
# identity would carry the previous buffer's tangent into the new one.

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(reuse_retained_shadow)}, ::Type{RT},
        prim::Annotation, args::Vararg{Annotation, N},
    ) where {RT, N}
    value = make_zero(prim.val)
    EnzymeRules.needs_shadow(config) || return EnzymeRules.needs_primal(config) ? value : nothing
    shadow = EnzymeRules.width(config) == 1 ? make_zero(value) :
        map(make_zero, ntuple(Returns(value), Val(EnzymeRules.width(config))))
    EnzymeRules.needs_primal(config) || return shadow
    return EnzymeRules.forward_rule_return_type(config, RT)(value, shadow)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(reuse_retained_shadow)}, ::Type{RT},
        prim::Annotation, args::Vararg{Annotation, N},
    ) where {RT, N}
    value = make_zero(prim.val)
    shadow = if EnzymeRules.needs_shadow(config)
        EnzymeRules.width(config) == 1 ? make_zero(value) :
            map(make_zero, ntuple(Returns(value), Val(EnzymeRules.width(config))))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? value : nothing, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(reuse_retained_shadow)}, dret, tape,
        prim::Annotation, args::Vararg{Annotation, N},
    ) where {N}
    return ntuple(Returns(nothing), Val(N + 1))
end
