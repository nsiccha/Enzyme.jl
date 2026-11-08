@inline function EnzymeCore.make_zero(x::FT)::FT where {FT <: AbstractFloat}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(x::Complex{FT})::Complex{FT} where {FT <: AbstractFloat}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(
        x::Array{FT, N},
    )::Array{FT, N} where {FT <: AbstractFloat, N}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(
        x::Array{Complex{FT}, N},
    )::Array{Complex{FT}, N} where {FT <: AbstractFloat, N}
    return Base.zero(x)
end

@static if VERSION < v"1.11-"
else
    @inline function EnzymeCore.make_zero(
            x::GenericMemory{kind, FT},
        )::GenericMemory{kind, FT} where {FT <: AbstractFloat, kind}
        return Base.zero(x)
    end
    @inline function EnzymeCore.make_zero(
            x::GenericMemory{kind, Complex{FT}},
        )::GenericMemory{kind, Complex{FT}} where {FT <: AbstractFloat, kind}
        return Base.zero(x)
    end
end

# `make_zero(x)` and `make_zero!(x)` without a `seen` table, for a tuple or named
# tuple (nested to any depth) of floating-point numbers, integers, and Arrays and
# `Ref`s of floating-point numbers, or for such a `Ref` itself: the active point a
# gradient of several arguments takes. Only Base types qualify, so no package can
# give a part of such a value a `make_zero` method of its own.
#
# The table costs more than such a shadow: `IdDict()` is two allocations and 336
# bytes, and each lookup and insertion is a hash-table call. A copy needs it here
# only to give two references to one Array or `Ref` one shadow, so
# `make_zero_aliased` checks that instead. An in-place zeroing never needs it:
# zeroing one array twice is harmless, and nothing in such a value can reach
# itself. Any other value keeps the general path below.
const MakeZeroFloat = Union{Base.IEEEFloat, Complex{Float16}, Complex{Float32}, Complex{Float64}}
const MakeZeroInactive = Union{Base.BitInteger, Bool, Nothing}

# The most Arrays and `Ref`s a copy compares pairwise instead of using a table
# (at 32 `Ref`s of one type the 496 comparisons cost less than half of what the
# table does), and the most fields of one tuple the fast paths take.
const MAKE_ZERO_FLAT_LEAVES = 32
const MAKE_ZERO_FLAT_FIELDS = 32

# The number of Arrays and `Ref`s in a value of type `T` built only from the types
# above, or -1 when `T` is any other type. With `scalars = false` (zeroing in
# place) a floating-point number is excluded too: no tuple field can be zeroed in
# place, so the general path raises its error for it.
Base.@assume_effects :foldable function make_zero_flat_leaves(
        @nospecialize(T::Type), scalars::Bool,
    )::Int
    Base.isconcretetype(T) || return -1
    if T <: MakeZeroFloat
        return scalars ? 0 : -1
    elseif T <: MakeZeroInactive
        return 0
    elseif T <: Array || T <: Base.RefValue
        FT = eltype(T)
        return Base.isconcretetype(FT) && FT <: MakeZeroFloat ? 1 : -1
    elseif T <: Tuple || T <: NamedTuple
        nf = fieldcount(T)
        nf <= MAKE_ZERO_FLAT_FIELDS || return -1
        n = 0
        for i in 1:nf
            k = make_zero_flat_leaves(fieldtype(T, i), scalars)
            k < 0 && return -1
            n += k
        end
        return n
    end
    return -1
end

# The shadow of a value `make_zero_flat_leaves` accepts, as the general path makes it.
@inline make_zero_flat(x::MakeZeroFloat) = zero(x)
@inline make_zero_flat(x::MakeZeroInactive) = x
@inline make_zero_flat(x::Array) = Base.zero(x)
@inline make_zero_flat(::Base.RefValue{T}) where {T} = Base.RefValue{T}(zero(T))
@inline make_zero_flat(x::Tuple) = ntuple(MakeZeroFlatField(x), Val(length(x)))
@inline make_zero_flat(x::NamedTuple{A}) where {A} = NamedTuple{A}(make_zero_flat(values(x)))

# Field `i` of such a tuple, zeroed: `ntuple` with a callable unrolls at every
# length the fast path takes, where Base's `map` stops at 32 fields.
struct MakeZeroFlatField{P}
    prev::P
end
@inline (f::MakeZeroFlatField)(i::Int) = make_zero_flat(getfield(f.prev, i))

# The Arrays and `Ref`s of such a value, as a tuple.
@inline make_zero_flat_refs(x::Union{Array, Base.RefValue}) = (x,)
@inline make_zero_flat_refs(::Union{MakeZeroFloat, MakeZeroInactive}) = ()
@inline make_zero_flat_refs(x::NamedTuple) = make_zero_flat_refs(values(x))
@inline make_zero_flat_refs(::Tuple{}) = ()
@inline function make_zero_flat_refs(x::Tuple)
    return (make_zero_flat_refs(first(x))..., make_zero_flat_refs(Base.tail(x))...)
end

# Whether two elements of `x` are one object. `===` of two values of different
# types folds to `false`, so only elements of one type are compared at run time.
@inline make_zero_aliased(::Tuple{}) = false
@inline function make_zero_aliased(x::Tuple)
    return make_zero_aliases(first(x), Base.tail(x)) || make_zero_aliased(Base.tail(x))
end
@inline make_zero_aliases(x, ::Tuple{}) = false
@inline make_zero_aliases(x, ys::Tuple) = x === first(ys) || make_zero_aliases(x, Base.tail(ys))

@inline function EnzymeCore.make_zero(
        prev::T, ::Val{false} = Val(false),
    )::T where {T <: Union{Tuple, NamedTuple, Base.RefValue}}
    n = make_zero_flat_leaves(T, true)
    if 0 <= n <= MAKE_ZERO_FLAT_LEAVES && !make_zero_aliased(make_zero_flat_refs(prev))
        return make_zero_flat(prev)
    end
    return EnzymeCore.make_zero(T, IdDict(), prev, Val(false))
end

# Zero in place a value `make_zero_flat_leaves(T, false)` accepts.
@inline make_zero_flat!(x::Array{T}) where {T} = (fill!(x, zero(T)); nothing)
@inline make_zero_flat!(x::Base.RefValue{T}) where {T} = (x[] = zero(T); nothing)
@inline make_zero_flat!(::MakeZeroInactive) = nothing
@inline make_zero_flat!(x::NamedTuple) = make_zero_flat!(values(x))
@inline make_zero_flat!(::Tuple{}) = nothing
@inline function make_zero_flat!(x::Tuple)
    make_zero_flat!(first(x))
    return make_zero_flat!(Base.tail(x))
end


@inline function EnzymeCore.make_zero(
	::Type{Array{Union{}, N}},
        seen::IdDict,
	prev::Array{Union{}, N},
        ::Val{copy_if_inactive} = Val(false),
    )::Array{Union{}, N} where {copy_if_inactive, N}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = copy(prev)
    seen[prev] = newa
    return newa
end

@inline function EnzymeCore.make_zero(
        ::Type{Array{FT, N}},
        seen::IdDict,
        prev::Array{FT, N},
        ::Val{copy_if_inactive} = Val(false),
    )::Array{FT, N} where {copy_if_inactive, FT <: AbstractFloat, N}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end
@inline function EnzymeCore.make_zero(
        ::Type{Array{Complex{FT}, N}},
        seen::IdDict,
        prev::Array{Complex{FT}, N},
        ::Val{copy_if_inactive} = Val(false),
    )::Array{Complex{FT}, N} where {copy_if_inactive, FT <: AbstractFloat, N}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end

@static if VERSION < v"1.11-"
else
    @inline function EnzymeCore.make_zero(
            ::Type{GenericMemory{kind, FT}},
            seen::IdDict,
            prev::GenericMemory{kind, FT},
            ::Val{copy_if_inactive} = Val(false),
        )::GenericMemory{kind, FT} where {copy_if_inactive, FT <: AbstractFloat, kind}
        if haskey(seen, prev)
            return seen[prev]
        end
        newa = Base.zero(prev)
        seen[prev] = newa
        return newa
    end
    @inline function EnzymeCore.make_zero(
            ::Type{GenericMemory{kind, Complex{FT}}},
            seen::IdDict,
            prev::GenericMemory{kind, Complex{FT}},
            ::Val{copy_if_inactive} = Val(false),
        )::GenericMemory{kind, Complex{FT}} where {copy_if_inactive, FT <: AbstractFloat, kind}
        if haskey(seen, prev)
            return seen[prev]
        end
        newa = Base.zero(prev)
        seen[prev] = newa
        return newa
    end
end

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT <: AbstractFloat}
    return RT(0)
end

@inline function EnzymeCore.make_zero(
        ::Type{Complex{RT}},
        seen::IdDict,
        prev::Complex{RT},
        ::Val{copy_if_inactive} = Val(false),
    )::Complex{RT} where {copy_if_inactive, RT <: AbstractFloat}
    return Complex{RT}(0)
end

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT <: Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    if guaranteed_const(RT)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    newa = RT(undef, size(prev))
    seen[prev] = newa
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            innerty = Core.Typeof(pv)
            @inbounds newa[I] =
                EnzymeCore.make_zero(innerty, seen, pv, Val(copy_if_inactive))
        end
    end
    return newa
end

@static if VERSION < v"1.11-"
else
    @inline function EnzymeCore.make_zero(
            ::Type{RT},
            seen::IdDict,
            prev::RT,
            ::Val{copy_if_inactive} = Val(false),
        )::RT where {copy_if_inactive, RT <: GenericMemory}
        if haskey(seen, prev)
            return seen[prev]
        end
        if guaranteed_const(RT)
            return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
        end
        newa = RT(undef, size(prev))
        seen[prev] = newa
        for I in eachindex(prev)
            if isassigned(prev, I)
                pv = prev[I]
                innerty = Core.Typeof(pv)
                @inbounds newa[I] =
                    EnzymeCore.make_zero(innerty, seen, pv, Val(copy_if_inactive))
            end
        end
        return newa
    end
end

@static if VERSION < v"1.11-"
else
    # A `MemoryRef` must point into the shadow of its `Memory`. The generic struct method
    # would zero the `Memory` but copy the `ptr_or_offset` field, leaving the shadow
    # `MemoryRef` pointing into the primal data.
    @inline function EnzymeCore.make_zero(
            ::Type{RT},
            seen::IdDict,
            prev::RT,
            ::Val{copy_if_inactive} = Val(false),
        )::RT where {copy_if_inactive, RT <: GenericMemoryRef}
        if haskey(seen, prev)
            return seen[prev]
        end
        if guaranteed_const(RT)
            return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
        end
        mem = prev.mem
        newmem = EnzymeCore.make_zero(Core.Typeof(mem), seen, mem, Val(copy_if_inactive))
        # `memoryrefoffset` is 1-based; no bounds check, so that refs into empty
        # memories (e.g. of an empty `Vector`) are supported as well
        newref = Base.memoryref(newmem)
        offset = Base.memoryrefoffset(prev)
        if offset != 1
            newref = @inbounds Base.memoryref(newref, offset)
        end
        seen[prev] = newref
        return newref
    end
end

# Field `i` of a tuple of type `RT`, zeroed. A callable struct rather than a closure,
# so that `fieldtype(RT, i)` folds for each constant `i`, and each field's
# `make_zero` dispatches statically.
struct MakeZeroTupleField{RT, copy_if_inactive, S, P}
    seen::S
    prev::P
end

@inline function MakeZeroTupleField{RT, copy_if_inactive}(
        seen::S, prev::P,
    ) where {RT, copy_if_inactive, S, P}
    return MakeZeroTupleField{RT, copy_if_inactive, S, P}(seen, prev)
end

@inline function (f::MakeZeroTupleField{RT, copy_if_inactive})(
        i::Int,
    ) where {RT, copy_if_inactive}
    return EnzymeCore.make_zero(
        fieldtype(RT, i), f.seen, getfield(f.prev, i), Val(copy_if_inactive)
    )
end

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT <: Tuple}
    if !Base.isconcretetype(RT)
        # A tuple type with an abstract or `Vararg` field: zero the tuple as the
        # concrete type it has.
        return EnzymeCore.make_zero(Core.Typeof(prev), seen, prev, Val(copy_if_inactive))
    end
    f = MakeZeroTupleField{RT, copy_if_inactive}(seen, prev)
    N = fieldcount(RT)
    # Unrolled up to `MAKE_ZERO_FLAT_FIELDS` fields, as Base's `map` over a tuple is;
    # a longer tuple loops, and its `make_zero` calls still dispatch statically
    # when all its fields have one type.
    return N <= MAKE_ZERO_FLAT_FIELDS ? ntuple(f, Val(N)) : ntuple(f, N)
end

@inline function EnzymeCore.make_zero(
        ::Type{NamedTuple{A, RT}},
        seen::IdDict,
        prev::NamedTuple{A, RT},
        ::Val{copy_if_inactive} = Val(false),
    )::NamedTuple{A, RT} where {copy_if_inactive, A, RT}
    prevtup = RT(prev)
    TT = Core.Typeof(prevtup)  # RT can be abstract
    return NamedTuple{A, RT}(EnzymeCore.make_zero(TT, seen, prevtup, Val(copy_if_inactive)))
end

@inline function EnzymeCore.make_zero(
        ::Type{Core.Box},
        seen::IdDict,
        prev::Core.Box,
        ::Val{copy_if_inactive} = Val(false),
    ) where {copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    res = Core.Box()
    seen[prev] = res
    res.contents = EnzymeCore.make_zero(Core.Typeof(prev2), seen, prev2, Val(copy_if_inactive))
    return res
end

@inline function EnzymeCore.make_zero(
        ::Type{Core.SimpleVector},
        seen::IdDict,
        prev::Core.SimpleVector,
        ::Val{copy_if_inactive} = Val(false),
    )::Core.SimpleVector where {copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]
    end
    data = []
    for v in prev
        push!(data, EnzymeCore.make_zero(Core.Typeof(v), seen, v, Val(copy_if_inactive)))
    end
    newa = Core.svec(data)
    seen[prev] = newa
    return newa
end

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT <: Dict}
    if haskey(seen, prev)
        return seen[prev]
    end
    res = empty(prev)
    seen[prev] = res
    for (k, v) in prev
        new_v = EnzymeCore.make_zero(Core.Typeof(v), seen, v, Val(copy_if_inactive))
        res[k] = new_v
    end
    return res
end

# Runtime objects that make_zero shares with the primal instead of copying, also
# under `copy_if_inactive`. A
# `Task` holds no derivative, and a field-wise copy is not a task: it is smaller
# than the runtime's task struct, so the next garbage collection reads past it
# and crashes. A `Core.Binding` is the identity of a global variable. Both are
# reached from data through a lock's owner, a condition's wait queue, or a
# `GlobalRef`. `Task` stays active: the rules for spawned tasks make their own
# shadow tasks.
const MakeZeroShared = Union{Task, Core.Binding}

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT <: MakeZeroShared}
    return prev
end

# Write `v` into field `i` of the mutable struct `x`: a `const` field through the
# runtime, an `@atomic` field with an atomic store, any other with `setfield!`.
@inline function make_zero_setfield!(x::T, i::Int, @nospecialize(v)) where {T}
    if Base.isconst(T, i)
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i - 1, v)
    elseif Base.isfieldatomic(T, i)
        setfield!(x, i, v, :sequentially_consistent)
    else
        setfield!(x, i, v)
    end
    return nothing
end

@inline function EnzymeCore.make_zero(
        ::Type{RT},
        seen::IdDict,
        prev::RT,
        ::Val{copy_if_inactive} = Val(false),
    )::RT where {copy_if_inactive, RT}
    if guaranteed_const(RT)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if haskey(seen, prev)
        return seen[prev]
    end
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)
    if ismutable(prev)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)::RT
        seen[prev] = y
        for i in 1:nf
            if isdefined(prev, i)
                xi = getfield(prev, i)
                T = Core.Typeof(xi)
                xi = EnzymeCore.make_zero(T, seen, xi, Val(copy_if_inactive))
                make_zero_setfield!(y, i, xi)
            end
        end
        return y
    end
    if nf == 0
        return prev
    end
    flds = Vector{Any}(undef, nf)
    for i in 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            xi = EnzymeCore.make_zero(Core.Typeof(xi), seen, xi, Val(copy_if_inactive))
            flds[i] = xi
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)
    seen[prev] = y
    return y
end

function make_zero_immutable!(prev::T, seen::S)::T where {T <: AbstractFloat, S}
    return zero(T)
end

function make_zero_immutable!(
        prev::Complex{T},
        seen::S,
    )::Complex{T} where {T <: AbstractFloat, S}
    return zero(Complex{T})
end

function make_zero_immutable!(prev::T, seen::S)::T where {T <: Tuple, S}
    if guaranteed_const(T)
        return prev  # unreachable from make_zero!
    end
    return ntuple(Val(length(T.parameters))) do i
        Base.@_inline_meta
        p = prev[i]
        SBT = Core.Typeof(p)
        if guaranteed_const(SBT)
            p  # covered by several tests even if not shown in coverage
        elseif !ismutabletype(SBT)
            make_zero_immutable!(p, seen)
        else
            EnzymeCore.make_zero!(p, seen)
            p
        end
    end
end

function make_zero_immutable!(prev::NamedTuple{a, b}, seen::S)::NamedTuple{a, b} where {a, b, S}
    if guaranteed_const(NamedTuple{a, b})
        return prev  # unreachable from make_zero!
    end
    return NamedTuple{a, b}(
        ntuple(Val(length(b.parameters))) do i
            Base.@_inline_meta
            p = prev[a[i]]
            SBT = Core.Typeof(p)
            if guaranteed_const(SBT)
                p  # covered by several tests even if not shown in coverage
            elseif !ismutabletype(SBT)
                make_zero_immutable!(p, seen)
            else
                EnzymeCore.make_zero!(p, seen)
                p
            end
        end
    )
end


function make_zero_immutable!(prev::T, seen::S)::T where {T, S}
    if guaranteed_const(T)
        return prev  # unreachable from make_zero!
    end
    @assert !ismutabletype(T)
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    flds = Vector{Any}(undef, nf)
    for i in 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            ST = Core.Typeof(xi)
            flds[i] = if guaranteed_const(ST)
                xi
            elseif !ismutabletype(ST)
                make_zero_immutable!(xi, seen)
            else
                EnzymeCore.make_zero!(xi, seen)
                xi
            end
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nf)::T
end

macro register_make_zero_inplace(sym)
    return quote
        @inline function $sym(
                prev::Base.RefValue{T},
                seen::ST,
            )::Nothing where {T <: AbstractFloat, ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            prev[] = zero(T)
            return nothing
        end

        @inline function $sym(
                prev::Base.RefValue{Complex{T}},
                seen::ST,
            )::Nothing where {T <: AbstractFloat, ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            prev[] = zero(Complex{T})
            return nothing
        end
        @inline function $sym(
                prev::Array{T, N},
                seen::ST,
            )::Nothing where {T <: AbstractFloat, N, ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(T))
            return nothing
        end

        @inline function $sym(
                prev::Array{Complex{T}, N},
                seen::ST,
            )::Nothing where {T <: AbstractFloat, N, ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(Complex{T}))
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
            @inline function $sym(
                    prev::GenericMemory{kind, T},
                    seen::ST,
                )::Nothing where {T <: AbstractFloat, kind, ST}
                if !isnothing(seen)
                    if prev in seen
                        return nothing
                    end
                    push!(seen, prev)
                end
                fill!(prev, zero(T))
                return nothing
            end

            @inline function $sym(
                    prev::GenericMemory{kind, Complex{T}},
                    seen::ST,
                )::Nothing where {T <: AbstractFloat, kind, ST}
                if !isnothing(seen)
                    if prev in seen
                        return nothing
                    end
                    push!(seen, prev)
                end
                fill!(prev, zero(Complex{T}))
                return nothing
            end
        end

        @inline function $sym(
                prev::Base.RefValue{T},
            )::Nothing where {T <: AbstractFloat}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(
                prev::Base.RefValue{Complex{T}},
            )::Nothing where {T <: AbstractFloat}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(prev::Array{T, N})::Nothing where {T <: AbstractFloat, N}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(
                prev::Array{Complex{T}, N},
            )::Nothing where {T <: AbstractFloat, N}
            $sym(prev, nothing)
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
            @inline function $sym(
                    prev::GenericMemory{kind, T}
                )::Nothing where {T <: AbstractFloat, kind}
                $sym(prev, nothing)
                return nothing
            end

            @inline function $sym(
                    prev::GenericMemory{kind, Complex{T}}
                )::Nothing where {T <: AbstractFloat, kind}
                $sym(prev, nothing)
                return nothing
            end
        end

        @inline function $sym(prev::Array{T, N}, seen::ST)::Nothing where {T, N, ST}
            if guaranteed_const(T)
                return nothing
            end
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            for I in eachindex(prev)
                if isassigned(prev, I)
                    pv = prev[I]
                    SBT = Core.Typeof(pv)
                    if guaranteed_const(SBT)
                        continue
                    elseif !ismutabletype(SBT)
                        @inbounds prev[I] = make_zero_immutable!(pv, seen)
                    else
                        $sym(pv, seen)
                    end
                end
            end
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
            @inline function $sym(prev::GenericMemory{kind, T}, seen::ST)::Nothing where {T, kind, ST}
                if guaranteed_const(T)
                    return nothing
                end
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
                for I in eachindex(prev)
                    if isassigned(prev, I)
                        pv = prev[I]
                        SBT = Core.Typeof(pv)
                        if guaranteed_const(SBT)
                            continue
                        elseif !ismutabletype(SBT)
                            @inbounds prev[I] = make_zero_immutable!(pv, seen)
                        else
                            $sym(pv, seen)
                        end
                    end
                end
                return nothing
            end
        end

        @inline function $sym(
                prev::Base.RefValue{T},
                seen::ST,
            )::Nothing where {T, ST}
            if guaranteed_const(T)
                return nothing
            end
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            pv = prev[]
            SBT = Core.Typeof(pv)
            if guaranteed_const(SBT)
                return nothing
            elseif !ismutabletype(SBT)
                prev[] = make_zero_immutable!(pv, seen)
            else
                $sym(pv, seen)
            end
            return nothing
        end

        @inline function $sym(prev::Core.Box, seen::ST)::Nothing where {ST}
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            pv = prev.contents
            SBT = Core.Typeof(pv)
            if guaranteed_const(SBT)
                return nothing
            elseif !ismutabletype(SBT)
                prev.contents = make_zero_immutable!(pv, seen)
            else
                $sym(pv, seen)
            end
            return nothing
        end

        @inline function $sym(prev::Dict, seen::ST)::Nothing where {ST}
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            for (k, v) in prev
                SBT = Core.Typeof(v)
                if guaranteed_const(SBT)
                    continue
                elseif !ismutabletype(SBT)
                    prev[k] = make_zero_immutable!(v, seen)
                else
                    $sym(v, seen)
                end
            end
            return nothing
        end

        @inline function $sym(prev::MakeZeroShared, seen::ST)::Nothing where {ST}
            return nothing
        end

        @inline $sym(prev) = $sym(prev, Base.IdSet())

        # A tuple or named tuple of Arrays and `Ref`s of floating-point numbers
        # (and integers) needs no `seen` set (`make_zero_flat_leaves`).
        @inline function $sym(prev::T)::Nothing where {T <: Union{Tuple, NamedTuple}}
            if make_zero_flat_leaves(T, false) >= 0
                return make_zero_flat!(prev)
            end
            $sym(prev, Base.IdSet())
            return nothing
        end
    end
end

@register_make_zero_inplace(Enzyme.make_zero!)
@register_make_zero_inplace(Enzyme.remake_zero!)

@inline function EnzymeCore.make_zero!(prev::T, seen::S)::Nothing where {T, S}
    if guaranteed_const(T)
        return nothing
    end
    if prev in seen
        return nothing
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    push!(seen, prev)
    ntuple(Val(nf)) do i
        @inline
        if isdefined(prev, i)
            xi = getfield(prev, i)
            SBT = Core.Typeof(xi)
            activitystate = active_reg_nothrow(SBT)
            if activitystate == AnyState  # guaranteed_const
            elseif ismutabletype(T) && !ismutabletype(SBT)
                yi = make_zero_immutable!(xi, seen)
                make_zero_setfield!(prev, i, yi)
            elseif activitystate == DupState
                EnzymeCore.make_zero!(xi, seen)
            else
                msg = "cannot set $xi to zero in-place, as it contains differentiable values in immutable positions\nIf the argument is known to have all immutable positions already zero (e.g. was the result of Enzyme.make_zero), Enzyme.remake_zero! will skip this error check."
                throw(ArgumentError(msg))
            end
        end
        return nothing
    end
    return nothing
end

@inline function EnzymeCore.remake_zero!(prev::T, seen::S)::Nothing where {T, S}
    if guaranteed_const(T)
        return nothing
    end
    if prev in seen
        return nothing
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    push!(seen, prev)
    ntuple(Val(nf)) do i
        @inline
        if isdefined(prev, i)
            xi = getfield(prev, i)
            SBT = Core.Typeof(xi)
            activitystate = active_reg_nothrow(SBT)
            if activitystate == AnyState  # guaranteed_const
            elseif ismutabletype(T) && !ismutabletype(SBT)
                yi = make_zero_immutable!(xi, seen)
                make_zero_setfield!(prev, i, yi)
            elseif activitystate == DupState
                EnzymeCore.make_zero!(xi, seen)
            elseif activitystate == MixedState
                EnzymeCore.remake_zero!(xi, seen)
            end
        end
        return nothing
    end
    return nothing
end
