
function referenceCaller(fn::Ref{Clos}, args...) where {Clos}
    fval = fn[]
    fval = fval::Clos
    return fval(args...)
end

# The derivative of a `Threads.@threads` loop runs the derivative of the loop's body on
# every thread. `ForwardPFor`, `AugmentedPFor` and `ReversePFor` are the callables it hands
# to `threading_run` for that, one per derivative of the body.
#
# `thunk` holds the body's derivative by its address, which the rule that emits the loop's
# derivative takes from the module it compiles (`threadsfor_common`). Code that runs calls
# it through that address. An outer derivative of such code, as in Forward over Reverse,
# differentiates the callable itself, and a call through an address it does not know has no
# derivative. So the rule of the outer derivative differentiates the callable with
# `Rederived = true` instead (`rederived_body`): the same data, called by a method that
# asks `thunk` for the body's derivative. That one is a constant where it is compiled, so
# the outer derivative takes its code up like that of any other nested derivative.
#
# `S` is `(ModifiedBetween, RuntimeActivity, StrongZero)`: the settings of the body's
# derivative that the type of its thunk does not carry. Its `AtomicAdd` setting is the
# default, as for every loop body (`threadsfor_common`). The body runs on every thread, and
# all of them add into the same shadows, so the number of threads decides how it
# accumulates. A package image is written with one thread for sessions with any number: the
# derivative it keeps for sessions with several takes the body's derivative up as such a
# session compiles it (see `Compiler.ForSeveralThreads`), like everything else that is
# linked into it.
struct ForwardPFor{ThunkTy, FT, S, Rederived}
    thunk::ThunkTy
    ft::FT
end

struct AugmentedPFor{ThunkTy, FT, AnyJL, byRef, TT, S, Rederived}
    thunk::ThunkTy
    ft::FT
    tapes::TT
end

struct ReversePFor{ThunkTy, FT, AnyJL, byRef, TT, S, Rederived}
    thunk::ThunkTy
    ft::FT
    tapes::TT
end

"""
    rederived_body(T)

The type an outer derivative differentiates in place of the loop body `T` of a
`Threads.@threads` loop: for one of the callables Enzyme's own derivative of such a loop
runs (`ForwardPFor`, `AugmentedPFor`, `ReversePFor`), the same type with `Rederived = true`,
whose fields and layout are the same; for any other type, `T` itself.
"""
rederived_body(@nospecialize(T::Type)) = T
rederived_body(::Type{ForwardPFor{ThunkTy, FT, S, false}}) where {ThunkTy, FT, S} =
    ForwardPFor{ThunkTy, FT, S, true}
rederived_body(::Type{AugmentedPFor{ThunkTy, FT, AnyJL, byRef, TT, S, false}}) where {ThunkTy, FT, AnyJL, byRef, TT, S} =
    AugmentedPFor{ThunkTy, FT, AnyJL, byRef, TT, S, true}
rederived_body(::Type{ReversePFor{ThunkTy, FT, AnyJL, byRef, TT, S, false}}) where {ThunkTy, FT, AnyJL, byRef, TT, S} =
    ReversePFor{ThunkTy, FT, AnyJL, byRef, TT, S, true}

# The forward derivative of a loop body, for the thunk type and settings the loop's
# derivative holds it with.
@inline function rederived_forward(
        ::Type{<:ForwardModeThunk{<:Any, FA, <:Any, TT, Width, ReturnPrimal}}, ::Val{S},
    ) where {FA, TT, Width, ReturnPrimal, S}
    return thunk(
        Val(0),
        FA,
        Const,
        TT,
        Val(API.DEM_ForwardMode),
        Val(Width),
        Val(S[1]),
        Val(ReturnPrimal),
        Val(false),
        FFIABI,
        Val(false),
        Val(S[2]),
        Val(S[3]),
        Val(nothing),
    )
end

# The augmented forward and the reverse derivative of a loop body, likewise. Both callables
# of one loop get them from this one compilation, so the tape the first writes is the tape
# the second reads. The type of an adjoint thunk does not say whether the augmented forward
# pass compiled with it returns the primal result, so the caller does.
@inline function rederived_split(
        ::Type{<:Union{AugmentedForwardThunk{<:Any, FA, <:Any, TT, Width}, AdjointThunk{<:Any, FA, <:Any, TT, Width}}},
        ::Val{S},
        ::Val{ReturnPrimal} = Val(false),
    ) where {FA, TT, Width, S, ReturnPrimal}
    return thunk(
        Val(0),
        FA,
        Const,
        TT,
        Val(API.DEM_ReverseModePrimal),
        Val(Width),
        Val(S[1]),
        Val(ReturnPrimal),
        Val(false),
        FFIABI,
        Val(false),
        Val(S[2]),
        Val(S[3]),
        Val(nothing),
    )
end

# The tapes of a loop are stored as the loop's derivative allocated them, for the tape
# type of the body's derivative it was compiled with, and so is the tape of a task
# (`AugmentedTask`). A rederived body has to write that same type.
@inline rederived_tape_check(
    ::AbstractThunk, ::Type{<:Union{Vector{TapeType}, Ptr{TapeType}, Base.RefValue{TapeType}}},
) where {TapeType} = throw(
    AssertionError(
        "The derivative of the body of a Threads.@threads loop or of a task, compiled again for an outer derivative, keeps another tape than the $TapeType the first derivative stores",
    ),
)
@inline rederived_tape_check(
    ::Union{
        AugmentedForwardThunk{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, TapeType},
        AdjointThunk{<:Any, <:Any, <:Any, <:Any, <:Any, TapeType},
    },
    ::Type{<:Union{Vector{TapeType}, Ptr{TapeType}, Base.RefValue{TapeType}}},
) where {TapeType} = nothing

@inline function pfor_forward(thunk, ft, tid_args...)
    if length(tid_args) == 0
        thunk(ft)
    else
        thunk(ft, Const(tid_args[1]))
    end
    return nothing
end

@inline function pfor_augmented(thunk, ft, tapes, ::Val{AnyJL}, ::Val{byRef}, tid) where {AnyJL, byRef}
    tres = if byRef
        thunk(Const(referenceCaller), ft, Const(tid))
    else
        thunk(ft, Const(tid))
    end

    if !AnyJL
        unsafe_store!(tapes, tres[1], tid)
    else
        @inbounds tapes[tid] = tres[1]
    end
    return nothing
end

@inline function pfor_reverse(thunk, ft, tapes, ::Val{AnyJL}, ::Val{byRef}, tid) where {AnyJL, byRef}
    tres = if !AnyJL
        unsafe_load(tapes, tid)
    else
        @inbounds tapes[tid]
    end

    if byRef
        thunk(Const(referenceCaller), ft, Const(tid), tres)
    else
        thunk(ft, Const(tid), tres)
    end
    return nothing
end

function (st::ForwardPFor{ThunkTy, FT, S, false})(tid_args...) where {ThunkTy, FT, S}
    return pfor_forward(st.thunk, st.ft, tid_args...)
end

function (st::ForwardPFor{ThunkTy, FT, S, true})(tid_args...) where {ThunkTy, FT, S}
    return pfor_forward(rederived_forward(ThunkTy, Val(S)), st.ft, tid_args...)
end

function (st::AugmentedPFor{ThunkTy, FT, AnyJL, byRef, TT, S, false})(tid) where {ThunkTy, FT, AnyJL, byRef, TT, S}
    return pfor_augmented(st.thunk, st.ft, st.tapes, Val(AnyJL), Val(byRef), tid)
end

function (st::AugmentedPFor{ThunkTy, FT, AnyJL, byRef, TT, S, true})(tid) where {ThunkTy, FT, AnyJL, byRef, TT, S}
    augmented = rederived_split(ThunkTy, Val(S))[1]
    rederived_tape_check(augmented, TT)
    return pfor_augmented(augmented, st.ft, st.tapes, Val(AnyJL), Val(byRef), tid)
end

function (st::ReversePFor{ThunkTy, FT, AnyJL, byRef, TT, S, false})(tid) where {ThunkTy, FT, AnyJL, byRef, TT, S}
    return pfor_reverse(st.thunk, st.ft, st.tapes, Val(AnyJL), Val(byRef), tid)
end

function (st::ReversePFor{ThunkTy, FT, AnyJL, byRef, TT, S, true})(tid) where {ThunkTy, FT, AnyJL, byRef, TT, S}
    adjoint = rederived_split(ThunkTy, Val(S))[2]
    rederived_tape_check(adjoint, TT)
    return pfor_reverse(adjoint, st.ft, st.tapes, Val(AnyJL), Val(byRef), tid)
end

function runtime_pfor_fwd(
        thunk::ThunkTy,
        ft::FT,
        ::Val{S},
        threading_args...,
    )::Cvoid where {ThunkTy, FT, S}
    Base.Threads.threading_run(ForwardPFor{ThunkTy, FT, S, false}(thunk, ft), threading_args...)
    return
end

function runtime_pfor_augfwd(
        thunk::ThunkTy,
        ft::FT,
        ::Val{AnyJL},
        ::Val{byRef},
        ::Val{S},
        threading_args...,
    ) where {ThunkTy, FT, AnyJL, byRef, S}
    TapeType = EnzymeRules.tape_type(ThunkTy)

    n = Base.Threads.threadpoolsize()
    tapes = if AnyJL
        Vector{TapeType}(undef, n)
    else
        Base.unsafe_convert(
            Ptr{TapeType},
            Libc.malloc(sizeof(TapeType) * n),
        )
    end

    Base.Threads.threading_run(
        AugmentedPFor{ThunkTy, FT, AnyJL, byRef, typeof(tapes), S, false}(thunk, ft, tapes),
        threading_args...,
    )
    return tapes
end

function runtime_pfor_rev(
        thunk::ThunkTy,
        ft::FT,
        ::Val{AnyJL},
        ::Val{byRef},
        ::Val{S},
        tapes,
        threading_args...,
    ) where {ThunkTy, FT, AnyJL, byRef, S}
    Base.Threads.threading_run(
        ReversePFor{ThunkTy, FT, AnyJL, byRef, typeof(tapes), S, false}(thunk, ft, tapes),
        threading_args...,
    )
    if !AnyJL
        Libc.free(tapes)
    end
    return nothing
end

# The derivative of a task (`Threads.@spawn`) starts tasks that run the derivative of the
# task's body, which takes no arguments: one for its forward derivative, or one for its
# augmented forward pass and one for its reverse pass, which the derivative's reverse pass
# runs. `ForwardTask`, `AugmentedTask` and `ReverseTask` are the callables it starts them
# with. Like those of a `Threads.@threads` loop (see `ForwardPFor`), they hold the body's
# derivative by its address, so an outer derivative differentiates them with
# `Rederived = true` (`rederived_task`), whose methods ask `thunk` for it. `S` holds the
# same settings as for a loop. The body's derivative returns its primal result.
struct ForwardTask{ThunkTy, FT, S, Rederived}
    thunk::ThunkTy
    ft::FT
end

# `tape` holds the tape that the augmented forward pass writes and the reverse pass reads.
struct AugmentedTask{ThunkTy, FT, TapeRef, S, Rederived}
    thunk::ThunkTy
    ft::FT
    tape::TapeRef
end

struct ReverseTask{ThunkTy, FT, TapeRef, S, Rederived}
    thunk::ThunkTy
    ft::FT
    tape::TapeRef
end

# A task whose body has no derivative holds a `PrimalErrorThunk`, which is not rederived.
rederived_body(::Type{ForwardTask{ThunkTy, FT, S, false}}) where {ThunkTy <: ForwardModeThunk, FT, S} =
    ForwardTask{ThunkTy, FT, S, true}
rederived_body(::Type{AugmentedTask{ThunkTy, FT, TapeRef, S, false}}) where {ThunkTy <: AugmentedForwardThunk, FT, TapeRef, S} =
    AugmentedTask{ThunkTy, FT, TapeRef, S, true}
rederived_body(::Type{ReverseTask{ThunkTy, FT, TapeRef, S, false}}) where {ThunkTy <: AdjointThunk, FT, TapeRef, S} =
    ReverseTask{ThunkTy, FT, TapeRef, S, true}

@inline task_fields(st::ForwardTask) = (st.thunk, st.ft)
@inline task_fields(st::Union{AugmentedTask, ReverseTask}) = (st.thunk, st.ft, st.tape)

"""
    rederived_task(fn, dfn)

The callable that an outer derivative starts a task with, and its shadow, in place of the
callable `fn` of a task and its shadow `dfn`: for one of the callables Enzyme's own
derivative of a task starts (`ForwardTask`, `AugmentedTask`, `ReverseTask`), the same data
in the type `rederived_body` gives; for anything else, `fn` and `dfn` themselves.
"""
rederived_task(fn, dfn) = (fn, dfn)
function rederived_task(fn::T, dfn::T) where {T <: Union{ForwardTask, AugmentedTask, ReverseTask}}
    R = rederived_body(T)
    R === T && return (fn, dfn)
    return (R(task_fields(fn)...), R(task_fields(dfn)...))
end

# The derivative of a task's body is compiled with `ReturnPrimal` (`runtime_newtask_augfwd`).
@inline task_split(::Type{ThunkTy}, ::Val{S}) where {ThunkTy, S} = rederived_split(ThunkTy, Val(S), Val(true))

@inline task_forward(thunk, ft) = thunk(ft)[1]

@inline function task_augmented(thunk, ft, tape)
    res = thunk(ft)
    tape[] = res[1]
    return res[2]
end

@inline function task_reverse(thunk, ft, tape)
    thunk(ft, tape[])
    return 0
end

function (st::ForwardTask{ThunkTy, FT, S, false})() where {ThunkTy, FT, S}
    return task_forward(st.thunk, st.ft)
end

function (st::ForwardTask{ThunkTy, FT, S, true})() where {ThunkTy, FT, S}
    return task_forward(rederived_forward(ThunkTy, Val(S)), st.ft)
end

function (st::AugmentedTask{ThunkTy, FT, TapeRef, S, false})() where {ThunkTy, FT, TapeRef, S}
    return task_augmented(st.thunk, st.ft, st.tape)
end

function (st::AugmentedTask{ThunkTy, FT, TapeRef, S, true})() where {ThunkTy, FT, TapeRef, S}
    augmented = task_split(ThunkTy, Val(S))[1]
    rederived_tape_check(augmented, TapeRef)
    return task_augmented(augmented, st.ft, st.tape)
end

function (st::ReverseTask{ThunkTy, FT, TapeRef, S, false})() where {ThunkTy, FT, TapeRef, S}
    return task_reverse(st.thunk, st.ft, st.tape)
end

function (st::ReverseTask{ThunkTy, FT, TapeRef, S, true})() where {ThunkTy, FT, TapeRef, S}
    adjoint = task_split(ThunkTy, Val(S))[2]
    rederived_tape_check(adjoint, TapeRef)
    return task_reverse(adjoint, st.ft, st.tape)
end

function runtime_newtask_fwd(
        fn::FT1,
        dfn::FT2,
        post::Any,
        ssize::Int,
        runtimeActivity::Val{RuntimeActivity},
        strongZero::Val{StrongZero},
        ::Val{width},
    ) where {FT1, FT2, width, RuntimeActivity, StrongZero}
    tfn, tdfn = rederived_task(fn, dfn)
    FT = Core.Typeof(tfn)
    ghos = guaranteed_const(FT)
    forward = thunk(
        Val(0),
        (ghos ? Const : Duplicated){FT},
        Const,
        Tuple{},
        Val(API.DEM_ForwardMode),
        Val(Int(width)),
        Val((false,)),
        Val(true),
        Val(false),
        FFIABI,
        Val(false),
        runtimeActivity,
        strongZero
    ) #=erriffuncwritten=#
    ft = ghos ? Const(tfn) : Duplicated(tfn, tdfn)
    S = ((false,), RuntimeActivity, StrongZero)
    st = ForwardTask{Core.Typeof(forward), Core.Typeof(ft), S, false}(forward, ft)
    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), st, post, ssize)
end

struct Return2
    ret1::Any
    ret2::Any
end

function runtime_newtask_augfwd(
        fn::FT1,
        dfn::FT2,
        post::Any,
        ssize::Int,
        runtimeActivity::Val{RuntimeActivity},
        strongZero::Val{StrongZero},
        ::Val{width},
        ::Val{ModifiedBetween},
    ) where {FT1, FT2, width, ModifiedBetween, RuntimeActivity, StrongZero}
    # TODO make this AD subcall type stable
    tfn, tdfn = rederived_task(fn, dfn)
    FT = Core.Typeof(tfn)
    ghos = guaranteed_const(FT)
    FA = if ghos
        Const{FT}
    elseif tdfn isa Base.RefValue && !(tfn isa Base.RefValue)
        MixedDuplicated{FT}
    else
        Duplicated{FT}
    end
    forward, adjoint = runtime_thunk(
        Val(0),
        FA,
        Const,
        Tuple{},
        Val(API.DEM_ReverseModePrimal),
        Val(Int(width)),
        Val(ModifiedBetween),
        Val(true),
        Val(false),
        FFIABI,
        Val(false),
        runtimeActivity,
        strongZero
    ) #=erriffuncwritten=#
    ft = if ghos
        Const(tfn)
    elseif tdfn isa Base.RefValue && !(tfn isa Base.RefValue)
        MixedDuplicated(tfn, tdfn)
    else
        Duplicated(tfn, tdfn)
    end
    return newtask_split(forward, adjoint, ft, post, ssize, Val((ModifiedBetween, RuntimeActivity, StrongZero)))
end

# The tape of a task's body, which `runtime_thunk` only knows once it has compiled the body's
# derivative, so `runtime_newtask_augfwd` gets the thunks untyped and calls this with them.
task_tape_type(::Type{T}) where {T <: AugmentedForwardThunk} = EnzymeRules.tape_type(T)
task_tape_type(::Type) = Any

function newtask_split(
        forward::AugT, adjoint::AdjT, ft::FT, post::Any, ssize::Int, ::Val{S},
    ) where {AugT, AdjT, FT, S}
    tape = Ref{task_tape_type(AugT)}()
    TapeRef = Core.Typeof(tape)
    ftask = ccall(
        :jl_new_task, Ref{Task}, (Any, Any, Int),
        AugmentedTask{AugT, FT, TapeRef, S, false}(forward, ft, tape), post, ssize,
    )
    rtask = ccall(
        :jl_new_task, Ref{Task}, (Any, Any, Int),
        ReverseTask{AdjT, FT, TapeRef, S, false}(adjoint, ft, tape), post, ssize,
    )
    return Return2(ftask, rtask)
end

@inline function threadsfor_common(orig, gutils, B, mode, enzyme_ctx, tape = nothing)

    mod = orig.parent.parent.parent

    llvmfn = orig.called_operand
    mi = nothing
    fwdmodenm = nothing
    augfwdnm = nothing
    adjointnm = nothing
    TapeType = nothing
    attributes = llvmfn.function_attributes
    for fattr in collect(attributes)
        if isa(fattr, LLVM.StringAttribute)
            if fattr.kind == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if fattr.kind == "enzymejl_tapetype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                TapeType = Base.unsafe_pointer_to_objref(ptr)
            end
            if fattr.kind == "enzymejl_forward"
                fwdmodenm = fattr.value
            end
            if fattr.kind == "enzymejl_augforward"
                augfwdnm = fattr.value
            end
            if fattr.kind == "enzymejl_adjoint"
                adjointnm = fattr.value
            end
        end
    end

    funcT = mi.specTypes.parameters[2]

    # Since JuliaLang/julia#63184, `threading_run` does not specialize on the loop
    # body, so the body is passed boxed and its type is only known at the call site.
    # One `threading_run` is then shared by every loop, so the derivatives compiled
    # for this loop are recorded on the call rather than on the function.
    boxed = funcT === Any
    if boxed
        legal, funcT, _ = abs_typeof(orig.operands[1], enzyme_ctx)
        if !legal
            throw(AssertionError("Could not determine the type of the Threads.@threads loop body $(string(orig.operands[1]))"))
        end
        fwdmodenm = nothing
        augfwdnm = nothing
        adjointnm = nothing
        TapeType = nothing
        attributes = orig.function_attributes
        for fattr in collect(attributes)
            if isa(fattr, LLVM.StringAttribute)
                if fattr.kind == "enzymejl_tapetype"
                    ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                    TapeType = Base.unsafe_pointer_to_objref(ptr)
                end
                if fattr.kind == "enzymejl_forward"
                    fwdmodenm = fattr.value
                end
                if fattr.kind == "enzymejl_augforward"
                    augfwdnm = fattr.value
                end
                if fattr.kind == "enzymejl_adjoint"
                    adjointnm = fattr.value
                end
            end
        end
    end

    # Enzyme's own derivative of a loop runs a callable that reaches the body's derivative
    # through an address. Differentiating that callable takes the one that asks for the
    # body's derivative instead (see `ForwardPFor`).
    rederived = rederived_body(funcT) !== funcT
    funcT = rederived_body(funcT)

    # TODO actually do modifiedBetween
    e_tt = Tuple{Const{Int}}
    modifiedBetween = (mode != API.DEM_ForwardMode, false)

    world = enzyme_world()

    pfuncT = funcT

    mi2 = my_methodinstance(mode == API.DEM_ForwardMode ? Forward : Reverse, funcT, Tuple{map(eltype, e_tt.parameters)...}, world)
    @assert mi2 !== nothing

    refed = false

    # TODO: Clean this up and add to `nested_codegen!` asa feature
    width = Int(get_width(gutils))

    dupClosure = !guaranteed_const_nongen(funcT, world)
    if dupClosure
        if is_constant_value(gutils, orig.operands[1])
            dupClosure = false
            if !boxed && inline_roots_type(funcT) != 0
                if !is_constant_value(gutils, orig.operands[2])
                    dupClosure = true
                end
            end
        end
    end

    pdupClosure = dupClosure

    subfunc = nothing

    dFT = (dupClosure ? (width == 1 ? Duplicated : (BatchDuplicated{T, Int(width)} where {T})) : Const){funcT}

    if mode == API.DEM_ForwardMode
        if fwdmodenm === nothing
            etarget = Compiler.EnzymeTarget()
            eparams = Compiler.EnzymeCompilerParams(
                Tuple{dFT, e_tt.parameters...},
                API.DEM_ForwardMode,
                width,
                Const{Nothing},
                true,
                true,
                modifiedBetween,
                false,
                false,
                UnknownTapeType,
                FFIABI,
                false,
                get_runtime_activity(gutils),
                get_strong_zero(gutils),
                Compiler.linked_default_atomic_add(enzyme_ctx),
            ) #=ErrIfFuncWritten=#
            ejob = Compiler.CompilerJob(
                mi2,
                CompilerConfig(etarget, eparams; kernel = false),
                world,
            )

            cmod, edges, fwdmodenm, _, _, _, value_table = _thunk(ejob, false) #=postopt=#

            LLVM.link!(mod, cmod)
            merge_julia_value_table!(enzyme_ctx, value_table)

            push!(attributes, StringAttribute("enzymejl_forward", fwdmodenm))
            push!(
                mod.functions[fwdmodenm].function_attributes,
                EnumAttribute(:alwaysinline),
            )
            permit_inlining!(mod.functions[fwdmodenm])
        end
        thunkTy = ForwardModeThunk{
            Ptr{Cvoid},
            dFT,
            Const{Nothing},
            e_tt,
            width,
            false,
        }  #=returnPrimal=#
        subfunc = mod.functions[fwdmodenm]

    elseif mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        if dupClosure
            if !guaranteed_nonactive(funcT, world)
                refed = true
                e_tt = Tuple{width == 1 ? Duplicated{Base.RefValue{funcT}} : BatchDuplicated{Base.RefValue{funcT}, Int(width)}, e_tt.parameters...}
                funcT = Core.Typeof(referenceCaller)
                dupClosure = false
                modifiedBetween = (false, modifiedBetween...)
                mi2 = my_methodinstance(mode == API.DEM_ForwardMode ? Forward : Reverse, funcT, Tuple{map(eltype, e_tt.parameters)...}, world)
                @assert mi2 !== nothing
                dFT = (dupClosure ? (width == 1 ? Duplicated : (BatchDuplicated{T, Int(width)} where {T})) : Const){funcT}
            end
        end

        if augfwdnm === nothing || adjointnm === nothing
            etarget = Compiler.EnzymeTarget()
            # TODO modifiedBetween
            eparams = Compiler.EnzymeCompilerParams(
                Tuple{dFT, e_tt.parameters...},
                API.DEM_ReverseModePrimal,
                width,
                Const{Nothing},
                true,
                true,
                modifiedBetween,
                false,
                false,
                UnknownTapeType,
                FFIABI,
                false,
                get_runtime_activity(gutils),
                get_strong_zero(gutils),
                Compiler.linked_default_atomic_add(enzyme_ctx),
            ) #=ErrIfFuncWritten=#
            ejob = Compiler.CompilerJob(
                mi2,
                CompilerConfig(etarget, eparams; kernel = false),
                world,
            )

            cmod, edges, adjointnm, augfwdnm, TapeType, _, value_table = _thunk(ejob, false) #=postopt=#

            LLVM.link!(mod, cmod)
            merge_julia_value_table!(enzyme_ctx, value_table)

            push!(attributes, StringAttribute("enzymejl_augforward", augfwdnm))
            push!(
                mod.functions[augfwdnm].function_attributes,
                EnumAttribute(:alwaysinline),
            )
            permit_inlining!(mod.functions[augfwdnm])

            push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm))
            push!(
                mod.functions[adjointnm].function_attributes,
                EnumAttribute(:alwaysinline),
            )
            permit_inlining!(mod.functions[adjointnm])

            push!(
                attributes,
                StringAttribute(
                    "enzymejl_tapetype",
                    string(convert(UInt, unsafe_to_pointer(TapeType))),
                ),
            )

        end

        if mode == API.DEM_ReverseModePrimal
            thunkTy = AugmentedForwardThunk{
                Ptr{Cvoid},
                dFT,
                Const{Nothing},
                e_tt,
                width,
                true,
                TapeType,
            } #=returnPrimal=#
            subfunc = mod.functions[augfwdnm]
        else
            thunkTy = AdjointThunk{
                Ptr{Cvoid},
                dFT,
                Const{Nothing},
                e_tt,
                width,
                TapeType,
            }
            subfunc = mod.functions[adjointnm]
        end
    else
        @assert "Unknown mode"
    end

    ppfuncT = pfuncT
    dpfuncT = width == 1 ? pfuncT : NTuple{Int(width), pfuncT}

    if refed
        dpfuncT = Base.RefValue{dpfuncT}
        pfuncT = Base.RefValue{pfuncT}
    end

    dfuncT = pfuncT
    if pdupClosure
        if width == 1
            dfuncT = Duplicated{dfuncT}
        else
            dfuncT = BatchDuplicated{dfuncT, Int(width)}
        end
    else
        dfuncT = Const{dfuncT}
    end

    vals = LLVM.Value[]

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.at_end(LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils))))
    ll_th = convert(LLVMType, thunkTy)
    al = alloca!(alloctx, ll_th)
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Tracked))
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Derived))
    push!(vals, al)
    @assert inline_roots_type(thunkTy) == 0

    copies = Tuple{LLVM.Value, LLVM.Value, LLVM.LLVMType}[]
    if !isghostty(dfuncT)

        llty = convert(LLVMType, dfuncT)

        num_arg_roots = inline_roots_type(llty)

        alloctx = LLVM.IRBuilder()
        position!(alloctx, LLVM.at_end(LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils))))

        llty_foralloca = if VERSION >= v"1.12" && num_arg_roots != 0
            strip_tracked_pointers(llty)
        else
            llty
        end

        al = alloca!(alloctx, llty_foralloca)
        al2 = if num_arg_roots != 0
            create_rooted_array(alloctx, num_arg_roots)
        end

        if !isghostty(ppfuncT)
            v = new_from_original(gutils, orig.operands[1])
            pllty = convert(LLVMType, ppfuncT)

            pv = nothing

            fwdbuilder = if mode == API.DEM_ReverseModeGradient
                B2 = LLVM.IRBuilder()
                position!(B2, LLVM.before(new_from_original(gutils, orig)))
                B2
            else
                B
            end

            if v.value_type != pllty
                if boxed
                    v = addrspacecast!(fwdbuilder, v, LLVM.PointerType(pllty, Derived))
                end
                pv = v
                v = load!(fwdbuilder, pllty, v)
            end

            if !boxed && inline_roots_type(ppfuncT) != 0
                v2 = new_from_original(gutils, orig.operands[2])
                v = recombine_value!(fwdbuilder, v, v2)
            end

            if mode == API.DEM_ReverseModeGradient
                v = lookup_value(gutils, v, B)
            end

        else
            v = makeInstanceOf(B, ppfuncT)
        end

        if refed
            val0 = val = emit_allocobj!(B, pfuncT)
            val = bitcast!(B, val, LLVM.PointerType(pllty, val.value_type.addrspace))
            val = addrspacecast!(B, val, LLVM.PointerType(pllty, Derived))

            if !(pv isa Nothing)
                push!(copies, (pv, val, pllty))
            end

            store!(B, v, val)

            if any_jltypes(pllty)
                emit_writebarrier!(B, get_julia_inner_types(B, val0, v))
            end
        else
            val0 = v
        end

        ptr = inbounds_gep!(
            B,
            llty,
            al,
            [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)],
        )

        if al2 !== nothing
            extract_roots_from_value!(B, val0, al2)
            T_jlvalue = LLVM.StructType(LLVMType[])
            T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
            al3 = gep!(B, T_prjlvalue, al2, LLVM.Value[ConstantInt(CountTrackedPointers(val0.value_type).count)])
        end

        store!(B, val0, ptr)

        if pdupClosure

            if !isghostty(ppfuncT)
                dv = invert_pointer(gutils, orig.operands[1], B)

                fwdbuilder = if mode == API.DEM_ReverseModeGradient
                    B2 = LLVM.IRBuilder()
                    position!(B2, LLVM.before(new_from_original(gutils, orig)))
                    B2
                else
                    B
                end

                spllty = LLVM.LLVMType(API.EnzymeGetShadowType(width, pllty))
                pv = nothing

                dv2 = if !boxed && inline_roots_type(ppfuncT) != 0
                    invert_pointer(gutils, orig.operands[2], B)
                end

                # The closure's tracked pointers live in the roots array, so its bits
                # fields alone can make it inactive. Its shadow is then the primal
                # itself, so in forward mode start from zero rather than copying the
                # primal floats.
                const_bits = mode == API.DEM_ForwardMode && dv2 !== nothing &&
                    is_constant_value(gutils, orig.operands[1])

                if const_bits
                    if width == 1
                        dv = recombine_value!(fwdbuilder, LLVM.null(spllty), dv2)
                    else
                        shadowres = UndefValue(spllty)
                        for idx in 1:width
                            arg2 = extract_value!(fwdbuilder, dv2, idx - 1)
                            arg = recombine_value!(fwdbuilder, LLVM.null(pllty), arg2)
                            shadowres = insert_value!(fwdbuilder, shadowres, arg, idx - 1)
                        end
                        dv = shadowres
                    end
                elseif dv.value_type != spllty
                    pv = dv
                    if width == 1
                        if boxed
                            dv = addrspacecast!(fwdbuilder, dv, LLVM.PointerType(spllty, Derived))
                            pv = dv
                        end
                        dv = load!(fwdbuilder, spllty, dv)
                        if dv2 !== nothing
                            dv = recombine_value!(fwdbuilder, dv, dv2)
                        end
                    else
                        shadowres = UndefValue(spllty)
                        for idx in 1:width
                            arg = extract_value!(fwdbuilder, dv, idx - 1)
                            if boxed
                                arg = addrspacecast!(fwdbuilder, arg, LLVM.PointerType(pllty, Derived))
                            end
                            arg = load!(fwdbuilder, pllty, arg)
                            if dv2 !== nothing
                                arg2 = extract_value!(fwdbuilder, dv2, idx - 1)
                                arg = recombine_value!(fwdbuilder, arg, arg2)
                            end
                            shadowres = insert_value!(fwdbuilder, shadowres, arg, idx - 1)
                        end
                        dv = shadowres
                    end
                end

                if mode == API.DEM_ReverseModeGradient
                    dv = lookup_value(gutils, dv, B)
                end
            else
                @assert false
            end

            if refed
                dval0 = dval = emit_allocobj!(B, dpfuncT)
                dval =
                    bitcast!(B, dval, LLVM.PointerType(spllty, dval.value_type.addrspace))
                dval = addrspacecast!(B, dval, LLVM.PointerType(spllty, Derived))
                store!(B, dv, dval)
                if any_jltypes(spllty)
                    emit_writebarrier!(B, get_julia_inner_types(B, dval0, dv))
                end
                pvl = lookup_value(gutils, pv, B)
                if mode == API.DEM_ReverseModeGradient
                    if width == 1
                        copy_floats_into!(B, spllty, dval, pvl)
                    else
                        for idx in 1:width
                            arg = extract_value!(B, pvl, idx - 1)
                            g0 = inbounds_gep!(B, spllty, dval, LLVM.Value[LLVM.ConstantInt(Int64(0)), LLVM.ConstantInt(Int32(idx - 1))])
                            copy_floats_into!(B, pllty, g0, arg)
                        end
                    end
                end
                if pv !== nothing
                    push!(copies, (pv, dval, spllty))
                end
            else
                dval0 = dv
            end

            dptr = inbounds_gep!(
                B,
                llty,
                al,
                [
                    LLVM.ConstantInt(LLVM.IntType(64), 0),
                    LLVM.ConstantInt(LLVM.IntType(32), 1),
                ],
            )

            if al2 !== nothing
                extract_roots_from_value!(B, dval0, al3)
            end
            store!(B, dval0, dptr)
        end

        al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

        push!(vals, al)

        if num_arg_roots != 0
            push!(vals, al2)

        end
    end

    if tape !== nothing
        push!(vals, tape)
    end

    push!(vals, new_from_original(gutils, orig.arguments[end]))

    # What the body's derivative was compiled with beyond what `thunkTy` says (see `ForwardPFor`).
    settings = (modifiedBetween, get_runtime_activity(gutils), get_strong_zero(gutils))

    return refed, subfunc.name, dfuncT, vals, thunkTy, TapeType, copies, settings
end

@register_fwd function threadsfor_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    mod = orig.parent.parent.parent

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    _, sname, dfuncT, vals, thunkTy, _, _, settings =
        threadsfor_common(orig, gutils, B, API.DEM_ForwardMode, enzyme_context())

    tt = Tuple{thunkTy, dfuncT, Val{settings}, Bool}
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_fwd, tt)
    push!(entry.function_attributes, EnumAttribute(:alwaysinline))

    pval = mod.functions[sname]
    if VERSION < v"1.12"
        pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(pval.value_type, [pval])
    store!(B, pval, vals[1])

    cal = LLVM.call!(B, entry.function_type, entry, vals)
    debug_from_orig!(gutils, cal, orig)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeReplaceOriginalToNew(gutils, orig, cal)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end
    return false
end

@register_aug function threadsfor_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    mod = orig.parent.parent.parent

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    byRef, sname, dfuncT, vals, thunkTy, _, copies, settings =
        threadsfor_common(orig, gutils, B, API.DEM_ReverseModePrimal, enzyme_context())

    tt = Tuple{
        thunkTy,
        dfuncT,
        Val{any_jltypes(EnzymeRules.tape_type(thunkTy))},
        Val{byRef},
        Val{settings},
        Bool,
    }
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_augfwd, tt)
    push!(entry.function_attributes, EnumAttribute(:alwaysinline))

    pval = mod.functions[sname]
    if VERSION < v"1.12"
        pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(pval.value_type, [pval])
    store!(B, pval, vals[1])

    tape = LLVM.call!(B, entry.function_type, entry, vals)
    debug_from_orig!(gutils, tape, orig)

    if !any_jltypes(EnzymeRules.tape_type(thunkTy))
        if tape.value_type != convert(LLVMType, Ptr{Cvoid})
            tape = LLVM.ConstantInt(0)
            @safe_warn "Illegal calling convention for threadsfor augfwd"
        end
    end

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeReplaceOriginalToNew(gutils, orig, tape)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    unsafe_store!(tapeR, tape.ref)

    return false
end

@register_rev function threadsfor_rev(B, orig, gutils, tape)
    mod = orig.parent.parent.parent
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return
    end

    byRef, sname, dfuncT, vals, thunkTy, TapeType, copies, settings =
        threadsfor_common(orig, gutils, B, API.DEM_ReverseModeGradient, enzyme_context(), tape)

    STT = if !any_jltypes(TapeType)
        Ptr{TapeType}
    else
        Vector{TapeType}
    end

    tt = Tuple{
        thunkTy,
        dfuncT,
        Val{any_jltypes(EnzymeRules.tape_type(thunkTy))},
        Val{byRef},
        Val{settings},
        STT,
        Bool,
    }
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_rev, tt)
    push!(entry.function_attributes, EnumAttribute(:alwaysinline))

    pval = mod.functions[sname]
    if VERSION < v"1.12"
        pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(pval.value_type, [pval])
    store!(B, pval, vals[1])

    cal = LLVM.call!(B, entry.function_type, entry, vals)
    debug_from_orig!(gutils, cal, orig)

    for (pv, val, pllty) in copies
        ld = load!(B, pllty, val)
        pv = lookup_value(gutils, pv, B)
        store!(B, ld, pv)
    end
    return nothing
end

@register_fwd function newtask_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)
    mode = get_mode(gutils)


    vals = LLVM.Value[
        unsafe_to_llvm(B, runtime_newtask_fwd),
        new_from_original(gutils, orig.operands[1]),
        invert_pointer(gutils, orig.operands[1], B),
        new_from_original(gutils, orig.operands[2]),
        (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(
            B,
            new_from_original(gutils, orig.operands[3]),
        ),
        unsafe_to_llvm(B, Val(get_runtime_activity(gutils))),
        unsafe_to_llvm(B, Val(get_strong_zero(gutils))),
        unsafe_to_llvm(B, Val(width)),
    ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)

    # TODO: GC, ret
    if shadowR != C_NULL
        unsafe_store!(shadowR, ntask.ref)
    end

    if normalR != C_NULL
        unsafe_store!(normalR, ntask.ref)
    end

    return false
end

@register_aug function newtask_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    # fn, dfn = augmentAndGradient(fn)
    # t = jl_new_task(fn)
    # # shadow t
    # dt = jl_new_task(dfn)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    @safe_warn "active variables passed by value to jl_new_task are not yet supported"
    width = get_width(gutils)
    mode = get_mode(gutils)

    uncacheable = get_uncacheable(gutils, orig)
    ModifiedBetween = (uncacheable[1] != 0,)


    vals = LLVM.Value[
        unsafe_to_llvm(B, runtime_newtask_augfwd),
        new_from_original(gutils, orig.operands[1]),
        invert_pointer(gutils, orig.operands[1], B),
        new_from_original(gutils, orig.operands[2]),
        (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(
            B,
            new_from_original(gutils, orig.operands[3]),
        ),
        unsafe_to_llvm(B, Val(get_runtime_activity(gutils))),
        unsafe_to_llvm(B, Val(get_strong_zero(gutils))),
        unsafe_to_llvm(B, Val(width)),
        unsafe_to_llvm(B, Val(ModifiedBetween)),
    ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)
    sret = ntask

    AT = LLVM.ArrayType(T_prjlvalue, 2)
    sret = LLVM.addrspacecast!(B, sret, LLVM.PointerType(T_jlvalue, Derived))
    sret = LLVM.pointercast!(B, sret, LLVM.PointerType(AT, Derived))

    if shadowR != C_NULL
        shadow = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)]),
        )
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

@register_rev function newtask_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig.operands[1])
        return true
    end

    inv = invert_pointer(gutils, orig.operands[1], B)
    width = get_width(gutils)
    if width == 1
        nops = LLVM.Value[inv, new_from_original(gutils, orig.operands[2])]
        valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, false) #=lookup=#
        debug_from_orig!(gutils, cal, orig)
        cal.callconv = orig.callconv
    else
        for idx in 1:width
            nops = LLVM.Value[
                extract_value(B, inv, idx - 1),
                new_from_original(gutils, orig.operands[2]),
            ]
            valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, false) #=lookup=#

            debug_from_orig!(gutils, cal, orig)
            cal.callconv = orig.callconv
        end
    end

    return false
end

@register_aug function set_task_tid_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function set_task_tid_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function enq_work_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, orig.value_type)))
        for idx in 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end

    return false
end

@register_aug function enq_work_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    enq_work_fwd(B, orig, gutils, normalR, shadowR)
end

function find_match(mod, name)
    for f in mod.functions
        at = get(f.function_attributes, "enzyme_math", nothing)
        if at !== nothing && at.value == name
            return f
        end
    end
    return nothing
end

@register_rev function enq_work_rev(B, orig, gutils, tape)
    # jl_wait(shadow(t))
    origops = orig.operands
    mod = orig.parent.parent.parent
    waitfn = find_match(mod, "jl_wait")
    if waitfn === nothing
        emit_error(
            B,
            orig,
            "Enzyme: could not find jl_wait fn to create shadow of jl_enq_work",
        )
        return nothing
    end
    @assert waitfn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, waitfn.function_type, waitfn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    cal.callconv = orig.callconv
    return nothing
end

@register_fwd function wait_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, orig.value_type)))
        for idx in 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_aug function wait_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, orig.value_type)))
        for idx in 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_rev function wait_rev(B, orig, gutils, tape)
    # jl_enq_work(shadow(t))
    origops = orig.operands
    mod = orig.parent.parent.parent
    enq_work_fn = find_match(mod, "jl_enq_work")
    if enq_work_fn === nothing
        emit_error(
            B,
            orig,
            "Enzyme: could not find jl_enq_work fn to create shadow of wait",
        )
        return nothing
    end
    @assert enq_work_fn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, enq_work_fn.function_type, enq_work_fn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    cal.callconv = orig.callconv
    return nothing
end
