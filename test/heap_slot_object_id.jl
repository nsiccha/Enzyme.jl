using Enzyme, Test

# A function Enzyme differentiates as a callee, or in split mode, moves its
# stack slots to the heap. On Julia 1.10 (typed pointers) Julia passes such a
# slot to `jl_object_id_` through an addrspacecast that also changes its pointee
# type to `i8`. `objectid` of an immutable value that holds references is how
# `view` reaches it: `unalias`, `mightalias`, then Base's default `dataids`. The
# moved slot kept its own pointer type there, and the derivative failed LLVM's
# verifier ("Call parameter type does not match function signature!").

struct SlotPoint{B, A} <: AbstractVector{Float64}
    base::B
    active::A
end
Base.size(p::SlotPoint) = size(p.base)
Base.IndexStyle(::Type{<:SlotPoint}) = IndexLinear()
@inline function Base.getindex(p::SlotPoint, i::Int)
    @boundscheck checkbounds(p.base, i)
    return i <= length(p.active) ? @inbounds(p.active[i]) : @inbounds(p.base[i])
end

# Holds its arguments in one tuple, a stack slot, as ReactiveKernels' statement
# wrapper does.
@noinline @generated slot_call(f::F, args::Vararg{Any, N}) where {F, N} =
    :(f($((:(args[$i]) for i in 1:N)...)))

function slot_sumsq(v)
    t = 0.0
    for x in v
        t += x * x
    end
    return t
end

slot_window(active, base) = view(SlotPoint(base, active), 1:length(base))

slot_view_loss(active, base, rows) =
    slot_sumsq(slot_call(view, slot_window(active, base), rows))

function slot_copy_loss(active, base, rows)
    out = Vector{Float64}(undef, length(rows))
    slot_call(copyto!, out, view(slot_window(active, base), rows))
    return slot_sumsq(out)
end

# The slot in the function that is differentiated, in split mode.
function slot_view_direct(active, base, rows)
    args = (slot_window(active, base), rows)
    return slot_sumsq(view(args...))
end

function slot_expected(active, base, rows)
    g = zero(active)
    for r in rows
        r <= length(active) && (g[r] += 2 * active[r])
    end
    return g
end

@testset "A moved stack slot passed to jl_object_id_" begin
    base = [3.0, 4.0, 2.0, 1.0, -0.5]
    for rows in ([1, 2, 4], [2, 2, 5, 1], Int[])
        active = [0.5, -1.0]
        expected = slot_expected(active, base, rows)
        primal = slot_sumsq([SlotPoint(base, active)[r] for r in rows])
        for f in (slot_view_loss, slot_copy_loss)
            dactive = zero(active)
            b = copy(base)
            res = autodiff(ReverseWithPrimal, f, Active, Duplicated(active, dactive), Const(b), Const(rows))
            @test res[2] ≈ primal
            @test dactive ≈ expected
            @test b == base
            dactive = zero(active)
            autodiff(set_runtime_activity(Reverse), f, Active, Duplicated(active, dactive), Const(b), Const(rows))
            @test dactive ≈ expected
        end

        dactive = zero(active)
        args = (Duplicated(active, dactive), Const(copy(base)), Const(rows))
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(slot_view_direct)}, Active,
            typeof.(args)...,
        )
        tape, value, _ = fwd(Const(slot_view_direct), args...)
        @test value ≈ primal
        rev(Const(slot_view_direct), args..., 1.0, tape)
        @test dactive ≈ expected
    end
end
