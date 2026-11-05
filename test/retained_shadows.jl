using Enzyme, Test

# A vector that reads selected entries from an active vector and the rest from a
# constant one. Storing a view of it into a `Duplicated` output gives the
# constant vector a zero-filled copy as its shadow.
struct RetainedOverlay{T, B, A, S} <: AbstractVector{T}
    base::B
    active::A
    slots::S
end
RetainedOverlay(base, active, slots) =
    RetainedOverlay{eltype(base), typeof(base), typeof(active), typeof(slots)}(base, active, slots)
Base.size(p::RetainedOverlay) = size(p.base)
Base.IndexStyle(::Type{<:RetainedOverlay}) = IndexLinear()
@inline Base.getindex(p::RetainedOverlay, i::Int) =
    iszero(p.slots[i]) ? p.base[i] : p.active[p.slots[i]]

function retained_view(active, seed, box, base, slots)
    values = view(RetainedOverlay(base, active, slots), 1:2)
    box[] = values
    return sum(seed .* values)
end

function unretained_view(active, seed, base, slots)
    values = view(RetainedOverlay(base, active, slots), 1:2)
    return sum(seed .* values)
end

function retained_fixture(n, ::Type{T} = Float64) where {T}
    base = fill(T(2), n)
    active = T[0.2, -0.3]
    dx = zero(active)
    slots = zeros(Int32, n)
    slots[1:2] .= (1, 2)
    seed = T[0.7, -0.4]
    box = Ref(view(RetainedOverlay(base, active, slots), 1:2))
    return active, dx, seed, box, make_zero(box), base, slots
end

function retained_gradient!(active, dx, seed, box, dbox, base, slots)
    fill!(dx, 0)
    make_zero!(dbox)
    return autodiff(
        ReverseWithPrimal, retained_view, Active, Duplicated(active, dx),
        Const(seed), Duplicated(box, dbox), Const(base), Const(slots),
    )[2]
end

function unretained_gradient!(active, dx, seed, base, slots)
    fill!(dx, 0)
    autodiff(Reverse, unretained_view, Active, Duplicated(active, dx), Const(seed), Const(base), Const(slots))
    return dx
end

function min_allocated(f::F, args) where {F}
    f(args...)
    bytes = typemax(Int)
    for _ in 1:3
        bytes = min(bytes, @allocated f(args...))
    end
    return bytes
end

sink_of(dbox) = parent(dbox[]).base

function retained_other(active, seed, box, base, slots, other)
    return retained_view(active, seed, box, base, slots) + sum(abs2, other)
end

@noinline function retained_store!(box, values)
    box[] = values
    return nothing
end
function retained_in_callee(active, seed, box, base, slots)
    values = view(RetainedOverlay(base, active, slots), 1:2)
    retained_store!(box, values)
    return sum(seed .* values)
end

const RETAINED_CALLS = Ref(0)
function retained_global(active, seed, box, base, slots)
    RETAINED_CALLS[] += 1
    return retained_view(active, seed, box, base, slots)
end

function retained_inner_gradient!(dx, active, seed, box, dbox, base, slots)
    autodiff(
        Reverse, retained_view, Active, Duplicated(active, dx), Const(seed),
        Duplicated(box, dbox), Const(base), Const(slots),
    )
    return nothing
end

function retained_helper_sum(base, shadow)
    return sum(Enzyme.Compiler.reuse_retained_shadow(base, shadow, Val((1,))))
end

@testset "Zero shadows of constant data reused at a write-only output" begin
    for T in (Float32, Float64)
        small = retained_fixture(16, T)
        large = retained_fixture(16000, T)
        for args in (small, large)
            active, dx, seed, box, dbox, base, slots = args
            supplied = sink_of(dbox)
            @test retained_gradient!(args...) ≈ sum(seed .* active)
            @test dx == seed
            @test box[] == active
            @test parent(box[]).base === base
            @test parent(dbox[]).active === dx
            # The caller's own zero copy is never reused: Enzyme did not allocate it.
            sink = sink_of(dbox)
            @test sink !== supplied
            @test sink !== base
            @test all(iszero, sink)
            @test all(==(T(2)), base)
            @test retained_gradient!(args...) ≈ sum(seed .* active)
            @test dx == seed
            @test sink_of(dbox) === sink
            @test all(iszero, sink)
            @test all(==(T(2)), base)
        end
        small_bytes = min_allocated(retained_gradient!, small)
        large_bytes = min_allocated(retained_gradient!, large)
        active, dx, seed, _, _, base, slots = large
        control_bytes = min_allocated(unretained_gradient!, (active, dx, seed, base, slots))
        @test large_bytes <= small_bytes + 512
        @test large_bytes < sizeof(base)
        @test control_bytes < sizeof(base)
    end

    @testset "A kept reference to an earlier output shadow" begin
        args = retained_fixture(16)
        _, _, _, _, dbox, _, _ = args
        retained_gradient!(args...)
        kept = dbox[]
        retained_gradient!(args...)
        # Documented: the next call may reuse the buffer the kept shadow holds.
        @test parent(kept).base === sink_of(dbox)
    end

    @testset "A copy that is not zero, or has another size" begin
        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        retained_gradient!(active, dx, seed, box, dbox, base, slots)
        old = sink_of(dbox)
        fill!(old, 7)
        fill!(dx, 0)
        autodiff(
            Reverse, retained_view, Active, Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test dx == seed
        @test sink_of(dbox) !== old
        @test all(iszero, sink_of(dbox))
        @test all(==(7), old)
        @test all(==(2), base)

        old[1] = -0.0
        old[2:end] .= 0
        dbox[] = view(RetainedOverlay(old, dx, slots), 1:2)
        fill!(dx, 0)
        autodiff(
            Reverse, retained_view, Active, Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test sink_of(dbox) !== old
        @test isequal(old[1], -0.0)

        # A constant array of another size gets a copy of its own size.
        longer = fill(3.0, 20)
        longer_slots = zeros(Int32, 20)
        longer_slots[1:2] .= (1, 2)
        previous = sink_of(dbox)
        retained_gradient!(active, dx, seed, box, dbox, longer, longer_slots)
        @test dx == seed
        @test length(sink_of(dbox)) == 20
        @test sink_of(dbox) !== previous
        @test all(==(3), longer)

        # Another constant array of the same size can use the same zero copy.
        other = fill(5.0, 20)
        previous = sink_of(dbox)
        retained_gradient!(active, dx, seed, box, dbox, other, longer_slots)
        @test dx == seed
        @test parent(box[]).base === other
        @test sink_of(dbox) === previous
        @test all(iszero, previous)
        @test all(==(5), other)
    end

    @testset "An uninitialized output" begin
        active, dx, seed, box, _, base, slots = retained_fixture(16)
        dbox = Ref{typeof(box[])}()
        autodiff(
            Reverse, retained_view, Active, Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test dx == seed
        @test isassigned(dbox)
        @test length(sink_of(dbox)) == length(base)
        @test all(iszero, sink_of(dbox))
    end

    @testset "A copy that another argument or a global can reach" begin
        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        retained_gradient!(active, dx, seed, box, dbox, base, slots)
        sink = sink_of(dbox)
        # The copy is also the shadow of another argument: never reused.
        fill!(slots, 0)
        fill!(dx, 0)
        autodiff(
            Reverse, retained_other, Active, Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots), Duplicated(base, sink),
        )
        @test all(==(4), sink)
        @test all(iszero, dx)
        @test sink_of(dbox) !== sink
        @test all(iszero, sink_of(dbox))
        @test all(==(2), base)

        # A function that touches global memory never reuses the copy.
        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        retained_gradient!(active, dx, seed, box, dbox, base, slots)
        sink = sink_of(dbox)
        make_zero!(dbox)
        fill!(dx, 0)
        autodiff(
            Reverse, retained_global, Active, Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test dx == seed
        @test sink_of(dbox) !== sink
        @test all(iszero, sink_of(dbox))

        # Only a store by the differentiated function itself qualifies: a callee
        # cannot see its caller's other arguments and memory.
        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        for _ in 1:2
            make_zero!(dbox)
            fill!(dx, 0)
            sink = sink_of(dbox)
            autodiff(
                Reverse, retained_in_callee, Active, Duplicated(active, dx), Const(seed),
                Duplicated(box, dbox), Const(base), Const(slots),
            )
            @test dx == seed
            @test sink_of(dbox) !== sink
            @test all(iszero, sink_of(dbox))
            @test all(==(2), base)
        end
    end

    # Shadow memory may be accessed only where the primal accesses the
    # corresponding memory (FAQ, "Identical types in `Duplicated` / Memory
    # Layout"). A caller-supplied copy whose untouched pages are not accessible
    # is valid, so it must never be scanned.
    if Sys.islinux() && Sys.WORD_SIZE == 64
        @testset "A caller-supplied copy with inaccessible pages" begin
            n = 16384
            bytes = n * sizeof(Float64)
            pagesize = Int(ccall(:getpagesize, Cint, ()))
            # PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
            mem = ccall(
                :mmap, Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
                C_NULL, bytes, 3, 0x22, -1, 0,
            )
            @test mem != Ptr{Cvoid}(-1)
            protected = unsafe_wrap(Array, Ptr{Float64}(mem), n; own = false)
            @test ccall(:mprotect, Cint, (Ptr{Cvoid}, Csize_t, Cint), mem + pagesize, bytes - pagesize, 0) == 0
            active, dx, seed, box, _, base, slots = retained_fixture(n)
            dbox = Ref(view(RetainedOverlay(protected, dx, slots), 1:2))
            GC.@preserve protected begin
                autodiff(
                    Reverse, retained_view, Active, Duplicated(active, dx), Const(seed),
                    Duplicated(box, dbox), Const(base), Const(slots),
                )
            end
            @test dx == seed
            @test sink_of(dbox) !== protected
            @test all(==(2), base)
            ccall(:munmap, Cint, (Ptr{Cvoid}, Csize_t), mem, bytes)
        end
    end

    @testset "Other differentiation modes allocate a new copy" begin
        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        result = autodiff(
            Forward, retained_view, Duplicated(active, ones(2)), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test result[1] ≈ sum(seed)
        @test sink_of(dbox) !== base
        @test all(iszero, sink_of(dbox))

        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        retained_gradient!(active, dx, seed, box, dbox, base, slots)
        sink = sink_of(dbox)
        make_zero!(dbox)
        fill!(dx, 0)
        autodiff(
            set_runtime_activity(Reverse), retained_view, Active, Duplicated(active, dx),
            Const(seed), Duplicated(box, dbox), Const(base), Const(slots),
        )
        @test dx == seed
        @test all(==(2), base)

        d1 = zero(active)
        d2 = zero(active)
        db1 = make_zero(box)
        db2 = make_zero(box)
        autodiff(
            Reverse, retained_view, Active, BatchDuplicated(active, (d1, d2)),
            Const(seed), BatchDuplicated(box, (db1, db2)), Const(base), Const(slots),
        )
        @test d1 == seed
        @test d2 == seed
        @test sink_of(db1) !== sink_of(db2)
        @test all(iszero, sink_of(db1))
        @test all(iszero, sink_of(db2))

        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(retained_view)}, Active,
            Duplicated{typeof(active)}, Const{typeof(seed)}, Duplicated{typeof(box)},
            Const{typeof(base)}, Const{typeof(slots)},
        )
        fill!(dx, 0)
        make_zero!(dbox)
        tape, primal, _ = fwd(
            Const(retained_view), Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots),
        )
        rev(
            Const(retained_view), Duplicated(active, dx), Const(seed),
            Duplicated(box, dbox), Const(base), Const(slots), one(primal), tape,
        )
        @test primal ≈ sum(seed .* active)
        @test dx == seed
        @test all(iszero, sink_of(dbox))
    end

    @testset "Higher derivatives" begin
        # Differentiating the helper itself: fresh zeros, nothing flows back.
        base = fill(2.0, 4)
        shadow = Ref(zero(base))
        tangent = Ref(ones(4))
        @test autodiff(Forward, retained_helper_sum, Const(base), Duplicated(shadow, tangent))[1] == 0
        @test tangent[] == ones(4)
        autodiff(Reverse, retained_helper_sum, Active, Const(base), Duplicated(shadow, tangent))
        @test tangent[] == ones(4)

        active, dx, seed, box, dbox, base, slots = retained_fixture(16)
        ddx = zero(dx)
        autodiff(
            Forward, retained_inner_gradient!, Duplicated(dx, ddx),
            Duplicated(active, ones(2)), Duplicated(seed, [2.0, 3.0]),
            Duplicated(box, make_zero(box)), Duplicated(dbox, make_zero(dbox)),
            Const(base), Const(slots),
        )
        @test dx == seed
        @test ddx == [2.0, 3.0]
        @test all(==(2), base)
    end

    @testset "Registry of allocated copies" begin
        sinks = Enzyme.Compiler.RETAINED_SINKS
        x = zeros(3)
        @test !Enzyme.Compiler.is_retained_sink(x)
        Enzyme.Compiler.register_retained_sink!(x)
        @test Enzyme.Compiler.is_retained_sink(x)
        @test !Enzyme.Compiler.is_retained_sink(copy(x))
        for _ in 1:1000
            Enzyme.Compiler.register_retained_sink!(zeros(1))
        end
        GC.gc()
        for _ in 1:1100
            Enzyme.Compiler.register_retained_sink!(zeros(1))
        end
        @test length(sinks.entries) < 2000
        @test Enzyme.Compiler.is_retained_sink(x)
    end
end
