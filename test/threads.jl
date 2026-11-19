using Enzyme
using Test

@testset "Threads $(Threads.nthreads())" begin
    function tasktest(M, x)
        xr = Ref(x)
        task = Threads.@spawn begin
            @inbounds M[1] = xr[]
        end
        @inbounds M[2] = x
        wait(task)
        nothing
    end

    R = Float64[0.0, 0.0]
    dR = Float64[2.0, 3.0]

    # We define the local variable `autodiff` as a `Ref` wrapper only to be able to capture
    # and test the warning emitted by the `@generated` function `Enzyme.autodiff`.
    autodiff = Ref{Any}(Enzyme.autodiff)
    @test 5.0 ≈ @test_warn r"active variables passed by value to jl_new_task are not yet supported" autodiff[](Reverse, tasktest, Duplicated(R, dR), Active(2.0))[1][2]
    @test Float64[2.0, 2.0] ≈ R
    @test Float64[0.0, 0.0] ≈ dR

    Enzyme.autodiff(Forward, tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
    @test Float64[1.0, 1.0] ≈ dR

    function tasktest2(M, x)
        task = Threads.@spawn begin
            return
        end
        Base.wait(task)
        nothing
    end
    # The empty return previously resulted in an illegal instruction error
    @test 0.0 ≈ @test_warn r"active variables passed by value to jl_new_task are not yet supported" autodiff[](Reverse, tasktest2, Duplicated(R, dR), Active(2.0))[1][2]
    @test () === Enzyme.autodiff(Forward, tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
end

@testset "Advanced Threads $(Threads.nthreads())" begin
    function foo(y)
        Threads.@threads for i in 1:3
            y[i] *= 2
        end
        nothing
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(Reverse, foo, Duplicated(x, dx))
    @test 2.0 ≈ x[1]
    @test 4.0 ≈ x[2]
    @test 6.0 ≈ x[3]
    @test 2.0 ≈ dx[1]
    @test 2.0 ≈ dx[2]
    @test 2.0 ≈ dx[3]

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(Forward, foo, Duplicated(x, dx))
    @test 2.0 ≈ x[1]
    @test 4.0 ≈ x[2]
    @test 6.0 ≈ x[3]
    @test 2.0 ≈ dx[1]
    @test 2.0 ≈ dx[2]
    @test 2.0 ≈ dx[3]
end

@testset "Advanced, Active-var Threads $(Threads.nthreads())" begin
    function f_multi(out, in)
        Threads.@threads for idx in 1:length(out)
            out[idx] = in
        end
        return nothing
    end

    out = [1.0, 2.0]
    dout = [1.0, 1.0]
    res = autodiff(Reverse, f_multi, Const, Duplicated(out, dout), Active(2.0))
    @test res[1][2] ≈ 2.0
end

@testset "Closure-less threads $(Threads.nthreads())" begin
    function bf(i, x)
        x[i] *= x[i]
        nothing
    end

    function psquare0(x)
        Enzyme.pmap(bf, 10, x)
    end

    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)

    Enzyme.autodiff(Reverse, psquare0, Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs

    function psquare1(x)
        Enzyme.@parallel x for i in 1:10
            @inbounds x[i] *= x[i]
        end
    end

    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)

    Enzyme.autodiff(Reverse, psquare1, Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs

    function psquare2(x, y)
        Enzyme.@parallel x y for i in 1:10
            @inbounds x[i] *= y[i]
        end
    end

    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)

    Enzyme.autodiff(Reverse, psquare2, Duplicated(xs, dxs), Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs
end

# TODO on 1.8 having `Inactive threads` after `UndefVar` in the main `runtest.jl` leads to a GC verification bug
@testset "Inactive threads" begin
    function thr_inactive(x, y)
        if x
            Threads.@threads for N in 1:5:20
                println("The number of this iteration is $N")
            end
        end
        y
    end
    @test 1.0 ≈ autodiff(Reverse, thr_inactive, Const(false), Active(2.14))[1][2]
    @test 1.0 ≈ autodiff(Forward, thr_inactive, Const(false), Duplicated(2.14, 1.0))[1]

    @test 1.0 ≈ autodiff(Reverse, thr_inactive, Const(true), Active(2.14))[1][2]
    @test 1.0 ≈ autodiff(Forward, thr_inactive, Const(true), Duplicated(2.14, 1.0))[1]
end

@testset "Batched Forward" begin
    function f2(du)
        Threads.@threads for i in eachindex(du)
            du[i] *= 2
        end
        return nothing
    end
    du = zeros(2)
    ty = ntuple(i -> (i + 1) * ones(2), Val(4))
    y_and_ty = BatchDuplicated(du, ty)
    autodiff(Forward, f2, Const, y_and_ty)
    @test ty[1] ≈ [4.0, 4.0]
    @test ty[2] ≈ [6.0, 6.0]
    @test ty[3] ≈ [8.0, 8.0]
    @test ty[4] ≈ [10.0, 10.0]
end

@testset "GEP non-inline type analysis" begin
    function spread_no_broadcast!(bufs, q, ::Val{N}) where {N}
        Threads.@threads for t in 1:N
            for i in t:N:length(q)
                @inbounds bufs[t][i, i + 1, i + 2] += q[i]
            end
        end
        return sum(sum, bufs)
    end

    q = [1.0, 2.0]
    dq = [0.0, 0.0]
    bufs = [zeros(4, 4, 4) for _ in 1:2]
    dbufs = [zeros(4, 4, 4) for _ in 1:2]

    autodiff(
        ReverseWithPrimal, spread_no_broadcast!, Active, Duplicated(bufs, dbufs),
        Duplicated(q, dq), Const(Val(2))
    )
    @test dq ≈ [1.0, 1.0]
end


# Callees reached from inside a threaded loop: one by static dispatch (compiled into the loop
# body's job), one by dynamic dispatch (compiled by the runtime rules). Top-level so that the
# loop body captures nothing but the array.
function atomic_sumsq(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]^2
    end
    return s
end
@noinline atomic_sumsq_noinline(x) = atomic_sumsq(x)
const atomic_sumsq_ref = Ref{Any}(atomic_sumsq)
function atomic_threaded_callees(x, N)
    out = zeros(N)
    Threads.@threads for i in 1:N
        @inbounds out[i] = isodd(i) ? atomic_sumsq_noinline(x) : (atomic_sumsq_ref[](x)::Float64)
    end
    return sum(out)
end
const atomic_threaded_ref = Ref{Any}(atomic_threaded_callees)
atomic_via_dynamic(x, N) = atomic_threaded_ref[](x, N)::Float64

@testset "Atomic accumulation into shadows $(Threads.nthreads())" begin
    function sumsq(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]^2
        end
        return s
    end
    function spawn_sumsq!(out, x)
        task = Threads.@spawn begin
            @inbounds out[1] = sumsq(x)
        end
        @inbounds out[2] = sumsq(x)
        wait(task)
        nothing
    end
    function threaded_sumsq(x)
        out = zeros(4)
        Threads.@threads for i in 1:4
            @inbounds out[i] = i * sumsq(x)
        end
        return sum(out)
    end
    function atomics(f, A, TT, setting)
        ir = sprint() do io
            Enzyme.Compiler.enzyme_code_llvm(io, f, A, TT; dump_module = true, AtomicAdd = setting)
        end
        return count("atomicrmw", ir)
    end

    @test atomic_add(Reverse) === nothing
    @test atomic_add(set_atomic_add(Reverse, false)) === false
    @test atomic_add(set_atomic_add(ReverseSplitWithPrimal, true)) === true

    multithreaded = Threads.nthreads() > 1
    VecDup = Tuple{Duplicated{Vector{Float64}}}
    # By default a CPU derivative accumulates atomically exactly when Julia runs several threads
    @test (atomics(sumsq, Active, VecDup, nothing) > 0) == multithreaded
    @test atomics(sumsq, Active, VecDup, true) > 0
    @test atomics(sumsq, Active, VecDup, false) == 0
    # A function that runs work on other threads cannot opt out: `false` is refused rather than
    # compiled with atomics anyway (its primal code has atomics of its own, so the default is
    # compared against `true` rather than against zero)
    SpawnTT = Tuple{Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}}
    for (f, A, TT) in ((threaded_sumsq, Active, VecDup), (spawn_sumsq!, Const, SpawnTT))
        @test_throws Enzyme.Compiler.EnzymeAtomicAddError atomics(f, A, TT, false)
        @test (atomics(f, A, TT, nothing) == atomics(f, A, TT, true)) == multithreaded
    end

    x = collect(1.0:10.0)
    for mode in (Reverse, set_atomic_add(Reverse, false), set_atomic_add(Reverse, true))
        dx = zero(x)
        autodiff(mode, Const(sumsq), Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test gradient(mode, sumsq, x)[1] ≈ 2 .* x
    end
    for mode in (Reverse, set_atomic_add(Reverse, true))
        dx = zero(x)
        autodiff(mode, Const(threaded_sumsq), Active, Duplicated(x, dx))
        @test dx ≈ 20 .* x

        dx = zero(x)
        autodiff(mode, Const(spawn_sumsq!), Const, Duplicated(zeros(2), ones(2)), Duplicated(x, dx))
        @test dx ≈ 4 .* x
    end
    @test_throws Enzyme.Compiler.EnzymeAtomicAddError autodiff(set_atomic_add(Reverse, false), Const(threaded_sumsq), Active, Duplicated(x, zero(x)))
    @test_throws Enzyme.Compiler.EnzymeAtomicAddError autodiff(set_atomic_add(Reverse, false), Const(spawn_sumsq!), Const, Duplicated(zeros(2), ones(2)), Duplicated(x, zero(x)))

    fwd, rev = autodiff_thunk(set_atomic_add(ReverseSplitNoPrimal, false), Const{typeof(sumsq)}, Active, Duplicated{Vector{Float64}})
    dx = zero(x)
    tape, = fwd(Const(sumsq), Duplicated(x, dx))
    rev(Const(sumsq), Duplicated(x, dx), 1.0, tape)
    @test dx ≈ 2 .* x

    # By default, `autodiff` calls on several threads may share one shadow
    dx = zero(x)
    Threads.@threads for i in 1:1000
        autodiff(Reverse, Const(sumsq), Active, Duplicated(x, dx))
    end
    @test dx ≈ 2000 .* x

    # Callees of a threaded loop body never see the caller's setting: the body and the
    # `@noinline` callee it calls statically are compiled by the threadsfor rule, a dynamically
    # dispatched callee by the runtime rules, all with the default
    CalleesTT = Tuple{Duplicated{Vector{Float64}}, Const{Int}}
    @test_throws Enzyme.Compiler.EnzymeAtomicAddError atomics(atomic_threaded_callees, Active, CalleesTT, false)
    N = 1000
    for mode in (Reverse, set_atomic_add(Reverse, true))
        dx = zero(x)
        autodiff(mode, Const(atomic_threaded_callees), Active, Duplicated(x, dx), Const(N))
        @test dx ≈ 2N .* x
    end
    # A function whose dynamically dispatched callee runs the threaded loop may opt out: its
    # own module starts no parallel work, and the callee is compiled with the default
    @test atomics(atomic_via_dynamic, Active, CalleesTT, false) == 0
    for mode in (Reverse, set_atomic_add(Reverse, false), set_atomic_add(Reverse, true))
        dx = zero(x)
        autodiff(mode, Const(atomic_via_dynamic), Active, Duplicated(x, dx), Const(N))
        @test dx ≈ 2N .* x
    end
end

# Callees reached through dynamic dispatch, which inherit the caller's `AtomicAdd` setting. Each
# case has a callee type of its own, so that `callee_settings` tells the cases apart.
struct AtomicCallee{Case} end
(::AtomicCallee)(x) = atomic_sumsq(x)
atomic_callee(case) = Ref{Any}(AtomicCallee{case}())

atomic_generic(x, f) = f[](x)::Float64
atomic_latest(x, f) = Base.invokelatest(f[], x)::Float64
# A splat of unknown length: `xs` holds the one argument
atomic_splat(xs, f) = f[](xs...)::Float64
# A method Julia compiles without a specialized entry point, as all it takes and returns is
# boxed: a call of it is resolved statically, and still made through the runtime (`jl_invoke`)
struct AtomicBoxed{Case} end
@noinline (::AtomicBoxed)(@nospecialize(x)) = Base.inferencebarrier(atomic_sumsq(x::Vector{Float64}))
atomic_boxed(x, ::AtomicBoxed{Case}) where {Case} = AtomicBoxed{Case}()(x)::Float64

# A callee that reaches another one through dynamic dispatch
struct AtomicMiddle{Case} end
(::AtomicMiddle{Case})(x) where {Case} = atomic_generic(x, atomic_callee(Case))

# A callee that runs a threaded loop, and reaches other callees outside and inside of it
struct AtomicThreaded{Case} end
function (::AtomicThreaded{Case})(x) where {Case}
    outside = atomic_generic(x, atomic_callee((Case, :outside)))
    inside = atomic_callee((Case, :inside))
    out = zeros(8)
    Threads.@threads for i in 1:8
        @inbounds out[i] = atomic_generic(x, inside)
    end
    return outside + sum(out)
end

# A callee that reaches another one while a task it spawned runs
struct AtomicSpawning{Case} end
function (::AtomicSpawning{Case})(x) where {Case}
    out = zeros(1)
    task = Threads.@spawn begin
        @inbounds out[1] = atomic_sumsq(x)
    end
    between = atomic_generic(x, atomic_callee((Case, :between)))
    wait(task)
    return between + @inbounds out[1]
end

# The `AtomicAdd` settings with which the run-time rules asked for the derivative of a callee of
# type `FT`, read off the specializations of the `runtime_thunk` method they call:
# (runtime_thunk, world, FA, A, TT, ..., strongzero, atomicadd)
function callee_settings(::Type{FT}) where {FT}
    settings = Set{Any}()
    for m in methods(Enzyme.Compiler.runtime_thunk), mi in Base.specializations(m)
        mi isa Core.MethodInstance || continue
        params = Base.unwrap_unionall(mi.specTypes).parameters
        (length(params) == 15 && params[3] === Type{Const{FT}}) || continue
        push!(settings, params[15].parameters[1])
    end
    return settings
end
callee_settings(case) = callee_settings(AtomicCallee{case})

# A sum whose derivative shows how it was accumulated, where the derivative contracts a
# multiplication into the addition after it: plain accumulation then gives -2^-54 and atomic
# accumulation, which rounds the product first, gives 0
function atomic_gather(x, c)
    a, b, indices = c
    s = 0.0
    @inbounds for i in eachindex(a)
        s += (x[indices[i]] * a[i]) * b[i]
    end
    return s
end
const atomic_gather_ref = Ref{Any}(atomic_gather)
atomic_gather_generic(x, c) = atomic_gather_ref[](x, c)::Float64
atomic_gather_latest(x, c) = Base.invokelatest(atomic_gather, x, c)::Float64

@testset "Atomic accumulation in dynamically dispatched callees $(Threads.nthreads())" begin
    Inherited = Enzyme.InheritedPlainAdd
    ForSeveralThreads = Enzyme.Compiler.ForSeveralThreads
    multithreaded = Threads.nthreads() > 1
    # What a callee inherits from a caller with each setting
    inherited = ((nothing, nothing), (false, Inherited), (true, true))
    VecDup = Tuple{Duplicated{Vector{Float64}}}
    function atomics(f, setting)
        ir = sprint() do io
            Enzyme.Compiler.enzyme_code_llvm(io, f, Active, VecDup; dump_module = true, AtomicAdd = setting)
        end
        return count("atomicrmw", ir)
    end
    function threaded_sumsq(x)
        out = zeros(4)
        Threads.@threads for i in 1:4
            @inbounds out[i] = i * atomic_sumsq(x)
        end
        return sum(out)
    end

    # A callee that inherits plain accumulation accumulates plainly. One that runs work in
    # parallel is compiled with the default instead of refused, as an explicit `false` is
    @test atomics(atomic_sumsq, Inherited) == 0
    @test_throws Enzyme.Compiler.EnzymeAtomicAddError atomics(threaded_sumsq, false)
    @test atomics(threaded_sumsq, Inherited) == atomics(threaded_sumsq, nothing)
    # The variants a package image keeps for sessions with several threads: the default made
    # as such a session makes it, also where a callee falls back to it
    @test atomics(atomic_sumsq, ForSeveralThreads(nothing)) == atomics(atomic_sumsq, true)
    @test atomics(atomic_sumsq, ForSeveralThreads(Inherited)) == 0
    @test atomics(threaded_sumsq, ForSeveralThreads(Inherited)) == atomics(threaded_sumsq, ForSeveralThreads(nothing))
    @test atomics(threaded_sumsq, ForSeveralThreads(Inherited)) > atomics(threaded_sumsq, Inherited) || multithreaded
    # The compile cache is keyed by the hash of a job, which must tell the settings apart
    jobs = [
        Enzyme.Compiler.get_job(atomic_sumsq, Active, VecDup; AtomicAdd = setting)
            for setting in (nothing, false, true, Inherited)
    ]
    @test allunique(map(hash, jobs))

    x = collect(1.0:10.0)
    # Every kind of dynamic call hands the caller's setting to its callee
    for (kind, caller) in ((:generic, atomic_generic), (:latest, atomic_latest))
        for (setting, callee) in inherited
            case = (kind, setting)
            dx = zero(x)
            autodiff(set_atomic_add(Reverse, setting), Const(caller), Active, Duplicated(x, dx), Const(atomic_callee(case)))
            @test dx == 2 .* x
            @test callee_settings(case) == Set{Any}((callee,))
        end
    end
    for (setting, callee) in inherited
        case = (:splat, setting)
        xs = [copy(x)]
        dxs = [zero(x)]
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_splat), Active, Duplicated(xs, dxs), Const(atomic_callee(case)))
        @test dxs[1] == 2 .* x
        @test callee_settings(case) == Set{Any}((callee,))
    end
    for (setting, callee) in inherited
        case = (:boxed, setting)
        dx = zero(x)
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_boxed), Active, Duplicated(x, dx), Const(AtomicBoxed{case}()))
        @test dx == 2 .* x
        @test callee_settings(AtomicBoxed{case}) == Set{Any}((callee,))
    end
    # and so does a callee to the callees it reaches
    for (setting, callee) in inherited
        case = (:nested, setting)
        dx = zero(x)
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_generic), Active, Duplicated(x, dx), Const(Ref{Any}(AtomicMiddle{case}())))
        @test dx == 2 .* x
        @test callee_settings(AtomicMiddle{case}) == Set{Any}((callee,))
        @test callee_settings(case) == Set{Any}((callee,))
    end
    # With split modes too
    for (setting, callee) in inherited
        case = (:split, setting)
        f = atomic_callee(case)
        fwd, rev = autodiff_thunk(
            set_atomic_add(ReverseSplitNoPrimal, setting), Const{typeof(atomic_generic)}, Active,
            Duplicated{Vector{Float64}}, Const{typeof(f)},
        )
        dx = zero(x)
        tape, = fwd(Const(atomic_generic), Duplicated(x, dx), Const(f))
        rev(Const(atomic_generic), Duplicated(x, dx), Const(f), 1.0, tape)
        @test dx == 2 .* x
        @test callee_settings(case) == Set{Any}((callee,))
    end

    # A callee gives the bits of a direct call with the same setting, also after the callee was
    # compiled with another one in this process
    c = ([1 + 2.0^-27, 1.0], [1 - 2.0^-27, -1.0], [1, 1])
    for setting in (nothing, false, true, nothing, false), caller in (atomic_gather_generic, atomic_gather_latest)
        mode = set_atomic_add(Reverse, setting)
        dx = zeros(2)
        autodiff(mode, Const(atomic_gather), Active, Duplicated([0.7, 1.3], dx), Const(c))
        ddx = zeros(2)
        autodiff(mode, Const(caller), Active, Duplicated([0.7, 1.3], ddx), Const(c))
        @test ddx == dx
        @test iszero(dx[2]) && abs(dx[1]) <= 2.0^-54
    end

    # A callee that runs a threaded loop is not refused under an inherited `false`: it keeps the
    # default, and so do the callees it reaches, outside of the loop and inside. The iterations
    # share one shadow, which stays exact
    for (setting, own, callees) in ((nothing, nothing, nothing), (false, Inherited, nothing), (true, true, true))
        case = (:threaded, setting)
        dx = zero(x)
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_generic), Active, Duplicated(x, dx), Const(Ref{Any}(AtomicThreaded{case}())))
        @test dx == 18 .* x
        @test callee_settings(AtomicThreaded{case}) == Set{Any}((own,))
        @test callee_settings((case, :outside)) == Set{Any}((callees,))
        # The loop body is compiled with the default under every setting
        @test callee_settings((case, :inside)) == Set{Any}((nothing,))
    end
    # A callee that spawns a task keeps the default too, with what it reaches while the task runs
    for (setting, own, callees) in ((nothing, nothing, nothing), (false, Inherited, nothing), (true, true, true))
        case = (:spawning, setting)
        dx = zero(x)
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_generic), Active, Duplicated(x, dx), Const(Ref{Any}(AtomicSpawning{case}())))
        @test dx == 4 .* x
        @test callee_settings(AtomicSpawning{case}) == Set{Any}((own,))
        @test callee_settings((case, :between)) == Set{Any}((callees,))
    end

    # Shared shadows under contention: many iterations of a threaded loop, each reaching its
    # callee through dynamic dispatch, behind a caller that asked for plain accumulation
    N = 2000
    for setting in (false, nothing)
        dx = zero(x)
        autodiff(set_atomic_add(Reverse, setting), Const(atomic_via_dynamic), Active, Duplicated(x, dx), Const(N))
        @test dx == 2N .* x
    end
    # The callee inside that loop was only ever compiled with the default
    @test callee_settings(typeof(atomic_sumsq)) == Set{Any}((nothing,))
end

# The inner derivatives of nested ones: taken with `autodiff`, whose thunk is compiled while
# the outer function is inferred and imported into the outer derivative, and with
# `autodiff_deferred`, which the outer compilation compiles itself.
function atomic_inner_gradient!(dx, x)
    autodiff(Reverse, Const(atomic_sumsq), Active, Duplicated(x, dx))
    return nothing
end
function atomic_deferred_gradient!(dx, x)
    autodiff_deferred(Reverse, Const(atomic_sumsq), Active, Duplicated(x, dx))
    return nothing
end

# A package image is written by a process with one thread and keeps, for sessions with several,
# a variant of each derivative compiled as such a session compiles it
# (`Compiler.ForSeveralThreads`). That has to hold for every derivative compiled on its behalf
# and linked into it as well: the body of a threaded loop, which gets the default under every
# setting, and the inner derivative of a nested one. This compiles those variants directly, so
# what it counts does not depend on the number of threads of this process.
@testset "Derivatives compiled for several threads $(Threads.nthreads())" begin
    function threaded_sumsq(x)
        out = zeros(4)
        Threads.@threads for i in 1:4
            @inbounds out[i] = i * atomic_sumsq(x)
        end
        return sum(out)
    end
    function atomics(f, A, TT, setting; mode = Enzyme.API.DEM_ReverseModeCombined)
        ir = sprint() do io
            Enzyme.Compiler.enzyme_code_llvm(io, f, A, TT; dump_module = true, AtomicAdd = setting, mode)
        end
        return count("atomicrmw", ir)
    end
    several = Enzyme.Compiler.ForSeveralThreads
    multithreaded = Threads.nthreads() > 1
    VecDup = Tuple{Duplicated{Vector{Float64}}}

    # A serial function for several threads is the function compiled with atomics
    @test atomics(atomic_sumsq, Active, VecDup, several(nothing)) == atomics(atomic_sumsq, Active, VecDup, true) > 0
    @test atomics(atomic_sumsq, Active, VecDup, several(false)) == 0

    # The body of a threaded loop is compiled with the default whatever the function's own
    # setting is, so with one thread `true` leaves it accumulating plainly. For several threads
    # the body accumulates atomically under either setting, as it does in a process with
    # several
    for_several = atomics(threaded_sumsq, Active, VecDup, several(nothing))
    @test for_several == atomics(threaded_sumsq, Active, VecDup, several(true))
    @test (for_several > atomics(threaded_sumsq, Active, VecDup, true)) == !multithreaded
    multithreaded && @test for_several == atomics(threaded_sumsq, Active, VecDup, nothing)
    @test_throws Enzyme.Compiler.EnzymeAtomicAddError atomics(threaded_sumsq, Active, VecDup, several(false))

    # Forward over Reverse: the inner derivative accumulates into shadows the caller may share
    # between threads, with the default setting. Forward mode has no accumulation of its own
    MatDup = Tuple{Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}}
    for inner in (atomic_inner_gradient!, atomic_deferred_gradient!)
        forward(setting) = atomics(inner, Const, MatDup, setting; mode = Enzyme.API.DEM_ForwardMode)
        @test forward(several(nothing)) > 0
        @test (forward(nothing) > 0) == multithreaded
        multithreaded && @test forward(several(nothing)) == forward(nothing)
    end
end

@testset "Forward constant captured by threads closure" begin
    function thr_const_capture(x, a)
        y = zeros(length(x))
        Threads.@threads for i in eachindex(x)
            y[i] = a * x[i]
        end
        return sum(y)
    end
    thr_const_local(x) = thr_const_capture(x, 2.0)

    x = [1.0, 2.0, 3.0, 4.0]
    @test 8.0 ≈ autodiff(Forward, thr_const_local, Duplicated(copy(x), ones(4)))[1]
    @test 0.0 ≈ autodiff(Forward, thr_const_local, Duplicated(copy(x), zeros(4)))[1]
    @test 8.0 ≈ autodiff(Forward, thr_const_capture, Duplicated(copy(x), ones(4)), Const(2.0))[1]
    @test 10.0 ≈ autodiff(Forward, thr_const_capture, Duplicated(copy(x), zeros(4)), Duplicated(2.0, 1.0))[1]
    res = autodiff(Forward, thr_const_local, BatchDuplicated(copy(x), (ones(4), 2 .* ones(4))))[1]
    @test 8.0 ≈ res[1]
    @test 16.0 ≈ res[2]
end

@testset "Threads.@spawn with active-by-value closure capture (#3226)" begin
    function f_spawn_unused(x)
        y = (x, 2.0)
        t = Threads.@spawn y[1]^2
        wait(t)
        return y[1]
    end
    function f_spawn_active(x)
        r = Ref(0.0)
        y = (x, 2.0)
        t = Threads.@spawn begin
            r[] = y[1]^2 * y[2]
            nothing
        end
        wait(t)
        return r[]
    end
    @test only(only(autodiff(set_runtime_activity(Reverse), f_spawn_unused, Active, Active(2.0)))) ≈ 1.0
    @test only(only(autodiff(set_runtime_activity(Reverse), f_spawn_active, Active, Active(3.0)))) ≈ 12.0
end
