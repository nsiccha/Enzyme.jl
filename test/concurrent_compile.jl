using Enzyme, Test

# First calls of different derivatives on several threads at once. A combined reverse-mode
# derivative compiles at its first call, holding Enzyme's compile lock; a split-mode one
# (`autodiff_thunk`) compiles in `thunk`'s generator while its caller is inferred, and when the
# runtime triggered that inference Julia holds its codegen lock (the type-inference lock on
# 1.11 to 1.13). The first needs the codegen lock to emit code, so a generator that waited for
# the compile lock deadlocked with it (see `Enzyme.Compiler.cached_compilation`).
#
# The workload runs in a child process with several threads, so that a deadlock fails this
# test (a watchdog in the child, then a timeout here) instead of hanging the test run. The
# child's main task, which runs the watchdog, is on its interactive thread: the workload's
# tasks run on the default threads, so a deadlocked one cannot hold the watchdog's thread.

const CONCURRENT_COMPILE_CHILD = raw"""
using Enzyme
using Enzyme.EnzymeRules
using Base.Threads

const NA = 12   # first calls of lazily compiled derivatives
const NB = 6    # first calls of split-mode derivatives, per eager task
const NBT = 2   # eager tasks
const STALL = 300.0

# A Julia rule, so that a lazily compiled derivative emits Julia code in the middle of its
# compile, which needs the codegen lock.
cc_cube(x) = x^3
function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(cc_cube)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(cc_cube)}, dret::Active, tape, x::Active)
    return (3 * x.val^2 * dret.val,)
end

for i in 1:max(NA, NB * NBT)
    fa = Symbol(:cc_fa_, i)
    fb = Symbol(:cc_fb_, i)
    ca = Symbol(:cc_calla_, i)
    cb = Symbol(:cc_callb_, i)
    @eval begin
        $fa(x) = cc_cube(sin(x * $i)) + cc_cube(cos(x)) + exp(x / $i)
        $fb(x) = cc_cube(tanh(x * $i)) + x * x * $i + sqrt(x + $i)
        @noinline function $cb(x)
            fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof($fb)}, Active, Active{Float64})
            tape, _, _ = fwd(Const($fb), Active(x))
            return rev(Const($fb), Active(x), 1.0, tape)[1][1]
        end
        @noinline $ca(x) = autodiff(Reverse, $fa, Active, Active(x))[1][1]
    end
end

dfa(i, x) = 3 * sin(x * i)^2 * cos(x * i) * i - 3 * cos(x)^2 * sin(x) + exp(x / i) / i
dfb(i, x) = 3 * tanh(x * i)^2 * (1 - tanh(x * i)^2) * i + 2 * x * i + 1 / (2 * sqrt(x + i))

const done = Atomic{Int}(0)
const wrong = Atomic{Int}(0)
const failed = Atomic{Int}(0)
const last_progress = Atomic{Float64}(time())

function run_lazy(x)
    for i in 1:NA
        try
            g = Base.invokelatest(getfield(Main, Symbol(:cc_calla_, i)), x)
            isapprox(g, dfa(i, x); rtol = 1.0e-10) || atomic_add!(wrong, 1)
        catch err
            atomic_add!(failed, 1)
            Core.println("lazy ", i, ": ", sprint(showerror, err))
        end
        atomic_add!(done, 1)
        last_progress[] = time()
    end
end

function run_eager(t, x)
    for k in 1:NB
        i = (t - 1) * NB + k
        try
            g = Base.invokelatest(getfield(Main, Symbol(:cc_callb_, i)), x)
            isapprox(g, dfb(i, x); rtol = 1.0e-10) || atomic_add!(wrong, 1)
        catch err
            atomic_add!(failed, 1)
            Core.println("eager ", i, ": ", sprint(showerror, err))
        end
        atomic_add!(done, 1)
        last_progress[] = time()
    end
end

function report(tag)
    Core.println(tag, " ", done[], " ", NA + NB * NBT, " ", wrong[], " ", failed[], " ", nthreads())
end

function main()
    x = 0.37
    tasks = Task[Threads.@spawn run_lazy(x)]
    for t in 1:NBT
        push!(tasks, Threads.@spawn run_eager(t, x))
    end
    last_progress[] = time()
    while true
        sleep(1.0)
        if all(istaskdone, tasks)
            report("DONE")
            ccall(:_exit, Cvoid, (Cint,), 0)
        end
        if time() - last_progress[] > STALL
            report("STALL")
            ccall(:_exit, Cvoid, (Cint,), 3)
        end
    end
end

# Compile Enzyme's own machinery on one thread first, so that only the concurrent first calls
# are left for the workload.
cc_warm(x) = cc_cube(sin(x)) + x
autodiff(Reverse, cc_warm, Active, Active(0.3))
let (fwd, rev) = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cc_warm)}, Active, Active{Float64})
    tape, _, _ = fwd(Const(cc_warm), Active(0.3))
    rev(Const(cc_warm), Active(0.3), 1.0, tape)
end
report("WARM")
main()
"""

@testset "Concurrent first calls of lazily and eagerly compiled derivatives" begin
    cmd = `$(Base.julia_cmd()) --threads=4,1 --startup-file=no --project=$(Base.active_project()) -e $CONCURRENT_COMPILE_CHILD`
    out = IOBuffer()
    proc = run(pipeline(cmd; stdout = out, stderr = out); wait = false)
    # The child's own watchdog ends a stalled run; this ends one whose watchdog cannot run.
    timer = Timer(1200) do _
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
    close(timer)
    output = String(take!(out))
    done = filter(startswith("DONE "), split(output, '\n'))
    @test success(proc)
    @test length(done) == 1
    if length(done) == 1
        _, finished, total, wrong, failed, nthreads = split(done[1])
        @test finished == total
        @test wrong == "0"
        @test failed == "0"
        @test nthreads == "4"
    else
        println(stderr, "Child output:\n", output)
    end
end
