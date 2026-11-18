using Enzyme
using Test

# Derivatives of derivatives of `Threads.@threads` loops. The outer derivative differentiates
# the code the inner one runs the loop with: the calls of `threading_run` it makes, and the
# callables it hands them, which reach the derivative of the loop's body.

function nt_sumsq(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]^2
    end
    return s
end

# Every iteration adds into the one shadow of `x`: gradient `128 x`, Hessian `128 I`.
function nt_dynamic(x)
    out = zeros(64)
    Threads.@threads for i in 1:64
        @inbounds out[i] = nt_sumsq(x)
    end
    return sum(out)
end

function nt_static(x)
    out = zeros(64)
    Threads.@threads :static for i in 1:64
        @inbounds out[i] = nt_sumsq(x)
    end
    return sum(out)
end

# Not quadratic, and a weight per iteration: gradient `55 a exp(a x)`, Hessian diagonal
# `55 a^2 exp(a x)`.
const nt_rates = [0.3, -0.2, 0.5, 0.1, -0.4, 0.25]

function nt_expsum(x)
    s = 0.0
    @inbounds for j in eachindex(x)
        s += exp(nt_rates[j] * x[j])
    end
    return s
end

function nt_weighted(x)
    out = zeros(10)
    Threads.@threads for i in 1:10
        @inbounds out[i] = i * nt_expsum(x)
    end
    return sum(out)
end

# A loop that writes an argument, whose body holds an active scalar.
function nt_scaled!(y, x, a)
    Threads.@threads for i in eachindex(x)
        @inbounds y[i] = a * x[i]^2
    end
    return nothing
end

function nt_scaled_gradient!(dx, x, a)
    y = zero(x)
    dy = ones(length(x))
    return autodiff(Reverse, Const(nt_scaled!), Const, Duplicated(y, dy), Duplicated(x, dx), Active(a))[1][3]
end

function nt_gradient!(f::F, dx, x) where {F}
    autodiff(Reverse, Const(f), Active, Duplicated(x, dx))
    return nothing
end

function nt_runtime_gradient!(f::F, dx, x) where {F}
    autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(x, dx))
    return nothing
end

nt_directional(f::F, x, v) where {F} = autodiff(Forward, Const(f), Duplicated(x, v))[1]

function nt_gradient_sqnorm(f::F, x) where {F}
    dx = zero(x)
    nt_gradient!(f, dx, x)
    return sum(abs2, dx)
end

# The gradient of the squared norm of the gradient, or `nothing` where compiling it stops
# with the internal error the testset below describes.
function nt_reverse_over_reverse(f::F, x) where {F}
    g = zero(x)
    try
        autodiff(Reverse, Const(nt_gradient_sqnorm), Active, Const(f), Duplicated(x, g))
    catch err
        err isa Enzyme.Compiler.EnzymeInternalError || rethrow()
        return nothing
    end
    return g
end

@testset "Nested derivatives of Threads.@threads loops $(Threads.nthreads())" begin
    x = collect(1.0:6.0)
    v = collect(6.0:-1.0:1.0)
    w = [1.0, -1.0, 2.0, 0.5, 0.0, 3.0]

    @testset "$name schedule" for (name, f) in (("default", nt_dynamic), ("static", nt_static))
        # The exact values are integers, so a lost or doubled update of a shadow shows.
        @testset "Forward over Reverse" begin
            dx = zero(x)
            ddx = zero(x)
            autodiff(Forward, Const(nt_gradient!), Const, Const(f), Duplicated(dx, ddx), Duplicated(x, v))
            @test dx == 128 .* x
            @test ddx == 128 .* v
        end

        @testset "Forward over Reverse, a batch of two directions" begin
            dx = zero(x)
            ddx1 = zero(x)
            ddx2 = zero(x)
            autodiff(
                Forward, Const(nt_gradient!), Const, Const(f),
                BatchDuplicated(dx, (ddx1, ddx2)), BatchDuplicated(x, (v, w)),
            )
            @test dx == 128 .* x
            @test ddx1 == 128 .* v
            @test ddx2 == 128 .* w
        end

        @testset "Reverse over Forward" begin
            hv = zero(x)
            autodiff(Reverse, Const(nt_directional), Active, Const(f), Duplicated(x, hv), Const(v))
            @test hv == 128 .* v
        end

        @testset "Forward over Forward" begin
            @test autodiff(Forward, Const(nt_directional), Const(f), Duplicated(x, w), Const(v))[1] ==
                128 * sum(v .* w)
        end

        @testset "Reverse over Reverse" begin
            if VERSION < v"1.11"
                g = zero(x)
                autodiff(Reverse, Const(nt_gradient_sqnorm), Active, Const(f), Duplicated(x, g))
                @test g ≈ 2 * 128^2 .* x
            elseif VERSION < v"1.12"
                # Not measured: no Julia 1.11 where these tests are developed
                @test_skip something(nt_reverse_over_reverse(f, x), zero(x)) ≈ 2 * 128^2 .* x
            else
                # Julia 1.12 and 1.13: the tape of the loop's body holds an object, and the
                # outer reverse pass moves that object to the stack although the loop's
                # callable stores it in the array of tapes. Compiling stops with
                # `EnzymeInternalError` ("Illegal address space propagation").
                @test_broken something(nt_reverse_over_reverse(f, x), zero(x)) ≈ 2 * 128^2 .* x
            end
        end
    end

    @testset "A weight per iteration" begin
        gradient = 55 .* nt_rates .* exp.(nt_rates .* x)
        hessian = 55 .* nt_rates .^ 2 .* exp.(nt_rates .* x)

        dx = zero(x)
        ddx = zero(x)
        autodiff(Forward, Const(nt_gradient!), Const, Const(nt_weighted), Duplicated(dx, ddx), Duplicated(x, v))
        @test dx ≈ gradient
        @test ddx ≈ hessian .* v

        dx = zero(x)
        ddx = zero(x)
        autodiff(
            set_runtime_activity(Forward), Const(nt_runtime_gradient!), Const, Const(nt_weighted),
            Duplicated(dx, ddx), Duplicated(x, v),
        )
        @test dx ≈ gradient
        @test ddx ≈ hessian .* v

        hv = zero(x)
        autodiff(Reverse, Const(nt_directional), Active, Const(nt_weighted), Duplicated(x, hv), Const(v))
        @test hv ≈ hessian .* v
    end

    @testset "An active scalar held by the loop body" begin
        # The gradient of `sum(a x^2)` is `2 a x` in `x` and `sum(x^2)` in `a`. With the
        # tangents `v` of `x` and 1 of `a`, theirs are `2 x + 2 a v` and `2 sum(x v)`.
        a = 2.0
        dx = zero(x)
        ddx = zero(x)
        dda, da = autodiff(
            ForwardWithPrimal, Const(nt_scaled_gradient!), Duplicated,
            Duplicated(dx, ddx), Duplicated(x, v), Duplicated(a, 1.0),
        )
        @test dx ≈ 2a .* x
        @test da ≈ sum(abs2, x)
        @test ddx ≈ 2 .* x .+ 2a .* v
        @test dda ≈ 2 * sum(x .* v)
    end
end

# A task (`Threads.@spawn`) adds into the shadow of `x` beside the caller: gradient `4 x`,
# Hessian `4 I`.
function nt_spawned(x)
    out = zeros(2)
    task = Threads.@spawn begin
        @inbounds out[1] = nt_sumsq(x)
        nothing
    end
    @inbounds out[2] = nt_sumsq(x)
    wait(task)
    return out[1] + out[2]
end

@testset "Nested derivatives through a task $(Threads.nthreads())" begin
    x = collect(1.0:6.0)
    v = collect(6.0:-1.0:1.0)
    w = [1.0, -1.0, 2.0, 0.5, 0.0, 3.0]

    @testset "First order" begin
        dx = zero(x)
        nt_gradient!(nt_spawned, dx, x)
        @test dx == 4 .* x
        @test nt_directional(nt_spawned, x, v) == 4 * sum(x .* v)
    end

    @testset "Forward over Reverse" begin
        dx = zero(x)
        ddx = zero(x)
        autodiff(Forward, Const(nt_gradient!), Const, Const(nt_spawned), Duplicated(dx, ddx), Duplicated(x, v))
        @test dx == 4 .* x
        @test ddx == 4 .* v
    end

    @testset "Reverse over Forward" begin
        hv = zero(x)
        autodiff(Reverse, Const(nt_directional), Active, Const(nt_spawned), Duplicated(x, hv), Const(v))
        @test hv == 4 .* v
    end

    @testset "Forward over Forward" begin
        @test autodiff(Forward, Const(nt_directional), Const(nt_spawned), Duplicated(x, w), Const(v))[1] ==
            4 * sum(v .* w)
    end

    @testset "Reverse over Reverse" begin
        g = zero(x)
        autodiff(Reverse, Const(nt_gradient_sqnorm), Active, Const(nt_spawned), Duplicated(x, g))
        @test g ≈ 32 .* x
    end
end

@testset "The task bodies an outer derivative differentiates" begin
    Compiler = Enzyme.Compiler
    S = ((false,), false, false)
    FT = Const{typeof(nt_sumsq)}
    Forward = Compiler.ForwardModeThunk{Ptr{Cvoid}, FT, Const{Float64}, Tuple{}, 1, true}
    Augmented = Compiler.AugmentedForwardThunk{Ptr{Cvoid}, FT, Const{Float64}, Tuple{}, 1, true, Float64}
    Adjoint = Compiler.AdjointThunk{Ptr{Cvoid}, FT, Const{Float64}, Tuple{}, 1, Float64}
    tape = Ref{Float64}()
    for st in (
            Compiler.ForwardTask{Forward, FT, S, false}(Forward(C_NULL), Const(nt_sumsq)),
            Compiler.AugmentedTask{Augmented, FT, typeof(tape), S, false}(Augmented(C_NULL), Const(nt_sumsq), tape),
            Compiler.ReverseTask{Adjoint, FT, typeof(tape), S, false}(Adjoint(C_NULL), Const(nt_sumsq), tape),
        )
        T = typeof(st)
        R = Compiler.rederived_body(T)
        @test R.name === T.name
        @test R.parameters[end] === true
        @test R.parameters[1:(end - 1)] == T.parameters[1:(end - 1)]
        @test fieldtypes(R) == fieldtypes(T)
        rst, rdst = Compiler.rederived_task(st, st)
        @test typeof(rst) === R
        @test typeof(rdst) === R
        @test Compiler.task_fields(rst) === Compiler.task_fields(st)
        # Already the one to differentiate
        @test Compiler.rederived_task(rst, rdst)[1] === rst
    end
    # A task that holds no derivative of a body
    @test Compiler.rederived_task(nt_sumsq, nt_sumsq) === (nt_sumsq, nt_sumsq)
end

@testset "A rule's calls keep their arguments in the code an outer derivative takes up" begin
    # The code Enzyme keeps of a derivative, for an outer derivative to differentiate, holds
    # calls that only a rule can differentiate: those of `threading_run` among them. The
    # rule reads their arguments by position, so none may have been folded into the callee.
    LLVM = Enzyme.Compiler.LLVM
    augmented, _ = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(nt_dynamic)}, Active, Duplicated{Vector{Float64}})
    kept = Enzyme.Compiler.cached_autodiff(augmented.primal)
    @test kept !== nothing
    ruled = String[]
    folded = String[]
    LLVM.Context() do _
        mod = parse(LLVM.Module, Vector{UInt8}(codeunits(kept.bitcode)))
        for fn in LLVM.functions(mod)
            attrs = Dict{String, String}()
            for attr in collect(LLVM.function_attributes(fn))
                attr isa LLVM.StringAttribute || continue
                attrs[LLVM.kind(attr)] = LLVM.value(attr)
            end
            get(attrs, "enzyme_math", "") in Enzyme.Compiler.parallel_enzyme_math || continue
            push!(ruled, attrs["enzyme_math"])
            haskey(attrs, "enzyme_parmremove") && push!(folded, LLVM.name(fn))
        end
        LLVM.dispose(mod)
    end
    @test "jl_threadsfor" in ruled
    @test isempty(folded)
end

# An outer derivative of a loop reaches the body's derivative in split mode: one callable
# writes its tape, another one reads it later. Split mode on its own, without threads:
# the tape goes through a function Julia does not inline.
function nt_overwriting!(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]^2
        # What the loop read is gone afterwards, so the reverse pass takes it from the tape
        x[i] = 0.0
    end
    return s
end

@inline nt_split(f::F) where {F} =
    autodiff_thunk(ReverseSplitNoPrimal, Const{F}, Active, Duplicated{Vector{Float64}})

@noinline function nt_split_reverse(f::F, x, dx, tape) where {F}
    nt_split(f)[2](Const(f), Duplicated(x, dx), 1.0, tape)
    return nothing
end

function nt_split_gradient!(f::F, dx, x0) where {F}
    x = copy(x0)
    tape, _, _ = nt_split(f)[1](Const(f), Duplicated(x, dx))
    nt_split_reverse(f, x, dx, tape)
    return nothing
end

@testset "An outer derivative keeps the tangent of a tape" begin
    x = collect(1.0:6.0)
    v = collect(6.0:-1.0:1.0)
    @testset "$f" for f in (nt_sumsq, nt_overwriting!)
        # A tape that caches numbers is not constant: called constant, its tangent is
        # dropped, and the second derivative below is zero with no error.
        TapeType = EnzymeRules.tape_type(nt_split(f)[1])
        @test !Enzyme.Compiler.guaranteed_const(TapeType)

        dx = zero(x)
        nt_split_gradient!(f, dx, x)
        @test dx == 2 .* x

        dx = zero(x)
        ddx = zero(x)
        autodiff(Forward, Const(nt_split_gradient!), Const, Const(f), Duplicated(dx, ddx), Duplicated(x, v))
        @test dx == 2 .* x
        @test ddx == 2 .* v
    end
end

@testset "A tape's pointer to memory of unknown layout" begin
    Compiler = Enzyme.Compiler
    LLVM = Compiler.LLVM
    Memory = Core.LLVMPtr{Compiler.TapeMemory, 0}
    # It may point to values that carry derivatives
    @test !Compiler.guaranteed_const(Memory)
    @test !Compiler.guaranteed_const(Tuple{UInt64, Memory})
    # A buffer of integers does not
    @test Compiler.guaranteed_const(Tuple{UInt64, Core.LLVMPtr{UInt64, 0}})
    # Its type tree says only that it is a pointer. With typed pointers (Julia 1.10) a tape
    # named the table of a cache's chunks a pointer to pointers to elements, while the
    # slots after the first one can hold offsets, which the reverse pass reads as integers:
    # under `--check-bounds=yes` the loop bodies above keep such a cache.
    ctx = LLVM.Context()
    dl = string(LLVM.DataLayout(LLVM.JITTargetMachine()))
    @test string(Enzyme.typetree(Memory, ctx, dl)) == "{[-1]:Pointer}"
    @test string(Enzyme.typetree(Core.LLVMPtr{Float64, 0}, ctx, dl)) == "{[-1]:Pointer, [-1,-1]:Float@double}"
    LLVM.Context() do _
        table = LLVM.PointerType(LLVM.PointerType(LLVM.DoubleType()))
        buffer = LLVM.PointerType(LLVM.DoubleType())
        @test Compiler.tape_type(LLVM.StructType([LLVM.Int64Type(), table])) == Tuple{UInt64, Memory}
        if LLVM.is_opaque(buffer)
            @test Compiler.tape_type(LLVM.StructType([LLVM.Int64Type(), buffer])) == Tuple{UInt64, Memory}
        else
            @test Compiler.tape_type(LLVM.StructType([LLVM.Int64Type(), buffer])) == Tuple{UInt64, Core.LLVMPtr{Float64, 0}}
        end
    end
end

@testset "The loop bodies an outer derivative differentiates" begin
    Compiler = Enzyme.Compiler
    Thunk = Compiler.ForwardModeThunk{Ptr{Cvoid}, Const{typeof(nt_sumsq)}, Const{Nothing}, Tuple{Const{Int}}, 1, false}
    settings = ((false, false), false, false)
    for T in (
            Compiler.ForwardPFor{Thunk, Const{typeof(nt_sumsq)}, settings, false},
            Compiler.AugmentedPFor{Thunk, Const{typeof(nt_sumsq)}, true, false, Vector{Float64}, settings, false},
            Compiler.ReversePFor{Thunk, Const{typeof(nt_sumsq)}, false, true, Ptr{Float64}, settings, false},
        )
        R = Compiler.rederived_body(T)
        @test R !== T
        @test R.name === T.name
        @test R.parameters[end] === true
        @test R.parameters[1:(end - 1)] == T.parameters[1:(end - 1)]
        @test fieldtypes(R) == fieldtypes(T)
        @test sizeof(R) == sizeof(T)
        # Already the one to differentiate
        @test Compiler.rederived_body(R) === R
    end
    # Any other loop body is differentiated as it is
    @test Compiler.rederived_body(typeof(nt_sumsq)) === typeof(nt_sumsq)
    @test Compiler.rederived_body(Any) === Any
end

# A package image is usually written by a process with one thread, and what it keeps of a
# derivative runs in sessions with any number. The packages go into a directory of their
# own, first on the children's load path, as in test/precompile.jl.
const NT_KEPT_PACKAGE = """
module EnzymeNestedThreadsKept
using Enzyme
function sumsq(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]^2
    end
    return s
end
function threaded(x)
    out = zeros(64)
    Threads.@threads for i in 1:64
        @inbounds out[i] = sumsq(x)
    end
    return sum(out)
end
function gradient!(dx, x)
    autodiff(Reverse, Const(threaded), Active, Duplicated(x, dx))
    return nothing
end
directional(x, v) = autodiff(Forward, Const(threaded), Duplicated(x, v))[1]
function forward_over_reverse(x, v)
    dx = zero(x)
    ddx = zero(x)
    autodiff(Forward, Const(gradient!), Const, Duplicated(dx, ddx), Duplicated(x, v))
    return dx, ddx
end
function reverse_over_forward(x, v)
    hv = zero(x)
    autodiff(Reverse, Const(directional), Active, Duplicated(x, hv), Const(v))
    return hv
end
const X = collect(1.0:6.0)
const V = collect(6.0:-1.0:1.0)
const EXPECTED = ((128 .* X, 128 .* V), 128 .* V)
const WORKLOAD = (forward_over_reverse(X, V), reverse_over_forward(X, V))
end
"""

# Prints: whether the session has package images, whether the values the image holds and
# the first calls are right, how many derivatives the session compiled, and how many of 300
# calls of each derivative were wrong.
const NT_KEPT_SESSION = """
print(Int(Base.JLOptions().use_pkgimages), " ")
flush(stdout)
using Enzyme, EnzymeNestedThreadsKept
const K = EnzymeNestedThreadsKept
first_calls = (K.forward_over_reverse(K.X, K.V), K.reverse_over_forward(K.X, K.V))
compiled = Enzyme.Compiler.retained_derivative_stats().compiled
wrong = [0, 0]
for _ in 1:300
    K.forward_over_reverse(K.X, K.V) == K.EXPECTED[1] || (wrong[1] += 1)
    K.reverse_over_forward(K.X, K.V) == K.EXPECTED[2] || (wrong[2] += 1)
end
print(K.WORKLOAD == K.EXPECTED, " ", first_calls == K.EXPECTED, " ", compiled, " ", wrong[1], " ", wrong[2])
"""

# Options under which Julia ignores the package images it has, which the test process may
# be running under.
function nt_ignores_pkgimages(arg::AbstractString)
    return startswith(arg, "--check-bounds") || startswith(arg, "--code-coverage") ||
        startswith(arg, "--track-allocation") || startswith(arg, "--inline") ||
        startswith(arg, "--pkgimages") || startswith(arg, "--threads") || startswith(arg, "-t")
end

function nt_kept_session(load_path, threads)
    exec = String[arg for arg in Base.julia_cmd().exec if !nt_ignores_pkgimages(arg)]
    cmd = addenv(
        `$(Cmd(exec)) --pkgimages=yes --threads=$threads --project=$(Base.active_project()) --startup-file=no -e $NT_KEPT_SESSION`,
        "JULIA_LOAD_PATH" => join([load_path, "@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":"),
    )
    out = IOBuffer()
    err = IOBuffer()
    ok = success(pipeline(cmd; stdout = out, stderr = err))
    ok || println(stderr, "Child process failed:\n", String(take!(err)))
    return ok, split(String(take!(out)))
end

@testset "A package image keeps them for any number of threads" begin
    load_path = mktempdir()
    try
        write(joinpath(load_path, "EnzymeNestedThreadsKept.jl"), NT_KEPT_PACKAGE)
        # The first session, with one thread, is the one that writes the image.
        @testset "a session with $threads thread(s)" for threads in (1, 2, 4)
            ok, fields = nt_kept_session(load_path, threads)
            @test ok
            @test length(fields) == 6
            length(fields) == 6 || continue
            @test fields[1] == "1"
            @test fields[2] == "true"
            @test fields[3] == "true"
            # Where an image keeps derivatives, the session links them and compiles none.
            Enzyme.Compiler.retention_supported() && @test fields[4] == "0"
            @test fields[5] == "0"
            @test fields[6] == "0"
        end
    finally
        try
            rm(load_path; recursive = true, force = true)
        catch  # Windows may still hold the image files open
        end
    end
end
