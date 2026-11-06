using Enzyme, Test

# Enzyme keeps what it compiles in globals of its own, and a thunk is held as the address of
# the code the JIT emitted for it. Neither survives the process that produced it, so a
# package that differentiates while it precompiles could write an image that refers to a dead
# session (EnzymeAD/Enzyme.jl#1549). What follows pins down that it no longer does for the
# derivatives `autodiff` compiles; these cases are what precompilation support must not
# regress.
#
# Enzyme keeps the caches its own workload fills out of an image by emptying them before the
# image is written. The derivatives of combined reverse mode and forward mode are compiled
# at their first call, and the code that calls one holds no address of that session (see
# `Enzyme.Compiler.LazyThunk`). What it holds instead, where `retention_supported()`, is the
# derivative with every address replaced by a name, which a later session links (see
# `Enzyme.Compiler.retained_derivatives!`, and the testsets at the end of this file). A split-mode thunk (`autodiff_thunk` with `ReverseSplit...`) is still compiled
# while its caller is inferred, because its type carries the tape type; while an image is
# written, the code of that call fetches the thunks at run time instead of holding them (see
# `Enzyme.Compiler.image_split_thunks`). The one-hot seeds of forward-mode `gradient` and
# `jacobian` are built by code that names objects by their addresses, which is replaced by
# plain Julia while an image is written (see `Enzyme.onehot_tuple`). The inner derivative of
# a nested one is still compiled while the outer one is inferred, as a constant of the
# inferred code; from Julia 1.11 that code sits in Julia's own cache, which writes what is
# valid in every later world into the image, so while an image is written it is made valid
# only in the world it was inferred in. Julia 1.10 reports it as newly inferred too, and
# widens that bound, so there Enzyme's interpreter keeps its results from being reported.

# Write the packages into a directory of their own and put it first on the child's load
# path. The depot is the one this test runs under, so the packages precompile next to the
# Enzyme the parent is testing instead of rebuilding the world in a fresh depot.
function precompile_test_harness(f)
    load_path = mktempdir()
    try
        f(load_path)
    finally
        try
            rm(load_path; recursive = true, force = true)
        catch  # Windows may still hold the image files open
        end
    end
    return nothing
end

# Options under which Julia ignores the package images it has and compiles from source
# instead. The test process may well be running under them, `--check-bounds=yes` and coverage
# being what CI tests with, and `Base.julia_cmd` would pass them on to the children, where
# they would leave nothing here testing what it says it does.
function ignores_pkgimages(arg::AbstractString)
    return startswith(arg, "--check-bounds") || startswith(arg, "--code-coverage") ||
        startswith(arg, "--track-allocation") || startswith(arg, "--inline") ||
        startswith(arg, "--pkgimages")
end

function child_julia_cmd()
    exec = String[arg for arg in Base.julia_cmd().exec if !ignores_pkgimages(arg)]
    push!(exec, "--pkgimages=yes")
    return Cmd(exec)
end

# Each child reports first whether it does have package images, so that a child which never
# loaded one cannot quietly stand in for one that did.
const CHILD_PREAMBLE = """
print(Int(Base.JLOptions().use_pkgimages), " ", Int(Base.JLOptions().code_coverage), " ")
flush(stdout)
"""

function child_command(load_path, code)
    return addenv(
        `$(child_julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_LOAD_PATH" => join([load_path, "@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":"),
    )
end

# Run a child, and give back whether it exited cleanly along with the fields it printed. The
# preamble is flushed before anything else runs, so those two come back even from a child
# that dies partway.
function run_child(load_path, code)
    out = IOBuffer()
    err = IOBuffer()
    ok = success(
        pipeline(
            child_command(load_path, CHILD_PREAMBLE * code);
            stdout = out, stderr = err,
        )
    )
    # Show why a child failed; its output is all there is to go by.
    ok || println(stderr, "Child process failed:\n", String(take!(err)))
    fields = split(String(take!(out)))
    @test length(fields) >= 2 && fields[1] == "1" && fields[2] == "0"
    return ok, fields[3:end]
end

const USES_AD = """
module EnzymePrecompileUsesAD
using Enzyme

scalar(x) = x * x * x
harmonic(x) = sin(x) / x

# Only differentiated once the package is loaded, never while it precompiles.
rev(x) = Enzyme.autodiff(Reverse, scalar, Active(x))[1][1]
fwd(x) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]
end
"""

const AD_WHILE_PRECOMPILING = """
module EnzymePrecompileAtBuild
using Enzyme

scalar(x) = x * x * x
harmonic(x) = sin(x) / x

rev(x) = Enzyme.autodiff(Reverse, scalar, Active(x))[1][1]
fwd(x) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]

# Differentiate as the package precompiles, so the thunks are compiled in the process that
# writes the image and whatever refers to them is what the image carries.
const PRECOMPILED = (rev(2.0), fwd(2.0))
end
"""

const rev_at_3 = 3 * 3.0^2
const fwd_at_3 = (cos(3.0) * 3.0 - sin(3.0)) / 3.0^2

@testset "a package that differentiates only once it is loaded" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompileUsesAD.jl"), USES_AD)
        code = """
        using Enzyme, EnzymePrecompileUsesAD
        print(EnzymePrecompileUsesAD.rev(3.0), " ", EnzymePrecompileUsesAD.fwd(3.0))
        """

        # The first child writes the image, the second loads it.
        for _ in 1:2
            ok, fields = run_child(load_path, code)
            @test ok
            if ok
                rev3, fwd3 = fields
                @test parse(Float64, rev3) ≈ rev_at_3
                @test parse(Float64, fwd3) ≈ fwd_at_3
            end
        end
    end
end

@testset "a package that differentiates while it precompiles" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompileAtBuild.jl"), AD_WHILE_PRECOMPILING)

        # Differentiating during precompilation itself works, and the values it computed are
        # in the image: the first child precompiles the package, the second only loads it.
        values_code = """
        using Enzyme, EnzymePrecompileAtBuild
        print(EnzymePrecompileAtBuild.PRECOMPILED[1], " ", EnzymePrecompileAtBuild.PRECOMPILED[2])
        """
        for _ in 1:2
            ok, fields = run_child(load_path, values_code)
            @test ok
            if ok
                rev2, fwd2 = fields
                @test parse(Float64, rev2) ≈ 3 * 2.0^2
                @test parse(Float64, fwd2) ≈ (cos(2.0) * 2.0 - sin(2.0)) / 2.0^2
            end
        end

        # Differentiating again from a loaded image works too. The derivatives were first
        # called while the package precompiled, so the code the image holds kept no address
        # of that process: it links them again here, or where that is not supported, compiles them.
        call_code = """
        using Enzyme, EnzymePrecompileAtBuild
        print(EnzymePrecompileAtBuild.rev(3.0), " ", EnzymePrecompileAtBuild.fwd(3.0))
        """
        ok, fields = run_child(load_path, call_code)
        @test ok
        if ok
            rev3, fwd3 = fields
            @test parse(Float64, rev3) ≈ rev_at_3
            @test parse(Float64, fwd3) ≈ fwd_at_3
        end
    end
end

const MODES_WHILE_PRECOMPILING = """
module EnzymePrecompileModes
using Enzyme

f(v) = sum(abs2, v) + sin(v[1])
g(v) = v .* v .+ sin(v[1])

# Combined reverse mode of a function of a vector, as `gradient` takes it.
rev_gradient(x) = Enzyme.gradient(Reverse, f, x)[1]

# Split mode, as `autodiff_thunk` hands it out.
function split_gradient(x)
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(x)})
    dx = zero(x)
    tape, _, _ = fwd(Const(f), Duplicated(x, dx))
    rev(Const(f), Duplicated(x, dx), 1.0, tape)
    return dx
end

# Reverse mode of a function returning a vector, which `jacobian` takes in split mode.
rev_jacobian(x) = vec(Enzyme.jacobian(Reverse, g, x)[1])

# Forward mode seeded with one-hot vectors.
fwd_gradient(x) = Enzyme.gradient(Forward, f, x)[1]

# Forward over reverse: the inner derivative is compiled while the outer one is.
inner(x) = Enzyme.gradient(Reverse, f, x)[1]
hvp(x) = Enzyme.autodiff(Forward, inner, Duplicated(x, [1.0, 0.0, 0.0]))[1]

const X = [0.3, -0.2, 0.1]
const PRECOMPILED = (rev_gradient(X), split_gradient(X), rev_jacobian(X), fwd_gradient(X), hvp(X))
end
"""

const modes_x = [0.5, 0.25, -1.0]
const modes_gradient = [2 * modes_x[1] + cos(modes_x[1]), 2 * modes_x[2], 2 * modes_x[3]]
const modes_expected = (
    rev_gradient = modes_gradient,
    split_gradient = modes_gradient,
    # Column by column: the first input reaches every output through `sin(v[1])`.
    rev_jacobian = [
        modes_gradient[1], cos(modes_x[1]), cos(modes_x[1]),
        0.0, modes_gradient[2], 0.0,
        0.0, 0.0, modes_gradient[3],
    ],
    fwd_gradient = modes_gradient,
    hvp = [2 - sin(modes_x[1]), 0.0, 0.0],
)

@testset "a package that takes derivatives of every kind while it precompiles" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompileModes.jl"), MODES_WHILE_PRECOMPILING)

        # The first child precompiles the package, the second only loads it.
        values_code = """
        using Enzyme, EnzymePrecompileModes
        print(join(reduce(vcat, EnzymePrecompileModes.PRECOMPILED), " "))
        """
        for _ in 1:2
            ok, fields = run_child(load_path, values_code)
            @test ok
            @test length(fields) == 21
        end

        # Each kind in a session of its own, so that one that crashes does not hide the rest.
        for (name, expected) in pairs(modes_expected)
            @testset "$name" begin
                call_code = """
                using Enzyme, EnzymePrecompileModes
                print(join(EnzymePrecompileModes.$name($modes_x), " "))
                """
                ok, fields = run_child(load_path, call_code)
                @test ok && length(fields) == length(expected) &&
                    all(isapprox(parse(Float64, s), e; atol = 1.0e-12) for (s, e) in zip(fields, expected))
            end
        end
    end
end

const PLAIN_AND_NESTED = """
module EnzymePrecompilePlainAndNested
using Enzyme

@noinline callee(x) = x^3
f(v) = sum(callee, v)
gradient_of(g, x) = Enzyme.gradient(Reverse, g, x)[1]
inner(x) = Enzyme.gradient(Reverse, f, x)[1]
hessian_vector(x, v) = Enzyme.autodiff(Forward, inner, Duplicated(x, v))[1]

const PRECOMPILED = (gradient_of(f, [1.0, 2.0, 3.0]), hessian_vector([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]))
end
"""

@testset "a package that takes a plain and a nested derivative of one function" begin
    # The inner derivative of a nested one holds the address of what it calls. On Julia 1.10
    # the code of that call used to reach the image as the native code of `gradient`, for
    # every later session to jump to. Only the nested derivative goes through that code, so
    # the plain one is what crashed.
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompilePlainAndNested.jl"), PLAIN_AND_NESTED)
        code = """
        using Enzyme, EnzymePrecompilePlainAndNested
        const P = EnzymePrecompilePlainAndNested
        print(join(P.gradient_of(P.f, [0.5, 1.5, 2.5, 3.5]), " "))
        """
        ok, fields = run_child(load_path, code)
        @test ok
        ok && @test parse.(Float64, fields) ≈ 3 .* [0.5, 1.5, 2.5, 3.5] .^ 2
    end
end

@testset "Enzyme's image holds the passes that act on branches" begin
    # The first derivative of a function with branches or selects runs passes that the
    # workload's `x^2` never reaches, such as `simplify_conditioned_selects!`. The workload
    # also differentiates `cos` so that Enzyme's image holds their code, and their helpers
    # are compiled once for every LLVM value type, so that code is all a later function
    # needs. Without both, the first Reverse gradient of a function calling `cos` or `log`
    # took 8-11 s on Julia 1.10, against about 2 s with registered Enzyme.
    precompile_test_harness() do load_path
        code = """
        using Enzyme
        function precompiled_specializations(f)
            world = Base.get_world_counter()
            n = 0
            for m in methods(f), mi in Base.specializations(m)
                mi isa Core.MethodInstance && isdefined(mi, :cache) || continue
                ci = mi.cache
                while true
                    if ci.min_world <= world <= ci.max_world && ci.invoke != C_NULL
                        n += 1
                        break
                    end
                    isdefined(ci, :next) || break
                    ci = ci.next
                end
            end
            return n
        end
        specializations(f) = sum(m -> Base.count(mi -> mi isa Core.MethodInstance, Base.specializations(m)), methods(f))
        f = Enzyme.Compiler.specialize_conditioned_select!
        native = precompiled_specializations(f)
        before = specializations(f)
        branchy(x) = x > 1.0 ? log(x) : exp(-x)
        grad = Enzyme.autodiff(Reverse, branchy, Active(2.0))[1][1]
        print(native, " ", before, " ", specializations(f), " ", grad)
        """
        ok, fields = run_child(load_path, code)
        @test ok
        if ok
            native, before, after, grad = fields
            @test parse(Int, native) > 0
            # Nothing of it was compiled again for this function.
            @test parse(Int, after) == parse(Int, before)
            @test parse(Float64, grad) ≈ 0.5
        end
    end
end

@testset "Enzyme's own precompilation" begin
    precompile_test_harness() do load_path
        # Enzyme differentiates in its `@compile_workload`, and the caches that fills are
        # globals of Enzyme's, so whatever is left in them is serialized into Enzyme's image
        # along with the addresses that session's JIT handed out. `clear_caches!` at the end
        # of the workload is what keeps them out of it.
        code = """
        using Enzyme
        C = Enzyme.Compiler
        sizes = (length(C.cache), length(C.autodiff_cache), length(Enzyme.tape_cache),
                 length(C.FRULE_CACHE), length(C.RRULE_CACHE), length(C.INACTIVE_CACHE),
                 length(C.ATTRIBUTE_PRESERVING_RULE_CACHE), length(C.READONLY_RULE_CACHE), length(C.NOALIAS_CACHE),
                 length(C.Interpreter.SigCache), length(C.ActivityCache),
                 length(C.ActivityMethodCache), Int(C.ActivityWorldCache[]),
                 length(C.JIT.hnd_string_map), length(C.JIT.hnd_int_map))
        print(sum(sizes), " ", Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
        """
        ok, fields = run_child(load_path, code)
        @test ok
        if ok
            cached, grad = fields
            # Nothing the session that built the image compiled or looked up is left.
            @test parse(Int, cached) == 0
            # And a session that starts from nothing differentiates.
            @test parse(Float64, grad) ≈ 8.0
        end

        # An emptied cache is not the whole of it. The workload differentiates a function
        # that stays in Enzyme, and Enzyme's own image holds the code of that call just as a
        # package image does. The derivative was first called while Enzyme precompiled, so
        # that code kept no address of that session, and asking for it again compiles it.
        workload_code = """
        using Enzyme
        mods = [getfield(Enzyme, n) for n in names(Enzyme; all = true) if
                startswith(string(n), "#") && isdefined(Enzyme, n) &&
                getfield(Enzyme, n) isa Module]
        fns = [m.f for m in mods if isdefined(m, :f)]
        if isempty(fns)
            print("none")   # the workload stopped leaving a function behind
        else
            print(Enzyme.autodiff(Reverse, first(fns), Active(2.0))[1][1])
        end
        """
        ok, fields = run_child(load_path, workload_code)
        if ok && fields == ["none"]
            @test_skip ok
        else
            @test ok
        end
    end
end

# How many specializations of three functions every derivative is compiled through hold native
# code that is valid in the current world, read before anything is differentiated. On Julia
# 1.10 and 1.12 alike, loading StaticArrays left none of these three with any, and the
# extension's image gives each back what Enzyme's own image has, in either load order. Not
# every compiler function is restored that completely (`thunk_generator` is not, on 1.12).
#
# `specialize_conditioned_select!` is reached only by a function with branches, so it is here
# for the part of the workload that has them. On Julia 1.10 loading StaticArrays discards it
# too, and an extension differentiating only `x^2` left it with none in either load order. On
# Julia 1.12 StaticArrays leaves it alone, so there this count cannot tell the two apart.
const PRECOMPILED_COMPILER = """
function precompiled_specializations(f)
    world = Base.get_world_counter()
    n = 0
    for m in methods(f), mi in Base.specializations(m)
        mi isa Core.MethodInstance && isdefined(mi, :cache) || continue
        ci = mi.cache
        while true
            if ci.min_world <= world <= ci.max_world && ci.invoke != C_NULL
                n += 1
                break
            end
            isdefined(ci, :next) || break
            ci = ci.next
        end
    end
    return n
end
C = Enzyme.Compiler
for f in (C.handle_compiled, C.lower_convention, C.check_ir!, C.specialize_conditioned_select!)
    print(precompiled_specializations(f), " ")
end
print(Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
"""

@testset "StaticArrays leaves Enzyme's compiler precompiled" begin
    # StaticArrays' `Base.cconvert(::Type{Ptr{T}}, ...)` methods match the `cconvert` calls
    # Enzyme's compiler makes with an argument inferred as `Any`, so loading it discards the
    # native code Enzyme's image holds for that compiler, in either load order, and the first
    # derivative compiled it all again. `EnzymeStaticArraysExt` runs `@compiler_workload`, as
    # Enzyme's own image does, so that its image holds the compiler as it stands with both
    # packages loaded.
    # Enzyme alone is the control: what its image holds is what the other two must keep.
    precompile_test_harness() do load_path
        for packages in ("Enzyme", "StaticArrays, Enzyme", "Enzyme, StaticArrays")
            ok, fields = run_child(load_path, "using $packages\n" * PRECOMPILED_COMPILER)
            @test ok
            if ok
                @test length(fields) == 5
                @test all(n -> parse(Int, n) > 0, fields[1:4])
                @test parse(Float64, fields[5]) ≈ 8.0
            end
        end
    end
end

const RETAINED = """
module EnzymeRetained
using Enzyme, LinearAlgebra
using Enzyme.EnzymeRules

@noinline callee(x) = x^3
@inline forced(x) = 2x
inlined(x) = 5x
middle(x) = inlined(x) + 1
through_callee(v) = sum(callee, v)
through_forced(v) = sum(forced, v)
through_middle(v) = sum(middle, v)

# The slope comes from a helper that only the reverse pass of the rule calls.
slope(x) = 4.0
ruled(x) = 4.0 * x
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(ruled)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? ruled(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, x.val)
end
function EnzymeRules.reverse(::EnzymeRules.RevConfig, ::Const{typeof(ruled)}, dret::Active, tape, x::Active)
    return (slope(tape) * dret.val,)
end
through_rule(v) = sum(ruled, v)

gradient_of(f, x) = Enzyme.gradient(Reverse, f, x)[1]

function split_gradient(x)
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(through_callee)}, Active, Duplicated{typeof(x)})
    dx = zero(x)
    tape, _, _ = fwd(Const(through_callee), Duplicated(x, dx))
    rev(Const(through_callee), Duplicated(x, dx), 1.0, tape)
    return dx
end

inner(x) = Enzyme.gradient(Reverse, through_callee, x)[1]
hessian_vector(x, v) = Enzyme.autodiff(Forward, inner, Duplicated(x, v))[1]

# The data is an input to the derivative, which holds neither its values nor its size.
least_squares(b, X, y) = 0.5 * sum(abs2, y .- X * b)
function least_squares_gradient(b, X, y)
    db = zero(b)
    Enzyme.autodiff(Reverse, least_squares, Active, Duplicated(b, db), Const(X), Const(y))
    return db
end

function workload()
    x = [1.0, 2.0, 3.0]
    return (
        gradient_of(through_callee, x), gradient_of(through_forced, x), gradient_of(through_middle, x),
        gradient_of(through_rule, x), split_gradient(x), hessian_vector(x, [1.0, 0.0, 0.0]),
        least_squares_gradient([1.0, 2.0], [1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, 2.0, 3.0]),
    )
end

const WORKLOAD = workload()
end
"""

# Defining these changes the result of everything in `RETAINED` that depends on one of them.
const RETAINED_CHANGES = """
EnzymeRetained.callee(x::Float64) = x^2
EnzymeRetained.forced(x::Float64) = 3x
EnzymeRetained.inlined(x::Float64) = 7x
EnzymeRetained.slope(x::Float64) = 9.0
"""

const RETAINED_PIRATE = """
module EnzymeRetainedPirate
using EnzymeRetained
$RETAINED_CHANGES
end
"""

# Other values and sizes than the image was built with.
const RETAINED_CHECK = """
const R = EnzymeRetained
x = [0.5, 1.5, 2.5, 3.5]
results = (
    R.gradient_of(R.through_callee, x), R.gradient_of(R.through_forced, x), R.gradient_of(R.through_middle, x),
    R.gradient_of(R.through_rule, x), R.split_gradient(x), R.hessian_vector(x, [1.0, 1.0, 0.0, 0.0]),
    R.least_squares_gradient([1.0, 2.0, 3.0], [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0; 1.0 0.0 1.0; 0.0 1.0 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]),
)
stats = Enzyme.Compiler.retained_derivative_stats()
print(stats.compiled, " ", stats.materialized, " ", isempty(stats.rejected), " ", isempty(stats.absent), " ")
print(join(reduce(vcat, results), " "))
"""

function retained_expected(; changed::Bool)
    x = [0.5, 1.5, 2.5, 3.5]
    X = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0; 1.0 0.0 1.0; 0.0 1.0 0.0]
    y = [1.0, 0.0, 1.0, 0.0, 1.0]
    b = [1.0, 2.0, 3.0]
    d_callee = changed ? 2 .* x : 3 .* x .^ 2
    d2_callee = changed ? [2.0, 2.0, 0.0, 0.0] : [6 * x[1], 6 * x[2], 0.0, 0.0]
    return vcat(
        d_callee, fill(changed ? 3.0 : 2.0, 4), fill(changed ? 7.0 : 5.0, 4),
        fill(changed ? 9.0 : 4.0, 4), d_callee, d2_callee, X' * (X * b .- y),
    )
end

# Run a session against the packages in `load_path`, and give back what it linked and compiled
# and whether its gradients are the expected ones.
function retained_session(load_path, setup, expected; threads = 1)
    code = "using Enzyme, EnzymeRetained\n$setup\n$RETAINED_CHECK"
    exec = `$(child_julia_cmd()) --threads=$threads --project=$(Base.active_project()) --startup-file=no -e $(CHILD_PREAMBLE * code)`
    cmd = addenv(exec, "JULIA_LOAD_PATH" => join([load_path, "@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":"))
    out = IOBuffer()
    err = IOBuffer()
    ok = success(pipeline(cmd; stdout = out, stderr = err))
    ok || println(stderr, "Child process failed:\n", String(take!(err)))
    @test ok
    fields = split(String(take!(out)))
    @test length(fields) >= 2 && fields[1] == "1" && fields[2] == "0"
    @test length(fields) == 6 + length(expected)
    length(fields) == 6 + length(expected) || return (compiled = -1, linked = -1, clean = false, close = false)
    grads = parse.(Float64, fields[7:end])
    return (
        compiled = parse(Int, fields[3]),
        linked = parse(Int, fields[4]),
        clean = fields[5] == "true" && fields[6] == "true",
        close = all(isapprox(g, e; atol = 1.0e-10) for (g, e) in zip(grads, expected)),
    )
end

# The number of derivatives `RETAINED_CHECK` takes from the image.
const RETAINED_KINDS = 7

@testset "a package image keeps the derivatives it compiled" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymeRetained.jl"), RETAINED)
        write(joinpath(load_path, "EnzymeRetainedPirate.jl"), RETAINED_PIRATE)
        retained = Enzyme.Compiler.retention_supported()
        unchanged = retained_expected(; changed = false)
        changed = retained_expected(; changed = true)

        @testset "fresh processes link it, for values and sizes the image never saw" begin
            for _ in 1:2
                s = retained_session(load_path, "", unchanged)
                @test s.close
                if retained
                    @test s.compiled == 0
                    @test s.linked == RETAINED_KINDS
                    @test s.clean
                end
            end
        end

        @testset "with threads, it takes the variant that accumulates atomically" begin
            s = retained_session(load_path, "", unchanged; threads = 2)
            @test s.close
            if retained
                @test s.compiled == 0
                @test s.linked == RETAINED_KINDS
            end
        end

        @testset "a method defined before its first call reaches it" begin
            s = retained_session(load_path, RETAINED_CHANGES, changed)
            @test s.close
            # Only the least squares gradient does not depend on what changed.
            retained && @test s.linked == 1
        end

        @testset "a package that changes a method reaches it" begin
            s = retained_session(load_path, "using EnzymeRetainedPirate", changed)
            @test s.close
            retained && @test s.linked == 1
        end

        @testset "a method defined after its first call reaches it" begin
            setup = """
            first_x = [1.0, 2.0]
            before = Enzyme.autodiff(Reverse, EnzymeRetained.through_callee, Active, Duplicated(first_x, zero(first_x)))
            $RETAINED_CHANGES
            """
            s = retained_session(load_path, setup, changed)
            @test s.close
        end

        @testset "options that change the code keep it from being used" begin
            s = retained_session(load_path, "Enzyme.API.strictAliasing!(false)", unchanged)
            @test s.close
            if retained
                @test s.linked == 0
                @test s.compiled >= RETAINED_KINDS
                @test !s.clean
            end
        end

        @testset "another element type compiles" begin
            code = """
            using Enzyme, EnzymeRetained
            g = EnzymeRetained.gradient_of(EnzymeRetained.through_callee, Float32[1, 2])
            stats = Enzyme.Compiler.retained_derivative_stats()
            print(stats.compiled, " ", stats.materialized, " ", join(g, " "))
            """
            ok, fields = run_child(load_path, code)
            @test ok
            @test parse(Int, fields[1]) == 1
            @test parse(Int, fields[2]) == 0
            @test parse.(Float32, fields[3:4]) == Float32[3, 12]
        end

        retained && @testset "what the image holds" begin
            code = """
            using Enzyme, EnzymeRetained
            const C = Enzyme.Compiler
            function cells()
                found = Any[]
                for m in methods(C.thunk), mi in Base.specializations(m)
                    mi === nothing && continue
                    occursin("EnzymeRetained.through_callee", string(mi.specTypes)) || continue
                    occursin("DEM_ReverseModeCombined", string(mi.specTypes)) || continue
                    ci = mi.cache
                    while !(ci.inferred isa String) && isdefined(ci, :next)
                        ci = ci.next
                    end
                    src = Base._uncompressed_ir(ci, ci.inferred)
                    for st in src.code, arg in (st isa Expr ? st.args : ())
                        arg isa C.LazyThunk && push!(found, arg)
                        if arg isa Expr
                            for a in arg.args
                                a isa C.LazyThunk && push!(found, a)
                            end
                        end
                    end
                end
                return found
            end
            function lock_cmpxchg(object)
                return any(i -> object[i] == 0xf0 && object[i + 2] == 0x0f && object[i + 3] == 0xb1, 1:(length(object) - 3))
            end
            cell = only(unique(objectid, cells()))
            r = cell.retained
            print(lock_cmpxchg(r.serial.object), " ", lock_cmpxchg(r.threaded.object), " ")
            print(C.address_in_object(r.serial.object) === nothing, " ", C.address_in_object(r.threaded.object) === nothing)
            """
            ok, fields = run_child(load_path, code)
            @test ok
            # Only the variant for several threads accumulates with a locked exchange.
            @test fields[1] == "false"
            @test fields[2] == "true"
            # Neither holds an address that is mapped in this process, or in the one that wrote it.
            @test fields[3] == "true"
            @test fields[4] == "true"
        end
    end
end

# A rule whose augmented primal has a constant result. From an image on Julia 1.13 its
# `CodeInstance` already has the constant-return entry point, and codegen emits nothing for it,
# so a session that has to compile through it stops at an assertion in GPUCompiler. This happens
# with or without a derivative kept in the image, which is used in place of compiling when it can
# be (Enzyme todo `0wovnyy`).
const RETAINED_CONST_RULE = """
module EnzymeRetainedConstRule
using Enzyme
using Enzyme.EnzymeRules
ruled(x) = 4.0 * x
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(ruled)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? ruled(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(::EnzymeRules.RevConfig, ::Const{typeof(ruled)}, dret::Active, tape, x::Active)
    return (4.0 * dret.val,)
end
through_rule(v) = sum(ruled, v)
gradient_of(f, x) = Enzyme.gradient(Reverse, f, x)[1]
const WORKLOAD = gradient_of(through_rule, [1.0, 2.0])
end
"""

@testset "a rule with a constant result, held in an image" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymeRetainedConstRule.jl"), RETAINED_CONST_RULE)
        call = """
        print(join(EnzymeRetainedConstRule.gradient_of(EnzymeRetainedConstRule.through_rule, [1.0, 2.0, 3.0]), " "))
        """
        # Linked from the image, where that is supported.
        ok, fields = run_child(load_path, "using Enzyme, EnzymeRetainedConstRule\n" * call)
        if Enzyme.Compiler.retention_supported() || VERSION.minor != 13
            @test ok
            ok && @test parse.(Float64, fields) == [4.0, 4.0, 4.0]
        end
        # Compiled in the session, as it is when the derivative cannot be used.
        out = IOBuffer()
        code = "using Enzyme, EnzymeRetainedConstRule\nEnzyme.API.strictAliasing!(false)\n" * call
        cmd = child_command(load_path, CHILD_PREAMBLE * code)
        compiled = success(pipeline(cmd; stdout = out, stderr = devnull))
        if VERSION.minor == 13
            @test_broken compiled
        else
            @test compiled
        end
    end
end

const RETAINED_OFF = """
module EnzymeNotRetained
using Enzyme
Enzyme.Compiler.RetainDerivatives[] = false
f(v) = sum(abs2, v)
const WORKLOAD = Enzyme.gradient(Reverse, f, [1.0, 2.0])
Enzyme.Compiler.RetainDerivatives[] = true
end
"""

@testset "a package can keep its derivatives out of its image" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymeNotRetained.jl"), RETAINED_OFF)
        code = """
        using Enzyme, EnzymeNotRetained
        g = Enzyme.gradient(Reverse, EnzymeNotRetained.f, [3.0, 4.0, 5.0])[1]
        stats = Enzyme.Compiler.retained_derivative_stats()
        print(stats.compiled, " ", stats.materialized, " ", join(g, " "))
        """
        ok, fields = run_child(load_path, code)
        @test ok
        @test parse(Int, fields[1]) == 1
        @test parse(Int, fields[2]) == 0
        @test parse.(Float64, fields[3:5]) == [6.0, 8.0, 10.0]
    end
end
