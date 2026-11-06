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
# at their first call, and one first called while precompiling is not kept by the code that
# calls it (see `Enzyme.Compiler.LazyThunk`), so no address of that session reaches an image
# that way. A split-mode thunk (`autodiff_thunk` with `ReverseSplit...`) is still compiled
# while its caller is inferred, because its type carries the tape type; while an image is
# written, the code of that call fetches the thunks at run time instead of holding them (see
# `Enzyme.Compiler.image_split_thunks`). The one-hot seeds of forward-mode `gradient` and
# `jacobian` are built by code that names objects by their addresses, which is replaced by
# plain Julia while an image is written (see `Enzyme.onehot_tuple`). The inner derivative of
# a nested one is still compiled while the outer one is inferred, as a constant of the
# inferred code; from Julia 1.11 that code sits in Julia's own cache, which writes what is
# valid in every later world into the image, so while an image is written it is made valid
# only in the world it was inferred in.

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
        # of that process and compiles them again here.
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
for f in (C.handle_compiled, C.lower_convention, C.check_ir!)
    print(precompiled_specializations(f), " ")
end
print(Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
"""

@testset "StaticArrays leaves Enzyme's compiler precompiled" begin
    # StaticArrays' `Base.cconvert(::Type{Ptr{T}}, ...)` methods match the `cconvert` calls
    # Enzyme's compiler makes with an argument inferred as `Any`, so loading it discards the
    # native code Enzyme's image holds for that compiler, in either load order, and the first
    # derivative compiled it all again. `EnzymeStaticArraysExt` runs Enzyme's workload once
    # more so that its own image holds the compiler as it stands with both packages loaded.
    # Enzyme alone is the control: what its image holds is what the other two must keep.
    precompile_test_harness() do load_path
        for packages in ("Enzyme", "StaticArrays, Enzyme", "Enzyme, StaticArrays")
            ok, fields = run_child(load_path, "using $packages\n" * PRECOMPILED_COMPILER)
            @test ok
            if ok
                @test length(fields) == 4
                @test all(n -> parse(Int, n) > 0, fields[1:3])
                @test parse(Float64, fields[4]) ≈ 8.0
            end
        end
    end
end
