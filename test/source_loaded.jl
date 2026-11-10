using Enzyme, Test, TOML

# Enzyme loaded from source rather than from its package image. `jl_code_for_staged` runs a
# generator in the world its method was defined in, so a generator that reads a function
# Enzyme defines later sees, in that world, a const binding not defined yet. Julia 1.12 and
# later refuse a generator that read ("UndefVarError: `vaEltypes` not defined in `Enzyme`",
# from `rev_with_return` in src/rules/jitrules.jl). A package image gives every binding of
# the module the world it was loaded in, so the order of definitions matters only when the
# source is loaded: under `--compiled-modules=existing` for a package with no cache (how a
# benchmark harness met it), `--compiled-modules=no`, or after a precompile that failed.
#
# The child below gets a copy of this Enzyme under a path no cache was written for, and runs
# with `--compiled-modules=existing`: every other dependency still loads from its image, built
# under this process's flags, and Enzyme and EnzymeCore from source. It then
# differentiates through each family of generated functions that generate code from Enzyme's
# own helpers. Julia 1.10 has neither the flag nor world-dependent bindings.

const SOURCE_LOADED_CHILD = raw"""
using Enzyme
origin = Base.pkgorigins[Base.PkgId(Enzyme)]
println("FROM_SOURCE ", origin.cachepath === nothing, " ", pkgdir(Enzyme) == ENV["ENZYME_SOURCE_COPY"])

function case(f, name)
    ok = try
        f()
    catch err
        println(stderr, name, ": ", sprint(showerror, err))
        false
    end
    println("CASE ", name, " ", ok === true ? "ok" : "FAIL")
end

# The splat of a `Vector{Colon}` into `setindex!` is a `Core._apply_iterate` that Enzyme
# differentiates at run time through `runtime_iterate_*`, whose generators build the call
# with `vaEltypes` (Bijectors' `_set_lastindex!` has this shape).
function splat_setindex(x, idx)
    y = zeros(length(x))
    setindex!(y, x, idx...)
    return sum(abs2, y)
end
x = [1.0, 2.0, 3.0]
idx = [Colon()]

case("splat reverse") do
    dx = zero(x)
    autodiff(Reverse, splat_setindex, Active, Duplicated(x, dx), Const(idx))
    dx == 2 .* x
end
case("splat reverse with primal") do
    dx = zero(x)
    res = autodiff(ReverseWithPrimal, splat_setindex, Active, Duplicated(x, dx), Const(idx))
    dx == 2 .* x && res[2] == 14.0
end
case("splat reverse, runtime activity") do
    dx = zero(x)
    autodiff(set_runtime_activity(Reverse), splat_setindex, Active, Duplicated(x, dx), Const(idx))
    dx == 2 .* x
end
case("splat forward") do
    autodiff(Forward, splat_setindex, Duplicated(x, [0.0, 1.0, 0.0]), Const(idx))[1] == 4.0
end
case("splat batched forward") do
    d = autodiff(Forward, splat_setindex, BatchDuplicated(x, ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])), Const(idx))[1]
    d[1] == 2.0 && d[2] == 6.0
end

# A call Julia dispatches at run time (`runtime_generic_*`).
@noinline sumsq(v) = sum(abs2, v)
dynamic(v) = Base.inferencebarrier(sumsq)(v) * 3.0
case("dynamic call reverse") do
    dx = zero(x)
    autodiff(Reverse, dynamic, Active, Duplicated(x, dx))
    dx == 6 .* x
end
case("dynamic call forward") do
    autodiff(Forward, dynamic, Duplicated(x, [1.0, 0.0, 0.0]))[1] == 6.0
end

# One-hot seeds of forward-mode `gradient` (`onehot_internal`).
case("forward gradient") do
    gradient(Forward, v -> sum(abs2, v), x)[1] == 2 .* x
end

# Split mode (`thunk`), and an active argument through a closure.
cube(v) = v[1]^3 + v[2]
case("split mode") do
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cube)}, Active, Duplicated{Vector{Float64}})
    v = [2.0, 5.0]
    dv = zero(v)
    tape, primal, _ = fwd(Const(cube), Duplicated(v, dv))
    rev(Const(cube), Duplicated(v, dv), 1.0, tape)
    primal == 13.0 && dv == [12.0, 1.0]
end

# `make_zero` of a struct.
struct HoldsArrays
    a::Vector{Float64}
    b::Tuple{Float64, Vector{Float64}}
end
case("make_zero") do
    z = make_zero(HoldsArrays([1.0, 2.0], (3.0, [4.0])))
    z.a == [0.0, 0.0] && z.b[1] == 0.0 && z.b[2] == [0.0]
end

# A broadcast, through Enzyme's broadcast overrides.
case("broadcast") do
    dx = zero(x)
    autodiff(Reverse, v -> sum(exp.(v) .* v), Active, Duplicated(x, dx))
    dx ≈ exp.(x) .* (1 .+ x)
end

# Forward over Reverse: the inner derivative is compiled while the outer one is inferred.
case("forward over reverse") do
    inner(v) = gradient(Reverse, w -> w[1]^2 * w[2], v)[1]
    hv = autodiff(Forward, inner, Duplicated([2.0, 3.0], [1.0, 0.0]))[1]
    hv == [6.0, 4.0]
end
"""

const SOURCE_LOADED_CASES = [
    "splat reverse",
    "splat reverse with primal",
    "splat reverse, runtime activity",
    "splat forward",
    "splat batched forward",
    "dynamic call reverse",
    "dynamic call forward",
    "forward gradient",
    "split mode",
    "make_zero",
    "broadcast",
    "forward over reverse",
]

@testset "Enzyme loaded from source" begin
    if VERSION < v"1.12"
        @test_skip "needs --compiled-modules=existing and world-dependent bindings (Julia 1.12)"
    else
        load_path = mktempdir()
        try
            # Everything Enzyme reads when it loads: its project, source, extensions, the
            # dogfood declarations under deps/, and its EnzymeCore, which then loads from
            # source too (as it did for the benchmark's checkout).
            enzyme_copy = joinpath(load_path, "Enzyme")
            mkpath(joinpath(enzyme_copy, "lib"))
            for entry in ("Project.toml", "src", "ext", "deps", joinpath("lib", "EnzymeCore"))
                cp(joinpath(pkgdir(Enzyme), entry), joinpath(enzyme_copy, entry))
            end
            # An environment that is this one with Enzyme and EnzymeCore at the copy: the
            # manifest's other paths made absolute, its preferences (the libEnzyme build)
            # kept. A copy put first on the load path instead would see none of Enzyme's
            # dependencies, which Julia looks up only in the environment it found Enzyme in.
            project = Base.active_project()
            env = joinpath(load_path, "env")
            mkpath(env)
            proj = TOML.parsefile(project)
            delete!(proj, "sources")
            open(io -> TOML.print(io, proj), joinpath(env, "Project.toml"), "w")
            manifest_file = Base.project_file_manifest_path(project)
            manifest = TOML.parsefile(manifest_file)
            for (name, entries) in manifest["deps"], entry in entries
                if name == "Enzyme"
                    entry["path"] = enzyme_copy
                elseif name == "EnzymeCore" && haskey(entry, "path")
                    entry["path"] = joinpath(enzyme_copy, "lib", "EnzymeCore")
                elseif haskey(entry, "path")
                    entry["path"] = abspath(dirname(manifest_file), entry["path"])
                end
            end
            open(io -> TOML.print(io, manifest), joinpath(env, "Manifest.toml"), "w")
            prefs = joinpath(dirname(project), "LocalPreferences.toml")
            isfile(prefs) && cp(prefs, joinpath(env, "LocalPreferences.toml"))
            cmd = addenv(
                `$(Base.julia_cmd()) --compiled-modules=existing --project=$env --startup-file=no -e $SOURCE_LOADED_CHILD`,
                "ENZYME_SOURCE_COPY" => enzyme_copy,
            )
            out = IOBuffer()
            err = IOBuffer()
            ok = success(pipeline(cmd; stdout = out, stderr = err))
            lines = split(String(take!(out)), '\n')
            # A child that found a cache for the copy, or loaded the parent's Enzyme, would not
            # test what this says it does.
            @test "FROM_SOURCE true true" in lines
            passed = [("CASE $name ok" in lines) for name in SOURCE_LOADED_CASES]
            for p in passed
                @test p
            end
            @test ok
            # Each failed case wrote its error to the child's stderr.
            (ok && all(passed)) || println(stderr, "Enzyme loaded from source, child stderr:\n", String(take!(err)))
        finally
            try
                rm(load_path; recursive = true, force = true)
            catch  # Windows may still hold files open
            end
        end
    end
end
