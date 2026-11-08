using Enzyme, Test

# The dogfood repair registry: one file per repair under deps/dogfood_repairs,
# read through Enzyme.dogfood_repairs, has_dogfood_repair and
# require_dogfood_repairs. A repair in the native library counts only when the
# loaded libEnzyme is the build deps/dogfood_libenzyme.toml declares.

const TOML = Enzyme.TOML
const ROOT = pkgdir(Enzyme)
const REGISTRY = joinpath(ROOT, "deps", "dogfood_repairs")
const COMMIT = r"^[0-9a-f]{40}$"

@testset "registry files" begin
    files = filter(endswith(".toml"), readdir(REGISTRY))
    @test !isempty(files)
    @test "heap_slot_object_id.toml" in files
    for file in files
        name = chopsuffix(file, ".toml")
        entry = TOML.parsefile(joinpath(REGISTRY, file))
        @test occursin(r"^[a-z][a-z0-9_]*$", name)
        @test issubset(keys(entry), ("summary", "libenzyme_build", "tests"))
        summary = get(entry, "summary", nothing)
        @test summary isa String && !isempty(summary) && !occursin('\n', summary)
        build = get(entry, "libenzyme_build", nothing)
        @test build === nothing || (build isa String && occursin(COMMIT, build))
        tests = get(entry, "tests", String[])
        @test tests isa Vector
        for test in tests
            @test test isa String && isfile(joinpath(ROOT, test))
        end
    end
    declared = TOML.parsefile(joinpath(ROOT, "deps", "dogfood_libenzyme.toml"))["commit"]
    @test occursin(COMMIT, declared)
end

@testset "registry logic" begin
    dir = mktempdir()
    native = repeat("a", 40)
    write(joinpath(dir, "julia_only.toml"), "summary = \"a repair in Julia code\"\n")
    write(joinpath(dir, "native.toml"), "summary = \"a repair in libEnzyme\"\nlibenzyme_build = \"$native\"\n")
    write(joinpath(dir, "README.md"), "not a repair\n")
    repairs = Enzyme.read_dogfood_repairs(dir)
    @test repairs == Dict{Symbol, Union{Nothing, String}}(:julia_only => nothing, :native => native)
    @test isempty(Enzyme.dogfood_absent(repairs, true, (:julia_only, "native")))
    @test Enzyme.dogfood_absent(repairs, false, (:julia_only, :native, :gone)) ==
        [(:native, :library), (:gone, :not_in_tree)]
    @test isempty(Enzyme.read_dogfood_repairs(joinpath(dir, "absent")))

    commit = repeat("b", 40)
    @test Enzyme.libenzyme_path_is_build("/srv/builds/$commit/llvm15/libEnzyme-15.so", commit, 15)
    @test !Enzyme.libenzyme_path_is_build("/srv/builds/$commit/llvm18/libEnzyme-18.so", commit, 15)
    @test !Enzyme.libenzyme_path_is_build("/srv/builds/$(repeat("c", 40))/llvm15/libEnzyme-15.so", commit, 15)
    @test !Enzyme.libenzyme_path_is_build("/depot/artifacts/0123/lib/libEnzyme-15.so", commit, 15)
    @test !Enzyme.libenzyme_path_is_build("libEnzyme-15.so", commit, 15)
    # A preference may name the build through a link.
    build_dir = mkpath(joinpath(dir, "builds", commit, "llvm15"))
    library = joinpath(build_dir, "libEnzyme-15.so")
    write(library, "")
    link = joinpath(dir, "libEnzyme-link.so")
    symlink(library, link)
    @test Enzyme.libenzyme_path_is_build(link, commit, 15)
end

@testset "this process" begin
    lib = Enzyme.dogfood_libenzyme()
    println("DOGFOOD_LIBENZYME declared=", lib.declared, " loaded=", lib.loaded, " matches=", lib.matches)
    @test lib.declared == TOML.parsefile(joinpath(ROOT, "deps", "dogfood_libenzyme.toml"))["commit"]
    @test lib.loaded == Enzyme.API.Enzyme_jll.libEnzyme_path
    expected = get(ENV, "ENZYME_EXPECT_DOGFOOD_LIBENZYME", "")
    if !isempty(expected)
        @test lib.matches == (expected == "1")
    end

    registry = Enzyme.read_dogfood_repairs(REGISTRY)
    present = Enzyme.dogfood_repairs()
    @test issorted(present)
    for (name, build) in registry
        carried = build === nothing || lib.matches
        @test (name in present) == carried
        @test Enzyme.has_dogfood_repair(name) == carried
        @test Enzyme.has_dogfood_repair(String(name)) == carried
    end
    @test length(present) == count(build -> build === nothing || lib.matches, values(registry))
    @test !Enzyme.has_dogfood_repair(:no_such_repair)

    @test Enzyme.require_dogfood_repairs() === nothing
    @test Enzyme.require_dogfood_repairs(:concurrent_gc_safepoints; requester = "a test") === nothing
    if lib.matches
        @test Enzyme.require_dogfood_repairs(:heap_slot_object_id, "select_strong_zero") === nothing
    else
        @test_throws Enzyme.DogfoodRepairMissing Enzyme.require_dogfood_repairs(:heap_slot_object_id)
    end

    err = try
        Enzyme.require_dogfood_repairs(:no_such_repair, :heap_slot_object_id; requester = "a test")
        nothing
    catch e
        e
    end
    @test err isa Enzyme.DogfoodRepairMissing
    @test err.absent == (
        lib.matches ? [(:no_such_repair, :not_in_tree)] :
            [(:no_such_repair, :not_in_tree), (:heap_slot_object_id, :library)]
    )
    message = sprint(showerror, err)
    @test startswith(message, "DogfoodRepairMissing: this process lacks Enzyme dogfood repairs that a test requires:")
    @test occursin("no_such_repair: this Enzyme.jl source does not carry it", message)
    @test occursin("loaded " * lib.loaded, message)
    @test occursin(ROOT, message)
    @test occursin("Move Enzyme to a newer commit of the dogfood line.", message)
    @test occursin("not the build this source declares", message) == !lib.matches
    @test occursin("Load the libEnzyme build", message) == !lib.matches

    # Advice follows the reasons: a library mismatch alone does not ask for a newer commit.
    library_only = sprint(
        showerror, Enzyme.DogfoodRepairMissing("a test", [(:native, :library)], ROOT, lib.declared, "/stock/libEnzyme-15.so")
    )
    @test occursin("Load the libEnzyme build", library_only)
    @test !occursin("newer commit", library_only)
end
