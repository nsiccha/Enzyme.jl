# Which reviewed repairs of the dogfood line the running process carries.
#
# Every repair this tree carries has a file `deps/dogfood_repairs/<name>.toml`
# (fields in `deps/dogfood_repairs/README.md`). A repair whose file names a
# `libenzyme_build` lives in the native library, so it counts only when the
# library that loaded is the build `deps/dogfood_libenzyme.toml` declares. The
# registry is read from this package's own tree when first asked, not when the
# package image is written, so a checkout that moved answers for what it holds.

import TOML

"""
    Enzyme.DogfoodRepairMissing <: Exception

Thrown by [`Enzyme.require_dogfood_repairs`](@ref) when the running process
lacks a requested dogfood repair. `absent` holds one `(name, reason)` pair per
missing repair: `reason` is `:not_in_tree` when this Enzyme.jl source carries
no such repair, and `:library` when the repair is in the native library and the
libEnzyme that loaded is not the build this source declares.
"""
struct DogfoodRepairMissing <: Exception
    requester::String
    absent::Vector{Tuple{Symbol, Symbol}}
    source::String
    declared::Union{Nothing, String}
    loaded::String
end

function Base.showerror(io::IO, err::DogfoodRepairMissing)
    print(io, "DogfoodRepairMissing: ", err.requester, " needs Enzyme dogfood repairs this process lacks:")
    for (name, reason) in err.absent
        print(io, "\n  - ", name, ": ")
        if reason === :library
            print(io, "it is in the native library, and the libEnzyme that loaded is not the build this source declares")
        else
            print(io, "this Enzyme.jl source does not carry it (the source predates the repair, or the name is misspelled)")
        end
    end
    print(io, "\nEnzyme.jl source: ", err.source)
    print(io, "\nlibEnzyme: declared build ", something(err.declared, "none"), ", loaded ", err.loaded)
    print(
        io, "\nMove Enzyme to a newer commit of the dogfood line, and load the libEnzyme build its ",
        "deps/dogfood_libenzyme.toml declares (Enzyme_jll's libEnzyme_path preference). ",
        "Enzyme.dogfood_repairs() lists the repairs this process carries."
    )
    return
end

const DOGFOOD_REPAIRS_LOCK = ReentrantLock()
const DOGFOOD_REPAIRS_CACHE = Ref{Union{Nothing, Dict{Symbol, Union{Nothing, String}}}}(nothing)

dogfood_root() = something(pkgdir(@__MODULE__), dirname(@__DIR__))

# name => the first libEnzyme build carrying the repair, or `nothing` for a
# repair in Julia code alone.
function read_dogfood_repairs(dir::AbstractString)
    repairs = Dict{Symbol, Union{Nothing, String}}()
    isdir(dir) || return repairs
    for file in readdir(dir)
        endswith(file, ".toml") || continue
        entry = TOML.parsefile(joinpath(dir, file))
        build = get(entry, "libenzyme_build", nothing)
        repairs[Symbol(chopsuffix(file, ".toml"))] = build === nothing ? nothing : String(build)
    end
    return repairs
end

function dogfood_registry()
    lock(DOGFOOD_REPAIRS_LOCK)
    try
        cached = DOGFOOD_REPAIRS_CACHE[]
        cached === nothing || return cached
        repairs = read_dogfood_repairs(joinpath(dogfood_root(), "deps", "dogfood_repairs"))
        DOGFOOD_REPAIRS_CACHE[] = repairs
        return repairs
    finally
        unlock(DOGFOOD_REPAIRS_LOCK)
    end
end

function declared_libenzyme_build(root::AbstractString)
    path = joinpath(root, "deps", "dogfood_libenzyme.toml")
    isfile(path) || return nothing
    commit = get(TOML.parsefile(path), "commit", nothing)
    return commit === nothing ? nothing : String(commit)
end

# A dogfood build is installed as `<root>/<commit>/llvm<M>/libEnzyme-<M>.<ext>`
# on every host that holds one; the stock library lives in Enzyme_jll's artifact.
function libenzyme_path_is_build(path::AbstractString, commit::AbstractString, llvm_major::Integer)
    candidates = ispath(path) ? (String(path), realpath(path)) : (String(path),)
    for candidate in candidates
        parts = splitpath(candidate)
        length(parts) >= 3 || continue
        parts[end - 2] == commit && parts[end - 1] == "llvm$(llvm_major)" && return true
    end
    return false
end

"""
    Enzyme.dogfood_libenzyme() -> (; declared, loaded, matches)

The libEnzyme build this Enzyme.jl source declares in
`deps/dogfood_libenzyme.toml` (`declared`, a commit, or `nothing`), the library
this process loaded (`loaded`, `Enzyme_jll.libEnzyme_path`), and whether the
loaded library is that build (`matches`). With the stock library, or a
preference that still names an older build, `matches` is `false`.
"""
function dogfood_libenzyme()
    declared = declared_libenzyme_build(dogfood_root())
    loaded = String(API.Enzyme_jll.libEnzyme_path)
    matches = declared !== nothing && libenzyme_path_is_build(loaded, declared, Base.libllvm_version.major)
    return (; declared, loaded, matches)
end

function dogfood_absent(repairs::AbstractDict, libenzyme_matches::Bool, names)
    absent = Tuple{Symbol, Symbol}[]
    for name in names
        key = Symbol(name)
        if !haskey(repairs, key)
            push!(absent, (key, :not_in_tree))
        elseif repairs[key] !== nothing && !libenzyme_matches
            push!(absent, (key, :library))
        end
    end
    return absent
end

"""
    Enzyme.dogfood_repairs() -> Vector{Symbol}

The names of the dogfood repairs this process carries, sorted: every repair in
this Enzyme.jl source's `deps/dogfood_repairs`, less those in the native
library when the loaded libEnzyme is not the declared build
([`Enzyme.dogfood_libenzyme`](@ref)). Registered Enzyme and dogfood commits
before this function lack it; check `isdefined(Enzyme, :dogfood_repairs)`.
"""
function dogfood_repairs()
    matches = dogfood_libenzyme().matches
    names = Symbol[]
    for (name, build) in dogfood_registry()
        (build === nothing || matches) && push!(names, name)
    end
    return sort!(names)
end

"""
    Enzyme.has_dogfood_repair(name) -> Bool

Whether this process carries the dogfood repair `name` (a `Symbol` or string),
as [`Enzyme.dogfood_repairs`](@ref) lists it.
"""
has_dogfood_repair(name::Union{Symbol, AbstractString}) =
    isempty(dogfood_absent(dogfood_registry(), dogfood_libenzyme().matches, (name,)))

"""
    Enzyme.require_dogfood_repairs(names...; requester = "the caller")

Return `nothing` when this process carries every named dogfood repair, and
throw [`Enzyme.DogfoodRepairMissing`](@ref) naming each missing one otherwise.
Call it before compiling a derivative that needs those repairs, so a too-old
dogfood commit, or a library that is not the declared build, is refused with
its cause instead of failing inside the derivative. `requester` names the
feature that needs them in the message.

Registered Enzyme and dogfood commits before this function lack it, so check
`isdefined(Enzyme, :require_dogfood_repairs)` first and refuse when it is
`false`.
"""
function require_dogfood_repairs(names::Union{Symbol, AbstractString}...; requester::AbstractString = "the caller")
    lib = dogfood_libenzyme()
    absent = dogfood_absent(dogfood_registry(), lib.matches, names)
    isempty(absent) && return nothing
    throw(DogfoodRepairMissing(String(requester), absent, dogfood_root(), lib.declared, lib.loaded))
end
