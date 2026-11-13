# Derivatives kept in package images.
#
# A package image holds Julia's native code for what a package compiled while it precompiled,
# but a derivative is not Julia's code: Enzyme compiles it itself and hands it to a JIT, and
# the code holds the addresses of the Julia objects and C functions it refers to. Those mean
# nothing to another process, which is why a derivative compiled while an image is written
# used to be dropped and compiled again at its first call in every later session.
#
# While an image is written, the generator of `thunk` compiles the derivative with every
# reference by name instead (`retained_derivative`) and emits an object file, which goes into
# the image inside the `LazyThunk` that the generated code holds. The first call in a later
# session links that object into the JIT and resolves the names against that session
# (`materialize_retained`), which compiles nothing. The code holds no data: the arguments,
# their values and their sizes are inputs as they always are.
#
# Whether the object may still be used is decided by Julia, and not by anything here. The
# generated code carries the derivative's edges, Julia checks those when it loads the image,
# and an invalidated `CodeInstance` is not used: the code is inferred again, its generator
# makes a cell with nothing retained, and the derivative compiles at its first call. What
# Julia cannot know about, the process the object was compiled for, is in `conditions`. The
# generated code holds an edge on nothing else: it reaches the compiler by a dynamic call (see
# `in_calling_world`), so that invalidating the compiler does not cost the derivative.

"""
    RetainDerivatives

Whether a derivative compiled while a package image is written is kept in the image. Set it
in the package before the workload to keep them out.
"""
const RetainDerivatives = Ref(true)

"""
    retains_derivatives(job)

Whether `job` should be compiled for keeping in the package image being written. Only the
FFI ABI on the host holds a derivative by an address the JIT hands out.
"""
function retains_derivatives(@nospecialize(job::CompilerJob))::Bool
    return retention_supported() && RetainDerivatives[] && generating_output() &&
        Base.JLOptions().incremental != 0 &&
        job.config.params.ABI <: FFIABI &&
        job.config.target isa EnzymeTarget
end

"""
    retention_supported()

Whether this platform keeps derivatives in package images at all: x86-64 Linux with GPUCompiler
2 on Julia 1.10, 1.12 or 1.13, which is where linking an object has been measured.
"""
retention_supported()::Bool = HAS_GPUCOMPILER_2 && Sys.islinux() && Sys.ARCH === :x86_64 && VERSION.minor in (10, 12, 13)

"""
    ForeignAddress(symbol, library)

A C symbol of a shared library, the address of which a retained derivative refers to.
"""
struct ForeignAddress
    symbol::String
    library::String
end

@static if HAS_GPUCOMPILER_2
    const RetainedRelocation = GPUCompiler.Relocation
else
    const RetainedRelocation = Any
end

"""
    RetainedDerivative

The machine code of a derivative compiled while a package image was written, and what it
takes to use it in another process.

- `conditions`: the facts about the process that the code depends on (see
  [`retention_conditions`](@ref)).
- `object`: the object file.
- `adjoint`, `primal`: the names of its entry points.
- `TapeType`, `edges`: as in a [`CompileResult`](@ref).
- `relocations`: the words of its data to write (a slot holding a Julia value).
- `values`: symbols that stand for the address of a Julia value.
- `foreign`: symbols that stand for the address of a C function.
"""
struct RetainedDerivative
    conditions::String
    object::Vector{UInt8}
    adjoint::String
    primal::Union{Nothing, String}
    TapeType::Type
    edges::Vector{Any}
    relocations::Vector{RetainedRelocation}
    values::Vector{Pair{String, Any}}
    foreign::Vector{Pair{String, ForeignAddress}}
end

"""
    RetainedDerivatives(serial, threaded)

What an image keeps of one derivative, or why it does not. A derivative accumulates into
shadow memory atomically when it runs with more than one thread (see `atomic_add_for`), and an
image is written with one thread, so a job that leaves that choice open has two: `serial` and
`threaded`. Either is a [`RetainedDerivative`](@ref) or the reason there is none. A job that
settled the choice with `set_atomic_add` has one, in `serial`, and `threaded` is `nothing`.
"""
struct RetainedDerivatives
    serial::Union{RetainedDerivative, String}
    threaded::Union{RetainedDerivative, String, Nothing}
end

# libEnzyme's options that change the code it generates, and that Enzyme's API can set.
const RETAINED_LIBENZYME_OPTIONS = (
    :EnzymeFastMath,
    :EnzymeStrictAliasing,
    :EnzymeInline,
    :EnzymeZeroCache,
    :EnzymeNonPower2Cache,
    :EnzymeJuliaAddrLoad,
)

"""
    retention_conditions()

The facts about this process that the machine code of a derivative depends on and that Julia
does not check when it loads a package image: the CPU it was compiled for, the offset of the
thread-local storage it reads the task through, the library that differentiated it, and the
options that library and Enzyme were set to. Julia itself checks the rest: an image is only
loaded by the Julia, the packages and the options it was written with.
"""
function retention_conditions()::String
    io = IOBuffer()
    print(io, cpu_name(), '|', cpu_features(), '|')
    print(io, unsafe_load(cglobal(:jl_tls_offset, Int)), '|')
    print(io, Enzyme_jll.libEnzyme_path, '|')
    print(io, RunAttributor[], '|', RunAttributorAfterAD[], '|', UnfoldDataPhiLoads[], '|')
    handle = Libdl.dlopen(API.libEnzyme)
    for option in RETAINED_LIBENZYME_OPTIONS
        print(io, Int(API.EnzymeGetCLBool(Libdl.dlsym(handle, option))), ',')
    end
    return String(take!(io))
end

# Addresses of the values `unsafe_to_ptr` handed out while an image is written, for finding
# them again in a module. It does not survive the process, and it is not kept at all outside
# of writing an image.
function captured_address_value(addr::UInt)
    lock(Enzyme.captured_constants_lock)
    try
        return get(Enzyme.captured_addresses, addr, nothing)
    finally
        unlock(Enzyme.captured_constants_lock)
    end
end

struct SymbolAddressInfo
    fname::Cstring
    fbase::Ptr{Cvoid}
    sname::Cstring
    saddr::Ptr{Cvoid}
end

"""
    foreign_address(addr)::Union{ForeignAddress, Nothing}

The symbol of a shared library that `addr` is exactly the address of, if any. A library may be
at another address in another process, but its symbols keep their names.
"""
function foreign_address(addr::UInt)::Union{ForeignAddress, Nothing}
    info = Ref{SymbolAddressInfo}()
    ptr = Ptr{Cvoid}(addr)
    found = ccall(:dladdr, Cint, (Ptr{Cvoid}, Ptr{SymbolAddressInfo}), ptr, info)
    found == 0 && return nothing
    info[].fname == C_NULL && return nothing
    library = unsafe_string(info[].fname)
    # An ELF indirect function can resolve to a hidden implementation (glibc's memchr,
    # for example), so dladdr knows its library but not its exported name. The FFI map
    # already records the addresses Enzyme resolves by name; use that same information
    # here, and verify the name in this library before keeping it for another process.
    symbol = if info[].saddr == ptr && info[].sname != C_NULL
        unsafe_string(info[].sname)
    else
        get(FFI.ptr_map, ptr, nothing)
    end
    symbol === nothing && return nothing
    resolve_symbol_name(symbol, library, ptr) === :match || return nothing
    return ForeignAddress(symbol, library)
end

# The constant expressions `inttoptr (iN <address> to T)` reachable from the constant `c`.
function collect_inttoptrs!(found::Dict{LLVM.API.LLVMValueRef, LLVM.Value}, seen::Set{LLVM.API.LLVMValueRef}, @nospecialize(c::LLVM.Value))
    if c isa LLVM.ConstantExpr
        c.ref in seen && return
        push!(seen, c.ref)
        if opcode(c) == LLVM.API.LLVMIntToPtr
            operands(c)[1] isa LLVM.ConstantInt && (found[c.ref] = c)
            return
        end
        for op in operands(c)
            collect_inttoptrs!(found, seen, op)
        end
    elseif c isa LLVM.ConstantStruct || c isa LLVM.ConstantArray || c isa LLVM.ConstantVector
        c.ref in seen && return
        push!(seen, c.ref)
        for op in operands(c)
            collect_inttoptrs!(found, seen, op)
        end
    end
    return
end

function module_inttoptrs(mod::LLVM.Module)::Vector{LLVM.Value}
    found = Dict{LLVM.API.LLVMValueRef, LLVM.Value}()
    seen = Set{LLVM.API.LLVMValueRef}()
    for gv in globals(mod)
        init = LLVM.initializer(gv)
        init === nothing || collect_inttoptrs!(found, seen, init)
    end
    for fn in functions(mod), bb in blocks(fn), inst in instructions(bb), op in operands(inst)
        op isa LLVM.Constant && collect_inttoptrs!(found, seen, op)
    end
    return collect(values(found))
end

# Below this no address is ever mapped; it is where Julia's small type tags and sentinels live.
const RETAINED_FIRST_ADDRESS = UInt(4096)

"""
    RetainedSymbols

The names a module being compiled for an image uses in place of addresses.
"""
mutable struct RetainedSymbols
    const prefix::String
    const values::Vector{Pair{String, Any}}
    const foreign::Vector{Pair{String, ForeignAddress}}
    # One global per object: LLVM folds a comparison of two different globals to false.
    const by_value::IdDict{Any, String}
    const by_address::Dict{UInt, String}
    count::Int
end
RetainedSymbols(prefix::String) = RetainedSymbols(prefix, Pair{String, Any}[], Pair{String, ForeignAddress}[], IdDict{Any, String}(), Dict{UInt, String}(), 0)

function new_address_global!(mod::LLVM.Module, symbols::RetainedSymbols)::String
    symbols.count += 1
    name = symbols.prefix * "addr" * string(symbols.count)
    gv = LLVM.GlobalVariable(mod, LLVM.StructType(LLVM.LLVMType[]), name)
    API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
    API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
    return name
end

"""
    symbolic_addresses!(mod, symbols)

Replace every literal address in `mod` with a global that the process linking the code gives
that address, and record what it is the address of in `symbols`: a Julia value `unsafe_to_ptr`
gave out, or a symbol of a shared library. Throw an error for any other address.
"""
function symbolic_addresses!(mod::LLVM.Module, symbols::RetainedSymbols)
    for expr in module_inttoptrs(mod)
        addr = convert(UInt, operands(expr)[1])
        addr < RETAINED_FIRST_ADDRESS && continue
        name = get(symbols.by_address, addr, nothing)
        if name === nothing
            val = captured_address_value(addr)
            if val !== nothing
                value = something(val)
                name = get(symbols.by_value, value, nothing)
                if name === nothing
                    name = new_address_global!(mod, symbols)
                    symbols.by_value[value] = name
                    push!(symbols.values, name => value)
                end
            else
                foreign = foreign_address(addr)
                foreign === nothing && error("the address $(repr(addr)) in `$(string(expr))` is neither a Julia value Enzyme inserted nor a symbol of a library")
                name = new_address_global!(mod, symbols)
                push!(symbols.foreign, name => foreign)
            end
            symbols.by_address[addr] = name
        end
        replace_uses!(expr, LLVM.const_pointercast(globals(mod)[name], value_type(expr)))
    end
    return nothing
end

"""
    symbolic_inserted_values!(mod, inserted, symbols)

Record what each global `ejl_<key>` that the compilation inserted into `mod` stands for (see
`insert_julia_value!`), where [`bake_inserted_values!`](@ref) would write in its address.
"""
function symbolic_inserted_values!(mod::LLVM.Module, inserted::Dict{String, Any}, symbols::RetainedSymbols)
    for (key, val) in inserted
        name = "ejl_" * key
        haskey(globals(mod), name) || continue
        value = unbind(val)
        symbols.by_value[value] = name
        push!(symbols.values, name => value)
    end
    return nothing
end

"""
    finish_retained_module!(mod, symbols, relocs, entries)

Make the optimized `mod` fit to emit as an object for any process: every name it defines or
that this process's JIT would resolve specially is unique to this object, and what the loader
writes into is a definition it can find (see `GPUCompiler.emit_patchable_relocations!`). Return
the records that are left, and the new names of `entries`.
"""
function finish_retained_module!(mod::LLVM.Module, symbols::RetainedSymbols, relocs, entries::Vector{String})
    prefix = symbols.prefix

    # What optimization left unused needs no word.
    for rec in copy(relocs.records)
        haskey(globals(mod), rec.name) || continue
        gv = globals(mod)[rec.name]
        if isempty(LLVM.uses(gv))
            LLVM.erase!(gv)
        end
    end
    GPUCompiler.prune_dead_relocations!(mod, relocs)

    for fn in functions(mod)
        name = LLVM.name(fn)
        if startswith(name, "ejlstr\$") || startswith(name, "ejlptr\$")
            _, fname, lib = split(name, "\$")
            library = startswith(name, "ejlstr\$") ? String(lib) : unsafe_string(reinterpret(Ptr{UInt8}, parse(Int, lib)))
            new = prefix * name
            LLVM.name!(fn, new)
            push!(symbols.foreign, new => ForeignAddress(String(fname), library))
        end
    end

    # The names of inserted values are the same in every object that refers to the object.
    renamed = Pair{String, Any}[]
    for (name, value) in symbols.values
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            if isempty(LLVM.uses(gv))
                LLVM.erase!(gv)
                continue
            end
            new = startswith(name, prefix) ? name : prefix * name
            LLVM.name!(gv, new)
            push!(renamed, new => value)
        end
    end
    empty!(symbols.values)
    append!(symbols.values, renamed)
    used = Pair{String, ForeignAddress}[]
    for entry in symbols.foreign
        name = first(entry)
        if haskey(globals(mod), name)
            gv = globals(mod)[name]
            if isempty(LLVM.uses(gv))
                LLVM.erase!(gv)
                continue
            end
        elseif !haskey(functions(mod), name)
            continue
        end
        push!(used, entry)
    end
    empty!(symbols.foreign)
    append!(symbols.foreign, used)

    # Everything else this object defines stays inside it.
    record_names = Set{String}(rec.name for rec in relocs.records)
    for gv in globals(mod)
        LLVM.isdeclaration(gv) && continue
        LLVM.name(gv) in record_names && continue
        startswith(LLVM.name(gv), "llvm.") && continue
        LLVM.linkage(gv) in (LLVM.API.LLVMPrivateLinkage, LLVM.API.LLVMInternalLinkage) && continue
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    end
    new_entries = String[]
    for fn in functions(mod)
        LLVM.isdeclaration(fn) && continue
        LLVM.name(fn) in entries && continue
        LLVM.linkage(fn) in (LLVM.API.LLVMPrivateLinkage, LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMLinkerPrivateLinkage) && continue
        LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    end
    for entry in entries
        new = prefix * entry
        LLVM.name!(functions(mod)[entry], new)
        push!(new_entries, new)
    end

    GPUCompiler.emit_patchable_relocations!(mod, relocs)
    return copy(relocs.records), new_entries
end

# The regions this process has mapped, above the addresses that any integer might be.
function mapped_regions()::Vector{Tuple{UInt, UInt}}
    regions = Tuple{UInt, UInt}[]
    for line in eachline("/proc/self/maps")
        range = first(split(line, ' '))
        from, to = split(range, '-')
        start = parse(UInt, from; base = 16)
        stop = parse(UInt, to; base = 16)
        stop > (UInt(1) << 32) && push!(regions, (max(start, UInt(1) << 32), stop))
    end
    sort!(regions)
    return regions
end

"""
    address_in_object(object)::Union{UInt, Nothing}

A word in the object's loaded code or data that matches an address mapped in this process,
if there is one. This is a conservative check after replacing literal pointers in the IR.
An x86-64 instruction holds a 64-bit address as eight consecutive bytes wherever it is, so
every offset within each file-backed allocated section is checked. ELF metadata (including
relocation records) is not loaded data: adjacent fields can look like an address by chance.
"""
function address_in_object(object::Vector{UInt8})::Union{UInt, Nothing}
    regions = mapped_regions()
    isempty(regions) && return nothing
    low = regions[1][1]
    high = regions[end][2]
    elf = only(ObjectFile.readmeta(IOBuffer(object), ObjectFile.ELF.ELFHandle))
    for section in ObjectFile.Sections(elf)
        header = ObjectFile.deref(section)
        header.sh_flags & ObjectFile.ELF.SHF_ALLOC == 0 && continue
        header.sh_type == ObjectFile.ELF.SHT_NOBITS && continue
        start = Int(ObjectFile.section_offset(section)) + 1
        stop = start + Int(ObjectFile.section_size(section)) - 1
        @inbounds for i in start:(stop - 7)
            # User addresses leave the two top bytes zero; nearly all other bytes do not.
            (object[i + 7] == 0x00 && object[i + 6] == 0x00) || continue
            word = UInt(object[i]) | UInt(object[i + 1]) << 8 | UInt(object[i + 2]) << 16 | UInt(object[i + 3]) << 24 |
                UInt(object[i + 4]) << 32 | UInt(object[i + 5]) << 40
            (low <= word < high) || continue
            k = searchsortedlast(regions, (word, typemax(UInt)))
            k >= 1 && regions[k][1] <= word < regions[k][2] && return word
        end
    end
    return nothing
end

const RETAINED_ID = Threads.Atomic{UInt}(0)

function retained_prefix(@nospecialize(job::CompilerJob))::String
    id = Threads.atomic_add!(RETAINED_ID, UInt(1))
    return "enzret" * string(hash((hash(job), id, time_ns(), getpid())); base = 16) * "\$"
end

"""
    emit_retained_object(mod)::Vector{UInt8}

The object file of `mod`. Its code goes through the JIT's loader without the JIT's compile
layer, so it takes the code model the JIT itself compiles with on 64 bits: the loader places
code and data apart, further than a 32-bit offset reaches.
"""
function emit_retained_object(mod::LLVM.Module)::Vector{UInt8}
    opt_level = Base.JLOptions().opt_level
    optlevel = opt_level < 2 ? LLVM.API.LLVMCodeGenLevelNone :
        opt_level == 2 ? LLVM.API.LLVMCodeGenLevelDefault : LLVM.API.LLVMCodeGenLevelAggressive
    triple = LLVM.triple(JIT.get_jit())
    tm = LLVM.TargetMachine(
        LLVM.Target(; triple), triple, cpu_name(), cpu_features();
        optlevel, reloc = LLVM.API.LLVMRelocStatic, code = LLVM.API.LLVMCodeModelLarge,
    )
    try
        return LLVM.emit(tm, mod, LLVM.API.LLVMObjectFile)
    finally
        LLVM.dispose(tm)
    end
end

"""
    keep_native_edges!(edges, primal, world)

Infer the primal and the functions in `edges` with Julia's native interpreter.

A change to a function in the middle of a call chain reaches the code at the top through the
edges Julia recorded while inferring it. Julia writes into an image, checks when it loads it,
and restores afterwards the edges of code in its own cache. Enzyme's interpreter keeps its
results apart on Julia 1.10, where a derivative that is the only thing to have inferred a
function, as a custom rule's helper is, would leave nothing to invalidate it by. The results
of native inference are not used here: they are there for their edges.
"""
function keep_native_edges!(edges::Vector{Any}, primal::Core.MethodInstance, world::UInt)
    interp = Core.Compiler.NativeInterpreter(world)
    infer_natively(interp, primal)
    for e in edges
        e isa Core.MethodInstance && e.def isa Method && infer_natively(interp, e)
    end
    return nothing
end

@static if VERSION >= v"1.12-"
    infer_natively(interp, mi::Core.MethodInstance) =
        Core.Compiler.typeinf_ext_toplevel(interp, mi, Core.Compiler.SOURCE_MODE_NOT_REQUIRED)
else
    infer_natively(interp, mi::Core.MethodInstance) = Core.Compiler.typeinf_ext_toplevel(interp, mi)
end

"""
    compile_retained(job)::RetainedDerivative

Compile `job` as `_thunk` does, but with every reference by name, down to an object file.
"""
function compile_retained(@nospecialize(job::CompilerJob))::RetainedDerivative
    config = CompilerConfig(job.config; optimize = false)
    job = CompilerJob(job.source, config, job.world)
    mod, meta = compile(:llvm, job)
    adjoint_name = name(meta.adjointf)
    primal_name = meta.augmented_primalf === nothing ? nothing : name(meta.augmented_primalf)

    LLVM.@dispose pm = LLVM.ModulePassManager() begin
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        LLVM.run!(pm, mod)
    end
    fixup_callconv!(mod, JIT.get_tm())

    symbols = RetainedSymbols(retained_prefix(job))
    symbolic_inserted_values!(mod, meta.value_table.inserted, symbols)
    symbolic_addresses!(mod, symbols)

    post_optimize!(mod, JIT.get_tm(); callconv = false)

    # Optimization may fold what it could not before. Nothing but names may be left.
    entries = primal_name === nothing ? [adjoint_name] : [adjoint_name, primal_name]
    records, new_entries = finish_retained_module!(mod, symbols, meta.relocations, entries)
    for expr in module_inttoptrs(mod)
        addr = convert(UInt, operands(expr)[1])
        addr < RETAINED_FIRST_ADDRESS || error("optimizing left the address $(repr(addr)) in `$(string(expr))`")
    end

    LLVM.verify(mod)
    object = emit_retained_object(mod)
    address = address_in_object(object)
    address === nothing || error("the object holds the address $(repr(address)), which is mapped in this process")

    # A generated `thunk` holds an edge on the primal only, and leaves the callees to their
    # callers' edges. This code holds the callees too, so it depends on each directly.
    edges = Any[e for e in meta.edges]
    for mi in keys(meta.compiled)
        push!(edges, mi)
    end
    keep_native_edges!(edges, job.source, job.world)

    return RetainedDerivative(
        retention_conditions(),
        object,
        new_entries[1],
        primal_name === nothing ? nothing : new_entries[2],
        meta.TapeType,
        edges,
        records,
        symbols.values,
        symbols.foreign,
    )
end

# What this session did with retained derivatives (see `retained_derivative_stats`).
mutable struct RetainedStatistics
    # Derivatives taken from an image, and the time it took to link them.
    materialized::Int
    seconds::Float64
    # The time it took to start tracking what they depend on.
    validity_seconds::Float64
    # Why a derivative an image holds was not used in this process.
    rejected::Dict{String, Int}
    # Why an image holds no derivative for a cell that was called.
    absent::Dict{String, Int}
end
const retained_statistics = RetainedStatistics(0, 0.0, 0.0, Dict{String, Int}(), Dict{String, Int}())

function reset_retained_statistics!()
    retained_statistics.materialized = 0
    retained_statistics.seconds = 0.0
    retained_statistics.validity_seconds = 0.0
    empty!(retained_statistics.rejected)
    empty!(retained_statistics.absent)
    return nothing
end

# Derivatives compiled in this process, retained or not.
const compiled_derivatives = Threads.Atomic{Int}(0)

"""
    retained_derivative_stats()

What this session did with derivatives kept in package images:

- `compiled`: derivatives this process compiled;
- `materialized`, `seconds`: derivatives taken from an image, and the time it took to link them;
- `validity_seconds`: the time it took to start tracking what those depend on;
- `rejected`: why a derivative an image holds could not be used here, with counts;
- `absent`: why an image holds none for a call that wanted one, with counts. That includes
  [`NOT_FROM_AN_IMAGE`](@ref), for the code of a call that was not loaded from an image.
"""
function retained_derivative_stats()
    lock(cache_publish_lock)
    try
        stats = retained_statistics
        return (;
            compiled = compiled_derivatives[],
            materialized = stats.materialized,
            seconds = stats.seconds,
            validity_seconds = stats.validity_seconds,
            rejected = copy(stats.rejected),
            absent = copy(stats.absent),
        )
    finally
        unlock(cache_publish_lock)
    end
end

function count_retained!(field::Symbol, reason::String)
    lock(cache_publish_lock)
    try
        counts = getfield(retained_statistics, field)
        counts[reason] = get(counts, reason, 0) + 1
    finally
        unlock(cache_publish_lock)
    end
    return nothing
end

"""
    NOT_FROM_AN_IMAGE

The reason in `retained_derivative_stats().absent` for a derivative whose call was generated in
this session. No image holds that derivative, or Julia invalidated the code in the image that
did: Julia then generates the code of the call again, and a session keeps nothing.
"""
const NOT_FROM_AN_IMAGE = "the code of this call was generated in this session: no image holds it, or a method it depends on changed"

# What a cell holds for its first call, and the reason if that is nothing.
function retained_of_cell(retained::Union{Nothing, RetainedDerivatives})::Union{Nothing, RetainedDerivatives}
    if retained === nothing && retention_supported() && !generating_output()
        count_retained!(:absent, NOT_FROM_AN_IMAGE)
    end
    return retained
end

# What this process linked, by the name of its entry point: a cell compiled again and a second
# variant of a job must not define the same names twice.
const materialized_derivatives = Dict{String, CompileResult}()

"""
    materialize_retained(retained)::CompileResult

Link the object of `retained` into this process and return it as `_link` would.
"""
function materialize_retained(retained::RetainedDerivative)::CompileResult
    lock(cache_publish_lock)
    done = try
        get(materialized_derivatives, retained.adjoint, nothing)
    finally
        unlock(cache_publish_lock)
    end
    done === nothing || return done

    started = time()
    lljit = JIT.get_jit()
    jd = JIT.jit[].jd
    pairs = JIT.symbol_pair_type()[]
    for (symbol, value) in retained.values
        push!(pairs, JIT.absolute_symbol_pair(LLVM.mangle(lljit, symbol), unsafe_to_ptr(value)))
    end
    for (symbol, foreign) in retained.foreign
        ptr = JIT.fix_ptr_lookup("ejlstr\$" * foreign.symbol * "\$" * foreign.library)
        push!(pairs, JIT.absolute_symbol_pair(LLVM.mangle(lljit, symbol), ptr))
    end
    isempty(pairs) || LLVM.define(jd, LLVM.absolute_symbols(pairs))

    LLVM.add!(lljit, jd, LLVM.MemoryBuffer(retained.object, retained.adjoint, true))
    adjoint = pointer(JIT.lookup(jd, retained.adjoint))
    primal = retained.primal === nothing ? C_NULL : pointer(JIT.lookup(jd, retained.primal))
    for rec in retained.relocations
        word = GPUCompiler.resolve_relocation_target(rec.target)
        site = pointer(JIT.lookup(jd, rec.name))
        unsafe_store!(Ptr{UInt}(site + rec.offset), word)
    end
    result = CompileResult(adjoint, primal, retained.TapeType, retained.edges)

    lock(cache_publish_lock)
    try
        materialized_derivatives[retained.adjoint] = result
        retained_statistics.materialized += 1
        retained_statistics.seconds += time() - started
    finally
        unlock(cache_publish_lock)
    end
    return result
end

# `track_validity` for a derivative taken from an image.
function track_retained_validity(edges::Vector{Any})
    started = time()
    validity = track_validity(edges)
    elapsed = time() - started
    lock(cache_publish_lock)
    try
        retained_statistics.validity_seconds += elapsed
    finally
        unlock(cache_publish_lock)
    end
    return validity
end

"""
    retained_result(retained, job)::Union{CompileResult, Nothing}

The derivative an image kept for `job`, linked into this process, or `nothing` when there is
none that fits this process.
"""
function retained_result(retained::RetainedDerivatives, @nospecialize(job::CompilerJob))::Union{CompileResult, Nothing}
    threaded = job.config.params.atomicAdd === nothing && Threads.nthreads() > 1
    variant = threaded ? retained.threaded : retained.serial
    if variant === nothing
        count_retained!(:absent, "compiled without a threaded variant")
        return nothing
    elseif variant isa String
        count_retained!(:absent, variant)
        return nothing
    end
    if variant.conditions != retention_conditions()
        count_retained!(:rejected, "compiled for other conditions: " * variant.conditions)
        return nothing
    end
    try
        return materialize_retained(variant)
    catch err
        count_retained!(:rejected, sprint(showerror, err))
        return nothing
    end
end

# `job`, with the choice `atomic_add_for` leaves to the number of threads made.
function with_atomic_add(@nospecialize(job::CompilerJob), atomic::Bool)
    p = job.config.params
    params = EnzymeCompilerParams(
        p.params, p.TT, p.mode, p.width, p.rt, p.run_enzyme, p.abiwrap, p.modifiedBetween,
        p.returnPrimal, p.shadowInit, p.expectedTapeType, p.ABI, p.err_if_func_written,
        p.runtimeActivity, p.strongZero, atomic,
    )
    return CompilerJob(job.source, CompilerConfig(job.config; params), job.world)
end

function retained_variant(@nospecialize(job::CompilerJob))::Union{RetainedDerivative, String}
    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    return try
        retained = compile_retained(job)
        # A derivative that does not link here will not link anywhere else.
        materialize_retained(retained)
        retained
    catch err
        sprint(showerror, err)
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end
end

"""
    retained_derivatives!(job, edges)::Union{RetainedDerivatives, Nothing}

Compile what a package image being written should keep of the derivative of `job`, if any,
and add what it depends on to `edges`, those of the code the generator of `thunk` returns:
that code is what Julia checks when it loads the image, and what it invalidates afterwards.

A failure here is not an error. It is the reason the session that calls the derivative
compiles it, and fails there if it has to, as it would have anyway.
"""
function retained_derivatives!(@nospecialize(job::CompilerJob), edges::Vector{Any})::Union{RetainedDerivatives, Nothing}
    retains_derivatives(job) || return nothing
    serial = retained_variant(job)
    threaded = if job.config.params.atomicAdd === nothing
        retained_variant(with_atomic_add(job, true))
    else
        nothing
    end
    if threaded isa RetainedDerivative && serial isa RetainedDerivative && threaded.TapeType !== serial.TapeType
        threaded = "the threaded variant has another tape type"
    end
    for variant in (serial, threaded)
        variant isa RetainedDerivative || continue
        for e in variant.edges
            push!(edges, e)
        end
    end
    return RetainedDerivatives(serial, threaded)
end
