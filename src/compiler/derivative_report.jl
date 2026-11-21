# What the code of a compiled derivative calls at run time, and a way for a caller to ask.
#
# Enzyme compiles a derivative as its own LLVM module, optimizes it and hands it to its JIT.
# Nothing Julia offers describes that code: `@code_warntype` and `@allocated` see the primal
# function and Julia's heap. Right before the JIT takes a module, `scan_derivative` reads it
# once and lists every call that leaves it: calls Julia dispatches at run time, Enzyme's own
# run-time differentiation of a call it could not resolve, boxes, allocations on Julia's heap,
# native allocations (Enzyme's tape and caches), and other calls into the runtime or a library.
# The list stays with the compiled code (`derivative_records`), and with the machine code a
# package image keeps (`RetainedDerivative`), so that `derivative_report` can describe a
# derivative a piece of code runs, whenever it was compiled.
#
# Nothing here runs code of the caller inside the compiler: a compile may happen in a generator,
# which runs in the world its method was defined in.

"""
    DerivativeCallSite

A call of a compiled function of a derivative's module, as listed in the `call_sites` of a
[`DerivativeFinding`](@ref) or of another call site.

- `in_function`: the compiled function that holds the call.
- `sweep`: the sweep the call is in, as for a finding.
- `frames`: the source lines of the call, innermost first: its own, those it was inlined into,
  and those of the call of its function while that function has exactly one call site. Empty
  for a call without debug information.
- `in_loop`: whether the call is in a loop, or in a function called from one.
- `throw_only`: whether the call is only reached on a path that ends in an error.
- `count`: how many calls of the module this site stands for: calls in `in_function` with the
  same source lines.
- `call_sites`: where `frames` stop because the function holding them is called from several
  places, its call sites; empty when they reach the derivative's entry. For a call inside a
  recursion (a function that calls itself, directly or through others), the calls that enter
  the recursion from outside it. Call sites are listed once per function and shared, so the
  same vector can appear under several findings and sites.

[`call_chains`](@ref) lists the complete chains of source lines this structure stands for.
"""
struct DerivativeCallSite
    in_function::String
    sweep::Symbol
    frames::Vector{StackTraces.StackFrame}
    in_loop::Bool
    throw_only::Bool
    count::Int
    call_sites::Vector{DerivativeCallSite}
end

"""
    DerivativeFinding

One kind of call in the compiled code of a derivative, found by [`derivative_report`](@ref).

- `kind`: `:dynamic_dispatch` (Julia looks the method up when the call runs),
  `:enzyme_runtime_fallback` (Enzyme differentiates the callee when the call runs, and compiles
  a further derivative for it then), `:box`, `:gc_allocation` (Julia's heap),
  `:native_allocation` (`malloc` and friends: Enzyme's tape and caches, and temporaries of
  foreign calls), or `:runtime_call` (any other call that leaves the derivative's code: into
  the Julia runtime, a native library such as BLAS, or separately compiled Julia code; a call
  through a pointer computed at run time has the callee `(function pointer)`).
- `callee`: the function called, by name. An address the code calls directly is resolved to
  its symbol.
- `in_function`: the compiled function that holds the call.
- `sweep`: `:forward` or `:reverse` for the sweep the call is in, `:combined` in a function that
  holds both sweeps where the call cannot be proven to be in the reverse one, and `:primal` in
  code that is not differentiated.
- `frames`: the source lines the call comes from, from debug info, innermost first: its own,
  those it was inlined into, and those of the call of its function while that function has
  exactly one call site.
- `call_sites`: where `frames` stop because the function holding them is called from several
  places, its call sites (see [`DerivativeCallSite`](@ref)), each with its own frames and call
  sites; empty when `frames` reach the derivative's entry. [`call_chains`](@ref) lists one
  complete chain of source lines per call path.
- `in_loop`: whether the call is in a loop, or in a function called from one.
- `throw_only`: whether the call is only reached on a path that ends in an error,
  including when every call of its helper is on such a path. Code the derivative
  provably cannot reach has no findings.
- `bytes`: for an allocation, its size when that is a constant, and `nothing` when it depends
  on run-time values (or is not an argument of the call, as for an array).
- `count`: how many calls of the module this finding stands for.
"""
struct DerivativeFinding
    kind::Symbol
    callee::String
    in_function::String
    sweep::Symbol
    frames::Vector{StackTraces.StackFrame}
    in_loop::Bool
    throw_only::Bool
    bytes::Union{Nothing, Int}
    count::Int
    call_sites::Vector{DerivativeCallSite}
end

DerivativeFinding(kind, callee, in_function, sweep, frames, in_loop, throw_only, bytes, count) =
    DerivativeFinding(kind, callee, in_function, sweep, frames, in_loop, throw_only, bytes, count, DerivativeCallSite[])

# Call sites are shared between findings and sites, so a comparison visits each pair of
# vectors once: its cost stays linear in what they hold, however often they are shared.
function same_call_sites(a::Vector{DerivativeCallSite}, b::Vector{DerivativeCallSite}, seen::IdDict{Any, Base.IdSet{Any}})::Bool
    a === b && return true
    length(a) == length(b) || return false
    compared = get!(Base.IdSet{Any}, seen, a)
    b in compared && return true
    push!(compared, b)
    for (x, y) in zip(a, b)
        same_call_site(x, y, seen) || return false
    end
    return true
end

function same_call_site(a::DerivativeCallSite, b::DerivativeCallSite, seen::IdDict{Any, Base.IdSet{Any}})::Bool
    return a.in_function == b.in_function && a.sweep === b.sweep && a.frames == b.frames &&
        a.in_loop == b.in_loop && a.throw_only == b.throw_only && a.count == b.count &&
        same_call_sites(a.call_sites, b.call_sites, seen)
end

Base.:(==)(a::DerivativeCallSite, b::DerivativeCallSite) = same_call_site(a, b, IdDict{Any, Base.IdSet{Any}}())

# The hash leaves out the call sites above: equal sites hash equal, and hashing stays linear.
function Base.hash(a::DerivativeCallSite, h::UInt)
    return hash(a.count, hash(a.throw_only, hash(a.in_loop, hash(a.frames, hash(a.sweep, hash(a.in_function, hash(:DerivativeCallSite, h)))))))
end

function Base.:(==)(a::DerivativeFinding, b::DerivativeFinding)
    return a.kind === b.kind && a.callee == b.callee && a.in_function == b.in_function &&
        a.sweep === b.sweep && a.frames == b.frames && a.in_loop == b.in_loop &&
        a.throw_only == b.throw_only && a.bytes == b.bytes && a.count == b.count &&
        same_call_sites(a.call_sites, b.call_sites, IdDict{Any, Base.IdSet{Any}}())
end

function Base.hash(a::DerivativeFinding, h::UInt)
    return hash(a.count, hash(a.bytes, hash(a.throw_only, hash(a.in_loop, hash(a.frames, hash(a.sweep, hash(a.in_function, hash(a.callee, hash(a.kind, h)))))))))
end

"""
    call_chains(finding::DerivativeFinding)::Vector{Vector{StackTraces.StackFrame}}
    call_chains(site::DerivativeCallSite)

The complete chains of source lines that reach `finding`, innermost first, one per path
through its [`call_sites`](@ref DerivativeFinding) and theirs: each is the finding's `frames`
followed by the frames of one call site at each level. A finding whose frames reach the
derivative's entry has one chain, its `frames`.

The number of chains is the number of call paths, which multiplies across levels: a function
called from three places that calls the finding's function from two has six. A call site
without debug information adds no frames to its chains.
"""
call_chains(f::Union{DerivativeFinding, DerivativeCallSite}) =
    append_call_chains!(Vector{StackTraces.StackFrame}[], f.frames, f.call_sites)

function append_call_chains!(
        chains::Vector{Vector{StackTraces.StackFrame}},
        prefix::Vector{StackTraces.StackFrame},
        sites::Vector{DerivativeCallSite},
    )::Vector{Vector{StackTraces.StackFrame}}
    if isempty(sites)
        push!(chains, copy(prefix))
        return chains
    end
    for site in sites
        append_call_chains!(chains, vcat(prefix, site.frames), site.call_sites)
    end
    return chains
end

# Julia's entry points for a call whose method is looked up, or reached through the generic
# calling convention, when it runs. Names are without their `jl_` or `ijl_` prefix.
const DISPATCH_RUNTIME_FUNCTIONS = Set{String}(
    [
        "apply_generic", "invoke", "apply", "call", "call0", "call1", "call2", "call3",
        "f__apply_iterate", "f__apply_latest", "f__apply_pure", "f__call_latest",
        "f__call_in_world", "f__call_in_world_total", "f_invoke", "f_invokelatest",
        "f_invoke_in_world", "f_opaque_closure_call", "gf_invoke", "gf_invoke_by_method",
        "gf_invoke_lookup", "gf_invoke_lookup_worlds", "unknown_fptr",
    ]
)

# Julia's entry points that allocate an object on its heap.
const GC_ALLOCATION_RUNTIME_FUNCTIONS = Set{String}(
    [
        "gc_pool_alloc", "gc_pool_alloc_instrumented", "gc_small_alloc",
        "gc_small_alloc_instrumented", "gc_big_alloc", "gc_big_alloc_instrumented",
        "gc_alloc_typed", "gc_alloc", "gc_allocobj", "gc_managed_malloc",
        "alloc_array_1d", "alloc_array_2d", "alloc_array_3d", "alloc_array_nd", "new_array",
        "array_copy", "ptr_to_array", "ptr_to_array_1d", "reshape_array",
        "array_grow_end", "array_grow_beg", "array_grow_at", "array_sizehint",
        "alloc_genericmemory", "alloc_genericmemory_unchecked", "genericmemory_copy",
        "genericmemory_copy_slice", "string_to_genericmemory", "ptr_to_genericmemory",
        "alloc_string", "cstr_to_string", "pchar_to_string",
        "alloc_svec", "alloc_svec_uninit", "svec", "svec1", "svec2", "svec3",
        "new_struct", "new_structv", "new_structt", "new_struct_uninit", "new_bits",
    ]
)

const NATIVE_ALLOCATION_FUNCTIONS = Set{String}(
    ["malloc", "calloc", "realloc", "posix_memalign", "aligned_alloc", "memalign"]
)

# The 1-based position of the size among the arguments of an allocation function.
const ALLOCATION_SIZE_ARGUMENT = Dict{String, Int}(
    "malloc" => 1, "realloc" => 2, "posix_memalign" => 3, "aligned_alloc" => 2, "memalign" => 2,
    "gc_pool_alloc" => 3, "gc_pool_alloc_instrumented" => 3, "gc_small_alloc" => 3,
    "gc_small_alloc_instrumented" => 3, "gc_big_alloc" => 2, "gc_big_alloc_instrumented" => 2,
    "gc_alloc_typed" => 2, "gc_alloc" => 2, "gc_managed_malloc" => 1,
)

# `name` without the prefix Julia's runtime functions carry (`ijl_` inside libjulia-internal).
function runtime_function_name(name::String)::Union{Nothing, String}
    startswith(name, "ijl_") && return String(SubString(name, 5))
    startswith(name, "jl_") && return String(SubString(name, 4))
    return nothing
end

# The name to report for a callee: Julia's runtime functions by their public `jl_` name.
function reported_callee_name(name::String)::String
    startswith(name, "ijl_") && return String(SubString(name, 2))
    return name
end

# A reference LLVM hands out, as a key.
reference_key(ref::Ptr)::UInt = reinterpret(UInt, ref)

is_constant_pointer_cast(v::LLVM.Value) =
    v isa LLVM.ConstantExpr && v.opcode in (LLVM.Opcode.BitCast, LLVM.Opcode.AddrSpaceCast)

# `v` without the constant casts around it.
function strip_constant_casts(@nospecialize(v::LLVM.Value))::LLVM.Value
    while is_constant_pointer_cast(v)
        v = v.operands[1]
    end
    return v
end

# Can something call this function without a direct edge in our call graph?
function has_indirect_callers(v::LLVM.Value)::Bool
    for use in v.uses
        user = use.user
        if user isa LLVM.CallBase
            # The callee is the last operand. Passing the same function as an argument
            # of that very call is a separate use of its address.
            use.ref == LLVM.API.LLVMGetOperandUse(user, length(user.operands) - 1) && continue
        elseif is_constant_pointer_cast(user)
            has_indirect_callers(user) || continue
        end
        return true
    end
    return false
end

# The value of argument `i` of `call` when it is a constant integer.
function constant_argument(call::LLVM.Instruction, i::Int)::Union{Nothing, Int}
    ops = call.operands
    # The callee is the last operand.
    i < length(ops) || return nothing
    arg = ops[i]
    arg isa LLVM.ConstantInt || return nothing
    return convert(Int, arg)
end

# The symbol at `addr`, as `check_ir!` finds it for a call through a literal address.
function address_callee_name(addr::UInt)::String
    ptr = Ptr{Cvoid}(addr)
    known = get(FFI.ptr_map, ptr, nothing)
    known === nothing || return known
    frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint), ptr, 0)
    length(frames) >= 1 || return ""
    fn, file, _, _, fromC, _ = last(frames)
    name = string(fn)
    isempty(name) && return ""
    # A name for a C address is the nearest symbol, which may belong to another address.
    if fromC && resolve_symbol_name(name, string(file), ptr) === :mismatch
        return ""
    end
    return name
end

# What a call instruction calls: `:internal` (a function this module defines, with its index),
# `:named` (by name), `:indirect` (through a pointer computed at run time) or `:skip`.
function called_function(
        call::LLVM.Instruction,
        defined::Dict{String, Int},
        global_names::Union{Nothing, Dict{String, String}},
        address_names::Dict{UInt, String},
    )::Tuple{Symbol, String, Int}
    callee = call.called_operand
    callee isa LLVM.InlineAsm && return (:skip, "", 0)
    callee = strip_constant_casts(callee)
    if callee isa LLVM.Function
        name = callee.name
        idx = get(defined, name, 0)
        idx == 0 || return (:internal, name, idx)
        startswith(name, "llvm.") && return (:skip, "", 0)
        return (:named, foreign_function_name(name, global_names), 0)
    elseif callee isa LLVM.GlobalVariable
        return (:named, foreign_function_name(callee.name, global_names), 0)
    elseif callee isa LLVM.ConstantExpr && callee.opcode == LLVM.Opcode.IntToPtr
        addr = callee.operands[1]
        addr isa LLVM.ConstantInt || return (:indirect, "", 0)
        address = convert(UInt, addr)
        name = get(address_names, address, nothing)
        if name === nothing
            name = address_callee_name(address)
            address_names[address] = name
        end
        return (:named, isempty(name) ? "(unknown address)" : name, 0)
    elseif callee isa LLVM.LoadInst
        # A foreign function called through a slot the linker or Julia fills in.
        slot = strip_constant_casts(callee.operands[1])
        if slot isa LLVM.GlobalVariable
            return (:named, foreign_function_name(slot.name, global_names), 0)
        end
    end
    return (:indirect, "", 0)
end

# The function a named declaration or slot stands for.
function foreign_function_name(name::String, global_names::Union{Nothing, Dict{String, String}})::String
    if global_names !== nothing
        known = get(global_names, name, nothing)
        known === nothing || return known
    end
    # Enzyme's own names for a foreign function: `ejlstr$<function>$<library>`.
    if startswith(name, "ejlstr\$") || startswith(name, "ejlptr\$")
        parts = split(name, '\$')
        length(parts) >= 2 && return String(parts[2])
    end
    # A slot of Julia's procedure linkage table.
    m = match(r"^jlplt_(.*)_\d+_got$", name)
    m === nothing || return String(m.captures[1])
    return name
end

# The kind of a call to the function `name`, and the name to report.
function classify_callee(call::LLVM.Instruction, name::String)::Tuple{Symbol, String}
    runtime = runtime_function_name(name)
    if runtime !== nothing
        if runtime in DISPATCH_RUNTIME_FUNCTIONS
            # Enzyme's run-time rules are Julia functions the code dispatches to.
            ops = call.operands
            if length(ops) >= 2
                target = strip_constant_casts(ops[1])
                if target isa LLVM.GlobalVariable
                    tname = target.name
                    at = findfirst("ejl_enz_runtime_", tname)
                    if at !== nothing
                        return (:enzyme_runtime_fallback, String(SubString(tname, last(at) - 7)))
                    end
                end
            end
            return (:dynamic_dispatch, reported_callee_name(name))
        elseif startswith(runtime, "box_")
            return (:box, reported_callee_name(name))
        elseif runtime in GC_ALLOCATION_RUNTIME_FUNCTIONS
            return (:gc_allocation, reported_callee_name(name))
        elseif runtime in NATIVE_ALLOCATION_FUNCTIONS
            return (:native_allocation, reported_callee_name(name))
        end
        return (:runtime_call, reported_callee_name(name))
    end
    name in NATIVE_ALLOCATION_FUNCTIONS && return (:native_allocation, name)
    return (:runtime_call, name)
end

# The size an allocation asks for, when it is a constant.
function allocation_bytes(call::LLVM.Instruction, kind::Symbol, name::String)::Union{Nothing, Int}
    (kind === :gc_allocation || kind === :native_allocation) || return nothing
    key = something(runtime_function_name(name), name)
    if key == "calloc"
        n = constant_argument(call, 1)
        size = constant_argument(call, 2)
        (n === nothing || size === nothing) && return nothing
        return n * size
    end
    position = get(ALLOCATION_SIZE_ARGUMENT, key, 0)
    position == 0 && return nothing
    return constant_argument(call, position)
end

# The strongly connected components of the graph with successor lists `succ`: the number of
# each vertex's component.
function strongly_connected_components(succ::Vector{Vector{Int}})::Vector{Int}
    n = length(succ)
    index = zeros(Int, n)
    low = zeros(Int, n)
    onstack = falses(n)
    component = zeros(Int, n)
    stack = Int[]
    work = Tuple{Int, Int}[]
    counter = 0
    ncomponents = 0
    for root in 1:n
        index[root] == 0 || continue
        push!(work, (root, 1))
        while !isempty(work)
            v, position = work[end]
            if index[v] == 0
                counter += 1
                index[v] = counter
                low[v] = counter
                push!(stack, v)
                onstack[v] = true
            end
            descended = false
            successors = succ[v]
            while position <= length(successors)
                w = successors[position]
                position += 1
                if index[w] == 0
                    work[end] = (v, position)
                    push!(work, (w, 1))
                    descended = true
                    break
                elseif onstack[w]
                    low[v] = min(low[v], index[w])
                end
            end
            descended && continue
            pop!(work)
            if !isempty(work)
                u = work[end][1]
                low[u] = min(low[u], low[v])
            end
            if low[v] == index[v]
                ncomponents += 1
                while true
                    w = pop!(stack)
                    onstack[w] = false
                    component[w] = ncomponents
                    w == v && break
                end
            end
        end
    end
    return component
end

# Which vertices are on a cycle of the graph with successor lists `succ`.
function on_cycle(succ::Vector{Vector{Int}})::BitVector
    n = length(succ)
    component = strongly_connected_components(succ)
    sizes = zeros(Int, n)
    for v in 1:n
        sizes[component[v]] += 1
    end
    cyclic = falses(n)
    for v in 1:n
        cyclic[v] = sizes[component[v]] > 1 || v in succ[v]
    end
    return cyclic
end

# Which vertices reach one of `targets` in the graph with successor lists `succ`.
function reaches(succ::Vector{Vector{Int}}, targets::BitVector)::BitVector
    n = length(succ)
    pred = Vector{Int}[Int[] for _ in 1:n]
    for v in 1:n, w in succ[v]
        push!(pred[w], v)
    end
    return reachable_from(pred, targets)
end

# Which vertices can be reached from any of `roots`.
function reachable_from(succ::Vector{Vector{Int}}, roots::BitVector)::BitVector
    reached = copy(roots)
    work = findall(roots)
    while !isempty(work)
        w = pop!(work)
        for v in succ[w]
            if !reached[v]
                reached[v] = true
                push!(work, v)
            end
        end
    end
    return reached
end

# What `scan_derivative` knows about the blocks of one function.
struct ScannedFunction
    name::String
    fn::LLVM.Function
    blocks::Vector{LLVM.BasicBlock}
    # Whether the function's entry reaches the block.
    reachable::BitVector
    # Whether a block reaches a return: one that does not only leads to an error.
    returns::BitVector
    # Whether a block is on a cycle of the control flow graph.
    cyclic::BitVector
    # Whether a block is in the reverse sweep of a function that holds both sweeps.
    reverse::BitVector
    # The sweep of every block, or `:combined` for a function that holds both.
    sweep::Symbol
end

function scanned_function(fn::LLVM.Function, defined::Dict{String, Int})::ScannedFunction
    name = fn.name
    blks = collect(fn.blocks)
    n = length(blks)
    position = Dict{UInt, Int}()
    for (i, bb) in enumerate(blks)
        position[reference_key(bb.ref)] = i
    end
    succ = Vector{Int}[Int[] for _ in 1:n]
    returning = falses(n)
    inverted = falses(n)
    for (i, bb) in enumerate(blks)
        inverted[i] = startswith(bb.name, "invert")
        term = bb.terminator
        term === nothing && continue
        if term isa LLVM.RetInst
            returning[i] = true
        end
        for target in term.successors
            j = get(position, reference_key(target.ref), 0)
            j == 0 || push!(succ[i], j)
        end
    end
    sweep = if startswith(name, "augmented_") || startswith(name, "fwddiffe")
        :forward
    elseif startswith(name, "diffe")
        # A function for the reverse sweep alone has a forward sweep beside it. In a batch its
        # name carries the width (`diffe2julia_f` beside `augmented_julia_f`).
        haskey(defined, "augmented_" * String(lstrip(isdigit, SubString(name, 6)))) ? :reverse : :combined
    else
        :primal
    end
    reverse = falses(n)
    if sweep === :combined && n > 0
        # Enzyme names the blocks of the reverse sweep `invert…`. A block that the entry only
        # reaches through one of them is in the reverse sweep, whatever it is called now.
        reached = falses(n)
        if !inverted[1]
            reached[1] = true
            work = Int[1]
            while !isempty(work)
                v = pop!(work)
                for w in succ[v]
                    if !reached[w] && !inverted[w]
                        reached[w] = true
                        push!(work, w)
                    end
                end
            end
        end
        for i in 1:n
            reverse[i] = !reached[i]
        end
    end
    entry = falses(n)
    entry[1] = true
    return ScannedFunction(name, fn, blks, reachable_from(succ, entry), reaches(succ, returning), on_cycle(succ), reverse, sweep)
end

# The source lines of `inst` itself, innermost first: its own and those it was inlined into.
function instruction_frames(inst::LLVM.Instruction)::Vector{StackTraces.StackFrame}
    frames = StackTraces.StackFrame[]
    loc = inst.debug_location
    while loc !== nothing
        scope = loc.scope
        line = Int(loc.line)
        # A wrapper Enzyme generates has no source line.
        if scope !== nothing && line > 0
            name = replace(something(scope.name, ""), r";$" => "")
            isempty(name) && (name = "unknown")
            file = scope.file
            path = file === nothing ? "unknown" :
                joinpath(something(file.directory, ""), something(file.filename, ""))
            push!(frames, StackTraces.StackFrame(Symbol(name), Symbol(path), line))
        end
        loc = loc.inlined_at
    end
    return frames
end

debug_location_key(inst::LLVM.Instruction)::UInt =
    (loc = inst.debug_location; loc === nothing ? UInt(0) : reference_key(loc.ref))

# The calls of one compiled function from one other function at one source position: one call
# site of the function. Calls without debug information form one site per calling function.
mutable struct CallGroup
    const caller::Int
    const call::LLVM.Instruction
    sweep::Symbol
    count::Int
    in_loop::Bool
    throw_only::Bool
end

# What `scan_derivative` knows about the module's calls, to say where a call comes from.
struct CallSiteGraph
    scanned::Vector{ScannedFunction}
    # The calls of each function, in blocks the entry of their function reaches:
    # (calling function, block, call).
    incoming::Vector{Vector{Tuple{Int, Int, LLVM.Instruction}}}
    reachable::BitVector
    returning::BitVector
    in_loop::BitVector
    # The strongly connected component of each function in the call graph, and its members.
    component::Vector{Int}
    members::Dict{Int, Vector{Int}}
    groups::Dict{Int, Vector{CallGroup}}
    own_frames::Dict{UInt, Vector{StackTraces.StackFrame}}
    walks::Dict{Tuple{UInt, Int}, Tuple{Vector{StackTraces.StackFrame}, Int}}
    sites::Dict{Int, Vector{DerivativeCallSite}}
    entries::Dict{Int, Vector{DerivativeCallSite}}
end

function CallSiteGraph(
        scanned::Vector{ScannedFunction},
        callees::Vector{Vector{Int}},
        incoming::Vector{Vector{Tuple{Int, Int, LLVM.Instruction}}},
        reachable::BitVector,
        returning::BitVector,
        in_loop::BitVector,
    )
    component = strongly_connected_components(callees)
    members = Dict{Int, Vector{Int}}()
    for (k, c) in enumerate(component)
        push!(get!(Vector{Int}, members, c), k)
    end
    return CallSiteGraph(
        scanned, incoming, reachable, returning, in_loop, component, members,
        Dict{Int, Vector{CallGroup}}(), Dict{UInt, Vector{StackTraces.StackFrame}}(),
        Dict{Tuple{UInt, Int}, Tuple{Vector{StackTraces.StackFrame}, Int}}(),
        Dict{Int, Vector{DerivativeCallSite}}(), Dict{Int, Vector{DerivativeCallSite}}(),
    )
end

function own_frames(g::CallSiteGraph, inst::LLVM.Instruction)::Vector{StackTraces.StackFrame}
    key = debug_location_key(inst)
    key == 0 && return StackTraces.StackFrame[]
    frames = get(g.own_frames, key, nothing)
    if frames === nothing
        frames = instruction_frames(inst)
        g.own_frames[key] = frames
    end
    return frames
end

# The call sites of function `f` in code the derivative can reach.
function call_groups(g::CallSiteGraph, f::Int)::Vector{CallGroup}
    known = get(g.groups, f, nothing)
    known === nothing || return known
    groups = CallGroup[]
    index = Dict{Tuple{Int, UInt}, Int}()
    for (k, b, call) in g.incoming[f]
        g.reachable[k] || continue
        sf = g.scanned[k]
        sweep = sf.sweep === :combined && sf.reverse[b] ? :reverse : sf.sweep
        looped = g.in_loop[k] || sf.cyclic[b]
        thrown = !g.returning[k] || !sf.returns[b]
        key = (k, debug_location_key(call))
        i = get(index, key, 0)
        if i == 0
            push!(groups, CallGroup(k, call, sweep, 1, looped, thrown))
            index[key] = length(groups)
        else
            group = groups[i]
            group.count += 1
            group.sweep === sweep || (group.sweep = sf.sweep)
            group.in_loop |= looped
            group.throw_only &= thrown
        end
    end
    g.groups[f] = groups
    return groups
end

# The source lines `inst` in function `k` comes from, innermost first: its own, those it was
# inlined into, and those of the call of its function while that function has exactly one call
# site. Also the function where they stop.
function walked_frames(g::CallSiteGraph, inst::LLVM.Instruction, k::Int)::Tuple{Vector{StackTraces.StackFrame}, Int}
    key = (debug_location_key(inst), k)
    known = get(g.walks, key, nothing)
    known === nothing || return known
    frames = copy(own_frames(g, inst))
    f = k
    visited = BitSet((k,))
    while true
        groups = call_groups(g, f)
        length(groups) == 1 || break
        group = only(groups)
        group.caller in visited && break
        append!(frames, own_frames(g, group.call))
        push!(visited, group.caller)
        f = group.caller
    end
    result = (frames, f)
    g.walks[key] = result
    return result
end

function call_site_isless(a::DerivativeCallSite, b::DerivativeCallSite)::Bool
    a.throw_only == b.throw_only || return !a.throw_only
    a.in_function == b.in_function || return a.in_function < b.in_function
    la = isempty(a.frames) ? 0 : first(a.frames).line
    lb = isempty(b.frames) ? 0 : first(b.frames).line
    la == lb || return la < lb
    return length(a.frames) < length(b.frames)
end

# The call site for the calls of `group`, which call a function of component `component`.
function call_site(g::CallSiteGraph, group::CallGroup, component::Int)::DerivativeCallSite
    frames, stop = walked_frames(g, group.call, group.caller)
    above = if g.component[stop] == component
        # Its lines stop inside the recursion it calls into: say how that recursion is entered.
        recursion_entries(g, component)
    else
        call_sites(g, stop)
    end
    return DerivativeCallSite(
        g.scanned[group.caller].name, group.sweep, frames, group.in_loop, group.throw_only,
        group.count, above,
    )
end

# The call sites of function `f`, listed when lines stop there.
function call_sites(g::CallSiteGraph, f::Int)::Vector{DerivativeCallSite}
    known = get(g.sites, f, nothing)
    known === nothing || return known
    sites = DerivativeCallSite[]
    for group in call_groups(g, f)
        push!(sites, call_site(g, group, g.component[f]))
    end
    sort!(sites; lt = call_site_isless)
    g.sites[f] = sites
    return sites
end

# The calls that enter the recursion `component` from outside it. Each is in a function the
# recursion does not reach, so listing what is above them cannot come back here.
function recursion_entries(g::CallSiteGraph, component::Int)::Vector{DerivativeCallSite}
    known = get(g.entries, component, nothing)
    known === nothing || return known
    sites = DerivativeCallSite[]
    for f in g.members[component], group in call_groups(g, f)
        g.component[group.caller] == component && continue
        push!(sites, call_site(g, group, component))
    end
    sort!(sites; lt = call_site_isless)
    g.entries[component] = sites
    return sites
end

const FINDING_KIND_ORDER = (:dynamic_dispatch, :enzyme_runtime_fallback, :box, :gc_allocation, :native_allocation, :runtime_call)

function finding_kind_rank(kind::Symbol)::Int
    for (i, k) in enumerate(FINDING_KIND_ORDER)
        k === kind && return i
    end
    return length(FINDING_KIND_ORDER) + 1
end

# Findings on returning paths first, then by kind.
function finding_isless(a::DerivativeFinding, b::DerivativeFinding)::Bool
    a.throw_only == b.throw_only || return !a.throw_only
    ra = finding_kind_rank(a.kind)
    rb = finding_kind_rank(b.kind)
    ra == rb || return ra < rb
    a.in_function == b.in_function || return a.in_function < b.in_function
    a.callee == b.callee || return a.callee < b.callee
    return something(a.bytes, -1) < something(b.bytes, -1)
end

"""
    scan_derivative(mod, global_names = nothing)::Vector{DerivativeFinding}

The calls in the optimized module of a derivative that leave it (see
[`DerivativeFinding`](@ref)). `global_names` gives the function that a global of the module
stands for, where a module compiled for a package image names an address.

The module is only read.
"""
function scan_derivative(mod::LLVM.Module, global_names::Union{Nothing, Dict{String, String}} = nothing)::Vector{DerivativeFinding}
    definitions = [fn for fn in mod.functions if !isempty(fn.blocks)]
    defined = Dict{String, Int}(fn.name => k for (k, fn) in enumerate(definitions))
    scanned = ScannedFunction[scanned_function(fn, defined) for fn in definitions]
    nf = length(scanned)
    callees = Vector{Int}[Int[] for _ in 1:nf]
    returning_callees = Vector{Int}[Int[] for _ in 1:nf]
    loop_calls = Tuple{Int, Int}[]
    roots = falses(nf)
    external = Tuple{Int, Int, LLVM.Instruction, String}[]
    incoming = Vector{Tuple{Int, Int, LLVM.Instruction}}[Tuple{Int, Int, LLVM.Instruction}[] for _ in 1:nf]
    address_names = Dict{UInt, String}()
    for (k, sf) in enumerate(scanned)
        # A local definition with no callers can survive optimization. Only exported
        # or indirectly callable functions need to be treated as independent entries.
        roots[k] = !(sf.fn.linkage in (LLVM.Linkage.Internal, LLVM.Linkage.Private)) || has_indirect_callers(sf.fn)
        for (b, bb) in enumerate(sf.blocks)
            sf.reachable[b] || continue
            for inst in bb.instructions
                inst isa LLVM.CallBase || continue
                how, name, idx = called_function(inst, defined, global_names, address_names)
                if how === :internal
                    push!(callees[k], idx)
                    push!(incoming[idx], (k, b, inst))
                    sf.returns[b] && push!(returning_callees[k], idx)
                    sf.cyclic[b] && push!(loop_calls, (k, idx))
                elseif how === :named
                    push!(external, (k, b, inst, name))
                elseif how === :indirect
                    push!(external, (k, b, inst, ""))
                end
            end
        end
    end

    # A recursive group cannot make itself reachable on a returning path: some
    # entry must reach it through call sites that can all return.
    reachable = reachable_from(callees, roots)
    returning = reachable_from(returning_callees, roots)
    loop_roots = reachable .& on_cycle(callees)
    for (caller, callee) in loop_calls
        reachable[caller] && (loop_roots[callee] = true)
    end
    in_loop = reachable_from(callees, loop_roots)
    graph = CallSiteGraph(scanned, callees, incoming, reachable, returning, in_loop)

    counts = Dict{DerivativeFinding, Int}()
    order = DerivativeFinding[]
    # Where the lines of the calls in each function stop: the same for all of them.
    stops = Dict{String, Int}()
    for (k, b, inst, name) in external
        reachable[k] || continue
        sf = scanned[k]
        # A call through a pointer computed at run time has no name to classify.
        kind, callee = if isempty(name)
            (:runtime_call, "(function pointer)")
        else
            classify_callee(inst, name)
        end
        frames, stop = walked_frames(graph, inst, k)
        stops[sf.name] = stop
        sweep = sf.sweep === :combined && sf.reverse[b] ? :reverse : sf.sweep
        finding = DerivativeFinding(
            kind, callee, sf.name, sweep, frames,
            in_loop[k] || sf.cyclic[b],
            !returning[k] || !sf.returns[b],
            isempty(name) ? nothing : allocation_bytes(inst, kind, name),
            1,
        )
        seen = get(counts, finding, 0)
        seen == 0 && push!(order, finding)
        counts[finding] = seen + 1
    end
    findings = DerivativeFinding[]
    for f in order
        push!(findings, DerivativeFinding(f.kind, f.callee, f.in_function, f.sweep, f.frames, f.in_loop, f.throw_only, f.bytes, counts[f], call_sites(graph, stops[f.in_function])))
    end
    sort!(findings; lt = finding_isless)
    return findings
end

"""
    DerivativeRecord

What Enzyme knows about one compiled derivative: the job it was compiled for, how this session
got its code (`:compiled`, `:image` for code linked from a package image, or `:inlined` for a
module handed to Julia for inlining into its caller), and what [`scan_derivative`](@ref) found
in it. `findings` is `nothing` for code that was not scanned, and `note` then says why.
"""
mutable struct DerivativeRecord
    const job::CompilerJob
    const source::Symbol
    const findings::Union{Nothing, Vector{DerivativeFinding}}
    const note::String
end

# The record of the derivative at each address the JIT gave out: both entry points of a
# split-mode derivative lead to the same one. Guarded by `cache_publish_lock`.
const derivative_records = Dict{Ptr{Cvoid}, DerivativeRecord}()

# The record of each derivative compiled for inlining into its caller, by the type of the
# `Val` its thunk holds in place of an address. Guarded by `cache_publish_lock`.
const inlined_derivative_records = IdDict{Any, DerivativeRecord}()

# Derivatives handed to a caller under a report that Enzyme has no record of. There should be
# none: every path that compiles or links a derivative registers it.
const unrecorded_derivative_calls = Threads.Atomic{Int}(0)

function derivative_record(ptr::Ptr{Cvoid})::Union{Nothing, DerivativeRecord}
    lock(cache_publish_lock)
    try
        return get(derivative_records, ptr, nothing)
    finally
        unlock(cache_publish_lock)
    end
end

function inlined_derivative_record(@nospecialize(code::Val))::Union{Nothing, DerivativeRecord}
    lock(cache_publish_lock)
    try
        return get(inlined_derivative_records, typeof(code), nothing)
    finally
        unlock(cache_publish_lock)
    end
end

# Keep `record` for the derivative whose thunk holds `code`: the address of its compiled code,
# or the `Val` that stands for a module compiled for inlining.
function register_derivative!(ptr::Ptr{Cvoid}, record::DerivativeRecord)
    ptr == C_NULL && return nothing
    lock(cache_publish_lock)
    try
        derivative_records[ptr] = record
    finally
        unlock(cache_publish_lock)
    end
    return nothing
end

function register_derivative!(@nospecialize(code::Val), record::DerivativeRecord)
    lock(cache_publish_lock)
    try
        inlined_derivative_records[typeof(code)] = record
    finally
        unlock(cache_publish_lock)
    end
    return nothing
end

register_derivative!(::Nothing, ::DerivativeRecord) = nothing

# Scan `mod`, the module of `job` as the JIT gets it. A failure of the scan must not cost the
# derivative: the record keeps the reason, and a report shows it.
function scanned_record(job::CompilerJob, mod::LLVM.Module, source::Symbol, global_names::Union{Nothing, Dict{String, String}} = nothing)::DerivativeRecord
    findings, note = scanned_findings(mod, global_names)
    return DerivativeRecord(job, source, findings, note)
end

function scanned_findings(mod::LLVM.Module, global_names::Union{Nothing, Dict{String, String}} = nothing)::Tuple{Union{Nothing, Vector{DerivativeFinding}}, String}
    try
        return (scan_derivative(mod, global_names), "")
    catch err
        return (nothing, "the scan of the compiled code failed: " * sprint(showerror, err))
    end
end

const INLINED_DERIVATIVE_NOTE = "compiled for inlining into its caller, which Julia optimizes as part of that caller"

"""
    DerivativeCollector

What a running [`derivative_report`](@ref) has seen so far. `parent` is the collector of a
report around this one, which sees the same derivatives.
"""
mutable struct DerivativeCollector
    const keep_ir::Bool
    const parent::Union{Nothing, DerivativeCollector}
    const lock::Threads.SpinLock
    const records::Vector{DerivativeRecord}
    const origins::Vector{Symbol}
    const ir::Vector{Union{Nothing, Vector{UInt8}}}
    const index::IdDict{DerivativeRecord, Int}
end

DerivativeCollector(keep_ir::Bool, parent::Union{Nothing, DerivativeCollector}) =
    DerivativeCollector(keep_ir, parent, Threads.SpinLock(), DerivativeRecord[], Symbol[], Union{Nothing, Vector{UInt8}}[], IdDict{DerivativeRecord, Int}())

# The collector of the report the current task runs under, if any.
const DERIVATIVE_COLLECTOR = ScopedValues.ScopedValue{Union{Nothing, DerivativeCollector}}(nothing)

# The number of reports running in the process. A call of a derivative reads it and nothing
# else while it is zero. It is a field and not a `Threads.Atomic`: with a read of one of those
# in its code, Julia 1.12 no longer inlined `autodiff` into its caller, and the call then
# returned its result boxed.
mutable struct RunningReports
    @atomic count::Int
end
const DERIVATIVE_REPORTS = RunningReports(0)

@inline reports_running()::Bool = (@atomic :acquire DERIVATIVE_REPORTS.count) != 0

@inline function current_derivative_collector()::Union{Nothing, DerivativeCollector}
    reports_running() || return nothing
    return DERIVATIVE_COLLECTOR[]
end

function collect_derivative!(collector::DerivativeCollector, record::DerivativeRecord, origin::Symbol, ir::Union{Nothing, Vector{UInt8}})
    current = collector
    while current !== nothing
        lock(current.lock)
        try
            if !haskey(current.index, record)
                push!(current.records, record)
                push!(current.origins, origin)
                push!(current.ir, current.keep_ir ? ir : nothing)
                current.index[record] = length(current.records)
            end
        finally
            unlock(current.lock)
        end
        current = current.parent
    end
    return nothing
end

# Whether any report the current task runs under wants the bitcode of what it compiles.
function collects_ir(collector::Union{Nothing, DerivativeCollector})::Bool
    current = collector
    while current !== nothing
        current.keep_ir && return true
        current = current.parent
    end
    return false
end

function module_bitcode(mod::LLVM.Module)::Vector{UInt8}
    buf = convert(MemoryBuffer, mod)
    try
        return convert(Vector{UInt8}, buf)
    finally
        dispose(buf)
    end
end

"""
    compiled_derivative_record(job, mod) -> (record, ir)

The record of the derivative of `job` that was just compiled into `mod`, and the bitcode of
`mod` if a running report asked for it. Called between compiling and handing `mod` to the JIT,
which takes the module.
"""
function compiled_derivative_record(job::CompilerJob, mod::LLVM.Module)::Tuple{DerivativeRecord, Union{Nothing, Vector{UInt8}}}
    if job.config.params.ABI <: InlineABI
        return (DerivativeRecord(job, :inlined, nothing, INLINED_DERIVATIVE_NOTE), nothing)
    end
    record = scanned_record(job, mod, :compiled)
    ir = collects_ir(current_derivative_collector()) ? module_bitcode(mod) : nothing
    return (record, ir)
end

"""
    publish_derivative_record!(record, ir, adjoint, primal)

Keep `record` for the derivative whose thunks hold `adjoint` and `primal` (addresses the JIT
gave out, or the `Val`s of a derivative compiled for inlining), and tell a running report
that it was compiled.
"""
function publish_derivative_record!(record::DerivativeRecord, ir::Union{Nothing, Vector{UInt8}}, @nospecialize(adjoint::Union{Nothing, Ptr{Cvoid}, Val}), @nospecialize(primal::Union{Nothing, Ptr{Cvoid}, Val}))
    register_derivative!(adjoint, record)
    register_derivative!(primal, record)
    collector = current_derivative_collector()
    collector === nothing || collect_derivative!(collector, record, record.source === :inlined ? :inlined : :compiled, ir)
    return nothing
end

"""
    linked_derivative_record!(job, findings, note, adjoint, primal)

Keep the record of a derivative that was linked from a package image, which kept `findings`
with its machine code, and tell a running report that it was linked.
"""
function linked_derivative_record!(job::CompilerJob, findings::Union{Nothing, Vector{DerivativeFinding}}, note::String, adjoint::Ptr{Cvoid}, primal::Ptr{Cvoid})
    record = derivative_record(adjoint)
    if record === nothing
        record = DerivativeRecord(job, :image, findings, note)
        register_derivative!(adjoint, record)
        register_derivative!(primal, record)
    end
    collector = current_derivative_collector()
    collector === nothing || collect_derivative!(collector, record, :image, nothing)
    return nothing
end

"""
    note_derivative_use(ptr)

Tell a running [`derivative_report`](@ref) that the caller is about to call the derivative
at `ptr`. While no report is running this reads one counter.

The functions that hand a compiled derivative to the code that calls it run this:
`lazy_thunk`, and through [`noted_thunks`](@ref) `lazy_split_thunks` and the code that the
generator of `thunk` returns. It is not in the code that makes the call (`enzyme_call`),
because code compiled for a device shares that code: a kernel calls the derivative it got
from `autodiff_deferred`, and it can neither read a counter of the host nor call into it.
So two uses go unseen once a derivative is compiled: a call through `autodiff_deferred`, and
a call of a split-mode thunk that its caller got from `autodiff_thunk` earlier and kept.

Code that Enzyme itself differentiates gets the inner derivative of a nested one as a
constant, without this: that derivative becomes part of the outer one's code, which is what
a report then describes.
"""
@inline note_derivative_use(ptr::Ptr{Cvoid}) = (Enzyme.within_autodiff() || !reports_running()) ? nothing : used_derivative(ptr)

"""
    noted_thunks(thunks)

`thunks`, one thunk or the two of a split-mode derivative, after telling a running
[`derivative_report`](@ref) that the caller is about to call them (see
[`note_derivative_use`](@ref)).
"""
@inline function noted_thunks(thunks::T) where {T}
    (Enzyme.within_autodiff() || !reports_running()) || used_thunks(thunks)
    return thunks
end

@noinline function used_derivative(ptr::Ptr{Cvoid})
    collector = DERIVATIVE_COLLECTOR[]
    collector === nothing && return nothing
    record = derivative_record(ptr)
    if record === nothing
        Threads.atomic_add!(unrecorded_derivative_calls, 1)
        return nothing
    end
    collect_derivative!(collector, record, record.source === :image ? :image : :cached, nothing)
    return nothing
end

# A derivative compiled for inlining into its caller has no code of its own to inspect.
function used_inlined_derivative(@nospecialize(code::Val))
    collector = DERIVATIVE_COLLECTOR[]
    collector === nothing && return nothing
    record = inlined_derivative_record(code)
    if record === nothing
        Threads.atomic_add!(unrecorded_derivative_calls, 1)
        return nothing
    end
    collect_derivative!(collector, record, :inlined, nothing)
    return nothing
end

# The first field of a thunk is what its call runs: the address of the compiled code, or
# for a derivative compiled for inlining the `Val` of its module and function.
function used_thunk(@nospecialize(thunk))
    thunk isa AbstractThunk || return nothing
    code = getfield(thunk, 1)
    if code isa Ptr{Cvoid}
        used_derivative(code)
    elseif code isa Val
        used_inlined_derivative(code)
    end
    return nothing
end

@noinline function used_thunks(@nospecialize(thunks))
    if thunks isa Tuple
        for thunk in thunks
            used_thunk(thunk)
        end
    else
        used_thunk(thunks)
    end
    return nothing
end

"""
    ReportedDerivative

One derivative in a [`DerivativeReport`](@ref).

- `func`: the type of the function that was differentiated.
- `mode`: `:reverse`, `:reverse_split` (the forward and the reverse sweep are called
  separately) or `:forward`.
- `arguments`: the tuple type of the argument annotations, as Enzyme compiled it.
- `return_annotation`: the annotation of the return value.
- `width`: the number of derivatives computed at once.
- `runtime_activity`, `strong_zero`, `atomic_add`: the settings of the mode. `atomic_add` is
  `nothing` where the number of threads decided it, and `:inherited_plain` for a derivative
  Enzyme compiled for a call dispatched at run time inside a derivative that accumulates
  plainly (see [`set_atomic_add`](@ref)): it does so too, unless it runs work in parallel.
- `origin`: `:compiled` while the report ran, `:cached` for code compiled earlier in the
  session, `:image` for code linked from a package image, or `:inlined` for a derivative
  compiled for inlining into its caller (`InlineABI`), whenever that was. A split-mode
  derivative is compiled while Julia infers the code that asks for it, which may be before
  that code runs: it is then `:cached`.
- `findings`: what the compiled code calls (see [`DerivativeFinding`](@ref)), or `nothing` if
  it was not inspected: an `:inlined` derivative has no code of its own, since Julia
  optimizes it as part of its caller. `note` says why.
- `ir`: the LLVM bitcode of the optimized module, for a derivative compiled while a report
  with `ir = true` ran, and `nothing` otherwise.
"""
struct ReportedDerivative
    func::Type
    mode::Symbol
    arguments::Type
    return_annotation::Type
    width::Int
    runtime_activity::Bool
    strong_zero::Bool
    atomic_add::Union{Nothing, Bool, Symbol}
    origin::Symbol
    findings::Union{Nothing, Vector{DerivativeFinding}}
    note::String
    ir::Union{Nothing, Vector{UInt8}}
end

function reported_mode(mode::API.CDerivativeMode)::Symbol
    mode == API.DEM_ForwardMode && return :forward
    mode == API.DEM_ReverseModeCombined && return :reverse
    return :reverse_split
end

reported_atomic_add(setting::Union{Nothing, Bool}) = setting
reported_atomic_add(::Type{InheritedPlainAdd}) = :inherited_plain
reported_atomic_add(variant::ForSeveralThreads) = reported_atomic_add(variant.setting)

function reported_derivative(record::DerivativeRecord, origin::Symbol, ir::Union{Nothing, Vector{UInt8}})::ReportedDerivative
    params = record.job.config.params
    annotations = params.TT.parameters
    func = eltype(annotations[1])
    arguments = Tuple{annotations[2:end]...}
    return ReportedDerivative(
        func, reported_mode(params.mode), arguments, params.rt, params.width,
        params.runtimeActivity, params.strongZero, reported_atomic_add(params.atomicAdd),
        origin, record.findings, record.note, ir,
    )
end

"""
    DerivativeReport

What [`derivative_report`](@ref) returns: `value`, the result of the code it ran, and
`derivatives`, one [`ReportedDerivative`](@ref) for each derivative that code compiled, linked
from a package image or asked Enzyme for again, in the order they were first seen.
"""
struct DerivativeReport
    value::Any
    derivatives::Vector{ReportedDerivative}
end

"""
    derivative_report(f; ir = false)::DerivativeReport

Run `f()` and report the derivatives Enzyme used for it: each one that was compiled, linked
from a package image, or asked for again while `f()` ran, on the calling task and the tasks it
started. `f` may take derivatives directly (`autodiff`, `autodiff_thunk`, `gradient`,
`jacobian`) or through another package.

For each derivative the report says what was differentiated and lists the calls in its
compiled code that leave it: run-time dispatch, Enzyme's run-time differentiation of a call it
could not resolve, boxes, allocations on Julia's heap, native allocations such as Enzyme's
tape, and other calls into the runtime or a library, each with its source lines. See
[`ReportedDerivative`](@ref) and [`DerivativeFinding`](@ref). The report judges nothing: which
findings matter is up to the caller.

Enzyme reads the code of every derivative once, when it compiles it, and keeps what it found
with the compiled code, so the report does not depend on when a derivative was compiled, and
costs no second compile. With `ir = true`, a derivative compiled while `f()` runs also carries
the LLVM bitcode of its optimized module.

A derivative nested in another one, such as the inner one of Forward over Reverse, runs as
part of the outer one's code and is reported there. A derivative compiled for inlining into
its caller (`InlineABI`) is listed without findings.

A derivative compiled before `f()` ran is seen when `f()` asks Enzyme for it, which every
call of `autodiff`, `autodiff_thunk`, `gradient` and `jacobian` does. Two uses of such a
derivative are not seen: a call through `autodiff_deferred`, and a call of a split-mode thunk
that `f` got from `autodiff_thunk` before the report and kept. Code that runs on a device
shares the code of these calls, so they cannot carry the check. Take the report around the
first call to see them.

```julia
report = Enzyme.derivative_report() do
    Enzyme.gradient(Reverse, f, x)
end
for d in report.derivatives, finding in something(d.findings, [])
    finding.throw_only && continue
    finding.kind in (:dynamic_dispatch, :enzyme_runtime_fallback) && @warn "dispatch at run time" d.func finding
end
```
"""
function derivative_report(f::F; ir::Bool = false) where {F}
    collector = DerivativeCollector(ir, DERIVATIVE_COLLECTOR[])
    @atomic DERIVATIVE_REPORTS.count += 1
    value = try
        @with DERIVATIVE_COLLECTOR => collector f()
    finally
        @atomic DERIVATIVE_REPORTS.count -= 1
    end
    derivatives = ReportedDerivative[]
    lock(collector.lock)
    try
        for i in eachindex(collector.records)
            push!(derivatives, reported_derivative(collector.records[i], collector.origins[i], collector.ir[i]))
        end
    finally
        unlock(collector.lock)
    end
    return DerivativeReport(value, derivatives)
end

const ORIGIN_DESCRIPTIONS = Dict{Symbol, String}(
    :compiled => "compiled during the report",
    :cached => "compiled earlier in this session",
    :image => "linked from a package image",
    :inlined => "compiled for inlining into its caller",
)

function print_derivative_heading(io::IO, d::ReportedDerivative)
    print(io, d.mode, " derivative of ", d.func, " with ")
    print(io, "(")
    params = d.arguments.parameters
    for (i, p) in enumerate(params)
        i == 1 || print(io, ", ")
        print(io, p)
    end
    print(io, ") -> ", d.return_annotation)
    d.width == 1 || print(io, ", width ", d.width)
    print(io, "; ", get(ORIGIN_DESCRIPTIONS, d.origin, String(d.origin)))
    return nothing
end

# How many calls of each kind `findings` holds, on paths that return or on those that do not.
function print_finding_counts(io::IO, findings::Vector{DerivativeFinding}, throw_only::Bool)
    first = true
    for kind in FINDING_KIND_ORDER
        n = 0
        for f in findings
            if f.kind === kind && f.throw_only == throw_only
                n += f.count
            end
        end
        n == 0 && continue
        first || print(io, ", ")
        first = false
        print(io, n, " ", kind)
    end
    first && print(io, "none")
    return nothing
end

function Base.show(io::IO, f::DerivativeFinding)
    print(io, f.kind, " ", f.callee)
    f.bytes === nothing || print(io, " (", f.bytes, " bytes)")
    f.count == 1 || print(io, " ×", f.count)
    print(io, " in ", f.in_function, " (", f.sweep, ")")
    f.in_loop && print(io, ", in a loop")
    f.throw_only && print(io, ", on an error path")
    return nothing
end

function print_frames(io::IO, frames::Vector{StackTraces.StackFrame}, indent::String)
    for (i, frame) in enumerate(frames)
        print(io, "\n", indent, i == 1 ? "at " : "   ", frame.func, " @ ", frame.file, ":", frame.line)
    end
    return nothing
end

# Call sites are shared, so each vector is printed once per finding, and referred to after.
function print_call_sites(io::IO, sites::Vector{DerivativeCallSite}, indent::String, printed::Base.IdSet{Any})
    isempty(sites) && return nothing
    n = length(sites)
    if sites in printed
        print(io, "\n", indent, "called from the same ", n, n == 1 ? " site" : " sites", " as above")
        return nothing
    end
    push!(printed, sites)
    print(io, "\n", indent, "called from ", n, n == 1 ? " site:" : " sites:")
    for site in sites
        print(io, "\n", indent, "- in ", site.in_function, " (", site.sweep, ")")
        site.count == 1 || print(io, " ×", site.count)
        site.in_loop && print(io, ", in a loop")
        site.throw_only && print(io, ", on an error path")
        print_frames(io, site.frames, indent * "  ")
        print_call_sites(io, site.call_sites, indent * "  ", printed)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", f::DerivativeFinding)
    show(io, f)
    print_frames(io, f.frames, "    ")
    print_call_sites(io, f.call_sites, "    ", Base.IdSet{Any}())
    return nothing
end

function Base.show(io::IO, site::DerivativeCallSite)
    print(io, "call in ", site.in_function, " (", site.sweep, ")")
    site.count == 1 || print(io, " ×", site.count)
    isempty(site.frames) || print(io, " at ", first(site.frames).func, " @ ", first(site.frames).file, ":", first(site.frames).line)
    n = length(site.call_sites)
    n == 0 || print(io, ", called from ", n, n == 1 ? " site" : " sites")
    return nothing
end

function Base.show(io::IO, d::ReportedDerivative)
    print_derivative_heading(io, d)
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", d::ReportedDerivative)
    print_derivative_heading(io, d)
    findings = d.findings
    if findings === nothing
        print(io, "\n  not inspected: ", d.note)
        return nothing
    end
    print(io, "\n  on paths that return: ")
    print_finding_counts(io, findings, false)
    print(io, "\n  on error paths: ")
    print_finding_counts(io, findings, true)
    for f in findings
        (f.throw_only || f.kind === :runtime_call) && continue
        print(io, "\n  - ")
        show(io, mime, f)
    end
    return nothing
end

function Base.show(io::IO, r::DerivativeReport)
    print(io, "DerivativeReport(", length(r.derivatives), length(r.derivatives) == 1 ? " derivative)" : " derivatives)")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", r::DerivativeReport)
    show(io, r)
    for (i, d) in enumerate(r.derivatives)
        print(io, "\n[", i, "] ")
        show(io, mime, d)
    end
    return nothing
end
