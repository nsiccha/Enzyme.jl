using Enzyme, Test

const LLVM = Enzyme.Compiler.LLVM

# Julia 1.12 and later leave part of a fresh `Memory`'s header to the code that allocates it,
# with no safepoint in between:
#
# - A `Memory` of a constant length that fits a pool cell (`zeros(64)`) is an object
#   allocation, `julia.gc_alloc_obj`, followed by stores of its data pointer and its length.
# - Any other `Memory` comes from `jl_alloc_genericmemory_unchecked` with its data pointer
#   set, and the calling code stores its length and, where elements hold references, then
#   zeroes its data.
#
# Enzyme allocates the shadow, which is a safepoint, next to such an allocation: Forward
# mode after the original, Reverse mode before it. A collection that starts there marks the
# earlier `Memory` with whatever its cell held before. If the data pointer is not the inline
# data, the collector takes the word after the header for the owner of the data and follows
# it (EnzymeAD/Enzyme.jl#3698):
#
#     signal 11: Segmentation fault
#     gc_try_claim_and_push at gc-stock.c:1688
#     gc_mark_outrefs at gc-stock.c:2401
#
# and where elements hold references it scans `length` of them (`gc_mark_objarray`,
# `gc_mark_memory8`). A cell that last held a `Memory` of the same size has a valid header
# left in it, which is why this crashed only once in many thousand calls. Enzyme now stores
# the header of an empty `Memory` right after such an allocation, wherever a safepoint
# comes before the allocating code's own stores.

const MHI_LENGTH = 64

@noinline function mhi_weighted(out, w)
    s = 0.0
    @inbounds for i in eachindex(out)
        s += out[i] * w
    end
    return s
end

# 2080 * x[1] * x[2], through a `Memory{Float64}` of constant length.
function mhi_filled(x)
    out = zeros(MHI_LENGTH)
    @inbounds for i in 1:MHI_LENGTH
        out[i] = x[1] * i
    end
    return mhi_weighted(out, x[2])
end

@noinline function mhi_total(parts)
    s = 0.0
    for p in parts, v in p
        s += v
    end
    return s
end

# n * (n + 1) / 2 * (x[1] + x[2]), through a `Memory` whose elements are references: of
# constant length in `mhi_boxed`, of a length known only at run time in `mhi_dynamic`.
function mhi_boxed(x)
    parts = Vector{Vector{Float64}}(undef, 4)
    for i in 1:4
        parts[i] = x .* i
    end
    return mhi_total(parts)
end
function mhi_dynamic(x, n)
    parts = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        parts[i] = x .* i
    end
    return mhi_total(parts)
end

mhi_gradient!(dx, x) = (autodiff(Reverse, Const(mhi_filled), Active, Duplicated(x, dx)); nothing)
mhi_tangent(x, v) = autodiff(Forward, Const(mhi_filled), Duplicated(x, v))[1]

@testset "Derivatives through a fresh array" begin
    x = [1.5, 2.0]
    v = [1.0, -0.5]
    @test autodiff(Forward, Const(mhi_filled), Duplicated(x, v))[1] ≈ 2080 * (v[1] * x[2] + x[1] * v[2])
    batch = autodiff(Forward, Const(mhi_filled), BatchDuplicated(x, ([1.0, 0.0], [0.0, 1.0])))[1]
    @test batch[1] ≈ 2080 * x[2]
    @test batch[2] ≈ 2080 * x[1]

    dx = zero(x)
    mhi_gradient!(dx, x)
    @test dx ≈ 2080 .* [x[2], x[1]]

    dx = zero(x)
    ddx = zero(x)
    autodiff(Forward, Const(mhi_gradient!), Const, Duplicated(dx, ddx), Duplicated(x, v))
    @test dx ≈ 2080 .* [x[2], x[1]]
    @test ddx ≈ 2080 .* [v[2], v[1]]

    dx = zero(x)
    autodiff(Reverse, Const(mhi_tangent), Active, Duplicated(x, dx), Const(v))
    @test dx ≈ 2080 .* [v[2], v[1]]

    @test autodiff(Forward, Const(mhi_boxed), Duplicated(x, v))[1] ≈ 10 * (v[1] + v[2])
    dx = zero(x)
    autodiff(Reverse, Const(mhi_boxed), Active, Duplicated(x, dx))
    @test dx ≈ [10.0, 10.0]

    @test autodiff(Forward, Const(mhi_dynamic), Duplicated(x, v), Const(5))[1] ≈ 15 * (v[1] + v[2])
    batch = autodiff(Forward, Const(mhi_dynamic), BatchDuplicated(x, ([1.0, 0.0], [0.0, 1.0])), Const(5))[1]
    @test batch[1] ≈ 15
    @test batch[2] ≈ 15
    dx = zero(x)
    autodiff(Reverse, Const(mhi_dynamic), Active, Duplicated(x, dx), Const(5))
    @test dx ≈ [15.0, 15.0]
end

function mhi_callee_name(call)
    callee = LLVM.called_operand(call)
    return callee isa LLVM.Function ? LLVM.name(callee) : ""
end

# The calls of `allocator` in `mod`, of `bytes` bytes where that is given.
function mhi_allocations(mod, allocator, bytes = nothing)
    allocs = LLVM.CallInst[]
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        inst isa LLVM.CallInst && mhi_callee_name(inst) == allocator || continue
        if bytes !== nothing
            size = LLVM.operands(inst)[2]
            size isa LLVM.ConstantInt && convert(Int, size) == bytes || continue
        end
        push!(allocs, inst)
    end
    return allocs
end

function mhi_base(@nospecialize(v))
    while v isa LLVM.GetElementPtrInst || v isa LLVM.AddrSpaceCastInst || v isa LLVM.BitCastInst
        v = LLVM.operands(v)[1]
    end
    return v
end

# How many stores into what `alloc` allocates come before the next call that can collect.
function mhi_stores_before_safepoint(alloc)
    stores = 0
    ref = LLVM.API.LLVMGetNextInstruction(alloc)
    while ref != C_NULL
        inst = LLVM.Instruction(ref)
        if inst isa LLVM.CallInst
            name = mhi_callee_name(inst)
            startswith(name, "llvm.") || name == "julia.pointer_from_objref" || break
        elseif inst isa LLVM.StoreInst && mhi_base(LLVM.operands(inst)[2]) == alloc
            stores += 1
        end
        ref = LLVM.API.LLVMGetNextInstruction(inst)
    end
    return stores
end

# The module of a derivative of `f`, optimized up to, but not including, GC lowering.
function mhi_module(body, f, mode, RT, argtypes, width)
    return Enzyme.Compiler.JuliaContext() do ctx
        _, mod = Enzyme.Compiler.reflect(f, RT, argtypes; second_stage = false, mode, width)
        Enzyme.Compiler.post_optimize!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        body(mod)
    end
end

@static if VERSION >= v"1.12-"
    @testset "A fresh Memory has its header before the next safepoint" begin
        batch = BatchDuplicated{T, 2} where {T}
        modes = (
            (Enzyme.API.DEM_ForwardMode, Duplicated, Duplicated{Vector{Float64}}, 1),
            (Enzyme.API.DEM_ForwardMode, batch, BatchDuplicated{Vector{Float64}, 2}, 2),
            (Enzyme.API.DEM_ReverseModeCombined, Active, Duplicated{Vector{Float64}}, 1),
        )
        for (mode, RT, AT, width) in modes
            # A constant length: the object allocation, then its length and data pointer.
            mhi_module(mhi_filled, mode, RT, Tuple{AT}, width) do mod
                allocs = mhi_allocations(mod, "julia.gc_alloc_obj", 16 + 8 * MHI_LENGTH)
                # The original and one shadow for each direction.
                @test length(allocs) == 1 + width
                for alloc in allocs
                    @test mhi_stores_before_safepoint(alloc) >= 2
                end
            end
            # A length known at run time: the allocator sets the data pointer.
            mhi_module(mhi_dynamic, mode, RT, Tuple{AT, Const{Int}}, width) do mod
                allocs = mhi_allocations(mod, "jl_alloc_genericmemory_unchecked")
                @test length(allocs) >= 1 + width
                for alloc in allocs
                    @test mhi_stores_before_safepoint(alloc) >= 1
                end
            end
        end
    end
end

# Each case takes a derivative many times while objects of the size class of its `Memory`,
# whose bytes are no address, are allocated and dropped, so that a fresh `Memory` lands on a
# cell without a valid header. Without the header stores, every case but the last (plain
# Reverse, which allocates the shadow first and zeroes it) died in the collector in each of
# four runs on Julia 1.12.7.
const MEMORY_HEADER_CHILD = raw"""
using Enzyme, Random

# 16 + 512 bytes, as a `Memory` of 64 `Float64` or of 64 references.
mutable struct Poison
    words::NTuple{66, UInt64}
end
const WORD = 0x4141414141414141
@noinline function poison!(keep, n)
    for i in 1:n
        keep[i] = Poison(ntuple(Returns(WORD), Val(66)))
    end
    return nothing
end

@noinline function weighted(out, w)
    s = 0.0
    @inbounds for i in eachindex(out)
        s += out[i] * w
    end
    return s
end
function filled(x)
    out = zeros(64)
    @inbounds for i in 1:64
        out[i] = x[1] * i
    end
    return weighted(out, x[2])
end
@noinline function total(parts)
    s = 0.0
    for p in parts, v in p
        s += v
    end
    return s
end
function boxed(x)
    parts = Vector{Vector{Float64}}(undef, 64)
    fill!(parts, x .* 2.0)
    return total(parts)
end
function dynamic(x, n)
    parts = Vector{Vector{Float64}}(undef, n)
    fill!(parts, x .* 2.0)
    return total(parts)
end
gradient!(dx, x) = (autodiff(Reverse, Const(filled), Active, Duplicated(x, dx)); nothing)
tangent(x, v) = autodiff(Forward, Const(filled), Duplicated(x, v))[1]

forward(x, v) = autodiff(Forward, Const(filled), Duplicated(x, v))[1]
batch(x, v) = sum(values(autodiff(Forward, Const(filled), BatchDuplicated(x, (v, copy(v))))[1]))
function forward_over_reverse(x, v)
    dx = zero(x); ddx = zero(x)
    autodiff(Forward, Const(gradient!), Const, Duplicated(dx, ddx), Duplicated(x, v))
    return ddx[2]
end
function reverse_over_forward(x, v)
    dx = zero(x)
    autodiff(Reverse, Const(tangent), Active, Duplicated(x, dx), Const(v))
    return dx[2]
end
boxed_forward(x, v) = autodiff(Forward, Const(boxed), Duplicated(x, v))[1]
dynamic_forward(x, v) = autodiff(Forward, Const(dynamic), Duplicated(x, v), Const(64))[1]
function reverse(x, v)
    dx = zero(x)
    gradient!(dx, x)
    return dx[1]
end

function run(f::F, rounds, rng) where {F}
    x = [1.5, 2.0]; v = [1.0, 0.0]
    keep = Vector{Poison}(undef, 512)
    expected = f(x, v)
    wrong = 0
    for _ in 1:rounds
        poison!(keep, rand(rng, 64:512))
        fill!(keep, keep[1])            # drop all but one
        for _ in 1:rand(rng, 16:96)
            wrong += f(x, v) != expected
        end
    end
    return wrong
end

const CASES = (
    ("forward", forward, 4160.0),
    ("batch", batch, 8320.0),
    ("forward_over_reverse", forward_over_reverse, 2080.0),
    ("reverse_over_forward", reverse_over_forward, 2080.0),
    ("boxed_forward", boxed_forward, 128.0),
    ("dynamic_forward", dynamic_forward, 128.0),
    ("reverse", reverse, 4160.0),
)
for (name, f, expected) in CASES
    isempty(ARGS[2:end]) || name in ARGS[2:end] || continue
    first = f([1.5, 2.0], [1.0, 0.0])
    wrong = run(f, parse(Int, ARGS[1]), Xoshiro(1))
    println("DONE ", name, " ", first == expected, " ", wrong)
end
"""

@testset "A collection between a Memory's allocation and its header" begin
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $MEMORY_HEADER_CHILD 3000`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
    timer = Timer(900) do _
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
    close(timer)
    output = String(take!(out))
    done = filter(startswith("DONE "), split(output, '\n'))
    @test success(proc)
    @test done == [
        "DONE forward true 0", "DONE batch true 0", "DONE forward_over_reverse true 0",
        "DONE reverse_over_forward true 0", "DONE boxed_forward true 0",
        "DONE dynamic_forward true 0", "DONE reverse true 0",
    ]
    if !success(proc) || length(done) != 7
        println(stderr, "Child output:\n", output)
    end
end
