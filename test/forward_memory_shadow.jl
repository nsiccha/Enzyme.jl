using Enzyme, Test

struct ForwardMemoryHolder
    v::Vector{Float64}
end

# `[a, b]` is a constant-length memory that Julia's codegen allocates inline
# with `julia.gc_alloc_obj` and initializes before its next safepoint. Batched
# Forward emits the shadow allocations between the two, so a collection there
# used to mark memories whose `length` and `ptr` were still stale.
function forward_memory_of_holders(x)
    a = ForwardMemoryHolder(x[2:end])
    b = ForwardMemoryHolder(fill(0.0, length(x) - 1))
    hs = [a, b]
    s = 0.0
    for h in hs
        s += sum(h.v)
    end
    return s * x[1]
end

function repeat_batched_forward(mode, x, basis, n)
    expected = (sum(x[2:end]), x[1], x[1], x[1])
    wrong = 0
    for _ in 1:n
        d, = autodiff(mode, forward_memory_of_holders, BatchDuplicated(x, basis))
        wrong += Tuple(d) == expected ? 0 : 1
    end
    return wrong
end

@testset "Batched Forward of a constant-length memory survives collections" begin
    x = [0.4, 0.2, 0.4, 0.6]
    basis = ntuple(i -> [Float64(j == i) for j in 1:4], 4)
    for mode in (Forward, Enzyme.set_abi(Forward, InlineABI))
        d, = autodiff(mode, forward_memory_of_holders, BatchDuplicated(x, basis))
        @test collect(Tuple(d)) ≈ [1.2, 0.4, 0.4, 0.4]
        # Without the shadow-allocation initialization this aborted with
        # "GC error (probable corruption)" or a segfault in `gc_mark_objarray`
        # within a few seconds of automatic collections (Julia 1.13.1).
        @test repeat_batched_forward(mode, x, basis, 300_000) == 0
    end
end
