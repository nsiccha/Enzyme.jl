using Enzyme, Libdl, Test

# The reverse pass caches each inner loop's trip count in an integer buffer.
# That buffer is constant in the outer forward pass, but still needs freeing.
function variable_trip_sum(x)
    total = 0.0
    for xi in x
        s = 0.0
        j = 0
        while j < 20 && s < 0.7
            s = 0.9s + sin(xi + 0.1s)
            j += 1
        end
        total += s
    end
    return total
end

function variable_trip_gradient!(g, x)
    fill!(g, 0.0)
    autodiff(Reverse, Const(variable_trip_sum), Active, Duplicated(x, g))
    return nothing
end

function variable_trip_hvp!(h, g, x, v)
    fill!(h, 0.0)
    autodiff(Forward, Const(variable_trip_gradient!), Duplicated(g, h), Duplicated(x, v))
    return nothing
end

# Native loop caches are not counted by Julia's allocation counter. On glibc,
# mallinfo2 reports bytes still in use even after Julia's garbage collector ran.
struct LoopCacheMallInfo
    arena::Csize_t
    ordblks::Csize_t
    smblks::Csize_t
    hblks::Csize_t
    hblkhd::Csize_t
    usmblks::Csize_t
    fsmblks::Csize_t
    uordblks::Csize_t
    fordblks::Csize_t
    keepcost::Csize_t
end

function loop_cache_live_bytes(mallinfo2)
    GC.gc(true)
    GC.gc(true)
    info = ccall(mallinfo2, LoopCacheMallInfo, ())
    return Int(info.uordblks)
end

function loop_cache_growth(mallinfo2, h, g, x, v, count)
    variable_trip_hvp!(h, g, x, v)
    before = loop_cache_live_bytes(mallinfo2)
    for _ in 1:count
        variable_trip_hvp!(h, g, x, v)
    end
    return loop_cache_live_bytes(mallinfo2) - before
end

@testset "Forward over Reverse releases constant loop caches" begin
    if VERSION >= v"1.12-"
        # Julia 1.12 and 1.13 already reject this kernel before measuring memory:
        # the outer Forward pass differentiates an untyped memoryref byte offset.
        # Tracked with Enzyme todo 14jm2ow; both old and repaired libraries fail.
        # The native LLVM regression covers the free itself on every backend.
        @test_skip VERSION < v"1.12-"
    else
        for x in (Float64[], [-0.2], [0.1, 0.2, 0.5, 0.9], collect(range(0.1, 0.5; length = 72)))
            v = cos.(x)
            g, h = zero(x), zero(x)
            variable_trip_hvp!(h, g, x, v)
            plus, minus = zero(x), zero(x)
            step = 1.0e-5
            variable_trip_gradient!(plus, x .+ step .* v)
            variable_trip_gradient!(minus, x .- step .* v)
            @test all(isfinite, g)
            @test h ≈ (plus .- minus) ./ (2step) rtol = 1.0e-6 atol = 1.0e-7
        end

        libc = Sys.islinux() ? Libdl.dlopen_e("libc.so.6") : C_NULL
        mallinfo2 = libc == C_NULL ? C_NULL : Libdl.dlsym_e(libc, :mallinfo2)
        if mallinfo2 != C_NULL
            x = collect(range(0.1, 0.5; length = 72))
            v = cos.(x)
            g, h = zero(x), zero(x)
            # Compile the measurement itself before checking two independent batches.
            loop_cache_growth(mallinfo2, h, g, x, v, 1)
            for _ in 1:2
                # The bug retains about 1.2 MB per batch. Allow small unrelated
                # allocator bookkeeping, but not a cache per evaluation.
                @test loop_cache_growth(mallinfo2, h, g, x, v, 2000) < 65536
            end
        else
            @test_skip mallinfo2 != C_NULL
        end
        libc == C_NULL || Libdl.dlclose(libc)
    end
end
