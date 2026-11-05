using Enzyme, LinearAlgebra, Libdl, Test
import EnzymeCore.EnzymeRules

# `mul!` reaches BLAS through `gemv!`, which Julia does not inline. The reverse
# pass of that call reads the matrix again, so the call's augmented pass copies
# the matrix whenever a write after the call may overwrite it. A store of a
# pointer, such as a buffer stored back into the Ref it came from, cannot
# overwrite a matrix of Float64. A store of a number can (an Int store may
# write floats reinterpreted as integers), so it keeps the copy.

const TLW_N, TLW_D = 2000, 20

@inline function tlw_lik(mu, y, sigma)
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        z = (y[i] - mu[i]) / sigma
        s += -0.5 * z * z - log(sigma)
    end
    return s
end

# The live native bytes, read by a custom reverse rule placed after the
# product: its reverse runs before the product's, while the product's tape
# (and any copy of the matrix in it) is still allocated. Julia's allocation
# counter does not see native caches; glibc's mallinfo2 does.
struct TLWMallInfo
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

const TLW_MALLINFO2 = Ref{Ptr{Cvoid}}(C_NULL)
const TLW_LIVE_IN_REVERSE = Ref(0)

function tlw_live_bytes()
    info = ccall(TLW_MALLINFO2[], TLWMallInfo, ())
    return Int(info.uordblks + info.hblkhd)
end

# `x + 0.0` is x, but LLVM cannot fold it (signed zeros), so the call's own
# result stays in use and the rule's reverse runs where the call was.
@noinline tlw_probe(x::Float64) = x + 0.0

function EnzymeRules.augmented_primal(
        config, ::Const{typeof(tlw_probe)}, ::Type{RT}, x::Active
    ) where {RT}
    primal = EnzymeRules.needs_primal(config) ? x.val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function tlw_record_live_bytes()
    if TLW_MALLINFO2[] != C_NULL
        TLW_LIVE_IN_REVERSE[] = tlw_live_bytes()
    end
    return nothing
end

function EnzymeRules.reverse(
        config, ::Const{typeof(tlw_probe)}, dret::Active, tape, x::Active
    )
    tlw_record_live_bytes()
    return (dret.val,)
end

function EnzymeRules.reverse(
        config, ::Const{typeof(tlw_probe)}, ::Type{<:Const}, tape, x::Active
    )
    tlw_record_live_bytes()
    return (0.0,)
end

# The reference: a freshly allocated product, no later write, no probe.
function tlw_plain(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    return tlw_lik(X * b, y, exp(u[2])) + u[1]
end

# The same with the probe.
function tlw_fresh(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    return tlw_probe(tlw_lik(X * b, y, exp(u[2]))) + u[1]
end

# The product into a buffer held in a Ref, nothing written after it.
function tlw_none(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    slot = caches[1]
    mul!(slot[], X, b)
    return tlw_probe(tlw_lik(slot[], y, exp(u[2]))) + u[1]
end

# The same buffer stored back into the Ref: a pointer store.
function tlw_storeback(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    slot = caches[1]
    slot[] = mul!(slot[], X, b)
    return tlw_probe(tlw_lik(slot[], y, exp(u[2]))) + u[1]
end

# An integer store: a counter in another Ref. The copy stays.
function tlw_counter(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    slot = caches[1]
    mul!(slot[], X, b)
    caches[2][] += 1
    return tlw_probe(tlw_lik(slot[], y, exp(u[2]))) + u[1]
end

# A Float64 store into another array, which may share the matrix's memory:
# the copy must stay.
function tlw_floatstore(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    slot = caches[1]
    mul!(slot[], X, b)
    caches[3][1] = 1.0
    return tlw_probe(tlw_lik(slot[], y, exp(u[2]))) + u[1]
end

# Later writes that do change what the product read.

# A pointer store replacing the buffer the product wrote: the reverse pass
# needs the old buffer.
function tlw_swap(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    slot = caches[1]
    mu = slot[]
    mul!(mu, X, b)
    slot[] = caches[4][]
    return tlw_probe(tlw_lik(mu, y, exp(u[2]))) + u[1]
end

# A pointer store replacing the matrix itself, held in a Ref.
function tlw_matrixswap(u, consts, caches)
    xref, y = consts[3], consts[2]
    b = view(u, 3:(TLW_D + 2))
    mu = caches[1][]
    mul!(mu, xref[], b)
    xref[] = consts[4]
    return tlw_probe(tlw_lik(mu, y, exp(u[2]))) + u[1]
end

# A Float64 store into the matrix's own memory, through a vector sharing it.
function tlw_overwrite(u, consts, caches)
    X, y = consts[1], consts[2]
    b = view(u, 3:(TLW_D + 2))
    mu = caches[1][]
    mul!(mu, X, b)
    consts[5][1] = 5.0
    return tlw_probe(tlw_lik(mu, y, exp(u[2]))) + u[1]
end

function tlw_setup()
    X = [sin(0.37i + 1.3j) for i in 1:TLW_N, j in 1:TLW_D]
    y = [cos(0.11i) for i in 1:TLW_N]
    other = [cos(0.29i + 0.7j) for i in 1:TLW_N, j in 1:TLW_D]
    consts = (X, y, Ref(X), other, vec(X))
    caches = (Ref(zeros(TLW_N)), Ref(0), zeros(2), Ref(zeros(TLW_N)))
    dcaches = (Ref(zeros(TLW_N)), Ref(0), zeros(2), Ref(zeros(TLW_N)))
    u = [0.1 * sin(k) for k in 1:(TLW_D + 2)]
    return X, consts, caches, dcaches, u
end

# The gradient of f at u, on fresh copies of the data each time (some of the
# functions above change it), and the native bytes live in its reverse pass.
function tlw_gradient(f)
    X, consts, caches, dcaches, u = tlw_setup()
    du = zero(u)
    # Free earlier calls' data now, so no collection during this call does.
    GC.gc()
    before = TLW_MALLINFO2[] == C_NULL ? 0 : tlw_live_bytes()
    autodiff(Reverse, f, Active, Duplicated(u, du), Const(consts), Duplicated(caches, dcaches))
    return du, TLW_LIVE_IN_REVERSE[] - before
end

@testset "Later pointer stores keep BLAS inputs uncopied" begin
    libc = Sys.islinux() ? Libdl.dlopen_e("libc.so.6") : C_NULL
    TLW_MALLINFO2[] = libc == C_NULL ? C_NULL : Libdl.dlsym_e(libc, :mallinfo2)
    reference, _ = tlw_gradient(tlw_plain)
    @test any(!iszero, reference[3:end])
    matrix_bytes = TLW_N * TLW_D * sizeof(Float64)

    @testset "gradient: $f" for f in (
            tlw_fresh, tlw_none, tlw_storeback, tlw_counter, tlw_floatstore,
            tlw_swap, tlw_matrixswap, tlw_overwrite,
        )
        tlw_gradient(f)
        du, _ = tlw_gradient(f)
        @test du ≈ reference rtol = 1.0e-12
    end

    if TLW_MALLINFO2[] != C_NULL
        @testset "matrix not copied: $f" for f in (
                tlw_none, tlw_storeback, tlw_swap, tlw_matrixswap,
            )
            tlw_gradient(f)
            _, live = tlw_gradient(f)
            @test live < matrix_bytes ÷ 2
        end
        # Stores of numbers keep the copy; tlw_overwrite needs it. These are
        # also the control that the measurement sees a copy.
        @testset "matrix copied: $f" for f in (
                tlw_counter, tlw_floatstore, tlw_overwrite,
            )
            tlw_gradient(f)
            _, live = tlw_gradient(f)
            @test live >= matrix_bytes
        end
    else
        @test_skip TLW_MALLINFO2[] != C_NULL
    end
end
