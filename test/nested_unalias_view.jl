using Enzyme, Test

# Nested derivatives through `append!` and `copyto!` from a `SubArray`.
#
# Base's `copyto!(dest, doffs, src, soffs, n)` reads its source through
# `unalias(dest, src)`: the source itself, or a copy of it when it may alias `dest`.
# The augmented primal Enzyme builds for it loads a field through a phi of the
# `SubArray` argument and that copy. A nested derivative (Forward over Reverse) runs
# `nodecayed_phis!` on that augmented primal, which had no parent object for the
# argument, so on Julia 1.10 reaching the phi threw "Enzyme could not determine how to
# keep a pointer phi in this function rooted for the garbage collector". See
# test/passes.jl, "nodecayed_phis! loads through a phi of an argument and an object in
# its predecessors".

@noinline function nuv_merged(rv, n, i)
    m = sizehint!(Float64[], length(rv) + n)
    for k in 1:n
        push!(m, rv[k])
    end
    append!(m, @view rv[i:end])
    return m
end

# 4 * sum(abs2, x): every entry once.
nuv_append(x) = sum(abs2, nuv_merged(x .* 2.0, 1, 2))
nuv_append_hess(x) = fill(8.0, length(x))

# The source is a view of the destination, so `unalias` copies it.
function nuv_self(x)
    m = x .* 2.0
    append!(m, view(m, 1:2))
    return sum(abs2, m)
end
nuv_self_hess(x) = [i <= 2 ? 16.0 : 8.0 for i in eachindex(x)]

# `copyto!` with offsets.
function nuv_copyto(x)
    m = zeros(length(x) + 1)
    copyto!(m, 2, view(x, 2:length(x)), 1, length(x) - 1)
    m[1] = 3 * x[1]
    return sum(abs2, m)
end
nuv_copyto_hess(x) = [i == 1 ? 18.0 : 2.0 for i in eachindex(x)]

function nuv_grad(x, f)
    dx = zero(x)
    autodiff(Reverse, Const(f), Active, Duplicated(x, dx))
    return dx
end

nuv_dir(x, v, f) = only(autodiff(Forward, Const(f), Duplicated(x, v)))

const NUV_CASES = (
    (nuv_append, nuv_append_hess),
    (nuv_self, nuv_self_hess),
    (nuv_copyto, nuv_copyto_hess),
)

@testset "Nested derivatives through copyto! from a SubArray" begin
    for (f, hess) in NUV_CASES, n in (4, 7)
        @testset "$(nameof(f)), length $n" begin
            x = [0.3 * i - 1.1 for i in 1:n]
            v = [(-1.0)^i * (0.5 + 0.25 * i) for i in 1:n]
            w = [0.1 * i for i in 1:n]
            # The functions are quadratic, so the gradient is the diagonal Hessian
            # times x.
            hx = hess(x)
            @test nuv_grad(x, f) ≈ hx .* x
            @test nuv_dir(x, v, f) ≈ sum(hx .* x .* v)

            # Forward over Reverse.
            @test only(autodiff(Forward, Const(nuv_grad), Duplicated(x, v), Const(f))) ≈ hx .* v
            @test only(autodiff(set_runtime_activity(Forward), Const(nuv_grad), Duplicated(x, v), Const(f))) ≈ hx .* v
            hvs = autodiff(Forward, Const(nuv_grad), BatchDuplicated(x, (v, w)), Const(f))[1]
            @test hvs[1] ≈ hx .* v
            @test hvs[2] ≈ hx .* w

            # Reverse over Forward.
            hv = zero(x)
            autodiff(Reverse, Const(nuv_dir), Active, Duplicated(x, hv), Const(v), Const(f))
            @test hv ≈ hx .* v

            # The inputs are unchanged.
            @test x == [0.3 * i - 1.1 for i in 1:n]
            @test v == [(-1.0)^i * (0.5 + 0.25 * i) for i in 1:n]
        end
    end
end
