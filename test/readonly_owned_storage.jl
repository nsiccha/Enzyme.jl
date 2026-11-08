using Enzyme, Test

# A fresh `Matrix{Float64}(undef, ...)` filled by a function Julia does not
# inline, then read by another. Before Julia 1.11 the filler receives the array
# header, which it only reads: it loads the data pointer from the header and
# writes the elements through it. LLVM therefore marks the header `readonly`.
# libEnzyme took that operand attribute to mean that nothing wrote the
# allocation, recreated it uninitialized in the reverse pass, and the reverse
# of the reader read that memory: the gradient of everything upstream of the
# filled array came back as whatever the allocator handed out (snag
# rkppl-mvnormal-s-1df3d9b3, an RKPPL MvNormal slice prior).

@noinline function owned_lower_solve!(
        Z::Matrix{Float64}, X::Matrix{Float64},
        F::Matrix{Float64}
    )
    groups, k = size(X)
    groups == 0 && return Z
    size(F) == (k, k) || throw(DimensionMismatch("F"))
    @inbounds for g in 1:groups, i in 1:k
        s = X[g, i]
        for j in 1:(i - 1)
            s -= F[i, j] * Z[j, g]
        end
        Z[i, g] = s / F[i, i]
    end
    return Z
end

@noinline function owned_sum_squares(Z::Matrix{Float64}, c::Float64)
    k, groups = size(Z)
    groups == 0 && return 0.0
    squares = 0.0
    @inbounds for g in 1:groups, i in 1:k
        z = Z[i, g]
        squares += z * z
    end
    return -0.5squares - groups * c
end

@noinline function owned_density(X::Matrix{Float64}, F::Matrix{Float64})
    groups, k = size(X)
    Z = owned_lower_solve!(Matrix{Float64}(undef, k, groups), X, F)
    return owned_sum_squares(Z, log(F[1, 1]))
end

# Each consumer below keeps the derivative of owned_density from being computed
# inside the reverse pass, so it is split into an augmented forward pass and a
# reverse pass, and the reverse pass needs the filled Z.
@noinline owned_scale(lp::Float64, u::Vector{Float64}) = lp * u[end]

function owned_through_call(u::Vector{Float64}, F::Matrix{Float64}, n::Int)
    X = reshape(u[1:(2n)], n, 2) .- 0.0
    return owned_scale(owned_density(X, F), u)
end

function owned_through_branch(u::Vector{Float64}, F::Matrix{Float64}, n::Int)
    X = reshape(u[1:(2n)], n, 2) .- 0.0
    lp = owned_density(X, F)
    return u[end] > 0 ? lp * u[end] : lp
end

# The gradient of owned_density(X, F) * u[end] in closed form. Row g of X
# solves to z1 = x1 / F11 and z2 = (x2 - F21 z1) / F22, and the density is
# -0.5 Σ (z1² + z2²) - n log F11.
function owned_expected(u, F, n)
    X = reshape(u[1:(2n)], n, 2)
    dX = zeros(n, 2)
    density = -n * log(F[1, 1])
    for g in 1:n
        z1 = X[g, 1] / F[1, 1]
        z2 = (X[g, 2] - F[2, 1] * z1) / F[2, 2]
        density -= 0.5 * (z1^2 + z2^2)
        dX[g, 1] = -z1 / F[1, 1] + z2 * F[2, 1] / (F[2, 2] * F[1, 1])
        dX[g, 2] = -z2 / F[2, 2]
    end
    return vcat(vec(dX) .* u[end], density)
end

# Hand freed, nonzero memory to the next allocations, so that an uninitialized
# array read by mistake holds values that cannot pass for the right ones.
function owned_dirty_allocator(n)
    for _ in 1:200
        fill(1.0e300, 2, n)
    end
    GC.gc(false)
    return nothing
end

# The smallest shape: one vector, filled and summed by functions Julia does not
# inline, beside a loop that allocates and fills one per iteration.
@noinline function owned_fill_scaled!(z::Vector{Float64}, x::Vector{Float64}, k::Int)
    @inbounds for i in eachindex(z, x)
        z[i] = x[i] * k
    end
    return z
end

@noinline function owned_vector_sum_squares(z::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(z)
        s += z[i] * z[i]
    end
    return s
end

@noinline owned_filled_vector(x::Vector{Float64}) =
    owned_vector_sum_squares(owned_fill_scaled!(Vector{Float64}(undef, length(x)), x, 3))

owned_vector_through_call(x::Vector{Float64}) =
    owned_scale(owned_filled_vector(x), x)

function owned_vector_loop(x::Vector{Float64}, n::Int)
    t = 0.0
    for k in 1:n
        z = Vector{Float64}(undef, length(x))
        owned_fill_scaled!(z, x, k)
        t += owned_vector_sum_squares(z)
    end
    return t
end

@testset "Readonly operands that write a fresh vector's storage" begin
    for m in (1, 4, 16)
        x = [0.1 * i + 0.05 for i in 1:m]
        value = 9 * sum(abs2, x)
        expected = 18 .* x .* x[end]
        expected[end] += value
        for _ in 1:3
            owned_dirty_allocator(m)
            dx = zero(x)
            autodiff(Reverse, owned_vector_through_call, Active, Duplicated(copy(x), dx))
            @test dx ≈ expected rtol = 1.0e-12
        end
        for n in (1, 3)
            owned_dirty_allocator(m)
            dx = zero(x)
            autodiff(Reverse, owned_vector_loop, Active, Duplicated(copy(x), dx), Const(n))
            @test dx ≈ 2 * sum(k^2 for k in 1:n) .* x rtol = 1.0e-12
        end
    end
end

@testset "Readonly operands that write a fresh array's storage" begin
    F = [1.0 0.0; 0.5 sqrt(1.75)]
    for n in (1, 2, 5), f in (owned_through_call, owned_through_branch)
        u = vcat([0.1 * i - 0.35 for i in 1:(2n)], 0.4)
        expected = owned_expected(u, F, n)
        @test f(u, F, n) ≈ expected[end] * u[end]
        for _ in 1:3
            owned_dirty_allocator(n)
            du = zero(u)
            v = copy(u)
            autodiff(Reverse, f, Active, Duplicated(v, du), Const(F), Const(n))
            @test du ≈ expected rtol = 1.0e-12
            @test v == u
        end
        owned_dirty_allocator(n)
        du = zero(u)
        _, primal = autodiff(
            ReverseWithPrimal, f, Active, Duplicated(copy(u), du),
            Const(F), Const(n)
        )
        @test primal ≈ expected[end] * u[end]
        @test du ≈ expected rtol = 1.0e-12
    end
end
