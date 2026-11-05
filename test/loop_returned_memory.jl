using Enzyme, LinearAlgebra, Test

# A loop that replaces an array with a freshly allocated, scaled copy, followed
# by a matrix product that reads the result. The non-inlined `/` returns newly
# initialized storage; Reverse must keep the shadow of the array it reads.
# With dogfood libEnzyme `c1743bf9` this segfaulted in
# `overload_broadcast_getindex` as soon as the loop body ran once.
function halving_product(x)
    magnitude = maximum(abs, x)
    scaled = x
    while magnitude > 0.5
        scaled = scaled / 2
        magnitude /= 2
    end
    return Matrix{Float64}(I, size(x)...) * scaled
end

function halvings(x)
    magnitude = maximum(abs, x)
    count = 0
    while magnitude > 0.5
        magnitude /= 2
        count += 1
    end
    return count
end

# Scaling and squaring a truncated Taylor series, as StanBlocks' Julia
# `matrix_exp` does (it avoids LAPACK balancing, which has no Enzyme rule).
function scaled_squared_exp(x)
    n = size(x, 1)
    magnitude = zero(eltype(x))
    for i in 1:n
        row = zero(eltype(x))
        for j in 1:n
            row += abs(x[i, j])
        end
        magnitude = max(magnitude, row)
    end
    scaled = x
    squarings = 0
    while magnitude > 0.5
        scaled = scaled / 2
        magnitude /= 2
        squarings += 1
    end
    result = Matrix{Float64}(I, n, n)
    term = result
    for k in 1:24
        term = term * scaled / k
        result = result + term
    end
    for _ in 1:squarings
        result = result * result
    end
    return result
end

# Gradient of sum(exp(X)): the Fréchet derivative of exp at X' applied to a
# matrix of ones, read from the upper right block of a block exponential.
function exp_sum_gradient(x)
    n = size(x, 1)
    block = [x' ones(n, n); zeros(n, n) x']
    return exp(block)[1:n, (n + 1):(2n)]
end

test_matrix(n, scale) = [i == j ? -scale : scale * (i - j) / (10n) for i in 1:n, j in 1:n]

@testset "Reverse through a loop of fresh scaled copies before a product" begin
    for n in (1, 2, 3), scale in (0.1, 1.0, 10.0)
        x = test_matrix(n, scale)
        original = copy(x)
        dx = zeros(n, n)
        _, primal = autodiff(
            ReverseWithPrimal, sum ∘ halving_product, Active, Duplicated(x, dx)
        )
        @test primal ≈ sum(x) / 2.0^halvings(x)
        @test dx ≈ fill(2.0^-halvings(x), n, n)
        @test x == original
    end
end

@testset "Reverse through a scaling-and-squaring matrix exponential" begin
    for n in (1, 2, 3), scale in (0.1, 1.0, 10.0)
        x = test_matrix(n, scale)
        original = copy(x)
        dx = zeros(n, n)
        _, primal = autodiff(
            ReverseWithPrimal, sum ∘ scaled_squared_exp, Active, Duplicated(x, dx)
        )
        @test primal ≈ sum(exp(x)) rtol = 1.0e-10
        @test dx ≈ exp_sum_gradient(x) rtol = 1.0e-8
        @test x == original
    end
end
