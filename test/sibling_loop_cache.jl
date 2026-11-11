using Enzyme, Test

@inline function sibling_cache_row_product(A, b, row)
    axes(A, 2) == axes(b, 1) || throw(DimensionMismatch())
    total = 0.0
    for j in axes(A, 2)
        total += A[row, j] * b[j]
    end
    return total
end

# The later write forces the row coefficients to be cached. Both products
# read the same row, but each reverse loop must use its own iteration index.
function sibling_cache_pair(bs, A, rows, overwrite_all)
    total = 0.0
    for row in rows
        a = sibling_cache_row_product(A, bs[1], row)
        b = sibling_cache_row_product(A, bs[2], row)
        total += sin(a) + cos(b)
    end
    if overwrite_all
        fill!(A, 0.0)
    else
        A[1, 1] = 0.0
    end
    return total
end

function sibling_cache_single(bs, A, row, overwrite_all)
    a = sibling_cache_row_product(A, bs[1], row)
    b = sibling_cache_row_product(A, bs[2], row)
    if overwrite_all
        fill!(A, 0.0)
    else
        A[row, 1] = 0.0
    end
    return sin(a) + cos(b)
end

@testset "Sibling loop cached loads" begin
    for width in (1, 3, 5), overwrite_all in (false, true), nested in (false, true)
        A = [sin(0.7i - 1.1j) for i in 1:4, j in 1:width]
        bs = ([0.2j - 0.4 for j in 1:width], [0.1j - 0.8 for j in 1:width])
        rows = nested ? [3, 1, 4] : [3]
        d = map(zero, bs)
        expected = map(zero, bs)
        expected_primal = 0.0
        for row in rows
            a = sum(A[row, :] .* bs[1])
            b = sum(A[row, :] .* bs[2])
            expected_primal += sin(a) + cos(b)
            expected[1] .+= cos(a) .* A[row, :]
            expected[2] .-= sin(b) .* A[row, :]
        end
        f = nested ? sibling_cache_pair : sibling_cache_single
        context = nested ? rows : only(rows)
        _, primal = Enzyme.autodiff(
            ReverseWithPrimal, Const(f), Active,
            Duplicated(bs, d), Const(copy(A)), Const(context), Const(overwrite_all)
        )
        @test primal ≈ expected_primal
        @test d[1] ≈ expected[1]
        @test d[2] ≈ expected[2]
    end
end
