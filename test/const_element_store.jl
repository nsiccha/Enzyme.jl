using Enzyme, Test

# A fresh container receives arrays loaded from a constant argument, and a consumer
# that is not inlined reads it into an active result. Activity analysis keeps the
# container active, because the consumer could write active data through it, so
# storing a constant element into it needs a shadow. Memory reached from a constant
# argument has none, and a derivative that flows into it is discarded, so a
# zero-filled copy of the element serves as its shadow.

struct ConstElementRecord
    xs::Vector{Float64}
end

# One allocation, one contiguous copy per element.
@noinline function concat_elements(per)
    n = 0
    for p in per
        n += length(p)
    end
    out = Vector{Float64}(undef, n)
    k = 1
    for p in per
        unsafe_copyto!(out, k, p, 1, length(p))
        k += length(p)
    end
    return out
end

function alias_elements(groups)
    per = similar(groups)
    for i in eachindex(groups)
        @inbounds per[i] = groups[i]
    end
    return per
end

@noinline function sum_nested_elements(per)
    s = 0.0
    for p in per, q in p, v in q
        s += v
    end
    return s
end

elements_vcat(r, g) = sum(reduce(vcat, alias_elements(g)) .* r)
elements_fold(r, g) = sum(concat_elements(alias_elements(g)) .* r)
elements_map(r, g) = sum(concat_elements(map(identity, g)) .* r)
elements_fields(r, g) = sum(concat_elements(map(s -> s.xs, ConstElementRecord.(g))) .* r)

# The container also holds an active array.
function elements_mixed(x, g)
    per = similar(g, length(g) + 1)
    @inbounds per[1] = x
    for i in eachindex(g)
        @inbounds per[i + 1] = g[i]
    end
    return sum(abs2, concat_elements(per))
end

# The elements are themselves arrays of arrays.
elements_nested(r, g) = sum_nested_elements(alias_elements(g)) * r[1]

const CONST_ELEMENT_GROUPS = [[0.5, 1.0, 1.0], [0.5], [1.0, 1.0]]
elements_global(r) = sum(concat_elements(alias_elements(CONST_ELEMENT_GROUPS)) .* r)

@testset "Fresh container of constant-argument elements" begin
    groups0 = [[0.5, 1.0, 1.0], [0.5], [1.0, 1.0]]
    rates0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    want = reduce(vcat, groups0)

    for f in (elements_vcat, elements_fold, elements_map, elements_fields)
        g = deepcopy(groups0)
        d = zero(rates0)
        res = autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(copy(rates0), d), Const(g))
        @test res[2] ≈ f(rates0, groups0)
        @test d == want
        @test g == groups0

        d1 = zero(rates0)
        d2 = zero(rates0)
        autodiff(Reverse, Const(f), Active, BatchDuplicated(copy(rates0), (d1, d2)), Const(deepcopy(groups0)))
        @test d1 == want
        @test d2 == want
    end

    for f in (elements_vcat, elements_fold)
        dr = collect(1.0:6.0)
        @test autodiff(Forward, Const(f), Duplicated(copy(rates0), dr), Const(deepcopy(groups0)))[1] ≈
            sum(want .* dr)
        bres = autodiff(
            Forward, Const(f), BatchDuplicated(copy(rates0), (ones(6), dr)), Const(deepcopy(groups0))
        )[1]
        @test bres[1] ≈ sum(want)
        @test bres[2] ≈ sum(want .* dr)
    end

    x = [2.0, 3.0]
    dx = zero(x)
    autodiff(Reverse, Const(elements_mixed), Active, Duplicated(x, dx), Const(deepcopy(groups0)))
    @test dx ≈ 2 .* x

    nested = [[[1.0, 2.0], [3.0]], [[4.0]]]
    dn = [0.0]
    autodiff(Reverse, Const(elements_nested), Active, Duplicated([2.0], dn), Const(deepcopy(nested)))
    @test dn ≈ [10.0]
    @test nested == [[[1.0, 2.0], [3.0]], [[4.0]]]

    # A closure's captured data is memory reached from the constant closure argument.
    g = deepcopy(groups0)
    captured = r -> sum(concat_elements(alias_elements(g)) .* r)
    dc = zero(rates0)
    autodiff(Reverse, Const(captured), Active, Duplicated(copy(rates0), dc))
    @test dc == want
    @test g == groups0

    # Not built yet: elements of a global array are constant memory too, but only
    # memory reached from a constant argument is recognized.
    @test_broken begin
        dg = zero(rates0)
        autodiff(Reverse, Const(elements_global), Active, Duplicated(copy(rates0), dg))
        dg == want
    end
end
