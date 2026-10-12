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
Base.:(==)(a::ConstElementRecord, b::ConstElementRecord) = a.xs == b.xs

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
# The argument holds structs stored inline, so each element is a field load.
elements_record_fields(r, records) = sum(concat_elements(map(s -> s.xs, records)) .* r)

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

    records0 = ConstElementRecord.(deepcopy(groups0))
    for (f, data0) in (
            (elements_vcat, groups0), (elements_fold, groups0), (elements_map, groups0),
            (elements_fields, groups0), (elements_record_fields, records0),
        )
        data = deepcopy(data0)
        d = zero(rates0)
        res = autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(copy(rates0), d), Const(data))
        @test res[2] ≈ f(rates0, data0)
        @test d == want
        @test data == data0

        d1 = zero(rates0)
        d2 = zero(rates0)
        autodiff(Reverse, Const(f), Active, BatchDuplicated(copy(rates0), (d1, d2)), Const(deepcopy(data0)))
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

# Constant argument data passed through dynamically dispatched calls, as arises
# downstream of a value that Julia infers as `Any`. The argument itself is stored
# into a box for the call or inserted into an aggregate that a call returns, and an
# element loaded from it is returned from a call. Each place needs a shadow for
# constant data, and a zero-filled copy serves, as for the elements above.
dyncall_uninferred(t, a) = Base.inferencebarrier(t .* exp(a))
dyncall_lpdf(y, m) = -0.5 * (y - m)^2

struct DynCallWrap{T}
    x::T
end
@noinline dyncall_wread(w::DynCallWrap, i) = w.x[i]
@noinline dyncall_eread(x, i) = x[i]
@noinline dyncall_ident(x) = x
@noinline dyncall_wwrite!(w::DynCallWrap, v) = (w.x[1] = v; nothing)

dyncall_inferred(a, t, y) = sum(dyncall_lpdf.(y, t .* exp(a)))
# A broadcast of the constant argument with an uninferred active array.
dyncall_broadcast(a, t, y) = sum(dyncall_lpdf.(y, dyncall_uninferred(t, a)))
# The constant argument in a struct boxed for a dynamic call.
function dyncall_wrapper(a, t, y)
    m = dyncall_uninferred(t, a)
    w = DynCallWrap(y)
    s = 0.0
    for i in eachindex(y)
        s += dyncall_lpdf(dyncall_wread(w, Base.inferencebarrier(i)), m[i])
    end
    return s
end
# A dynamic call returning an element of the constant argument.
function dyncall_element(a, ts, ys)
    s = 0.0
    for i in eachindex(ys)
        m = dyncall_uninferred(ts[i], a)
        yi = dyncall_eread(ys, Base.inferencebarrier(i))
        s += sum(dyncall_lpdf.(yi, m))
    end
    return s
end
# A dynamic call returning the constant argument itself.
dyncall_returned(a, y) = sum(dyncall_ident(Base.inferencebarrier(y)) .* a)
# An element of a constant tuple argument, which is passed as an aggregate.
function dyncall_tuple(a, cs)
    w = DynCallWrap(cs[2])
    s = 0.0
    for i in eachindex(cs[2])
        s += dyncall_lpdf(dyncall_wread(w, Base.inferencebarrier(i)), a * i)
    end
    return s
end
dyncall_first(x, f, args...) = f(x[1], args...)
# Active data written into the constant argument through the box.
function dyncall_storage(a, y)
    w = DynCallWrap(y)
    dyncall_wwrite!(Base.inferencebarrier(w), 2a)
    return dyncall_wread(Base.inferencebarrier(w), 1)
end

@testset "Constant argument data through dynamic calls" begin
    t = collect(range(0.5, 4.0; length = 7))
    y = 1.3 .* t .+ 0.1 .* sin.(1:7)
    ts = [t[1:3], t[4:7]]
    ys = [y[1:3], y[4:7]]
    want = autodiff(Reverse, dyncall_inferred, Active, Active(0.1), Const(t), Const(y))[1][1]

    for (f, args) in ((dyncall_broadcast, (t, y)), (dyncall_wrapper, (t, y)), (dyncall_element, (ts, ys)))
        data = deepcopy(args)
        @test autodiff(Reverse, f, Active, Active(0.1), map(Const, data)...)[1][1] ≈ want
        @test autodiff(set_runtime_activity(Reverse), f, Active, Active(0.1), map(Const, data)...)[1][1] ≈ want
        @test data == args

        d1 = [0.0]
        d2 = [0.0]
        autodiff(Reverse, dyncall_first, Active, BatchDuplicated([0.1], (d1, d2)), Const(f), map(Const, data)...)
        @test d1[1] ≈ want
        @test d2[1] ≈ want

        @test autodiff(Forward, f, Duplicated(0.1, 1.0), map(Const, data)...)[1] ≈ want
        bres = autodiff(Forward, f, BatchDuplicated(0.1, (1.0, 2.0)), map(Const, data)...)[1]
        @test bres[1] ≈ want
        @test bres[2] ≈ 2 * want
        @test data == args
    end

    y2 = [1.0, 2.0]
    @test autodiff(Reverse, dyncall_returned, Active, Active(0.3), Const(y2))[1][1] ≈ 3.0
    @test y2 == [1.0, 2.0]

    # sum_i (c_i - a i) i at a = 0.3 with c = [1, 2, 3].
    cs = ([1.0], [1.0, 2.0, 3.0])
    @test autodiff(Reverse, dyncall_tuple, Active, Active(0.3), Const(cs))[1][1] ≈ 9.8

    # A closure's captured data is a field of the constant closure argument.
    captured = let t = t, y = y
        a -> dyncall_wrapper(a, t, y)
    end
    @test autodiff(Reverse, Const(captured), Active, Active(0.1))[1][1] ≈ want

    # Active data written into constant memory is not differentiated (Enzyme FAQ,
    # "Activity of temporary storage"); runtime activity gives the same zero.
    @test autodiff(Reverse, dyncall_storage, Active, Active(0.3), Const([1.0, 2.0]))[1][1] == 0.0
    @test autodiff(set_runtime_activity(Reverse), dyncall_storage, Active, Active(0.3), Const([1.0, 2.0]))[1][1] == 0.0
end

# The constant data boxed in a loop: the copy is computed before the box is
# allocated, so a garbage collection that it triggers never sees the box with its
# field uninitialized.
function dyncall_loaded_box(a, t, yy)
    m = dyncall_uninferred(t, a)
    s = 0.0
    for i in eachindex(m)
        w = DynCallWrap(yy[1])
        s += dyncall_lpdf(dyncall_wread(w, Base.inferencebarrier(i)), m[i])
    end
    return s
end

@testset "Shadow copies computed outside object initialization" begin
    t = collect(range(0.5, 4.0; length = 512))
    y = 1.3 .* t .+ 0.1 .* sin.(1:512)
    want = autodiff(set_runtime_activity(Reverse), dyncall_wrapper, Active, Active(0.1), Const(t), Const(y))[1][1]
    want_loaded = autodiff(
        set_runtime_activity(Reverse), dyncall_loaded_box, Active, Active(0.1), Const(t), Const([y])
    )[1][1]
    for _ in 1:50
        @test autodiff(Reverse, dyncall_wrapper, Active, Active(0.1), Const(t), Const(y))[1][1] ≈ want
        @test autodiff(Reverse, dyncall_loaded_box, Active, Active(0.1), Const(t), Const([y]))[1][1] ≈
            want_loaded
    end
end
