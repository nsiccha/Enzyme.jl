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

# A comprehension over an index vector builds a generator whose function captures
# an active array together with constant arrays. When that function, or the loop
# that collects the generator, is not inlined, the captures are stored into a
# struct passed to the call, which moves constant argument data into active
# memory as above. Julia 1.10 does not inline the function for a reshaped view,
# even when `collect` is inlined at the call site; Julia 1.12 does not inline
# `collect_to_with_first!` for any of these arrays.
generator_comprehension(b, x, idx) = [b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx)]
function generator_inline_collect(b, x, idx)
    values = (b[idx[i], 1] + b[idx[i], 2] * x[i] for i in eachindex(idx))
    return @inline collect(values)
end
generator_objective(reader, b, x, idx) = sum(reader(b, x, idx))

function generator_storages(::Type{T}) where {T}
    u = T[-0.2, 0.1, 0.3, -0.1]
    return (reshape(copy(u), 2, 2), reshape(view(copy(u), 1:4), 2, 2), view(reshape(copy(u), 2, 2), :, :))
end

@testset "Constant arrays captured by a generator with active data" begin
    @testset "$(nameof(reader)), $(typeof(b)), n = $n" for T in (Float32, Float64), b in generator_storages(T),
            reader in (generator_comprehension, generator_inline_collect), n in (0, 1, 4, 17)
        x = T[(i - 2) / 4 for i in 1:n]
        idx = [mod1(i, 2) for i in 1:n]
        want = zeros(T, 2, 2)
        for i in 1:n
            want[idx[i], 1] += 1
            want[idx[i], 2] += x[i]
        end
        inputs = deepcopy((b, x, idx))
        db = Enzyme.make_zero(b)
        autodiff(Reverse, generator_objective, Active, Const(reader), Duplicated(b, db), Const(x), Const(idx))
        @test collect(db) ≈ want
        @test (b, x, idx) == inputs
    end
end

# Copying a view of a constant array into a new vector goes through Base's
# `unalias(dest, src)`, which returns `src` itself unless the two may alias. That
# call is not inlined and its result is active, so the returned parent array, a
# field of the constant argument, needs a shadow. On Julia 1.13 GVN's load PRE
# loads that field in two predecessors and returns their phi.
@inline function view_copy_objective(w, s)
    t = zero(s)
    @inbounds for i in eachindex(w)
        t += w[i] * log(s)
    end
    return t
end
view_copy_convert(x, s, idx) = view_copy_objective(convert(Vector{Float64}, view(x, idx)), s)
view_copy_constructor(x, s, idx) = view_copy_objective(Vector{Float64}(view(x, idx)), s)
view_copy_copyto(x, s, idx) = view_copy_objective(copyto!(similar(idx, Float64), view(x, idx)), s)
view_copy_collect(x, s, idx) = view_copy_objective(collect(view(x, idx)), s)
view_copy_range(x, s, idx) = view_copy_objective(Vector{Float64}(view(x, 2:5)), s)
view_copy_column(A, s, idx) = view_copy_objective(Vector{Float64}(view(A, :, 2)), s)
view_copy_rows(A, s, idx) = view_copy_objective(Vector{Float64}(view(A, idx, 2)), s)
view_copy_first(sv, f, x, idx) = f(x, sv[1], idx)

@testset "Copies of views of constant arrays" begin
    x = [0.5, 1.5, 2.0, 0.25, 3.0, 1.25, 0.75]
    A = reshape(collect(1.0:21.0), 7, 3)
    idx = [1, 3, 5, 7]
    s = 2.5
    @testset "$(nameof(f))" for (f, data, copied) in (
            (view_copy_convert, x, x[idx]), (view_copy_constructor, x, x[idx]),
            (view_copy_copyto, x, x[idx]), (view_copy_collect, x, x[idx]),
            (view_copy_range, x, x[2:5]), (view_copy_column, A, A[:, 2]),
            (view_copy_rows, A, A[idx, 2]),
        )
        want = sum(copied) / s
        inputs = deepcopy((data, idx))
        @test autodiff(Reverse, f, Active, Const(data), Active(s), Const(idx))[1][2] ≈ want
        @test autodiff(Forward, f, Const(data), Duplicated(s, 1.0), Const(idx))[1] ≈ want
        bres = autodiff(Forward, f, Const(data), BatchDuplicated(s, (1.0, 2.0)), Const(idx))[1]
        @test bres[1] ≈ want
        @test bres[2] ≈ 2 * want

        d1 = [0.0]
        d2 = [0.0]
        autodiff(Reverse, view_copy_first, Active, BatchDuplicated([s], (d1, d2)), Const(f), Const(data), Const(idx))
        @test d1[1] ≈ want
        @test d2[1] ≈ want

        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active, Const{typeof(data)}, Active{Float64}, Const{typeof(idx)}
        )
        tape, prim, _ = fwd(Const(f), Const(data), Active(s), Const(idx))
        @test prim ≈ sum(copied) * log(s)
        @test rev(Const(f), Const(data), Active(s), Const(idx), 1.0, tape)[1][2] ≈ want
        @test (data, idx) == inputs
    end
end

# Before Julia 1.12, `unalias(dest, src)` returns the view `src` through its `sret`
# buffer with one `memcpy`, as does any non-inlined function returning a view. Enzyme
# copied the parent pointer of a constant source into the shadow unchanged, so when
# the copy met active data, Reverse added derivatives into the `Const` array and
# Forward took its values as the tangent.
function view_copy_into_rows(u, c)
    B = zeros(2, 2)
    copyto!(view(B, 1, :), u[[1, 3]])
    copyto!(view(B, 2, :), view(c, 2:3))
    return sum(B .* [1.0 2.0; 3.0 4.0])
end
function view_copy_into_vector(u, c)
    v = zeros(2)
    copyto!(v, view(c, 2:3))
    return sum(v .* u[1:2]) + u[3]^2
end
@noinline view_copy_pass(v) = v
view_copy_returned(u, c) = sum(view_copy_pass(view(c, 2:3)) .* u[1:2])

@testset "Copies of views of constant arrays into active data" begin
    u = [0.1i for i in 1:6]
    c = [0.3i for i in 1:6]
    @testset "$(nameof(f))" for (f, du_want, dc_want) in (
            (view_copy_into_rows, [1.0, 0, 2.0, 0, 0, 0], [0, 3.0, 4.0, 0, 0, 0]),
            (view_copy_into_vector, [c[2], c[3], 2u[3], 0, 0, 0], [0, u[1], u[2], 0, 0, 0]),
            (view_copy_returned, [c[2], c[3], 0, 0, 0, 0], [0, u[1], u[2], 0, 0, 0]),
        )
        c0 = copy(c)
        for mode in (Reverse, set_runtime_activity(Reverse))
            du = zero(u)
            autodiff(mode, f, Active, Duplicated(u, du), Const(c))
            @test du ≈ du_want
            @test c == c0
        end

        du = zero(u)
        dc = zero(c)
        autodiff(Reverse, f, Active, Duplicated(u, du), Duplicated(c, dc))
        @test du ≈ du_want
        @test dc ≈ dc_want

        @test autodiff(Forward, f, Duplicated(u, ones(6)), Const(c))[1] ≈ sum(du_want)
        bres = autodiff(Forward, f, BatchDuplicated(u, (ones(6), 2 .* ones(6))), Const(c))[1]
        @test bres[1] ≈ sum(du_want)
        @test bres[2] ≈ 2 * sum(du_want)

        du = zero(u)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(u)}, Const{typeof(c)}
        )
        tape, prim, _ = fwd(Const(f), Duplicated(u, du), Const(c))
        @test prim ≈ f(u, c0)
        rev(Const(f), Duplicated(u, du), Const(c), 1.0, tape)
        @test du ≈ du_want
        @test c == c0
    end
end

# Julia 1.13 constructs a reshaped Array by copying the source MemoryRef into
# a new header. A constant row reduction can make that header active when it
# shares the reshaped array with a broadcast over active data.
@static if VERSION >= v"1.13-"
    function const_reshape_weighted_rows(X, a)
        A = reshape(a, 1, length(a))
        return sum(sum((A .- 1) .* log.(X); dims = 2))
    end

    function const_reshape_shifted_rows(X, a)
        A = reshape(a, 1, length(a))
        offset = sum(A; dims = 2)
        return sum(offset .+ sum((A .- 1) .* log.(X); dims = 2))
    end

    @testset "Constant array reshape used by reduction and active broadcast" begin
        for groups in (2, 5), width in (2, 3),
                f in (const_reshape_weighted_rows, const_reshape_shifted_rows)
            X = reshape(collect(range(0.2, 0.8; length = groups * width)), groups, width)
            a = collect(range(1.5, 2.5; length = width))
            saved_X, saved_a = copy(X), copy(a)
            expected = (reshape(a, 1, width) .- 1) ./ X
            dX = zero(X)
            result = autodiff(ReverseWithPrimal, f, Active, Duplicated(X, dX), Const(a))
            @test last(result) ≈ f(X, a)
            @test dX ≈ expected
            @test X == saved_X && a == saved_a
        end

        X = [0.25 0.75; 0.6 0.4]
        a = [2.0, 3.0]
        d1, d2 = zero(X), zero(X)
        autodiff(
            Reverse, const_reshape_shifted_rows, Active,
            BatchDuplicated(X, (d1, d2)), Const(a),
        )
        @test d1 ≈ (reshape(a, 1, 2) .- 1) ./ X
        @test d2 ≈ d1
        @test X == [0.25 0.75; 0.6 0.4] && a == [2.0, 3.0]
    end

    # Enzyme selects a rule from the primal argument types alone, so active
    # arrays reach the same reshape rule and must share derivatives with the
    # reshaped result. A linear view of a vector reshapes with `Val(1)`, which
    # returns the vector itself; a linear view of a matrix reshapes it.
    active_reshape_view(x) = sum(abs2, view(x, 2:3))
    active_reshape_linear_view(x) = sum(abs2, view(reshape(x, 2, 2), 2:3))
    function active_reshape_write(x)
        y = copy(x)
        A = reshape(y, 2, 2)
        A .*= 2
        return sum(abs2, y)
    end
    active_reshape_mixed(x, a) = sum(reshape(x, 2, 2) .* reshape(a, 1, 2))
    function active_reshape_gradient(x)
        dx = zero(x)
        autodiff(Reverse, active_reshape_write, Active, Duplicated(x, dx))
        return dx
    end

    @testset "Active array reshape alongside the constant reshape rule" begin
        x0 = [1.0, 2.0, 3.0, 4.0]
        for (f, want) in (
                (active_reshape_view, [0.0, 4.0, 6.0, 0.0]),
                (active_reshape_linear_view, [0.0, 4.0, 6.0, 0.0]),
                (active_reshape_write, 8 .* x0),
            )
            for mode in (Reverse, set_runtime_activity(Reverse))
                x = copy(x0)
                dx = zero(x)
                autodiff(mode, f, Active, Duplicated(x, dx))
                @test dx ≈ want
                @test x == x0
            end
            x = copy(x0)
            d1, d2 = zero(x), zero(x)
            autodiff(Reverse, f, Active, BatchDuplicated(x, (d1, d2)))
            @test d1 ≈ want
            @test d2 ≈ want
        end

        x = copy(x0)
        dx = zero(x)
        a = [2.0, 3.0]
        res = autodiff(ReverseWithPrimal, active_reshape_mixed, Active, Duplicated(x, dx), Const(a))
        @test last(res) ≈ active_reshape_mixed(x0, a)
        @test dx ≈ [2.0, 2.0, 3.0, 3.0]
        @test x == x0 && a == [2.0, 3.0]

        # Forward mode, including forward over the reverse rule.
        v = [1.0, 0.0, 0.0, 1.0]
        @test autodiff(Forward, active_reshape_mixed, Duplicated(copy(x0), v), Const(a))[1] ≈ 5.0
        @test autodiff(Forward, active_reshape_mixed, Const(copy(x0)), Duplicated(copy(a), [1.0, 0.0]))[1] ≈ 3.0
        bres = autodiff(Forward, active_reshape_mixed, BatchDuplicated(copy(x0), (ones(4), v)), Const(a))[1]
        @test bres[1] ≈ 10.0
        @test bres[2] ≈ 5.0
        @test autodiff(Forward, active_reshape_gradient, Duplicated(copy(x0), v))[1] ≈ 8 .* v
    end
end

# Arrays that non-inlined calls (`findall`, `diff`) compute from constant data are
# stored into a NamedTuple, which is passed with an active scalar to a function
# that is not inlined. Activity analysis treats the `Vector{Float64}` that `diff`
# returns as active, so the NamedTuple is active, and the constant `Vector{Int}`
# that `findall` returns needs a shadow there. Its type comes from the callee's
# return type; an array of integers cannot hold derivatives, so it is its own
# shadow.
@inline returned_record(times) = (; idx = findall(>(0.0), times), dt = diff(times))
@noinline returned_record_weight(r, rate) = sum(exp.(-rate .* r.dt); init = 0.0) + length(r.idx)
function returned_record_total(rates, ts)
    t = 0.0
    for s in eachindex(rates)
        t += returned_record_weight(returned_record(ts[s]), rates[s])
    end
    return t
end

@testset "Arrays returned by calls into an active record" begin
    ts0 = [[0.0, 0.5, 1.5, 2.0], [0.0, 1.0], [0.0], [0.0, 0.25, 0.75]]
    rates0 = [0.3, 0.7, 1.1, 0.2]
    want = [-sum(diff(t) .* exp.(-r .* diff(t))) for (r, t) in zip(rates0, ts0)]

    for mode in (ReverseWithPrimal, set_runtime_activity(ReverseWithPrimal))
        ts = deepcopy(ts0)
        d = zero(rates0)
        res = autodiff(mode, returned_record_total, Active, Duplicated(copy(rates0), d), Const(ts))
        @test res[2] ≈ returned_record_total(rates0, ts0)
        @test d ≈ want
        @test ts == ts0
    end

    d1 = zero(rates0)
    d2 = zero(rates0)
    autodiff(Reverse, returned_record_total, Active, BatchDuplicated(copy(rates0), (d1, d2)), Const(deepcopy(ts0)))
    @test d1 ≈ want
    @test d2 ≈ want

    dr = collect(1.0:4.0)
    @test autodiff(Forward, returned_record_total, Duplicated(copy(rates0), dr), Const(deepcopy(ts0)))[1] ≈
        sum(want .* dr)
    bres = autodiff(
        Forward, returned_record_total, BatchDuplicated(copy(rates0), (ones(4), dr)), Const(deepcopy(ts0))
    )[1]
    @test bres[1] ≈ sum(want)
    @test bres[2] ≈ sum(want .* dr)

    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(returned_record_total)}, Active,
        Duplicated{Vector{Float64}}, Const{Vector{Vector{Float64}}},
    )
    x = Duplicated(copy(rates0), zero(rates0))
    ts = Const(deepcopy(ts0))
    tape, prim, _ = fwd(Const(returned_record_total), x, ts)
    @test prim ≈ returned_record_total(rates0, ts0)
    rev(Const(returned_record_total), x, ts, 1.0, tape)
    @test x.dval ≈ want
end
