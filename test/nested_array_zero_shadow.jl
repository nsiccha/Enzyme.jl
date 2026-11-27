using Enzyme, Test

# A branch selects constant data or data the function makes, and the data is an
# array whose elements are not floating-point numbers themselves: an array of
# arrays, or an array of tuples. Static Reverse then needs a shadow for the
# constant side and makes it at run time with `make_zero`.
#
# `make_zero` of such an array keeps a table of the shadows it has made, so that
# two references to one inner array get one shadow. The table is untyped. Reading
# a shadow back through a method's return type annotation converted the entry
# with a call dispatched at run time, and on Julia 1.10 the insertion was reached
# through `jl_invoke`. Both are inside code the derivative calls on every
# evaluation, so a caller that requires a type-stable derivative refused it.

returning(d, kind::Symbol) = [f for f in d.findings if !f.throw_only && f.kind === kind]
derivative_of(report, f) = only(d for d in report.derivatives if d.func === typeof(f))

function own_tables(x::Vector{Float64})
    own = Vector{Vector{Float64}}(undef, 2)
    own[1] = 2.0 .* x
    own[2] = x .+ 1.0
    return own
end

function sum_squares(t::Vector{Vector{Float64}})
    s = 0.0
    for v in t
        s += sum(abs2, v)
    end
    return s
end

function pick_nested(x::Vector{Float64}, data::Vector{Vector{Float64}}, flag::Bool)
    t = flag ? data : own_tables(x)
    return sum_squares(t) * x[1]
end

# The same selection once per group, as a generated model body makes it.
function pick_nested_groups(
        x::Vector{Float64}, tables::Vector{Vector{Vector{Float64}}}, flags::Vector{Bool},
    )
    total = 0.0
    for g in eachindex(tables)
        t = flags[g] ? tables[g] : own_tables(x)
        total += sum_squares(t) * x[1]
    end
    return total
end

function pick_tuples(x::Vector{Float64}, data::Vector{NTuple{2, Vector{Float64}}}, flag::Bool)
    own = Vector{NTuple{2, Vector{Float64}}}(undef, 1)
    own[1] = (2.0 .* x, x .+ 1.0)
    t = flag ? data : own
    s = 0.0
    for (a, b) in t
        s += sum(abs2, a) + sum(b)
    end
    return s * x[1]
end

# The derivatives by hand. `own_tables(x)` holds `2x` and `x .+ 1`.
own_squares(x) = sum(abs2, 2.0 .* x) + sum(abs2, x .+ 1.0)
own_squares_gradient(x) = 8.0 .* x .+ 2.0 .* (x .+ 1.0)
first_unit(x) = [i == 1 ? 1.0 : 0.0 for i in eachindex(x)]

function pick_nested_gradient(x, data, flag)
    flag && return sum_squares(data) .* first_unit(x)
    return own_squares(x) .* first_unit(x) .+ x[1] .* own_squares_gradient(x)
end

function reverse_report(f, x, rest...)
    dx = zero(x)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, f, Active, Duplicated(x, dx), map(Const, rest)...)
    end
    return dx, derivative_of(report, f)
end

@testset "A branch selects constant or active nested arrays" begin
    x = [0.5, -1.0, 2.0]
    data = [[1.0, 2.0], [3.0]]
    @testset "flag = $flag" for flag in (true, false)
        dx, derivative = reverse_report(pick_nested, x, data, flag)
        @test dx ≈ pick_nested_gradient(x, data, flag)
        @test isempty(returning(derivative, :dynamic_dispatch))
        @test isempty(returning(derivative, :enzyme_runtime_fallback))
        @test data == [[1.0, 2.0], [3.0]]
    end

    tuples = [([1.0], [2.0, 3.0])]
    dx, derivative = reverse_report(pick_tuples, x, tuples, true)
    @test dx ≈ 6.0 .* first_unit(x)
    @test isempty(returning(derivative, :dynamic_dispatch))
    dx, derivative = reverse_report(pick_tuples, x, tuples, false)
    @test dx ≈ (sum(abs2, 2.0 .* x) + sum(x .+ 1.0)) .* first_unit(x) .+ x[1] .* (8.0 .* x .+ 1.0)
    @test isempty(returning(derivative, :dynamic_dispatch))
end

# Both sides of the branch are arrays the function makes: an empty one where a
# group is skipped, else one triple per member. Nothing is stored into the empty
# one, so it is constant and its shadow is made at run time. An array of float
# triples is not an array of floating-point numbers, so `make_zero` took the path
# with the table. A generated model body that allocates each group's output with
# `similar` reaches this shape once the compiler splits off the empty case.
@inline function triples_or_empty(x::Vector{Float64}, n::Int, keep::Bool)
    w = Vector{NTuple{3, Float64}}(undef, 0)
    if keep
        w = Vector{NTuple{3, Float64}}(undef, n)
        for i in 1:n
            w[i] = (x[1] * i, x[2], x[3] + i)
        end
    end
    return w
end

function sum_triples(x::Vector{Float64}, n::Int, keep::Bool)
    total = 0.0
    for (a, b, c) in triples_or_empty(x, n, keep)
        total += a * b + c
    end
    return total
end

@testset "A branch selects an empty or a filled fresh array of float triples" begin
    x = [0.5, -1.0, 2.0]
    @testset "n = $n, keep = $keep" for n in (0, 3), keep in (false, true)
        dx, derivative = reverse_report(sum_triples, x, n, keep)
        steps = keep ? sum(1:n; init = 0) : 0
        @test dx ≈ [x[2] * steps, x[1] * steps, keep ? n : 0]
        @test isempty(returning(derivative, :dynamic_dispatch))
        @test isempty(returning(derivative, :enzyme_runtime_fallback))
    end
end

# On Julia 1.11 and later static activity cannot tell that `tables[g]` is constant
# data here and asks for runtime activity, with and without this repair.
@static if VERSION < v"1.11-"
    @testset "The selection inside a loop over groups" begin
        x = [0.5, -1.0, 2.0]
        data = [[1.0, 2.0], [3.0]]
        flags = [true, false, true]
        dx, derivative = reverse_report(pick_nested_groups, x, [data, data, data], flags)
        expected = sum(pick_nested_gradient(x, data, flag) for flag in flags)
        @test dx ≈ expected
        @test isempty(returning(derivative, :dynamic_dispatch))
        @test isempty(returning(derivative, :enzyme_runtime_fallback))
    end
end

@testset "make_zero of nested arrays" begin
    inner = [1.0, 2.0]
    nested = [inner, [3.0], inner]
    shadow = make_zero(nested)
    @test shadow isa Vector{Vector{Float64}}
    @test shadow == [[0.0, 0.0], [0.0], [0.0, 0.0]]
    @test shadow !== nested
    # Two references to one inner array keep one shadow.
    @test shadow[1] === shadow[3]
    @test shadow[1] !== inner
    @test nested == [[1.0, 2.0], [3.0], [1.0, 2.0]]

    deep = [[inner, [4.0]], [inner]]
    dshadow = make_zero(deep)
    @test dshadow isa Vector{Vector{Vector{Float64}}}
    @test dshadow == [[[0.0, 0.0], [0.0]], [[0.0, 0.0]]]
    @test dshadow[1][1] === dshadow[2][1]

    matrices = [ones(2, 2), ones(1, 3)]
    @test make_zero(matrices) == [zeros(2, 2), zeros(1, 3)]

    tuples = [(inner, [5.0]), (inner, inner)]
    tshadow = make_zero(tuples)
    @test tshadow isa Vector{NTuple{2, Vector{Float64}}}
    @test tshadow == [([0.0, 0.0], [0.0]), ([0.0, 0.0], [0.0, 0.0])]
    @test tshadow[1][1] === tshadow[2][1] === tshadow[2][2]

    # Inactive members stay as they are, active ones are zeroed.
    mixed = [(1.5, 2), (2.5, 3)]
    @test make_zero(mixed) == [(0.0, 2), (0.0, 3)]
    indices = [[1, 2], [3]]
    @test make_zero(indices) === indices

    # An unassigned slot stays unassigned.
    partial = Vector{Vector{Float64}}(undef, 2)
    partial[2] = [1.0]
    pshadow = make_zero(partial)
    @test !isassigned(pshadow, 1)
    @test pshadow[2] == [0.0]

    # A table the caller filled in is used as it is.
    seen = IdDict()
    given = [0.0, 0.0]
    seen[inner] = given
    reused = make_zero(typeof(nested), seen, nested)
    @test reused[1] === given
    @test reused[3] === given
    @test seen[nested] === reused

    # An entry of another type than its key is an error. It used to be converted,
    # which made a copy that nothing else referred to.
    seen = IdDict()
    seen[inner] = 1:2
    @test_throws TypeError make_zero(typeof(inner), seen, inner)
end
