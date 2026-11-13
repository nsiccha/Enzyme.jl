using Enzyme, LogExpFunctions, Test

@noinline bound_fold_logsum(x, y) = x >= y ? (isinf(x) & (x == y) ? x : x + log1pexp(y - x)) : y + log1pexp(x - y)

function bound_objective(x, data, scratch, ::Val{C}, ::Val{L}, ::Val{W}) where {C, L, W}
    events, previous, never, t, components = data
    base = copy(view(x, 1:t))
    weights = view(x, (length(x) - components + 1):length(x))
    logweights = log.(weights)
    total = 0.0
    for i in eachindex(events)
        bc = if C
            Base.Broadcast.broadcasted(+, Base.Broadcast.broadcasted(+, base, x[t + 2] * x[t + 2 + i]), Base.Broadcast.broadcasted(*, x[t + 1], previous[i]))
        else
            Base.Broadcast.broadcasted(+, base, x[t + 2] * x[t + 2 + i])
        end
        eta = L ? Base.Broadcast.instantiate(bc) : Base.materialize(bc)
        rows = length(axes(eta, 1))
        rows == length(events[i]) == length(never[i]) || throw(DimensionMismatch("rows"))
        mix = -Inf
        for k in eachindex(weights)
            score = 0.0
            for j in 1:rows
                a = k == 1 ? eta[j] : never[i][j]
                score -= log1pexp(ifelse(events[i][j], -a, a))
            end
            mix = bound_fold_logsum(mix, logweights[k] + score)
        end
        total += mix
    end
    if W
        for j in eachindex(scratch)
            scratch[j] = x[j]^2
        end
        total += sum(scratch)
    else
        for j in eachindex(scratch)
            total += x[j]^2
        end
    end
    return total
end

# Every component overwrites the values the next component reads. Distinct
# cache slots are necessary even though the row count is shared.
function mutating_components(x, lengths, components)
    total = 0.0
    for rows in lengths, k in 1:components, j in 1:rows
        x[j] = x[j]^2
        total += x[j]
    end
    return total
end

function mutating_components_reference(x, lengths, components)
    values = copy(x)
    derivative = ones(length(x))
    gradient = zero(x)
    total = 0.0
    for rows in lengths, k in 1:components, j in 1:rows
        derivative[j] *= 2 * values[j]
        values[j] *= values[j]
        gradient[j] += derivative[j]
        total += values[j]
    end
    return total, gradient, values
end

# The reverse must also retain the original extent when a later store changes
# the bound itself, rather than rereading it while indexing the shared tape.
function mutating_component_bounds(x, lengths, components)
    total = 0.0
    for i in eachindex(lengths)
        rows = lengths[i]
        for k in 1:components, j in 1:rows
            x[j] = x[j]^2
            total += x[j]
        end
        lengths[i] = 0
    end
    return total
end

@testset "Distinct entries across mutating components" begin
    for lengths in (Int[], [0, 0], [3, 1, 0, 4]), components in (0, 1, 2, 3)
        x = [0.2, -0.4, 0.6, -0.8]
        value, expected, final = mutating_components_reference(x, lengths, components)
        dx = zero(x)
        result = autodiff(
            ReverseWithPrimal, mutating_components, Active,
            Duplicated(x, dx), Const(lengths), Const(components)
        )
        @test result[2] ≈ value
        @test dx ≈ expected
        @test x == final

        x = [0.2, -0.4, 0.6, -0.8]
        dx = zero(x)
        mutable_lengths = copy(lengths)
        result = autodiff(
            ReverseWithPrimal, mutating_component_bounds, Active,
            Duplicated(x, dx), Const(mutable_lengths), Const(components)
        )
        @test result[2] ≈ value
        @test dx ≈ expected
        @test x == final
        @test all(iszero, mutable_lengths)
    end
end

function bound_grad!(dx, x, data, scratch, dscratch, c, l, w)
    fill!(dx, 0); fill!(dscratch, 0)
    Enzyme.autodiff(Reverse, bound_objective, Active, Duplicated(x, dx), Const(data), Duplicated(scratch, dscratch), Const(c), Const(l), Const(w))
    return dx
end


# A runtime broadcast extent can vary with the event while remaining the same
# across its component loop. Native cache chunks should cover all components
# of one event, and a merged index with identical arithmetic on both arms can
# be recomputed. The native lit tests check the generated tape placement;
# these cases exercise broadcast shape, nonlinear values, and mutation.
function shape_input(t, lengths, components, alias_scratch)
    n = length(lengths)
    previous = [Float64[sin(i + j) for j in 1:r] for (i, r) in enumerate(lengths)]
    rows = [t == 1 ? r : t for r in lengths]
    events = [Bool[iszero((i * j) % 7) for j in 1:r] for (i, r) in enumerate(rows)]
    never = [Float64[cos(i + j) for j in 1:r] for (i, r) in enumerate(rows)]
    x = vcat(sin.(1:(n + t + 2)), fill(1.0 / components, components))
    scratch = cos.(1:t)
    if alias_scratch
        previous[1] = scratch
        never[end] = scratch
    end
    return x, (events, previous, never, t, components), scratch
end

function shape_gradient(input, lazy)
    x, data, scratch = deepcopy(input)
    dx, ds = zero(x), zero(scratch)
    bound_grad!(dx, x, data, scratch, ds, Val(true), Val(lazy), Val(true))
    return dx, scratch, ds
end

function shape_value(input, direction, step)
    x, data, scratch = deepcopy(input)
    x .+= step .* direction
    return bound_objective(x, data, scratch, Val(true), Val(true), Val(true))
end

function shape_split_gradient(input)
    x, data, scratch = deepcopy(input)
    args = (
        Duplicated(x, zero(x)), Const(data), Duplicated(scratch, zero(scratch)),
        Const(Val(true)), Const(Val(true)), Const(Val(true)),
    )
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(bound_objective)}, Active,
        map(typeof, args)...,
    )
    tape, value, _ = fwd(Const(bound_objective), args...)
    rev(Const(bound_objective), args..., 1.0, tape)
    return value, args[1].dval, scratch, args[3].dval
end

function component_gradient(x, lengths, components)
    values, dx = copy(x), zero(x)
    autodiff(
        Reverse, mutating_components, Active,
        Duplicated(values, dx), Const(lengths), Const(components),
    )
    return dx
end

component_gradient_sum(x, lengths, components) = sum(component_gradient(x, lengths, components))

@testset "Runtime broadcast cache shapes" begin
    cases = (
        (5, [5, 1, 5, 1, 1, 5], false),
        (1, [0, 1, 3, 7, 2, 0, 1, 9], false),
        (3, Int[], false),
        (0, [0, 1, 0, 1], false),
        (5, [5, 1, 5, 5], true),
    )
    for (t, lengths, alias_scratch) in cases, components in (1, 2, 3)
        input = shape_input(t, lengths, components, alias_scratch)
        g, scratch, ds = shape_gradient(input, true)
        ref, scratch_ref, ds_ref = shape_gradient(input, false)
        @test g ≈ ref rtol = 1.0e-12 atol = 1.0e-12
        @test scratch == scratch_ref
        @test ds ≈ ds_ref rtol = 1.0e-12 atol = 1.0e-12
        direction = cos.(eachindex(g))
        fd = (shape_value(input, direction, 1.0e-5) - shape_value(input, direction, -1.0e-5)) / 2.0e-5
        @test sum(g .* direction) ≈ fd rtol = 1.0e-6 atol = 1.0e-6
    end
end

@testset "Split reverse across ragged buffer growth" begin
    # Large jumps cross multiple growth boundaries; empty events consume no
    # values. Aliases require both the old values and the saved event offsets.
    cases = (
        (1, [0, 1, 0, 33, 1, 257, 0, 2], false),
        (5, [5, 1, 5, 1, 1, 5], true),
        (1, Int[], false),
        (0, [0, 1, 0, 1], false),
    )
    for (t, lengths, alias_scratch) in cases, components in (1, 2, 3)
        input = shape_input(t, lengths, components, alias_scratch)
        expected, scratch_ref, ds_ref = shape_gradient(input, false)
        value, got, scratch, ds = shape_split_gradient(input)
        @test value ≈ shape_value(input, zero(input[1]), 0.0)
        @test got ≈ expected rtol = 1.0e-12 atol = 1.0e-12
        @test scratch == scratch_ref
        @test ds ≈ ds_ref rtol = 1.0e-12 atol = 1.0e-12
    end
end

@testset "Ragged cache higher derivatives" begin
    # Growth must preserve the old storage while differentiating its copies.
    # A raw realloc can pass first derivatives but invalidate the second reverse.
    for T in (Float64, Float32), lengths in ([1, 0, 2, 4], [0, 0]), components in (0, 1, 2)
        x = T[0.2, -0.4, 0.6, -0.8]
        direction = T.(cos.(1:4))
        tolerance = T === Float32 ? 1.0e-5 : 1.0e-6
        h = 1.0e-5
        up = mutating_components_reference(x .+ h .* direction, lengths, components)[2]
        down = mutating_components_reference(x .- h .* direction, lengths, components)[2]
        expected = (up .- down) ./ (2h)
        got = only(
            autodiff(
                Forward, component_gradient, Duplicated(x, direction),
                Const(lengths), Const(components),
            )
        )
        @test got ≈ expected rtol = tolerance atol = tolerance

        up = mutating_components_reference(x .+ h, lengths, components)[2]
        down = mutating_components_reference(x .- h, lengths, components)[2]
        expected = (up .- down) ./ (2h)
        got = zero(x)
        autodiff(
            Reverse, component_gradient_sum, Active,
            Duplicated(x, got), Const(lengths), Const(components),
        )
        @test got ≈ expected rtol = tolerance atol = tolerance
    end
end

if VERSION >= v"1.13"
    @testset "Bound broadcast native tape sites" begin
        x, data, scratch = shape_input(5, fill(5, 6), 2, false)
        argtypes = Tuple{
            Duplicated{typeof(x)}, Const{typeof(data)},
            Duplicated{typeof(scratch)}, Const{Val{true}}, Const{Val{true}}, Const{Val{true}},
        }
        io = IOBuffer()
        Enzyme.Compiler.enzyme_code_llvm(
            io, bound_objective, Active, argtypes;
            raw = true, dump_module = true, debuginfo = :none
        )
        ir = String(take!(io))
        # LLVM 20 previously emitted 16 sites, including an integer index
        # cache. Recompute that index and remove its outer pointer table.
        @test count(r"call[^\n]*@malloc\(", ir) <= 14
    end
end
