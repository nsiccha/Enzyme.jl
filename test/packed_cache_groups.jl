using Enzyme, Test

# A combined Reverse derivative stores the packed ragged caches of one element
# type and loop nest as records of one growing buffer with one offset table
# (repair `packed_cache_groups`). The same code compiled with the option off
# keeps one table and one buffer per cache, so each case compares the grouped
# derivative with that one, and with central differences. The two can differ
# in the last bit: LLVM may fuse or order the adjoint arithmetic differently
# when the cached values sit at other addresses.

const GROUPED = Enzyme.has_dogfood_repair(:packed_cache_groups)

function with_ungrouped(f)
    Enzyme.API.EnzymeSetCLBool(:EnzymeGroupPackedCaches, false)
    try
        return f()
    finally
        Enzyme.API.EnzymeSetCLBool(:EnzymeGroupPackedCaches, true)
    end
end

# Per group: a fresh propagator table filled in one inner loop, then a walk
# over the group's events with a three-state carry, a three-way branch and a
# running total in a second inner loop.
for name in (:event_walk, :event_walk_ungrouped)
    @eval function $name(x, steps, kinds, amounts, dts, y)
        total = 0.0
        k1 = exp(x[1]); v = exp(x[2]); ka = exp(x[3]); q = exp(x[4])
        @inbounds for g in eachindex(kinds)
            dt = dts[g]; st = steps[g]; ty = kinds[g]; am = amounts[g]; yg = y[g]
            P = Vector{NTuple{9, Float64}}(undef, length(dt))
            for i in eachindex(dt)
                a = exp(-k1 * dt[i]); b = exp(-ka * dt[i]); c = exp(-q * dt[i])
                P[i] = (b, ka * (a - b), q * (c - a), 0.0, a, q * (a - c), 0.0, q * (c - a) / 2, c)
            end
            s = (0.0, 0.0, 0.0); given = 0.0
            out = Vector{Float64}(undef, length(ty))
            for j in eachindex(ty)
                p = P[st[j]]
                m = (
                    p[1] * s[1] + p[4] * s[2] + p[7] * s[3],
                    p[2] * s[1] + p[5] * s[2] + p[8] * s[3],
                    p[3] * s[1] + p[6] * s[2] + p[9] * s[3],
                )
                t = ty[j]
                s = t == 2 ? (m[1] + am[j], m[2], m[3]) : t == 3 ? (m[1], m[2] + am[j], m[3]) : m
                given = t == 1 ? given : given + am[j]
                out[j] = m[2] / v + 1.0e-3 * given
            end
            for j in eachindex(out)
                total += (out[j] - yg[j])^2
            end
        end
        return total
    end
end

function event_data(events, tables)
    G = length(events)
    kinds = [Int[1 + (g + 2j) % 3 for j in 1:events[g]] for g in 1:G]
    dts = [Float64[1.0 + 0.5 * i + 0.1 * g for i in 1:tables[g]] for g in 1:G]
    steps = [Int[1 + (g * j) % max(tables[g], 1) for j in 1:events[g]] for g in 1:G]
    amounts = [Float64[10.0 + 3.0 * sin(g + j) for j in 1:events[g]] for g in 1:G]
    y = [Float64[0.5 + 0.25 * cos(g * j) for j in 1:events[g]] for g in 1:G]
    return (steps, kinds, amounts, dts, y)
end

function walk_gradient(f, x, data)
    dx = zero(x)
    result = autodiff(ReverseWithPrimal, f, Active, Duplicated(copy(x), dx), map(Const, data)...)
    return result[2], dx
end

function walk_split_gradient(f, x, data)
    args = (Duplicated(copy(x), zero(x)), map(Const, data)...)
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, map(typeof, args)...)
    tape, value, _ = fwd(Const(f), args...)
    rev(Const(f), args..., 1.0, tape)
    return value, args[1].dval
end

# Groups of differing sizes, empty groups, no groups, and a jump that crosses
# several capacity doublings at once.
const EVENT_CASES = (
    ([8, 8], [3, 4]),
    ([1, 0, 5, 33, 2], [1, 1, 2, 4, 3]),
    (Int[], Int[]),
    ([0, 0, 0], [0, 2, 0]),
    ([2, 257, 1, 0, 9], [2, 3, 1, 1, 5]),
    (fill(8, 40), fill(4, 40)),
)

@testset "Packed cache groups: event walk" begin
    x = [log(0.1), log(10.0), log(1.0), log(0.3)]
    direction = [0.3, -0.7, 0.5, 0.2]
    for (events, tables) in EVENT_CASES
        data = event_data(events, tables)
        value, g = walk_gradient(event_walk, x, data)
        @test value == event_walk(x, data...)
        if GROUPED
            ref_value, ref = with_ungrouped(() -> walk_gradient(event_walk_ungrouped, x, data))
            @test value == ref_value
            @test g ≈ ref rtol = 1.0e-12 atol = 1.0e-12
        end
        h = 1.0e-6
        fd = (event_walk(x .+ h .* direction, data...) - event_walk(x .- h .* direction, data...)) / 2h
        @test sum(g .* direction) ≈ fd rtol = 1.0e-6 atol = 1.0e-8
        # Split mode passes caches on a tape and keeps one table per cache.
        split_value, split = walk_split_gradient(event_walk, x, data)
        @test split_value ≈ value rtol = 1.0e-14
        @test split ≈ g rtol = 1.0e-12 atol = 1.0e-12
    end
end

# Several values cached per inner iteration, in Float32 or Float64, and a
# nest that caches values of both element types (two groups).
for name in (:ragged_products, :ragged_products_ungrouped)
    @eval function $name(x::AbstractVector{T}, lengths) where {T}
        total = zero(T)
        for i in eachindex(lengths)
            acc = zero(T)
            for j in 1:lengths[i]
                a = exp(x[1] * sin(T(j)))
                b = sin(x[2] + a)
                c = a * b * x[3]
                acc += c * c + log1p(a * a) * b
            end
            total += acc / (one(T) + T(i))
        end
        return total
    end
end

for name in (:mixed_products, :mixed_products_ungrouped)
    @eval function $name(x, lengths)
        total = 0.0
        for i in eachindex(lengths)
            acc = 0.0
            for j in 1:lengths[i]
                a = exp(x[1] * sin(j))
                b = Float32(sin(x[2] + a))
                c = sin(b * Float32(x[3]))
                acc += a * a * c + Float64(b * c)
            end
            total += acc / (1 + i)
        end
        return total
    end
end

function product_gradient(f, x, lengths)
    dx = zero(x)
    autodiff(Reverse, f, Active, Duplicated(copy(x), dx), Const(lengths))
    return dx
end

const LENGTHS = ([3, 0, 5, 1], Int[], [0, 0], [1, 33, 0, 2, 257], fill(6, 30))

@testset "Packed cache groups: element types" begin
    for lengths in LENGTHS
        for (f, fu, T) in (
                (ragged_products, ragged_products_ungrouped, Float64),
                (ragged_products, ragged_products_ungrouped, Float32),
                (mixed_products, mixed_products_ungrouped, Float64),
            )
            x = T[0.3, -0.2, 0.7]
            g = product_gradient(f, x, lengths)
            if GROUPED
                ref = with_ungrouped(() -> product_gradient(fu, x, lengths))
                @test g ≈ ref rtol = 1.0e-12 atol = 1.0e-12
            end
            direction = T[0.5, -0.25, 1.0]
            # The mixed nest rounds through Float32, as a Float32 nest does.
            rounded = T === Float32 || f === mixed_products
            h = rounded ? 1.0e-2 : 1.0e-6
            xb = Float64.(x)
            fd = (f(xb .+ h .* direction, lengths) - f(xb .- h .* direction, lengths)) / 2h
            tol = rounded ? 2.0e-3 : 1.0e-6
            @test sum(Float64.(g) .* direction) ≈ fd rtol = tol atol = tol
        end
    end
end

# Each value overwrites the input the next iteration reads, so every slot of
# a record must keep its own value.
for name in (:mutating_records, :mutating_records_ungrouped)
    @eval function $name(x, lengths, components)
        total = 0.0
        for rows in lengths, k in 1:components, j in 1:rows
            a = x[j]
            b = (sin(a) + k) / (k + 2)
            x[j] = a * b
            total += x[j] * cos(a)
        end
        return total
    end
end

function mutating_records_reference(x, lengths, components)
    x = copy(x)
    total = 0.0
    for rows in lengths, k in 1:components, j in 1:rows
        a = x[j]
        b = (sin(a) + k) / (k + 2)
        x[j] = a * b
        total += x[j] * cos(a)
    end
    return total, x
end

function mutating_gradient(f, x, lengths, components)
    values, dx = copy(x), zero(x)
    result = autodiff(ReverseWithPrimal, f, Active, Duplicated(values, dx), Const(lengths), Const(components))
    return result[2], dx, values
end

@testset "Packed cache groups: mutation" begin
    for lengths in (Int[], [0, 0], [3, 1, 0, 4], [4, 4, 4, 4, 4]), components in (0, 1, 3)
        x = [0.2, -0.4, 0.6, -0.8]
        value, g, final = mutating_gradient(mutating_records, x, lengths, components)
        ref_value, ref_final = mutating_records_reference(x, lengths, components)
        @test value ≈ ref_value
        @test final == ref_final
        if GROUPED
            u = with_ungrouped(() -> mutating_gradient(mutating_records_ungrouped, x, lengths, components))
            @test value == u[1]
            @test g ≈ u[2] rtol = 1.0e-12 atol = 1.0e-12
        end
        direction = [0.3, -0.7, 0.5, 0.2]
        h = 1.0e-6
        fd = (
            mutating_records_reference(x .+ h .* direction, lengths, components)[1] -
                mutating_records_reference(x .- h .* direction, lengths, components)[1]
        ) / 2h
        @test sum(g .* direction) ≈ fd rtol = 1.0e-6 atol = 1.0e-8
    end
end

# The buffer and its growth are differentiated again: Forward over Reverse
# and Reverse over Reverse.
inner_gradient(x, lengths) = product_gradient(ragged_products, x, lengths)
inner_gradient_ungrouped(x, lengths) = product_gradient(ragged_products_ungrouped, x, lengths)
inner_gradient_sum(x, lengths) = sum(inner_gradient(x, lengths))
inner_gradient_sum_ungrouped(x, lengths) = sum(inner_gradient_ungrouped(x, lengths))

@testset "Packed cache groups: higher derivatives" begin
    for T in (Float64, Float32), lengths in ([3, 0, 5, 1], [0, 0], [1, 33, 0, 2])
        x = T[0.3, -0.2, 0.7]
        direction = T[0.5, -0.25, 1.0]
        hvp = only(autodiff(Forward, inner_gradient, Duplicated(x, direction), Const(lengths)))
        rr = zero(x)
        autodiff(Reverse, inner_gradient_sum, Active, Duplicated(copy(x), rr), Const(lengths))
        if GROUPED
            hvp_u = with_ungrouped(
                () -> only(autodiff(Forward, inner_gradient_ungrouped, Duplicated(x, direction), Const(lengths)))
            )
            @test hvp ≈ hvp_u rtol = 1.0e-12 atol = 1.0e-12
            rr_u = zero(x)
            with_ungrouped(() -> autodiff(Reverse, inner_gradient_sum_ungrouped, Active, Duplicated(copy(x), rr_u), Const(lengths)))
            @test rr ≈ rr_u rtol = 1.0e-12 atol = 1.0e-12
        end
        xb = Float64.(x)
        h = 1.0e-5
        fd = (inner_gradient(xb .+ h .* direction, lengths) .- inner_gradient(xb .- h .* direction, lengths)) ./ 2h
        tol = T === Float32 ? 1.0e-3 : 1.0e-6
        @test Float64.(hvp) ≈ fd rtol = tol atol = tol
        fd_sum = [
            (inner_gradient_sum(xb .+ h .* (1:3 .== k), lengths) - inner_gradient_sum(xb .- h .* (1:3 .== k), lengths)) / 2h
                for k in 1:3
        ]
        @test Float64.(rr) ≈ fd_sum rtol = tol atol = tol
    end
end

function allocation_sites(f, argtypes)
    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(io, f, Active, argtypes; raw = true, dump_module = true, debuginfo = :none)
    ir = String(take!(io))
    return count(r"call[^\n]*@malloc\(", ir)
end

if GROUPED
    @testset "Packed cache groups: one table and growth per group" begin
        data = event_data([8, 8], [3, 4])
        argtypes = Tuple{Duplicated{Vector{Float64}}, map(d -> Const{typeof(d)}, data)...}
        grouped = allocation_sites(event_walk, argtypes)
        ungrouped = with_ungrouped(() -> allocation_sites(event_walk, argtypes))
        # Each inner loop that caches values allocates its table and grows its
        # buffer at one site each, rather than at two sites per cached value.
        # Which values are cached depends on the Julia version: 26 sites
        # against 4 on Julia 1.10, 8 against 4 on 1.12 and 1.13. Allocations
        # that bounds checking adds are the same either way.
        @test grouped <= ungrouped - 4
    end
end
