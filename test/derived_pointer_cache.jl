using Enzyme, Test

# The reverse pass needs the element `y[k]` of every iteration. Nothing writes it, so it can be
# recomputed, but `y` and `k` must be saved, as the function overwrites what it read them from.
# With two values to save and one address computed from them, the smallest thing to save is
# that address. It is a derived pointer, which cannot be saved, and neither can the data pointer
# of `y` it is computed from: libEnzyme raised "Illegal cached pointer". It now saves `y` and `k`
# and recomputes the rest. Recomputing from what the function overwrote would read `arrays[2]`
# or index 1, which the expected gradients below tell apart.
@noinline function overwrites_array_and_index(x, arrays, idxs, n)
    y = arrays[1]
    k = idxs[1]
    out = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        d = y[k] - x[j]
        out[j] = -0.5 * d * d
    end
    idxs[1] = 1
    arrays[1] = arrays[2]
    return sum(out)
end

# The array comes out of a vector of arrays the function allocates and then overwrites.
@noinline function overwrites_held_array_and_index(x, arrays, idxs, n)
    held = Vector{Vector{Float64}}(undef, 2)
    held[1] = arrays[1]
    held[2] = arrays[2]
    k = idxs[1]
    y = held[k - 1]
    out = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        d = y[k] - x[j]
        out[j] = -0.5 * d * d
    end
    held[1] = held[2]
    idxs[1] = 1
    return sum(out) + length(held[1])
end

# With the same shape inside a loop, each iteration has its own `y` and `k` to save.
@noinline function overwrites_in_outer_loop(x, arrays, idxs, n)
    s = 0.0
    for i in 1:2
        y = arrays[i]
        k = idxs[i]
        out = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            d = y[k] - x[j]
            out[j] = -0.5 * d * d
        end
        idxs[i] = 1
        arrays[i] = arrays[3 - i]
        s += sum(out)
    end
    return s
end

# A control: with only the array overwritten, there is one value to save and no reason to
# consider its address.
@noinline function overwrites_array_only(x, arrays, idxs, n)
    y = arrays[1]
    k = idxs[1]
    out = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        d = y[k] - x[j]
        out[j] = -0.5 * d * d
    end
    arrays[1] = arrays[2]
    return sum(out)
end

derived_pointer_inputs() = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [2, 3])

@testset "A derived pointer is never saved for the reverse pass" begin
    x = [0.3, -1.2, 2.0, 0.7]
    one_element = 2.0 .- x   # arrays[1][2]
    two_elements = (2.0 .- x) .+ (6.0 .- x)   # and arrays[2][3]
    cases = (
        (overwrites_array_and_index, one_element),
        (overwrites_held_array_and_index, one_element),
        (overwrites_in_outer_loop, two_elements),
        (overwrites_array_only, one_element),
    )
    for (f, expected) in cases
        @testset "$(nameof(f))" begin
            for mode in (Reverse, set_runtime_activity(Reverse))
                arrays, idxs = derived_pointer_inputs()
                dx = zero(x)
                autodiff(mode, f, Active, Duplicated(x, dx), Const(arrays), Const(idxs), Const(length(x)))
                @test dx ≈ expected
                @test x == [0.3, -1.2, 2.0, 0.7]
            end

            arrays, idxs = derived_pointer_inputs()
            dx = zero(x)
            result = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(arrays), Const(idxs), Const(length(x)))
            @test dx ≈ expected
            arrays, idxs = derived_pointer_inputs()
            @test result[2] == f(x, arrays, idxs, length(x))

            # An empty loop reads no element, so nothing depends on `x`.
            arrays, idxs = derived_pointer_inputs()
            dx = zero(x)
            autodiff(Reverse, f, Active, Duplicated(x, dx), Const(arrays), Const(idxs), Const(0))
            @test all(iszero, dx)
        end
    end
end
