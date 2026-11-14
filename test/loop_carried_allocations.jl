using Enzyme, Test

# Each step makes a new array whose shape depends on the previous step's array.
# Replaying allocations with one cache slot must not substitute the current
# array for the previous iteration's array before its header is initialized.
# This is the ordinary Julia form of a scan nested inside a broadcasted plate.
function nested_scan_carry(w, constants)
    groups = constants.groups
    group_axes = axes(groups)
    total = 0.0
    if !isempty(CartesianIndices(group_axes))
        prepared = Base.Broadcast.preprocess(nothing, groups)
        initial = zeros(length(w))
        @simd for gi in group_axes[1]
            index = CartesianIndex(gi)
            indices = eachindex(@inbounds Base.Broadcast._broadcast_getindex(prepared, index))
            trajectory = similar(@inbounds(Base.Broadcast._broadcast_getindex(prepared, index)), Float64)
            previous = initial
            for i in indices
                x = @inbounds Base.Broadcast._broadcast_getindex(prepared, index)[i]
                ax = Base.Broadcast.broadcast_shape(axes(previous), axes(w))
                next = similar(previous, Float64, ax)
                step_total = 0.0
                if !isempty(CartesianIndices(ax))
                    prevbc = Base.Broadcast.preprocess(nothing, previous)
                    wbc = Base.Broadcast.preprocess(nothing, w)
                    for j in CartesianIndices(ax)
                        value = @inbounds Base.Broadcast._broadcast_getindex(prevbc, j) + x * Base.Broadcast._broadcast_getindex(wbc, j)
                        @inbounds next[j] = value
                        step_total += value
                    end
                end
                @inbounds trajectory[i] = step_total
                previous = next
            end
            total += sum(trajectory)
        end
    end
    return total
end

@testset "Allocations carried across scan iterations" begin
    groupsets = (
        [[0.3, -0.2, 0.5], Float64[], [0.1], [0.4, 0.2]],
        [Float64[], [0.4, 0.2], [0.1], Float64[], [0.3, -0.2, 0.5]],
        [[0.2]],
        [[0.1, -0.2, 0.3, -0.4, 0.5, 0.6]],
        [Float64[]],
        Vector{Float64}[],
    )
    for groups in groupsets, width in (0, 1, 3, 7)
        w = collect(1.0:width)
        original_groups = deepcopy(groups)
        original_w = copy(w)
        coefficient = 0.0
        for xs in groups
            prefix = 0.0
            for x in xs
                prefix += x
                coefficient += prefix
            end
        end
        expected_value = coefficient * sum(w)
        expected_gradient = fill(coefficient, width)
        constants = (; groups)
        for repeat in 1:3
            dw = zero(w)
            result = autodiff(
                ReverseWithPrimal, Const(nested_scan_carry), Active,
                Duplicated(w, dw), Const(constants),
            )
            @test result[2] ≈ expected_value
            @test dw ≈ expected_gradient
            @test w == original_w
            @test groups == original_groups
            GC.gc()
        end

        dw = zero(w)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(nested_scan_carry)}, Active,
            Duplicated{typeof(w)}, Const{typeof(constants)},
        )
        tape, value, _ = fwd(Const(nested_scan_carry), Duplicated(w, dw), Const(constants))
        GC.gc()
        rev(Const(nested_scan_carry), Duplicated(w, dw), Const(constants), 1.0, tape)
        @test value ≈ expected_value
        @test dw ≈ expected_gradient
        @test w == original_w
        @test groups == original_groups
    end
end
