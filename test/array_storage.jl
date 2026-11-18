using Enzyme, Test

@static if VERSION >= v"1.11-"
    function storage_objective!(x, count, hint, first, shrink)
        Base._deleteend!(x, count)
        sizehint!(x, hint; first, shrink)
        return sum(abs2, x)
    end

    function storage_mutation_only!(x, count, hint)
        Base._deleteend!(x, count)
        sizehint!(x, hint)
        return nothing
    end

    @testset "Storage survives collection between split passes" begin
        original = collect(1.0:8.0)
        x, dx = copy(original), zero(original)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(storage_objective!)}, Active,
            Duplicated{Vector{Float64}}, Const{Int}, Const{Int}, Const{Bool}, Const{Bool}
        )
        tape, primal, _ = fwd(Const(storage_objective!), Duplicated(x, dx), Const(3), Const(100), Const(false), Const(true))
        # Both vectors now refer to new storage. Reverse must retain what it
        # needs from before the resize even if a GC runs between the passes.
        GC.gc()
        rev(Const(storage_objective!), Duplicated(x, dx), Const(3), Const(100), Const(false), Const(true), 1.0, tape)
        @test primal == sum(abs2, original[1:5])
        @test x == original[1:5]
        @test dx == [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0]
    end

    @testset "Inactive storage mutation lanes" begin
        original = collect(1.0:8.0)
        for count in (0, 3, 8), hint in (0, 100)
            x = copy(original)
            autodiff(set_runtime_activity(Reverse), storage_mutation_only!, Const, Duplicated(x, x), Const(count), Const(hint))
            @test x == original[1:(8 - count)]
            for inactive_lane in (1, 2)
                x, dx = copy(original), zero(original)
                shadows = inactive_lane == 1 ? (x, dx) : (dx, x)
                autodiff(set_runtime_activity(Reverse), storage_mutation_only!, Const, BatchDuplicated(x, shadows), Const(count), Const(hint))
                @test x == original[1:(8 - count)]
                @test dx == zero(original)
            end
        end
    end

    @testset "Forward sizehint return aliases" begin
        x, dx, dy = collect(1.0:8.0), ones(8), fill(2.0, 8)
        result = autodiff(ForwardWithPrimal, sizehint!, Duplicated(x, dx), Const(100))
        @test result[1] === dx
        @test result[2] === x
        result = autodiff(ForwardWithPrimal, sizehint!, BatchDuplicated(x, (dx, dy)), Const(200))
        @test result[1][1] === dx
        @test result[1][2] === dy
        @test result[2] === x
        @test dx == ones(8)
        @test dy == fill(2.0, 8)
    end

    @testset "Forward storage changes" begin
        original = Float64[1, 2, 3, 4, 5, 6, 7, 8]
        tangent = cos.(eachindex(original))
        for count in (0, 1, 3, 8), hint in (0, 100), first in (false, true), shrink in (false, true)
            n = length(original) - count
            first && n == 0 && continue
            x, dx = copy(original), copy(tangent)
            derivative = only(autodiff(Forward, storage_objective!, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink)))
            @test derivative ≈ 2sum(original[1:n] .* tangent[1:n])
            @test dx == tangent[1:n]
            @test x == original[1:n]
        end
    end

    # sizehint! reserves storage beyond the vector's length, uninitialized. The
    # shadow's reserved storage must start at zero like any new shadow: push!,
    # pushfirst!, append! and resize! grow into it, and the reverse pass accumulates
    # adjoints there. The adjoints of the elements the vector held at the call must
    # reach the old storage however the vector grew afterwards: pushfirst! moves its
    # header within the reserved storage, and growth past it reallocates.
    # Each objective is sum(abs2, 2x), so its gradient is 8x.
    function reserved_push(x, hint)
        v = sizehint!(Float64[], hint)
        for i in eachindex(x)
            push!(v, 2 * x[i])
        end
        return sum(abs2, v)
    end

    function reserved_append_view(x, hint)
        y = x .* 2.0
        v = sizehint!(Float64[], hint)
        push!(v, y[1])
        append!(v, @view y[2:end])
        return sum(abs2, v)
    end

    function reserved_resize(x, hint)
        v = sizehint!(Float64[], hint)
        resize!(v, length(x))
        for i in eachindex(x)
            v[i] = 2 * x[i]
        end
        return sum(abs2, v)
    end

    function reserved_front(x, hint)
        v = [2 * x[end]]
        sizehint!(v, hint; first = true)
        for i in (length(x) - 1):-1:1
            pushfirst!(v, 2 * x[i])
        end
        return sum(abs2, v)
    end

    function reserved_regrow(x, hint)
        v = [2 * x[1]]
        sizehint!(v, 2)
        for i in 2:length(x)
            push!(v, 2 * x[i])
        end
        return sum(abs2, v)
    end

    # Leave freed storage of the reserved sizes holding nonzero values, so that
    # storage allocated next, uninitialized, is likely to start nonzero.
    @noinline function dirty_freed_storage(lens)
        for len in lens, _ in 1:64
            Base.donotdelete(fill(-1.0e300, len))
        end
        GC.gc(false)
        return nothing
    end

    function reserved_gradient(mode, f, x, hint)
        dx = zero(x)
        autodiff(mode, Const(f), Active, Duplicated(x, dx), Const(hint))
        return dx
    end

    function reserved_batch_gradient(f, x, hint)
        dx1, dx2 = zero(x), zero(x)
        autodiff(Reverse, Const(f), Active, BatchDuplicated(x, (dx1, dx2)), Const(hint))
        return dx1, dx2
    end

    # Two separate defects, present without this rule too: batched Reverse through
    # pushfirst! or a push! that reallocates loses adjoints (Enzyme todo `1t9iw45`),
    # and Forward over Reverse through sizehint! then growth past the reserved
    # storage throws (Enzyme todo `1o8ugkl`). So batch and nest only the shapes that
    # do neither.
    reserved_batch_shape(f) = f !== reserved_front && f !== reserved_regrow
    reserved_nested_shape(f) = f !== reserved_regrow

    @testset "Growth into storage reserved by sizehint!" begin
        for f in (reserved_push, reserved_append_view, reserved_resize, reserved_front, reserved_regrow), n in (4, 7)
            x = [0.3 * i - 1.1 for i in 1:n]
            v = [(-1.0)^i * (0.5 + 0.25 * i) for i in 1:n]
            want = 8 .* x
            for hint in (n + 1, 3n)
                lens = (2, n, n + 1, 3n)
                wrong = 0
                for _ in 1:20
                    dirty_freed_storage(lens)
                    reserved_gradient(Reverse, f, x, hint) ≈ want || (wrong += 1)
                    dirty_freed_storage(lens)
                    reserved_gradient(set_runtime_activity(Reverse), f, x, hint) ≈ want || (wrong += 1)
                    reserved_batch_shape(f) || continue
                    dirty_freed_storage(lens)
                    dx1, dx2 = reserved_batch_gradient(f, x, hint)
                    dx1 ≈ want && dx2 ≈ want || (wrong += 1)
                end
                @test wrong == 0
            end
            reserved_nested_shape(f) || continue
            # Forward over Reverse: the Hessian is 8I.
            wrong = 0
            for _ in 1:20
                dirty_freed_storage((2, n, n + 1))
                hv = only(autodiff(Forward, Const(reserved_gradient), Const(Reverse), Const(f), Duplicated(x, v), Const(n + 1)))
                hv ≈ 8 .* v || (wrong += 1)
            end
            @test wrong == 0
        end
    end

    function storage_alias_objective(x, count, hint, first, shrink)
        y = copy(x)
        old = y.ref.mem
        Base._deleteend!(y, count)
        sizehint!(y, hint; first, shrink)
        return sum(abs2, y) + 3sum(old)
    end

    @testset "Array storage restoration" begin
        for T in (Float32, Float64), mode in (Reverse, set_runtime_activity(Reverse))
            original = T[1, 2, 3, 4, 5, 6, 7, 8]
            for count in (0, 1, 3, 8), first in (false, true), shrink in (false, true)
                n = length(original) - count
                # Base.sizehint! itself rejects some first=true calls on an empty vector.
                first && n == 0 && continue
                for hint in (0, n, 100)
                    x, dx = copy(original), zero(original)
                    autodiff(mode, storage_objective!, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                    expected = [i <= n ? 2original[i] : zero(T) for i in eachindex(original)]
                    @test x == original[1:n]
                    @test dx == expected

                    x, dx1, dx2 = copy(original), zero(original), zero(original)
                    autodiff(mode, storage_objective!, Active, BatchDuplicated(x, (dx1, dx2)), Const(count), Const(hint), Const(first), Const(shrink))
                    @test dx1 == expected
                    @test dx2 == expected

                    x, dx = copy(original), zero(original)
                    autodiff(mode, storage_alias_objective, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                    @test dx == expected .+ 3
                    @test x == original

                    if count == 0
                        x, dx = copy(original), fill(T(17), length(original))
                        autodiff(mode, storage_objective!, Active, Duplicated(x, dx), Const(count), Const(hint), Const(first), Const(shrink))
                        @test dx == expected .+ 17
                    end
                end
            end
        end
    end
end
