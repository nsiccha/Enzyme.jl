using Enzyme, Test

# ReverseWithPrimal returns the primal value the function computed, on every
# path. libEnzyme may run a call's forward pass inside its reverse pass (a
# combined call). Before the dogfood libEnzyme repair it then moved the store of
# the return value after that call, into the call's reverse block, so a path
# that skipped the call returned whatever the return slot held: the seed, zero
# or an unrelated value. The gradient was right.

@noinline function square_first!(x)
    x[1] = x[1] * x[1]
    return nothing
end

# The return value does not depend on the call.
function skip_call(n, u)
    s = u[2] * 3.0
    if n > 1
        square_first!(u)
    end
    return s
end

# The return value reads the length of an array whose contents a call fills
# only when it is not empty (Base's `_copyto_impl!` behind `u[2:last]`).
function empty_slice_length(c, u)
    a = u[1]
    v = u[2:c[1]]
    return -a^2 / 2 + length(v)
end

# The reporter's shape (a ReactiveKernels kernel): a run-time empty slice
# copied, reshaped and summed, and a second slice that is discarded.
function empty_slice_sum(c, u)
    a = u[1]
    v = reshape(copy(u[2:c[1]]), (c[2],))
    p = sum(x -> -(x - a)^2 / 2, v; init = 0.0)
    lj = (u[2:c[1]]; 0.0)
    return -a^2 / 2 + p + lj
end

function reverse_with_primal(mode, f, c, u)
    du = zero(u)
    res = autodiff(mode, Const(f), Active, Const(c), Duplicated(copy(u), du))
    return res[2], du
end

@testset "Combined calls on a skipped path keep the primal return" begin
    modes = (ReverseWithPrimal, set_runtime_activity(ReverseWithPrimal))
    @testset "$(nameof(f)), c = $c" for (f, c, u, grad) in (
            (skip_call, 1, [0.3, 0.5], [0.0, 3.0]),
            (skip_call, 2, [0.3, 0.5], [0.0, 3.0]),
            (empty_slice_length, (1, 0), [0.3, 0.5], [-0.3, 0.0]),
            (empty_slice_length, (2, 0), [0.3, 0.5], [-0.3, 0.0]),
            (empty_slice_length, (3, 0), [0.3, 0.5, 0.7], [-0.3, 0.0, 0.0]),
            (empty_slice_sum, (1, 0), [0.2], [-0.2]),
            (empty_slice_sum, (2, 1), [0.2, 0.5], [0.3 - 0.2, -0.3]),
            (empty_slice_sum, (3, 2), [0.2, 0.5, -0.1], [-0.2 + 0.3 - 0.3, -0.3, 0.3]),
        )
        primal = f(c, copy(u))
        for mode in modes
            val, du = reverse_with_primal(mode, f, c, u)
            @test val === primal
            @test du ≈ grad
        end
        # The gradient-only and split modes agree.
        du = zero(u)
        autodiff(Reverse, Const(f), Active, Const(c), Duplicated(copy(u), du))
        @test du ≈ grad
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            Const{typeof(c)}, Duplicated{typeof(u)}
        )
        du = zero(u)
        uu = copy(u)
        tape, val, _ = fwd(Const(f), Const(c), Duplicated(uu, du))
        rev(Const(f), Const(c), Duplicated(uu, du), 1.0, tape)
        @test val === primal
        @test du ≈ grad
    end
    @testset "gradient(ReverseWithPrimal, …)" begin
        res = gradient(ReverseWithPrimal, empty_slice_sum, Const((1, 0)), [0.2])
        @test res.val === empty_slice_sum((1, 0), [0.2])
        @test res.derivs[2] ≈ [-0.2]
    end
end
