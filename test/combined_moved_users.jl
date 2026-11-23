using Enzyme, Test

# libEnzyme may run a call's forward pass inside its reverse pass (a combined
# call). It then moves the instructions that use the call's result after the
# call, into the call's reverse block. Before the dogfood libEnzyme repair, a
# returned sum that used the results of two such calls in different blocks was
# moved by both: the first call's reverse block kept the part of the sum that
# used only its result, and the second call's took the rest, which reads that
# part before the first call's reverse code computes it. ReverseWithPrimal then
# aborted the process with "Instruction does not dominate all uses!" while
# compiling. The repair runs the second call augmented instead.

@noinline g(z) = log1p(z * z)
dg(z) = 2z / (1 + z * z)

# Two calls in different blocks (a loop between them) feed one returned sum.
function two_blocks(x, y, n)
    a = g(x)
    s1 = 3.0 * a
    t = 0.0
    for i in 1:n
        t += i * y
    end
    b = g(y)
    s2 = 5.0 * b
    return (s1 + t) + s2
end
two_blocks_grad(x, y, n) = (3dg(x), n * (n + 1) / 2 + 5dg(y))

# Three calls, each in its own block.
function three_blocks(x, y, n)
    a = g(x)
    t = 0.0
    for i in 1:n
        t += i * x
    end
    b = g(y)
    u = 0.0
    for i in 1:n
        u += y
    end
    c = g(x * y)
    return ((2.0 * a + t) + (b + u)) - 4.0 * c
end
three_blocks_grad(x, y, n) =
    (2dg(x) + n * (n + 1) / 2 - 4y * dg(x * y), dg(y) + n - 4x * dg(x * y))

# Both calls in one block: these stay combined.
function one_block(x, y, n)
    a = g(x)
    b = g(y)
    return (3.0 * a + n) + 5.0 * b
end
one_block_grad(x, y, n) = (3dg(x), 5dg(y))

# The reporter's shape: a Student-t log kernel `log1p(z^2 / nu)` with branches,
# taken twice on standardized array reads around a loop over the same array.
@noinline function student_log1p(nu, z)
    result = zero(nu + z)
    if nu == Inf
        result = z^2
    else
        t = z^2 / nu
        if (t < Inf) & (nu >= 0)
            result = log1p(t)
        else
            result = 2log(abs(z)) - log(nu) + log1p((nu / z) / z)
        end
    end
    return result
end
function student_pair(x, c)
    @inbounds x1 = x[1]
    @inbounds x2 = x[2]
    a = student_log1p(c[1], (x1 - c[2]) / c[3])
    s1 = c[4] + (-0.5 * (c[1] + 1)) * a - c[5]
    t = 0.0
    for i in 3:length(x)
        @inbounds t += x[i]^2
    end
    b = student_log1p(c[6], (x2 - c[7]) / c[8])
    s2 = c[9] + (-0.5 * (c[6] + 1)) * b - c[10]
    return (s1 + -0.5 * t) + s2
end
function student_pair_grad(x, c)
    z1 = (x[1] - c[2]) / c[3]
    z2 = (x[2] - c[7]) / c[8]
    d1 = -0.5 * (c[1] + 1) * (2z1 / c[1]) / (1 + z1^2 / c[1]) / c[3]
    d2 = -0.5 * (c[6] + 1) * (2z2 / c[6]) / (1 + z2^2 / c[6]) / c[8]
    return [d1; d2; -x[3:end]]
end

@testset "Two combined calls whose results feed one return" begin
    modes = (ReverseWithPrimal, set_runtime_activity(ReverseWithPrimal))
    @testset "$(nameof(f)), n = $n" for f in (two_blocks, three_blocks, one_block),
            n in (0, 1, 4)
        grad = getfield(@__MODULE__, Symbol(nameof(f), :_grad))
        x, y = 0.3, -0.7
        for mode in modes
            res = autodiff(mode, Const(f), Active, Active(x), Active(y), Const(n))
            @test res[2] === f(x, y, n)
            @test res[1][1] ≈ grad(x, y, n)[1]
            @test res[1][2] ≈ grad(x, y, n)[2]
        end
        res = autodiff(Reverse, Const(f), Active, Active(x), Active(y), Const(n))
        @test res[1][1] ≈ grad(x, y, n)[1]
        @test res[1][2] ≈ grad(x, y, n)[2]
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            Active{Float64}, Active{Float64}, Const{Int}
        )
        tape, val, _ = fwd(Const(f), Active(x), Active(y), Const(n))
        dres = rev(Const(f), Active(x), Active(y), Const(n), 1.0, tape)
        @test val === f(x, y, n)
        @test dres[1][1] ≈ grad(x, y, n)[1]
        @test dres[1][2] ≈ grad(x, y, n)[2]
    end
    @testset "student_pair, length $(length(x))" for x in (
            [0.3, -0.7], [0.3, -0.7, 0.2, 0.4, -0.1],
        )
        c = (3.0, 0.1, 2.0, 0.5, 0.2, 4.0, -0.2, 1.5, 0.3, 0.1)
        for mode in (ReverseWithPrimal, set_runtime_activity(ReverseWithPrimal))
            dx = zero(x)
            res = autodiff(mode, Const(student_pair), Active, Duplicated(copy(x), dx), Const(c))
            @test res[2] === student_pair(x, c)
            @test dx ≈ student_pair_grad(x, c)
        end
        res = gradient(ReverseWithPrimal, student_pair, x, Const(c))
        @test res.val === student_pair(x, c)
        @test res.derivs[1] ≈ student_pair_grad(x, c)
    end
end
