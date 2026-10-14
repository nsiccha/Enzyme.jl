using Enzyme, LinearAlgebra, Test

# An unannotated function argument whose type can hold derivative data must be
# proven readonly (`err_if_func_written`). These callables only read the data
# they hold, so they must differentiate exactly as `Const(f)` does.
#
# `Matrix + UniformScaling` takes the `UniformScaling` by reference, so the
# copy that the calling-convention wrapper makes of a by-value argument
# (`.innerparm`) survives optimization. LLVM splits the store that fills that
# copy into one store per field; the check used to report those stores as
# writes to the argument. It also used to treat the fresh matrix such a call
# returns as data of the argument, and so rejected passing it to LAPACK, and
# where a load of a held array was untyped (Julia 1.10), it followed the float
# read from it into the fresh result.

struct ROShifted{T}
    data::T
end
function (c::ROShifted)(x)
    A, s = c.data
    return sum(x^2 .* A + s)
end

struct ROSwapped{T}
    data::T
end
function (c::ROSwapped)(x)
    s, A = c.data
    return sum(x^2 .* A + s)
end

struct ROFields{M, S}
    A::M
    s::S
end
(c::ROFields)(x) = sum(x^2 .* c.A + c.s)

struct ROThree{T}
    data::T
end
function (c::ROThree)(x)
    A, B, s = c.data
    return sum((x^2 .* A + s) .* B)
end

struct ROCholesky{T}
    data::T
end
function (c::ROCholesky)(rho)
    D, s = c.data
    K = exp.(-0.5 .* D ./ rho^2) + s
    return logdet(cholesky(Symmetric(K)))
end

@noinline ro_tuple_sum(t) = sum(t[1]) * t[2]

# The tuple passed by reference is a stack slot holding the captured matrix and
# active data.
struct ROLocalTuple{M}
    A::M
end
(c::ROLocalTuple)(x) = ro_tuple_sum((c.A, x * x))

@noinline ro_first(t) = t[1]

struct ROReturned{T}
    data::T
end
function (c::ROReturned)(x)
    A = ro_first(c.data)
    return sum(x^2 .* A) + c.data[2].λ
end

# Controls: each writes data the callable holds, so the check must still raise.
struct ROWriteMatrix{T}
    data::T
end
function (c::ROWriteMatrix)(x)
    A, s = c.data
    A[1] = x
    return sum(x .* A + s)
end

struct ROWriteRef{T}
    data::T
end
function (c::ROWriteRef)(x)
    A, s, r = c.data
    r[] = x
    return sum(x .* A + s) + r[]
end

struct ROWriteReturned{T}
    data::T
end
function (c::ROWriteReturned)(x)
    A = ro_first(c.data)
    A[1] = x
    return sum(A + c.data[2])
end

ro_A() = [1.0 0.5; 0.5 1.0]
ro_B() = [2.0 0.0; 1.0 3.0]
ro_D() = [0.0 1.0 4.0; 1.0 0.0 1.0; 4.0 1.0 0.0]

@testset "Readonly check: by-value argument copies and fresh call results" begin
    for f in (
            ROShifted((ro_A(), 1.0e-10I)), ROSwapped((1.0e-10I, ro_A())),
            ROFields(ro_A(), 1.0e-10I), ROThree((ro_A(), ro_B(), 1.0e-10I)),
            ROLocalTuple(ro_A()), ROReturned((ro_A(), 1.0e-10I)),
        )
        expected = autodiff(ReverseWithPrimal, Const(f), Active, Active(1.5))
        @test autodiff(ReverseWithPrimal, f, Active, Active(1.5)) == expected
        dexpected = autodiff(ForwardWithPrimal, Const(f), Duplicated(1.5, 1.0))
        @test autodiff(ForwardWithPrimal, f, Duplicated(1.5, 1.0)) == dexpected
    end
    @test autodiff(Reverse, ROShifted((ro_A(), 1.0e-10I)), Active, Active(1.5))[1][1] ≈ 9.0

    f = ROCholesky((ro_D(), 1.0e-10I))
    @test autodiff(ReverseWithPrimal, f, Active, Active(1.3)) ==
        autodiff(ReverseWithPrimal, Const(f), Active, Active(1.3))

    for f in (
            ROWriteMatrix((ro_A(), 1.0e-10I)), ROWriteRef((ro_A(), 1.0e-10I, Ref(0.0))),
            ROWriteReturned((ro_A(), 1.0e-10I)),
        )
        @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, f, Active, Active(1.5))
        @test f.data[1] == ro_A()
    end
end
