using Enzyme
using Test

# The callbacks are native code behind a runtime pointer. Passing the pointer
# as an argument prevents Enzyme from importing the Julia callback's body.
foreign_square(x::Cdouble)::Cdouble = x * x
function foreign_write(context::Ptr{Cvoid}, x::Ptr{Cdouble}, out::Ptr{Cdouble}, err::Ptr{Ptr{Cchar}})::Cint
    unsafe_store!(out, unsafe_load(x)^2)
    return 0
end

struct ForeignCall
    pointer::Ptr{Cvoid}
    context::Ptr{Cvoid}
end

foreign_scalar(pointer, x) = ccall(pointer, Cdouble, (Cdouble,), x)
function foreign_buffer(f::ForeignCall, x)
    out = Ref(0.0)
    err = Ref{Ptr{Cchar}}(C_NULL)
    status = ccall(f.pointer, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ref{Cdouble}, Ref{Ptr{Cchar}}), f.context, x, out, err)
    status == 0 || error("foreign call failed")
    return out[]
end

foreign_inactive(pointer, x, y) = x^2 + foreign_scalar(pointer, y)
const foreign_libm = Base.Math.libm
foreign_supported(x) = ccall((:sin, foreign_libm), Cdouble, (Cdouble,), x)

struct ForeignPrimal{F}
    f::F
end
(f::ForeignPrimal)(x) = foreign_buffer(f.f, x)
foreign_value_gradient(x) = (x[1]^2, [2x[1]])

@testset "Active calls through constant native pointers" begin
    scalar = @cfunction(foreign_square, Cdouble, (Cdouble,))
    buffer = ForeignCall(@cfunction(foreign_write, Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Ptr{Cchar}})), C_NULL)
    @test foreign_scalar(scalar, 0.3) ≈ 0.09
    @test foreign_buffer(buffer, [0.3]) ≈ 0.09
    @test foreign_supported(0.3) ≈ sin(0.3)

    # Both the active scalar result and active memory must reach the diagnostic.
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Reverse, foreign_scalar, Active, Const(scalar), Active(0.3))
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Forward, foreign_scalar, Const(scalar), Duplicated(0.3, 1.0))
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError gradient(Reverse, foreign_buffer, Const(buffer), [0.3])
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Forward, foreign_buffer, Const(buffer), Duplicated([0.3], [1.0]))

    # An inactive call still executes normally; known foreign-call rules still apply.
    @test autodiff(Reverse, foreign_inactive, Active, Const(scalar), Active(0.3), Const(0.4))[1][2] ≈ 0.6
    @test autodiff(Forward, foreign_inactive, Const(scalar), Duplicated(0.3, 1.0), Const(0.4))[1] ≈ 0.6
    @test autodiff(Reverse, foreign_supported, Active, Active(0.3))[1][1] ≈ cos(0.3)
    @test autodiff(Forward, foreign_supported, Duplicated(0.3, 1.0))[1] ≈ cos(0.3)

    supplied = WithGradient(ForeignPrimal(buffer), foreign_value_gradient)
    @test gradient(Reverse, Const(supplied), [0.3])[1] ≈ [0.6]
    @test autodiff(Forward, Const(supplied), Duplicated([0.3], [2.0]))[1] ≈ 1.2
end
