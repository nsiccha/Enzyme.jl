using Enzyme, Test, LinearAlgebra

# A `ccall` names its library in one of several ways, and Julia compiles them differently:
# a constant of the same module is resolved when the method is compiled, while Julia 1.12
# leaves a module-qualified constant (`Mod.lib`) to a run-time `getproperty` call, a
# non-constant global is read when the call first runs, and so is any other expression
# (which Julia 1.13 no longer accepts). A library read from a global must differentiate like
# the first. Behind any other expression a function Enzyme has a rule for by name does too,
# and a BLAS or LAPACK routine or a function without a rule raises an error.

module CcallLibraries
    import LinearAlgebra
    const constant = Base.libm_name
    variable = Base.libm_name
    module Inner
        const constant = Base.libm_name
    end
    name() = Base.libm_name
    lapack() = LinearAlgebra.BLAS.libblastrampoline
end

const ccall_library_constant = Base.libm_name
ccall_library_variable = Base.libm_name
ccall_lapack_variable = BLAS.libblastrampoline

cos_constant(x) = ccall((:cos, ccall_library_constant), Float64, (Float64,), x)
cos_qualified_constant(x) = ccall((:cos, CcallLibraries.constant), Float64, (Float64,), x)
cos_nested_constant(x) = ccall((:cos, CcallLibraries.Inner.constant), Float64, (Float64,), x)
cos_variable(x) = ccall((:cos, ccall_library_variable), Float64, (Float64,), x)
cos_qualified_variable(x) = ccall((:cos, CcallLibraries.variable), Float64, (Float64,), x)
cos_at_ccall(x) = @ccall CcallLibraries.constant.cos(x::Float64)::Float64
# No library: Julia looks the symbol up in the process. While compiling, Julia 1.10 to 1.12
# attribute a C library function that libjulia-internal links (libc's `memchr` and `memcmp`,
# and `cos`) to libjulia-internal; Julia 1.13 names no library for it. Outside a package image,
# Julia 1.10 calls such a function through the address it resolved while compiling, which
# Enzyme names only for the functions it knows by address; `cos` is not one of them, so
# `cos_without_library` is tested from Julia 1.11 on, which always looks the symbol up.
cos_without_library(x) = ccall(:cos, Float64, (Float64,), x)
# `findfirst` of an ASCII character in a `String` calls `memchr`, `cmp` of two calls `memcmp`.
first_a(x, s) = x * something(findfirst('a', s), 0)
compared(x, s) = x * cmp(s, "m")

# A call that carries no derivative: its argument is a constant.
scaled_by_constant(x) = x * ccall((:cos, CcallLibraries.constant), Float64, (Float64,), 2.0)
scaled_by_variable(x) = x * ccall((:cos, ccall_library_variable), Float64, (Float64,), 2.0)

# Enzyme has no derivative for `remquo`, which also takes a `Ref`.
function remquo_qualified_constant(x)
    quotient = Ref{Cint}(0)
    return ccall((:remquo, CcallLibraries.constant), Float64, (Float64, Float64, Ref{Cint}), x, 2.0, quotient)
end

const scalar_libraries = Any[
    cos_constant, cos_qualified_constant, cos_nested_constant, cos_variable, cos_qualified_variable,
    cos_at_ccall,
]
@static if VERSION >= v"1.11-"
    push!(scalar_libraries, cos_without_library)
end
const lapack_spellings = Pair{Symbol, Any}[
    :qualified_constant => :(BLAS.libblastrampoline),
    :variable => :ccall_lapack_variable,
]

@static if VERSION < v"1.13-"
    cos_computed(x) = ccall((:cos, CcallLibraries.name()), Float64, (Float64,), x)
    scaled_by_computed(x) = x * ccall((:cos, CcallLibraries.name()), Float64, (Float64,), 2.0)
    function remquo_computed(x)
        quotient = Ref{Cint}(0)
        return ccall((:remquo, CcallLibraries.name()), Float64, (Float64, Float64, Ref{Cint}), x, 2.0, quotient)
    end
    push!(scalar_libraries, cos_computed)
end
const lapack_definitions = copy(lapack_spellings)
@static if VERSION < v"1.13-"
    push!(lapack_definitions, :computed => :(CcallLibraries.lapack()))
end

# LAPACK's `lacpy` copies one triangle of `A` into `B`. Its integers and its `uplo` go by
# reference, as the `Ref`s of a hand-written wrapper.
for (spelling, library) in lapack_definitions
    @eval function $(Symbol(:lacpy_, spelling, :!))(B::Matrix{Float64}, A::Matrix{Float64}, uplo::Char)
        m, n = size(A)
        ccall(
            (BLAS.@blasfunc(dlacpy_), $library), Cvoid,
            (
                Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ptr{Float64}, Ref{BLAS.BlasInt},
                Ptr{Float64}, Ref{BLAS.BlasInt}, Clong,
            ),
            uplo, m, n, A, max(1, stride(A, 2)), B, max(1, stride(B, 2)), 1,
        )
        return B
    end
end

function weighted(out, w)
    s = 0.0
    for i in eachindex(out, w)
        s += w[i] * out[i]
    end
    return s
end

const lacpy_source = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0; 13.0 14.0 15.0 16.0] ./ 8

copied_active(lacpy!, x, out, w, uplo) = weighted(lacpy!(out, x, uplo), w)
function copied_constant(lacpy!, x, out, w, uplo)
    lacpy!(out, lacpy_source, uplo)
    out[1, 1] += sum(x)
    return weighted(out, w)
end

reverse_derivative(f, x) = autodiff(Reverse, f, Active, Active(x))[1][1]

@testset "ccall libraries" begin
    x = 0.3
    @testset "$f" for f in scalar_libraries
        @test f(x) ≈ cos(x)
        @test autodiff(Reverse, f, Active, Active(x))[1][1] ≈ -sin(x)
        @test autodiff(set_runtime_activity(Reverse), f, Active, Active(x))[1][1] ≈ -sin(x)
        @test autodiff(Forward, f, Duplicated(x, 1.0))[1] ≈ -sin(x)
        derivative, primal = autodiff(ReverseWithPrimal, f, Active, Active(x))
        @test primal == f(x)
        @test derivative[1] ≈ -sin(x)
        @test autodiff(Forward, reverse_derivative, Const(f), Duplicated(x, 1.0))[1] ≈ -cos(x)
    end

    scaled = Any[scaled_by_constant, scaled_by_variable]
    @static if VERSION < v"1.13-"
        push!(scaled, scaled_by_computed)
    end
    @testset "$f" for f in scaled
        @test autodiff(Reverse, f, Active, Active(0.7))[1][1] ≈ cos(2.0)
        @test autodiff(Forward, f, Duplicated(0.7, 1.0))[1] ≈ cos(2.0)
    end

    # On Julia 1.13 Reverse mode through `memchr` raised "No create nofree of empty function
    # (ejlptr$memchr$0)"; Forward mode, `memcmp` and `cos` got their derivatives.
    @testset "C library functions without a library: $s" for (s, at, order) in
        (("banana", 2, -1), ("xyz", 0, 1), ("λa", 3, 1), ("m", 0, 0))
        @test autodiff(Reverse, first_a, Active, Active(3.0), Const(s))[1][1] == at
        @test autodiff(Forward, first_a, Duplicated(3.0, 1.0), Const(s))[1] == at
        @test autodiff(set_runtime_activity(Reverse), first_a, Active, Active(3.0), Const(s))[1][1] == at
        @test autodiff(Reverse, compared, Active, Active(3.0), Const(s))[1][1] == order
        @test autodiff(Forward, compared, Duplicated(3.0, 1.0), Const(s))[1] == order
    end

    without_rule = Any[remquo_qualified_constant]
    @static if VERSION < v"1.13-"
        push!(without_rule, remquo_computed)
    end
    @testset "$f" for f in without_rule
        @test f(0.7) == 0.7
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Reverse, f, Active, Active(0.7))
        @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Forward, f, Duplicated(0.7, 1.0))
    end

    W = reshape(collect(1.0:16.0), 4, 4)
    w = vec(W)
    @testset "lacpy, $spelling" for (spelling, _) in lapack_spellings
        lacpy! = getfield(@__MODULE__, Symbol(:lacpy_, spelling, :!))
        @testset "$(uplo)" for (uplo, copied) in (('U', triu(W)), ('L', tril(W)))
            kept = W .- copied
            @test lacpy!(fill(-1.0, 4, 4), lacpy_source, uplo) == [copied[i, j] != 0 ? lacpy_source[i, j] : -1.0 for i in 1:4, j in 1:4]

            X = lacpy_source .+ 1.0
            dX = zeros(4, 4)
            dB = zeros(4, 4)
            autodiff(Reverse, copied_active, Active, Const(lacpy!), Duplicated(copy(X), dX), Duplicated(zeros(4, 4), dB), Const(w), Const(uplo))
            @test dX == copied
            @test dB == kept

            dx = zeros(4)
            dB = zeros(4, 4)
            before = copy(lacpy_source)
            autodiff(Reverse, copied_constant, Active, Const(lacpy!), Duplicated([1.0, 2.0, 3.0, 4.0], dx), Duplicated(zeros(4, 4), dB), Const(w), Const(uplo))
            @test dx == fill(W[1, 1], 4)
            @test dB == kept
            @test lacpy_source == before

            dX = zeros(4, 4)
            dB = zeros(4, 4)
            autodiff(set_runtime_activity(Reverse), copied_active, Active, Const(lacpy!), Duplicated(copy(X), dX), Duplicated(zeros(4, 4), dB), Const(w), Const(uplo))
            @test dX == copied
            @test dB == kept

            tangent = reshape(collect(0.5:0.5:8.0), 4, 4)
            seed = fill(-2.0, 4, 4)
            dB = copy(seed)
            autodiff(Forward, lacpy!, Duplicated(zeros(4, 4), dB), Duplicated(copy(X), copy(tangent)), Const(uplo))
            @test dB == [copied[i, j] != 0 ? tangent[i, j] : seed[i, j] for i in 1:4, j in 1:4]
        end
    end

    # A LAPACK routine behind a library that stays a run-time value has no derivative, and
    # says so: before, compiling its derivative ended the process.
    @static if VERSION < v"1.13-"
        @testset "lacpy, computed" begin
            X = lacpy_source .+ 1.0
            @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(
                Reverse, copied_active, Active, Const(lacpy_computed!), Duplicated(copy(X), zeros(4, 4)),
                Duplicated(zeros(4, 4), zeros(4, 4)), Const(w), Const('U'),
            )
            @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(
                Forward, lacpy_computed!, Duplicated(zeros(4, 4), zeros(4, 4)), Duplicated(copy(X), ones(4, 4)), Const('U'),
            )
            @test lacpy_computed!(fill(-1.0, 4, 4), lacpy_source, 'U') == [i <= j ? lacpy_source[i, j] : -1.0 for i in 1:4, j in 1:4]
        end
    end
end
