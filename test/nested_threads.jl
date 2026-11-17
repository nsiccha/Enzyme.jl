using Enzyme
using Test

# Derivatives of derivatives of `Threads.@threads` loops. The outer derivative differentiates
# the code the inner one runs the loop with: the calls of `threading_run` it makes, and the
# callables it hands them, which reach the derivative of the loop's body.

function nt_sumsq(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]^2
    end
    return s
end

# Every iteration adds into the one shadow of `x`: gradient `128 x`, Hessian `128 I`.
function nt_dynamic(x)
    out = zeros(64)
    Threads.@threads for i in 1:64
        @inbounds out[i] = nt_sumsq(x)
    end
    return sum(out)
end

function nt_static(x)
    out = zeros(64)
    Threads.@threads :static for i in 1:64
        @inbounds out[i] = nt_sumsq(x)
    end
    return sum(out)
end

# Not quadratic, and a weight per iteration: gradient `55 a exp(a x)`, Hessian diagonal
# `55 a^2 exp(a x)`.
const nt_rates = [0.3, -0.2, 0.5, 0.1, -0.4, 0.25]

function nt_expsum(x)
    s = 0.0
    @inbounds for j in eachindex(x)
        s += exp(nt_rates[j] * x[j])
    end
    return s
end

function nt_weighted(x)
    out = zeros(10)
    Threads.@threads for i in 1:10
        @inbounds out[i] = i * nt_expsum(x)
    end
    return sum(out)
end

# A loop that writes an argument, whose body holds an active scalar.
function nt_scaled!(y, x, a)
    Threads.@threads for i in eachindex(x)
        @inbounds y[i] = a * x[i]^2
    end
    return nothing
end

function nt_scaled_gradient!(dx, x, a)
    y = zero(x)
    dy = ones(length(x))
    return autodiff(Reverse, Const(nt_scaled!), Const, Duplicated(y, dy), Duplicated(x, dx), Active(a))[1][3]
end

function nt_gradient!(f::F, dx, x) where {F}
    autodiff(Reverse, Const(f), Active, Duplicated(x, dx))
    return nothing
end

function nt_runtime_gradient!(f::F, dx, x) where {F}
    autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(x, dx))
    return nothing
end

nt_directional(f::F, x, v) where {F} = autodiff(Forward, Const(f), Duplicated(x, v))[1]

function nt_gradient_sqnorm(f::F, x) where {F}
    dx = zero(x)
    nt_gradient!(f, dx, x)
    return sum(abs2, dx)
end

@testset "Nested derivatives of Threads.@threads loops $(Threads.nthreads())" begin
    x = collect(1.0:6.0)
    v = collect(6.0:-1.0:1.0)
    w = [1.0, -1.0, 2.0, 0.5, 0.0, 3.0]

    @testset "$name schedule" for (name, f) in (("default", nt_dynamic), ("static", nt_static))
        # The exact values are integers, so a lost or doubled update of a shadow shows.
        @testset "Forward over Reverse" begin
            dx = zero(x)
            ddx = zero(x)
            autodiff(Forward, Const(nt_gradient!), Const, Const(f), Duplicated(dx, ddx), Duplicated(x, v))
            @test dx == 128 .* x
            @test ddx == 128 .* v
        end

        @testset "Forward over Reverse, a batch of two directions" begin
            dx = zero(x)
            ddx1 = zero(x)
            ddx2 = zero(x)
            autodiff(
                Forward, Const(nt_gradient!), Const, Const(f),
                BatchDuplicated(dx, (ddx1, ddx2)), BatchDuplicated(x, (v, w)),
            )
            @test dx == 128 .* x
            @test ddx1 == 128 .* v
            @test ddx2 == 128 .* w
        end

        @testset "Reverse over Forward" begin
            hv = zero(x)
            autodiff(Reverse, Const(nt_directional), Active, Const(f), Duplicated(x, hv), Const(v))
            @test hv == 128 .* v
        end

        @testset "Forward over Forward" begin
            @test autodiff(Forward, Const(nt_directional), Const(f), Duplicated(x, w), Const(v))[1] ==
                128 * sum(v .* w)
        end

        @testset "Reverse over Reverse" begin
            g = zero(x)
            autodiff(Reverse, Const(nt_gradient_sqnorm), Active, Const(f), Duplicated(x, g))
            @test g ≈ 2 * 128^2 .* x
        end
    end

    @testset "A weight per iteration" begin
        gradient = 55 .* nt_rates .* exp.(nt_rates .* x)
        hessian = 55 .* nt_rates .^ 2 .* exp.(nt_rates .* x)

        dx = zero(x)
        ddx = zero(x)
        autodiff(Forward, Const(nt_gradient!), Const, Const(nt_weighted), Duplicated(dx, ddx), Duplicated(x, v))
        @test dx ≈ gradient
        @test ddx ≈ hessian .* v

        dx = zero(x)
        ddx = zero(x)
        autodiff(
            set_runtime_activity(Forward), Const(nt_runtime_gradient!), Const, Const(nt_weighted),
            Duplicated(dx, ddx), Duplicated(x, v),
        )
        @test dx ≈ gradient
        @test ddx ≈ hessian .* v

        hv = zero(x)
        autodiff(Reverse, Const(nt_directional), Active, Const(nt_weighted), Duplicated(x, hv), Const(v))
        @test hv ≈ hessian .* v
    end

    @testset "An active scalar held by the loop body" begin
        # The gradient of `sum(a x^2)` is `2 a x` in `x` and `sum(x^2)` in `a`. With the
        # tangents `v` of `x` and 1 of `a`, theirs are `2 x + 2 a v` and `2 sum(x v)`.
        a = 2.0
        dx = zero(x)
        ddx = zero(x)
        dda, da = autodiff(
            ForwardWithPrimal, Const(nt_scaled_gradient!), Duplicated,
            Duplicated(dx, ddx), Duplicated(x, v), Duplicated(a, 1.0),
        )
        @test dx ≈ 2a .* x
        @test da ≈ sum(abs2, x)
        @test ddx ≈ 2 .* x .+ 2a .* v
        @test dda ≈ 2 * sum(x .* v)
    end
end

@testset "A rule's calls keep their arguments in the code an outer derivative takes up" begin
    # The code Enzyme keeps of a derivative, for an outer derivative to differentiate, holds
    # calls that only a rule can differentiate: those of `threading_run` among them. The
    # rule reads their arguments by position, so none may have been folded into the callee.
    LLVM = Enzyme.Compiler.LLVM
    augmented, _ = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(nt_dynamic)}, Active, Duplicated{Vector{Float64}})
    kept = Enzyme.Compiler.cached_autodiff(augmented.primal)
    @test kept !== nothing
    ruled = String[]
    folded = String[]
    LLVM.Context() do _
        mod = parse(LLVM.Module, Vector{UInt8}(codeunits(kept.bitcode)))
        for fn in LLVM.functions(mod)
            attrs = Dict{String, String}()
            for attr in collect(LLVM.function_attributes(fn))
                attr isa LLVM.StringAttribute || continue
                attrs[LLVM.kind(attr)] = LLVM.value(attr)
            end
            get(attrs, "enzyme_math", "") in Enzyme.Compiler.parallel_enzyme_math || continue
            push!(ruled, attrs["enzyme_math"])
            haskey(attrs, "enzyme_parmremove") && push!(folded, LLVM.name(fn))
        end
        LLVM.dispose(mod)
    end
    @test "jl_threadsfor" in ruled
    @test isempty(folded)
end

@testset "The loop bodies an outer derivative differentiates" begin
    Compiler = Enzyme.Compiler
    Thunk = Compiler.ForwardModeThunk{Ptr{Cvoid}, Const{typeof(nt_sumsq)}, Const{Nothing}, Tuple{Const{Int}}, 1, false}
    settings = ((false, false), false, false)
    for T in (
            Compiler.ForwardPFor{Thunk, Const{typeof(nt_sumsq)}, settings, false},
            Compiler.AugmentedPFor{Thunk, Const{typeof(nt_sumsq)}, true, false, Vector{Float64}, settings, false},
            Compiler.ReversePFor{Thunk, Const{typeof(nt_sumsq)}, false, true, Ptr{Float64}, settings, false},
        )
        R = Compiler.rederived_body(T)
        @test R !== T
        @test R.name === T.name
        @test R.parameters[end] === true
        @test R.parameters[1:(end - 1)] == T.parameters[1:(end - 1)]
        @test fieldtypes(R) == fieldtypes(T)
        @test sizeof(R) == sizeof(T)
        # Already the one to differentiate
        @test Compiler.rederived_body(R) === R
    end
    # Any other loop body is differentiated as it is
    @test Compiler.rederived_body(typeof(nt_sumsq)) === typeof(nt_sumsq)
    @test Compiler.rederived_body(Any) === Any
end
