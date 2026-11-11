using Enzyme, Test

# Forward over Reverse where the reverse pass reads integers back from a large
# tape. Type analysis keeps no types past `enzyme-max-type-offset` bytes of the
# tape, so a tape read must carry its own type: otherwise the outer forward pass
# takes the integer for a possible pointer, counts its arithmetic as active, and
# libEnzyme crashed on the xor in the reverse of `filter!`'s `resize!`.

@noinline keep_nonnegative!(y) = filter!(d -> d >= 0.0, y)

const TAPE_SHIFTS = ntuple(i -> 0.25 * i, Val(10))

@eval @noinline function many_tapes(x)
    out = Float64[]
    $([:(push!(out, sum(abs2, sort!(keep_nonnegative!(x .- $c))))) for c in TAPE_SHIFTS]...)
    return out
end

tape_objective(x) = sum(abs2, many_tapes(x))

function tape_gradient(x)
    dx = zero(x)
    autodiff(Reverse, tape_objective, Active, Duplicated(x, dx))
    return dx
end

# With S_i = Σ_k (x_k - c_i)²₊ and f = Σ_i S_i²:
# ∂f/∂x_k = Σ_i 4 S_i (x_k - c_i)₊
# ∂²f/∂x_k∂x_l = Σ_i [8 (x_k - c_i)₊ (x_l - c_i)₊ + δ_kl 4 S_i 1(x_k ≥ c_i)]
function tape_closed_form(x, v)
    g = zero(x)
    hv = zero(x)
    for c in TAPE_SHIFTS
        p = max.(x .- c, 0.0)
        S = sum(abs2, p)
        g .+= 4 .* S .* p
        hv .+= 8 .* p .* sum(p .* v) .+ 4 .* S .* (x .>= c) .* v
    end
    return g, hv
end

@testset "Forward over Reverse reads integers from a large tape" begin
    # Off the filter's boundaries x_k == c_i, where f has a kink in its Hessian.
    x = [1.61, 3.13, 2.57, 4.09]
    v = [1.0, 0.5, -1.0, 2.0]
    g, hv = tape_closed_form(x, v)
    @test tape_objective(x) ≈ sum(sum(abs2, max.(x .- c, 0.0))^2 for c in TAPE_SHIFTS)
    @test tape_gradient(x) ≈ g
    @test only(autodiff(Forward, tape_gradient, Duplicated(x, v))) ≈ hv
    @test x == [1.61, 3.13, 2.57, 4.09]
end

# A function returning several arrays through an sret, whose derivative returns a
# tape through an sret too. Forward over Reverse rebuilds that tape, so its constant
# integers meet active memory. Julia's GC lowering also needs the returnRoots of such
# an sret, even though nothing reads them.
@noinline function stick_breaks(U)
    z = 1.0 ./ (1.0 .+ exp.(-(U .+ permutedims(log.(4 .- (1:3))))))
    l = log1p.(-z)
    lr = cumsum(hcat(0.0 .* U[:, 1:1], l); dims = 2)
    return z, l, lr
end

@noinline function stick_constrain(U)
    z, _, lr = stick_breaks(U)
    return exp.(lr) .* hcat(z, 1.0 .+ 0.0 .* U[:, 1:1])
end

stick_objective(u) = sum(abs2, stick_constrain(reshape(u, 2, 3)))

function stick_gradient(u)
    du = zero(u)
    autodiff(Reverse, stick_objective, Active, Duplicated(u, du))
    return du
end

function central_difference(gradient, u, v, h)
    return (gradient(u .+ h .* v) .- gradient(u .- h .* v)) ./ (2h)
end

function central_gradient(f, u, h)
    return [(f(u .+ h .* (eachindex(u) .== i)) - f(u .- h .* (eachindex(u) .== i))) / (2h) for i in eachindex(u)]
end

@testset "Forward over Reverse rebuilds a tape returned through an sret" begin
    u = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2]
    v = [1.0, 0.5, -1.0, 2.0, 0.25, -0.5]
    h = 1.0e-6
    # On Julia 1.13 (LLVM 20), compiling this plain Reverse pass segfaulted in LLVM's
    # ScalarEvolution loop guards before dogfood libEnzyme f6cb9e61 (Enzyme todo 0e58pgu).
    @test stick_gradient(u) ≈ central_gradient(stick_objective, u, h) rtol = 1.0e-7
    # On Julia 1.13 the outer Forward pass raises EnzymeRuntimeActivityError in the
    # dynamic call that builds CartesianIndices, with every libEnzyme (Enzyme todo 0k6qyml).
    if VERSION >= v"1.13-"
        @test_broken only(autodiff(Forward, stick_gradient, Duplicated(u, v))) ≈ central_difference(stick_gradient, u, v, h) rtol = 1.0e-7
    else
        @test only(autodiff(Forward, stick_gradient, Duplicated(u, v))) ≈ central_difference(stick_gradient, u, v, h) rtol = 1.0e-7
    end
end

# The same computation with the offsets taken from the size and an early return. The
# reverse pass caches the indices of its loops, and their loads reach the mixed-activity
# handler. A load of an integer or of a pointer to integers is its own shadow there,
# whatever the memory it was read from looks like.
function simplex_breaks(U)
    K = size(U, 2) + 1
    z = 1.0 ./ (1.0 .+ exp.(-(U .+ permutedims(log.(K .- (1:(K - 1)))))))
    l = log1p.(-z)
    lr = cumsum(hcat(0.0 .* U[:, 1:1], l); dims = 2)
    return z, l, lr
end

function simplex_constrain(U)
    size(U, 2) == 0 && return ones(size(U, 1), 1)
    z, _, lr = simplex_breaks(U)
    return exp.(lr) .* hcat(z, 1.0 .+ 0.0 .* U[:, 1:1])
end

simplex_objective(u) = sum(abs2, simplex_constrain(reshape(u, 2, 3)))

function simplex_gradient(u)
    du = zero(u)
    autodiff(Reverse, simplex_objective, Active, Duplicated(u, du))
    return du
end

@testset "Forward over Reverse keeps the indices cached in a tape constant" begin
    u = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2]
    v = [1.0, 0.5, -1.0, 2.0, 0.25, -0.5]
    h = 1.0e-6
    @test simplex_gradient(u) ≈ central_gradient(simplex_objective, u, h) rtol = 1.0e-7
    if VERSION >= v"1.13-"
        @test_broken only(autodiff(Forward, simplex_gradient, Duplicated(u, v))) ≈ central_difference(simplex_gradient, u, v, h) rtol = 1.0e-7
    else
        @test only(autodiff(Forward, simplex_gradient, Duplicated(u, v))) ≈ central_difference(simplex_gradient, u, v, h) rtol = 1.0e-7
    end
end
