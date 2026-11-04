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
    # On Julia 1.12 and later, Reverse through a filter! that drops an element puts
    # derivatives on the wrong entries (Enzyme todo 063gmv7).
    if VERSION >= v"1.12-"
        @test_broken tape_gradient(x) ≈ g
        @test_broken only(autodiff(Forward, tape_gradient, Duplicated(x, v))) ≈ hv
    else
        @test tape_gradient(x) ≈ g
        @test only(autodiff(Forward, tape_gradient, Duplicated(x, v))) ≈ hv
    end
    @test x == [1.61, 3.13, 2.57, 4.09]
end
