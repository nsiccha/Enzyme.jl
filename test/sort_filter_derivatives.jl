using Enzyme, Test

@noinline function filtered_differences(ts, d)
    values = Float64[ts[i] - d[j] for j in eachindex(d) for i in eachindex(ts)]
    return sort!(filter!(positive_difference, values))
end
positive_difference(x) = x >= 0.0
filtered_loss(ts, d) = sum(exp, filtered_differences(ts, d))
function filtered_gradient(ts, d)
    g, gd = zero(ts), zero(d)
    autodiff(Reverse, filtered_loss, Active, Duplicated(ts, g), Duplicated(d, gd))
    return g, gd
end
filtered_direction(ts, d, v, w) = only(autodiff(Forward, filtered_loss, Duplicated(ts, v), Duplicated(d, w)))
function filtered_rof(ts, d, v, w)
    hv, hd = zero(ts), zero(d)
    autodiff(Reverse, filtered_direction, Active, Duplicated(ts, hv), Duplicated(d, hd), Const(v), Const(w))
    return hv, hd
end
function filtered_expected(ts, d, v, w)
    g, gd, hv, hd = zero(ts), zero(d), zero(ts), zero(d)
    for j in eachindex(d), i in eachindex(ts)
        q = ts[i] - d[j]
        if q >= 0
            e = exp(q)
            g[i] += e
            gd[j] -= e
            z = e * (v[i] - w[j])
            hv[i] += z
            hd[j] -= z
        end
    end
    return g, gd, hv, hd
end
function check_filtered_products(ts, d, v, w)
    g, gd, hv, hd = filtered_expected(ts, d, v, w)
    actual_g, actual_d = filtered_gradient(ts, d)
    @test actual_g ≈ g
    @test actual_d ≈ gd
    @test filtered_direction(ts, d, v, w) ≈ sum(g .* v) + sum(gd .* w)
    actual_hv, actual_hd = only(autodiff(Forward, filtered_gradient, Duplicated(ts, v), Duplicated(d, w)))
    @test actual_hv ≈ hv
    @test actual_hd ≈ hd
    actual_hv, actual_hd = filtered_rof(ts, d, v, w)
    @test actual_hv ≈ hv
    @test actual_hd ≈ hd
    @test ts == [0.5, 1.5, 2.0, 3.5, 0.25]
    return @test d == [0.0, 1.0, 2.5]
end

filtered_warm_objective(x) = sum(abs2, sin.(x))
@testset "Filtered nested derivatives, fresh and after other derivatives" begin
    ts = [0.5, 1.5, 2.0, 3.5, 0.25]
    d = [0.0, 1.0, 2.5]
    v, w = cos.(eachindex(ts)), sin.(eachindex(d))
    check_filtered_products(ts, d, v, w)
    warm = zero(ts)
    autodiff(Reverse, filtered_warm_objective, Active, Duplicated(ts, warm))
    @test warm ≈ 2 .* sin.(ts) .* cos.(ts)
    check_filtered_products(ts, d, v, w)
end
