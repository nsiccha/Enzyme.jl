using Enzyme, Test

struct TapedGatherView <: AbstractVector{Float64}
    base::Vector{Float64}
    active::Vector{Float64}
    slots::Vector{Int32}
end

Base.size(p::TapedGatherView) = size(p.base)
Base.IndexStyle(::Type{TapedGatherView}) = IndexLinear()

@inline function Base.getindex(p::TapedGatherView, i::Int)
    @boundscheck checkbounds(p.base, i)
    k = @inbounds p.slots[i]
    return iszero(k) ? (@inbounds p.base[i]) : (@inbounds p.active[k])
end

@inline taped_entry(data, i, j) = data[i, CartesianIndices(Base.tail(axes(data)))[j]]

# Keep the binary search outside the differentiated gather so its loop state
# stays separate from the linear index that the gather's reverse pass saves.
Base.@noinline taped_row(rows, key) = searchsortedfirst(rows, key)

function taped_loss(x, base, slots, rows, ids, stride)
    data = reshape(TapedGatherView(base, x, slots), length(rows), :)
    values = [taped_entry(data, taped_row(rows, mod1(u, stride)), cld(u, stride)) for u in ids]
    return sum(abs2, values)
end

function taped_gradient!(g, x, base, slots, rows, ids, stride)
    autodiff(Reverse, taped_loss, Active, Duplicated(x, g), Const(base), Const(slots), Const(rows), Const(ids), Const(stride))
    return nothing
end

function taped_direction(x, v, base, slots, rows, ids, stride)
    return only(autodiff(Forward, taped_loss, Duplicated, Duplicated(x, v), Const(base), Const(slots), Const(rows), Const(ids), Const(stride)))
end

@testset "Static Hessian products through a reshaped gather" begin
    x = [1.0, 2.0, 3.0, 4.0]
    args = Const.((collect(10.0:15.0), Int32[0, 1, 3, 0, 4, 0], [1, 3], [1, 3, 4, 6, 7, 9], 3))
    # The gather reads x[1], x[3], and x[4] once each. The other three
    # entries come from the constant base. The tape saves the linear index.
    expected_gradient = [2.0, 0.0, 6.0, 8.0]
    v = [4.0, -2.0, 1.0, 3.0]
    w = [-1.0, 7.0, 2.0, 0.5]
    expected_v = [8.0, 0.0, 2.0, 6.0]
    expected_w = [-2.0, 0.0, 4.0, 1.0]

    g, hv = zero(x), zero(x)
    autodiff(Forward, taped_gradient!, Const, Duplicated(g, hv), Duplicated(x, v), args...)
    @test g == expected_gradient
    @test hv == expected_v

    g, hv, hw = zero(x), zero(x), zero(x)
    autodiff(Forward, taped_gradient!, Const, BatchDuplicated(g, (hv, hw)), BatchDuplicated(x, (v, w)), args...)
    @test g == expected_gradient
    @test hv == expected_v
    @test hw == expected_w

    for (dir, forward_over_reverse) in ((v, hv), (w, hw))
        reverse_over_forward = zero(x)
        autodiff(Reverse, taped_direction, Active, Duplicated(x, reverse_over_forward), Const(dir), args...)
        @test reverse_over_forward == forward_over_reverse
    end
end
