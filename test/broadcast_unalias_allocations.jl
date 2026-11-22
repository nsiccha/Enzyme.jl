using Enzyme, Test

function broadcast_product!(a, c, b)
    a .= c .* b
    return nothing
end

function broadcast_add_product!(a, c, b)
    a .+= c .* b
    return nothing
end

function broadcast_allocations(mode, f::F, n) where {F}
    a, da = fill(1.0, n), fill(0.25, n)
    b, db = fill(2.0, n), fill(0.5, n)
    c = fill(3.0, n)
    args = (Const(f), Const, Duplicated(a, da), Const(c), Duplicated(b, db))
    autodiff(mode, args...)
    adds = f === broadcast_add_product!
    @test all(==(adds ? 7.0 : 6.0), a)
    if mode === Forward
        @test all(==(adds ? 1.75 : 1.5), da)
        @test all(==(0.5), db)
    else
        @test all(==(adds ? 0.25 : 0.0), da)
        @test all(==(1.25), db)
    end
    @test all(==(3.0), c)
    return @allocated autodiff(mode, args...)
end

@testset "Disjoint broadcast inputs need no array-sized shadow" begin
    for f in (broadcast_product!, broadcast_add_product!), mode in (Forward, Reverse)
        small = broadcast_allocations(mode, f, 128)
        large = broadcast_allocations(mode, f, 8192)
        @test large <= small + 1024
    end
end

function broadcast_reversed_view(x, c)
    a = copy(x)
    a .+= c .* view(a, length(a):-1:1)
    return sum(a)
end

@testset "A broadcast reads its overlapping view before overwriting it" begin
    x, c = [1.0, -2.0, 4.0, 8.0], [2.0, 3.0, -1.0, 5.0]
    direction = [2.0, 1.0, -3.0, 4.0]
    value = sum(x .+ c .* reverse(x))
    gradient = 1 .+ reverse(c)
    tangent, primal = autodiff(ForwardWithPrimal, broadcast_reversed_view, Duplicated(x, direction), Const(c))
    @test primal == value
    @test tangent == sum(direction .* gradient)
    dx = zero(x)
    _, primal = autodiff(ReverseWithPrimal, broadcast_reversed_view, Active, Duplicated(x, dx), Const(c))
    @test primal == value
    @test dx == gradient
end
