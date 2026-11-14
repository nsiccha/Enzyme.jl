using Enzyme, Test

@noinline inline_roots_record(a) = (a, length(a))
@noinline inline_roots_pair(a) = (a, copy(a), length(a), length(a) + 1)
@noinline inline_roots_nested(a) =
    (length(a), ((a, 0), (a, 1)), ((a,), (a,)), ((1, 2), (3, 4)), false)

function inline_roots_loss(x, a)
    record = inline_roots_record(a)
    return x[1] * record[2] + sum(record[1])
end

function inline_roots_pair_loss(x, a)
    record = inline_roots_pair(a)
    return x[1] * record[3] + sum(record[1]) + sum(record[2]) * record[4]
end

function inline_roots_nested_loss(x, a)
    record = inline_roots_nested(a)
    return x[1] * (record[1] + record[2][1][2]) + sum(record[2][2][1]) + record[4][1][1]
end

@testset "InlineABI retains mixed aggregate return roots" begin
    # A retained helper returns both GC-tracked fields and scalar data. Integer
    # arrays are inactive on Julia 1.10, so the primal helper itself survives AD.
    for n in (1, 3, 5)
        a = collect(1:n)
        original = copy(a)
        x = [2.0]
        total = n * (n + 1) / 2
        for (f, value) in (
                (inline_roots_loss, 2n + total),
                (inline_roots_pair_loss, 2n + total * (n + 2)),
                (inline_roots_nested_loss, 2n + total + 1),
            )
            @test f(x, a) == value
            for abi in (FFIABI, InlineABI)
                forward = set_abi(ForwardWithPrimal, abi)
                @test autodiff(forward, f, Duplicated(x, [1.0]), Const(a)) == (Float64(n), value)
                batched = autodiff(forward, f, BatchDuplicated(x, ([1.0], [2.0])), Const(a))
                @test Tuple(first(batched)) == (Float64(n), 2.0n)
                @test last(batched) == value
                dx = zero(x)
                autodiff(set_abi(Reverse, abi), f, Active, Duplicated(x, dx), Const(a))
                @test dx == [Float64(n)]
                @test a == original
                @test x == [2.0]
            end
        end
    end
end
