using Enzyme, Test

struct BoxedRecord{T}
    first::T
    second::Float64
end

record_score(x::BoxedRecord{Float64}) = x.first + 2x.second
record_score(x::BoxedRecord{Vector{Float64}}) = x.first[1] + 2x.second
score_twice(records, f) = (f[](records[1]) + f[](records[1]))::Float64
concrete_score(records) = record_score(records[1]::BoxedRecord{Float64})
scale_score(records, x, f) = x * (f[](records[1])::Float64)

@static if VERSION >= v"1.11-"
    @testset "Reverse of ordinary indexing into boxed storage" begin
        f = Const(Ref{Any}(record_score))
        for container in (Memory{BoxedRecord}, Vector{BoxedRecord})
            records = container(undef, 1)
            records[1] = BoxedRecord(3.0, 4.0)
            drecords = container(undef, 1)
            drecords[1] = BoxedRecord(1.0, 3.0)
            _, score = autodiff(ReverseWithPrimal, Const(score_twice), Active, Duplicated(records, drecords), f)
            @test score == 22.0
            @test drecords[1] == BoxedRecord(3.0, 7.0)

            drecords[1] = BoxedRecord(1.0, 3.0)
            _, score = autodiff(ReverseWithPrimal, Const(concrete_score), Active, Duplicated(records, drecords))
            @test score == 11.0
            @test drecords[1] == BoxedRecord(2.0, 5.0)

            drecords[1] = BoxedRecord(1.0, 3.0)
            dsecond = copy(drecords)
            dsecond[1] = BoxedRecord(4.0, -7.0)
            autodiff(Reverse, Const(score_twice), Active, BatchDuplicated(records, (drecords, dsecond)), f)
            @test drecords[1] == BoxedRecord(3.0, 7.0)
            @test dsecond[1] == BoxedRecord(6.0, -3.0)

            (partials, score) = autodiff(Enzyme.set_runtime_activity(ReverseWithPrimal), Const(scale_score), Active, Duplicated(records, records), Active(2.0), f)
            @test partials[2] == 11.0
            @test score == 22.0
            @test records[1] == BoxedRecord(3.0, 4.0)

            records[1] = BoxedRecord([3.0], 4.0)
            seed = [1.0]
            drecords[1] = BoxedRecord(seed, 3.0)
            _, score = autodiff(ReverseWithPrimal, Const(score_twice), Active, Duplicated(records, drecords), f)
            @test score == 22.0
            @test drecords[1].first === seed
            @test seed == [3.0]
            @test drecords[1].second == 7.0
        end

        records = BoxedRecord[BoxedRecord(9.0, 9.0), BoxedRecord(3.0, 4.0), BoxedRecord(5.0, 6.0)]
        drecords = BoxedRecord[BoxedRecord(8.0, 8.0), BoxedRecord(1.0, 3.0), BoxedRecord(7.0, 7.0)]
        popfirst!(records)
        popfirst!(drecords)
        _, score = autodiff(ReverseWithPrimal, Const(concrete_score), Active, Duplicated(records, drecords))
        @test score == 11.0
        @test drecords == [BoxedRecord(2.0, 5.0), BoxedRecord(7.0, 7.0)]
        @test_throws BoundsError autodiff(Reverse, Const(concrete_score), Active, Duplicated(BoxedRecord[], BoxedRecord[]))
    end
end
