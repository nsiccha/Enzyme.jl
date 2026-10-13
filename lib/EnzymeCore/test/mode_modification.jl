using EnzymeCore
using EnzymeCore: InlineABI, ReverseMode, ReverseModeSplit, Split, Combined, set_runtime_activity, set_err_if_func_written, clear_err_if_func_written, set_strong_zero, set_abi, set_atomic_add, atomic_add, WithPrimal, NoPrimal
using Test

@testset "Split / unsplit mode" begin
    @test Split(Reverse) == ReverseSplitNoPrimal
    @test Split(ReverseWithPrimal) == ReverseSplitWithPrimal
    @test Split(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test Split(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test Split(set_runtime_activity(Reverse)) == set_runtime_activity(ReverseSplitNoPrimal)
    @test Split(set_err_if_func_written(Reverse)) == set_err_if_func_written(ReverseSplitNoPrimal)
    @test Split(set_abi(Reverse, InlineABI)) == set_abi(ReverseSplitNoPrimal, InlineABI)

    @test Split(Reverse, Val(:ReturnShadow), Val(:Width), Val(:ModifiedBetween), Val(:ShadowInit)) == ReverseModeSplit{false,:ReturnShadow,false,false,:Width,:ModifiedBetween,EnzymeCore.DefaultABI,false,false,:ShadowInit}()

    @test Combined(Reverse) == Reverse
    @test Combined(ReverseWithPrimal) == ReverseWithPrimal
    @test Combined(ReverseSplitNoPrimal) == Reverse
    @test Combined(ReverseSplitWithPrimal) == ReverseWithPrimal

    @test Combined(set_runtime_activity(ReverseSplitNoPrimal)) == set_runtime_activity(Reverse)
    @test Combined(set_err_if_func_written(ReverseSplitNoPrimal)) == set_err_if_func_written(Reverse)
    @test Combined(set_abi(ReverseSplitNoPrimal, InlineABI)) == set_abi(Reverse, InlineABI)
end

@testset "Atomic accumulation setting" begin
    @test atomic_add(Reverse) === nothing
    @test atomic_add(ReverseSplitNoPrimal) === nothing
    for setting in (true, false, nothing)
        mode = set_atomic_add(Reverse, setting)
        @test atomic_add(mode) === setting
        @test atomic_add(typeof(mode)) === setting
        @test atomic_add(set_runtime_activity(mode)) === setting
        @test atomic_add(set_strong_zero(mode)) === setting
        @test atomic_add(set_err_if_func_written(mode)) === setting
        @test atomic_add(set_abi(mode, InlineABI)) === setting
        @test atomic_add(WithPrimal(mode)) === setting
        @test atomic_add(Split(mode)) === setting
        @test Combined(Split(mode)) == mode

        split = set_atomic_add(ReverseSplitWithPrimal, setting)
        @test atomic_add(split) === setting
        @test atomic_add(NoPrimal(split)) === setting
        @test atomic_add(clear_err_if_func_written(set_err_if_func_written(split))) === setting
        @test atomic_add(Combined(split)) === setting
    end
    # Modes spelled without the setting, as code written before it existed does, leave it to Enzyme
    @test ReverseMode{false, false, false, EnzymeCore.DefaultABI, false, false}() === Reverse
    @test ReverseModeSplit{false, true, false, false, 0, true, EnzymeCore.DefaultABI, false, false, false}() === ReverseSplitNoPrimal
end
