using Enzyme, Test

# A branch selects a freshly allocated array with constant contents against an
# active array. Activity analysis proves the fresh array inactive, so its shadow
# is a zero array of the same shape rather than a runtime activity error.

@noinline consume_branch(v) = sum(abs2, v)

branch_literal(x, e) = sum(e ? Float64[] : x .* 2.0)
branch_literal_call(x, e) = consume_branch(e ? Float64[] : x .* 2.0)
branch_zeros0(x, e) = sum(e ? zeros(0) : x .* 2.0)
branch_zeros(x, e) = sum(e ? zeros(length(x)) : x .* 2.0)
branch_ones(x, e) = sum(e ? ones(length(x)) : x .* 2.0)
branch_fill(x, e) = sum(e ? fill(3.0, length(x)) : x .* 2.0)
branch_undef(x, e) = sum(e ? fill!(Vector{Float64}(undef, length(x)), 0.5) : x .* 2.0)
branch_similar0(x, e) = sum(e ? similar(x, 0) : x .* 2.0)
branch_matrix(x, e) = sum(e ? zeros(length(x), length(x)) : x .* x')
branch_complex(x, e) = sum(abs2, e ? zeros(ComplexF64, length(x)) : complex.(x, x))
branch_then_add(x, e) = sum(abs2, (e ? zeros(length(x)) : x .* 2.0) .+ x)
branch_return(x, e) = e ? zeros(length(x)) : x .* 2.0

@testset "Branch selecting a fresh constant array" begin
    x0 = [1.0, 2.0]
    cases = (
        (branch_literal, [0.0, 0.0], [2.0, 2.0]),
        (branch_literal_call, [0.0, 0.0], 8.0 .* x0),
        (branch_zeros0, [0.0, 0.0], [2.0, 2.0]),
        (branch_zeros, [0.0, 0.0], [2.0, 2.0]),
        (branch_ones, [0.0, 0.0], [2.0, 2.0]),
        (branch_fill, [0.0, 0.0], [2.0, 2.0]),
        (branch_undef, [0.0, 0.0], [2.0, 2.0]),
        (branch_similar0, [0.0, 0.0], [2.0, 2.0]),
        (branch_matrix, [0.0, 0.0], fill(sum(x0), 2) .* 2.0),
        (branch_complex, [0.0, 0.0], 4.0 .* x0),
        (branch_then_add, 2.0 .* x0, 18.0 .* x0),
    )
    for (f, dtrue, dfalse) in cases, e in (true, false)
        x = copy(x0)
        dx = zero(x)
        res = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(e))
        @test res[2] ≈ f(x0, e)
        @test dx ≈ (e ? dtrue : dfalse)
        @test x == x0
    end

    for e in (true, false)
        expected = e ? 0.0 : 2.0
        @test autodiff(Forward, branch_zeros, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1] ≈ expected
        res = autodiff(ForwardWithPrimal, branch_zeros, Duplicated(copy(x0), [1.0, 0.0]), Const(e))
        @test res[1] ≈ expected
        @test res[2] ≈ branch_zeros(x0, e)

        bres = autodiff(
            Forward, branch_zeros, BatchDuplicated(copy(x0), ([1.0, 0.0], [0.0, 1.0])), Const(e)
        )[1]
        @test bres[1] ≈ expected
        @test bres[2] ≈ expected

        d1 = zero(x0)
        d2 = zero(x0)
        autodiff(Reverse, branch_zeros, Active, BatchDuplicated(copy(x0), (d1, d2)), Const(e))
        @test d1 ≈ fill(expected, 2)
        @test d2 ≈ fill(expected, 2)

        dx = zero(x0)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal,
            Const{typeof(branch_zeros)},
            Active,
            Duplicated{Vector{Float64}},
            Const{Bool},
        )
        tape, primal, _ = fwd(Const(branch_zeros), Duplicated(copy(x0), dx), Const(e))
        rev(Const(branch_zeros), Duplicated(copy(x0), dx), Const(e), 1.0, tape)
        @test primal ≈ branch_zeros(x0, e)
        @test dx ≈ fill(expected, 2)

        dret = autodiff(Forward, branch_return, Duplicated, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1]
        @test dret ≈ (e ? [0.0, 0.0] : [2.0, 0.0])
    end
end

# Active data reaching the fresh array through an alias makes it active, so these
# never take the zero-shadow path; their gradients must stay exact.
alias_store_through_branch(x, e) =
    (c = zeros(length(x)); v = e ? c : similar(x); v .= x .* 2.0; sum(c))
alias_store_after_branch(x, e) = (c = zeros(length(x)); v = e ? c : x .* 1.0; c .= x; sum(v))
alias_two_branches(x, e) =
    (c = zeros(length(x)); v = e ? c : x .* 2.0; w = e ? c : x .* 3.0; v[1] = x[1]; sum(w))
alias_push(x, e) = (v = e ? Float64[] : x .* 2.0; push!(v, x[1]); sum(v))
alias_vec(x, e) = (m = zeros(length(x), 1); v = e ? vec(m) : x .* 2.0; m[2, 1] = x[1]^2; sum(v))

@testset "Fresh constant array aliased with active data" begin
    x0 = [1.0, 2.0]
    cases = (
        (alias_store_through_branch, [2.0, 2.0], [0.0, 0.0]),
        (alias_store_after_branch, [1.0, 1.0], [1.0, 1.0]),
        (alias_two_branches, [1.0, 0.0], [3.0, 3.0]),
        (alias_push, [1.0, 0.0], [3.0, 2.0]),
        (alias_vec, [2.0, 0.0], [2.0, 2.0]),
    )
    for (f, dtrue, dfalse) in cases, e in (true, false)
        x = copy(x0)
        dx = zero(x)
        autodiff(Reverse, f, Active, Duplicated(x, dx), Const(e))
        @test dx ≈ (e ? dtrue : dfalse)
    end
end

const BRANCH_GLOBAL = [1.0, 2.0]
branch_const_arg(x, c, e) = sum(e ? c : x .* 2.0)
branch_global(x, e) = sum(e ? BRANCH_GLOBAL : x .* 2.0)

@testset "Constant memory that is not a fresh allocation" begin
    x0 = [1.0, 2.0]
    # refused: a Const argument or a global may serve as storage for active data,
    # which static activity analysis cannot track; correctness requires runtime
    # activity (Enzyme FAQ, "Runtime Activity").
    @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError autodiff(
        Reverse, branch_const_arg, Active, Duplicated(copy(x0), zero(x0)), Const([5.0, 6.0]), Const(true)
    )
    @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError autodiff(
        Reverse, branch_global, Active, Duplicated(copy(x0), zero(x0)), Const(true)
    )
    dx = zero(x0)
    autodiff(
        set_runtime_activity(Reverse), branch_const_arg, Active,
        Duplicated(copy(x0), dx), Const([5.0, 6.0]), Const(true),
    )
    @test dx ≈ [0.0, 0.0]
end
