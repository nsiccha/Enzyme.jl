using Enzyme, Test

# A fresh cache is essential: inferring the inner reduction first hides the
# loss of keyword argument types during mutual recursion on Julia 1.10.
function recursion_fresh_interp()
    cache = @static if VERSION >= v"1.11.0-DEV.1552"
        gensym(:inference_recursion)
    else
        Enzyme.Compiler.GPUCompiler.CodeCache()
    end
    return Enzyme.Compiler.Interpreter.EnzymeInterpreter(
        cache, nothing, Base.get_world_counter(), false, false, false,
    )
end

recursion_nested_sum(xs) = sum(sum, xs)
recursion_nested_prod(xs) = prod(prod, xs)
recursion_sum_init(x) = sum(x; init = 1.0)
recursion_nested_init(xs) = sum(recursion_sum_init, xs; init = 2.0)
recursion_sum_loss(xs, w) = sum(w .* recursion_nested_sum(xs))

@testset "Nested reductions retain concrete keyword arguments" begin
    xs = [[1.0, 2.0], [3.0, 4.0]]
    for f in (recursion_nested_sum, recursion_nested_prod, recursion_nested_init)
        interp = recursion_fresh_interp()
        @test only(Base.code_typed(f, (typeof(xs),); interp, optimize = false))[2] === Float64
    end

    # The scalar result also feeds broadcast shape/type inference in consumers.
    w = [2.0, 3.0]
    dxs = Enzyme.make_zero(xs)
    result = autodiff(ReverseWithPrimal, recursion_sum_loss, Active, Duplicated(xs, dxs), Const(w))
    @test result[2] == 50.0
    @test dxs == [[5.0, 5.0], [5.0, 5.0]]
    @test xs == [[1.0, 2.0], [3.0, 4.0]]
    @test w == [2.0, 3.0]
end

struct RecursionOuterCell
    scale::Float64
end
struct RecursionInnerCell
    scale::Float64
end

function recursion_cell(f, args...)
    T = Base.promote_op(f, map(typeof, args)...)
    return zero(T) + f(args...)
end

(f::RecursionInnerCell)(x, y, z) = f.scale * (x + y + z)
(f::RecursionOuterCell)(x, y) = x > 0 ? recursion_cell(RecursionInnerCell(f.scale), x, x, y) : 0.0
recursion_cell_loss(scale, x, y) = recursion_cell(RecursionOuterCell(scale), x, y)

@testset "Nested result-type queries for different callables" begin
    interp = recursion_fresh_interp()
    @test only(Base.code_typed(recursion_cell_loss, (Float64, Float64, Float64); interp, optimize = false))[2] === Float64

    report = Enzyme.derivative_report() do
        autodiff(ReverseWithPrimal, recursion_cell_loss, Active, Active(0.5), Const(2.0), Const(3.0))
    end
    @test report.value == ((7.0, nothing, nothing), 3.5)
    findings = only(report.derivatives).findings
    @test !any(f -> !f.throw_only && f.kind in (:dynamic_dispatch, :enzyme_runtime_fallback), findings)
end

recursion_fixed(n) = n <= 0 ? 1 : recursion_fixed(n - 1) + 1
recursion_growing_type(x, n) = n <= 0 ? length(x) : recursion_type_step((x,), n - 1)
recursion_type_step(x, n) = recursion_growing_type(x, n)
recursion_growing_value(::Val{N}, n) where {N} = n <= 0 ? N : recursion_value_step(Val(N + 1), n - 1)
recursion_value_step(x, n) = recursion_growing_value(x, n)
recursion_other_cell(f::RecursionOuterCell) = RecursionInnerCell(f.scale)
recursion_other_cell(f::RecursionInnerCell) = RecursionOuterCell(f.scale)
function recursion_growing_arity(f, n, args...)
    return n <= 0 ? length(args) : recursion_growing_arity(recursion_other_cell(f), n - 1, args..., n)
end

@testset "Recursive inference still converges" begin
    interp = recursion_fresh_interp()
    @test only(Base.code_typed(recursion_fixed, (Int,); interp, optimize = false))[2] === Int
    # Growing signatures may be widened to Any. The important property here
    # is that inference terminates, rather than generating unbounded frames.
    for (f, types) in (
            (recursion_growing_type, (Tuple{Int}, Int)),
            (recursion_growing_value, (Val{1}, Int)),
            (recursion_growing_arity, (RecursionOuterCell, Int)),
        )
        interp = recursion_fresh_interp()
        @test only(Base.code_typed(f, types; interp, optimize = false))[2] in (Int, Any)
    end
    @test recursion_fixed(3) == 4
    @test recursion_growing_type((1,), 3) == 1
    @test recursion_growing_value(Val(1), 3) == 4
    @test recursion_growing_arity(RecursionOuterCell(1.0), 3) == 3
end
