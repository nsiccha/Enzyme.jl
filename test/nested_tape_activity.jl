using Enzyme, Test

# The outer pass of a nested derivative differentiates the inner reverse pass, which
# includes the tape the inner pass builds: an aggregate of pointers and scalars that
# is active as a whole as soon as one member is, such as the shadow of an array. Its
# other members can be constant. Each constant member then needs a shadow, and the
# outer pass raised `EnzymeRuntimeActivityError` where the mixed-activity handler had
# none. The function below reads such a tape through Base functions on `Const` data:
# `dt` has no derivative, and neither does anything computed from it.

nta_copy_unique_sort(dt) = unique!(sort!(copy(dt)))
nta_copy_unique(dt) = unique!(copy(dt))
nta_copy_sort(dt) = sort!(copy(dt))
nta_unique_of_sort(dt) = unique(sort(dt))
nta_sort_of_unique(dt) = sort!(unique(dt))
nta_grouped(dt) = Base._groupedunique!(sort(dt))
function nta_push_unique_sort(dt)
    l = Float64[]
    for j in eachindex(dt)
        x = convert(Float64, dt[j])
        (isempty(l) || !isequal(x, last(l))) && push!(l, x)
    end
    return unique!(sort!(l))
end

@noinline function nta_objective(lens::F, x, dt) where {F}
    s = 0.0
    for v in lens(dt)
        s += x[2] * exp(-x[1] * v)
    end
    return s
end

function nta_gradient(lens::F, x, dt) where {F}
    dx = zero(x)
    autodiff(Reverse, Const(nta_objective), Active, Const(lens), Duplicated(x, dx), Const(dt))
    return dx
end

nta_direction(lens::F, x, v, dt) where {F} =
    only(autodiff(Forward, Const(nta_objective), Duplicated, Const(lens), Duplicated(x, v), Const(dt)))

# The Hessian-vector product of `nta_objective` needs only the values `lens` computes.
function nta_analytic_hvp(x, v, values)
    h = zeros(2)
    for u in values
        e = exp(-x[1] * u)
        h[1] += x[2] * u^2 * e * v[1] - u * e * v[2]
        h[2] += -u * e * v[1]
    end
    return h
end

nta_forward_over_reverse(lens, x, v, dt) =
    only(autodiff(Forward, Const(nta_gradient), Duplicated, Const(lens), Duplicated(copy(x), copy(v)), Const(dt)))

function nta_reverse_over_forward(lens, x, v, dt)
    hv = zero(x)
    autodiff(Reverse, Const(nta_direction), Active, Const(lens), Duplicated(copy(x), hv), Const(v), Const(dt))
    return hv
end

function nta_batch_forward_over_reverse(lens, x, v, dt)
    return only(
        autodiff(
            Forward, Const(nta_gradient), BatchDuplicated, Const(lens),
            BatchDuplicated(copy(x), (copy(v), 2 .* v)), Const(dt),
        ),
    )
end

const NTA_DT = [2.0, 0.5, 1.0, 1.0, 3.0]
const NTA_X = [0.3, 1.7]
const NTA_V = [1.0, -0.5]

const NTA_HELPERS = (
    nta_copy_unique_sort, nta_copy_unique, nta_copy_sort, nta_unique_of_sort,
    nta_sort_of_unique, nta_grouped, nta_push_unique_sort,
)

@testset "Const data on the tape of a nested derivative" begin
    dt0 = copy(NTA_DT)
    @testset "$(nameof(lens))" for lens in NTA_HELPERS
        want = nta_analytic_hvp(NTA_X, NTA_V, lens(NTA_DT))
        @test nta_forward_over_reverse(lens, NTA_X, NTA_V, NTA_DT) ≈ want rtol = 1.0e-10
        batch = nta_batch_forward_over_reverse(lens, NTA_X, NTA_V, NTA_DT)
        @test batch[1] ≈ want rtol = 1.0e-10
        @test batch[2] ≈ 2 .* want rtol = 1.0e-10
        @test nta_reverse_over_forward(lens, NTA_X, NTA_V, NTA_DT) ≈ want rtol = 1.0e-10
        @test NTA_DT == dt0
    end
end

# A constant member the reverse pass frees or reads is the primal's own shadow, so a
# shadow pass that freed it too or read it as a tangent would show up here.
@testset "Repeated nested derivatives keep the heap and the tangents intact" begin
    want = nta_analytic_hvp(NTA_X, NTA_V, nta_copy_unique_sort(NTA_DT))
    good = 0
    for i in 1:2000
        good += isapprox(nta_forward_over_reverse(nta_copy_unique_sort, NTA_X, NTA_V, NTA_DT), want; rtol = 1.0e-10)
        iszero(i % 200) && GC.gc(false)
    end
    @test good == 2000
end

# The proof that no reverse pass reads a member of a tape.
const NTA_LLVM = Enzyme.LLVM

function nta_tape_module(reverse_passes::String)
    return """
    define { i64, { i64, i64 } } @augmented_f(i64 %a, i64 %b, i64 %c) {
    top:
      %0 = insertvalue { i64, i64 } undef, i64 %b, 0
      %member = insertvalue { i64, i64 } %0, i64 %c, 1
      %1 = insertvalue { i64, { i64, i64 } } undef, i64 %a, 0
      %2 = insertvalue { i64, { i64, i64 } } %1, { i64, i64 } %member, 1
      ret { i64, { i64, i64 } } %2
    }

    define i64 @unrelated({ i64, { i64, i64 } } %tapeArg) {
    top:
      %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 1, 1
      ret i64 %0
    }

    declare void @use({ i64, i64 })

    $reverse_passes
    """
end

const NTA_READS_FIRST = """
define i64 @diffef({ i64, { i64, i64 } } %tapeArg) {
top:
  %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 0
  ret i64 %0
}
"""

const NTA_READS_NESTED = """
define i64 @diffef({ i64, { i64, i64 } } %tapeArg) {
top:
  %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 1, 0
  ret i64 %0
}
"""

const NTA_READS_WHOLE_MEMBER = """
define void @diffef({ i64, { i64, i64 } } %tapeArg) {
top:
  %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 1
  call void @use({ i64, i64 } %0)
  ret void
}
"""

const NTA_USES_WHOLE_TAPE = """
define i64 @diffef({ i64, { i64, i64 } } %tapeArg) {
top:
  %0 = call i64 @unrelated({ i64, { i64, i64 } } %tapeArg)
  ret i64 %0
}
"""

const NTA_OTHER_VARIANT_READS = NTA_READS_NESTED * """
    define i64 @diffef.1({ i64, { i64, i64 } } %tapeArg) {
    top:
      %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 1, 1
      ret i64 %0
    }
    """

const NTA_BATCH_READS = NTA_READS_NESTED * """
    define i64 @diffe2f({ i64, { i64, i64 } } %tapeArg) {
    top:
      %0 = extractvalue { i64, { i64, i64 } } %tapeArg, 1, 1
      ret i64 %0
    }
    """

const NTA_OTHER_TAPE_TYPE = NTA_READS_NESTED * """
    define i64 @diffef.1({ i64, i64 } %tapeArg) {
    top:
      %0 = extractvalue { i64, i64 } %tapeArg, 1
      ret i64 %0
    }
    """

# `tape_member_never_read` of `%member`, which reaches the returned tape at `[1, 1]`.
function nta_second_nested_member_unread(reverse_passes::String; function_name = nothing)
    return NTA_LLVM.Context() do ctx
        mod = parse(NTA_LLVM.Module, nta_tape_module(reverse_passes))
        fn = NTA_LLVM.functions(mod)["augmented_f"]
        function_name === nothing || NTA_LLVM.name!(fn, function_name)
        for inst in NTA_LLVM.instructions(first(NTA_LLVM.blocks(fn)))
            NTA_LLVM.name(inst) == "member" && return Enzyme.Compiler.tape_member_never_read(inst)
        end
        error("no such insertvalue")
    end
end

@testset "tape_member_never_read" begin
    @test nta_second_nested_member_unread(NTA_READS_FIRST)
    @test nta_second_nested_member_unread(NTA_READS_NESTED)
    @test nta_second_nested_member_unread(NTA_READS_NESTED; function_name = "preprocess_augmented_f")
    @test nta_second_nested_member_unread(NTA_OTHER_TAPE_TYPE)
    # The reverse pass reads that member, or the aggregate holding it.
    @test !nta_second_nested_member_unread(NTA_READS_WHOLE_MEMBER)
    @test !nta_second_nested_member_unread(NTA_USES_WHOLE_TAPE)
    # Which variant is which is not recorded, so one that reads it counts.
    @test !nta_second_nested_member_unread(NTA_OTHER_VARIANT_READS)
    @test !nta_second_nested_member_unread(NTA_BATCH_READS)
    # No reverse pass to look at, or not a tape.
    @test !nta_second_nested_member_unread("")
    @test !nta_second_nested_member_unread(NTA_READS_NESTED; function_name = "g")
end
