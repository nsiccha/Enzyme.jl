using Enzyme, Test
using Enzyme.Compiler: LLVM

union_standardize(x, scale) = x / (scale == 0 && x == 0 ? 1.0 : scale)

function union_retained_density(x, scale, box)
    terms = similar(x)
    for i in eachindex(x)
        z = union_standardize(x[i], scale)
        terms[i] = -0.5z^2
    end
    box[] = terms
    return sum(terms)
end

@testset "Conditional integer/float denominator" begin
    for scale in (1, 1.0, -2, 3), x in (0.0, 0.2)
        @test autodiff(Reverse, union_standardize, Active, Active(x), Const(scale))[1][1] ≈ 1 / scale
        derivative, primal = autodiff(ReverseWithPrimal, union_standardize, Active, Active(x), Const(scale))
        @test derivative[1] ≈ 1 / scale
        @test primal ≈ x / scale
        @test autodiff(Forward, union_standardize, Duplicated(x, 1.0), Const(scale))[1] ≈ 1 / scale
    end
    @test autodiff(Reverse, union_standardize, Active, Active(0.0), Const(0))[1][1] == 1.0
    for scale in (1, 1.0, -2, 3)
        x = [0.1, 0.2, 0.3]
        dx = zero(x)
        box, dbox = Ref(zero(x)), Ref(zero(x))
        primal = autodiff(ReverseWithPrimal, union_retained_density, Active, Duplicated(x, dx), Const(scale), Duplicated(box, dbox))[2]
        @test primal ≈ -sum(abs2, x) / (2scale^2)
        @test dx ≈ -x ./ scale^2
        @test box[] ≈ -0.5 .* (x ./ scale) .^ 2
        @test x == [0.1, 0.2, 0.3]
    end
end

Base.@noinline union_choose_mean(active, a, b) = ifelse(active, a, b)
Base.@noinline union_split_mean(z, x, active) = 0.3 .+ union_choose_mean(active, z[1], z[2] .* x)
union_loss1(z, x, y, active) = -sum(abs2, y .- (0.3 .+ union_choose_mean(active, z[1], z[2] .* x))) / 2
union_loss2(z, x, y, active) = -sum(abs2, y .- union_split_mean(z, x, active)) / 2

function union_loss3(z, x, y, active)
    result = 0.0
    for i in eachindex(active)
        mu = 0.3 .+ ifelse(active[i], z[1], z[2] .* x[i])
        result -= sum(abs2, y[i] .- mu) / 2
    end
    return result
end

function union_loss4(z, x, y, active)
    result = 0.0
    for i in eachindex(active)
        mu = union_split_mean(z, x[i], active[i])
        result -= sum(abs2, y[i] .- mu) / 2
    end
    return result
end

@testset "Boxed scalar/vector union" begin
    for f in (union_loss1, union_loss2, union_loss3, union_loss4), active in (false, true)
        z = [0.2, -0.4]
        dz = zero(z)
        x, y = [-0.7, 0.2, 0.9], [0.1, -0.2, 0.4]
        args = f in (union_loss1, union_loss2) ? (x, y, active) : ([x], [y], BitVector([active]))
        primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(z, dz), Const.(args)...)[2]
        @test primal ≈ f(z, args...)
        @test dz ≈ (active ? [-1.2, 0.0] : [0.0, 0.666])
        @test z == [0.2, -0.4]
    end
    for f in (union_loss3, union_loss4)
        x = [[-0.7, 0.2, 0.9], [0.4], Float64[], [0.5, -0.2]]
        y = [[0.1, -0.2, 0.4], [0.7], Float64[], [-0.3, 0.8]]
        active = BitVector([false, true, true, false])
        z, dz = [0.2, -0.4], zeros(2)
        primal = autodiff(ReverseWithPrimal, f, Active, Duplicated(z, dz), Const(x), Const(y), Const(active))[2]
        @test primal ≈ f(z, x, y, active)
        expected = zeros(2)
        for i in eachindex(active)
            if active[i]
                expected[1] += sum(y[i] .- (0.3 + z[1]))
            else
                expected[2] += sum((y[i] .- (0.3 .+ z[2] .* x[i])) .* x[i])
            end
        end
        @test dz ≈ expected
    end
end

function conditioned_tag_fixture(; linkage = "internal", second = "-128", third = "1", chain = 0)
    extra = IOBuffer()
    source = "%tag"
    previous = "%exit"
    for i in 1:chain
        println(extra, "br label %chainblock$i\nchainblock$i:")
        println(extra, "%chain$i = phi i8 [ $source, $previous ]")
        source = "%chain$i"
        previous = "%chainblock$i"
    end
    return parse(
        LLVM.Module, """
        define $linkage { i64, i8 } @tagged(i1 %choice, i8 %unknown) {
        entry:
            %zero = zext i1 false to i8
            %high = or i8 %zero, $second
            %other = select i1 %choice, i8 %high, i8 $third
            br label %loop
        loop:
            %tag = phi i8 [ %other, %entry ], [ %again, %loop ]
            %again = select i1 %choice, i8 1, i8 %tag
            br i1 %choice, label %exit, label %loop
        exit:
            $(String(take!(extra)))
            %result = insertvalue { i64, i8 } undef, i8 $source, 1
            ret { i64, i8 } %result
        }
        define i64 @test(i1 %choice, i64 %a, i64 %b, i8 %unknown) {
        entry:
            %result = call { i64, i8 } @tagged(i1 %choice, i8 %unknown)
            %tag = extractvalue { i64, i8 } %result, 1
            %boxed = icmp slt i8 %tag, 0
            %payload = select i1 %boxed, i64 %b, i64 %a
            %masked = and i8 %tag, 127
            %isfloat = icmp eq i8 %masked, 1
            br i1 %isfloat, label %float, label %other
        float:
            ret i64 %payload
        other:
            ret i64 %payload
        }
        """
    )
end

function conditioned_returns(mod)
    result = LLVM.Value[]
    for bb in LLVM.blocks(LLVM.functions(mod)["test"])
        inst = LLVM.terminator(bb)
        inst isa LLVM.RetInst && push!(result, LLVM.operands(inst)[1])
    end
    return result
end

@testset "Conditioned selects preserve unproven uses" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            declare void @retain(i64)
            define double @test(i1 %condition, i1 %other, i64 %integer) {
            entry:
                %payload = select i1 %condition, i64 %integer, i64 4607182418800017408
                call void @retain(i64 %payload)
                %bits = bitcast i64 %payload to double
                %number = sitofp i64 %payload to double
                %known = select i1 %condition, double %number, double %bits
                %unknown = select i1 %other, double %number, double %bits
                %result = fadd double %known, %unknown
                ret double %result
            }
            """
        )
        @test Enzyme.Compiler.simplify_conditioned_selects!(mod) == 2
        LLVM.verify(mod)
        instructions = Dict(LLVM.name(inst) => inst for bb in LLVM.blocks(LLVM.functions(mod)["test"]) for inst in LLVM.instructions(bb))
        payload = instructions["payload"]
        @test LLVM.operands(instructions["bits"])[1] == payload
        @test LLVM.operands(instructions["number"])[1] == payload
        @test LLVM.operands(LLVM.operands(instructions["known"])[2])[1] == LLVM.parameters(LLVM.functions(mod)["test"])[3]
        @test LLVM.operands(LLVM.operands(instructions["known"])[3])[1] isa LLVM.ConstantInt

        # A finite domain through a call, aggregate and cyclic PHI proves both
        # arms. An interposable definition, unknown value or exhausted search
        # must not justify substituting either pointer layout.
        mod = conditioned_tag_fixture()
        @test Enzyme.Compiler.simplify_conditioned_selects!(mod) == 2
        @test conditioned_returns(mod) == collect(LLVM.parameters(LLVM.functions(mod)["test"]))[2:3]
        LLVM.verify(mod)
        for options in ((; linkage = "weak"), (; third = "%unknown"), (; third = "undef"), (; chain = 300))
            mod = conditioned_tag_fixture(; options...)
            @test Enzyme.Compiler.simplify_conditioned_selects!(mod) == 0
            @test all(isa.(conditioned_returns(mod), LLVM.SelectInst))
            LLVM.verify(mod)
        end
        # Tag 129 shares the masked test with 1 but requires the other layout.
        mod = conditioned_tag_fixture(; third = "-127")
        @test Enzyme.Compiler.simplify_conditioned_selects!(mod) == 1
        @test first(conditioned_returns(mod)) isa LLVM.SelectInst
        LLVM.verify(mod)

        # The successor is also reachable without taking the condition's edge.
        mod = parse(
            LLVM.Module, """
            define i64 @test(i1 %condition, i64 %a, i64 %b) {
            entry:
                %value = select i1 %condition, i64 %a, i64 %b
                br i1 %condition, label %join, label %other
            other:
                br label %join
            join:
                ret i64 %value
            }
            """
        )
        @test Enzyme.Compiler.simplify_conditioned_selects!(mod) == 0
        LLVM.verify(mod)

        # The masked test excludes tag 1, so the second test cannot take its
        # boxing arm. Leaving that arm in the IR merges a boxed float into an
        # array pointer PHI on Julia 1.12, even though the primal never takes it.
        mod = parse(
            LLVM.Module, """
            declare void @unreachable_box()
            define i64 @test(i1 %choice) {
            entry:
                %tag = select i1 %choice, i8 1, i8 -128
                %masked = and i8 %tag, 127
                %isfloat = icmp eq i8 %masked, 1
                br i1 %isfloat, label %float, label %other
            float:
                ret i64 3
            other:
                %box = icmp eq i8 %tag, 1
                br i1 %box, label %impossible, label %array
            impossible:
                call void @unreachable_box()
                ret i64 0
            array:
                ret i64 7
            }
            """
        )
        @test Enzyme.Compiler.simplify_conditioned_selects!(mod) > 0
        @test all(!(inst isa LLVM.CallInst) for bb in LLVM.blocks(LLVM.functions(mod)["test"]) for inst in LLVM.instructions(bb))
        LLVM.verify(mod)
    end
end
