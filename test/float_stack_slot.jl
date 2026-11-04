using Enzyme, LLVM, Test

# An `NTuple{N, Float64}` that Julia keeps in a stack slot, because the loop reads it at a
# runtime index, holds floats at every offset. Type analysis could not see that: the
# read types only the elements its index is known to take (a loop variable with an
# unknown trip count: only its start), and an element the function updates under a branch
# is otherwise only ever stored, reloaded and merged with itself. The phi merging the
# update, a float, with that reload then raised `EnzymeNoTypeError` ("Cannot deduce type
# of phi"), once LLVM had a reason to keep the reload (BRM's block-level adaptive-centering
# pass at five or more margins; snag reverse-untyped-a117a92b).
#
# LLVM forwards the store to the reload in functions small enough to write down here, so
# these functions write the IR directly and keep the reload with `volatile`.
slot_ptr(T) = VERSION >= v"1.12" ? "ptr" : "$T*"

function slot_ir(fp)
    slot = "[3 x $fp]"
    elt = slot_ptr(fp)
    arr = slot_ptr(slot)
    zero = "0.0"
    two = "2.0"
    return """
    define $fp @f($fp %x, i64 %n, i64 %k) alwaysinline {
    top:
      %slot = alloca $slot, align 8
      %e0 = getelementptr inbounds $slot, $arr %slot, i64 0, i64 0
      %e1 = getelementptr inbounds $slot, $arr %slot, i64 0, i64 1
      %e2 = getelementptr inbounds $slot, $arr %slot, i64 0, i64 2
      br label %head
    head:
      %j = phi i64 [ 0, %top ], [ %j1, %merge ]
      %s0 = phi $fp [ $zero, %top ], [ %m0, %merge ]
      %s1 = phi $fp [ $zero, %top ], [ %m1, %merge ]
      %s2 = phi $fp [ $zero, %top ], [ %m2, %merge ]
      %acc = phi $fp [ $zero, %top ], [ %acc2, %merge ]
      store $fp %s0, $elt %e0
      store $fp %s1, $elt %e1
      store $fp %s2, $elt %e2
      %guard = icmp sgt i64 %j, 0
      br i1 %guard, label %read, label %afterread
    read:
      %l = phi i64 [ 0, %head ], [ %l1, %read ]
      %m = phi $fp [ $zero, %head ], [ %m3, %read ]
      %q = getelementptr inbounds $slot, $arr %slot, i64 0, i64 %l
      %u = load $fp, $elt %q
      %uu = fmul $fp %u, %x
      %m3 = fadd $fp %m, %uu
      %l1 = add nuw nsw i64 %l, 1
      %cont = icmp slt i64 %l1, %j
      br i1 %cont, label %read, label %afterread
    afterread:
      %mm = phi $fp [ $zero, %head ], [ %m3, %read ]
      %acc2 = fadd $fp %acc, %mm
      %j1 = add nuw nsw i64 %j, 1
      %upd = icmp slt i64 %j1, %k
      br i1 %upd, label %update, label %keep
    update:
      %v = fmul $fp %x, $two
      %is0 = icmp eq i64 %j, 0
      %is1 = icmp eq i64 %j, 1
      %is2 = icmp eq i64 %j, 2
      %t0 = select i1 %is0, $fp %v, $fp %s0
      %t1 = select i1 %is1, $fp %v, $fp %s1
      %t2 = select i1 %is2, $fp %v, $fp %s2
      br label %merge
    keep:
      %r0 = load volatile $fp, $elt %e0
      %r1 = load volatile $fp, $elt %e1
      %r2 = load volatile $fp, $elt %e2
      br label %merge
    merge:
      %m0 = phi $fp [ %t0, %update ], [ %r0, %keep ]
      %m1 = phi $fp [ %t1, %update ], [ %r1, %keep ]
      %m2 = phi $fp [ %t2, %update ], [ %r2, %keep ]
      %more = icmp slt i64 %j1, %n
      br i1 %more, label %head, label %exit
    exit:
      ret $fp %acc2
    }
    """
end

const SLOT_IR_DOUBLE = slot_ir("double")
const SLOT_IR_FLOAT = slot_ir("float")
slot_loop(x::Float64, n::Int, k::Int) =
    Base.llvmcall((SLOT_IR_DOUBLE, "f"), Float64, Tuple{Float64, Int, Int}, x, n, k)
slot_loop(x::Float32, n::Int, k::Int) =
    Base.llvmcall((SLOT_IR_FLOAT, "f"), Float32, Tuple{Float32, Int, Int}, x, n, k)

# The same arithmetic without a slot: the elements a loop of `n` steps has written before
# step `j`, and their sum times `x`.
function slot_loop_plain(x, n, k)
    t = (zero(x), zero(x), zero(x))
    acc = zero(x)
    for j in 0:n-1
        for l in 1:j
            acc += t[l] * x
        end
        if j + 1 < k
            t = Base.setindex(t, 2x, j + 1)
        end
    end
    return acc
end

@testset "a float merged with a reload of its own stack slot element" begin
    for x in (1.5, 1.5f0), (n, k) in ((4, 3), (3, 3), (4, 4), (2, 2), (4, 2))
        @test slot_loop(x, n, k) == slot_loop_plain(x, n, k)
        expected = 2 * slot_loop_plain(x, n, k) / x
        @test autodiff(Reverse, slot_loop, Active, Active(x), Const(n), Const(k))[1][1] ≈ expected
        @test autodiff(set_runtime_activity(Reverse), slot_loop, Active, Active(x), Const(n), Const(k))[1][1] ≈ expected
        @test only(autodiff(Forward, slot_loop, Duplicated(x, one(x)), Const(n), Const(k))) ≈ expected
    end
end

# The Julia shape that reaches that IR in a larger function, at the lengths that keep the
# slot. It differentiated before the repair too, and pins that the slot's new type does
# not disturb the ordinary case: one update under a branch, and one on every step.
function tuple_conditional(x, ::Val{N}, K, G) where {N}
    acc = zero(eltype(x))
    for g in 1:G
        t = ntuple(_ -> zero(eltype(x)), Val(N))
        for j in 1:K
            m = zero(eltype(x))
            for l in 1:(j - 1)
                m += x[l] * t[l]
            end
            u = x[j + g]
            j < K && (t = Base.setindex(t, (u - m) / exp(x[j]), j))
            acc += log(exp(x[j])) + m * u
        end
    end
    return acc
end

function tuple_unconditional(x, ::Val{N}, K, G) where {N}
    acc = zero(eltype(x))
    for g in 1:G
        t = ntuple(_ -> zero(eltype(x)), Val(N + 1))
        for j in 1:K
            m = zero(eltype(x))
            for l in 1:(j - 1)
                m += x[l] * t[l]
            end
            u = x[j + g]
            t = Base.setindex(t, (u - m) / exp(x[j]), j)
            acc += log(exp(x[j])) + m * u
        end
    end
    return acc
end

function slot_central(f, x; h = 1.0e-6)
    y = copy(x)
    return map(eachindex(x)) do i
        y[i] = x[i] + h
        up = f(y)
        y[i] = x[i] - h
        down = f(y)
        y[i] = x[i]
        (up - down) / 2h
    end
end

@testset "a tuple read at a runtime index, $(nameof(f)), N = $N" for f in (tuple_conditional, tuple_unconditional), N in (2, 3, 4, 6, 8)
    K, G = N + 1, 3
    x = collect(range(-0.4, 0.7; length = K + G + 1))
    dx = zero(x)
    autodiff(Reverse, f, Active, Duplicated(x, dx), Const(Val(N)), Const(K), Const(G))
    @test dx ≈ slot_central(y -> f(y, Val(N), K, G), x) atol = 1.0e-6
end

@testset "which stack slots hold only floats" begin
    aggregate = Enzyme.Compiler.floating_point_aggregate_type
    API = Enzyme.API
    LLVM.Context() do _
        d = LLVM.DoubleType()
        f = LLVM.FloatType()
        i = LLVM.Int64Type()
        @test aggregate(LLVM.ArrayType(d, 6)) == API.DT_Double
        @test aggregate(LLVM.ArrayType(f, 6)) == API.DT_Float
        @test aggregate(LLVM.ArrayType(LLVM.ArrayType(d, 2), 3)) == API.DT_Double
        @test aggregate(LLVM.StructType(LLVM.LLVMType[LLVM.ArrayType(d, 2), d])) == API.DT_Double
        # A scalar is not an aggregate, and anything else in the slot keeps it untyped.
        @test aggregate(d) === nothing
        @test aggregate(i) === nothing
        @test aggregate(LLVM.ArrayType(i, 6)) === nothing
        @test aggregate(LLVM.StructType(LLVM.LLVMType[d, i])) === nothing
        @test aggregate(LLVM.StructType(LLVM.LLVMType[d, f])) === nothing
        @test aggregate(LLVM.StructType(LLVM.LLVMType[LLVM.ArrayType(d, 2), LLVM.ArrayType(i, 2)])) === nothing
        @test aggregate(LLVM.StructType(LLVM.LLVMType[])) === nothing
    end
end
