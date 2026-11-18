using Enzyme, LLVM, Test

# `split_constant_merges!` splits each pointer select and phi it collected, in block
# order. Splitting a select moves the rest of its block to a new block and rebuilds
# the phis of the blocks that follow it (`retarget_phis!`), so a phi collected there
# was erased before its turn came. Splitting that stale phi read freed memory: a
# segfault in `merge_load_tree`, or `UndefRefError` from `LLVM.parent` in
# `merge_predecessors_branch`, while compiling the Reverse gradient of a Soss GP
# posterior (snag dogfood-constant-886748d1). Each function below has a select whose
# block flows into a phi that is split too.
function merge_order_ir(typed::Bool)
    obj = typed ? "{} addrspace(10)*" : "ptr addrspace(10)"
    data = typed ? "double addrspace(13)*" : "ptr addrspace(13)"
    slot = typed ? "double addrspace(13)* addrspace(11)*" : "ptr addrspace(11)"
    loads = """
      %f = call $obj @fresh()
      %fs = addrspacecast $obj %f to $slot
      %fd = load $data, $slot %fs, align 8
      %as = addrspacecast $obj %a to $slot
      %ad = load $data, $slot %as, align 8
    """
    return """
    source_filename = "start"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
    target triple = "x86_64-linux-gnu"

    declare noalias nonnull $obj @fresh()

    ; The block of the select ends in a branch into the block of the phi.
    define double @select_into_phi(i1 %c, i1 %d, $obj %a) {
    top:
    $loads
      br i1 %d, label %pick, label %merge

    pick:
      %s = select i1 %c, $data %fd, $data %ad
      %u = load double, $data %s, align 8
      br label %merge

    merge:
      %p = phi $data [ %fd, %top ], [ %ad, %pick ]
      %w = phi double [ 0.0, %top ], [ %u, %pick ]
      %v = load double, $data %p, align 8
      %r = fadd double %v, %w
      ret double %r
    }

    ; The same, with the phi collected before the select: block order puts the
    ; merge first.
    define double @phi_before_select(i1 %c, i1 %d, $obj %a) {
    top:
    $loads
      br i1 %d, label %pick, label %merge

    merge:
      %p = phi $data [ %fd, %top ], [ %ad, %pick ]
      %w = phi double [ 0.0, %top ], [ %u, %pick ]
      %v = load double, $data %p, align 8
      %r = fadd double %v, %w
      ret double %r

    pick:
      %s = select i1 %c, $data %fd, $data %ad
      %u = load double, $data %s, align 8
      br label %merge
    }

    ; Two selects in the block, and two phis of pointers after it.
    define double @two_selects_two_phis(i1 %c, i1 %e, i1 %d, $obj %a) {
    top:
    $loads
      br i1 %d, label %pick, label %merge

    pick:
      %s = select i1 %c, $data %fd, $data %ad
      %u = load double, $data %s, align 8
      %t = select i1 %e, $data %ad, $data %fd
      %x = load double, $data %t, align 8
      %ux = fadd double %u, %x
      br label %merge

    merge:
      %p = phi $data [ %fd, %top ], [ %ad, %pick ]
      %q = phi $data [ %ad, %top ], [ %fd, %pick ]
      %w = phi double [ 0.0, %top ], [ %ux, %pick ]
      %v = load double, $data %p, align 8
      %y = load double, $data %q, align 8
      %vy = fadd double %v, %y
      %r = fadd double %vy, %w
      ret double %r
    }
    """
end

function merge_order_split!(mod, name)
    f = functions(mod)[name]
    Enzyme.Compiler.split_constant_merges!(mod, Enzyme.Compiler.ConstantFields(f, Set{LLVM.Value}()))
    return f
end

@testset "Splitting a select first does not leave stale phis to split" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, merge_order_ir(LLVM.supports_typed_pointers(ctx)))
        @test LLVM.verify(mod) === nothing
        @testset "$name" for name in ("select_into_phi", "phi_before_select", "two_selects_two_phis")
            f = merge_order_split!(mod, name)
            @test LLVM.verify(mod) === nothing
            insts = [inst for bb in blocks(f) for inst in instructions(bb)]
            # Every merge of the two arrays is gone: each load reads one of them.
            @test !any(isa(inst, LLVM.SelectInst) for inst in insts)
            @test !any(isa(inst, LLVM.PHIInst) && isa(value_type(inst), LLVM.PointerType) for inst in insts)
            for inst in insts
                isa(inst, LLVM.LoadInst) && value_type(inst) == LLVM.DoubleType() || continue
                @test !isa(operands(inst)[1], LLVM.PHIInst)
            end
        end
    end
end
