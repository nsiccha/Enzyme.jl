using Enzyme, LLVM, Test

# A loop-carried value read from an array on every edge into the loop, such as `yp` in a
# loop whose first iteration is peeled (`yp = y[first]; for t in first+1:last; ...;
# yp = y[t]; end`), becomes a load through a phi of element addresses: LLVM's instcombine
# folds the phi of the two loads into one load of a phi of their addresses (address space
# 13). On Julia 1.10, `nodecayed_phis!` then rewrote that phi into a phi of the array
# objects and a phi of byte offsets, and reloaded the data pointer from the object on
# every iteration. The reverse pass reloaded the object and its data pointer on every
# iteration of the loop's reverse, which cost the peeled loop 5–8% of its Reverse time
# over the same loop not peeled. `unfold_field_phi_loads!` now loads in the predecessors
# instead, which leaves a phi of the values, as in the loop that is not peeled.

function central_differences(f, x; h = 1.0e-6)
    g = zero(x)
    for i in eachindex(x)
        up = copy(x)
        down = copy(x)
        up[i] += h
        down[i] -= h
        g[i] = (f(up) - f(down)) / (2h)
    end
    return g
end

# The residuals of an ARMA(1,1) recursion, written plainly and with the first step peeled.
for (suffix, inl) in (("inl", true), ("noinl", false))
    plain = Symbol(:dpl_plain_, suffix)
    peeled = Symbol(:dpl_peeled_, suffix)
    fp = :(
        function $plain(y, m, f, h)
            out = similar(y, typeof(m * f * h))
            yp, ep = m, zero(m * f * h)
            for t in eachindex(y)
                e = y[t] - (m + f * yp + h * ep)
                @inbounds out[t] = e
                yp, ep = y[t], e
            end
            return out
        end
    )
    fq = :(
        function $peeled(y, m, f, h)
            out = similar(y, typeof(m * f * h))
            ix = eachindex(y)
            isempty(ix) && return out
            yp, ep = m, zero(m * f * h)
            t = first(ix)
            e = y[t] - (m + f * yp + h * ep)
            @inbounds out[t] = e
            yp, ep = y[t], e
            for t in (first(ix) + 1):last(ix)
                e = y[t] - (m + f * yp + h * ep)
                @inbounds out[t] = e
                yp, ep = y[t], e
            end
            return out
        end
    )
    @eval $(inl ? :(@inline $fp) : :(@noinline $fp))
    @eval $(inl ? :(@inline $fq) : :(@noinline $fq))
    for path in (plain, peeled)
        @eval function $(Symbol(:dpl_obj_, path))(p, y, idx)
            G = length(idx)
            s = 0.0
            for g in eachindex(idx)
                errs = $path(y[idx[g]], p[g], p[G + g], p[2G + g])
                acc = 0.0
                for e in errs
                    acc += -0.5 * e * e
                end
                s += acc
            end
            return s
        end
    end
end

const DPL_OBJECTIVES = (
    dpl_obj_dpl_plain_inl, dpl_obj_dpl_peeled_inl, dpl_obj_dpl_plain_noinl, dpl_obj_dpl_peeled_noinl,
)

# Paths of 0, 1, 2 and 7 steps: none, only the peeled step, one loop iteration, several.
function dpl_data()
    lengths = [0, 1, 2, 7]
    y = [sin(0.7i) + 0.1i for i in 1:sum(lengths)]
    idx = Vector{Int}[]
    start = 1
    for n in lengths
        push!(idx, collect(start:(start + n - 1)))
        start += n
    end
    G = length(lengths)
    p = vcat([0.1, -0.2, 0.3, 0.05], [0.5, 0.2, -0.4, 0.3], [0.25, -0.3, 0.1, 0.4])
    @assert length(p) == 3G
    return p, y, idx
end

function dpl_gradient(mode, f, p, y, idx)
    dp = zero(p)
    autodiff(mode, Const(f), Active, Duplicated(p, dp), Const(y), Const(idx))
    return dp
end

function with_unfold(f, enabled)
    Enzyme.Compiler.UnfoldDataPhiLoads[] = enabled
    try
        return f()
    finally
        Enzyme.Compiler.UnfoldDataPhiLoads[] = true
    end
end

@testset "Peeled and plain loops: the same derivatives" begin
    p, y, idx = dpl_data()
    fd = central_differences(q -> dpl_obj_dpl_plain_inl(q, y, idx), p)
    reference = dpl_gradient(Reverse, dpl_obj_dpl_plain_inl, p, y, idx)
    @test reference ≈ fd atol = 1.0e-6
    for f in DPL_OBJECTIVES
        @test f(p, y, idx) ≈ dpl_obj_dpl_plain_inl(p, y, idx)
        @test dpl_gradient(Reverse, f, p, y, idx) ≈ reference
        @test dpl_gradient(ReverseWithPrimal, f, p, y, idx) ≈ reference
        v = [cos(0.3i) for i in eachindex(p)]
        @test only(autodiff(Forward, Const(f), Duplicated(p, v), Const(y), Const(idx))) ≈
            sum(reference .* v)
    end
end

@testset "Peeled loop: split mode" begin
    p, y, idx = dpl_data()
    reference = dpl_gradient(Reverse, dpl_obj_dpl_plain_inl, p, y, idx)
    for f in (dpl_obj_dpl_peeled_inl, dpl_obj_dpl_peeled_noinl)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            Duplicated{Vector{Float64}}, Const{Vector{Float64}}, Const{Vector{Vector{Int}}}
        )
        dp = zero(p)
        tape, primal, _ = fwd(Const(f), Duplicated(p, dp), Const(y), Const(idx))
        rev(Const(f), Duplicated(p, dp), Const(y), Const(idx), 1.0, tape)
        @test primal ≈ f(p, y, idx)
        @test dp ≈ reference
    end
end

derivative_ir(f) = sprint() do io
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, Tuple{Duplicated{Vector{Float64}}, Const{Vector{Float64}}, Const{Vector{Vector{Int}}}};
        dump_module = true, second_stage = false, debuginfo = :none
    )
end

@testset "The peeled carry stays a phi of values" begin
    # Julia 1.10 reloads the data pointer of an object phi; later versions read it
    # through `julia.gc_loaded`. Either way the rewrite names its phis `nodecayed`.
    nodecayed(ir) = count(r"%\"?nodecayed(off)?\.[\w.]*value_phi", ir)
    unfolded = nodecayed(derivative_ir(dpl_obj_dpl_peeled_inl))
    @test unfolded == 0
    # A new method, since the switch acts when a derivative is compiled.
    folded_back(p, y, idx) = dpl_obj_dpl_peeled_inl(p, y, idx)
    folded = with_unfold(false) do
        nodecayed(derivative_ir(folded_back))
    end
    if VERSION < v"1.11-"
        @test folded > 0
    end
    p, y, idx = dpl_data()
    @test with_unfold(false) do
        dpl_gradient(Reverse, folded_back, p, y, idx)
    end ≈ dpl_gradient(Reverse, dpl_obj_dpl_peeled_inl, p, y, idx)
end

# `phi` of two element addresses into array data, loaded at the top of `merge`.
function data_phi_fixture(; write_first = false, other_use = false, atomic = false)
    load = atomic ? "load atomic double, double addrspace(13)* %elem monotonic, align 8" :
        "load double, double addrspace(13)* %elem, align 8"
    return parse(
        LLVM.Module, """
        declare void @clobber()
        declare void @retain(double addrspace(13)*)
        define double @test(i1 %first, i1 %ok, double addrspace(13)* %a, double addrspace(13)* %b) {
        entry:
            br i1 %first, label %left, label %right
        left:
            br label %merge
        right:
            %bnext = getelementptr inbounds double, double addrspace(13)* %b, i64 3
            br i1 %ok, label %merge, label %fail
        fail:
            ret double 0.0
        merge:
            %elem = phi double addrspace(13)* [ %a, %left ], [ %bnext, %right ]
            $(write_first ? "call void @clobber()" : "")
            $(other_use ? "call void @retain(double addrspace(13)* %elem)" : "")
            %v = $load
            ret double %v
        }
        """
    )
end

@testset "Data address phis are unfolded into phis of loads" begin
    LLVM.Context() do ctx
        mod = data_phi_fixture()
        fn = LLVM.functions(mod)["test"]
        @test Enzyme.Compiler.unfold_field_phi_loads!(fn)
        LLVM.verify(mod)
        merge = only(bb for bb in LLVM.blocks(fn) if LLVM.name(bb) == "merge")
        v = only(LLVM.operands(LLVM.terminator(merge)))
        @test v isa LLVM.PHIInst
        @test all(x isa LLVM.LoadInst for (x, _) in LLVM.incoming(v))
        # `right` also branches to `fail`, so its load sits on a new edge block.
        @test all(length(LLVM.successors(LLVM.terminator(b))) == 1 for (_, b) in LLVM.incoming(v))
        @test !any(
            inst isa LLVM.PHIInst && LLVM.value_type(inst) isa LLVM.PointerType
                for bb in LLVM.blocks(fn) for inst in LLVM.instructions(bb)
        )
        for options in ((; write_first = true), (; other_use = true), (; atomic = true))
            mod = data_phi_fixture(; options...)
            fn = LLVM.functions(mod)["test"]
            @test !Enzyme.Compiler.unfold_field_phi_loads!(fn)
            LLVM.verify(mod)
            merge = only(bb for bb in LLVM.blocks(fn) if LLVM.name(bb) == "merge")
            @test only(LLVM.operands(LLVM.terminator(merge))) isa LLVM.LoadInst
        end
        mod = data_phi_fixture()
        fn = LLVM.functions(mod)["test"]
        @test !with_unfold(() -> Enzyme.Compiler.unfold_field_phi_loads!(fn), false)
    end
end
