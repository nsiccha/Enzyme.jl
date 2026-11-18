# If `cond` is (a cast of) the type of a checked field read from an immutable
# tuple, return that read.
function checked_tuple_field_typeof(@nospecialize(cond::LLVM.Value), enzyme_ctx::Union{EnzymeContext, Nothing})
    while true
        if cond isa LLVM.PtrToIntInst || cond isa LLVM.AddrSpaceCastInst || cond isa LLVM.BitCastInst
            cond = cond.operands[1]
        elseif cond isa LLVM.CallInst && cond.called_operand isa LLVM.Function &&
                cond.called_operand.name == "julia.pointer_from_objref"
            cond = first(cond.arguments)
        else
            break
        end
    end
    cond isa LLVM.CallInst || return nothing
    cf = cond.called_operand
    cf isa LLVM.Function && cf.name == "julia.typeof" || return nothing
    source = first(cond.arguments)
    source isa LLVM.CallInst || return nothing
    sf = source.called_operand
    sf isa LLVM.Function || return nothing
    sf.name in ("jl_get_nth_field_checked", "ijl_get_nth_field_checked") || return nothing
    legal, tuplety, _ = abs_typeof(first(source.arguments), enzyme_ctx)
    legal && tuplety <: Union{Tuple, NamedTuple} && isconcretetype(tuplety) || return nothing
    return source
end

# Identify a runtime type test of a checked field from an immutable tuple.
function tuple_field_type_test(term::LLVM.Instruction, enzyme_ctx::Union{EnzymeContext, Nothing})
    if term isa LLVM.SwitchInst
        return checked_tuple_field_typeof(term.operands[1], enzyme_ctx)
    elseif term isa LLVM.BrInst && LLVM.API.LLVMGetNumSuccessors(term) == 2
        # Julia can lower a two-way type test to an equality branch instead
        # of a switch (notably with Julia 1.12).
        cmp = LLVM.Value(LLVM.API.LLVMGetCondition(term))
        cmp isa LLVM.ICmpInst || return nothing
        LLVM.API.LLVMGetICmpPredicate(cmp) in
            (LLVM.IntPredicate.EQ, LLVM.IntPredicate.NE) || return nothing
        # The other operand is the type tested against: a constant, or a load
        # of the global that holds it while Julia values stay symbolic until
        # the module is linked.
        lhs, rhs = cmp.operands
        source = checked_tuple_field_typeof(lhs, enzyme_ctx)
        source === nothing || return source
        return checked_tuple_field_typeof(rhs, enzyme_ctx)
    end
    return nothing
end

# Give each runtime type arm its own read from an immutable tuple. A boxed
# heterogeneous field otherwise shares one SSA value between distinct layouts,
# causing Enzyme's path-insensitive type analysis to combine their pointer depths.
# Preserve the original checked read and type test; only rematerialize the
# already-checked field where the original read strictly dominates the type arm,
# and only redirect uses that the new read dominates.
function narrow_tuple_fields!(mod::LLVM.Module, enzyme_ctx::Union{EnzymeContext, Nothing})
    changed = 0
    for f in mod.functions
        bbs = collect(f.blocks)
        isempty(bbs) && continue
        tests = Pair{LLVM.Instruction, LLVM.CallInst}[]
        for bb in bbs
            term = bb.terminator
            term === nothing && continue
            source = tuple_field_type_test(term, enzyme_ctx)
            source === nothing && continue
            push!(tests, term => source)
        end
        isempty(tests) && continue

        tree = LLVM.DomTree(f)
        for (term, source) in tests
            users = Set{LLVM.Value}()
            for use in source.uses
                push!(users, use.user)
            end
            successors = Set{LLVM.BasicBlock}()
            for i in 0:(Int(LLVM.API.LLVMGetNumSuccessors(term)) - 1)
                push!(successors, LLVM.BasicBlock(LLVM.API.LLVMGetSuccessor(term, i)))
            end
            for succ in successors
                source.parent != succ || continue
                anchor = nothing
                for inst in succ.instructions
                    inst isa LLVM.PHIInst && continue
                    anchor = inst
                    break
                end
                LLVM.dominates(tree, source, anchor) || continue
                builder = IRBuilder()
                position!(builder, LLVM.before(anchor))
                clone = LLVM.Value(LLVM.API.LLVMInstructionClone(source))
                LLVM.API.LLVMInsertIntoBuilderWithName(builder, clone, "tuplefield.narrow")
                dispose(builder)
                replaced = false
                for user in users
                    # PHI operands are used on incoming edges, not in their block.
                    user isa LLVM.Instruction && !(user isa LLVM.PHIInst) || continue
                    LLVM.dominates(tree, clone, user) || continue
                    for i in 1:length(user.operands)
                        if user.operands[i] == source
                            LLVM.API.LLVMSetOperand(user, i - 1, clone)
                            replaced = true
                        end
                    end
                end
                if replaced
                    changed += 1
                else
                    LLVM.API.LLVMInstructionEraseFromParent(clone)
                end
            end
        end
        dispose(tree)
    end
    return changed
end
