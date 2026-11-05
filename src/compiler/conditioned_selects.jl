# Julia can represent a union by selecting between values with different layouts.
# Before path-insensitive type analysis, specialize uses whose control condition
# proves which input was selected. For example, a conditional integer/float
# payload must be selected before its sitofp/bitcast, not after both casts.
# The same proof removes contradictory tag branches, whose unreachable boxing
# paths would otherwise merge incompatible layouts into a pointer PHI.

function select_condition_expression(value::LLVM.Value)
    return value isa LLVM.ICmpInst || value isa LLVM.TruncInst ||
        value isa LLVM.ZExtInst || value isa LLVM.SExtInst ||
        value isa LLVM.AndInst || value isa LLVM.OrInst || value isa LLVM.XorInst
end

# Recognize small integer predicates of a single SSA value (e.g. Julia's union
# tag tests `tag < 0` and `(tag & 127) == 1`). Unknown expressions stay opaque.
function select_condition_leaf(value::LLVM.Value, depth::Int = 8)
    depth == 0 && return nothing
    value isa LLVM.ConstantInt && return nothing
    if select_condition_expression(value)
        result = nothing
        for operand in operands(value)
            operand isa LLVM.ConstantInt && continue
            leaf = select_condition_leaf(operand, depth - 1)
            leaf === nothing && return nothing
            result === nothing || result == leaf || return nothing
            result = leaf
        end
        return result
    end
    return value
end

function evaluate_select_condition(value, variable, constant, depth::Int = 8)
    value == variable && return constant
    value isa LLVM.ConstantInt && return value
    depth == 0 && return nothing
    if value isa LLVM.ICmpInst || value isa LLVM.AndInst ||
            value isa LLVM.OrInst || value isa LLVM.XorInst
        lhs = evaluate_select_condition(operands(value)[1], variable, constant, depth - 1)
        rhs = evaluate_select_condition(operands(value)[2], variable, constant, depth - 1)
        lhs isa LLVM.ConstantInt && rhs isa LLVM.ConstantInt || return nothing
        # IRBuilder folds constant integer operands on all supported LLVM
        # versions; the older LLVMConst* entry points were removed in LLVM 18/19.
        builder = IRBuilder()
        result = if value isa LLVM.ICmpInst
            LLVM.icmp!(builder, LLVM.API.LLVMGetICmpPredicate(value), lhs, rhs)
        elseif value isa LLVM.AndInst
            LLVM.and!(builder, lhs, rhs)
        elseif value isa LLVM.OrInst
            LLVM.or!(builder, lhs, rhs)
        else
            LLVM.xor!(builder, lhs, rhs)
        end
        dispose(builder)
        return result
    elseif value isa LLVM.TruncInst || value isa LLVM.ZExtInst || value isa LLVM.SExtInst
        operand = evaluate_select_condition(operands(value)[1], variable, constant, depth - 1)
        operand isa LLVM.ConstantInt || return nothing
        builder = IRBuilder()
        result = if value isa LLVM.TruncInst
            LLVM.trunc!(builder, operand, value_type(value))
        elseif value isa LLVM.ZExtInst
            LLVM.zext!(builder, operand, value_type(value))
        else
            LLVM.sext!(builder, operand, value_type(value))
        end
        dispose(builder)
        return result
    end
    return nothing
end

# An overapproximation of all possible values. Follow aggregate return tags only
# through non-interposable definitions. An unknown leaf, including a parameter,
# undef or poison, invalidates the proof. The work and aggregate paths are bounded
# even for cyclic PHIs, recursive calls and very large functions.
function finite_select_values(root::LLVM.Value)
    todo = Tuple{LLVM.Value, Tuple}[(root, ())]
    seen = Set{Tuple{LLVM.Value, Tuple}}()
    result = LLVM.ConstantInt[]
    budget = 256
    while !isempty(todo)
        value, path = pop!(todo)
        (value, path) in seen && continue
        push!(seen, (value, path))
        budget -= 1
        budget >= 0 && length(path) <= 8 || return nothing
        if value isa LLVM.ConstantInt && isempty(path)
            value in result || push!(result, value)
        elseif value isa LLVM.PHIInst
            for (incoming, _) in LLVM.incoming(value)
                length(todo) < budget || return nothing
                push!(todo, (incoming, path))
            end
        elseif value isa LLVM.SelectInst
            push!(todo, (operands(value)[2], path), (operands(value)[3], path))
        elseif value isa LLVM.ExtractValueInst
            LLVM.API.LLVMGetNumIndices(value) <= 8 || return nothing
            indices = Tuple(unsafe_wrap(Array, LLVM.API.LLVMGetIndices(value), Int(LLVM.API.LLVMGetNumIndices(value))))
            push!(todo, (operands(value)[1], (indices..., path...)))
        elseif value isa LLVM.InsertValueInst && !isempty(path)
            LLVM.API.LLVMGetNumIndices(value) <= 8 || return nothing
            indices = Tuple(unsafe_wrap(Array, LLVM.API.LLVMGetIndices(value), Int(LLVM.API.LLVMGetNumIndices(value))))
            common = min(length(indices), length(path))
            if length(indices) <= length(path) && path[1:common] == indices
                push!(todo, (operands(value)[2], path[(length(indices) + 1):end]))
            elseif path[1:common] != indices[1:common]
                push!(todo, (operands(value)[1], path))
            else
                return nothing
            end
        elseif value isa LLVM.CallInst && called_operand(value) isa LLVM.Function
            fn = called_operand(value)
            LLVM.linkage(fn) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage) || return nothing
            found = false
            for bb in blocks(fn)
                budget -= 1
                budget >= 0 || return nothing
                term = terminator(bb)
                term isa LLVM.RetInst && !isempty(operands(term)) || continue
                push!(todo, (operands(term)[1], path))
                found = true
            end
            found || return nothing
        else
            folded = isempty(path) ? evaluate_select_condition(value, nothing, nothing) : nothing
            folded isa LLVM.ConstantInt || return nothing
            folded in result || push!(result, folded)
        end
    end
    return isempty(result) ? nothing : result
end

function implied_select_condition(target, condition, taken, domains)
    target == condition && return taken
    variable = select_condition_leaf(target)
    variable === nothing && return nothing
    if !haskey(domains, variable)
        domains[variable] = finite_select_values(variable)
    end
    values = domains[variable]
    values === nothing && return nothing
    answer = nothing
    for value in values
        branch = evaluate_select_condition(condition, variable, value)
        branch isa LLVM.ConstantInt || return nothing
        convert(Bool, branch) == taken || continue
        selected = evaluate_select_condition(target, variable, value)
        selected isa LLVM.ConstantInt || return nothing
        current = convert(Bool, selected)
        answer === nothing || answer == current || return nothing
        answer = current
    end
    return answer
end

# Cache both successful and failed proofs; a shared tag can have many uses.
function cached_select_condition(target, condition, taken, domains, proofs)
    key = (target, condition, taken)
    if !haskey(proofs, key)
        proofs[key] = implied_select_condition(target, condition, taken, domains)
    end
    return proofs[key]
end

function known_condition_at_use(condition, user, local_condition, local_taken, tree, tests, domains, proofs)
    selected = local_condition === nothing ? nothing :
        cached_select_condition(condition, local_condition, local_taken, domains, proofs)
    if selected === nothing
        leaf = something(select_condition_leaf(condition), condition)
        if haskey(tests, leaf)
            for (branch_condition, anchor, taken) in tests[leaf]
                anchor == user || LLVM.dominates(tree, anchor, user) || continue
                selected = cached_select_condition(condition, branch_condition, taken, domains, proofs)
                selected === nothing || break
            end
        end
    end
    return selected
end

function specialize_conditioned_select!(builder, value, user, local_condition, local_taken, tree, tests, domains, proofs, depth::Int = 8)
    depth == 0 && return value
    if value_type(value) == LLVM.Int1Type() && !(value isa LLVM.ConstantInt)
        known = known_condition_at_use(value, user, local_condition, local_taken, tree, tests, domains, proofs)
        known === nothing || return LLVM.ConstantInt(known)
    end
    if value isa LLVM.SelectInst && value_type(operands(value)[1]) == LLVM.Int1Type()
        selected = known_condition_at_use(operands(value)[1], user, local_condition, local_taken, tree, tests, domains, proofs)
        selected === nothing && return value
        return specialize_conditioned_select!(builder, operands(value)[selected ? 2 : 3], user, local_condition, local_taken, tree, tests, domains, proofs, depth - 1)
    elseif value isa LLVM.Instruction && LLVM.API.LLVMIsACastInst(value) != C_NULL
        input = operands(value)[1]
        replacement = specialize_conditioned_select!(builder, input, user, local_condition, local_taken, tree, tests, domains, proofs, depth - 1)
        replacement == input && return value
        # Clone only casts. Loads and calls must retain their execution order,
        # and other users must keep the original (potentially mixed) value.
        clone = LLVM.Instruction(LLVM.API.LLVMInstructionClone(value))
        LLVM.API.LLVMSetOperand(clone, 0, replacement)
        LLVM.API.LLVMInsertIntoBuilderWithName(builder, clone, "select.cast")
        return clone
    end
    return value
end

function simplify_conditioned_selects!(mod::LLVM.Module)
    changed = 0
    cfg_changed = false
    for fn in functions(mod)
        isempty(blocks(fn)) && continue
        original_instructions = LLVM.Instruction[]
        has_control = false
        for bb in blocks(fn), inst in instructions(bb)
            push!(original_instructions, inst)
            has_control |= inst isa LLVM.SelectInst || (inst isa LLVM.BrInst && LLVM.API.LLVMGetNumSuccessors(inst) == 2)
        end
        has_control || continue
        tests = Dict{LLVM.Value, Vector{Tuple{LLVM.Value, LLVM.Instruction, Bool}}}()
        unique_predecessors = Dict{LLVM.BasicBlock, Bool}()
        for bb in blocks(fn)
            term = terminator(bb)
            term isa LLVM.BrInst && LLVM.API.LLVMGetNumSuccessors(term) == 2 || continue
            condition = LLVM.Value(LLVM.API.LLVMGetCondition(term))
            leaf = something(select_condition_leaf(condition), condition)
            entries = get!(tests, leaf, Tuple{LLVM.Value, LLVM.Instruction, Bool}[])
            for edge in 0:1
                # Bound the number of dominance queries per operand. Additional
                # branches only provide opportunities; ignoring them is safe.
                length(entries) < 16 || break
                succ = LLVM.BasicBlock(LLVM.API.LLVMGetSuccessor(term, edge))
                if !haskey(unique_predecessors, succ)
                    unique_predecessors[succ] = length(LLVM.predecessors(succ)) == 1
                end
                unique_predecessors[succ] || continue
                for anchor in instructions(succ)
                    anchor isa LLVM.PHIInst && continue
                    push!(entries, (condition, anchor, edge == 0))
                    break
                end
            end
        end
        tree = isempty(tests) ? nothing : LLVM.DomTree(fn)
        builder = IRBuilder()
        domains = Dict{LLVM.Value, Union{Nothing, Vector{LLVM.ConstantInt}}}()
        proofs = Dict{Tuple{LLVM.Value, LLVM.Value, Bool}, Union{Nothing, Bool}}()
        for user in original_instructions
            # PHI operands are evaluated on incoming edges, not in this block.
            user isa LLVM.PHIInst && continue
            position!(builder, user)
            for index in 1:length(operands(user))
                value = operands(user)[index]
                local_condition = user isa LLVM.SelectInst && index in (2, 3) &&
                    value_type(operands(user)[1]) == LLVM.Int1Type() ? operands(user)[1] : nothing
                replacement = specialize_conditioned_select!(builder, value, user, local_condition, index == 2, tree, tests, domains, proofs)
                replacement == value && continue
                LLVM.API.LLVMSetOperand(user, index - 1, replacement)
                cfg_changed |= user isa LLVM.BrInst && index == 1
                changed += 1
            end
        end
        dispose(builder)
        tree === nothing || dispose(tree)
    end
    cfg_changed && run!(SimplifyCFGPass(), mod)
    changed == 0 || run!(DCEPass(), mod)
    return changed
end
