# A record's pointer shadow follows the chosen edge. Its floating-point adjoint
# accumulates on that edge in reverse. A PHI or select of the whole record
# makes libEnzyme try to use one differentiation rule for both.
record_has_float(ty::LLVM.LLVMType) = ty isa LLVM.FloatingPointType
record_has_float(ty::LLVM.StructType) = any(record_has_float, ty.elements)
record_has_float(ty::Union{LLVM.ArrayType, LLVM.VectorType}) = record_has_float(ty.element_type)

function record_merge_fields!(fields, path::Vector{Cuint}, @nospecialize(ty::LLVM.LLVMType))
    if ty isa Union{LLVM.StructType, LLVM.ArrayType} && record_has_float(ty) && floating_point_aggregate_type(ty) === nothing
        count = ty isa LLVM.StructType ? length(ty.elements) : Int(ty.length)
        for i in 1:count
            child = ty isa LLVM.StructType ? ty.elements[i] : ty.element_type
            push!(path, i - 1)
            record_merge_fields!(fields, path, child)
            pop!(path)
        end
    else
        # Keep homogeneous subaggregates together, e.g. an entire NTuple of floats.
        push!(fields, (copy(path), ty))
    end
    return nothing
end

function split_record_merge!(builder::IRBuilder, inst::Union{LLVM.PHIInst, LLVM.SelectInst})
    if inst isa LLVM.PHIInst
        insert_at = LLVM.after_phis(inst.parent)
        insert_at === nothing && return false
        # A terminator's result is unavailable before it on the incoming edge.
        for (value, bb) in inst.incoming
            (value == bb.terminator || bb.terminator isa LLVM.CatchSwitchInst) && return false
        end
    else
        insert_at = LLVM.before(inst)
        position!(builder, insert_at)
        # Even for an undef condition, all fields must take the same arm.
        cond = freeze!(builder, inst.operands[1])
    end

    fields = Tuple{Vector{Cuint}, LLVM.LLVMType}[]
    record_merge_fields!(fields, Cuint[], inst.value_type)
    record = LLVM.UndefValue(inst.value_type)
    for (path, fieldty) in fields
        if inst isa LLVM.PHIInst
            position!(builder, LLVM.before(inst))
            field = phi!(builder, fieldty)
            # LLVM requires identical values on repeated edges from one block.
            extracted = Dict{LLVM.BasicBlock, LLVM.Value}()
            for (value, bb) in inst.incoming
                if !haskey(extracted, bb)
                    position!(builder, LLVM.before(bb.terminator))
                    extracted[bb] = extract_value!(builder, value, path)
                end
                push!(field.incoming, (extracted[bb], bb))
            end
        else
            position!(builder, insert_at)
            lhs = extract_value!(builder, inst.operands[2], path)
            rhs = extract_value!(builder, inst.operands[3], path)
            field = select!(builder, cond, lhs, rhs)
        end
        position!(builder, insert_at)
        record = insert_value!(builder, record, field, path)
    end
    if record isa LLVM.Instruction
        for key in ("enzyme_type", "enzyme_truetype", "enzyme_active")
            haskey(inst.metadata, key) && (record.metadata[key] = inst.metadata[key])
        end
    end
    replace_uses!(inst, record)
    erase!(inst)
    return true
end

function split_record_merges!(mod::LLVM.Module)
    pending = Union{LLVM.PHIInst, LLVM.SelectInst}[]
    @dispose builder = IRBuilder() begin
        for f in mod.functions
            for bb in f.blocks, inst in bb.instructions
                inst isa Union{LLVM.PHIInst, LLVM.SelectInst} || continue
                ty = inst.value_type
                ty isa Union{LLVM.StructType, LLVM.ArrayType} || continue
                record_has_float(ty) || continue
                # Derived addresses require a separate GC root on each edge.
                # This pass runs after nodecayed_phis! has established those roots.
                tracked = CountTrackedPointers(ty)
                tracked.count != 0 && !tracked.derived || continue
                haskey(inst.metadata, "enzyme_inactive") || push!(pending, inst)
            end
            changed = false
            for inst in pending
                changed |= split_record_merge!(builder, inst)
            end
            empty!(pending)
            # Fold field reads through record construction, including reads on
            # a backedge between two formerly aggregate PHIs.
            changed && run!(InstSimplifyPass(), f)
        end
    end
    return nothing
end
