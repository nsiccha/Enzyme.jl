# BLAS routines that libEnzyme has no derivative for in the current mode are
# differentiated through their reference implementation: `EnzymeBitcodeReplacement`
# (libEnzymeBCLoad) links the routine's bitcode into the module, internal and
# `alwaysinline`, so the call is inlined and Enzyme differentiates its loops.
#
# That only reaches a call to a declaration named after the routine itself
# (`dtrmm_64_`, which is what Julia 1.10 emits). Julia 1.12 and 1.13 name a call
# through a library `ejlstr$<routine>$<library>` (`check_ir`), and record the
# routine in `enzyme_math`. For such a declaration BCLoader gives it a body that
# calls the routine's bitcode, but leaves it a separate function that keeps
# `enzyme_math`. libEnzyme matches a call's BLAS routine by `enzyme_math` before
# it looks at a body, so it differentiated the call with its native rule for the
# routine, the rule the fallback was chosen to avoid. For `trmm` and `trmv`, whose
# native rules have no Forward derivative, Forward mode then raised
# `EnzymeNoDerivativeError` ("cannot handle blas argument within trmm").
#
# `bitcode_wrapper_candidates` lists the declarations BCLoader may rewrite this way,
# before it runs; `inline_bitcode_wrappers!` makes each one it gave a body, for a
# routine it reports as replaced, an internal `alwaysinline` function without
# `enzyme_math`, so that the call is inlined into the routine's bitcode as on
# Julia 1.10. Functions that already had a body, such as the `ejlrt$` stubs of a
# library Julia resolves at run time, are not candidates and keep their rules.

function enzyme_math_name(f::LLVM.Function)
    attr = LLVM.API.LLVMGetStringAttributeAtIndex(
        f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
        "enzyme_math", length("enzyme_math"),
    )
    attr == C_NULL && return nothing
    return LLVM.StringAttribute(attr).value
end

# Declarations whose `enzyme_math` routine is not their own name.
function bitcode_wrapper_candidates(mod::LLVM.Module)
    candidates = LLVM.Function[]
    for f in mod.functions
        isempty(f.blocks) || continue
        math = enzyme_math_name(f)
        math === nothing && continue
        math == f.name && continue
        push!(candidates, f)
    end
    return candidates
end

# Returns the names of the functions it changed.
function inline_bitcode_wrappers!(candidates::Vector{LLVM.Function}, replaced::Vector{String})
    inlined = String[]
    for f in candidates
        isempty(f.blocks) && continue
        math = enzyme_math_name(f)
        math === nothing && continue
        math in replaced || continue
        delete!(f.function_attributes, "enzyme_math")
        delete!(f.function_attributes, PRESERVEPRIMAL_ATTR_KIND)
        delete!(f.function_attributes, :noinline)
        push!(f.function_attributes, EnumAttribute(:alwaysinline))
        f.linkage = LLVM.Linkage.Internal
        push!(inlined, f.name)
    end
    return inlined
end
