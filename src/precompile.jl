using PrecompileTools: PrecompileTools, @setup_workload

# What a package image differentiates so that it holds Enzyme's compiler. Enzyme's own image
# runs it below, and `EnzymeStaticArraysExt` runs it again because loading StaticArrays discards
# that image's code, so this is the one place that says what they compile. It is a macro and not
# a function so that each image differentiates functions of its own, for which nothing has
# compiled `autodiff` yet, whereas a function here would call what Enzyme's image already holds.
macro compiler_workload()
    return esc(
        quote
            $PrecompileTools.@setup_workload begin
                precompile_module = @eval module $(gensym())
                f(x) = x^2
                # Branches and selects, here those of Base's `cos`, are what the passes that act
                # on control flow (`simplify_conditioned_selects!`, for one) work on. `f` has
                # none, so without `g` the first derivative of a session compiles those passes'
                # code.
                g(x) = cos(x)
                end

                # This workload is for Enzyme's compiler, and has no use for these derivatives
                # afterwards.
                $Enzyme.Compiler.RetainDerivatives[] = false
                $PrecompileTools.@compile_workload begin
                    $Enzyme.autodiff($Enzyme.Reverse, precompile_module.f, $Enzyme.Active(2.0))
                    $Enzyme.autodiff($Enzyme.Reverse, precompile_module.g, $Enzyme.Active(2.0))
                end
                $Enzyme.Compiler.RetainDerivatives[] = true

                # Everything the workload cached belongs to this process: thunks are held as JIT
                # addresses, the rule and activity memos are keyed on worlds that mean nothing
                # once the image is loaded elsewhere. Serializing them hands every session that
                # loads the image a cache it must not use, so drop them and leave it with none.
                $Enzyme.Compiler.clear_caches!()
            end
        end
    )
end

# `__init__` sets up the JIT of a loaded Enzyme, and it has not run in the process that writes
# Enzyme's own image. An extension's image is written with Enzyme loaded, so this stays out of
# the macro: running it a second time there would make a second JIT.
@setup_workload begin
    Compiler.JIT.setup_globals()
end

@compiler_workload

# A derivative taken from a package image spends its first call on nothing but this, so have it
# compiled.
@setup_workload begin
    precompile(Compiler.retention_conditions, ())
    precompile(Compiler.retained_result, (Compiler.RetainedDerivatives, Compiler.CompilerJob))
    precompile(Compiler.materialize_retained, (Compiler.RetainedDerivative,))
    precompile(Compiler.track_retained_validity, (Vector{Any},))
    precompile(Compiler.first_call_of_thunk, (Compiler.LazyThunk,))
    precompile(Compiler.first_call_of_thunk, (Compiler.LazySplitThunks,))
    precompile(Compiler.compile_lazy_thunk!, (Compiler.LazyThunk,))
    precompile(Compiler.compile_lazy_thunk!, (Compiler.LazySplitThunks,))
    precompile(Compiler.lazy_compilation, (Compiler.CompilerJob, Compiler.RetainedDerivatives))
end
