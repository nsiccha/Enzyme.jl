using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    precompile_module = @eval module $(gensym())
    f(x) = x^2
    # Branches and selects, here those of Base's `cos`, are what the passes that act on
    # control flow (`simplify_conditioned_selects!`, for one) work on. `f` has none, so
    # without `g` the first derivative of a session compiles those passes' code.
    g(x) = cos(x)
    end

    Compiler.JIT.setup_globals()

    # This workload is for Enzyme's compiler, and has no use for these derivatives afterwards.
    Compiler.RetainDerivatives[] = false
    @compile_workload begin
        Enzyme.autodiff(Reverse, precompile_module.f, Active(2.0))
        Enzyme.autodiff(Reverse, precompile_module.g, Active(2.0))
    end
    Compiler.RetainDerivatives[] = true

    # A derivative taken from a package image spends its first call on nothing but this, so
    # have it compiled.
    precompile(Compiler.retention_conditions, ())
    precompile(Compiler.retained_result, (Compiler.RetainedDerivatives, Compiler.CompilerJob))
    precompile(Compiler.materialize_retained, (Compiler.RetainedDerivative,))
    precompile(Compiler.track_retained_validity, (Vector{Any},))
    precompile(Compiler.compile_lazy_thunk!, (Compiler.LazySplitThunks,))
    precompile(Compiler.lazy_compilation, (Compiler.CompilerJob, Compiler.RetainedDerivatives))

    # Everything the workload cached belongs to this process: thunks are held as JIT
    # addresses, the rule and activity memos are keyed on worlds that mean nothing once the
    # image is loaded elsewhere. Serializing them hands every session that loads Enzyme a
    # cache it must not use, so drop them and leave the image with none.
    Compiler.clear_caches!()
end
