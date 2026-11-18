using Enzyme, Test, LinearAlgebra
import LLVM

# `Enzyme.derivative_report(f)` runs `f()` and says, for each derivative Enzyme compiled,
# linked or was asked for meanwhile, what the compiled code of that derivative calls at run
# time.
# Enzyme reads the optimized module of every derivative once, before its JIT takes it
# (`Enzyme.Compiler.scan_derivative`), and keeps what it found with the compiled code.
#
# The first testsets run the scan on modules written out here, one shape of call each. The
# others take derivatives and check what the report says about them.

const Compiler = Enzyme.Compiler

# Typed pointers through LLVM 15, opaque ones later.
const TYPED_POINTERS = LLVM.version().major <= 15
pointer_type(pointee::String) = TYPED_POINTERS ? pointee * "*" : "ptr"

function scan(ir::String, global_names = nothing)
    return LLVM.Context() do _
        mod = parse(LLVM.Module, ir)
        try
            Compiler.scan_derivative(mod, global_names)
        finally
            LLVM.dispose(mod)
        end
    end
end

only_finding(findings, callee) = only(f for f in findings if f.callee == callee)

@testset "kinds of calls" begin
    object = pointer_type("i8")
    findings = scan(
        """
        @ejl_enz_runtime_generic_augfwd = external global i8
        @some_function = external global i8
        declare i64 @ijl_apply_generic($object, i64, i32)
        declare i64 @ijl_invoke($object, i64, i32, i64)
        declare i64 @ijl_box_float64(double)
        declare i64 @ijl_gc_pool_alloc(i64, i32, i32)
        declare i64 @ijl_gc_big_alloc(i64, i64)
        declare i64 @ijl_alloc_array_1d(i64, i64)
        declare i64 @malloc(i64)
        declare i64 @calloc(i64, i64)
        declare i64 @realloc(i64, i64)
        declare void @free(i64)
        declare void @ijl_gc_queue_root(i64)
        declare double @llvm.fabs.f64(double)
        declare void @dgemv_64_(i64)

        define double @diffejulia_f(i64 %n, double %x) {
        entry:
          %a = call i64 @ijl_apply_generic($object @ejl_enz_runtime_generic_augfwd, i64 0, i32 3)
          %b = call i64 @ijl_apply_generic($object @some_function, i64 0, i32 1)
          %c = call i64 @ijl_invoke($object @some_function, i64 0, i32 1, i64 0)
          %d = call i64 @ijl_box_float64(double %x)
          %e = call i64 @ijl_gc_pool_alloc(i64 0, i32 1136, i32 32)
          %f = call i64 @ijl_gc_big_alloc(i64 0, i64 %n)
          %g = call i64 @ijl_alloc_array_1d(i64 0, i64 %n)
          %h = call i64 @malloc(i64 8)
          %i = call i64 @malloc(i64 %n)
          %j = call i64 @calloc(i64 4, i64 16)
          %k = call i64 @realloc(i64 %h, i64 24)
          call void @free(i64 %k)
          call void @ijl_gc_queue_root(i64 %e)
          %l = call double @llvm.fabs.f64(double %x)
          call void @dgemv_64_(i64 %n)
          call void inttoptr (i64 16 to $(pointer_type("void ()")))()
          %m = call i8* asm "movq %fs:0, \$0", "=r"()
          ret double %l
        }
        """ |> ir -> TYPED_POINTERS ? ir : replace(ir, "i8* asm" => "ptr asm")
    )
    @test all(f -> f.in_function == "diffejulia_f" && f.sweep === :combined, findings)
    @test all(f -> !f.in_loop && !f.throw_only && f.count == 1, findings)

    fallback = only_finding(findings, "runtime_generic_augfwd")
    @test fallback.kind === :enzyme_runtime_fallback
    @test fallback.bytes === nothing
    @test only_finding(findings, "jl_apply_generic").kind === :dynamic_dispatch
    @test only_finding(findings, "jl_invoke").kind === :dynamic_dispatch
    @test only_finding(findings, "jl_box_float64").kind === :box

    pool = only_finding(findings, "jl_gc_pool_alloc")
    @test pool.kind === :gc_allocation && pool.bytes == 32
    big = only_finding(findings, "jl_gc_big_alloc")
    @test big.kind === :gc_allocation && big.bytes === nothing
    array = only_finding(findings, "jl_alloc_array_1d")
    @test array.kind === :gc_allocation && array.bytes === nothing

    mallocs = [f for f in findings if f.callee == "malloc"]
    @test all(f -> f.kind === :native_allocation, mallocs)
    @test sort([something(f.bytes, -1) for f in mallocs]) == [-1, 8]
    @test only_finding(findings, "calloc").bytes == 64
    @test only_finding(findings, "realloc").bytes == 24

    # Every other call that leaves the module is listed, without a judgement.
    for callee in ("free", "jl_gc_queue_root", "dgemv_64_", "(unknown address)")
        @test only_finding(findings, callee).kind === :runtime_call
    end
    # An intrinsic and inline assembly are not calls out of the module.
    @test !any(f -> startswith(f.callee, "llvm.") || occursin("asm", f.callee), findings)

    # Findings on returning paths come sorted by kind.
    ranks = [Compiler.finding_kind_rank(f.kind) for f in findings]
    @test issorted(ranks)
end

@testset "identical calls are counted, not repeated" begin
    findings = scan(
        """
        declare i64 @malloc(i64)
        define i64 @augmented_julia_f() {
        entry:
          %a = call i64 @malloc(i64 8)
          %b = call i64 @malloc(i64 8)
          %c = call i64 @malloc(i64 16)
          ret i64 %a
        }
        """
    )
    @test length(findings) == 2
    @test sort([(f.bytes, f.count) for f in findings]) == [(8, 2), (16, 1)]
    @test all(f -> f.sweep === :forward, findings)
end

@testset "error paths and loops" begin
    findings = scan(
        """
        declare i64 @malloc(i64)
        declare i64 @ijl_box_int64(i64)
        declare void @ijl_throw(i64)

        define i64 @diffejulia_f(i64 %n) {
        entry:
          %bad = icmp slt i64 %n, 0
          br i1 %bad, label %fail, label %loop
        loop:
          %i = phi i64 [ 0, %entry ], [ %next, %loop ]
          %p = call i64 @malloc(i64 8)
          %next = add i64 %i, 1
          %done = icmp eq i64 %next, %n
          br i1 %done, label %exit, label %loop
        exit:
          %q = call i64 @malloc(i64 16)
          ret i64 %q
        fail:
          %e = call i64 @ijl_box_int64(i64 %n)
          call void @ijl_throw(i64 %e)
          unreachable
        }
        """
    )
    in_loop = only(f for f in findings if f.bytes == 8)
    @test in_loop.in_loop && !in_loop.throw_only
    after = only(f for f in findings if f.bytes == 16)
    @test !after.in_loop && !after.throw_only
    box = only_finding(findings, "jl_box_int64")
    @test box.throw_only && !box.in_loop
    @test only_finding(findings, "jl_throw").throw_only
    # Findings on returning paths come first.
    @test issorted([f.throw_only for f in findings])
end

@testset "a function is judged by its callers" begin
    findings = scan(
        """
        declare i64 @malloc(i64)
        declare void @ijl_throw(i64)

        define internal i64 @julia_message(i64 %n) {
        entry:
          %m = call i64 @malloc(i64 %n)
          ret i64 %m
        }

        define internal i64 @julia_step(i64 %n) {
        entry:
          %m = call i64 @malloc(i64 32)
          ret i64 %m
        }

        define internal i64 @julia_twice(i64 %n) {
        entry:
          %a = call i64 @julia_step(i64 %n)
          ret i64 %a
        }

        define internal i64 @julia_recursive(i64 %n) {
        entry:
          %m = call i64 @malloc(i64 48)
          %stop = icmp eq i64 %n, 0
          br i1 %stop, label %done, label %again
        again:
          %less = sub i64 %n, 1
          %r = call i64 @julia_recursive(i64 %less)
          ret i64 %r
        done:
          ret i64 %m
        }

        define i64 @fwddiffejulia_f(i64 %n) {
        entry:
          %bad = icmp slt i64 %n, 0
          br i1 %bad, label %fail, label %loop
        loop:
          %i = phi i64 [ 0, %entry ], [ %next, %loop ]
          %s = call i64 @julia_twice(i64 %i)
          %next = add i64 %i, 1
          %done = icmp eq i64 %next, %n
          br i1 %done, label %exit, label %loop
        exit:
          %r = call i64 @julia_recursive(i64 %n)
          ret i64 %r
        fail:
          %e = call i64 @julia_message(i64 %n)
          call void @ijl_throw(i64 %e)
          unreachable
        }
        """
    )
    # Only called on the way to an error, although the function itself returns.
    message = only(f for f in findings if f.in_function == "julia_message")
    @test message.throw_only && !message.in_loop && message.sweep === :primal
    # Called from a loop, through another function.
    step = only(f for f in findings if f.in_function == "julia_step")
    @test step.in_loop && !step.throw_only && step.bytes == 32
    # A function that calls itself repeats what it does.
    recursive = only(f for f in findings if f.in_function == "julia_recursive")
    @test recursive.in_loop && !recursive.throw_only
end

@testset "sweeps" begin
    findings = scan(
        """
        declare i64 @malloc(i64)

        define i64 @augmented_julia_g() {
        entry:
          %a = call i64 @malloc(i64 1)
          ret i64 %a
        }

        define i64 @diffejulia_g() {
        entry:
          %a = call i64 @malloc(i64 2)
          ret i64 %a
        }

        define i64 @diffejulia_h(i1 %c) {
        entry:
          %a = call i64 @malloc(i64 3)
          br label %invertentry
        invertentry:
          %b = call i64 @malloc(i64 4)
          br label %later
        later:
          %d = call i64 @malloc(i64 5)
          ret i64 %d
        }

        define i64 @julia_k() {
        entry:
          %a = call i64 @malloc(i64 6)
          ret i64 %a
        }

        define i64 @augmented_julia_m() {
        entry:
          %a = call i64 @malloc(i64 7)
          ret i64 %a
        }

        define i64 @diffe2julia_m() {
        entry:
          %a = call i64 @malloc(i64 8)
          ret i64 %a
        }

        define i64 @diffe3julia_n() {
        entry:
          %a = call i64 @malloc(i64 9)
          ret i64 %a
        }
        """
    )
    sweep(bytes) = only(f for f in findings if f.bytes == bytes).sweep
    @test sweep(1) === :forward
    # The reverse sweep of a function that has its forward sweep beside it.
    @test sweep(2) === :reverse
    # In a function holding both, what only an `invert…` block leads to is the reverse sweep.
    @test sweep(3) === :combined
    @test sweep(4) === :reverse
    @test sweep(5) === :reverse
    @test sweep(6) === :primal
    # The reverse sweep of a batch carries the width in its name, the forward sweep does not.
    @test sweep(7) === :forward
    @test sweep(8) === :reverse
    @test sweep(9) === :combined
end

@testset "names for the globals of a module compiled for a package image" begin
    object = pointer_type("i8")
    callee = TYPED_POINTERS ? "bitcast ({}* @enzretaddr1 to i64 (i64)*)" : "@enzretaddr1"
    findings = scan(
        """
        @enzretaddr1 = external global {}
        @jlplt_cblas_ddot_1234_got = external global $(pointer_type("i64 (i64)"))
        declare i64 @"ejlstr\$erf\$libopenspecfun"(i64)

        define i64 @augmented_julia_f(i64 %n) {
        entry:
          %a = call i64 $callee(i64 %n)
          %slot = load $(pointer_type("i64 (i64)")), $(pointer_type(pointer_type("i64 (i64)"))) @jlplt_cblas_ddot_1234_got
          %b = call i64 %slot(i64 %n)
          %c = call i64 @"ejlstr\$erf\$libopenspecfun"(i64 %n)
          ret i64 %a
        }
        """, Dict("enzretaddr1" => "ijl_alloc_array_1d")
    )
    @test only_finding(findings, "jl_alloc_array_1d").kind === :gc_allocation
    @test only_finding(findings, "cblas_ddot").kind === :runtime_call
    @test only_finding(findings, "erf").kind === :runtime_call
end

@testset "a call through a pointer computed at run time" begin
    fptr = pointer_type("i64 (i64)")
    findings = scan(
        """
        define i64 @augmented_julia_f($fptr %f, i64 %n) {
        entry:
          %a = call i64 %f(i64 %n)
          ret i64 %a
        }
        """
    )
    @test only(findings).kind === :runtime_call
    @test only(findings).callee == "(function pointer)"
end

# The findings of `d` on paths that return.
returning(d) = [f for f in d.findings if !f.throw_only]
returning(d, kind::Symbol) = [f for f in d.findings if !f.throw_only && f.kind === kind]
derivative_of(report, f) = only(d for d in report.derivatives if d.func === typeof(f))

@noinline dispatched(v) = Base.inferencebarrier(sin)(v)
function sum_dispatched(x)
    s = 0.0
    for i in eachindex(x)
        s += dispatched(x[i])::Float64
    end
    return s
end

@testset "a call dispatched at run time" begin
    x = [0.3, -0.2, 0.5]
    report = Enzyme.derivative_report() do
        Enzyme.gradient(Reverse, sum_dispatched, x)
    end
    @test report isa Enzyme.DerivativeReport
    @test report.value[1] ≈ cos.(x)

    top = derivative_of(report, sum_dispatched)
    @test top isa Enzyme.ReportedDerivative
    @test top.mode === :reverse
    @test top.arguments === Tuple{Duplicated{Vector{Float64}}}
    @test top.return_annotation === Active{Float64}
    @test top.width == 1
    @test top.origin === :compiled
    @test top.note == ""
    @test top.ir === nothing

    fallbacks = returning(top, :enzyme_runtime_fallback)
    @test all(f -> f isa Enzyme.DerivativeFinding, fallbacks)
    # Enzyme differentiates the call when it runs, once in each sweep.
    @test Set(f.callee for f in fallbacks) == Set(["runtime_generic_augfwd", "runtime_generic_rev"])
    @test all(f -> f.in_loop, fallbacks)
    @test all(f -> any(frame -> frame.func === :dispatched, f.frames), fallbacks)
    @test only(f for f in fallbacks if f.callee == "runtime_generic_augfwd").sweep in (:forward, :combined)
    @test only(f for f in fallbacks if f.callee == "runtime_generic_rev").sweep === :reverse

    # The derivative of what the call reaches is compiled when the call first runs.
    callee = derivative_of(report, sin)
    @test callee.mode === :reverse_split
    @test callee.origin === :compiled
    @test callee.arguments === Tuple{Active{Float64}}
    @test isempty(returning(callee, :enzyme_runtime_fallback))
    @test isempty(returning(callee, :dynamic_dispatch))
    @test top.atomic_add === nothing
    @test callee.atomic_add === nothing
end

@noinline dispatched_cosine(v) = Base.inferencebarrier(cos)(v)
function sum_dispatched_cosine(x)
    s = 0.0
    for i in eachindex(x)
        s += dispatched_cosine(x[i])::Float64
    end
    return s
end

# A derivative is reported with the `AtomicAdd` setting it was compiled with. What a call
# dispatched at run time reaches inherits plain accumulation from its caller.
@testset "the accumulation setting of a derivative" begin
    x = [0.3, -0.2, 0.5]
    dx = zero(x)
    report = Enzyme.derivative_report() do
        autodiff(set_atomic_add(Reverse, false), sum_dispatched_cosine, Active, Duplicated(x, dx))
    end
    @test dx ≈ -sin.(x)
    @test derivative_of(report, sum_dispatched_cosine).atomic_add === false
    callee = derivative_of(report, cos)
    @test callee.atomic_add === :inherited_plain
    @test callee.findings !== nothing

    dx .= 0
    report = Enzyme.derivative_report() do
        autodiff(set_atomic_add(Reverse, true), sum_dispatched_cosine, Active, Duplicated(x, dx))
    end
    @test dx ≈ -sin.(x)
    @test derivative_of(report, sum_dispatched_cosine).atomic_add === true
    @test derivative_of(report, cos).atomic_add === true
end

function sum_squares(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i] * x[i]
    end
    return s
end

@testset "a derivative that calls nothing" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, sum_squares, Active, Duplicated(x, dx))
    end
    @test dx == 2 .* x
    d = only(report.derivatives)
    @test d.func === typeof(sum_squares)
    @test isempty(returning(d))
end

product_norm(x, A) = sum(abs2, A * x)

@testset "allocations and a library call" begin
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    x = [0.5, -1.0]
    dx = zero(x)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, product_norm, Active, Duplicated(x, dx), Const(A))
    end
    @test dx ≈ 2 .* (A' * (A * x))
    d = derivative_of(report, product_norm)
    @test d.arguments === Tuple{Duplicated{Vector{Float64}}, Const{Matrix{Float64}}}
    # The product's result, on Julia's heap.
    @test !isempty(returning(d, :gc_allocation))
    # BLAS is called through an address, which the report names.
    @test any(f -> occursin("gemv", f.callee), returning(d, :runtime_call))
    @test isempty(returning(d, :dynamic_dispatch))
    @test isempty(returning(d, :enzyme_runtime_fallback))
end

# The reverse sweep needs every intermediate `y`, which the forward sweep keeps.
function iterated_sine(x, n)
    y = x
    for _ in 1:n
        y = sin(y) * y
    end
    return y
end

@testset "a tape that grows with a run-time length" begin
    report = Enzyme.derivative_report() do
        autodiff(Reverse, iterated_sine, Active, Active(0.7), Const(5))
    end
    d = derivative_of(report, iterated_sine)
    tapes = returning(d, :native_allocation)
    @test !isempty(tapes)
    @test any(f -> f.bytes === nothing, tapes)
    @test isempty(returning(d, :gc_allocation))
end

checked_square(x, i) = x[i] * x[i]

@testset "a call that is only reached on the way to an error" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, checked_square, Active, Duplicated(x, dx), Const(2))
    end
    @test dx == [0.0, 4.0, 0.0]
    d = derivative_of(report, checked_square)
    # Julia 1.10 calls the runtime's bounds error, later versions a function of their own that
    # raises it, which is only called on the way to the error.
    bounds_error(f) = occursin("bounds_error", f.callee) || occursin("boundserror", f.in_function)
    errors = [f for f in d.findings if f.throw_only]
    @test any(bounds_error, errors)
    @test !any(bounds_error, returning(d))
end

cube_sum(x) = sum(v -> v^3, x)

@testset "a derivative compiled earlier is reported from what Enzyme kept" begin
    x = [1.0, 2.0]
    first = Enzyme.derivative_report(; ir = true) do
        Enzyme.gradient(Reverse, cube_sum, x)
    end
    d1 = derivative_of(first, cube_sum)
    @test d1.origin === :compiled
    # The bitcode of the optimized module, which LLVM reads back.
    @test d1.ir isa Vector{UInt8}
    @test d1.ir[1:4] == [0x42, 0x43, 0xc0, 0xde]
    names = LLVM.Context() do _
        mod = parse(LLVM.Module, d1.ir)
        try
            [fn.name for fn in mod.functions if !isempty(fn.blocks)]
        finally
            LLVM.dispose(mod)
        end
    end
    @test any(startswith("diffejulia_cube_sum"), names)

    second = Enzyme.derivative_report(; ir = true) do
        Enzyme.gradient(Reverse, cube_sum, x)
    end
    @test second.value == first.value
    d2 = derivative_of(second, cube_sum)
    @test d2.origin === :cached
    @test d2.findings == d1.findings
    @test d2.ir === nothing

    # Without a report nothing is collected, and the counter of running reports is back at zero.
    Enzyme.gradient(Reverse, cube_sum, x)
    @test !Compiler.reports_running()
end

function sum_squares_gradient!(dx, x)
    autodiff(Reverse, sum_squares, Active, Duplicated(x, dx))
    return nothing
end

# Every call of `autodiff` asks whether a report is running. That must not keep `autodiff`
# from being inlined into its caller, where it would return its result boxed.
@testset "a derivative call allocates nothing while no report runs" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)
    sum_squares_gradient!(dx, x)
    @test (@allocated sum_squares_gradient!(dx, x)) == 0
end

quartic(x) = sum(v -> v^4, x)
sextic(x) = sum(v -> v^6, x)
square(x) = x * x
@noinline inlined_square_derivative(x) = autodiff(Enzyme.set_abi(Reverse, Enzyme.InlineABI), square, Active, Active(x))[1][1]

@testset "other ways to take a derivative" begin
    x = [1.0, 2.0]

    @testset "Forward" begin
        report = Enzyme.derivative_report() do
            autodiff(Forward, quartic, Duplicated(x, [1.0, 0.0]))
        end
        @test report.value[1] ≈ 4.0
        @test derivative_of(report, quartic).mode === :forward
    end

    @testset "split Reverse" begin
        dx = zero(x)
        report = Enzyme.derivative_report() do
            fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(quartic)}, Active, Duplicated{Vector{Float64}})
            tape, _, _ = fwd(Const(quartic), Duplicated(x, dx))
            rev(Const(quartic), Duplicated(x, dx), 1.0, tape)
        end
        @test dx ≈ 4 .* x .^ 3
        d = derivative_of(report, quartic)
        @test d.mode === :reverse_split
        @test d.findings !== nothing
        # Julia compiles a split-mode derivative while it infers the code that asks for it,
        # which here is before the report runs that code.
        @test d.origin in (:compiled, :cached)
    end

    @testset "a batch" begin
        report = Enzyme.derivative_report() do
            autodiff(Forward, quartic, BatchDuplicated(x, ([1.0, 0.0], [0.0, 1.0])))
        end
        @test derivative_of(report, quartic).width == 2
    end

    @testset "deferred" begin
        dx = zero(x)
        report = Enzyme.derivative_report() do
            autodiff_deferred(Reverse, Const(quartic), Active, Duplicated(x, dx))
        end
        @test dx ≈ 4 .* x .^ 3
        d = derivative_of(report, quartic)
        @test d.origin === :compiled
        @test d.findings !== nothing
        # A later call is not seen: the code of a deferred call is also what a kernel on a
        # device runs, and that cannot ask the host whether a report is running.
        again = Enzyme.derivative_report() do
            autodiff_deferred(Reverse, Const(quartic), Active, Duplicated(x, dx))
        end
        @test isempty(again.derivatives)
    end

    @testset "split-mode thunks kept from before the report" begin
        dx = zero(x)
        fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(sextic)}, Active, Duplicated{Vector{Float64}})
        kept = Enzyme.derivative_report() do
            tape, _, _ = fwd(Const(sextic), Duplicated(x, dx))
            rev(Const(sextic), Duplicated(x, dx), 1.0, tape)
        end
        @test dx ≈ 6 .* x .^ 5
        # Calling a thunk does not ask Enzyme for the derivative, so the report does not see it.
        @test isempty(kept.derivatives)
        # Asking for the thunks again does.
        asked = Enzyme.derivative_report() do
            autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(sextic)}, Active, Duplicated{Vector{Float64}})
        end
        d = derivative_of(asked, sextic)
        @test d.origin === :cached
        @test d.mode === :reverse_split
        @test d.findings !== nothing
    end

    @testset "compiled for inlining into its caller" begin
        for _ in 1:2
            report = Enzyme.derivative_report() do
                inlined_square_derivative(3.0)
            end
            @test report.value ≈ 6.0
            # Julia optimizes this derivative as part of its caller, so Enzyme has no code of
            # its own to read: it is listed, without findings, whenever it was compiled.
            d = derivative_of(report, square)
            @test d.origin === :inlined
            @test d.findings === nothing
            @test occursin("inlining", d.note)
            @test occursin("not inspected", sprint(show, MIME"text/plain"(), d))
        end
    end

    @testset "Forward over Reverse" begin
        inner(y) = Enzyme.gradient(Reverse, quartic, y)[1]
        report = Enzyme.derivative_report() do
            autodiff(Forward, inner, Duplicated(x, [1.0, 0.0]))
        end
        @test report.value[1] ≈ [12.0, 0.0]
        outer = derivative_of(report, inner)
        @test outer.mode === :forward
        @test outer.findings !== nothing
        # The inner derivative runs as part of the outer one's code: it is not called by itself.
        @test !any(d -> d.func === typeof(quartic) && d.mode === :reverse && d.origin === :cached, report.derivatives)
    end
end

quintic(x) = sum(v -> v^5, x)

@testset "a report sees the tasks it starts and the reports inside it" begin
    x = [1.0, 2.0]
    inner = Ref{Any}(nothing)
    outer = Enzyme.derivative_report() do
        inner[] = Enzyme.derivative_report() do
            fetch(Threads.@spawn Enzyme.gradient(Reverse, quintic, x))
        end
    end
    @test inner[].value[1] ≈ 5 .* x .^ 4
    @test derivative_of(inner[], quintic).origin === :compiled
    @test derivative_of(outer, quintic).origin === :compiled
end

@testset "a failing block ends its report" begin
    @test_throws ErrorException Enzyme.derivative_report() do
        error("no derivative today")
    end
    @test !Compiler.reports_running()
end

@testset "printing" begin
    x = [0.3, -0.2, 0.5]
    report = Enzyme.derivative_report() do
        Enzyme.gradient(Reverse, sum_dispatched, x)
    end
    short = sprint(show, report)
    @test occursin("DerivativeReport", short)
    full = sprint(show, MIME"text/plain"(), report)
    @test occursin("derivative of typeof(sum_dispatched)", full)
    @test occursin("compiled earlier in this session", full)
    @test occursin("enzyme_runtime_fallback", full)
    @test occursin("on paths that return", full)
    finding = Base.first(returning(derivative_of(report, sum_dispatched), :enzyme_runtime_fallback))
    @test occursin("runtime_generic", sprint(show, finding))
    @test occursin(" at ", sprint(show, MIME"text/plain"(), finding))
end

deferred_quartic!(dx, x) = autodiff_deferred(Reverse, Const(quartic), Active, Duplicated(x, dx))
deferred_quartic_forward(x, dx) = autodiff_deferred(Forward, Const(quartic), Duplicated, Duplicated(x, dx))
direct_quartic!(dx, x) = autodiff(Reverse, Const(quartic), Active, Duplicated(x, dx))

# The optimized code of `f` for arguments of `types`, as text.
optimized_code(f, types) = sprint(show, only(code_typed(f, types; optimize = true))[1])
asks_for_reports(code) = occursin("RunningReports", code) || occursin("DERIVATIVE_REPORTS", code)
# Whether the call of the derivative is part of the code, and not behind a call that hides it.
shows_derivative_call(code) = occursin("llvmcall", code)

# A kernel that runs on a device takes derivatives with `autodiff_deferred`, and what it
# compiles is the code of that call. It can neither read a counter of the host nor call into
# the host, so the question whether a report is running must not be part of that code. It is
# asked where the host hands out a derivative instead, as for `autodiff`.
@testset "the code of a deferred call does not ask for a report" begin
    vectors = (Vector{Float64}, Vector{Float64})
    for deferred in (deferred_quartic!, deferred_quartic_forward)
        code = optimized_code(deferred, vectors)
        @test shows_derivative_call(code)
        @test !asks_for_reports(code)
    end
    @test deferred_quartic_forward([1.0, 2.0], [1.0, 0.0])[1] ≈ 4.0
    direct = optimized_code(direct_quartic!, vectors)
    @test shows_derivative_call(direct)
    @test asks_for_reports(direct)
end

# Every derivative that was handed out for a call has a record.
@testset "no derivative ran without a record" begin
    @test Compiler.unrecorded_derivative_calls[] == 0
end
