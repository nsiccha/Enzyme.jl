# Changelog

## Unreleased

- Reduce redundant reverse caches for fresh array buffers on Julia 1.10.
  Merged empty-Memory paths on later Julia versions remain conservative.
- Keep that reduction through a ReactiveKernels/BRM model kernel, which
  applies `vec` to the call's result and passes its inputs as fields of an
  aggregate argument, and after a broadcast that allocates its output on each
  branch. Writes into objects allocated after the call, writes through a
  header merged from such allocations, and calls that write only memory they
  create no longer mark the call's inputs as overwritten.
- Keep it through later calls that write only arrays created after the call:
  a function that writes the elements of an array it allocates (a broadcast
  or a readout into `similar` that Julia does not inline), a function that
  returns such an array, and a function that writes an argument which is
  such an array, as `sum(A; dims)` does through `reducedim_initarray` and
  `_mapreducedim!`. Splats of tuples or arrays into `tuple` through
  `Core._apply_iterate` count as writing only their result. Through a
  ReactiveKernels/BRM model kernel this cuts the reverse storage of a
  multi-channel field consumed that way from 113-1,003 to 7-29 bytes per
  row-feature on Julia 1.10.
- Default dogfood derivative arithmetic to strict floating-point semantics.
  This prevents reassociation from underflowing representable gradients such
  as `exp(log(t * 1e-190))`. Explicit `Enzyme.API.fast_math!(true)` remains
  available before derivative compilation.
- Compile and log from generated functions without calling a logger that has
  no methods in the running world, such as Test's `TestLogger` when Test loads
  after Enzyme. On Julia 1.10 such a task logger made nested derivatives fail
  for the rest of the process, and on Julia 1.10 and 1.12 such a global logger
  broke derivatives. Messages Enzyme logs while compiling are now written to
  `Core.stderr` by `SimpleLogger`, without color.

## 0.13.217

- Add `WithGradient(primal, value_and_gradient)` for explicitly supplied
  derivatives of scalar-valued functions, including opaque native calls.
  Supports Forward and Reverse modes, batching, and reusable gradient buffers.
- Repair constant native function-pointer activity analysis and diagnose
  missing derivatives before handling indirect-call shadows. Avoid an
  out-of-bounds parameter-attribute query when optimizing an indirect call.
  Preserve inactive local buffers and symbol lookup, and report missing reverse
  derivatives outside the augmented primal's exception handlers.
