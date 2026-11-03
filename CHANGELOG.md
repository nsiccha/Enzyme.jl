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
- Default dogfood derivative arithmetic to strict floating-point semantics.
  This prevents reassociation from underflowing representable gradients such
  as `exp(log(t * 1e-190))`. Explicit `Enzyme.API.fast_math!(true)` remains
  available before derivative compilation.

## 0.13.217

- Add `WithGradient(primal, value_and_gradient)` for explicitly supplied
  derivatives of scalar-valued functions, including opaque native calls.
  Supports Forward and Reverse modes, batching, and reusable gradient buffers.
- Repair constant native function-pointer activity analysis and diagnose
  missing derivatives before handling indirect-call shadows. Avoid an
  out-of-bounds parameter-attribute query when optimizing an indirect call.
  Preserve inactive local buffers and symbol lookup, and report missing reverse
  derivatives outside the augmented primal's exception handlers.
