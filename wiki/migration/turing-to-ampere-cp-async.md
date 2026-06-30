---
id: migration-turing-to-ampere-cp-async
title: Migrating Turing Shared-Memory Staging to Ampere cp.async
type: migration
from_arch: sm75
to_arch: sm86
tags: [cp-async, double-buffering, shared-memory-optimization, async-copy]
related: [hw-mma-sync-turing, hw-cp-async-ampere, technique-shared-memory-double-buffering]
sources: [doc-ptx-isa, doc-cuda-c-programming-guide]
confidence: source-reported
reproducibility: pseudocode
prerequisites: [hw-cp-async-ampere]
---

# Migrating Turing Shared-Memory Staging to Ampere `cp.async`

This page describes porting a GEMM-style staging loop from Turing (sm75) to
Ampere (sm86), replacing synchronous register-staged loads with asynchronous
`cp.async` copies.

> This page intentionally carries no `scope_relevance` field: both `from_arch`
> (sm75) and `to_arch` (sm86) are in-scope, so no scope justification is needed.

## Before (Turing, sm75)

```text
# global -> register -> shared, then a full barrier
reg = global_load(A_tile_ptr)
smem[stage] = reg
__syncthreads()
compute(smem[stage])     # ldmatrix + mma.sync
```

The load occupies registers and the `__syncthreads()` sits on the critical path.

## After (Ampere, sm86)

```text
# global -> shared directly, no registers, non-blocking
cp.async.cg smem[stage], A_tile_ptr
cp.async.commit_group
... issue next group ...
cp.async.wait_group 1     # this thread's async copies into smem are complete
__syncthreads()           # STILL REQUIRED: make all threads' smem writes
                          # visible block-wide before any lane consumes the tile
compute(smem[stage])      # ldmatrix + mma.sync unchanged
```

> Important: `cp.async.wait_group` only drains the **issuing thread's** async
> copy groups — it is *not* a block-wide barrier. Shared-memory tiling still
> needs a CTA barrier (`__syncthreads()`, or an `mbarrier`/`cuda::pipeline`
> barrier) before other warps read the staged tile. `cp.async` removes the
> register round-trip and lets copies overlap compute; it does **not** remove
> the need to synchronize the block.

## What changes and what does not

- **Changes**: the staging mechanism (synchronous register round-trip →
  `cp.async` groups). Pipelines can go deeper than two stages, because copies
  proceed without occupying registers or blocking the issuing warp.
- **Unchanged**: the compute step (`ldmatrix` + `mma.sync`), the overall
  double-buffering structure ([[technique-shared-memory-double-buffering]]),
  and the need for a `__syncthreads()` (or equivalent CTA barrier) before the
  block consumes a freshly staged tile. What `cp.async.wait_group` adds is a
  per-thread completion check for the async copies; it complements, and does not
  replace, the block barrier.

## Caveat

`cp.async` does not exist on Turing; this migration only applies in the
sm75 → sm86 direction. The reverse (running an Ampere kernel on a T4) requires
falling back to the synchronous path described in [[hw-mma-sync-turing]].

## References

- PTX ISA, `cp.async` family (`doc-ptx-isa`).
- CUDA C++ Programming Guide, asynchronous-copy model (`doc-cuda-c-programming-guide`).
