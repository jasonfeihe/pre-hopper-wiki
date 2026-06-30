---
id: doc-ptx-isa
title: NVIDIA PTX ISA — Warp-Level MMA, ldmatrix, and cp.async
url: https://docs.nvidia.com/cuda/parallel-thread-execution/
source_category: official-doc
architectures: [sm75, sm86, sm89]
tags: [mma-sync, ldmatrix, cp-async, tensor-core]
retrieved_at: '2026-06-30'
scope_relevance: >-
  PTX-level reference for the warp MMA, ldmatrix, and cp.async instructions used
  by Turing/Ampere/Ada kernels — all pre-Hopper.
---

# NVIDIA PTX ISA (summary)

Reference for the Parallel Thread Execution virtual ISA. Instructions captured
for the pre-Hopper scope:

- **`mma.sync.aligned`**: warp-synchronous matrix-multiply-accumulate.
  Introduced for Turing (sm_75) Tensor Cores; the canonical PTX entry point for
  warp-level MMA on Turing/Ampere/Ada. All 32 lanes of the warp must participate
  (it is warp-synchronous), so the instruction must not be executed under
  divergent control flow. Operand shapes and supported types vary by compute
  capability.
- **`ldmatrix`**: collective shared-memory-to-register matrix load that fills
  the register fragments consumed by `mma.sync`, performing the
  transpose/replication required by the Tensor Core fragment layout. Available
  from sm_75.
- **`cp.async`** (`cp.async.ca` / `cp.async.cg`) with `cp.async.commit_group` /
  `cp.async.wait_group`: asynchronous global-to-shared copy. Introduced for
  Ampere (sm_80) and available on sm_86 and sm_89; **not** available on Turing
  (sm_75). The `.ca` variant caches the copied line in L1 and L2; the `.cg`
  variant bypasses L1 and caches in L2 only. `cp.async.wait_group N` waits until
  at most `N` of the issuing thread's committed copy groups remain outstanding;
  it is a per-thread completion primitive, not a block-wide barrier.

The ISA document specifies the per-architecture availability and the
commit/wait grouping semantics that software pipelines rely on.
