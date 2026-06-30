---
id: hw-mma-sync-turing
title: Turing Warp-Level MMA (mma.sync) on T4
type: hardware
architectures: [sm75]
tags: [mma-sync, tensor-core, ldmatrix, fp16, int8]
confidence: verified
related: [technique-shared-memory-double-buffering]
sources: [doc-ptx-isa, doc-cuda-c-programming-guide]
aliases: [mma.sync, MMA, warp MMA]
hardware_features: [mma-sync, tensor-core, ldmatrix]
scope_relevance: >-
  Turing (sm75 / T4) is the earliest in-scope architecture; mma.sync is its
  defining tensor-core programming path and the baseline that later
  architectures extend.
---

# Turing Warp-Level MMA (`mma.sync`) on T4

Turing (compute capability 7.5, e.g. the Tesla **T4**) exposes its
**second-generation Tensor Cores** to PTX through the warp-synchronous
`mma.sync.aligned` instruction. A full warp cooperatively supplies the A, B, and
C/D matrix fragments, and the instruction computes `D = A * B + C` across the
warp's 32 lanes.

## Programming model

- Register fragments for A and B are typically loaded from shared memory with
  **`ldmatrix`**, which performs the transpose/replication needed by the Tensor
  Core fragment layout.
- Supported input types on Turing include **FP16** (with FP16 or FP32
  accumulation), **INT8**, and **INT4**.
- The instruction is *warp-synchronous*: all 32 lanes must participate, so
  divergence around an `mma.sync` is a correctness error, not just a performance
  problem.

## What Turing does NOT have

Turing has **no `cp.async`** asynchronous copy. Staging operand tiles into
shared memory therefore goes through ordinary global loads into registers
followed by shared stores, which keeps the load latency on the critical path.
Hiding that latency requires classic double-buffering with explicit
synchronization (see [[technique-shared-memory-double-buffering]]); the
asynchronous-copy pipeline that Ampere introduces is unavailable here.

## References

- PTX ISA: `mma.sync.aligned`, `ldmatrix` (`doc-ptx-isa`).
- CUDA C++ Programming Guide, compute-capability 7.5 specifications
  (`doc-cuda-c-programming-guide`).
