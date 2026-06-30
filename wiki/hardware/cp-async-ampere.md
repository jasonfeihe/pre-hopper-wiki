---
id: hw-cp-async-ampere
title: Ampere Asynchronous Copy (cp.async) on A10
type: hardware
architectures: [sm86]
tags: [cp-async, async-copy, shared-memory, tensor-core, bf16, tf32]
confidence: verified
related: [technique-shared-memory-double-buffering, migration-turing-to-ampere-cp-async]
sources: [doc-ptx-isa, doc-cuda-c-programming-guide]
aliases: [cp.async, async copy]
hardware_features: [cp-async, async-copy, shared-memory]
scope_relevance: >-
  cp.async is Ampere's signature memory-pipeline feature and the key capability
  the A10 (sm86) adds over Turing within the pre-Hopper range.
---

# Ampere Asynchronous Copy (`cp.async`) on A10

Ampere (compute capability 8.6, e.g. the **A10**) introduces the **`cp.async`**
instruction family: asynchronous copies from global memory directly into shared
memory that **bypass the register file** and do not block the issuing warp.

## Programming model

- `cp.async.ca` / `cp.async.cg` issue the copy (`.ca` caches in L1+L2, `.cg`
  bypasses L1).
- Copies are grouped and drained with `cp.async.commit_group` and
  `cp.async.wait_group N`, letting a kernel keep up to several copy groups in
  flight.
- Because the data lands in shared memory without occupying registers, a kernel
  can prefetch the next operand tile while the Tensor Cores consume the current
  one — the foundation of multi-stage software pipelines
  ([[technique-shared-memory-double-buffering]]).

Ampere also adds **BF16** and **TF32** Tensor Core input types on top of the
Turing FP16/INT8/INT4 set.

## Contrast with Turing

On Turing (sm75) the same staging must be done synchronously through registers
(see [[hw-mma-sync-turing]]); `cp.async` is unavailable there. Porting a Turing
staging loop to Ampere's asynchronous pipeline is covered in
[[migration-turing-to-ampere-cp-async]].

## References

- PTX ISA: `cp.async`, `cp.async.commit_group`, `cp.async.wait_group`
  (`doc-ptx-isa`).
- CUDA C++ Programming Guide, asynchronous-copy model and compute-capability 8.6
  specifications (`doc-cuda-c-programming-guide`).
