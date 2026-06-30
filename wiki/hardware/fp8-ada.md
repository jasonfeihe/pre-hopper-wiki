---
id: hw-fp8-ada
title: Ada Lovelace FP8 Tensor Cores on L40
type: hardware
architectures: [sm89]
tags: [fp8, tensor-core, cp-async, bf16, tf32]
confidence: verified
related: [hw-cp-async-ampere]
sources: [doc-cuda-c-programming-guide]
aliases: [FP8, E4M3, E5M2]
hardware_features: [fp8, tensor-core]
scope_relevance: >-
  FP8 (E4M3/E5M2) Tensor Core support is Ada Lovelace's signature addition over
  Ampere within the pre-Hopper range; L20/L40 are sm89.
---

# Ada Lovelace FP8 Tensor Cores on L40

Ada Lovelace (compute capability 8.9, e.g. **L20** and **L40**) carries
**fourth-generation Tensor Cores** whose headline addition over Ampere is native
**FP8** matrix support in two encodings:

- **E4M3** — 4 exponent bits, 3 mantissa bits: higher precision, narrower
  dynamic range.
- **E5M2** — 5 exponent bits, 2 mantissa bits: wider dynamic range, lower
  precision.

The choice between encodings is a precision/range tradeoff; specific
training/inference usage conventions are out of scope for this hardware page.

## Inherited from Ampere

Ada retains the Ampere feature set this KB documents elsewhere — including
**`cp.async`** asynchronous global-to-shared copy ([[hw-cp-async-ampere]]) and
the **BF16 / TF32** Tensor Core input types. FP8 is layered on top of, not in
place of, those capabilities.

## Scope note

FP8 here refers to the **Tensor Core compute** datatype available on sm89. It is
distinct from the later Hopper/Blackwell tensor-core paths, which are out of
scope for this pre-Hopper knowledge base.

## References

- CUDA C++ Programming Guide, compute-capability 8.9 specifications and FP8
  Tensor Core types (`doc-cuda-c-programming-guide`).
