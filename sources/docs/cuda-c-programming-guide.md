---
id: doc-cuda-c-programming-guide
title: NVIDIA CUDA C++ Programming Guide — Compute Capabilities & Async Copy
url: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
source_category: official-doc
architectures: [sm75, sm86, sm89]
tags: [tensor-core, cp-async, shared-memory, mma-sync]
retrieved_at: '2026-06-30'
scope_relevance: >-
  Canonical reference for per-compute-capability features across Turing (7.5),
  Ampere (8.6), and Ada Lovelace (8.9) — all in pre-Hopper scope.
---

# NVIDIA CUDA C++ Programming Guide (summary)

Authoritative reference for the CUDA programming model and the per-architecture
feature deltas relevant to this knowledge base.

Key points captured for the pre-Hopper scope:

- **Compute Capability 7.5 (Turing, T4)**: second-generation Tensor Cores with
  FP16, INT8, INT4 and INT1 matrix operations. Asynchronous `cp.async`-style
  global-to-shared copy is **not** available; shared-memory staging goes through
  ordinary load-then-store sequences.
- **Compute Capability 8.0/8.6 (Ampere; 8.6 = A10)**: third-generation Tensor
  Cores add BF16 and TF32; the `cp.async` family of instructions provides
  asynchronous global-to-shared copies that bypass the register file and enable
  software-pipelined shared-memory staging.
- **Compute Capability 8.9 (Ada Lovelace, L20/L40)**: fourth-generation Tensor
  Cores add FP8 (E4M3 and E5M2) matrix operations on top of the Ampere feature
  set, while retaining `cp.async`.

The guide documents the asynchronous-copy programming model (including the
arrive/wait barrier usage) and the per-compute-capability technical
specifications used throughout this wiki.
