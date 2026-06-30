---
id: technique-shared-memory-double-buffering
title: Shared-Memory Double-Buffering for GEMM Tiles
type: technique
architectures: [sm75, sm86, sm89]
tags: [double-buffering, software-pipelining, shared-memory-optimization, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mma-sync-turing]
related: [hw-mma-sync-turing, hw-cp-async-ampere]
sources: [doc-cuda-c-programming-guide, doc-ptx-isa]
techniques: [double-buffering, software-pipelining, shared-memory-optimization]
hardware_features: [shared-memory]
scope_relevance: >-
  Double-buffering is the cross-architecture staging technique that links the
  Turing synchronous path to the Ampere/Ada asynchronous one; applies across all
  in-scope architectures.
---

# Shared-Memory Double-Buffering for GEMM Tiles

Double-buffering structures a GEMM main loop around **two shared-memory
buffers** so that staging the *next* operand tile can overlap consuming the
current one. On Ampere/Ada the staging copy (`cp.async`) proceeds without
blocking the issuing warp, which is what enables the latency of the next load to
overlap the current tile's Tensor Core math; on Turing the staging is
synchronous, so the structure mainly reduces redundant `__syncthreads()` rather
than overlapping a non-blocking copy.

## Two-stage loop (compilable snippet)

The following Turing-compatible CUDA C++ fragment double-buffers FP16 operand
tiles through two shared-memory buffers. It compiles for `sm_75` with
`nvcc -arch=sm_75`; a stub `compute_tile` stands in for the
`ldmatrix`/`mma.sync` detail covered in [[hw-mma-sync-turing]].

```cuda
// nvcc -arch=sm_75 -c double_buffer.cu
#include <cuda_fp16.h>

constexpr int TILE = 1024;  // elements per operand tile

// Stub: the real implementation issues ldmatrix + mma.sync over the tile.
__device__ void compute_tile(const __half* smem_tile) {
    (void)smem_tile;
}

__global__ void gemm_main_loop(const __half* __restrict__ gmem, int K) {
    __shared__ __half smem[2][TILE];
    const int t = threadIdx.x;

    // Prologue: stage tile 0 (synchronous global->register->shared on Turing).
    for (int i = t; i < TILE; i += blockDim.x)
        smem[0][i] = gmem[i];
    __syncthreads();                           // tile 0 visible block-wide

    for (int k = 0; k < K; ++k) {
        const int cur  = k & 1;
        const int next = (k + 1) & 1;
        if (k + 1 < K) {                       // prefetch the next tile
            const __half* src = gmem + (k + 1) * TILE;
            for (int i = t; i < TILE; i += blockDim.x)
                smem[next][i] = src[i];
        }
        compute_tile(smem[cur]);               // consume current tile
        __syncthreads();                       // next buffer filled + safe to reuse
    }
}
```

## How the load step differs by architecture

- **Turing (sm75)**: the "begin loading" step is an ordinary global load into
  registers followed by a shared store; the `sync` is a full `__syncthreads()`.
  Latency hiding is limited to two stages in practice (see
  [[hw-mma-sync-turing]]).
- **Ampere (sm86) / Ada (sm89)**: the load step becomes a non-blocking
  **`cp.async`** group, drained per-thread with `cp.async.wait_group`, which
  removes the register round-trip and supports deeper (3+ stage) pipelines (see
  [[hw-cp-async-ampere]]). `cp.async.wait_group` is **not** a block barrier — a
  `__syncthreads()` (or `mbarrier`/`cuda::pipeline` barrier) is still required
  before other warps consume the staged tile.

The compute step (`ldmatrix` + `mma.sync`) is the same shape across all three
architectures; only the staging mechanism changes.

## References

- CUDA C++ Programming Guide, asynchronous-copy and shared-memory model
  (`doc-cuda-c-programming-guide`).
- PTX ISA, `cp.async` commit/wait grouping (`doc-ptx-isa`).
