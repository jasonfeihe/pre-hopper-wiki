# Seed Candidates — Reviewer Notes & Evidence Rationale

This document records the human review behind the hand-triaged candidate ledgers
(`candidates/*.yaml`) and the committed seed corpus (`tests/fixtures/seed/`). It
is the evidence trail for the inclusion decisions and the known coverage gaps.

## Scope recap

- In-scope architectures: **sm75 (Turing/T4), sm86 (Ampere/A10), sm89 (Ada/L20,L40)**.
- **sm80 / A100 is out of scope** (skip reason `sm80-only`) and is never mapped
  to sm86, even though the discovery window (2020+) spans the A100 era. Widening
  is future work (FUT-4).
- Domain: all GPU kernel optimization (DEC-4), including vector-search (cuVS).

## Included seed PRs (materialized as `source-pr` pages)

| Page id | Repo / PR | Arch | Evidence |
|---------|-----------|------|----------|
| `pr-vllm-29901` | vllm-project/vllm #29901 | sm75 | Title "marlin kernel support for turing (sm75)"; diff guards `__CUDA_ARCH__ == 750` with `mma.sync` + `ldmatrix` (direct-sm + arch-guard). |
| `pr-flashinfer-385` | flashinfer-ai/flashinfer #385 | sm86 | "Fix invalid kernel configuration for sm86"; diff guards `__CUDA_ARCH__ == 860` for the A10 100 KB/SM shared-memory cap (direct-sm + arch-guard). |
| `pr-flashinfer-1973` | flashinfer-ai/flashinfer #1973 | sm89 | "Add support for **L40** FusedMoE in cutlass path"; sm_89 CUTLASS FP8 path (direct-device + direct-sm). |

Each cites a real upstream URL and merge SHA in its fixture; the page text is
synthesized from the committed fixture, not fetched live.

## Evidence rationale (why these and not others)

- **Direct, optimization-target evidence only.** A defensible `sm75`/`sm86`/`sm89`
  label requires the PR to *target* that architecture for kernel work — an
  explicit `sm_75`/`sm_86`/`sm_89` / T4 / A10 / L20 / L40 mention, or an
  arch-guard code path naming the SM. A support-matrix listing or a
  "not supported on Turing" guard does **not** qualify.
- **The vLLM false-positive pattern.** vLLM surfaces many raw "Turing"/"sm75"
  hits that are capability guards / fallbacks ("Turing is not supported",
  "fall back to ..."). The classifier's `capability-guard-only` skip reason and
  `capability_guard_markers` exist precisely to reject these; only #29901, which
  *adds* a Turing kernel, was included.
- **sm90/sm100 excluded.** e.g. flashinfer #2157 ("fix xqa mha_sm90.cu") is
  `hopper-only`.

## Known coverage gaps (handed off to future curation)

- Only FlashInfer and vLLM ledgers are hand-triaged this loop; cutlass, sglang,
  pytorch, tensorrt-llm, and cuvs ledgers are valid stubs (`prs: []`) awaiting a
  curation pass (FUT-5).
- cuVS vector-search vocabulary exists (`knn`, `ivf`, `pq`, `graph-search`, …)
  but no cuVS PR is seeded yet; the empirical pilot suggests pre-Hopper-specific
  cuVS kernel PRs are sparse and need a dedicated search pass.
- The empirical reality (recorded during planning) is that genuine pre-Hopper
  kernel-optimization PRs are a thin slice of 2020+ activity; the high-quality
  skip ledger is itself a deliverable, proving the corpus boundary.

## How to extend

See `references/ingestion.md` for the discover → classify → generate → skip-log
workflow and the committed-fixture contract.
