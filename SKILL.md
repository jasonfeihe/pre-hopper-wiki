---
name: PreHopperWiki
description: >-
  Use when the user asks about optimizing NVIDIA PRE-HOPPER GPU kernels — Turing
  (sm75, T4), Ampere (sm86, A10), or Ada Lovelace (sm89, L20/L40): warp-level
  mma.sync, ldmatrix, cp.async asynchronous copy, shared-memory double-buffering,
  Tensor Core dtypes (FP16/INT8 on Turing, BF16/TF32 on Ampere, FP8 on Ada), or
  Turing->Ampere->Ada migration. Do NOT use for Hopper (sm90) / Blackwell (sm100)
  features such as tcgen05/TMEM/CLC/NVFP4, for A100 (sm80), or for generic CUDA
  questions unrelated to pre-Hopper tensor-core kernels.
argument-hint: "[natural-language-question] | [--tag foo --type hardware] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# PreHopperWiki — Pre-Hopper Kernel Optimization Wiki

A structured, cross-referenced knowledge base of GPU kernel optimization for
**pre-Hopper** NVIDIA GPUs:

| Architecture | Compute capability | Example GPUs |
|--------------|--------------------|--------------|
| Turing       | `sm75`             | T4           |
| Ampere       | `sm86`             | A10          |
| Ada Lovelace | `sm89`             | L20, L40     |

> Scope boundary: Hopper (`sm90`), Blackwell (`sm100`), and A100 (`sm80`) are
> **out of scope**. So are distributed-systems topics. This wiki is about
> pre-Hopper tensor-core kernel optimization.

## When To Use This Skill

Trigger this skill when the user asks about:

- **Turing (sm75 / T4)**: warp-level `mma.sync`, `ldmatrix`, FP16/INT8/INT4
  Tensor Cores, synchronous shared-memory staging.
- **Ampere (sm86 / A10)**: `cp.async` asynchronous global→shared copy,
  software-pipelined staging, BF16/TF32 Tensor Cores.
- **Ada Lovelace (sm89 / L20, L40)**: FP8 (E4M3/E5M2) Tensor Cores on top of the
  Ampere feature set.
- **Cross-architecture techniques**: shared-memory double-buffering,
  software pipelining, bank-conflict avoidance.
- **Migration**: porting kernels Turing → Ampere → Ada.

Do NOT use this skill for Hopper/Blackwell features, A100/sm80, or generic CUDA
questions unrelated to pre-Hopper tensor cores.

## How To Query

All commands run through `uv` from the repository root (the scripts auto-resolve
the knowledge-base root; no environment variable is required):

### Path 1 — unified search (natural language + filters)

```bash
uv run python scripts/query.py "shared memory double buffering"
uv run python scripts/query.py --tag mma-sync --type hardware
uv run python scripts/query.py --architecture T4          # alias → sm75
uv run python scripts/query.py --architecture Ada --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag` and
`--architecture` are alias-aware (e.g. `--architecture L40` matches `sm89`,
`--tag mma.sync` matches `mma-sync`).

### Path 2 — fetch a specific page by id or path

```bash
uv run python scripts/get_page.py hw-mma-sync-turing
uv run python scripts/get_page.py hw-cp-async-ampere --frontmatter-only
uv run python scripts/get_page.py technique-shared-memory-double-buffering --follow-sources
```

### Path 3 — regex text search

```bash
uv run python scripts/grep_wiki.py "cp\.async" --only wiki
uv run python scripts/grep_wiki.py "bank conflict" --context 3
```

### Path 4 — pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-hardware-feature.md`
- `queries/by-technique.md`
- `queries/by-kernel-type.md`
- `queries/by-language.md`
- `queries/by-problem.md`
- `queries/by-repo.md`

## Output Pattern

When answering from this knowledge base:

1. Cite specific pages by path (e.g. `wiki/hardware/mma-sync-turing.md`) and id
   (`hw-mma-sync-turing`).
2. Follow `sources:` ids to trace claims back to the cited NVIDIA docs.
3. Respect `confidence`: `verified` > `source-reported` > `inferred` >
   `experimental`.
4. Keep architecture/feature pairings correct — e.g. `cp.async` is Ampere+, not
   Turing.

See `README.md` for installation and the add-a-page workflow, `CLAUDE.md` and
`references/schema.md` for the schema and controlled vocabulary, and
`references/incremental-updates.md` for how the corpus grows over time.
