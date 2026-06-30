# Provenance & Attribution

## Design lineage

The architecture of this knowledge base — its three-layer
`sources/` → `wiki/` → `queries/` model, controlled-vocabulary YAML frontmatter
schema, schema/vocabulary/link validator, deterministic query-index
regenerator, and keyword/page/grep retrieval CLIs — is **inspired by** the
KernelWiki Blackwell/Hopper kernel-optimization knowledge base
(`~/rd/auto-cuda/KernelWiki`, reference snapshot reviewed **2026-06-30**).

## No source files were copied

All tooling in `scripts/` and all schema/vocabulary files in `data/` are
**original code written for this repository**. No KernelWiki source file was
copied, vendored, or forked into this repository. The reference knowledge base
was consulted only as a *design reference* for behavior and structure; the
implementations here are independent.

This is a deliberate decision (recorded as **DEC-4** in the implementation
plan): reimplementing from scratch keeps the codebase fully owned by this
project and avoids importing another repository's licensing/attribution
obligations.

### What differs from the reference, by design

- **Scope**: pre-Hopper GPUs only — Turing (`sm75`/T4), Ampere (`sm86`/A10),
  Ada Lovelace (`sm89`/L20, L40). The reference is Blackwell/Hopper-first.
- **No "Blackwell-first" policy**: there is no `blackwell_relevance` field and no
  validator rule requiring one. A neutral, optional `scope_relevance` field
  replaces it.
- **Root resolution** uses the `PREHOPPER_WIKI_ROOT` environment variable; there
  is no functional dependency on any Blackwell-named variable.
- **Environment**: `uv` + Python 3.12 (`pyproject.toml` + `uv.lock` +
  `.python-version`) rather than `pip` + `requirements.txt`.
- **Operational/ingestion machinery** (PR fetching, candidate ledgers,
  verbatim-asset provenance automation, freshness checking) is intentionally not
  reimplemented in this scaffold; it is future work.

## Licensing

This repository is licensed under the MIT License (see `LICENSE`). Because no
third-party source files are included, there are no external code-license
obligations to carry. Wiki content cites upstream NVIDIA documentation and other
public sources by URL in each page's `sources:` frontmatter; those citations are
references, not redistributions of the cited material.
