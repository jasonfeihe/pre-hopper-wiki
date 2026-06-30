# Incremental Updates

The pre-Hopper kernel wiki is **append-only and not frozen to a single point in
time** (`知识库不用卡死到某一个时间`). New sources and pages are added over time
without rewriting or re-pinning the existing corpus, and the query indices are
regenerated deterministically from frontmatter.

This document is the contract for that process. It is fully exercisable with the
core tooling alone (`validate.py` + `generate-indices.py`); it does **not**
depend on any automated ingestion or freshness pipeline (that automation is
future work — see `## Future Work / Out of Scope` in the plan).

## The `refresh-cutoff.yaml` baseline

`data/refresh-cutoff.yaml` records a single `cutoff_date`: the date through which
the corpus has been deliberately reviewed.

- It is a **fresh baseline for this repository** — it is not inherited from any
  other knowledge base.
- Adding a page does **not** require changing `cutoff_date`. The date marks
  review rounds, not individual edits.
- Advancing `cutoff_date` records that a new deliberate review round has
  happened. The validator only checks that the field is present and is an ISO
  `YYYY-MM-DD` date.

## Stable ids

Every page carries a unique `id` with a type-specific prefix (`hw-`,
`technique-`, `kernel-`, `pattern-`, `lang-`, `migration-`, `doc-`, `blog-`,
`pr-`, `contest-`). Ids are the stable handles other pages link to via
`sources:` and `related:`. Once published, **an id is never reused or
repurposed**; renaming a page means adding a new id, not mutating an old one.
The validator enforces id uniqueness and that every `sources:`/`related:`
reference resolves, so a dropped or renamed id surfaces immediately as a broken
link rather than silently disappearing.

## Adding a source or page (the workflow)

1. **Create the page** under the correct layer and directory:
   - a raw source under `sources/{docs,blogs,prs,contests}/`, or
   - a synthesized page under `wiki/{hardware,techniques,kernels,patterns,languages,migration}/`.
2. **Write valid frontmatter**: a unique prefixed `id`, the required fields for
   the page type (see `references/schema.md`), only in-scope architectures
   (`sm75`/`sm86`/`sm89`), and only controlled-vocabulary tags. A synthesized
   wiki page must cite at least one existing source in `sources:`.
3. **Validate**:
   ```bash
   uv run python scripts/validate.py
   ```
4. **Regenerate the indices**:
   ```bash
   uv run python scripts/generate-indices.py
   ```
5. Commit the new page together with the regenerated `queries/*.md`.

## Guarantees

- **Prior content is preserved.** Adding a page never edits existing pages;
  their ids and bodies are untouched. The new page simply appears in the
  regenerated indices alongside the existing entries.
- **Deterministic regeneration.** `generate-indices.py` collects pages in sorted
  path order and emits every grouping in sorted key order, so regenerating twice
  with no content change produces no diff. A hand-edited index is restored on the
  next regeneration — the indices are generated artifacts, not hand-maintained.
- **No silent loss.** Removing or renaming an existing id between runs is
  reported by the validator as a broken `sources:`/`related:` link, so provenance
  cannot be dropped unnoticed.

The `tests/test_tooling.py` suite includes an end-to-end check
(`IncrementalUpdateTests`) that seeds a small corpus, validates and indexes it,
then adds one new source page and one new wiki page, re-runs validate +
generate-indices, and asserts that (a) all previously-present ids still resolve,
(b) the new page appears in the regenerated indices, and (c) a second
regeneration produces no further diff — all using only the core tooling.
