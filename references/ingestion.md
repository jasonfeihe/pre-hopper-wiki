# Content-Ingestion Pipeline

How real upstream-repo PRs become validated `source-pr` pages in this knowledge
base. The pipeline is **policy-first** (a declarative inclusion policy decides
what is genuinely pre-Hopper-relevant) and **offline-deterministic** (every stage
runs from committed fixtures; live GitHub is opt-in and never used by tests/CI).

## Scope (resolved decisions)

- **Architectures**: exactly `sm75` (Turing/T4), `sm86` (Ampere/A10),
  `sm89` (Ada/L20, L40). **`sm80`/A100 is out of scope** (skip reason
  `sm80-only`) and is never mapped to `sm86`. Widening is future work.
- **Domain**: all GPU kernel optimization, including vector-search (cuVS):
  `knn`, `vector-search`, `ivf`, `pq`, `graph-search`, etc. in
  `data/tags.yaml`.
- **Discovery window**: `2020-01-01` → present (configurable via
  `--since`/`--until`).

## The four stages

```
discover ──▶ classify ──▶ generate ──▶ (skip-log)
(refresh)    (policy)      (pages)       (audit)
```

1. **Discover** — `scripts/refresh_candidate_ledger.py` surfaces candidate PRs
   and merges them into `candidates/<repo_slug>.yaml` as `decision: defer`
   (existing decisions are never rewritten). It records what each refresh saw in
   `data/refresh-search-results.yaml`.
2. **Classify** — `scripts/classify_candidate.py` applies
   `data/inclusion-policy.yaml` to a candidate + its committed metadata/diff
   fixture and returns an `include` verdict (with `architecture_evidence`) or a
   `skip` verdict (a reason from the policy's `skip_reasons` taxonomy).
3. **Generate** — `scripts/generate-pr-pages.py` reads a seed manifest, runs the
   classifier, and writes a schema-valid `sources/prs/<repo_slug>/PR-<N>.md` page
   on `include`, or appends a row to `data/pr-page-skipped.yaml` on `skip`.
4. **Validate + index** — `scripts/validate.py` then `scripts/generate-indices.py`
   (both already gate on the schema/vocabulary).

## What counts as defensible pre-Hopper relevance

A page earns an `sm75`/`sm86`/`sm89` label only with a **direct optimization-target
signal** (`data/inclusion-policy.yaml::evidence_tiers`):

- `direct-sm-mention` — explicit `sm_75`/`sm_86`/`sm_89` (or compute capability
  7.5/8.6/8.9) as the thing being optimized.
- `direct-device-mention` — explicit T4 / A10 / L20 / L40 (or Turing/Ada as the
  kernel target). "Ampere" alone is weak (it spans the out-of-scope A100/sm80).
- `arch-guard-codepath` — a `__CUDA_ARCH__ == 750/860/890` guard whose taken
  branch is kernel work.

A mention that appears **only inside a capability guard** ("Turing is not
supported", "fall back to ...") is `capability-guard-only`, not an include — this
is the empirically-common vLLM false-positive pattern. Architecture-token
matching is word-boundary aware, so `A10` does **not** match inside `A100`.

## Offline / fixture contract

- **Discovery** default mode replays committed `tests/fixtures/gh/<slug>.json`
  search responses. `--live` calls `gh search prs` for real; it is opt-in and
  excluded from tests/CI.
- **Classification & generation** read committed candidate fixtures under
  `tests/fixtures/seed/<slug>/PR-<N>.json` (PR metadata + a bounded diff/body
  excerpt). `captured_at` comes from the seed manifest, never the clock, so
  output is byte-stable.
- The seed manifest is `tests/fixtures/seed/seed-manifest.yaml`; each entry names
  a real PR URL/number (for provenance) and points at its committed fixture.

> Tests never touch the network. `tests/test_ingestion.py` includes a guard that
> runs the scripts with `gh` unavailable and asserts success.

## Commands

```bash
# Discover (fixture mode, default) — merges new candidates as defer:
uv run python scripts/refresh_candidate_ledger.py --repos cutlass,flashinfer --searched-at 2026-06-30

# Discover (live, opt-in — NOT used by tests/CI):
uv run python scripts/refresh_candidate_ledger.py --live --cutoff 2026-06-30 --since 2020-01-01

# Classify a single candidate fixture (inspect a verdict):
uv run python scripts/classify_candidate.py --fixture tests/fixtures/seed/vllm/PR-29901.json

# Generate source-pr pages from the seed manifest:
uv run python scripts/generate-pr-pages.py --dry-run      # preview
uv run python scripts/generate-pr-pages.py                # write

# Validate + regenerate indices:
uv run python scripts/validate.py
uv run python scripts/generate-indices.py

# Full offline test suite:
uv run python -m unittest discover -s tests
```

## Adding a new source-PR page

1. Identify a real PR with defensible in-scope architecture evidence.
2. Add a committed fixture `tests/fixtures/seed/<slug>/PR-<N>.json` (metadata +
   bounded diff/body excerpt that contains the evidence).
3. Add an entry to `tests/fixtures/seed/seed-manifest.yaml` (provenance + curated
   tags; tags must be in `data/tags.yaml`).
4. `uv run python scripts/generate-pr-pages.py`, then `validate.py` +
   `generate-indices.py`.
5. Record the reviewer rationale in `docs/seed-candidates.md` and reflect the
   decision in the repo's `candidates/<slug>.yaml` ledger.

## Related files

- `data/inclusion-policy.yaml` — evidence tiers + skip taxonomy.
- `data/schemas.yaml` — `inclusion-policy`, `candidate-ledger`,
  `refresh-search-results`, `pr-page-skipped-audit` schema entries.
- `candidates/*.yaml` + `candidates/README.md` + `candidates/tracked-repos.txt`.
- `docs/seed-candidates.md` — reviewer notes + evidence rationale.
- `references/incremental-updates.md` — the underlying append-only model.

## Deferred (future work)

Live at-scale crawling of the full corpus; an artifact/diff provenance subsystem
(`fetch_pr_diff` + `artifacts/` bundles + verbatim verification); an automated
blog/doc scraper; broader architecture coverage (A100/`sm80`); and deep
full-corpus curation beyond the seed and the reviewed ledgers.
