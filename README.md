# pre-hopper-wiki

A structured, cross-referenced knowledge base of GPU kernel optimization for
**pre-Hopper** NVIDIA GPUs — Turing (`sm75`, T4), Ampere (`sm86`, A10), and Ada
Lovelace (`sm89`, L20/L40) — packaged as a Claude Code skill.

It follows a three-layer `sources/` → `wiki/` → `queries/` model with
controlled-vocabulary YAML frontmatter, a validator, an index generator, and
retrieval CLIs. The tooling is **original code** scoped to pre-Hopper
architectures; its design lineage and attribution are documented in
`PROVENANCE.md`.

> Scope: only Turing/Ampere/Ada (`sm75`/`sm86`/`sm89`). Newer and other
> architectures are out of scope.

## Environment (uv + Python 3.12)

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.12. There is a
single runtime dependency, PyYAML.

```bash
uv sync            # create .venv and install PyYAML from uv.lock
uv run python -c "import yaml; print(yaml.__version__)"
```

`pyproject.toml` pins `requires-python = ">=3.12,<3.13"`; `.python-version` pins
`3.12`; `uv.lock` is committed for reproducible installs (`uv sync --frozen`).

## Layout

```
pre-hopper-wiki/
├── SKILL.md                     # Claude Code skill entry point
├── README.md                    # this file
├── CLAUDE.md                    # schema + navigation reference
├── PROVENANCE.md                # design lineage / attribution / licensing
├── pyproject.toml, uv.lock, .python-version
├── data/
│   ├── schemas.yaml             # page-type schemas
│   ├── tags.yaml                # controlled vocabulary (architectures, features, …)
│   ├── aliases.yaml             # device/instruction aliases → canonical tags
│   ├── version-claims.yaml      # version-sensitivity registry (stub)
│   └── refresh-cutoff.yaml      # incremental-update baseline date
├── scripts/
│   ├── _wiki_root.py            # shared root resolver (PREHOPPER_WIKI_ROOT)
│   ├── validate.py              # schema + vocabulary + link validator
│   ├── generate-indices.py      # regenerates queries/*.md
│   ├── query.py                 # unified keyword/filter search
│   ├── get_page.py              # fetch a page by id or path
│   └── grep_wiki.py             # regex text search
├── sources/{docs,blogs,prs,contests}/    # Layer 1: raw sources
├── wiki/{hardware,techniques,kernels,patterns,languages,migration}/  # Layer 2
├── queries/                     # Layer 3: auto-generated indices
├── references/{schema.md,incremental-updates.md}
└── tests/                       # test suite + invalid fixtures
```

## Usage

```bash
# Search
uv run python scripts/query.py "shared memory double buffering"
uv run python scripts/query.py --architecture T4 --compact     # alias → sm75

# Fetch a page
uv run python scripts/get_page.py hw-cp-async-ampere

# Regex search
uv run python scripts/grep_wiki.py "cp\.async" --only wiki

# Validate and regenerate indices
uv run python scripts/validate.py
uv run python scripts/generate-indices.py

# Run the test suite
uv run python -m unittest discover -s tests -v
```

## Workflows

### Add a source

1. Create `sources/docs/<slug>.md` (or `blogs/`, `prs/`, `contests/`) with valid
   frontmatter: a unique `doc-`/`blog-`/`pr-`/`contest-` id, `source_category`,
   in-scope `architectures`, controlled-vocabulary `tags`, and a date field.
2. `uv run python scripts/validate.py`
3. `uv run python scripts/generate-indices.py`

### Add a wiki page

1. Create `wiki/<type>/<slug>.md` with valid frontmatter: a unique
   `hw-`/`technique-`/`kernel-`/`pattern-`/`lang-`/`migration-` id, the required
   fields for the page type (see `references/schema.md`), in-scope
   `architectures`, controlled-vocabulary tags, and at least one cited source in
   `sources:`.
2. `uv run python scripts/validate.py`
3. `uv run python scripts/generate-indices.py`
4. Commit the page together with the regenerated `queries/*.md`.

### Validate and regenerate

```bash
uv run python scripts/validate.py          # must report 0 errors
uv run python scripts/generate-indices.py  # deterministic; rerun → no diff
```

The corpus is append-only and not pinned to a single point in time; see
`references/incremental-updates.md`.

## Quality gates

- `uv run python scripts/validate.py` reports 0 errors.
- `uv run python scripts/generate-indices.py` is deterministic (a second run
  produces no diff).
- Every synthesized wiki page cites at least one resolvable source.
- Only in-scope architectures (`sm75`/`sm86`/`sm89`) and controlled-vocabulary
  tags are accepted.
- `uv run python -m unittest discover -s tests` passes.

## License

MIT — see `LICENSE`. Design lineage and attribution: `PROVENANCE.md`.
