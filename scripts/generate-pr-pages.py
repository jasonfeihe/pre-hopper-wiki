#!/usr/bin/env python3
"""Generate `source-pr` pages from a seed manifest of candidate fixtures.

Reads a manifest (default tests/fixtures/seed/seed-manifest.yaml) whose entries
each point at a committed candidate fixture (PR metadata + bounded diff/body).
For each entry:

  * run the inclusion classifier (scripts/classify_candidate.classify);
  * on an `include` verdict, write a schema-valid
    sources/prs/<repo_slug>/PR-<N>.md page (architectures from the classifier,
    tags from the curated manifest entry, inclusion_reason citing the evidence);
  * on a `skip` verdict, append a row to data/pr-page-skipped.yaml (never a page).

PURELY offline: classification and page text derive only from committed
fixtures + the manifest + data/inclusion-policy.yaml. No network, no clock —
`captured_at` comes from the manifest (or --captured-at), never from the system
time, so output is byte-stable.

Page ids are collision-safe across repos: `pr-<repo_slug>-<N>`.

Usage:
    generate-pr-pages.py [--root DIR] [--manifest PATH] [--captured-at YYYY-MM-DD]
                         [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402
from classify_candidate import classify, load_policy  # noqa: E402

DEFAULT_MANIFEST = "tests/fixtures/seed/seed-manifest.yaml"

# Provenance marker written into every generated page's frontmatter. Only pages
# carrying this marker are considered generator-owned and thus eligible for stale
# removal; hand-authored source-pr pages (added via the append-only workflow in
# references/incremental-updates.md) have no marker and are never deleted.
GENERATED_BY = "generate-pr-pages"


def _vocab(root: Path, key: str) -> set:
    data = yaml.safe_load((root / "data" / "tags.yaml").read_text(encoding="utf-8")) or {}
    return set(data.get(key, []) or [])


def _allowed_statuses(root: Path) -> list[str]:
    """The source-pr `status` enum, read from the schema source of truth so the
    pre-emit check can never drift from validate.py. Falls back to the canonical
    set if the schema is unreadable, so the check never silently no-ops."""
    fallback = ["open", "merged", "closed"]
    try:
        schemas = yaml.safe_load((root / "data" / "schemas.yaml").read_text(encoding="utf-8")) or {}
        allowed = ((schemas.get("source-pr", {}) or {}).get("constraints", {}) or {}).get("status")
        return list(allowed) if allowed else fallback
    except (OSError, yaml.YAMLError):
        return fallback


def _page_id(root: Path, page: Path) -> str:
    """The collision-safe page id for a generated PR page path:
    sources/prs/<slug>/PR-<N>.md -> pr-<slug>-<N>."""
    slug, fname = page.relative_to(root).parts[-2:]
    return f"pr-{slug}-{fname[len('PR-'):-len('.md')]}"


def _is_generated_page(page: Path) -> bool:
    """True iff the page's frontmatter carries our provenance marker. Only such
    pages are generator-owned; hand-authored source-pr pages (append-only
    workflow) have no marker and must never be treated as stale."""
    try:
        text = page.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    front = text[3:end] if end != -1 else text
    try:
        fm = yaml.safe_load(front) or {}
    except yaml.YAMLError:
        return False
    return isinstance(fm, dict) and fm.get("generated_by") == GENERATED_BY


def _generated_pr_pages(root: Path) -> list[Path]:
    """Every generator-OWNED PR page currently on disk (i.e. carrying the
    provenance marker). Pages without the marker are hand-authored and excluded,
    so a regeneration never deletes append-only content."""
    base = root / "sources" / "prs"
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("PR-*.md") if _is_generated_page(p))


def _evidence_sentence(evidence: list[dict]) -> str:
    bits = [f"{e['architecture']} via {e['evidence_type']} ('{e['token']}')" for e in evidence]
    return "Pre-Hopper relevance: " + "; ".join(bits) + "."


def render_page(entry: dict, verdict: dict, captured_at: str) -> str:
    """Build a schema-valid source-pr page. Only fields the source-pr schema
    allows are emitted (no `architecture_evidence` field — its content goes into
    `inclusion_reason`)."""
    slug = entry["repo_slug"]
    num = entry["pr"]
    fm = {
        "id": f"pr-{slug}-{num}",
        "repo": entry["repo"],
        "pr": num,
        "title": entry["title"],
        "author": entry["author"],
        "date": entry["date"],
        "url": entry["url"],
        "source_category": "upstream-code",
        "architectures": verdict["architectures"],
        "tags": entry.get("tags", []),
        "captured_at": captured_at,
        "status": entry["status"],
    }
    if entry.get("status") == "merged" and entry.get("merge_sha"):
        fm["merge_sha"] = entry["merge_sha"]
    for opt in ("techniques", "hardware_features", "kernel_types", "languages"):
        if entry.get(opt):
            fm[opt] = entry[opt]
    fm["inclusion_reason"] = _evidence_sentence(verdict["architecture_evidence"])
    if entry.get("description"):
        fm["description"] = entry["description"]
    # Provenance: marks this page as generator-owned so stale cleanup can target
    # only pages this script produced (never hand-authored ones).
    fm["generated_by"] = GENERATED_BY

    front = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    body = entry.get("summary", f"Summary of {entry['repo']} PR #{num}: {entry['title']}.")
    return f"---\n{front}---\n\n# {entry['title']}\n\n{body}\n"


def validate_page_fields(entry: dict, verdict: dict, vocabs: dict, arch_vocab: set,
                         allowed_statuses: list[str]) -> list[str]:
    """Hard pre-emit checks so the generator never writes an invalid page. Every
    controlled-vocabulary list field on the entry must use only in-vocabulary
    values (mirrors validate.py so generation can't outrun the validator)."""
    errs = []
    label = f"{entry['repo_slug']} PR {entry['pr']}"
    if not verdict["architectures"]:
        errs.append(f"{label}: include verdict has empty architectures")
    for a in verdict["architectures"]:
        if a not in arch_vocab:
            errs.append(f"{label}: architecture '{a}' out of scope")
    # tags is validated against the union of every vocab set; the typed list
    # fields are validated against their own set.
    union = set().union(*vocabs.values()) if vocabs else set()
    for t in entry.get("tags", []):
        if t not in union:
            errs.append(f"{label}: tag '{t}' not in data/tags.yaml vocabulary")
    for field in ("techniques", "hardware_features", "kernel_types", "languages"):
        for v in entry.get(field, []):
            if v not in vocabs.get(field, set()):
                errs.append(f"{label}: {field} value '{v}' not in data/tags.yaml")
    # status must be in the source-pr schema enum BEFORE we emit a page, else the
    # page would only fail validate.py later (the generator's pre-emit contract).
    status = entry.get("status")
    if status not in allowed_statuses:
        errs.append(f"{label}: status '{status}' not in {allowed_statuses}")
    if status == "merged" and not entry.get("merge_sha"):
        errs.append(f"{label}: status merged but no merge_sha")
    return errs


def main():
    parser = argparse.ArgumentParser(description="Generate source-pr pages from a seed manifest")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    parser.add_argument("--manifest", help=f"Manifest path (default {DEFAULT_MANIFEST})")
    parser.add_argument("--captured-at", help="Override captured_at stamp (else from manifest)")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT
    manifest_path = Path(args.manifest) if args.manifest else root / DEFAULT_MANIFEST
    # A custom --manifest is a PARTIAL run: it is authoritative only for the ids it
    # declares, so reconciliation must not delete pages/skip-rows outside it (Codex
    # R11). The default committed manifest is a FULL rebuild of the whole
    # generator-owned corpus (keeps the R8 deleted-entry cleanup).
    partial = bool(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    captured_at = args.captured_at or manifest.get("captured_at")
    if not captured_at:
        print("ERROR: captured_at must come from the manifest or --captured-at (no clock read).", file=sys.stderr)
        sys.exit(2)

    policy = load_policy(root)
    vocabs = {
        "architectures": _vocab(root, "architectures"),
        "hardware_features": _vocab(root, "hardware_features"),
        "techniques": _vocab(root, "techniques"),
        "kernel_types": _vocab(root, "kernel_types"),
        "languages": _vocab(root, "languages"),
    }
    arch_vocab = vocabs["architectures"]
    allowed_statuses = _allowed_statuses(root)

    emitted, skipped = [], []
    hard_errors = []
    for entry in manifest.get("entries", []):
        fixture_path = root / entry["fixture"]
        candidate = json.loads(fixture_path.read_text(encoding="utf-8"))
        verdict = classify(candidate, policy, arch_vocab)
        if verdict["decision"] == "skip":
            skipped.append({
                "pr_id": f"pr-{entry['repo_slug']}-{entry['pr']}",
                "repo": entry["repo"],
                "pr_number": entry["pr"],
                "stage": "classify",
                "reason": verdict["reason"],
                "recorded_at": captured_at,
            })
            continue
        errs = validate_page_fields(entry, verdict, vocabs, arch_vocab, allowed_statuses)
        if errs:
            hard_errors.extend(errs)
            continue
        page = render_page(entry, verdict, captured_at)
        dest = root / "sources" / "prs" / entry["repo_slug"] / f"PR-{entry['pr']}.md"
        emitted.append((dest, page))

    if hard_errors:
        print("ERROR: cannot emit invalid pages:", file=sys.stderr)
        for e in hard_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Reconcile the generator-owned artifacts against the CURRENT manifest. The
    # emitted set is authoritative for pages; a GENERATOR-OWNED page (carrying the
    # provenance marker) whose id is not emitted this run is stale — whether it
    # flipped include->skip OR its manifest entry was deleted entirely (Codex R8).
    # Hand-authored pages without the marker are never swept (Codex R10). For a
    # PARTIAL run (custom --manifest) the manifest is authoritative ONLY for the
    # ids it declares, so a stale page must ALSO be one this manifest declares —
    # otherwise regenerating one repo would delete unrelated pages (Codex R11).
    emitted_ids = {_page_id(root, dest) for dest, _ in emitted}
    manifest_ids = {f"pr-{e['repo_slug']}-{e['pr']}" for e in manifest.get("entries", [])}

    def _is_stale(page: Path) -> bool:
        pid = _page_id(root, page)
        if pid in emitted_ids:
            return False
        # Partial run: only reconcile pages the active manifest is responsible for.
        return pid in manifest_ids if partial else True

    stale_pages = [p for p in _generated_pr_pages(root) if _is_stale(p)]

    if args.dry_run:
        for dest, _ in emitted:
            print(f"  would write {dest.relative_to(root)}")
        for stale in stale_pages:
            print(f"  would REMOVE stale page {stale.relative_to(root)}")
        for row in skipped:
            print(f"  would skip-log {row['pr_id']} ({row['reason']})")
        print(f"Dry run: {len(emitted)} page(s), {len(stale_pages)} stale removal(s), "
              f"{len(skipped)} skip(s).")
        return

    for dest, page in emitted:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8")
        print(f"  wrote {dest.relative_to(root)}")

    # Delete every stale page (include->skip flips AND deleted manifest entries),
    # pruning a now-empty repo dir. A PR cannot be both an included page and
    # absent/skipped in the current manifest.
    for stale in stale_pages:
        stale.unlink()
        print(f"  removed stale page {stale.relative_to(root)}")
        try:
            stale.parent.rmdir()  # only succeeds if now empty
        except OSError:
            pass

    # Merge skip rows into data/pr-page-skipped.yaml (sorted, deterministic).
    # Drop an existing row when a PR emitted this run reclaims its id (a PR cannot
    # be both a page and a skip). For a FULL rebuild, also drop rows whose id is no
    # longer in the manifest (Codex R8). For a PARTIAL run, rows for ids OUTSIDE
    # this manifest are preserved untouched — this manifest is not authoritative
    # for them (Codex R11).
    def _keep_existing(pid) -> bool:
        if pid in emitted_ids:
            return False  # reclaimed by a page this run
        if partial:
            # Only ids this manifest declares are reconciled; keep the rest as-is.
            # A declared id that is still a skip is re-added fresh by the loop below.
            return pid not in manifest_ids
        return pid in manifest_ids  # full rebuild: drop ids no longer in the manifest

    skip_path = root / "data" / "pr-page-skipped.yaml"
    existing_rows = []
    if skip_path.is_file():
        prior = yaml.safe_load(skip_path.read_text(encoding="utf-8")) or {}
        existing_rows = prior.get("rows", []) or []
    by_id = {r.get("pr_id"): r for r in existing_rows
             if isinstance(r, dict) and _keep_existing(r.get("pr_id"))}
    for row in skipped:
        by_id[row["pr_id"]] = row
    rows = sorted(by_id.values(), key=lambda r: (r.get("repo", ""), r.get("pr_number", 0)))
    out = ("## Skip audit emitted by scripts/generate-pr-pages.py. Every `reason`\n"
           "## must be a key in data/inclusion-policy.yaml::skip_reasons.\n"
           + yaml.safe_dump({"rows": rows}, sort_keys=False, allow_unicode=True))
    skip_path.write_text(out, encoding="utf-8")
    print(f"Wrote {len(emitted)} page(s); skip audit has {len(rows)} row(s).")


if __name__ == "__main__":
    main()
