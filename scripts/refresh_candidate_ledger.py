#!/usr/bin/env python3
"""Discover candidate PRs for the pre-Hopper wiki and merge them into per-repo
candidate ledgers (candidates/<repo_slug>.yaml).

Two modes:
  * fixture mode (DEFAULT): replays committed GitHub search responses from
    tests/fixtures/gh/<repo_slug>.json. NO network access. This is what the test
    suite and CI exercise, so discovery is deterministic and reproducible.
  * --live: calls `gh search prs` for real. Opt-in only; never used by tests/CI.

Newly-seen PRs are merged into the ledger as `decision: defer` (needs triage).
Existing decisions are NEVER rewritten — re-running is idempotent and additive.
Output (ledgers + data/refresh-search-results.yaml) is byte-stable for identical
inputs: rows sorted by PR number descending, pr_numbers_seen sorted ascending,
atomic temp-file-then-replace writes so a partial/failed run cannot corrupt a
ledger.

Discovery window defaults to 2020-01-01 -> the cutoff date (configurable via --since/--until). The window is
recorded in each ledger header and in refresh-search-results.yaml.

Usage:
    refresh_candidate_ledger.py [--root DIR] [--repos slug,slug] \
        [--since 2020-01-01] [--until 2026-06-30] [--cutoff 2026-06-30] [--live]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402

DEFAULT_SINCE = "2020-01-01"

# Tracked repos: slug -> full GitHub "owner/name" for the tracked repo set; cuVS included (vector-search is in scope).
TRACKED_REPOS = {
    "cutlass": "NVIDIA/cutlass",
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
    "flashinfer": "flashinfer-ai/flashinfer",
    "pytorch": "pytorch/pytorch",
    "tensorrt-llm": "NVIDIA/TensorRT-LLM",
    "cuvs": "NVIDIA/cuVS",
}

# Pre-Hopper-targeted discovery keywords (seeded from data/aliases.yaml /
# data/tags.yaml). These surface CANDIDATES; the classifier decides relevance.
DEFAULT_KEYWORDS = [
    "sm75", "sm_75", "T4", "Turing",
    "sm86", "sm_86", "A10", "Ampere",
    "sm89", "sm_89", "L40", "L20", "Ada",
    "mma.sync", "ldmatrix", "cp.async", "tensor core",
    "fp16", "int8", "tf32", "bf16", "fp8", "dp4a",
]


def atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically (temp file in the same dir, then replace)
    so a crash mid-write cannot truncate or corrupt an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def dump_yaml(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def filter_window(candidates: list[dict], since: str, until: str, slug: str) -> list[dict]:
    """Keep only candidates whose date is within [since, until] inclusive.

    Dates are ISO `YYYY-MM-DD` strings, so lexical comparison is chronological.
    A missing/unparseable date is a hard error: a malformed fixture must not
    silently leak out-of-window rows into the deterministic refresh result."""
    out = []
    for cand in candidates:
        d = str(cand.get("date") or "")
        if not _ISO_DATE.match(d):
            raise SystemExit(
                f"ERROR: candidate {slug}#{cand.get('number')} has missing/invalid "
                f"date {d!r}; expected YYYY-MM-DD."
            )
        if since <= d <= until:
            out.append(cand)
    return out


def fixture_candidates(root: Path, slug: str) -> list[dict]:
    """Replay a committed GitHub search response. Each item is a dict with at
    least number/title/date. No network."""
    fixture = root / "tests" / "fixtures" / "gh" / f"{slug}.json"
    if not fixture.is_file():
        return []
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    out = []
    for item in raw:
        out.append({
            "number": item["number"],
            "title": item.get("title", ""),
            "date": item.get("date") or item.get("createdAt", "")[:10],
        })
    return out


def live_candidates(full_repo: str, keywords: list[str], since: str, until: str) -> list[dict]:
    """Opt-in live discovery via `gh search prs`. Imported lazily and only when
    --live is passed, so the default code path has no network dependency."""
    import subprocess

    seen: dict[int, dict] = {}
    # GitHub search returns at most 1000 results per query (gh paginates internally
    # up to that cap). Ask for the full 1000 rather than the default 100, and WARN
    # when a keyword saturates the cap so missing candidates are surfaced for
    # triage instead of being silently dropped (Codex R10).
    search_cap = 1000
    for kw in keywords:
        cmd = [
            "gh", "search", "prs", "--repo", full_repo, "--merged",
            "--created", f"{since}..{until}", kw,
            "--limit", str(search_cap), "--json", "number,title,createdAt",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: live search failed for {full_repo} [{kw}]: {e}", file=sys.stderr)
            continue
        if res.returncode != 0:
            print(f"  WARN: gh non-zero for {full_repo} [{kw}]: {res.stderr.strip()}", file=sys.stderr)
            continue
        items = json.loads(res.stdout or "[]")
        if len(items) >= search_cap:
            print(f"  WARN: {full_repo} [{kw}] hit the {search_cap}-result search cap; "
                  f"some merged PRs in {since}..{until} may be missing — narrow the "
                  f"window/keyword and re-run to triage them.", file=sys.stderr)
        for item in items:
            n = item["number"]
            seen.setdefault(n, {
                "number": n,
                "title": item.get("title", ""),
                "date": (item.get("createdAt") or "")[:10],
            })
    return list(seen.values())


def merge_into_ledger(ledger: dict, candidates: list[dict], repo_full: str,
                      slug: str, searched_at: str, since: str, until: str,
                      keywords: list[str]) -> dict:
    """Additively merge candidates into a ledger. New PRs -> decision: defer.
    Existing rows are preserved verbatim (decisions never rewritten), EXCEPT that
    any row carrying a valid date outside the advertised [since, until] window is
    dropped: the ledger header is rewritten to this window, so a re-refresh with a
    tighter window must not leave rows the header no longer covers (Codex R7). A
    row with a missing/unparseable date is kept — hand-authored rows may predate
    date-stamping, and silently discarding a curated decision would be worse than
    an out-of-window row we cannot place."""
    existing = {r["number"]: r for r in ledger.get("prs", []) if isinstance(r, dict)}
    for cand in candidates:
        n = cand["number"]
        if n in existing:
            continue  # never rewrite an existing decision
        existing[n] = {
            "number": n,
            "title": cand["title"],
            "date": cand["date"],
            "decision": "defer",
            "reason": "surfaced by discovery refresh; needs triage",
        }

    def _in_window(row: dict) -> bool:
        d = str(row.get("date") or "")
        if not _ISO_DATE.match(d):
            return True  # undated hand-authored row: keep (can't place it)
        return since <= d <= until

    rows = sorted((r for r in existing.values() if _in_window(r)),
                  key=lambda r: r["number"], reverse=True)
    tally = {"include": 0, "exclude": 0, "defer": 0, "needs-review": 0}
    for r in rows:
        if r.get("decision") in tally:
            tally[r["decision"]] += 1
    # Preserve a curated keyword set and reviewer notes if the ledger already
    # has them — a refresh must not clobber hand-authored metadata.
    out = {
        "repo": repo_full,
        "searched_at": searched_at,
        "window_start": since,
        "keywords_used": ledger.get("keywords_used") or keywords,
        "total_candidates": len(rows),
        "included": tally["include"],
        "excluded": tally["exclude"],
        "deferred": tally["defer"],
        "needs_review": tally["needs-review"],
    }
    if ledger.get("notes"):
        out["notes"] = ledger["notes"]
    out["prs"] = rows
    return out


def main():
    parser = argparse.ArgumentParser(description="Refresh pre-Hopper candidate ledgers")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    parser.add_argument("--repos", help="Comma-separated repo slugs (default: all tracked)")
    parser.add_argument("--since", default=DEFAULT_SINCE, help=f"Window start (default {DEFAULT_SINCE})")
    parser.add_argument("--until", help="Window end (default: --cutoff)")
    parser.add_argument("--cutoff", help="Refresh cutoff date YYYY-MM-DD (default: today via fixture stamp)")
    parser.add_argument("--searched-at", help="Override the searched_at stamp (for deterministic tests)")
    parser.add_argument("--live", action="store_true",
                        help="Opt-in: call gh for real (NOT used by tests/CI; default replays fixtures)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT
    slugs = [s.strip() for s in args.repos.split(",")] if args.repos else list(TRACKED_REPOS)
    # Deterministic stamp for fixture mode: caller supplies it; never read the clock.
    searched_at = args.searched_at or args.cutoff
    if not searched_at:
        if args.live:
            print("ERROR: --live requires --cutoff/--searched-at (no implicit clock read).", file=sys.stderr)
            sys.exit(2)
        # fixture mode falls back to the repo's existing cutoff baseline
        rc = root / "data" / "refresh-cutoff.yaml"
        searched_at = (yaml.safe_load(rc.read_text(encoding="utf-8")) or {}).get("cutoff_date") if rc.is_file() else None
        if not searched_at:
            print("ERROR: could not determine searched_at; pass --searched-at.", file=sys.stderr)
            sys.exit(2)
    until = args.until or searched_at

    repos_results = []
    for slug in slugs:
        if slug not in TRACKED_REPOS:
            print(f"  WARN: '{slug}' is not a tracked repo; skipping.", file=sys.stderr)
            continue
        full = TRACKED_REPOS[slug]
        if args.live:
            cands = live_candidates(full, DEFAULT_KEYWORDS, args.since, until)
        else:
            cands = fixture_candidates(root, slug)
        # Apply the discovery window to BOTH modes: only PRs whose date is in
        # [since, until] are merged/recorded, matching the window the ledger and
        # refresh-search-results header advertise.
        cands = filter_window(cands, args.since, until, slug)

        ledger_path = root / "candidates" / f"{slug}.yaml"
        ledger = {}
        if ledger_path.is_file():
            ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
        merged = merge_into_ledger(ledger, cands, full, slug, searched_at, args.since, until, DEFAULT_KEYWORDS)
        atomic_write(ledger_path, dump_yaml(merged))
        print(f"  {slug}: {len(cands)} candidate(s) seen, ledger now {merged['total_candidates']} row(s)")

        repos_results.append({
            "repo_slug": slug,
            "searched_at": searched_at,
            "window_start": args.since,
            "pr_numbers_seen": sorted(c["number"] for c in cands),
            "last_pr_date_seen": max((c["date"] for c in cands), default=""),
        })

    # Preserve entries for repos NOT touched by this (possibly subset) refresh: a
    # targeted `--repos cutlass,flashinfer` run must not erase the other tracked
    # repos from the artifact, which is expected to cover the full tracked set
    # (Codex R6). Load existing entries, replace only the refreshed slugs, re-sort.
    rsr_path = root / "data" / "refresh-search-results.yaml"
    existing = {}
    if rsr_path.is_file():
        prior = yaml.safe_load(rsr_path.read_text(encoding="utf-8")) or {}
        for entry in prior.get("repos", []) or []:
            if isinstance(entry, dict) and entry.get("repo_slug"):
                existing[entry["repo_slug"]] = entry
    for entry in repos_results:
        existing[entry["repo_slug"]] = entry
    merged_repos = sorted(existing.values(), key=lambda r: r["repo_slug"])
    atomic_write(
        rsr_path,
        "## Generated by scripts/refresh_candidate_ledger.py. Byte-stable for\n"
        "## identical inputs; pr_numbers_seen sorted ascending; repos by slug.\n"
        + dump_yaml({"cutoff_date": searched_at, "repos": merged_repos}),
    )
    print(f"Wrote refresh-search-results.yaml for {len(merged_repos)} repo(s) "
          f"({len(repos_results)} refreshed this run).")

    # Advance the incremental-update baseline so a later default refresh does not
    # silently fall back to an older cutoff (Codex R7). MONOTONIC: only move the
    # date forward, never backward; an unchanged/older searched_at leaves the file
    # byte-identical (no spurious diff). The leading comment block and `notes` are
    # preserved.
    _advance_cutoff(root, searched_at)


def _advance_cutoff(root: Path, searched_at: str) -> None:
    rc_path = root / "data" / "refresh-cutoff.yaml"
    if not rc_path.is_file():
        return
    raw = rc_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    current = str(data.get("cutoff_date") or "")
    if not (_ISO_DATE.match(searched_at) and searched_at > current):
        return  # not strictly newer -> leave the file untouched (byte-identical)
    # Preserve the leading comment block (contiguous top-of-file lines starting
    # with '#') so the human-facing header/semantics survive the rewrite.
    header_lines = []
    for line in raw.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip() == "" and header_lines:
            header_lines.append(line)
        else:
            break
    header = ("\n".join(header_lines) + "\n") if header_lines else ""
    body = {"cutoff_date": searched_at}
    if data.get("notes"):
        body["notes"] = data["notes"]
    atomic_write(rc_path, header + dump_yaml(body))
    print(f"Advanced refresh-cutoff.yaml: {current or '(none)'} -> {searched_at}")


if __name__ == "__main__":
    main()
