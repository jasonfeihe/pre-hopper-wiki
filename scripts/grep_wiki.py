#!/usr/bin/env python3
"""Regex text search across the pre-Hopper kernel wiki bodies and source pages.

Usage:
    grep_wiki.py "bank conflict"
    grep_wiki.py "mma\\.sync" --only wiki
    grep_wiki.py "cp.async" "pipeline" --any
    grep_wiki.py "tensor core" --context 3

Returns matching lines with file path, line number, and N context lines.
Prints "No matches." (not a crash) when nothing matches or the KB is empty.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402


def iter_files(scope: str, root: Path):
    dirs = {
        "wiki": ["wiki"],
        "sources": ["sources"],
        "all": ["wiki", "sources"],
    }
    for sub in dirs.get(scope, ["wiki", "sources"]):
        base = root / sub
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.md")):
            if f.is_file():
                yield f


def grep_file(path: Path, patterns, context: int, any_match: bool):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    results = []
    for i, line in enumerate(lines):
        matched = (any(p.search(line) for p in patterns) if any_match
                   else all(p.search(line) for p in patterns))
        if matched:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "\n".join(
                f"{j+1}{'->' if j == i else ':'} {lines[j]}" for j in range(start, end)
            )
            results.append((i + 1, snippet))
    return results


def main():
    parser = argparse.ArgumentParser(description="Text search across the pre-Hopper kernel wiki")
    parser.add_argument("patterns", nargs="+", help="Regex pattern(s) — all must match a line unless --any")
    parser.add_argument("--only", choices=["wiki", "sources", "all"], default="all", help="Search scope (default all)")
    parser.add_argument("--context", type=int, default=1, help="Context lines around each match (default 1)")
    parser.add_argument("--any", action="store_true", help="Match if ANY pattern matches a line")
    parser.add_argument("--limit", type=int, default=20, help="Max files reported (default 20)")
    parser.add_argument("--files-only", action="store_true", help="Print only matching file paths")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT

    compiled = []
    for p in args.patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            print(f"ERROR: invalid regex {p!r}: {e}", file=sys.stderr)
            sys.exit(2)

    matched_files = []
    for path in iter_files(args.only, root):
        hits = grep_file(path, compiled, args.context, args.any)
        if hits:
            matched_files.append((path, hits))
    matched_files = matched_files[:args.limit]

    if args.files_only:
        for path, _ in matched_files:
            print(path.relative_to(root))
        return
    if not matched_files:
        print("No matches.")
        return
    print(f"# {len(matched_files)} file(s) match")
    for path, hits in matched_files:
        print(f"\n## {path.relative_to(root)}  ({len(hits)} match{'es' if len(hits) != 1 else ''})")
        for _, snippet in hits[:5]:
            print("```")
            print(snippet)
            print("```")


if __name__ == "__main__":
    main()
