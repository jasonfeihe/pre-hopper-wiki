#!/usr/bin/env python3
"""Retrieve a single page from the pre-Hopper kernel wiki by id or path.

Usage:
    get_page.py hw-mma-sync-turing               # by id
    get_page.py wiki/hardware/mma-sync-turing.md  # by path
    get_page.py hw-mma-sync-turing --body-only
    get_page.py hw-mma-sync-turing --frontmatter-only
    get_page.py technique-cp-async-pipelining --follow-sources

Prints a clean "not found" message (exit 1) for an unknown id/path rather than
crashing with a traceback.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402


def find_page(lookup: str, root: Path):
    if "/" in lookup or lookup.endswith(".md"):
        p = root / lookup
        return p if p.exists() else None
    for subdir in ("wiki", "sources"):
        base = root / subdir
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            try:
                content = md.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.match(r"^---\s*\r?\n(.*?)\r?\n---", content, re.DOTALL)
            if not m:
                continue
            try:
                fm = yaml.safe_load(m.group(1))
            except yaml.YAMLError:
                continue
            if isinstance(fm, dict) and fm.get("id") == lookup:
                return md
    return None


def split_frontmatter(content: str):
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", content, re.DOTALL)
    if not m:
        return None, content
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        fm = None
    return fm, m.group(2)


def main():
    parser = argparse.ArgumentParser(description="Get a pre-Hopper wiki page by id or path")
    parser.add_argument("lookup", help="Page id (e.g. hw-mma-sync-turing) or relative path")
    parser.add_argument("--body-only", action="store_true", help="Print only the body")
    parser.add_argument("--frontmatter-only", action="store_true", help="Print only the frontmatter as YAML")
    parser.add_argument("--follow-sources", action="store_true", help="Also print a 500-char excerpt of each cited source")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT

    page_path = find_page(args.lookup, root)
    if not page_path:
        print(f"ERROR: No page found for '{args.lookup}'", file=sys.stderr)
        sys.exit(1)

    content = page_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)

    if args.frontmatter_only:
        if fm:
            print(yaml.dump(fm, allow_unicode=True, sort_keys=False))
        return
    if args.body_only:
        print(body)
        return

    print(f"# {page_path.relative_to(root)}\n")
    print(content)

    if args.follow_sources and fm and fm.get("sources"):
        print("\n---\n## Cited Sources (excerpts)\n")
        for src_id in fm.get("sources", []):
            src_page = find_page(src_id, root)
            if src_page:
                _, src_body = split_frontmatter(src_page.read_text(encoding="utf-8"))
                excerpt = (src_body or "")[:500].strip()
                print(f"### {src_id}")
                print(f"`{src_page.relative_to(root)}`\n")
                print(excerpt + "\n")
            else:
                print(f"### {src_id}\n_(source id not found)_\n")


if __name__ == "__main__":
    main()
