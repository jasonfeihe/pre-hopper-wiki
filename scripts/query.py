#!/usr/bin/env python3
"""Unified query tool for the pre-Hopper kernel wiki.

Keyword search plus tag / type / repo / language / architecture / symptom /
confidence filters. Filters are alias-aware: `--architecture T4` matches `sm75`,
`--tag mma.sync` matches `mma-sync`, etc., via data/aliases.yaml.

Usage:
    query.py "shared memory bank conflicts"
    query.py --tag mma-sync --type hardware
    query.py --architecture T4
    query.py --type technique --compact

Returns a ranked list of matching pages with titles, paths, and key fields.
Returns a clean "No matching pages." (not a crash) when nothing matches or the
knowledge base is empty.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402

_ALIAS_CACHE = {}


def load_alias_expansions(root: Path):
    key = str(root)
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    out = {}
    aliases_path = root / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE[key] = {}
        return {}
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE[key] = out
    return out


def expand_keyword(kw: str, root: Path):
    aliases = load_alias_expansions(root)
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_frontmatter(path: Path):
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, None
    if not isinstance(fm, dict):
        return None, None
    return fm, m.group(2)


def load_all_pages(root: Path):
    pages = []
    for subdir in ("sources", "wiki"):
        base = root / subdir
        if not base.exists():
            continue
        for md in sorted(base.rglob("*.md")):
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append({
                "path": md.relative_to(root).as_posix(),
                "fm": fm,
                "body": body or "",
            })
    return pages


def detect_page_type(fm: dict, path: str):
    parts = path.split("/")
    if parts[0] == "wiki" and "type" in fm:
        return f"wiki-{fm['type']}"
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"
    return "unknown"


def score_keyword_match(fm, body, keywords, root):
    score = 0
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                         "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    body_lower = body.lower()
    for kw in keywords:
        best = 0
        for variant in expand_keyword(kw, root):
            v_l = variant.lower()
            s = 0
            if v_l in title_text:
                s += 10
            if v_l in tag_text:
                s += 5
            s += min(body_lower.count(v_l), 3)
            best = max(best, s)
        score += best
    return score


def filter_pages(pages, args, root):
    out = []
    for p in pages:
        fm, path = p["fm"], p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type and not (ptype.endswith(args.type) or ptype == args.type):
            continue

        if args.tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag, root)}
            if not any(str(t).lower() in tag_variants for t in all_tags):
                continue

        if args.repo:
            if args.repo.lower() not in str(fm.get("repo", "")).lower():
                continue

        if args.language:
            langs = set(fm.get("languages") or [])
            tags = set(fm.get("tags") or [])
            if args.language not in langs and args.language not in tags:
                continue

        if args.architecture:
            archs = {str(a).lower() for a in (fm.get("architectures") or [])}
            arch_variants = {v.lower() for v in expand_keyword(args.architecture, root)}
            if not (archs & arch_variants):
                continue

        if args.symptom and args.symptom not in set(fm.get("symptoms") or []):
            continue

        if args.confidence and str(fm.get("confidence", "")) != args.confidence:
            continue

        out.append(p)
    return out


def format_result(page, compact=False):
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")
    if compact:
        return f"  [{ptype}] {pid}: {title}  ({path})"
    lines = [f"## {title}",
             f"- **id**: `{pid}`",
             f"- **type**: `{ptype}`",
             f"- **path**: `{path}`"]
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "hardware_features", "techniques", "kernel_types", "languages"):
        if fm.get(k):
            lines.append(f"- **{k}**: {fm[k]}")
    if fm.get("sources"):
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the pre-Hopper kernel wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument("--type", help="Page type (hardware, technique, kernel, pattern, language, migration, pr, doc, blog, contest)")
    parser.add_argument("--tag", help="Filter by tag (tags/techniques/hardware_features/kernel_types/languages)")
    parser.add_argument("--repo", help="Filter by source repo (partial match)")
    parser.add_argument("--language", help="Filter by language/DSL (cuda-cpp, ptx, triton, cutlass, python)")
    parser.add_argument("--architecture", help="Filter by architecture (sm75/sm86/sm89 or an alias like T4/Ampere/L40)")
    parser.add_argument("--symptom", help="Filter by pattern symptom")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--compact", action="store_true", help="One line per result")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT

    pages = filter_pages(load_all_pages(root), args, root)

    keywords = [tok for q in args.query for tok in re.split(r"\s+", q.strip()) if tok]
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords, root)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[:args.limit]

    if args.paths_only:
        for p in pages:
            print(p["path"])
        return
    if not pages:
        print("No matching pages.")
        return
    print(f"# {len(pages)} result(s)\n")
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
