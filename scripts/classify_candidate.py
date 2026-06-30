#!/usr/bin/env python3
"""Classify a candidate PR against the pre-Hopper inclusion policy.

Given a candidate fixture (PR metadata + a bounded diff/body excerpt) and
data/inclusion-policy.yaml, decide whether the PR earns a `source-pr` page:

  * include  -> {architectures (subset of {sm75,sm86,sm89}), architecture_evidence}
  * skip     -> {reason} drawn from the policy's skip_reasons taxonomy

The decision is PURELY a function of the committed fixture + the policy: no
network, no clock. Running it twice on the same input yields the identical
verdict.

Classification order (first match wins):
  1. non-kernel / docs-only / framework-only by changed-path shape
  2. positive in-scope evidence tiers (direct-sm-mention, direct-device-mention,
     arch-guard-codepath) — but a token that appears ONLY inside a capability
     guard ("not supported on sm75", "fall back") does NOT count
  3. out-of-scope-only architecture mentions -> hopper-only / blackwell-only /
     sm80-only
  4. otherwise -> no-prehopper-evidence

This module is importable (classify(candidate, policy) -> verdict dict) and has a
small CLI for inspecting a single fixture.

Usage:
    classify_candidate.py --fixture tests/fixtures/seed/<slug>/PR-<N>.json [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402


def load_policy(root: Path) -> dict:
    path = root / "data" / "inclusion-policy.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _searchable_text(candidate: dict) -> str:
    """Concatenate the fixture's text fields into one lowercased haystack."""
    parts = [
        str(candidate.get("title", "")),
        str(candidate.get("body", "")),
        str(candidate.get("diff", "")),
    ]
    return "\n".join(parts).lower()


def _changed_paths(candidate: dict) -> list[str]:
    return [str(p) for p in (candidate.get("changed_paths") or candidate.get("files_changed") or [])]


def _matches_glob(path: str, globs: list[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(path, g) for g in globs)


def _token_regex(token_lower: str) -> re.Pattern:
    """Compile a word-boundary matcher for an architecture/device token.

    Plain `in` substring matching is wrong: 'A10' would match inside 'A100'
    (sm80, out of scope) and 'sm89' could match inside a longer run. We require
    the token not to be flanked by an alphanumeric character, so 'A10' matches
    'A10'/'A10G' boundaries but not 'A100'. Dots in tokens (cp.async) are
    escaped and treated literally.
    """
    return re.compile(r"(?<![0-9a-z])" + re.escape(token_lower) + r"(?![0-9a-z])")


def _token_spans(haystack: str, token_lower: str) -> list[int]:
    """Start offsets of every word-boundary occurrence of token in haystack."""
    return [m.start() for m in _token_regex(token_lower).finditer(haystack)]


def _occurrence_guarded(haystack: str, pos: int, token_len: int, markers: list[str]) -> bool:
    """True if THIS occurrence sits inside a capability-guard context. The window
    is the clause containing the token (bounded by . ; newline) PLUS a short
    look-ahead into the immediately-following clause, because guard wording often
    trails the arch token across a boundary, e.g. 'Turing (sm75). Not supported;
    fall back ...'. The look-behind stays clause-local so a guard about a
    DIFFERENT arch earlier in the text does not taint this mention."""
    start = max((haystack.rfind(ch, 0, pos) for ch in ".;\n"), default=-1) + 1
    end_candidates = [haystack.find(ch, pos + token_len) for ch in ".;\n"]
    end_candidates = [e for e in end_candidates if e != -1]
    clause_end = min(end_candidates) if end_candidates else len(haystack)
    # Look-ahead: include the next clause too (bounded), to catch trailing guards.
    lookahead = haystack[clause_end: clause_end + 80]
    window = haystack[start:clause_end] + " " + lookahead
    return any(m in window for m in markers)


def _has_clean_mention(haystack: str, token_lower: str, markers: list[str]) -> bool:
    """True when token occurs at a word boundary at least once OUTSIDE any
    capability-guard clause."""
    spans = _token_spans(haystack, token_lower)
    if not spans:
        return False
    if not markers:
        return True
    return any(not _occurrence_guarded(haystack, pos, len(token_lower), markers) for pos in spans)


def _only_guarded_mention(haystack: str, token_lower: str, markers: list[str]) -> bool:
    """True when token occurs (at a boundary) but every occurrence is guarded."""
    spans = _token_spans(haystack, token_lower)
    if not spans:
        return False
    return all(_occurrence_guarded(haystack, pos, len(token_lower), markers) for pos in spans)


def classify(candidate: dict, policy: dict, in_scope_archs: set[str] | None = None) -> dict:
    """Return a verdict dict:
        {"decision": "include", "architectures": [...], "architecture_evidence": [...]}
      or {"decision": "skip", "reason": "<taxonomy>"}
    """
    in_scope_archs = in_scope_archs or {"sm75", "sm86", "sm89"}
    haystack = _searchable_text(candidate)
    paths = _changed_paths(candidate)
    skip_reasons = policy.get("skip_reasons", {})
    markers = [m.lower() for m in (policy.get("capability_guard_markers") or [])]
    globs = policy.get("non_kernel_path_globs", {}) or {}
    kernel_exts = tuple(globs.get("kernel_path_extensions", []))
    ambiguous_exts = tuple(globs.get("ambiguous_path_extensions", []))

    def skip(reason: str) -> dict:
        # Defensive: only emit reasons the taxonomy knows.
        return {"decision": "skip", "reason": reason if reason in skip_reasons else "no-prehopper-evidence"}

    # --- (1) kernel-evidence gate -------------------------------------------
    # A source-pr page is a KERNEL page. Require positive evidence that the PR
    # touches device/kernel code. A path with an UNAMBIGUOUS device extension
    # (.cu/.cuh/.ptx) is sufficient. An AMBIGUOUS C++ path (.cpp/.h/...) counts
    # only when the text also carries a kernel signal, so a host-only
    # src/scheduler.cpp is not mistaken for kernel work. Absence of paths is not
    # a pass — the text must then carry a kernel signal (closes the empty-paths
    # bypass, Codex).
    kernel_text_signals = ("__cuda_arch__", "mma.sync", "ldmatrix", "cp.async",
                           "wmma", ".cu", ".cuh", ".ptx", "cutlass", "__global__",
                           "tensor core", "tensor-core", "warp-level", "kernel")
    has_kernel_text = any(s in haystack for s in kernel_text_signals)
    has_device_path = bool(kernel_exts) and any(p.endswith(kernel_exts) for p in paths)
    has_ambiguous_path = bool(ambiguous_exts) and any(p.endswith(ambiguous_exts) for p in paths)
    touches_kernel = has_device_path or (has_ambiguous_path and has_kernel_text)

    if paths and not touches_kernel:
        docs_globs = globs.get("docs-only", [])
        if docs_globs and all(_matches_glob(p, docs_globs) for p in paths):
            return skip("docs-only")
        fw_globs = globs.get("framework-only", [])
        if fw_globs and all(_matches_glob(p, fw_globs) for p in paths):
            return skip("framework-only")
        return skip("non-kernel")
    if not touches_kernel and not has_kernel_text:
        # No kernel path AND no kernel text signal (covers empty changed_paths).
        return skip("non-kernel")

    # --- (2) collect clean in-scope evidence AND clean out-of-scope mentions --
    architectures: set[str] = set()
    evidence: list[dict] = []
    for tier in policy.get("evidence_tiers", []):
        for token, arch in (tier.get("maps_to", {}) or {}).items():
            if arch not in in_scope_archs:
                continue
            if _has_clean_mention(haystack, token.lower(), markers):
                architectures.add(arch)
                evidence.append({
                    "architecture": arch,
                    "evidence_type": tier.get("id"),
                    "token": token,
                })

    oos = policy.get("out_of_scope_arch_tokens", {}) or {}
    oos_clean = []  # (token, reason) with a clean (non-guarded) mention
    oos_any = []    # (token, reason) present at all (clean or guarded)
    for token, reason in oos.items():
        if _token_spans(haystack, token.lower()):
            oos_any.append((token, reason))
            if _has_clean_mention(haystack, token.lower(), markers):
                oos_clean.append((token, reason))

    # --- (3) decision -------------------------------------------------------
    # A clean, in-scope direct/guard mention is sufficient for include. An
    # out-of-scope arch may legitimately appear as CONTRAST ("A10 has 100KB vs
    # sm80's 164KB", "no Hopper TMA on Ada"), so its mere presence does NOT veto
    # a clean in-scope optimization target. Genuinely ambiguous support-matrix
    # PRs (no direct target) have no clean in-scope evidence and fall through to
    # the skip paths + human ledger review. The sm89-in-fallback bypass is
    # handled upstream: guard/fallback detection makes that mention non-clean.
    if architectures:
        archs_sorted = sorted(architectures)
        evidence_sorted = sorted(evidence, key=lambda e: (e["architecture"], str(e["evidence_type"]), e["token"]))
        return {
            "decision": "include",
            "architectures": archs_sorted,
            "architecture_evidence": evidence_sorted,
        }

    # No clean in-scope evidence. Attribute the skip.
    if oos_any:
        # Prefer a clean out-of-scope mention's reason; else any present one.
        return skip((oos_clean or oos_any)[0][1])

    # An in-scope token present but ONLY as a capability guard / fallback.
    for tier in policy.get("evidence_tiers", []):
        for token, arch in (tier.get("maps_to", {}) or {}).items():
            if arch in in_scope_archs and _only_guarded_mention(haystack, token.lower(), markers):
                return skip("capability-guard-only")

    # --- (4) nothing defensible ---------------------------------------------
    return skip("no-prehopper-evidence")


def main():
    parser = argparse.ArgumentParser(description="Classify a candidate PR fixture")
    parser.add_argument("--fixture", required=True, help="Path to a candidate JSON fixture")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve() if args.root else _DEFAULT_ROOT
    candidate = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    verdict = classify(candidate, load_policy(root))
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
