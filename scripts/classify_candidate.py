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


def _token_arch_map(policy: dict) -> dict[str, str]:
    """Map every architecture/device token (lowercased) the policy knows to its
    architecture: positive tiers map to an in-scope sm; out-of-scope tokens map
    to a synthetic 'oos:<reason>' arch so they are distinguishable from in-scope
    archs during guard attribution."""
    out: dict[str, str] = {}
    for tier in policy.get("evidence_tiers", []):
        for tok, arch in (tier.get("maps_to", {}) or {}).items():
            out.setdefault(tok.lower(), arch)
    for tok, reason in (policy.get("out_of_scope_arch_tokens", {}) or {}).items():
        out.setdefault(tok.lower(), f"oos:{reason}")
    return out


# --- Capability-guard attribution (segment-based) --------------------------
#
# A capability-guard marker ("not supported", "fall back", "requires sm", ...)
# guards the architecture it grammatically governs. Attribution is scoped to the
# marker's own HARD SEGMENT (split on '.', ';', newline — NOT comma):
#   * within the segment, the marker governs the CONTIGUOUS COORDINATED arch run
#     adjacent to it — the nearest arch token AFTER it and any further tokens
#     joined to it by list connectors only ("not supported on sm75 and sm89"
#     guards BOTH); if no arch follows, the nearest run BEFORE it ("sm75 and sm89
#     not supported"). A non-connector gap breaks the run, so a contrastive
#     "...sm75, but optimized for sm89" still leaves sm89 clean.
#   * a marker in a segment with NO arch token carries back one step to the
#     immediately-previous non-empty segment only if that segment names exactly
#     one architecture ("Turing (sm75). Not supported").
# An occurrence is guarded iff some marker attribution targets its (segment, arch);
# an architecture is CLEAN iff any of its occurrences is unguarded. Comparing by
# architecture (not token) means synonyms like 'turing'/'sm75' are treated as one.
# This design and its case coverage were cross-checked with Codex.

_HARD_BOUNDARIES = ".;\n"

# Text that may join two arch tokens into ONE coordinated list governed by a
# single guard marker: list connectors ("and"/"or") and punctuation only. Any
# other word (e.g. ", but optimized for") breaks the run.
_CONNECTOR_RE = re.compile(r"^[\s,/&+]*(?:(?:and|or)[\s,/&+]*)*$", re.IGNORECASE)


def _coordinated_run(haystack: str, ordered: list) -> set[str]:
    """`ordered` is arch-token occurrences sorted by proximity to a guard marker
    (nearest first, extending away from it). Return the archs of the maximal
    prefix that forms a coordinated list — consecutive tokens separated only by
    list connectors. The first (nearest) token always anchors the run."""
    run = [ordered[0]]
    for prev, nxt in zip(ordered, ordered[1:]):
        lo, hi = (prev[1], nxt[0]) if prev[1] <= nxt[0] else (nxt[1], prev[0])
        if _CONNECTOR_RE.match(haystack[lo:hi]):
            run.append(nxt)
        else:
            break
    return {t[2] for t in run}



def _hard_segments(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans split on . ; newline."""
    segs = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _HARD_BOUNDARIES:
            if start < i:
                segs.append((start, i))
            start = i + 1
    if start < len(text):
        segs.append((start, len(text)))
    return segs


def _seg_id(offset: int, segments: list[tuple[int, int]]) -> int:
    for sid, (s, e) in enumerate(segments):
        if s <= offset < e:
            return sid
    return -1


def _guarded_segment_archs(haystack: str, token_arch_map: dict[str, str],
                           markers: list[str]) -> set[tuple[int, str]]:
    """Return the set of (segment_id, architecture) pairs that a capability guard
    attributes to. Any arch-token occurrence whose (segment, arch) is in this set
    is guarded."""
    segments = _hard_segments(haystack)
    # All arch-token occurrences: (start, end, arch, seg_id).
    token_occs = []
    for tok, arch in token_arch_map.items():
        for off in _token_spans(haystack, tok):
            token_occs.append((off, off + len(tok), arch, _seg_id(off, segments)))
    tokens_by_seg: dict[int, list] = {}
    for occ in token_occs:
        tokens_by_seg.setdefault(occ[3], []).append(occ)

    # All marker occurrences: (start, end, seg_id).
    marker_occs = []
    for m in markers:
        mstart = 0
        while True:
            i = haystack.find(m, mstart)
            if i == -1:
                break
            marker_occs.append((i, i + len(m), _seg_id(i, segments)))
            mstart = i + len(m)

    guarded: set[tuple[int, str]] = set()
    for mstart, mend, sid in marker_occs:
        seg_tokens = tokens_by_seg.get(sid, [])
        governed_archs: set[str] = set()
        target_sid = sid
        if seg_tokens:
            # The coordinated arch run AFTER the marker (nearest first), else the
            # run BEFORE it. A guard naming a list ("sm75 and sm89") governs all.
            after = sorted((t for t in seg_tokens if t[0] >= mstart),
                           key=lambda t: (t[0], t[1]))
            if after:
                governed_archs = _coordinated_run(haystack, after)
            else:
                before = sorted((t for t in seg_tokens if t[1] <= mstart),
                                key=lambda t: (t[1], t[0]), reverse=True)
                if before:
                    governed_archs = _coordinated_run(haystack, before)
        else:
            # Orphan marker segment: carry back one step to a single-arch segment.
            prev = sid - 1
            while prev >= 0 and not tokens_by_seg.get(prev):
                prev -= 1
            if prev >= 0:
                prev_arches = {t[2] for t in tokens_by_seg.get(prev, [])}
                if len(prev_arches) == 1:
                    governed_archs = prev_arches
                    target_sid = prev
        for arch in governed_archs:
            guarded.add((target_sid, arch))
    return guarded


def _arch_occurrences(haystack: str, arch: str, token_arch_map: dict[str, str],
                      segments: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """(offset, seg_id) for every token occurrence that maps to `arch`."""
    out = []
    for tok, a in token_arch_map.items():
        if a != arch:
            continue
        for off in _token_spans(haystack, tok):
            out.append((off, _seg_id(off, segments)))
    return out


def _is_arch_clean(haystack: str, arch: str, markers: list[str],
                   token_arch_map: dict[str, str]) -> bool:
    """True when `arch` is mentioned and at least one of its occurrences is not
    guarded by a capability marker."""
    segments = _hard_segments(haystack)
    occs = _arch_occurrences(haystack, arch, token_arch_map, segments)
    if not occs:
        return False
    if not markers:
        return True
    guarded = _guarded_segment_archs(haystack, token_arch_map, markers)
    return any((sid, arch) not in guarded for _, sid in occs)


def _is_arch_only_guarded(haystack: str, arch: str, markers: list[str],
                          token_arch_map: dict[str, str]) -> bool:
    """True when `arch` is mentioned but every occurrence is guarded."""
    segments = _hard_segments(haystack)
    occs = _arch_occurrences(haystack, arch, token_arch_map, segments)
    if not occs:
        return False
    guarded = _guarded_segment_archs(haystack, token_arch_map, markers)
    return all((sid, arch) in guarded for _, sid in occs)





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
    token_arch_map = _token_arch_map(policy)
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
    # only when the text also carries a DEVICE-SPECIFIC signal, so a host-only
    # src/scheduler.cpp is not mistaken for kernel work. The signal list is
    # deliberately device-specific: the bare word "kernel" is NOT a signal,
    # because host-side PRs routinely say "kernel scheduler"/"kernel launcher"
    # without touching device code (Codex R6). Absence of paths is not a pass —
    # the text must then carry a device signal (closes the empty-paths bypass).
    # The list is device-SPECIFIC: neither the generic word "kernel" nor the
    # library name "cutlass" is a signal, because host-side dispatch/plumbing code
    # routinely names CUTLASS without adding any device code (Codex R7). Only
    # device files (.cu/.cuh/.ptx) and device constructs qualify a .cpp/.h path.
    kernel_text_signals = ("__cuda_arch__", "mma.sync", "ldmatrix", "cp.async",
                           "wmma", ".cu", ".cuh", ".ptx", "__global__",
                           "__device__", "tensor core", "tensor-core", "warp-level")
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
    # Compute the capability-guarded (segment, arch) set ONCE, then reuse it for
    # both in-scope evidence and out-of-scope attribution. A token occurrence is
    # clean when its (segment, arch) pair is not guarded; an architecture is
    # included when ANY of its occurrences (across synonymous tokens) is clean.
    segments = _hard_segments(haystack)
    guarded = _guarded_segment_archs(haystack, token_arch_map, markers)

    def _token_has_clean(token_lower: str, arch: str) -> bool:
        return any((_seg_id(off, segments), arch) not in guarded
                   for off in _token_spans(haystack, token_lower))

    architectures: set[str] = set()
    evidence: list[dict] = []
    for tier in policy.get("evidence_tiers", []):
        for token, arch in (tier.get("maps_to", {}) or {}).items():
            if arch not in in_scope_archs:
                continue
            if _token_has_clean(token.lower(), arch):
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
            oos_arch = token_arch_map.get(token.lower(), f"oos:{reason}")
            if _token_has_clean(token.lower(), oos_arch):
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

    # An in-scope architecture present but ONLY as a capability guard / fallback.
    for arch in sorted(in_scope_archs):
        if _is_arch_only_guarded(haystack, arch, markers, token_arch_map):
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
