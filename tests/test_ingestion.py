#!/usr/bin/env python3
"""Tests for the content-ingestion pipeline (policy, classifier, generator,
refresh discovery). All offline — no test touches the network.

Run with:
    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PYTHON = sys.executable

sys.path.insert(0, str(SCRIPTS))
from classify_candidate import classify, load_policy  # noqa: E402

POLICY = load_policy(REPO)
ARCHS = {"sm75", "sm86", "sm89"}


def run_script(name, *args):
    return subprocess.run([PYTHON, str(SCRIPTS / name), *args],
                          capture_output=True, text=True)


class ClassifierTests(unittest.TestCase):
    """fixture-driven verdicts, deterministic, with the required negatives."""

    def _verdict(self, candidate):
        return classify(candidate, POLICY, ARCHS)

    def test_in_scope_includes(self):
        for token, arch in (("sm_75", "sm75"), ("sm_86", "sm86"), ("sm_89", "sm89"),
                            ("L40", "sm89"), ("T4", "sm75")):
            v = self._verdict({"title": f"optimize kernel for {token}",
                               "changed_paths": ["csrc/x.cu"]})
            self.assertEqual(v["decision"], "include", f"{token}: {v}")
            self.assertIn(arch, v["architectures"])
            self.assertTrue(v["architecture_evidence"])

    def test_deterministic(self):
        cand = {"title": "sm89 L40 fp8 path", "changed_paths": ["a.cu"]}
        self.assertEqual(self._verdict(cand), self._verdict(cand))

    def test_hopper_only_skipped(self):
        v = self._verdict({"title": "optimize sm90 H100 wgmma", "changed_paths": ["a.cu"]})
        self.assertEqual(v, {"decision": "skip", "reason": "hopper-only"})

    def test_blackwell_only_skipped(self):
        v = self._verdict({"title": "add tcgen05 sm100 path", "changed_paths": ["a.cu"]})
        self.assertEqual(v, {"decision": "skip", "reason": "blackwell-only"})

    def test_sm80_only_skipped(self):
        v = self._verdict({"title": "A100 sm80 only optimization", "changed_paths": ["a.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "sm80-only")

    def test_generic_no_arch_skipped(self):
        v = self._verdict({"title": "speed up the gemm loop", "changed_paths": ["a.cu"]})
        self.assertEqual(v, {"decision": "skip", "reason": "no-prehopper-evidence"})

    def test_framework_only_skipped(self):
        v = self._verdict({"title": "refactor scheduler", "changed_paths": ["vllm/core/scheduler.py"]})
        self.assertEqual(v["decision"], "skip")
        self.assertIn(v["reason"], ("framework-only", "non-kernel"))

    def test_docs_only_skipped(self):
        v = self._verdict({"title": "update readme", "changed_paths": ["docs/guide.md", "README.md"]})
        self.assertEqual(v["decision"], "skip")
        self.assertIn(v["reason"], ("docs-only", "non-kernel"))

    def test_capability_guard_not_a_false_include(self):
        # The ONLY sm75 mention is a capability guard -> must not include.
        v = self._verdict({"title": "MoE kernel",
                           "body": "Turing (sm75) is not supported; fall back to the cuda-core path.",
                           "changed_paths": ["csrc/moe.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "capability-guard-only")

    def test_clean_mention_beats_guard(self):
        # A guarded mention PLUS a clean optimization mention -> include.
        v = self._verdict({"title": "Add sm89 L40 FP8 kernel",
                           "body": "Note: sm75 not supported. Adds optimized sm_89 path.",
                           "changed_paths": ["csrc/x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])

    # --- regression tests for the Codex-found classifier bypasses -----------

    def test_bare_number_is_not_arch_evidence(self):
        # "increase buffer from 750 to 1024" must NOT be read as sm75 evidence.
        v = self._verdict({"title": "increase buffer from 750 to 1024",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "skip")

    def test_context_bearing_guard_is_evidence(self):
        v = self._verdict({"title": "add ada path",
                           "body": "guard __CUDA_ARCH__ == 890 around the mma",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])

    def test_trailing_guard_clause_bypass_closed(self):
        # The only sm75 mention trails into a guard in the next clause.
        v = self._verdict({"title": "MoE kernel",
                           "body": "Turing (sm75). Not supported; fall back to cuda core.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "capability-guard-only")

    def test_host_only_cpp_is_not_kernel(self):
        # sm89 in title but only a host-side .cpp with no kernel text signal.
        v = self._verdict({"title": "sm89 scheduler config",
                           "changed_paths": ["src/scheduler.cpp"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "non-kernel")

    def test_empty_paths_no_kernel_text_is_not_kernel(self):
        v = self._verdict({"title": "optimize for sm89"})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "non-kernel")

    def test_clc_only_is_blackwell(self):
        v = self._verdict({"title": "add clc cluster launch control", "changed_paths": ["x.cu"]})
        self.assertEqual(v, {"decision": "skip", "reason": "blackwell-only"})

    def test_out_of_scope_contrast_does_not_veto_clean_in_scope(self):
        # A clean sm86 target plus a CONTRASTIVE sm80 mention still includes.
        v = self._verdict({"title": "fix sm86 smem config",
                           "body": "A10 (sm86) has ~100KB/SM vs sm80's 164KB; clamp stages.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm86"])

    def test_trailing_guard_about_other_arch_does_not_taint(self):
        # Clean sm89 target; the trailing-sentence guard is about a DIFFERENT
        # arch (sm75), so it must not mark the sm89 mention as guarded.
        v = self._verdict({"title": "Adds optimized sm89 kernel. Not supported on sm75.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])

    def test_trailing_guard_about_same_arch_still_guards(self):
        # The only mention is sm75 and the trailing guard names no other arch,
        # so it genuinely refers back to sm75 -> capability-guard-only.
        v = self._verdict({"title": "MoE kernel",
                           "body": "Turing (sm75). Not supported; fall back to cuda core.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "capability-guard-only")


class GeneratorTests(unittest.TestCase):
    """generation from the committed seed manifest, offline."""

    def test_seed_generates_three_pages_one_per_arch(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            r = run_script("generate-pr-pages.py", "--root", str(kb))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            pages = list((kb / "sources" / "prs").rglob("PR-*.md"))
            self.assertGreaterEqual(len(pages), 3)
            archs = set()
            for p in pages:
                fm = _frontmatter(p)
                archs.update(fm.get("architectures", []))
                self.assertEqual(fm["source_category"], "upstream-code")
                self.assertTrue(fm["architectures"], f"{p} has empty architectures")
                self.assertIn("inclusion_reason", fm)
            self.assertEqual(archs, {"sm75", "sm86", "sm89"})

    def test_generated_repo_validates_and_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)
            self.assertEqual(run_script("generate-indices.py", "--root", str(kb)).returncode, 0)
            byrepo = (kb / "queries" / "by-repo.md").read_text(encoding="utf-8")
            self.assertIn("#29901", byrepo)
            self.assertIn("#1973", byrepo)

    def test_generation_needs_no_network(self):
        # Run with a poisoned PATH so `gh` (and most network tools) are absent,
        # and an env var that would break any accidental socket use.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            env = dict(os.environ)
            env["PATH"] = "/nonexistent"
            env["no_proxy"] = "*"
            r = subprocess.run([PYTHON, str(SCRIPTS / "generate-pr-pages.py"), "--root", str(kb)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            before = set((kb / "sources" / "prs").rglob("PR-*.md"))
            r = run_script("generate-pr-pages.py", "--root", str(kb), "--dry-run")
            self.assertEqual(r.returncode, 0)
            after = set((kb / "sources" / "prs").rglob("PR-*.md"))
            self.assertEqual(before, after)

    def test_invalid_tag_is_hard_error_not_a_page(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"][0]["tags"].append("tcgen05")  # Blackwell-only, not in vocab
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            r = run_script("generate-pr-pages.py", "--root", str(kb))
            self.assertEqual(r.returncode, 1)
            self.assertIn("not in data/tags.yaml", r.stderr)

    def test_skip_verdict_logs_not_pages(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            # Add a Hopper-only fixture + manifest entry -> must skip-log, no page.
            fx = kb / "tests" / "fixtures" / "seed" / "cutlass"
            fx.mkdir(parents=True, exist_ok=True)
            (fx / "PR-9001.json").write_text(json.dumps({
                "number": 9001, "title": "sm90 H100 wgmma only", "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"].append({
                "repo_slug": "cutlass", "repo": "NVIDIA/cutlass", "pr": 9001,
                "title": "sm90 H100 wgmma only", "author": "x", "date": "2025-01-01",
                "url": "https://example.com/9001", "status": "merged", "merge_sha": "abc123",
                "fixture": "tests/fixtures/seed/cutlass/PR-9001.json", "tags": [],
            })
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            self.assertFalse((kb / "sources" / "prs" / "cutlass" / "PR-9001.md").exists())
            skip = yaml.safe_load((kb / "data" / "pr-page-skipped.yaml").read_text(encoding="utf-8"))
            reasons = {r["pr_number"]: r["reason"] for r in skip["rows"]}
            self.assertEqual(reasons.get(9001), "hopper-only")
            # And the skip reason is in the policy taxonomy -> repo still validates.
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)


class RefreshTests(unittest.TestCase):
    """fixture-mode discovery, no network, idempotent, atomic."""

    def test_refresh_preserves_existing_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb),
                           "--repos", "flashinfer", "--searched-at", "2026-06-30")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            led = yaml.safe_load((kb / "candidates" / "flashinfer.yaml").read_text(encoding="utf-8"))
            rows = {x["number"]: x["decision"] for x in led["prs"]}
            self.assertEqual(rows[1973], "include")   # existing decision NOT rewritten
            self.assertEqual(rows[385], "needs-review")
            self.assertEqual(rows.get(999), "defer")  # new candidate added as defer

    def test_refresh_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            run_script("refresh_candidate_ledger.py", "--root", str(kb),
                       "--repos", "cutlass", "--searched-at", "2026-06-30")
            first = (kb / "candidates" / "cutlass.yaml").read_text(encoding="utf-8")
            run_script("refresh_candidate_ledger.py", "--root", str(kb),
                       "--repos", "cutlass", "--searched-at", "2026-06-30")
            second = (kb / "candidates" / "cutlass.yaml").read_text(encoding="utf-8")
            self.assertEqual(first, second)

    def test_refresh_default_mode_uses_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            env = dict(os.environ)
            env["PATH"] = "/nonexistent"  # no `gh` reachable
            r = subprocess.run([PYTHON, str(SCRIPTS / "refresh_candidate_ledger.py"),
                                "--root", str(kb), "--repos", "cutlass", "--searched-at", "2026-06-30"],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_refresh_subset_validator(self):
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            run_script("refresh_candidate_ledger.py", "--root", str(kb),
                       "--repos", "cutlass,flashinfer", "--searched-at", "2026-06-30")
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_refresh_result_repo_schema_enforced(self):
        # A repo entry missing schema-required fields (searched_at/window_start/
        # last_pr_date_seen) must be rejected, not silently OK.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            (kb / "data" / "refresh-search-results.yaml").write_text(
                "cutoff_date: '2026-06-30'\nrepos:\n- repo_slug: cutlass\n  pr_numbers_seen: []\n",
                encoding="utf-8")
            r = run_script("validate.py", "--root", str(kb))
            self.assertEqual(r.returncode, 1)
            self.assertIn("missing required field 'searched_at'", r.stdout)

    def test_refresh_honors_discovery_window(self):
        # Out-of-window PRs (before --since or after --until) must be dropped in
        # fixture mode, not merged into the ledger or refresh results.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            # Start from a clean cutlass ledger so this isolates window filtering
            # (the committed ledger may already carry fixture-discovered rows).
            (kb / "candidates" / "cutlass.yaml").write_text(
                "repo: NVIDIA/cutlass\nsearched_at: '2026-06-30'\nwindow_start: '2020-01-01'\n"
                "keywords_used: [sm75]\ntotal_candidates: 0\nincluded: 0\nexcluded: 0\n"
                "deferred: 0\nneeds_review: 0\nprs: []\n", encoding="utf-8")
            (kb / "tests" / "fixtures" / "gh" / "cutlass.json").write_text(json.dumps([
                {"number": 100, "title": "pre-window", "createdAt": "2019-12-31T00:00:00Z"},
                {"number": 200, "title": "in-window", "createdAt": "2024-01-01T00:00:00Z"},
                {"number": 300, "title": "future", "createdAt": "2027-01-01T00:00:00Z"},
            ]), encoding="utf-8")
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb), "--repos", "cutlass",
                           "--since", "2020-01-01", "--until", "2026-06-30", "--searched-at", "2026-06-30")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            led = yaml.safe_load((kb / "candidates" / "cutlass.yaml").read_text(encoding="utf-8"))
            nums = sorted(x["number"] for x in led["prs"])
            self.assertEqual(nums, [200], f"only in-window PR should merge, got {nums}")
            rsr = yaml.safe_load((kb / "data" / "refresh-search-results.yaml").read_text(encoding="utf-8"))
            cut = next(e for e in rsr["repos"] if e["repo_slug"] == "cutlass")
            self.assertEqual(cut["pr_numbers_seen"], [200])
            self.assertEqual(cut["last_pr_date_seen"], "2024-01-01")

    def test_refresh_rejects_undated_candidate(self):
        # A malformed fixture row without a valid date is a hard error, not a
        # silent leak.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            (kb / "tests" / "fixtures" / "gh" / "cutlass.json").write_text(json.dumps([
                {"number": 1, "title": "no date"},
            ]), encoding="utf-8")
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb), "--repos", "cutlass",
                           "--searched-at", "2026-06-30")
            self.assertNotEqual(r.returncode, 0)


class CommittedArtifactTests(unittest.TestCase):
    """the refresh-search-results deliverable must exist in the live repo
    (a clean checkout), cover the tracked repo set, and validate."""

    def test_refresh_search_results_committed_and_valid(self):
        rsr = REPO / "data" / "refresh-search-results.yaml"
        self.assertTrue(rsr.is_file(), "data/refresh-search-results.yaml must be committed")
        data = yaml.safe_load(rsr.read_text(encoding="utf-8"))
        self.assertIn("cutoff_date", data)
        slugs = {e["repo_slug"] for e in data["repos"]}
        # All seven tracked repos represented.
        self.assertEqual(slugs, {"cutlass", "sglang", "vllm", "flashinfer",
                                 "pytorch", "tensorrt-llm", "cuvs"})
        for e in data["repos"]:
            self.assertEqual(e["pr_numbers_seen"], sorted(e["pr_numbers_seen"]),
                             f"{e['repo_slug']} pr_numbers_seen not sorted")
            for k in ("searched_at", "window_start", "last_pr_date_seen"):
                self.assertIn(k, e)
        # The live repo validates with the committed artifact present.
        self.assertEqual(run_script("validate.py", "--root", str(REPO)).returncode, 0)

    def test_vllm_seed_has_no_incorrect_int4_shape(self):
        # The Turing INT4 m16n8k8 shape (a Codex hard error) must not reappear in
        # the committed seed manifest or generated vLLM page.
        manifest = (REPO / "tests" / "fixtures" / "seed" / "seed-manifest.yaml").read_text(encoding="utf-8")
        page = (REPO / "sources" / "prs" / "vllm" / "PR-29901.md").read_text(encoding="utf-8")
        self.assertNotIn("m16n8k8", manifest)
        self.assertNotIn("m16n8k8", page)


class ScaleTests(unittest.TestCase):
    """The ingestion-generated source-pr page shape validates and indexes
    at volume, offline, deterministically — separate from goal-1 ingestion."""

    N = 300
    ARCH_CYCLE = ["sm75", "sm86", "sm89"]

    def _make_synthetic_kb(self, dest: Path) -> Path:
        kb = _clone_kb(dest)
        # Remove the small seed pages so the scale test operates purely on the
        # newly-generated synthetic volume (operate on generated volume, not the existing pages).
        shutil.rmtree(kb / "sources" / "prs", ignore_errors=True)
        for i in range(self.N):
            arch = self.ARCH_CYCLE[i % 3]
            slug = "synthrepo"
            d = kb / "sources" / "prs" / slug
            d.mkdir(parents=True, exist_ok=True)
            fm = {
                "id": f"pr-{slug}-{i}", "repo": "synth/repo", "pr": i,
                "title": f"synthetic kernel pr {i}", "author": "synth",
                "date": "2024-01-01", "url": f"https://example.com/{i}",
                "source_category": "upstream-code", "architectures": [arch],
                "tags": ["tensor-core"], "captured_at": "2026-06-30", "status": "open",
            }
            front = yaml.safe_dump(fm, sort_keys=False)
            (d / f"PR-{i}.md").write_text(f"---\n{front}---\n\n# synthetic {i}\n\nbody\n", encoding="utf-8")
        return kb

    def test_scale_validates_and_indexes_deterministically(self):
        with tempfile.TemporaryDirectory() as d:
            kb = self._make_synthetic_kb(Path(d))
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)
            self.assertEqual(run_script("generate-indices.py", "--root", str(kb)).returncode, 0)
            byrepo1 = (kb / "queries" / "by-repo.md").read_text(encoding="utf-8")
            self.assertIn("synth/repo", byrepo1)
            self.assertIn(f"{self.N} PRs", byrepo1)
            # Determinism at scale: a second run produces no diff.
            self.assertEqual(run_script("generate-indices.py", "--root", str(kb)).returncode, 0)
            self.assertEqual(byrepo1, (kb / "queries" / "by-repo.md").read_text(encoding="utf-8"))

    def test_scale_still_catches_invalid_page(self):
        with tempfile.TemporaryDirectory() as d:
            kb = self._make_synthetic_kb(Path(d))
            bad = kb / "sources" / "prs" / "synthrepo" / "PR-bad.md"
            fm = {
                "id": "pr-synthrepo-bad", "repo": "synth/repo", "pr": 99999,
                "title": "out of scope", "author": "x", "date": "2024-01-01",
                "url": "https://example.com/x", "source_category": "upstream-code",
                "architectures": ["sm90"], "tags": ["tensor-core"],
                "captured_at": "2026-06-30", "status": "open",
            }
            bad.write_text("---\n" + yaml.safe_dump(fm, sort_keys=False) + "---\n\nx\n", encoding="utf-8")
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 1)


# ---- helpers ---------------------------------------------------------------
def _clone_kb(dest: Path) -> Path:
    """Copy the committed corpus + scripts + fixtures into a temp KB root so
    tests can mutate freely without touching the real repo."""
    for sub in ("scripts", "data", "wiki", "sources", "queries", "candidates",
                "tests/fixtures/seed", "tests/fixtures/gh", "references", "docs"):
        src = REPO / sub
        if src.exists():
            shutil.copytree(src, dest / sub, dirs_exist_ok=True)
    return dest


def _frontmatter(path: Path) -> dict:
    import re
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text, re.DOTALL)
    return yaml.safe_load(m.group(1)) if m else {}


if __name__ == "__main__":
    unittest.main(verbosity=2)
