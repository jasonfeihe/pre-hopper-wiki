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

    def test_bare_kernel_word_does_not_rescue_host_cpp(self):
        # R6 regression: the generic word "kernel" (as in "kernel scheduler") is
        # NOT device evidence, so a host-only .cpp that merely says "kernel" +
        # names an in-scope arch must still skip as non-kernel.
        v = self._verdict({"title": "Host-only A10 kernel scheduler change",
                           "body": "Refactor the kernel launcher scheduling on the host.",
                           "changed_paths": ["src/scheduler.cpp"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "non-kernel")

    def test_device_construct_in_cpp_still_counts(self):
        # A device construct (__global__/mma.sync) in an ambiguous .cpp IS
        # evidence -> a clean in-scope arch still includes.
        v = self._verdict({"title": "A10 attention",
                           "body": "add __global__ mma.sync path for sm86",
                           "changed_paths": ["src/attn.cpp"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm86"])

    def test_cutlass_word_does_not_rescue_host_cpp(self):
        # R7 regression: the library name "cutlass" is NOT device evidence, so a
        # host-only .cpp that names CUTLASS + an in-scope arch must skip.
        v = self._verdict({"title": "sm89 cutlass plumbing",
                           "body": "Host-side CUTLASS dispatch wiring only.",
                           "changed_paths": ["src/dispatch.cpp"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "non-kernel")

    def test_cutlass_device_file_still_includes(self):
        # A real CUTLASS .cu device file with a clean in-scope arch still includes
        # (the fix removes only the TEXT signal, not device-path evidence).
        v = self._verdict({"title": "sm89 cutlass gemm",
                           "body": "add cutlass fp8 path",
                           "changed_paths": ["csrc/gemm_sm89.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])

    def test_cu_filename_in_prose_does_not_rescue_host_cpp(self):
        # R12 regression: a device file EXTENSION mentioned in prose (e.g. "see
        # foo.cu") is NOT device evidence — only a real .cu/.cuh/.ptx changed path
        # or a device construct is. A host-only .cpp referencing a .cu filename +
        # an in-scope arch must still skip as non-kernel.
        v = self._verdict({"title": "sm89 dispatch refactor",
                           "body": "Host-side plumbing; see kernels in foo.cu / bar.cuh for context.",
                           "changed_paths": ["src/dispatch.cpp"]})
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

    def test_same_clause_comma_guard_does_not_taint_other_arch(self):
        # R3 regression: a guard sharing the SAME hard segment (comma-joined)
        # with a clean arch must attribute only to the arch it grammatically
        # governs (sm75 after "not supported on"), leaving sm89 a clean include.
        # Comma is NOT a hard boundary, so this exercises in-segment attribution.
        v = self._verdict({"title": "Adds optimized sm89 kernel, not supported on sm75.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])

    def test_a10g_device_maps_to_sm86(self):
        # R3 regression: the A10G device name resolves to sm86 like the A10.
        v = self._verdict({"title": "Tune A10G decode kernel",
                           "body": "ldmatrix path for the A10G inference card.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm86"])

    def test_multi_arch_guard_clause_guards_all_listed_archs(self):
        # R4 regression: a single guard clause naming a COORDINATED arch list
        # ("not supported on sm75 and sm89") must guard BOTH archs, not just the
        # nearest one -> the PR has no clean evidence -> capability-guard-only.
        v = self._verdict({"title": "MoE kernel",
                           "body": "Adds a path; not supported on sm75 and sm89.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "capability-guard-only")

    def test_multi_arch_guard_list_before_marker(self):
        # Symmetric: the coordinated list precedes the marker.
        v = self._verdict({"title": "MoE kernel",
                           "body": "sm75 and sm89 not supported; fall back to generic.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "skip")
        self.assertEqual(v["reason"], "capability-guard-only")

    def test_contrastive_after_guard_list_stays_clean(self):
        # A non-connector gap ("..., but optimized for") breaks the coordinated
        # run, so sm89 after the contrast is still a clean include.
        v = self._verdict({"title": "kernel",
                           "body": "not supported on sm75, but optimized for sm89.",
                           "changed_paths": ["x.cu"]})
        self.assertEqual(v["decision"], "include")
        self.assertEqual(v["architectures"], ["sm89"])


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

    def test_invalid_status_is_hard_error_not_a_page(self):
        # R5 regression: a mistyped status must be rejected PRE-EMIT (mirroring the
        # source-pr schema enum), never written into a page that validate.py would
        # later reject. Use a fresh in-scope entry so no committed page pre-exists.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            fx = kb / "tests" / "fixtures" / "seed" / "cutlass"
            fx.mkdir(parents=True, exist_ok=True)
            (fx / "PR-9200.json").write_text(json.dumps({
                "number": 9200, "title": "Optimize sm89 L40 fp8 mma.sync kernel",
                "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"].append({
                "repo_slug": "cutlass", "repo": "NVIDIA/cutlass", "pr": 9200,
                "title": "Optimize sm89 L40 fp8 mma.sync kernel", "author": "x",
                "date": "2025-01-01", "url": "https://example.com/9200",
                "status": "mergd",  # typo, not in [open, merged, closed]
                "fixture": "tests/fixtures/seed/cutlass/PR-9200.json", "tags": [],
            })
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            r = run_script("generate-pr-pages.py", "--root", str(kb))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("status 'mergd'", r.stderr)
            page = kb / "sources" / "prs" / "cutlass" / "PR-9200.md"
            self.assertFalse(page.exists(), "invalid status must not write a page")

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

    def test_skip_verdict_removes_stale_page(self):
        # R3 regression: a PR that was previously an include (page on disk) and
        # now classifies as skip must have its stale page REMOVED, not orphaned.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            # First pass: an in-scope fixture produces a real page.
            fx = kb / "tests" / "fixtures" / "seed" / "cutlass"
            fx.mkdir(parents=True, exist_ok=True)
            fixture = fx / "PR-9100.json"
            fixture.write_text(json.dumps({
                "number": 9100, "title": "Optimize sm89 L40 fp8 mma.sync kernel",
                "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"].append({
                "repo_slug": "cutlass", "repo": "NVIDIA/cutlass", "pr": 9100,
                "title": "Optimize sm89 L40 fp8 mma.sync kernel", "author": "x",
                "date": "2025-01-01", "url": "https://example.com/9100",
                "status": "merged", "merge_sha": "abc123",
                "fixture": "tests/fixtures/seed/cutlass/PR-9100.json", "tags": [],
            })
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            page = kb / "sources" / "prs" / "cutlass" / "PR-9100.md"
            self.assertTrue(page.exists(), "include should have produced a page")

            # Second pass: flip the fixture to Hopper-only so it now skips.
            fixture.write_text(json.dumps({
                "number": 9100, "title": "sm90 H100 wgmma only", "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            self.assertFalse(page.exists(), "skip must remove the stale include page")
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_deleted_manifest_entry_removes_generated_page(self):
        # R8 regression: removing a seed entry entirely must delete its previously
        # generated page, so regeneration is a faithful rebuild from the manifest.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            page = kb / "sources" / "prs" / "flashinfer" / "PR-385.md"
            self.assertTrue(page.exists(), "committed seed page should be present")
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"] = [e for e in data["entries"]
                               if not (e["repo_slug"] == "flashinfer" and e["pr"] == 385)]
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            self.assertFalse(page.exists(), "deleted manifest entry must remove its page")
            # Other pages survive; repo still validates.
            self.assertTrue((kb / "sources" / "prs" / "flashinfer" / "PR-1973.md").exists())
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_deleted_skip_only_entry_drops_audit_row(self):
        # R8 regression: a skip-only seed entry, once removed from the manifest,
        # must not linger in data/pr-page-skipped.yaml.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            fx = kb / "tests" / "fixtures" / "seed" / "cutlass"
            fx.mkdir(parents=True, exist_ok=True)
            (fx / "PR-9002.json").write_text(json.dumps({
                "number": 9002, "title": "sm90 H100 wgmma only", "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            entry = {
                "repo_slug": "cutlass", "repo": "NVIDIA/cutlass", "pr": 9002,
                "title": "sm90 H100 wgmma only", "author": "x", "date": "2025-01-01",
                "url": "https://example.com/9002", "status": "merged", "merge_sha": "abc123",
                "fixture": "tests/fixtures/seed/cutlass/PR-9002.json", "tags": [],
            }
            data["entries"].append(entry)
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            skip = yaml.safe_load((kb / "data" / "pr-page-skipped.yaml").read_text(encoding="utf-8"))
            self.assertIn("pr-cutlass-9002", {r["pr_id"] for r in skip["rows"]})
            # Remove the entry and regenerate -> its audit row must be gone.
            data["entries"] = [e for e in data["entries"] if e["pr"] != 9002]
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            skip = yaml.safe_load((kb / "data" / "pr-page-skipped.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("pr-cutlass-9002", {r["pr_id"] for r in skip["rows"]})
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_generated_pages_carry_provenance_marker(self):
        # R10: every generated seed page must carry the generated_by marker so the
        # stale sweep can distinguish generator-owned pages from hand-authored ones.
        for name in ("vllm/PR-29901.md", "flashinfer/PR-385.md", "flashinfer/PR-1973.md"):
            page = (REPO / "sources" / "prs" / name).read_text(encoding="utf-8")
            self.assertIn("generated_by: generate-pr-pages", page, f"{name} missing provenance marker")

    def test_hand_authored_page_survives_regeneration(self):
        # R10 regression: a hand-authored source-pr page (append-only workflow, NO
        # provenance marker) must NOT be deleted by a regeneration, while a
        # generator-owned page dropped from the manifest still is.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            hand = kb / "sources" / "prs" / "pytorch" / "PR-555.md"
            hand.parent.mkdir(parents=True, exist_ok=True)
            hand.write_text(
                "---\nid: pr-pytorch-555\nrepo: pytorch/pytorch\npr: 555\n"
                "title: Hand-authored sm86 page\nauthor: someone\ndate: '2024-03-03'\n"
                "url: https://github.com/pytorch/pytorch/pull/555\n"
                "source_category: upstream-code\narchitectures: [sm86]\ntags: [sm86]\n"
                "captured_at: '2026-06-30'\nstatus: merged\nmerge_sha: deadbeef\n---\n\n"
                "# Hand-authored sm86 page\n\nAdded via the append-only workflow.\n",
                encoding="utf-8")
            # Also drop a generator-owned entry (flashinfer#385) from the manifest.
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"] = [e for e in data["entries"]
                               if not (e["repo_slug"] == "flashinfer" and e["pr"] == 385)]
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            self.assertEqual(run_script("generate-pr-pages.py", "--root", str(kb)).returncode, 0)
            self.assertTrue(hand.exists(), "hand-authored page (no marker) must survive")
            self.assertFalse((kb / "sources" / "prs" / "flashinfer" / "PR-385.md").exists(),
                             "generator-owned page dropped from manifest must still be removed")
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_partial_custom_manifest_preserves_out_of_scope_pages(self):
        # R11 regression: a PARTIAL custom --manifest is authoritative only for the
        # ids it declares. Regenerating one repo must NOT delete unrelated
        # generator-owned pages, but must still reconcile the entries it declares.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            full = yaml.safe_load(
                (kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml").read_text(encoding="utf-8"))
            one = {"captured_at": full.get("captured_at", "2026-06-30"),
                   "entries": [e for e in full["entries"] if e["repo_slug"] == "vllm"]}
            custom = kb / "one-entry-manifest.yaml"
            custom.write_text(yaml.safe_dump(one), encoding="utf-8")
            r = run_script("generate-pr-pages.py", "--root", str(kb), "--manifest", str(custom))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            # Unrelated pages (not in this manifest) survive.
            self.assertTrue((kb / "sources" / "prs" / "flashinfer" / "PR-385.md").exists())
            self.assertTrue((kb / "sources" / "prs" / "flashinfer" / "PR-1973.md").exists())
            # The declared entry is (re)generated.
            self.assertTrue((kb / "sources" / "prs" / "vllm" / "PR-29901.md").exists())
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_partial_manifest_still_reconciles_its_own_include_to_skip(self):
        # R11: a partial manifest still removes a page for one of ITS OWN entries
        # when that entry flips include->skip (scoped reconciliation, not disabled).
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            full = yaml.safe_load(
                (kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml").read_text(encoding="utf-8"))
            e1973 = next(e for e in full["entries"] if e["pr"] == 1973)
            # Flip its fixture to hopper-only so it now classifies as skip.
            (kb / e1973["fixture"]).write_text(json.dumps({
                "number": 1973, "title": "sm90 H100 wgmma only", "changed_paths": ["a.cu"],
            }), encoding="utf-8")
            one = {"captured_at": full.get("captured_at", "2026-06-30"), "entries": [e1973]}
            custom = kb / "one.yaml"
            custom.write_text(yaml.safe_dump(one), encoding="utf-8")
            r = run_script("generate-pr-pages.py", "--root", str(kb), "--manifest", str(custom))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertFalse((kb / "sources" / "prs" / "flashinfer" / "PR-1973.md").exists(),
                             "an include->skip flip in the partial manifest must remove its own page")
            # An unrelated page still survives.
            self.assertTrue((kb / "sources" / "prs" / "vllm" / "PR-29901.md").exists())
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

    def test_explicit_default_manifest_path_is_full_rebuild(self):
        # R12 regression: passing the DEFAULT seed manifest path explicitly must
        # behave exactly like the default invocation — a full rebuild that removes
        # stale pages for entries deleted from that manifest (not a partial run).
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            manifest = kb / "tests" / "fixtures" / "seed" / "seed-manifest.yaml"
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            data["entries"] = [e for e in data["entries"]
                               if not (e["repo_slug"] == "flashinfer" and e["pr"] == 385)]
            manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
            r = run_script("generate-pr-pages.py", "--root", str(kb), "--manifest", str(manifest))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertFalse((kb / "sources" / "prs" / "flashinfer" / "PR-385.md").exists(),
                             "explicit default-path run must still clean up a deleted entry's page")
            self.assertTrue((kb / "sources" / "prs" / "flashinfer" / "PR-1973.md").exists())
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)



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

    def test_subset_refresh_preserves_untouched_repos(self):
        # R6 regression: a targeted `--repos flashinfer` refresh must NOT erase the
        # other tracked repos from data/refresh-search-results.yaml.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            rsr = kb / "data" / "refresh-search-results.yaml"
            before = {e["repo_slug"] for e in yaml.safe_load(rsr.read_text(encoding="utf-8"))["repos"]}
            self.assertEqual(len(before), 7, "fixture should start with all 7 tracked repos")
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb),
                           "--repos", "flashinfer", "--searched-at", "2026-06-30")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            after = {e["repo_slug"] for e in yaml.safe_load(rsr.read_text(encoding="utf-8"))["repos"]}
            self.assertEqual(after, before, "subset refresh must preserve untouched repos")
            self.assertEqual(run_script("validate.py", "--root", str(kb)).returncode, 0)

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

    def test_refresh_result_malformed_pr_numbers_seen_is_error_not_crash(self):
        # R9 regression: a scalar pr_numbers_seen must be a validation error, not a
        # TypeError at sorted() that crashes the validation-gated tooling.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            (kb / "data" / "refresh-search-results.yaml").write_text(
                "cutoff_date: '2026-06-30'\nrepos:\n- repo_slug: cutlass\n"
                "  searched_at: '2026-06-30'\n  window_start: '2020-01-01'\n"
                "  last_pr_date_seen: ''\n  pr_numbers_seen: 1\n",  # scalar, not a list
                encoding="utf-8")
            r = run_script("validate.py", "--root", str(kb))
            self.assertEqual(r.returncode, 1)
            self.assertIn("pr_numbers_seen must be a list", r.stdout)
            self.assertNotIn("Traceback", r.stderr)

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

    def test_refresh_tighter_window_drops_stale_rows(self):
        # R7 regression: re-refreshing with a TIGHTER window must drop existing
        # rows that fall outside it, so ledger rows stay consistent with the
        # rewritten window_start.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            (kb / "candidates" / "cutlass.yaml").write_text(
                "repo: NVIDIA/cutlass\nsearched_at: '2026-06-30'\nwindow_start: '2020-01-01'\n"
                "keywords_used: [sm75]\ntotal_candidates: 0\nincluded: 0\nexcluded: 0\n"
                "deferred: 0\nneeds_review: 0\nprs: []\n", encoding="utf-8")
            (kb / "tests" / "fixtures" / "gh" / "cutlass.json").write_text(json.dumps([
                {"number": 1989, "title": "old", "createdAt": "2023-08-01T00:00:00Z"},
                {"number": 2200, "title": "newer", "createdAt": "2024-06-01T00:00:00Z"},
            ]), encoding="utf-8")
            # Broad window: both merge.
            run_script("refresh_candidate_ledger.py", "--root", str(kb), "--repos", "cutlass",
                       "--since", "2020-01-01", "--until", "2026-06-30", "--searched-at", "2026-06-30")
            broad = yaml.safe_load((kb / "candidates" / "cutlass.yaml").read_text(encoding="utf-8"))
            self.assertEqual(sorted(x["number"] for x in broad["prs"]), [1989, 2200])
            # Tighter window: the 2023 row must be dropped.
            run_script("refresh_candidate_ledger.py", "--root", str(kb), "--repos", "cutlass",
                       "--since", "2024-01-01", "--until", "2026-06-30", "--searched-at", "2026-06-30")
            tight = yaml.safe_load((kb / "candidates" / "cutlass.yaml").read_text(encoding="utf-8"))
            self.assertEqual(sorted(x["number"] for x in tight["prs"]), [2200])
            self.assertEqual(tight["window_start"], "2024-01-01")
            self.assertEqual(tight["total_candidates"], 1)

    def test_refresh_advances_cutoff_monotonically(self):
        # R7 regression: a newer --searched-at advances refresh-cutoff.yaml; an
        # equal/older date leaves it byte-identical (never moves backward).
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            rc = kb / "data" / "refresh-cutoff.yaml"
            self.assertEqual(yaml.safe_load(rc.read_text(encoding="utf-8"))["cutoff_date"], "2026-06-30")
            # Newer date -> advance, notes preserved, header comment preserved.
            run_script("refresh_candidate_ledger.py", "--root", str(kb),
                       "--repos", "cutlass", "--searched-at", "2026-07-01")
            after = rc.read_text(encoding="utf-8")
            data = yaml.safe_load(after)
            self.assertEqual(data["cutoff_date"], "2026-07-01")
            self.assertTrue(data.get("notes"), "notes must be preserved")
            self.assertTrue(after.lstrip().startswith("#"), "header comment must be preserved")
            # Older date -> no backward move, byte-identical.
            run_script("refresh_candidate_ledger.py", "--root", str(kb),
                       "--repos", "cutlass", "--searched-at", "2026-06-30")
            self.assertEqual(rc.read_text(encoding="utf-8"), after, "must not move cutoff backward")

    def test_refresh_all_unknown_repos_is_noop_and_nonzero(self):
        # R13 regression: --repos resolving to only unknown slugs is a no-op — it
        # must NOT record a fictitious review round (rewrite refresh-search-results
        # / advance refresh-cutoff), and must exit non-zero.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            rsr = kb / "data" / "refresh-search-results.yaml"
            rc = kb / "data" / "refresh-cutoff.yaml"
            rsr_before = rsr.read_text(encoding="utf-8")
            rc_before = rc.read_text(encoding="utf-8")
            # A future --searched-at that WOULD advance the cutoff if it ran.
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb),
                           "--repos", "typo", "--searched-at", "2026-07-15")
            self.assertNotEqual(r.returncode, 0, "all-unknown --repos must exit non-zero")
            self.assertEqual(rsr.read_text(encoding="utf-8"), rsr_before,
                             "refresh-search-results.yaml must be untouched on a no-op refresh")
            self.assertEqual(rc.read_text(encoding="utf-8"), rc_before,
                             "refresh-cutoff.yaml must be untouched on a no-op refresh")

    def test_refresh_mixed_known_unknown_repos_still_writes(self):
        # R13: a mix of tracked + unknown slugs still refreshes the tracked one
        # (the no-op guard only fires when NOTHING tracked was refreshed).
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            r = run_script("refresh_candidate_ledger.py", "--root", str(kb),
                           "--repos", "cutlass,typo", "--searched-at", "2026-06-30")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

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

    def test_committed_ledger_exclude_reasons_are_taxonomy_keys(self):
        # R6: every `exclude` row in a committed ledger must cite a skip-taxonomy
        # key, not free-form prose (prose belongs in reason_detail).
        policy = yaml.safe_load((REPO / "data" / "inclusion-policy.yaml").read_text(encoding="utf-8"))
        taxonomy = set(policy["skip_reasons"].keys())
        for ledger in (REPO / "candidates").glob("*.yaml"):
            data = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
            for row in data.get("prs", []):
                if row.get("decision") == "exclude":
                    self.assertIn(row.get("reason"), taxonomy,
                                  f"{ledger.name} PR {row.get('number')} exclude reason not in taxonomy")

    def test_ledger_exclude_reason_off_taxonomy_is_rejected(self):
        # R6 regression: the validator must reject an exclude row whose reason is
        # not a skip-taxonomy key.
        with tempfile.TemporaryDirectory() as d:
            kb = _clone_kb(Path(d))
            led = kb / "candidates" / "flashinfer.yaml"
            data = yaml.safe_load(led.read_text(encoding="utf-8"))
            for row in data["prs"]:
                if row["decision"] == "exclude":
                    row["reason"] = "totally-made-up-reason"
                    break
            led.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            r = run_script("validate.py", "--root", str(kb))
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("exclude reason", r.stdout + r.stderr)


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
