#!/usr/bin/env python3
"""Test suite for the pre-Hopper kernel wiki tooling.

Run with:
    uv run python -m unittest discover -s tests -v
    # or
    uv run python tests/test_tooling.py

The suite builds throwaway knowledge-base roots in temp dirs (copying the real
data/*.yaml and selected fixture pages) so it never mutates the live wiki/.
It covers, among others:
  * empty-KB validation passes and emits six valid header-only indices
  * generate-indices is deterministic (twice-run no diff) and regenerates over a
    hand-edited index
  * each invalid fixture is rejected by validate.py
  * unparseable frontmatter is reported, not silently swallowed
  * a schema/vocabulary-invalid page makes all five entry-point scripts fail
  * PREHOPPER_WIKI_ROOT override + BLACKWELL_WIKI_ROOT no-op
  * the live seeded corpus validates clean
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
DATA = REPO / "data"
FIXTURES = REPO / "tests" / "fixtures" / "invalid"
PYTHON = sys.executable

REQUIRED_DATA = ["schemas.yaml", "tags.yaml", "version-claims.yaml", "refresh-cutoff.yaml"]
INDEX_FILES = [
    "by-problem.md", "by-technique.md", "by-hardware-feature.md",
    "by-repo.md", "by-kernel-type.md", "by-language.md",
]


def run_script(name, *args, root=None, env=None):
    cmd = [PYTHON, str(SCRIPTS / name), *args]
    if root is not None:
        cmd += ["--root", str(root)]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env)


def make_kb(tmp: Path, wiki_pages=None, source_pages=None):
    """Build a minimal KB root: real data/, empty tree, plus optional pages.

    wiki_pages / source_pages are lists of (relative_path, content_or_srcpath).
    """
    (tmp / "data").mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_DATA:
        shutil.copy(DATA / f, tmp / "data" / f)
    shutil.copy(DATA / "aliases.yaml", tmp / "data" / "aliases.yaml")
    for sub in ("hardware", "techniques", "kernels", "patterns", "languages", "migration"):
        (tmp / "wiki" / sub).mkdir(parents=True, exist_ok=True)
    (tmp / "sources" / "docs").mkdir(parents=True, exist_ok=True)
    (tmp / "queries").mkdir(parents=True, exist_ok=True)
    for rel, content in (wiki_pages or []) + (source_pages or []):
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    return tmp


class EmptyKBTests(unittest.TestCase):
    def test_empty_validates_and_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
            g = run_script("generate-indices.py", root=kb)
            self.assertEqual(g.returncode, 0, g.stdout + g.stderr)
            for idx in INDEX_FILES:
                p = kb / "queries" / idx
                self.assertTrue(p.is_file(), f"{idx} not written")
                self.assertIn("Auto-generated", p.read_text(encoding="utf-8"))

    def test_empty_retrieval_graceful(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            q = run_script("query.py", "anything", root=kb)
            self.assertEqual(q.returncode, 0)
            self.assertIn("No matching pages.", q.stdout)
            g = run_script("grep_wiki.py", "anything", root=kb)
            self.assertEqual(g.returncode, 0)
            self.assertIn("No matches.", g.stdout)


class DeterminismTests(unittest.TestCase):
    def test_twice_run_no_diff_and_regenerates_over_edit(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            run_script("generate-indices.py", root=kb)
            first = {f: (kb / "queries" / f).read_text(encoding="utf-8") for f in INDEX_FILES}
            run_script("generate-indices.py", root=kb)
            for f in INDEX_FILES:
                self.assertEqual(first[f], (kb / "queries" / f).read_text(encoding="utf-8"),
                                 f"{f} changed on rerun (non-deterministic)")
            # Manual edit must not persist across a regeneration.
            target = kb / "queries" / "by-technique.md"
            target.write_text("HAND EDITED — should be overwritten\n", encoding="utf-8")
            run_script("generate-indices.py", root=kb)
            self.assertEqual(first["by-technique.md"], target.read_text(encoding="utf-8"))


class NegativeFixtureTests(unittest.TestCase):
    def _reject(self, fixture_name):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            # provide a resolvable source so only the intended rule fails
            shutil.copy(REPO / "sources" / "docs" / "ptx-isa.md",
                        kb / "sources" / "docs" / "ptx-isa.md")
            shutil.copy(FIXTURES / fixture_name, kb / "wiki" / "hardware" / fixture_name)
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 1, f"{fixture_name} should fail validation:\n{v.stdout}")
            return v.stdout

    def test_out_of_scope_arch_rejected(self):
        self.assertIn("out of scope", self._reject("bad-arch.md"))

    def test_blackwell_tag_rejected(self):
        out = self._reject("blackwell-tag.md")
        self.assertIn("controlled vocabulary", out)

    def test_broken_link_rejected(self):
        self.assertIn("unknown id", self._reject("bad-link.md"))

    def test_empty_sources_rejected(self):
        self.assertIn("at least one source", self._reject("empty-sources.md"))

    def test_unparseable_reported_by_validate(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            shutil.copy(FIXTURES / "unparseable.md", kb / "wiki" / "hardware" / "unparseable.md")
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 1)
            self.assertIn("frontmatter", v.stdout)

    def test_unparseable_reported_by_generate_indices(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            shutil.copy(FIXTURES / "unparseable.md", kb / "wiki" / "hardware" / "unparseable.md")
            g = run_script("generate-indices.py", root=kb)
            self.assertEqual(g.returncode, 1, "generate-indices must fail on unparseable frontmatter")
            # The validation gate reports the offending file (to stderr) before
            # any index is written; the message names the frontmatter problem.
            self.assertIn("frontmatter", g.stdout + g.stderr)
            # And no index must have been written from the invalid corpus.
            idx = kb / "queries" / "by-hardware-feature.md"
            self.assertFalse(idx.is_file() and "Untitled" in idx.read_text(encoding="utf-8"))


class RootResolutionTests(unittest.TestCase):
    def test_env_override_and_blackwell_noop(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            # PREHOPPER_WIKI_ROOT points at kb; run from an unrelated cwd.
            with tempfile.TemporaryDirectory() as cwd:
                r = subprocess.run(
                    [PYTHON, str(SCRIPTS / "validate.py")],
                    capture_output=True, text=True, cwd=cwd,
                    env={**os.environ, "PREHOPPER_WIKI_ROOT": str(kb)},
                )
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            # Bad PREHOPPER_WIKI_ROOT hard-errors.
            r2 = subprocess.run(
                [PYTHON, str(SCRIPTS / "validate.py")],
                capture_output=True, text=True,
                env={**os.environ, "PREHOPPER_WIKI_ROOT": tempfile.gettempdir()},
            )
            self.assertEqual(r2.returncode, 2)
            # BLACKWELL_WIKI_ROOT is a no-op: setting it does not redirect the root.
            r3 = subprocess.run(
                [PYTHON, str(SCRIPTS / "validate.py")],
                capture_output=True, text=True, cwd=str(REPO),
                env={**os.environ, "BLACKWELL_WIKI_ROOT": tempfile.gettempdir()},
            )
            self.assertEqual(r3.returncode, 0, r3.stdout + r3.stderr)


class LiveCorpusTests(unittest.TestCase):
    def test_live_seeded_corpus_validates(self):
        v = run_script("validate.py", root=REPO)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)


# A small initial corpus (one source + one wiki page) used by the incremental test.
_DOC = """---
id: doc-seed
title: Seed Doc
url: https://example.com/seed
source_category: official-doc
architectures: [sm75]
tags: [tensor-core]
retrieved_at: '2026-06-30'
---
Seed source body.
"""

_HW = """---
id: hw-seed
title: Seed Hardware Page
type: hardware
architectures: [sm75]
tags: [tensor-core, mma-sync]
confidence: inferred
related: []
sources: [doc-seed]
---
Seed hardware body.
"""

# The new pages added during the incremental step.
_DOC_NEW = """---
id: doc-seed-2
title: Second Seed Doc
url: https://example.com/seed2
source_category: official-doc
architectures: [sm86]
tags: [cp-async]
retrieved_at: '2026-06-30'
---
Second source body.
"""

_HW_NEW = """---
id: hw-seed-2
title: Second Seed Hardware Page
type: hardware
architectures: [sm86]
tags: [cp-async, shared-memory]
confidence: inferred
related: [hw-seed]
sources: [doc-seed-2]
---
Second hardware body, links back to the first via related.
"""


class IncrementalUpdateTests(unittest.TestCase):
    def test_add_validate_regenerate_preserves_prior_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(
                Path(d),
                wiki_pages=[("wiki/hardware/hw-seed.md", _HW)],
                source_pages=[("sources/docs/doc-seed.md", _DOC)],
            )
            # Initial validate + index.
            self.assertEqual(run_script("validate.py", root=kb).returncode, 0)
            self.assertEqual(run_script("generate-indices.py", root=kb).returncode, 0)
            hwfeat_before = (kb / "queries" / "by-hardware-feature.md").read_text(encoding="utf-8")
            self.assertIn("hw-seed.md", hwfeat_before)

            # Incremental step: append one new source + one new wiki page.
            (kb / "sources" / "docs" / "doc-seed-2.md").write_text(_DOC_NEW, encoding="utf-8")
            (kb / "wiki" / "hardware" / "hw-seed-2.md").write_text(_HW_NEW, encoding="utf-8")

            # Uses only core tooling (no ingestion/freshness pipeline).
            self.assertEqual(run_script("validate.py", root=kb).returncode, 0)
            self.assertEqual(run_script("generate-indices.py", root=kb).returncode, 0)

            hwfeat_after = (kb / "queries" / "by-hardware-feature.md").read_text(encoding="utf-8")
            # Prior page still present (provenance preserved)...
            self.assertIn("hw-seed.md", hwfeat_after)
            # ...and the new page now appears.
            self.assertIn("hw-seed-2.md", hwfeat_after)

            # Original source page content is byte-for-byte untouched.
            self.assertEqual((kb / "sources" / "docs" / "doc-seed.md").read_text(encoding="utf-8"), _DOC)

            # Second regeneration is a no-op (deterministic).
            self.assertEqual(run_script("generate-indices.py", root=kb).returncode, 0)
            hwfeat_again = (kb / "queries" / "by-hardware-feature.md").read_text(encoding="utf-8")
            self.assertEqual(hwfeat_after, hwfeat_again)

    def test_dropped_id_surfaces_as_broken_link(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(
                Path(d),
                wiki_pages=[("wiki/hardware/hw-seed.md", _HW)],
                source_pages=[("sources/docs/doc-seed.md", _DOC)],
            )
            self.assertEqual(run_script("validate.py", root=kb).returncode, 0)
            # Remove the cited source: the dangling reference must be reported.
            (kb / "sources" / "docs" / "doc-seed.md").unlink()
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 1)
            self.assertIn("unknown id", v.stdout)


class AllScriptsValidationGateTests(unittest.TestCase):
    """A schema/vocabulary-invalid page must cause ALL FIVE entry-point scripts
    to fail/report, not just validate.py."""

    def _kb_with_invalid_page(self, tmp):
        kb = make_kb(Path(tmp))
        # Provide a resolvable source so only the intended invalid page is at fault.
        shutil.copy(REPO / "sources" / "docs" / "ptx-isa.md",
                    kb / "sources" / "docs" / "ptx-isa.md")
        # bad-arch.md targets sm90 (out of scope) — validate.py rejects it.
        shutil.copy(FIXTURES / "bad-arch.md", kb / "wiki" / "hardware" / "bad-arch.md")
        return kb

    def test_all_five_scripts_fail_on_invalid_kb(self):
        with tempfile.TemporaryDirectory() as d:
            kb = self._kb_with_invalid_page(d)
            # 1. validate.py
            self.assertEqual(run_script("validate.py", root=kb).returncode, 1)
            # 2. generate-indices.py: must fail AND not write the invalid page.
            g = run_script("generate-indices.py", root=kb)
            self.assertEqual(g.returncode, 1, g.stdout + g.stderr)
            idx = kb / "queries" / "by-hardware-feature.md"
            if idx.is_file():
                self.assertNotIn("Out Of Scope", idx.read_text(encoding="utf-8"),
                                 "invalid page leaked into a generated index")
            # 3. query.py: must fail instead of returning the invalid page.
            q = run_script("query.py", "--architecture", "sm90", root=kb)
            self.assertEqual(q.returncode, 1, q.stdout + q.stderr)
            self.assertNotIn("Out Of Scope", q.stdout)
            # 4. get_page.py: must fail instead of printing the invalid page.
            gp = run_script("get_page.py", "hw-fixture-bad-arch", root=kb)
            self.assertEqual(gp.returncode, 1, gp.stdout + gp.stderr)
            self.assertNotIn("Out Of Scope", gp.stdout)
            # 5. grep_wiki.py: must fail instead of matching the invalid page.
            gr = run_script("grep_wiki.py", "sm90", root=kb)
            self.assertEqual(gr.returncode, 1, gr.stdout + gr.stderr)
            self.assertNotIn("bad-arch.md", gr.stdout)

    def test_gate_message_points_to_validate(self):
        with tempfile.TemporaryDirectory() as d:
            kb = self._kb_with_invalid_page(d)
            for name in ("generate-indices.py", "query.py", "get_page.py", "grep_wiki.py"):
                r = run_script(name, "x", root=kb) if name in ("query.py", "get_page.py", "grep_wiki.py") \
                    else run_script(name, root=kb)
                self.assertIn("failed validation", r.stderr,
                              f"{name} did not emit the shared gate message")


class SchemaConstraintTests(unittest.TestCase):
    def test_bad_pr_status_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            (kb / "sources" / "prs").mkdir(parents=True, exist_ok=True)
            shutil.copy(FIXTURES / "bad-status.md", kb / "sources" / "prs" / "bad-status.md")
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 1)
            self.assertIn("status 'banana'", v.stdout)

    def test_version_claims_constraints_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            kb = make_kb(Path(d))
            # Malformed claim: wrong id prefix, bad tool, empty applies_to/source_ids.
            (kb / "data" / "version-claims.yaml").write_text(
                "claims:\n"
                "  - id: bad-claim\n"
                "    tool: notatool\n"
                "    claim_valid_for: x\n"
                "    last_verified_release: y\n"
                "    last_verified_at: '2026-06-30'\n"
                "    applies_to: []\n"
                "    source_ids: []\n",
                encoding="utf-8",
            )
            v = run_script("validate.py", root=kb)
            self.assertEqual(v.returncode, 1)
            out = v.stdout
            self.assertIn("vs-", out)          # id_prefix violation reported
            self.assertIn("notatool", out)     # tool enum violation reported


if __name__ == "__main__":
    unittest.main(verbosity=2)
