#!/usr/bin/env python3
"""Validate the pre-Hopper kernel wiki against its schema and controlled vocabulary.

Checks performed:
  1. Required data files exist (data/schemas.yaml, data/tags.yaml,
     data/version-claims.yaml) — missing files are a hard error, not a silent pass.
  2. Every sources/*.md and wiki/*.md page has parseable YAML frontmatter.
  3. Each page maps to a known page type; its required fields are present and it
     carries no unknown fields.
  4. Controlled-vocabulary fields (architectures, tags, hardware_features,
     techniques, kernel_types, languages, confidence, reproducibility,
     source_category) only use values enumerated in data/tags.yaml. In
     particular, out-of-scope architectures (anything other than the enumerated
     sm75/sm86/sm89) are rejected.
  5. Page ids are unique and carry the id-prefix their page type requires.
  6. Link integrity: every id referenced in `sources:` and `related:` resolves
     to an existing page id.
  7. Per-type constraints (id_prefix, type literal, allowed source_category,
     reproducibility_minimum, merge_sha-when-merged, and any other enum
     constraint declared in data/schemas.yaml such as source-pr.status).
  8. wiki/*.md pages live under a recognized type subdirectory.
  9. data/version-claims.yaml (including claim id_prefix, tool enum, and
     applies_to/source_ids minimums) and data/refresh-cutoff.yaml conform to
     their registry schemas.

There is deliberately NO "Blackwell-first" / `blackwell_relevance` rule: this
knowledge base is pre-Hopper-scoped, and no page type requires a relevance
field. A neutral, optional `scope_relevance` field is accepted where the schema
lists it.

`validate_root(root) -> list[str]` exposes these checks as an importable gate so
the other entry-point scripts (generate-indices, query, get_page, grep_wiki)
refuse to operate on an invalid knowledge base rather than producing false-pass
results.

Exit code 0 on success, 1 on any validation error.

Usage:
    validate.py            # validate the whole knowledge base
    validate.py --root DIR # validate a specific root (overrides autodetection)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT as _DEFAULT_ROOT  # noqa: E402

# Controlled-vocabulary frontmatter fields whose list/scalar values must be
# enumerated in data/tags.yaml. Maps the frontmatter key -> the tags.yaml key.
LIST_VOCAB_FIELDS = {
    "architectures": "architectures",
    "tags": None,  # tags are validated against the union of several vocab sets
    "hardware_features": "hardware_features",
    "techniques": "techniques",
    "kernel_types": "kernel_types",
    "languages": "languages",
}
SCALAR_VOCAB_FIELDS = {
    "confidence": "confidence",
    "reproducibility": "reproducibility",
    "source_category": "source_categories",
}

# The reproducibility ladder, ordered weakest -> strongest, for enforcing
# `reproducibility_minimum` constraints.
REPRO_LADDER = ["concept", "pseudocode", "snippet", "runnable", "benchmarked"]

# Constraint keys handled by dedicated logic (not by the generic enum pass).
SPECIAL_CONSTRAINT_KEYS = {
    "id_prefix", "type", "reproducibility_minimum", "merge_sha_required_when",
    "source_category",
}

# Recognized wiki subdirectories (a wiki/*.md page must live under one of these).
WIKI_TYPE_DIRS = {
    "hardware", "techniques", "kernels", "patterns", "languages", "migration",
}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def split_frontmatter(content: str):
    """Return (frontmatter_dict_or_None, parse_ok). parse_ok is False when the
    document has no frontmatter block or the block is not a YAML mapping."""
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", content, re.DOTALL)
    if not m:
        return None, False
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, False
    if not isinstance(fm, dict):
        return None, False
    return fm, True


def detect_page_type(fm: dict, rel_path: str):
    """Return the schema key for a page, or None if it cannot be determined.

    wiki pages declare their kind via the `type` field (-> wiki-<type>).
    sources pages are typed by their subdirectory (sources/docs -> source-doc).
    """
    parts = rel_path.split("/")
    if parts[0] == "wiki":
        t = fm.get("type")
        if isinstance(t, str):
            return f"wiki-{t}"
        return None
    if parts[0] == "sources" and len(parts) > 1:
        # sources/docs -> source-doc, sources/prs -> source-pr, etc.
        return f"source-{parts[1].rstrip('s')}"
    return None


class Validator:
    def __init__(self, root: Path):
        self.root = root
        self.errors: list[str] = []
        self.schemas = {}
        self.tags = {}
        self.pages = []          # list of (rel_path, fm)
        self.ids = {}            # id -> rel_path
        self._vocab_cache = {}

    def err(self, msg: str):
        self.errors.append(msg)

    # ---- loading -------------------------------------------------------

    def load_required_data(self) -> bool:
        ok = True
        schemas_path = self.root / "data" / "schemas.yaml"
        tags_path = self.root / "data" / "tags.yaml"
        vclaims_path = self.root / "data" / "version-claims.yaml"

        if not schemas_path.is_file():
            self.err("data/schemas.yaml: missing (required)")
            ok = False
        if not tags_path.is_file():
            self.err("data/tags.yaml: missing (required)")
            ok = False
        if not vclaims_path.is_file():
            self.err("data/version-claims.yaml: missing (required registry stub)")
            ok = False
        if not ok:
            return False

        try:
            self.schemas = load_yaml(schemas_path) or {}
        except yaml.YAMLError as e:
            self.err(f"data/schemas.yaml: unparseable ({e})")
            return False
        try:
            self.tags = load_yaml(tags_path) or {}
        except yaml.YAMLError as e:
            self.err(f"data/tags.yaml: unparseable ({e})")
            return False
        return True

    def vocab(self, tags_key: str) -> set:
        if tags_key not in self._vocab_cache:
            self._vocab_cache[tags_key] = set(self.tags.get(tags_key, []) or [])
        return self._vocab_cache[tags_key]

    def all_tag_vocab(self) -> set:
        """Union of every vocabulary set a free `tags` value may legitimately
        draw from."""
        out = set()
        for key in ("hardware_features", "techniques", "kernel_types",
                    "languages", "architectures"):
            out |= self.vocab(key)
        return out

    def collect_pages(self):
        for subdir in ("sources", "wiki"):
            base = self.root / subdir
            if not base.exists():
                continue
            for md in sorted(base.rglob("*.md")):
                rel = md.relative_to(self.root).as_posix()
                content = md.read_text(encoding="utf-8")
                fm, ok = split_frontmatter(content)
                if not ok:
                    self.err(f"{rel}: missing or unparseable YAML frontmatter")
                    continue
                self.pages.append((rel, fm))

    # ---- per-page validation ------------------------------------------

    def validate_page(self, rel: str, fm: dict):
        parts = rel.split("/")

        # wiki pages must live under a recognized type subdirectory.
        if parts[0] == "wiki":
            if len(parts) < 3 or parts[1] not in WIKI_TYPE_DIRS:
                self.err(
                    f"{rel}: wiki page is not under a recognized type "
                    f"subdirectory ({sorted(WIKI_TYPE_DIRS)})"
                )
                return

        ptype = detect_page_type(fm, rel)
        if ptype is None or ptype not in self.schemas:
            self.err(f"{rel}: cannot determine a known page type (got {ptype!r})")
            return

        schema = self.schemas[ptype]
        required = set(schema.get("required", []))
        optional = set(schema.get("optional", []))
        allowed = required | optional
        constraints = schema.get("constraints", {}) or {}

        # Required fields present. A required field must have its key present
        # and a non-null/non-empty-string value; empty LISTS are allowed for
        # most fields (e.g. `related: []`). The `sources` field is special: a
        # synthesized wiki page must cite at least one source, so an empty
        # sources list is rejected — mirroring the reference KB's
        # "every synthesized page is sourced" contract.
        for field in sorted(required):
            present = field in fm and fm[field] not in (None, "", {})
            if not present:
                self.err(f"{rel}: missing required field '{field}' for {ptype}")
            elif field == "sources" and ptype.startswith("wiki-") and fm[field] == []:
                self.err(f"{rel}: wiki page must cite at least one source (sources is empty)")

        # No unknown fields (version_sensitive pointer is always allowed).
        for field in fm:
            if field not in allowed and field != "version_sensitive":
                self.err(f"{rel}: unknown field '{field}' for {ptype}")

        # id prefix + uniqueness.
        pid = fm.get("id")
        if isinstance(pid, str):
            prefix = constraints.get("id_prefix")
            if prefix and not pid.startswith(prefix):
                self.err(f"{rel}: id '{pid}' must start with '{prefix}' for {ptype}")
            if pid in self.ids:
                self.err(f"{rel}: duplicate id '{pid}' (also in {self.ids[pid]})")
            else:
                self.ids[pid] = rel

        # type literal.
        if "type" in constraints and fm.get("type") != constraints["type"]:
            self.err(
                f"{rel}: type '{fm.get('type')}' does not match required "
                f"'{constraints['type']}' for {ptype}"
            )

        # source_category constraint.
        if "source_category" in constraints:
            allowed_cats = constraints["source_category"]
            if isinstance(allowed_cats, str):
                allowed_cats = [allowed_cats]
            if fm.get("source_category") not in allowed_cats:
                self.err(
                    f"{rel}: source_category '{fm.get('source_category')}' not in "
                    f"{allowed_cats} for {ptype}"
                )

        # merge_sha required when merged.
        if constraints.get("merge_sha_required_when") == "status == merged":
            if fm.get("status") == "merged" and not fm.get("merge_sha"):
                self.err(f"{rel}: status is 'merged' but merge_sha is missing")

        # reproducibility minimum.
        rmin = constraints.get("reproducibility_minimum")
        if rmin and fm.get("reproducibility") in REPRO_LADDER:
            if REPRO_LADDER.index(fm["reproducibility"]) < REPRO_LADDER.index(rmin):
                self.err(
                    f"{rel}: reproducibility '{fm['reproducibility']}' is below the "
                    f"minimum '{rmin}' for {ptype}"
                )

        # Controlled-vocabulary list fields.
        for field, tags_key in LIST_VOCAB_FIELDS.items():
            if field not in fm:
                continue
            values = fm[field]
            if not isinstance(values, list):
                self.err(f"{rel}: field '{field}' must be a list")
                continue
            valid = self.all_tag_vocab() if tags_key is None else self.vocab(tags_key)
            for v in values:
                if v not in valid:
                    if field == "architectures":
                        self.err(
                            f"{rel}: architecture '{v}' is out of scope / not in "
                            f"data/tags.yaml architectures {sorted(self.vocab('architectures'))}"
                        )
                    else:
                        self.err(
                            f"{rel}: {field} value '{v}' is not in the controlled "
                            f"vocabulary (data/tags.yaml)"
                        )

        # from_arch / to_arch (migration pages) must be enumerated architectures.
        for field in ("from_arch", "to_arch"):
            if field in fm and fm[field] not in self.vocab("architectures"):
                self.err(
                    f"{rel}: {field} '{fm[field]}' is out of scope / not in "
                    f"data/tags.yaml architectures"
                )

        # Scalar vocabulary fields.
        for field, tags_key in SCALAR_VOCAB_FIELDS.items():
            if field in fm and fm[field] is not None:
                if fm[field] not in self.vocab(tags_key):
                    self.err(
                        f"{rel}: {field} '{fm[field]}' is not in the controlled "
                        f"vocabulary (data/tags.yaml {tags_key})"
                    )

        # Generic enum constraints: any constraint key that names a real
        # frontmatter field and whose value is a list of allowed values is
        # enforced here (e.g. source-pr.status: [open, merged, closed]). The
        # specially-handled keys above are skipped so we don't double-report.
        for ckey, allowed in constraints.items():
            if ckey in SPECIAL_CONSTRAINT_KEYS or not isinstance(allowed, list):
                continue
            if ckey not in fm or fm[ckey] is None:
                continue
            values = fm[ckey] if isinstance(fm[ckey], list) else [fm[ckey]]
            for v in values:
                if v not in allowed:
                    self.err(
                        f"{rel}: {ckey} '{v}' is not one of {allowed} for {ptype}"
                    )

    def validate_links(self):
        """sources: and related: ids must resolve to a known page id."""
        for rel, fm in self.pages:
            for field in ("sources", "related"):
                refs = fm.get(field)
                if not refs:
                    continue
                if not isinstance(refs, list):
                    self.err(f"{rel}: field '{field}' must be a list of ids")
                    continue
                for ref in refs:
                    if ref not in self.ids:
                        self.err(
                            f"{rel}: {field} references unknown id '{ref}'"
                        )

    # ---- data registries ----------------------------------------------

    def validate_version_claims(self):
        path = self.root / "data" / "version-claims.yaml"
        try:
            data = load_yaml(path)
        except yaml.YAMLError as e:
            self.err(f"data/version-claims.yaml: unparseable ({e})")
            return
        if not isinstance(data, dict) or "claims" not in data:
            self.err("data/version-claims.yaml: missing required key 'claims'")
            return
        claims = data.get("claims")
        if claims is None:
            return  # `claims:` with no value is an acceptable empty stub
        if not isinstance(claims, list):
            self.err("data/version-claims.yaml: 'claims' must be a list")
            return
        schema = self.schemas.get("version-claims-registry", {})
        claim_schema = schema.get("claim_schema", {})
        req = set(claim_schema.get("required", []))
        cc = claim_schema.get("constraints", {}) or {}
        tool_enum = cc.get("tool")
        id_prefix = cc.get("id_prefix")
        applies_min = cc.get("applies_to_required_min")
        source_min = cc.get("source_ids_required_min")
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                self.err(f"data/version-claims.yaml: claim #{i} is not a mapping")
                continue
            label = claim.get("id", i)
            for field in sorted(req):
                if field not in claim:
                    self.err(
                        f"data/version-claims.yaml: claim "
                        f"'{label}' missing required field '{field}'"
                    )
            if id_prefix and isinstance(claim.get("id"), str) \
                    and not claim["id"].startswith(id_prefix):
                self.err(
                    f"data/version-claims.yaml: claim id '{claim['id']}' must "
                    f"start with '{id_prefix}'"
                )
            if tool_enum and "tool" in claim and claim["tool"] not in tool_enum:
                self.err(
                    f"data/version-claims.yaml: claim '{label}' tool "
                    f"'{claim['tool']}' is not one of {tool_enum}"
                )
            if applies_min and len(claim.get("applies_to", []) or []) < applies_min:
                self.err(
                    f"data/version-claims.yaml: claim '{label}' needs at least "
                    f"{applies_min} applies_to entr(y/ies)"
                )
            if source_min and len(claim.get("source_ids", []) or []) < source_min:
                self.err(
                    f"data/version-claims.yaml: claim '{label}' needs at least "
                    f"{source_min} source_ids entr(y/ies)"
                )

    def validate_refresh_cutoff(self):
        path = self.root / "data" / "refresh-cutoff.yaml"
        if not path.is_file():
            return  # optional file; the scaffold creates it but it is not required here
        try:
            data = load_yaml(path)
        except yaml.YAMLError as e:
            self.err(f"data/refresh-cutoff.yaml: unparseable ({e})")
            return
        if not isinstance(data, dict) or "cutoff_date" not in data:
            self.err("data/refresh-cutoff.yaml: missing required key 'cutoff_date'")
            return
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(data["cutoff_date"])):
            self.err(
                f"data/refresh-cutoff.yaml: cutoff_date "
                f"'{data['cutoff_date']}' is not an ISO date (YYYY-MM-DD)"
            )

    # ---- ingestion: policy, ledgers, skip audit, refresh results -------

    def _load_optional_yaml(self, rel: str):
        """Load an optional data file. Returns (data, present). Reports a parse
        error and returns (None, True) when the file exists but is unparseable."""
        path = self.root / rel
        if not path.is_file():
            return None, False
        try:
            return load_yaml(path), True
        except yaml.YAMLError as e:
            self.err(f"{rel}: unparseable ({e})")
            return None, True

    def skip_reason_taxonomy(self) -> set:
        """The closed set of allowed skip reasons, from inclusion-policy.yaml.
        Empty set when the policy is absent (callers guard on that)."""
        data, present = self._load_optional_yaml("data/inclusion-policy.yaml")
        if not present or not isinstance(data, dict):
            return set()
        reasons = data.get("skip_reasons")
        return set(reasons.keys()) if isinstance(reasons, dict) else set()

    def validate_inclusion_policy(self):
        data, present = self._load_optional_yaml("data/inclusion-policy.yaml")
        if not present:
            return  # optional file; absent is fine (the pipeline simply isn't seeded yet)
        if not isinstance(data, dict):
            self.err("data/inclusion-policy.yaml: top level must be a mapping")
            return
        schema = self.schemas.get("inclusion-policy", {})
        for field in schema.get("required", []):
            if field not in data or data[field] in (None, "", [], {}):
                self.err(f"data/inclusion-policy.yaml: missing required key '{field}'")

        tiers = data.get("evidence_tiers")
        if not isinstance(tiers, list) or not tiers:
            self.err("data/inclusion-policy.yaml: 'evidence_tiers' must be a non-empty list")
            tiers = []
        tier_schema = schema.get("evidence_tier_schema", {})
        tier_req = tier_schema.get("required", [])
        allowed_strength = (tier_schema.get("constraints", {}) or {}).get("strength")
        arch_vocab = self.vocab("architectures")
        for i, tier in enumerate(tiers):
            if not isinstance(tier, dict):
                self.err(f"data/inclusion-policy.yaml: evidence_tier #{i} is not a mapping")
                continue
            label = tier.get("id", i)
            for field in tier_req:
                if field not in tier:
                    self.err(f"data/inclusion-policy.yaml: tier '{label}' missing '{field}'")
            if allowed_strength and tier.get("strength") not in allowed_strength:
                self.err(
                    f"data/inclusion-policy.yaml: tier '{label}' strength "
                    f"'{tier.get('strength')}' not in {allowed_strength}"
                )
            # Every architecture a tier maps to must be in scope.
            maps_to = tier.get("maps_to", {})
            if isinstance(maps_to, dict):
                for token, arch in maps_to.items():
                    if arch not in arch_vocab:
                        self.err(
                            f"data/inclusion-policy.yaml: tier '{label}' maps "
                            f"'{token}' to out-of-scope architecture '{arch}'"
                        )

        reasons = data.get("skip_reasons")
        if not isinstance(reasons, dict) or not reasons:
            self.err("data/inclusion-policy.yaml: 'skip_reasons' must be a non-empty mapping")

    def validate_candidate_ledgers(self):
        base = self.root / "candidates"
        if not base.is_dir():
            return
        schema = self.schemas.get("candidate-ledger", {})
        required = schema.get("required", [])
        row_schema = schema.get("row_schema", {})
        row_req = row_schema.get("required", [])
        row_cc = row_schema.get("constraints", {}) or {}
        decision_enum = row_cc.get("decision")
        repo_vocab = self._tracked_repo_set()
        # Closed skip/exclude taxonomy: an `exclude` verdict in a ledger must cite
        # a reason from data/inclusion-policy.yaml::skip_reasons, same as the skip
        # audit (Codex R6). Empty taxonomy (policy absent) disables the check.
        taxonomy = self.skip_reason_taxonomy()
        for ledger in sorted(base.glob("*.yaml")):
            rel = ledger.relative_to(self.root).as_posix()
            try:
                data = load_yaml(ledger)
            except yaml.YAMLError as e:
                self.err(f"{rel}: unparseable ({e})")
                continue
            if not isinstance(data, dict):
                self.err(f"{rel}: top level must be a mapping")
                continue
            for field in required:
                if field not in data:
                    self.err(f"{rel}: missing required key '{field}'")
            if repo_vocab and data.get("repo") and data["repo"] not in repo_vocab:
                self.err(f"{rel}: repo '{data['repo']}' is not in the tracked repo set")
            rows = data.get("prs", [])
            if not isinstance(rows, list):
                self.err(f"{rel}: 'prs' must be a list")
                rows = []
            tally = {"include": 0, "exclude": 0, "defer": 0, "needs-review": 0}
            for row in rows:
                if not isinstance(row, dict):
                    self.err(f"{rel}: a prs row is not a mapping")
                    continue
                num = row.get("number", "?")
                for field in row_req:
                    if field not in row or row[field] in (None, ""):
                        self.err(f"{rel}: PR {num} missing required field '{field}'")
                dec = row.get("decision")
                if decision_enum and dec not in decision_enum:
                    self.err(f"{rel}: PR {num} decision '{dec}' not in {decision_enum}")
                elif dec in tally:
                    tally[dec] += 1
                if dec == "include" and not row.get("architecture_evidence"):
                    self.err(f"{rel}: PR {num} is 'include' but has no architecture_evidence")
                if dec == "exclude" and taxonomy:
                    reason = row.get("reason")
                    if reason not in taxonomy:
                        self.err(
                            f"{rel}: PR {num} exclude reason '{reason}' is not a key "
                            f"in data/inclusion-policy.yaml::skip_reasons"
                        )
            # Summary counts, when present, must match the real tallies.
            for key, count_field in (("include", "included"), ("exclude", "excluded"),
                                     ("defer", "deferred"), ("needs-review", "needs_review")):
                if count_field in data and data[count_field] != tally[key]:
                    self.err(
                        f"{rel}: {count_field}={data[count_field]} disagrees with "
                        f"actual {key} count {tally[key]}"
                    )
            if "total_candidates" in data and data["total_candidates"] != len(rows):
                self.err(
                    f"{rel}: total_candidates={data['total_candidates']} disagrees "
                    f"with actual row count {len(rows)}"
                )

    def _tracked_repo_set(self) -> set:
        """Repos the loop tracks, read from candidates/*.yaml repo fields plus
        refresh-search-results. Used to flag out-of-scope-repo ledgers. Returns
        the union of declared repos so a new ledger that declares its own repo
        is accepted, while a ledger pointing at an unrelated repo is flagged
        only when an explicit allowlist file is present."""
        allow = self.root / "candidates" / "tracked-repos.txt"
        if allow.is_file():
            return {ln.strip() for ln in allow.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")}
        return set()  # no allowlist -> do not constrain repo names

    def validate_skipped_audit(self):
        data, present = self._load_optional_yaml("data/pr-page-skipped.yaml")
        if not present:
            return
        if not isinstance(data, dict) or "rows" not in data:
            self.err("data/pr-page-skipped.yaml: missing required key 'rows'")
            return
        rows = data.get("rows") or []
        if not isinstance(rows, list):
            self.err("data/pr-page-skipped.yaml: 'rows' must be a list")
            return
        schema = self.schemas.get("pr-page-skipped-audit", {})
        row_schema = schema.get("row_schema", {})
        row_req = row_schema.get("required", [])
        stage_enum = (row_schema.get("constraints", {}) or {}).get("stage")
        taxonomy = self.skip_reason_taxonomy()
        for row in rows:
            if not isinstance(row, dict):
                self.err("data/pr-page-skipped.yaml: a row is not a mapping")
                continue
            label = row.get("pr_id", row.get("pr_number", "?"))
            for field in row_req:
                if field not in row or row[field] in (None, ""):
                    self.err(f"data/pr-page-skipped.yaml: row '{label}' missing '{field}'")
            if stage_enum and row.get("stage") not in stage_enum:
                self.err(
                    f"data/pr-page-skipped.yaml: row '{label}' stage "
                    f"'{row.get('stage')}' not in {stage_enum}"
                )
            reason = row.get("reason")
            if taxonomy and reason is not None and reason not in taxonomy:
                self.err(
                    f"data/pr-page-skipped.yaml: row '{label}' reason '{reason}' "
                    f"is not a key in data/inclusion-policy.yaml::skip_reasons"
                )

    def validate_refresh_subset(self):
        """Every PR number seen in a refresh (refresh-search-results.yaml) must
        appear as a row in the matching candidates/<repo_slug>.yaml ledger."""
        data, present = self._load_optional_yaml("data/refresh-search-results.yaml")
        if not present:
            return
        if not isinstance(data, dict):
            self.err("data/refresh-search-results.yaml: top level must be a mapping")
            return
        for field in self.schemas.get("refresh-search-results", {}).get("required", []):
            if field not in data:
                self.err(f"data/refresh-search-results.yaml: missing required key '{field}'")
        repos = data.get("repos") or []
        if not isinstance(repos, list):
            self.err("data/refresh-search-results.yaml: 'repos' must be a list")
            return
        repo_required = (self.schemas.get("refresh-search-results", {})
                         .get("repo_schema", {}).get("required", []))
        for entry in repos:
            if not isinstance(entry, dict):
                self.err("data/refresh-search-results.yaml: a repos entry is not a mapping")
                continue
            slug = entry.get("repo_slug", "?")
            # Enforce the per-repo schema's required fields (searched_at,
            # window_start, pr_numbers_seen, last_pr_date_seen, repo_slug).
            for field in repo_required:
                if field not in entry:
                    self.err(
                        f"data/refresh-search-results.yaml: repo '{slug}' "
                        f"missing required field '{field}'"
                    )
            seen = entry.get("pr_numbers_seen") or []
            if seen != sorted(seen):
                self.err(
                    f"data/refresh-search-results.yaml: repo '{slug}' "
                    f"pr_numbers_seen is not sorted ascending"
                )
            ledger_path = self.root / "candidates" / f"{slug}.yaml"
            if not ledger_path.is_file():
                if seen:
                    self.err(
                        f"data/refresh-search-results.yaml: repo '{slug}' has seen "
                        f"PRs but candidates/{slug}.yaml does not exist"
                    )
                continue
            try:
                ledger = load_yaml(ledger_path) or {}
            except yaml.YAMLError:
                continue  # ledger parse error already reported by validate_candidate_ledgers
            ledger_numbers = {r.get("number") for r in (ledger.get("prs") or [])
                              if isinstance(r, dict)}
            for n in seen:
                if n not in ledger_numbers:
                    self.err(
                        f"data/refresh-search-results.yaml: repo '{slug}' saw PR "
                        f"#{n} but it is not a row in candidates/{slug}.yaml"
                    )

    # ---- driver --------------------------------------------------------

    def collect_errors(self) -> list[str]:
        """Run every check and return the accumulated error strings, without
        printing or exiting. Used both by the CLI and by the shared
        validate_root() gate that the other scripts call."""
        if not self.load_required_data():
            return self.errors
        self.collect_pages()
        for rel, fm in self.pages:
            self.validate_page(rel, fm)
        self.validate_links()
        self.validate_version_claims()
        self.validate_refresh_cutoff()
        self.validate_inclusion_policy()
        self.validate_candidate_ledgers()
        self.validate_skipped_audit()
        self.validate_refresh_subset()
        return self.errors

    def run(self) -> int:
        self.collect_errors()
        return self._report()

    def _report(self) -> int:
        if self.errors:
            print(f"VALIDATION FAILED: {len(self.errors)} error(s)")
            for e in self.errors:
                print(f"  - {e}")
            return 1
        print(f"VALIDATION OK: {len(self.pages)} page(s) validated, 0 errors")
        return 0


def validate_root(root: Path) -> list[str]:
    """Validate a knowledge-base root and return the list of error strings.

    Returns an empty list when the knowledge base is valid (including a valid
    empty knowledge base). Never prints and never exits — callers decide how to
    report. This is the shared gate the non-validator scripts use before doing
    any indexing / search / fetch work, so that schema- or vocabulary-invalid
    knowledge-base state never produces a false-pass result.
    """
    return Validator(Path(root)).collect_errors()


def gate_or_exit(root: Path) -> None:
    """Shared entry-point guard for the non-validator scripts.

    Runs validate_root(); if the knowledge base has any validation errors, prints
    a concise failure message plus the errors to stderr and exits non-zero. A
    valid knowledge base (including a valid empty one) returns normally so the
    caller proceeds to its real work.
    """
    errors = validate_root(root)
    if errors:
        print(
            "ERROR: knowledge base failed validation; "
            "run `uv run python scripts/validate.py` to see details.",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate the pre-Hopper kernel wiki")
    parser.add_argument("--root", help="Knowledge-base root (default: autodetect)")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = _DEFAULT_ROOT

    sys.exit(Validator(root).run())


if __name__ == "__main__":
    main()
