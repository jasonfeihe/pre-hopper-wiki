# Candidate Ledgers

Per-repo YAML ledgers recording include / exclude / defer / needs-review
decisions for surfaced PR candidates, one file per tracked repository. They are
the decision-audit trail for the pre-Hopper content-ingestion pipeline.

- Schema: `candidate-ledger` in `data/schemas.yaml`; validated by
  `scripts/validate.py`.
- Tracked repos: see `candidates/tracked-repos.txt` (a ledger whose `repo` is
  not in that allowlist is flagged `out-of-scope-repo`).
- `include` rows must carry `architecture_evidence`. Summary counts must equal
  the actual per-decision tallies.
- New candidates discovered by `scripts/refresh_candidate_ledger.py` land as
  `decision: defer` (needs triage); existing decisions are never rewritten.
- Reviewer rationale and coverage gaps for the hand-triaged ledgers are in
  `docs/seed-candidates.md`.
