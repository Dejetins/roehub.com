# Market Data — append certificate and verified-prefix preservation v1

Status: implemented and locally verified by S10-r1 on 2026-10-14.
This additive refinement applies to frozen artifact publication and verified-prefix
preservation in the [PG adapter contract](market-data-real-adapters-postgresql-parquet-v1.md).
The original S01 contract remains byte-bound by predecessor receipts; this document
owns the subsequent 0040 refinement. Other D1–D7 decisions remain unchanged.

## Normative scope and precedence

For the implemented 0040 runtime, this addendum takes precedence in these exact
parts of the original contract:

- **D4 — exact coverage, verified frontier and bounded backfill:** refines the
  invalidation rule for writes outside the verified prefix. Strict tail appends
  preserve it; material corrections and late actual inside accepted absence
  invalidate it. Deletion, restore, reset, policy change and index rebuild retain
  their original invalidation obligations.
- **S01 reversible implementation decisions**, paragraphs beginning
  “Materialization revision compares” and “Summary revision is conservative”:
  the global revision and consistent observation remain unchanged. For an already
  frozen, bounded artifact only, a newer global revision does not force rejection
  when the append certificate below proves its inputs unchanged. All other
  revision consumers retain their original contract.

This is the current rule for those cases, not an optional alternative. It does not
authorize an operator to advance verification or suppress a correction marker.

## Current rules

Migration `0040` adds a conservative append certificate without changing that global
revision or its cache/observation consumers. Under the same generation row lock and
transaction as canonical/coverage writes, `last_correction_revision` records the
latest revision affecting the prior prefix (including gap fills/backward extension),
and `append_safe_revision` attests the current revision. Unknown/non-ready prior
coverage and an unattested older-writer revision also advance the correction marker.
An artifact with a fixed range ending no later than the captured last minute plus
one minute may publish after strict tail appends only when the attestation is current
and its captured revision is at least the correction marker. Generation, funding and
ready-index fences remain mandatory; generic future-range exports and legacy journal
identities retain exact-revision fencing. Migration initializes both markers to the
current revision, conservatively rejecting earlier snapshots. No counter is reset by
an operator to admit a build. Roll out via normal bootstrap and fresh writer sessions.
Strict tail appends preserve an existing verified prefix only when every changed
minute is outside `[origin, verified_until)`. This also checks the certificate's end:
accepted absence can extend past the last actual candle, and a late actual within
that prefix invalidates it. Preserving a certificate never advances its frontier;
only the existing bounded, lease/generation-fenced verifier can do that. Corrections
and unknown prior indexes conservatively invalidate the certificate.

Evidence and operational limits: [S10-r1 report](../../../.codex/agents/generated/postgresql-only-storage-v1/reports/S10-r1.md).
