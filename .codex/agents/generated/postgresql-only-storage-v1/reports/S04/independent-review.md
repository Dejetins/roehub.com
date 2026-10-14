# Independent production-risk review

Reviewer: independent subagent `/root/s04_review`, final review 2026-10-14.
Scope: changed S04 reader/verifier/maintenance/reconciliation code, tests and local
PG16 query evidence. No recursive review. Reviewer did not independently rerun tests.

Findings corrected and re-reviewed:

1. P1: acquiring a verifier lease for a genuinely new empty instrument failed.
   Fresh-only generation/summary bootstrap added; unknown/rebuilding not promoted.
2. P1: deletion-to-empty observation was omitted and left stale source publication.
   Explicit empty/unknown tombstones and persisted-source real-PG tests added.
3. P2: maintenance role could not execute the new bounded maintenance operations.
   Narrow grants added and exercised with an unprivileged maintenance role.
4. P2: delayed observations could overwrite newer readiness/empty state.
   Current generation/revision/readiness fence and monotonic observed_at guards
   added; delayed empty and delayed unknown cases tested.
5. P2: Q13 SQL plans were omitted for psycopg Composed queries; representative
   parity assertions were too weak for several outputs. Capture now resolves
   Composed SQL; complete DTO/array values, observations, reference/search/readiness
   and funding records are checked. Final reviewed evidence: 59 plans, 25 passes.

Final reviewer conclusion: no open material P1/P2 findings. Lease renew/release
preserves generation→lease locking and token checks. Funding universe streams
batches of 4096 rather than imposing an arbitrary output cap.

The executor separately reruns final gates and query probes. This review does not
prove S09 mixed soak, an installation-wide physical connection cap, activation or
production delivery. Those boundaries remain explicit next-stage obligations.
