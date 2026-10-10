# S05 independent production-risk review

2026-10-08. Reviewer: `/root/s05_review`. Skill: `production-risk-review`.
Independent review requested for changed scheduling, concurrency, ownership and
worker composition. No recursive review. Product, ledger and accepted artifacts
were not edited by this reviewer.

## Verdict

No unresolved material code findings in the reviewed final S05 implementation.
The local concurrency/worker proof below passed independently. This review does
not replace the executor's complete routed gates, migration checks, report,
receipt validation or lifecycle transition. It does not authorize activation.

## Scope and evidence basis

Reviewed the complete S05 prompt, relevant implementation-plan requirements and
S01/S03/S04 handoff contracts, current ledger status, S05 baseline and preserved
before-images. Focused on `artifact_cadence.py`, `dispatch_repository.py`,
`legacy_dispatch.py`, `managed_app.py`, `managed_worker.py`, `managed_recovery.py`,
`managed_access.py`, metrics, migration 0030 and its registration, management
repository changes, native progress observer and scheduler factory routing.
Existing S04 derivative planning/publication and S02/0028 ownership mechanisms
were inspected at crossed boundaries rather than treated as new implementations.

The final factory selects managed execution explicitly with
`ROEHUB_ARTIFACT_MANAGEMENT_INSTALLATION_ID`, requires configured storage and
disabled legacy scheduling, and supplies actual PostgreSQL/ClickHouse,
catalog/access, funding, native publisher and host-lock adapters. Capability and
recorded handoff remain separate gates. The composed cycle was exercised on
disposable inputs; no installed scheduler was started.

## Findings resolved during review

- **High: claim/lifetime race.** Lifetime ownership originally began after the
  durable claim. It now exists before claim/retry visibility and is keyed by
  attempt owner token; reconciliation cannot seize a live process by age.
  Missing/corrupt/unknown publication evidence retains quarantine. Exact committed
  receipt, pointer and validated manifest can repair a completion outage.
- **High: incompatible or unauthorized demand poisoned shared work.** Claim
  construction now checks each request, isolates unavailable catalog/authorization
  outcomes and groups by frozen compatibility. The bounded scan uses the complete
  configured queue bound before selecting the compatible group; oversized groups
  reject explicitly rather than silently publish a partial union.
- **High: derivative failure could erase mandatory-base success.** Insufficient
  derivative history and invalid derivative selections retain their own outcomes;
  eligible base/other derivative work can complete. Waiting requests only reenter
  when their identified source changes. Successful policy timestamps are limited
  to ready requests.
- **High: queue and retry starvation.** Durable keyset schedule traversal avoids
  hiding faster tenant overrides behind the internal cadence. Per-lane capacity
  refusal preserves other admissions. Round-robin request lanes, retry alternation
  and fenced rejection of failed retry preflight avoid permanently poisoned heads.
  Filesystem permission failures before claim become durable unavailable reasons.
- **High: coalescing could underreserve storage.** Effective combined history is
  used to recompute base storage admission before source freeze, followed by full
  derivative/storage/budget checks before physical publication. Original request
  targets remain inspectable when one latest physical target satisfies them.
- **Medium: progress and executable integration gaps.** Durable series track
  phase/timeframe/unit independently within an epoch, including the one-minute
  base phase. Actual native counters feed observations; retry creates a new epoch.
  Production factory routing, trusted access and explicit legacy demand conversion
  now connect the managed worker to the existing numerical publisher.

No scope-creep finding. Additive migration registration, actor/inherited-schedule
fields, progress persistence and legacy exclusion are required crossed boundaries.
Foreign files were assessed against the S05 baseline, not against HEAD alone.

## Independently executed checks

- Final command with `ROEHUB_BACKTEST_TEST_DSN` set only to the executor-created,
  trust-auth disposable container `roehub-s05-postgres` on loopback port `49912`:
  `.venv/bin/python -m pytest tests/real_infra/backtest/test_artifact_dispatch.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py tests/unit/apps/scheduler/test_backtest_artifact_publisher_metrics.py -q`
  — **33 passed in 12.98s**, exit 0. Test schemas were fixture-owned and dropped.
  This covers 21 dispatch cases plus scheduler/calendar and metrics regression.
- Independently reran `.venv/bin/ruff check` over every path in
  `S05-python-paths.json`: **All checks passed**, exit 0.
- Inspected executor `S05-types.json`: pyright **0 errors, 0 warnings**, exit 0.
  This is inspected executor evidence, not an independent type-check execution.
- Earlier independent DST smoke checked 1/3/7 calendar-day schedules through the
  New York spring gap and fall fold: first valid civil minute / first fold, with
  subsequent local-time anchors retained. The final pytest run also includes the
  calendar presets/anchor test.
- Earlier intermediate runs (8 scheduler, 10/12/18 evolving dispatch tests and
  the separate real process-death case) passed. An intermediate combined run had
  two failed test expectations after error-handling changes; these were disclosed,
  corrected and superseded by the final 33-pass run. A transient saved lint
  failure from a long test line was also superseded by the independent clean run.

The final tests exercise native selective union, per-request outcomes, committed
receipt recovery, real process death with unknown pointer quarantine, lifetime
exclusion, concurrent claims, retry/new epoch, source waits, pause, effective
storage budget, permissions, missed-target coalescing, overrides, owner-only
legacy handoff, and actual composed cycles. They use fixed/disposable data.

## Compatibility and limits

Additive persistence and optional inherited schedules are `compatible-change`
for upgraded consumers; migrate before using the changed repository. Explicit
managed/legacy ownership selection is an intentional operational change. Native
numerical engines, immutable Backtest recipes and physical 0028 reservation/pin
semantics remain unchanged (`none`). Existing S04 forward-reader restrictions
still apply; this review does not relax them.

Proof is local, bounded and deterministic/disposable. It is not a week-long run,
live provider verification, production deployment, full-market capacity proof or
browser/API acceptance. Unknown publication ownership intentionally continues
to consume capacity until safe reconciliation; no timeout reclaim was introduced.
The executor must retain those limits, final gate results and the exact S06
handoff in `S05.md` before accepting S05.
