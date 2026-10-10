# S09 independent integrated production-risk review

Date: 2026-10-08. Reviewer: `s09_review`. Mode: independent read-only product
review using `production-risk-review`; no recursive reviewer. Only this new report
is reviewer-owned. Baseline HEAD: `718a1d8c2dcc28ccb2c0704de3b663785494e0c3`.
The worktree is extensively dirty; scope comes from the accepted plan and S01,
S05, S06, S08 reports/source manifests, not the entire HEAD diff.

## Findings (initial review; repairs not yet reviewed)

1. **P1, high confidence — retry bypasses request reauthorization.**
   `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/postgres/management_repository.py:723`
   and `apps/scheduler/backtest_artifact_publisher/wiring/managed_app.py` retry lane.
   Initial `claim_group` rechecks membership and coordinate access for each request.
   After a prepublication transient failure, the retry lane calls `retry` and then
   `worker.execute` directly. `retry` checks owner fence/settings/root/budget but
   does not recheck actor membership or catalog access. Revoking access while a
   request is `retry_wait` therefore still permits private derivative work to run
   and become ready. Reauthorize all attached requests at retry admission, preserve
   individual denied outcomes, and omit denied demand from the retry's effective
   build. Test membership and catalog revocation between attempts, including mixed
   shared requests where mandatory base/another authorized request can continue.

2. **P2, high confidence — storage budget edits do not govern live usage admission.**
   `apps/scheduler/backtest_artifact_publisher/wiring/managed_worker.py:346` and
   `src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/base_storage_admission.py`.
   Factory composition snapshots `budget_bytes` into immutable `BaseStorageAdmission`.
   Storage configure updates PostgreSQL without replacing that object. Disk inventory
   is checked against the startup budget; the later current-settings check compares
   only `reservation + 1 GiB`, omitting existing usage. Example: startup budget 10 GiB,
   usage 4 GiB, changed budget 3 GiB, reservation 0.1 GiB: both checks permit new work
   although usage already exceeds the current budget. Increasing budget also remains
   ineffective until restart. S01 explicitly requires shrinking below usage to block
   new admission. Read current trusted settings for usage-inclusive admission and
   handle the configure/admission race under the existing exclusion; test decreasing
   and increasing the budget with the same running worker instance.

3. **P2, high confidence — ordinary profile edits trigger immediate/retroactive work.**
   `apps/api/wiring/artifact_service.py:155` admits an immediate request for any
   precompute profile patch or repeated `mode=precompute`, even if precompute was
   already enabled. R08 reserves immediate admission for initial preparation and
   enabling precompute; later edits should coalesce to the selected cadence.
   Independently, `dispatch_repository.py:174–196` includes policy control version
   in the cadence fingerprint. Editing a policy invalidates the `next_due > now`
   fast path and creates another dedupe key for the already elapsed occurrence.
   Restrict immediate admission to the actual transition and preserve the next
   future occurrence on later edits. Test profile/mode/no-op edits after an occurrence
   has completed, then prove the new profile is selected at the next scheduled run.

All three findings were sent to the S09 executor immediately, before this report.
They are implementation defects inferred directly from complete control-flow
inspection, not claimed observed production incidents. No reproducer or test suite
was executed by this reviewer during this initial pass.

## Inspected boundaries and favorable observations

- Managed process lifetime lock starts before claim/retry becomes durable. Global
  PostgreSQL admission allows one running/quarantined physical build; unknown outcomes
  consume capacity. Recovery requires acquired lifetime, exact attempt/epoch/owner
  receipt, pointer generation/hash and canonical payload validation. No TTL takeover
  was found in these paths.
- Storage HTTP separately requires operations permission, installation ownership,
  recent auth and Origin for mutation. The control service independently checks owner
  and configured installation. Requests select trusted root IDs, not arbitrary paths.
- Cleanup commits an intent before IO, reacquires the existing physical writer
  reservation, verifies current/non-current identities and every reader state, and
  quarantines uncertain outcomes. Same-key replay returns unknown instead of deleting
  again. Descriptor-relative no-follow traversal protects the inspected deletion path.
- Managed source freeze compares persisted content identity on retry. Publication
  completion follows actual native publication and fsynced receipt; source readiness
  is separate from request readiness. Shared derivative progress hides foreign totals.
- Production factory selection is explicit through installation ID and disabled
  legacy scheduling; construction itself does not grant handoff or activation.

## Missing requirements, tests and operational limits

The outstanding findings affect R07/R08/R12/R13. No unrelated scope expansion was
identified in the inspected implementation. This is a focused trust/concurrency/
recovery review, not exhaustive numerical or frontend correctness certification.
S08 explicitly uses synthetic browser principals and queued jobs; those observations
cannot prove a completed Backtest worker. S09's executor owns real integrated native
worker, PostgreSQL/ClickHouse, crash/restart/pin/quota/cleanup, browser regression and
all affected gates. This report supplies no independent test-pass claim for them.

Reviewed operational documentation preserves disabled activation, reader-first
compatibility and explicit owner reconciliation of unknown cleanup. The final S09
runbook must explain live budget semantics after repair and use the current supported
runtime; historical retired runbooks do not authorize startup. Production/provider/
S3/full-market/week-long behavior remains unverified. No credentials, user databases,
installed scheduler, branch, worktree, commit or publication was touched.

## Disposition

Initial review: **changes required**. The executor should repair and regress the
three findings, then request this same reviewer to inspect the changed behavior and
evidence. This is follow-up of one independent review, not recursive review. No
readiness or S09 acceptance verdict is implied.

## Follow-up 1: source repairs and scoring gate diagnosis

The same reviewer inspected the executor's working-tree repairs in
`artifact_service.py`, `managed_worker.py`, `managed_app.py`,
`dispatch_repository.py`, `base_storage_admission.py` and the new
`tests/real_infra/backtest/test_artifact_integrated.py`. No product edits were made
by the reviewer.

- Retry now reauthorizes each attached running request inside its admission
  transaction, records denied outcomes and stops an entirely unauthorized group.
  Worker derivative planning filters the frozen request set by actual running
  request IDs, preserving authorized/base work in a mixed group. This addresses
  finding 1 at source level; final regression proof remains executor-owned.
- API immediate admission now requires an actual transition into precompute.
  Cadence fingerprint excludes unrelated policy version and schedule edits skip
  retroactive occurrences. The added profile-edit test checks no immediate request,
  no same-occurrence duplicate and selection of revision 2 at the following day.
  This addresses finding 3 at source level; final gate results remain pending.
- Live budget is now read under installation exclusion for every storage check,
  including retries. The final effective reservation check still releases that
  transaction before updating reservation in a second transaction. A configure
  shrink between those transactions can invalidate usage-inclusive admission;
  its later check again compares only reservation plus reserve. Keep the final
  usage-inclusive check and reservation update under one installation exclusion.
  This residual of finding 2 was sent immediately to the executor. The sparse-file
  reduction regression covers the sequential case, not this transaction gap.

The six failures in `S09-shard-backtest-scoring.json` all enter the same intentional
`cost_permutation.py:159` effective-thread mismatch guard. The parametrized
`test_preflight_to_actual_short_orchestration` constructs orchestration directly
with default 12-thread compute policy, bypassing production child setup.
Reviewer executed only this read-only diagnostic command:

```text
.venv/bin/python -c 'import numba as nb; from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy; print({"effective_numba_threads": nb.get_num_threads(), "default_policy_threads": BacktestComputePolicy().threads.num_threads})'
exit 0: {'effective_numba_threads': 18, 'default_policy_threads': 12}
```

Production `full_job_child` applies `backtest_numba_environ` before importing its
compute wiring; the unit test does not. The smallest scoped repair is an explicit
test `BacktestComputePolicy(threads=BacktestNumbaThreadDecision(nb.get_num_threads(),
"test"))`, as already used by cost-permutation tests. Preserve the runtime guard.
Run this module and then the scoring shard. This diagnosis does not attribute
the failure to a particular historical stage or claim any gate passed.

## Follow-up 2: repaired disposition

The final effective-reservation `check_storage` now executes inside the same outer
repository transaction as the reservation update. Its nested transaction retains
installation exclusion until that outer commit, closing the configure/admission
gap identified in follow-up 1. The executor's `S09-regressions-first.txt` records
24 passed in 10.03 seconds; the reviewer read this output, did not independently
rerun the suite, and does not promote it to broader integration proof.

**Current code-review disposition: no unresolved material findings from this
focused review.** All three original findings are repaired at the inspected source
boundary. Final integrated acceptance still requires the executor's complete
R01–R13 evidence and gates. In particular, `prepare_artifact_inputs` alone is not
evidence of a completed admitted Backtest worker.

For that missing evidence, the existing real-infrastructure pattern at
`tests/real_infra/backtest/test_artifact_input_lifecycle.py`,
`test_native_child_failure_drills_release_owned_files_and_pins`, supplies real
PostgreSQL create/queued reader pin/lease and `BacktestChildProcessExecutor` with
actual child computation. Reuse its success path against the integrated published
snapshot, select the default production child without failure hooks, commit actual
returned top variants through `finish_with_top_variants`, and inspect persisted
preparation provenance and source generation/hash. The smaller in-process native
orchestration pattern in `test_materialization_lifecycle.py` has a synthetic
acknowledgement callback and does not by itself establish that PG/child lifecycle.
These are implementation suggestions to the executor, not claims of executed proof.

## Final whole-implementation / local-handoff assessment

The reviewer inspected `complete_backtest` in the final integrated test and read
`S09-integrated-fifth.txt` (1 passed, 3 deselected). It now uses actual HTTP route/use
case admission and PostgreSQL recipe/reader pin, claims a lease, invokes the default
production `BacktestChildProcessExecutor` child, commits returned results and verifies
persisted `succeeded`, preparation provenance, original manifest hash, released reader
pin and cleaned attempt directory. The source is the same disposable ClickHouse/
managed publication exercised earlier in that test. This closes the earlier missing
worker-proof observation. Test identity/scope resolution remains synthetic; it is not
production session-issuance proof or an autonomous installed worker-daemon observation.

Final executor evidence inspected: `S09-real-infra-final.json` records **123 passed**
with 3 multiprocessing/fork deprecation warnings; `S09-pyright-final.json` records
**1,598 files, zero errors/warnings**; `S09-frontend.txt` records **323 passed**.
These are reviewed executor results, not separate reviewer reruns. Appended canonical
runtime documentation, installation runbook and Navigator README distinguish browser
fixture limits from actual completed worker evidence, preserve explicit activation,
reader-first migration and quarantine recovery, and describe repaired budget/cadence
semantics. No new material trust/concurrency/recovery finding arose from this pass.

### Residual failed tooling gates and materiality

Reviewer independently ran `.venv/bin/ruff check . --output-format concise`: exit 1,
exactly **25 diagnostics**, exclusively the two receipt-bound pack updater files:
23 E501, one I001 in `test_ledger_update.py`, one F841 at its line 249. F841 assigns
the return value of `self.repair_fixture()` without using that local; the fixture
call still executes and the test checks rejection and byte preservation. None of
these diagnostics identifies a missing execution step, unsafe concurrency branch or
runtime authorization defect. `S09-final-gates.json` includes 14 passing updater
protocol tests, including competing claims, directory lock, evidence/receipt binding
and replacement-state validation. `S09-tooling-attribution.json` binds unchanged
S09 baseline/current updater bytes. This lint failure is a disclosed ancillary local
residual; changing bound protocol bytes solely to silence formatting is unwarranted.

The failed runtime-input scan attempts UTF-8 decoding of unchanged binary
`apps/navigator-web/src/assets/exchanges/binance.ico`; baseline/current SHA-256 match
in tooling attribution. The scan fails closed instead of certifying incomplete
inventory. This is a release-inventory limitation, not evidence of an artifact worker
security/recovery defect. It must remain a failed gate.

The OSS policy mismatch includes unchanged exchange assets, pre-existing evidence,
and this pack's newly created screenshot PNGs/build outputs. The new pack assets
cannot be wholly attributed to foreign changes. Missing policy provenance does not
show a runtime authorization/concurrency defect; however, distribution rights and
release packaging are not established. Do not invent licenses or relabel the OSS
gate passed. Publication needs a separately authorized asset/provenance resolution.

**Final review verdict:** no unresolved material local implementation findings in
the assessed boundaries; the three earlier defects are repaired and functionally
regressed. Local handoff may use these results provided the final report/receipt
binds the complete required R01–R13 proof and final gate outcomes. **Whole-CI/release
is not green**: retain the Ruff, runtime-input inventory and OSS failures explicitly.
This assessment does not waive required functional proof, claim production readiness,
authorize publication, or turn pending generated-map verification into a pass.

## Receipt-preserving final type-gate correction

The preceding zero-error Pyright result is **historical, not the final type-gate
state**. Receipt audit found S08 directly binds `storage_fixture.py` and
`storage_preflight.py`; the executor restored their exact accepted bytes instead
of changing historical receipts. Reviewer compared both files with their S09
before-images: no differences. No type-checker suppression or policy weakening
was applied.

`S09-pyright-bound-fixtures.json` is the final inspected full Pyright result:
**1,598 files, 4 errors, 0 warnings**. Three errors concern statically unknown
`__wrapped__` members at `storage_fixture.py:41` (two calls) and
`storage_preflight.py:32`. Reviewer performed a read-only import of the underlying
test modules and verified `fixture.managed`, `catalog`, and `native.built` each
have a callable `__wrapped__` at runtime; no disposable DB or browser server was
started. These diagnostics do not identify a missing runtime callable.

The fourth error at `storage_fixture.py:310` is `str` versus psycopg `LiteralString`
for migration SQL read from the fixed repository path
`migrations/postgres/0031_backtest_artifact_storage_v1.sql`. This fixture does not
compose that SQL from browser input, identifiers, credentials or external text;
the reported type mismatch does not reveal an injection boundary. These files
are disposable browser proof support, not production service entrypoints.

**Materiality:** the four exact, receipt-bound fixture diagnostics are ancillary
local type debt, not an unresolved security/concurrency/recovery defect. Preserve
the failure and its ownership attribution; do not claim full Pyright passes. The
123 real-infrastructure and 323 frontend results remain recorded executions, not
an assertion that every historical fixture was rerun after byte restoration.
Restored fixture runtime semantics are unchanged by removal of the intermediate
typing-only adaptation; no product code was reverted. The local implementation
review verdict remains no unresolved material findings, conditional on final
required functional evidence/receipt validity. Whole-CI/release remains **not
green**, now explicitly including full Pyright in addition to Ruff, inventory and
OSS. Resolving bound-artifact type debt requires a separately legitimate evidence
lifecycle, not silently rewriting accepted evidence.

Final supplemental evidence read: `S09-restored-fixture-runtime.json`, exit 0,
executes the restored `storage_preflight.setup_preflight` in a temporary directory
and produces actual native NPY/preflight, then verifies wrapped managed/catalog
callbacks. Thus current runtime evidence exists for the restored helper; it is not
merely inferred from old S08 execution. The executor also reports S08 dependency
entry validation passes after restoration; final receipt validation remains its
responsibility. The directing parent explicitly accepted this narrowly attributed
four-diagnostic local debt, with the zero-error type result superseded and no blanket
gate waiver. Final disposition above applies to precisely these four diagnostics
and does not excuse any additional functional or material type failure.
