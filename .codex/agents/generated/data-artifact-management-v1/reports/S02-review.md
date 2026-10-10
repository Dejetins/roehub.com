# S02 independent production risk review

2026-10-08. Reviewer: `s02_production_review`, independently delegated subagent using `production-risk-review`. Scope: S02 only, against its prompt, S01 accepted contracts and the current worktree. No product edits, recursive review, publication or scheduler activation by this reviewer.

## Verdict

No unresolved material findings within the S02 application/repository/migration proof boundary. The three initial defects below were repaired and independently re-read. Final management PostgreSQL/profile tests were also independently rerun by the reviewer: **42 passed, zero skipped, exit 0, 3.99s**.

This verdict supports S02 acceptance through the existing updater; it does not itself accept the ledger, authorize the next stage or establish production/runtime readiness.

## Findings and disposition

1. **F1 — P1, high confidence: conflicting immutable profile payloads inside one bulk. Fixed.** `management_repository.py::submit` originally checked only already-persisted revisions. Two bulk items could use the same new `(profile_id, revision)` with different content: the second `ON CONFLICT DO NOTHING` kept the first persisted profile while its request snapshot used the second. The pre-write batch identity map now rejects the conflict. `test_conflicting_profile_inside_same_bulk_rolls_back` verifies no commands, profiles or requests survive.
2. **F2 — P2, high confidence: retry bypassed a reduced installation budget. Fixed.** `management_repository.py::retry` originally checked capability/root but omitted the current reservation bound applied by `claim`. It now rejects `reservation_bytes + 1 GiB > budget_bytes` before creating a new attempt. `test_retry_budget_shrink_blocks_before_new_attempt` confirms the retry remains unclaimed and the attempt count stays unchanged.
3. **F3 — P2, high confidence: command recovery omitted current coordinate authorization. Fixed.** `management_repository.py::command` originally returned coordinate/request metadata after membership authorization alone. It now resolves access for every bounded response coordinate before returning the result. `test_command_recovery_checks_current_coordinate_access` proves denial after access revocation.

The initial concurrent-retry evidence gap is closed by `test_concurrent_retry_has_one_new_attempt_and_fences_loser` and `test_retry_racing_new_claim_preserves_global_budget`. The trigger-injected failure after a first child insert additionally proves database rollback of the entire command/profile/policy/request envelope.

## Inspected scope and evidence

Read the S02 stage contract, S01 report and relevant plan requirement mapping; inspected the management models/ports, runtime-derived catalog, explicit legacy projection, PostgreSQL repository, additive 0029 SQL, manifest/bootstrap/storage integration and changed unit/real-infrastructure tests. Checked command/child organization FKs, current role and installation-owner checks, CAS and immutable revisions, bounded admission, global claim locking, token/incarnation/attempt/epoch fencing, retry/quarantine semantics, finite profiles and legacy preservation. No unrelated product scope drift identified.

Final additions were inspected: explicit bounded `signal_params` come from the existing defaults provider and participate in catalog identity/validation; legacy snapshots include compute/source/signal defaults in their effective hash. Replay lookup precedes the system desired-target upsert, avoiding regression from replaying an already-admitted older occurrence. Strict grid/risk values reject booleans/nonfinite values. These changes remain inside the selected persistence/profile work.

Reviewer execution:

```sh
ROEHUB_BACKTEST_TEST_DSN='host=127.0.0.1 port=25482 user=postgres dbname=postgres' .venv/bin/python -m pytest -q tests/real_infra/backtest/test_artifact_management.py tests/unit/contexts/backtest_artifacts/test_management_profiles.py
```

Result: **42 passed in 3.99s**, including all 17 PostgreSQL tests. The target is the executor-owned disposable container documented in `S02-verification.md`; each test uses and drops its own random schema. This reviewer did not operate on preview/user data. An earlier independent unit/profile plus storage-lifecycle run gave **34 passed in 1.02s** before the final signal-default test; the final command above verifies the subsequently changed catalog/import/model code.

Reviewed durable executor evidence in `S02-verification.md` and scope/limits in `S02.md`. The executor records 48 existing physical ownership tests, 309 routed artifact tests, 102 artifact/migration units before final additions, fresh and repeated full migration orchestration, and owned Ruff/Pyright passes. Those broader commands were not separately rerun by this reviewer, and overlapping counts are not summed. Source bindings are recorded in `S02-source-bindings.json`.

## Compatibility and remaining proof boundary

The schema is additive and preserves earlier migrations/recipe identities; explicit legacy import preserves implicit precompute independently of new on-demand policies. Classification: `compatible-change` for the new persistence interfaces, `none` for unchanged existing runtime consumers. Installed-version compatibility and runtime activation remain `unknown`.

The current slice has no production API composition or scheduler activation. `CoordinateAccess` must be supplied from trusted current source authorization in S06; the test resolver does not prove that future adapter. Source identities and committed-pointer receipts are trusted internal inputs which S03 must obtain from canonical source/publication evidence. Queue claim does not replace physical slot ownership and does not authorize filesystem writes. Actual occupied/free disk accounting, compatible inventory union, scheduling/DST/fair dispatch, quarantine reconciliation and public API/CSRF remain the explicitly assigned later-stage work. The reviewed report states these limits; no browser, numerical artifact, cleanup or deployment proof is claimed.

No additional owner decision or documentation correction is required for S02. Next safe step is the executor's stage-result/receipt validation and atomic acceptance, followed by a separate S03 execution only under its authorized lifecycle.
