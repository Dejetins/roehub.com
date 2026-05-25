# Stage 09D: E2E Cleanup Controlled Backfill

Дата проверки: 2026-05-26.

Статус: local validation complete; direct-main, CI/deploy and Mac Studio cleanup
evidence pending.

Scope ограничен mandatory cleanup/backfill path for old disabled development
exchange connections. Stage 09D does not add public DELETE, does not physically
delete exchange connections, credential versions or audit events, and does not
change permission semantics or trading execution.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| 09C prerequisite | Stage 09C must be accepted before 09D starts. | Iteration ledger marks 09C as `accepted; direct-main delivered; CI/deploy and Mac Studio runtime evidence complete`; 09C report is accepted. | Accepted. | None. |
| Cleanup predicate | Only disabled old test/development rows with label prefix `stage08_`, `e2e_` or `smoke_` are eligible. | `select_cleanup_candidates` requires matching label prefix, `status='disabled'`, `disabled_at IS NOT NULL`, and `archived_at IS NULL`; tests prove active, already archived and manual labels are skipped. | Accepted locally. | Production execution must use dry-run output before mutation. |
| Supported lifecycle path | Cleanup execution must archive through the existing archive command, not ad hoc SQL mutation. | `tools/exchange_connection_cleanup.py --execute` calls `HttpExchangeControlClient.archive_connection(..., cleanup_source='stage09d')`; SQL is read-only for candidate discovery. | Accepted locally. | Requires exchange-control internal token on target runtime. |
| Audit and metrics | Cleanup archives must emit `exchange_connection_archived` audit evidence and `exchange_connection_cleanup_total` metric evidence. | Execution path records audit via `AccountSettingsUseCase.record_exchange_connection_archive`; internal archive request accepts bounded `cleanup_source` and records `exchange_connection_cleanup_total{source,result}` in exchange-control. | Accepted locally. | Production metric increments after Mac Studio execution. |
| Default visibility | Archived cleanup rows must be hidden from the default account/API/UI list. | Existing 09B API/web tests still pass; default list remains `status=active`, and explicit history filters expose disabled/archived. | Accepted locally. | Production-browser proof remains 09E; 09D can use API/runtime evidence. |
| No hard delete | physical hard delete запрещен. | No `DELETE` route or SQL delete added; cleanup tool performs SELECT for candidates and archive command for execution. | Accepted. | Archived rows are forward-only history. |

## Cleanup Predicate

| Predicate | Required value | Reason |
|---|---|---|
| `label` | starts with `stage08_`, `e2e_` or `smoke_` | Restricts cleanup to known development/e2e rows. |
| `status` | `disabled` | Active user records are never touched. |
| `disabled_at` | not null | Confirms lifecycle disable happened before archive. |
| `archived_at` | null | Keeps archive idempotent and avoids repeat audit noise. |
| `owner_user_id` | optional CLI filter | Allows narrower operator-scoped cleanup when a test account is known. |
| `created_at` | optional `--created-after` / `--created-before` | Allows time-window narrowing for Stage 08/09 evidence runs. |

## Implementation Evidence

| Surface | Change | Evidence |
|---|---|---|
| Operator tool | Added `tools/exchange_connection_cleanup.py` with default `--dry-run`; `--execute` is explicit. | Tool selects redacted candidates and prints JSON evidence without secrets, ciphertext, HMAC, tokens or raw exchange responses. |
| Candidate selection | Conservative Python and SQL predicates require disabled matching-prefix rows only. | `tests/unit/contexts/exchange_control/test_exchange_connection_cleanup.py::test_cleanup_candidate_selection_is_conservative`. |
| Archive execution | Execution archives through `ExchangeControlClient.archive_connection`, passing bounded `cleanup_source`. | `test_execute_cleanup_archives_through_client_and_records_audit`. |
| Audit | Cleanup execution writes `exchange_connection_archived` through the accepted account settings use-case. | Same test asserts redacted audit metadata path and event literal. |
| Metrics | Internal archive request accepts optional `cleanup_source` and records `exchange_connection_cleanup_total{source,result}`. | Runtime test asserts `result="archived",source="stage09d"` appears in `/metrics` and no `connection_id` label is present. |
| Public UI/API defaults | Stage 09B default active-only behavior remains unchanged. | Required API/web tests passed with Stage 09D changes. |

## Dry-Run And Execution Evidence

| Environment | Mode | Command / evidence | Result | Secret safety |
|---|---|---|---|---|
| local-dev | unit dry-run | `select_cleanup_candidates` and `summarize_candidates` over mixed rows. | Selected only disabled `stage08_` and `e2e_` rows; active/manual/already-archived rows skipped. | Redacted refs only; full UUID absent from evidence string. |
| local-dev | execution simulation | `execute_cleanup` with fake archive client and audit recorder. | Archive client called once with `cleanup_source='stage09d'`; `exchange_connection_archived` audit path recorded. | No secret-bearing fields in candidate/result model. |
| Mac Studio | dry-run | Pending after direct-main deploy. | Pending. | Pending. |
| Mac Studio | execution | Pending after dry-run acceptance. | Pending. | Pending. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No new public route, DTO field or browser action is added in 09D. |
| Internal API | `compatible-change` | Existing local-only archive command accepts optional `cleanup_source`; absent field preserves 09A-09C callers. |
| Operator tooling | `compatible-change` | Adds a dry-run-by-default tool under `tools/`; execution requires explicit `--execute` and internal exchange-control auth. |
| Persistence | `compatible-change` | Selected disabled cleanup rows move to existing `archived` lifecycle state; no physical delete or schema migration. |
| Audit | `compatible-change` | Reuses accepted `exchange_connection_archived` event with redacted metadata. |
| Metrics / ops | `compatible-change` | `exchange_connection_cleanup_total{source,result}` now increments with bounded labels and no user/connection/credential labels. |
| Browser-visible behavior | `none` | 09B active-only default remains unchanged; archived rows are visible only through explicit history filters. |
| Config / env | `none` | No new required env var; tool reuses `IDENTITY_PG_DSN`, `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL`, and `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`. |
| Trading execution | `none` | No order placement, execution authorization, order ledger or signal-to-execution behavior added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `92 passed`, 3 known httpx cookie deprecation warnings. | Local run on 2026-05-26. |
| `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed. | Local run on 2026-05-26. |
| `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed: `0 errors`. | Local run on 2026-05-26. |
| `python -m tools.docs.generate_docs_index --check` | Passed. | Docs index updated for the 09D report and checked on 2026-05-26. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Pending. |
| CI / deploy | Pending. | Pending. |
| Mac Studio smoke | Pending. | Pending. |
| Mac Studio cleanup dry-run/execution | Pending. | Pending. |

## Residual Risk And Stage 09E Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Archived cleanup rows are forward-only history. | 09D/09E operators | Recovery is explicit archived history view or a future unarchive design, not silent unarchive. |
| Production authenticated browser proof remains separate. | 09E | Prove create -> validate or deterministic skip/failure -> disable -> archive -> default hidden with real authenticated browser flow. |
| Cleanup predicate must remain conservative. | Future cleanup runs | Keep `--dry-run` first and use owner/time filters whenever available; never archive active or manually created user records. |
