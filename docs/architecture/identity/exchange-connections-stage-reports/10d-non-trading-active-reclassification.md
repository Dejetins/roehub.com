# Stage 10D: Non-Trading Active Reclassification

Дата проверки: 2026-05-27.

Статус: in progress; implementation and local gates complete pending
direct-main deploy and Mac Studio runtime dry-run/execute evidence.

Scope: controlled reclassification/backfill of existing `active` exchange
connections that are not trading-ready under Stage 10 semantics. Stage 10D does
not revalidate exchange credentials, does not place or simulate orders, does
not design reactivation, and physical hard delete запрещен.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 10C must be accepted before 10D starts. | Iteration ledger marks 10C accepted with direct-main CI/deploy and public runtime evidence complete. | Accepted. | None. |
| Candidate selection | Dry-run selects only `active` rows that are not trading-ready by Stage 10 fields or Stage 09 fallback evidence. | Unit truth table covers `permission_mismatch`, `effective_permissions=read`, `exchange_permissions=read`, `effective_capability!=trading`, and `connection_readiness!=ready_for_trading`; trading-ready and non-active rows are excluded. | Local accepted. | Production count pending Mac Studio dry-run. |
| Execution path | Execution moves eligible rows out of Active through supported lifecycle semantics. | `ExchangeConnectionService.reclassify_non_trading_active_connection` re-checks active ownership and refuses `ready_for_trading`; internal command uses disable lifecycle with `status_reason=reclassified_non_trading_ready`. | Local accepted. | Runtime execution pending dry-run approval. |
| Readiness preservation | Reclassified read-only rows remain non-trading with reason `read_only_not_supported`. | Domain regression verifies disabled reclassified readonly row keeps `effective_capability=none`, `connection_readiness=rejected`, and reason `read_only_not_supported`. | Local accepted. | Existing rows with incomplete old metadata may resolve to `validation_required`. |
| Audit and metrics | Redacted audit and bounded metric exist. | Additive audit event `exchange_connection_reclassified`, `target_id`, and metric `exchange_connection_reclassification_total{source,result,reason}` are implemented and unit-covered. | Local accepted. | Runtime metric/audit rows pending Mac Studio execute. |

## Reclassification Contract

| Scenario | Selected by dry-run | Execute behavior |
|---|---|---|
| Active `ready_for_trading` / `effective_capability=trading` | No. | No mutation. Service rejects if called directly with `exchange_connection_trading_ready`. |
| Active readonly evidence | Yes: `exchange_permissions=read`, `effective_permissions=read`, or `read_only_not_supported`. | `disabled`, `status_reason=reclassified_non_trading_ready`, readiness remains rejected/read-only. |
| Active `permission_mismatch` | Yes. | `disabled` through supported lifecycle path. |
| Active unsafe permissions | Yes when Stage 10 readiness/capability is non-trading. | `disabled` through supported lifecycle path. |
| Active validation unavailable / required | Yes when not `ready_for_trading`. | `disabled` through supported lifecycle path. |
| Disabled or archived rows | No. | No mutation. |

## Operator Command

| Mode | Command | Contract |
|---|---|---|
| dry-run | `uv run python -m tools.exchange_connections.reclassify_non_trading_active --dry-run --json` | Prints redacted refs, `candidate_count`, reasons, and `physical hard delete запрещен`; no mutation. |
| execute | `uv run python -m tools.exchange_connections.reclassify_non_trading_active --execute --json` | Re-loads candidates and calls local-only exchange-control disable lifecycle with `status_reason=reclassified_non_trading_ready`; records `exchange_connection_reclassified` audit. |

## Runtime Evidence

| Surface | Command | Sanitized result | Verdict |
|---|---|---|---|
| Dry-run | `uv run python -m tools.exchange_connections.reclassify_non_trading_active --dry-run --json` | Pending post-deploy Mac Studio run. | Pending. |
| Execute | `uv run python -m tools.exchange_connections.reclassify_non_trading_active --execute --json` | Pending approved dry-run. | Pending. |
| Active API | `curl -fsS "$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active" ... \| jq -e 'all(.items[]; .connection_readiness == "ready_for_trading" and .effective_capability == "trading")'` | Pending post-execute proof. | Pending. |
| DB active rows | `SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE status='active' ...` | Pending post-execute proof; no secret-bearing columns selected. | Pending. |
| Audit | `SELECT event_type, target_id, metadata_json, created_at FROM identity_audit_events WHERE event_type IN ('exchange_connection_reclassified','exchange_connection_disabled') ...` | Pending post-execute proof. | Pending. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics \| rg 'exchange_connection_reclassification_total\|exchange_connection_trading_readiness_total'` | Pending post-deploy proof. | Pending. |

## Data Safety

| Guard | Implementation |
|---|---|
| No physical delete | No delete SQL or hard-delete command is added; tool output includes `physical hard delete запрещен`. |
| Dry-run first | The default mode is `--dry-run`; `--execute` is an explicit mutually exclusive mode. |
| Redaction | Dry-run/execute JSON uses hashed `connection_ref`, `owner_ref`, and `label_ref`; no API secrets, ciphertext, HMAC, tokens, cookies, or raw exchange responses are printed. |
| Lifecycle | Execution routes through exchange-control disable semantics; the service re-evaluates readiness and refuses already trading-ready active rows. |
| Forward-only | Reclassification moves rows to History/disabled. Reactivation is not designed in Stage 10D. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No public route or DTO field changes. The Active API result changes only because existing data is repaired out of Active. |
| Internal API | `compatible-change` | Internal disable command accepts optional `status_reason` and `reclassification_source`; existing callers keep default user disconnect behavior. |
| Persistence | `compatible-change` | Additive audit `target_id` column and expanded check constraint for `exchange_connection_reclassified`; no secret-bearing schema changes. |
| Ops / runtime | `compatible-change` | Adds bounded `exchange_connection_reclassification_total{source,result,reason}` and operator tool module. |
| Browser-visible | `compatible-change` | `/settings` Active becomes cleaner after data repair; no UI markup changes. |
| Trading execution | `none` | No exchange order, order simulation, execution process, or order ledger code is added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Focused reclassification tests | Passed: `38 passed`. | `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_connection_reclassification.py tests/unit/contexts/exchange_control/test_exchange_connection_readiness.py tests/unit/contexts/exchange_control/test_exchange_control_runtime.py tests/unit/apps/migrations/test_identity_exchange_audit_events_sql.py`. |
| Focused ruff | Passed. | `uv run ruff check ... changed files ...`. |
| Focused pyright | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tools/exchange_connections/reclassify_non_trading_active.py`. |
| Required gates | Pending. | Full Stage 10D gates run after report/ledger are complete. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Pending local gates and runtime/deploy evidence. |
| CI / deploy | Pending. | Pending direct-main push. |
| Runtime | Pending. | Dry-run -> execute -> API/DB/audit/metrics proof still required for acceptance. |

## Stage 10E Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Existing active non-ready rows must be reclassified before Stage 10E readiness. | 10D | Complete Mac Studio dry-run/execute and prove Active API contains only `ready_for_trading` rows. |
| Full production trading-ready success proof may need env-backed trade credentials. | 10E | If trade-enabled credentials are absent, mark the success half partial/blocked rather than inferring readiness. |
