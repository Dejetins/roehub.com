# Stage 10D: Non-Trading Active Reclassification

Дата проверки: 2026-05-27.

Статус: accepted; implementation commits `08e9ba5f`, `e7b6b0af`,
`cd207388`, `5e8b2083` direct-main delivered; CI/deploy and Mac Studio
runtime dry-run/execute/API/DB/audit/metrics evidence complete.

Scope: controlled reclassification/backfill of existing `active` exchange
connections that are not trading-ready under Stage 10 semantics. Stage 10D does
not revalidate exchange credentials, does not place or simulate orders, does
not design reactivation, and physical hard delete запрещен.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 10C must be accepted before 10D starts. | Iteration ledger marks 10C accepted with direct-main CI/deploy and public runtime evidence complete. | Accepted. | None. |
| Candidate selection | Dry-run selects only `active` rows that are not trading-ready by Stage 10 fields or Stage 09 fallback evidence. | Mac Studio dry-run selected 1 active Bybit mainnet candidate with `permission_mismatch`, `effective_permissions=read`, `exchange_permissions=read`, `connection_readiness=rejected`, reason `read_only_not_supported`; final dry-run count is `0`. | Accepted. | None. |
| Execution path | Execution moves eligible rows out of Active through supported lifecycle semantics. | Candidate moved to `disabled` with `status_reason=reclassified_non_trading_ready`; service refuses active `ready_for_trading` and is idempotent for already reclassified disabled rows. | Accepted. | Reactivation remains intentionally undesigned. |
| Readiness preservation | Reclassified read-only rows remain non-trading with reason `read_only_not_supported`. | DB row `bybit_test` is `disabled`, `connection_readiness=rejected`, reason `read_only_not_supported`; unit regression covers the same behavior. | Accepted. | None. |
| Audit and metrics | Redacted audit and bounded metric exist. | Audit row `exchange_connection_reclassified` has `target_id` and redacted metadata; `/metrics` exposes `exchange_connection_reclassification_total{source="stage10d",result="disabled",reason="read_only_not_supported"} 1.0`. | Accepted. | None. |

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
| execute | `uv run python -m tools.exchange_connections.reclassify_non_trading_active --execute --json --exchange-control-url http://127.0.0.1:9205` | Re-loads candidates and calls local-only exchange-control disable lifecycle with `status_reason=reclassified_non_trading_ready`; records `exchange_connection_reclassified` audit. Idempotent repair emits metric for already reclassified rows after service restart. |

## Runtime Evidence

| Surface | Command | Sanitized result | Verdict |
|---|---|---|---|
| Initial dry-run | `ssh macstudio '... uv run python -m tools.exchange_connections.reclassify_non_trading_active --dry-run --json'` | `candidate_count=1`; candidate refs are hashed; reasons include `permission_mismatch`, `effective_permissions=read`, `exchange_permissions=read`, `effective_capability=none`, `connection_readiness=rejected`, `read_only_not_supported`; safety says `physical hard delete запрещен`. | Pass. |
| Execute | `ssh macstudio '... uv run python -m tools.exchange_connections.reclassify_non_trading_active --execute --json --exchange-control-url http://127.0.0.1:9205'` | Lifecycle execute moved the candidate to disabled. A first audit insert exposed missing production `target_id`; bootstrap was repaired. Final execute returned `candidate_count=0`, `audit_repair_count=0`, `metric_repair_count=1`, no additional mutation. | Pass. |
| Final dry-run | Same dry-run command after execute. | `candidate_count=0`. | Pass. |
| Active API | Temporary local production session, then `curl --max-time 10 -fsS "http://127.0.0.1:8000/ui/account/exchange-connections?status=active" ... \| jq -e 'all(.items[]; .connection_readiness == "ready_for_trading" and .effective_capability == "trading")'`. | `true`; temporary Stage 10D sessions revoked, final active temporary-session count `0`. Public `https://roehub.com` from Mac Studio timed out on 443, so the accepted proof used the deployed local API route from the production smoke surface. | Pass. |
| DB active rows | `SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE status='active' ORDER BY created_at DESC LIMIT 20;` | `0 rows`; no active exchange connections remain. No secret-bearing columns selected. | Pass. |
| Audit | `SELECT event_type, target_id, metadata_json, created_at FROM identity_audit_events WHERE event_type IN ('exchange_connection_reclassified','exchange_connection_disabled') ORDER BY created_at DESC LIMIT 20;` | One `exchange_connection_reclassified` row for the redacted candidate target; metadata contains only exchange, market, environment, previous/new status, reason `read_only_not_supported`, and source `stage10d`. | Pass. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics \| rg 'exchange_connection_reclassification_total\|exchange_connection_trading_readiness_total'`. | `exchange_connection_reclassification_total{source="stage10d",result="disabled",reason="read_only_not_supported"} 1.0`; `exchange_connection_trading_readiness_total{exchange="bybit",result="rejected",reason="read_only_not_supported"} 1.0`. | Pass. |

## Data Safety

| Guard | Implementation / evidence |
|---|---|
| No physical delete | No delete SQL or hard-delete command was added; tool output includes `physical hard delete запрещен`. |
| Dry-run first | The initial Mac Studio dry-run selected exactly 1 redacted candidate before any mutation. |
| Exact mutation | The approved candidate became `disabled/reclassified_non_trading_ready`; final dry-run and DB active query show no remaining candidates. |
| Redaction | Dry-run/execute JSON uses hashed `connection_ref`, `owner_ref`, and `label_ref`; no API secrets, ciphertext, HMAC, tokens, cookies, or raw exchange responses are printed. |
| Lifecycle | Execution routes through exchange-control disable semantics; service re-evaluates readiness, refuses active trading-ready rows, and only treats already reclassified disabled rows idempotently for audit/metric repair. |
| Forward-only | Reclassification moves rows to History/disabled. Reactivation is not designed in Stage 10D. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No public route or DTO field changes. The Active API result changes only because existing data is repaired out of Active. |
| Internal API | `compatible-change` | Internal disable command accepts optional `status_reason` and `reclassification_source`; existing callers keep default user disconnect behavior. |
| Persistence | `compatible-change` | Additive audit `target_id` column and expanded check constraint for `exchange_connection_reclassified`; `0007` was kept current-idempotent so bootstrap can re-run after rows exist. |
| Ops / runtime | `compatible-change` | Adds bounded `exchange_connection_reclassification_total{source,result,reason}` and operator tool module. |
| Browser-visible | `compatible-change` | `/settings` Active is now backed by only trading-ready rows; no UI markup changes. |
| Trading execution | `none` | No exchange order, order simulation, execution process, or order ledger code is added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Required pytest | Passed: `117 passed`, `3 warnings` from existing httpx cookie deprecation. | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations`. |
| Required ruff | Passed. | `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tools tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations`. |
| Required pyright | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`. |
| Docs index | Passed. | `python -m tools.docs.generate_docs_index --check`; index regenerated earlier with `python -m tools.docs.generate_docs_index`. |
| Runtime acceptance | Passed. | Mac Studio dry-run -> execute -> final dry-run -> API -> DB -> audit -> metrics evidence above. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commits / push | `08e9ba5f Add exchange connection reclassification tool`; `e7b6b0af Apply reclassification audit migration in bootstrap`; `cd207388 Make reclassification metric repair idempotent`; `5e8b2083 Keep audit event check idempotent`; all pushed to `origin/main`. | Pass. |
| CI | `26477834060`, `26478061441`, `26478357900`, `26478477763`. | Success; latest CI `26478477763` success. |
| Deploy | Final deploys for `5e8b2083`: Deploy Backend `26478530468` success; Publish App Image `26478530410` success; Deploy Web `26478530397` success. Earlier backend deploy for `cd207388` failed on idempotent audit check and was repaired by `5e8b2083`. | Pass. |
| Runtime | Mac Studio `/opt/roehub/app` deployed bundle ran Stage 10D tool and acceptance calls successfully. | Pass. |

## Stage 10E Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Active list is now clean. | 10E | Production readiness can start from final Stage 10D proof: active exchange connections query returned `0 rows`; future Active rows must be produced only by Stage 10B trading-ready auto-validation. |
| Full production trading-ready success proof may need env-backed trade credentials. | 10E | If trade-enabled credentials are absent, mark the success half partial/blocked rather than inferring readiness. |
| Public edge from Mac Studio timed out on 443 during Stage 10D API proof. | 10E | Prefer authenticated browser/public proof from a client that can reach `https://roehub.com`; local production API route remains available for host-local smoke. |
