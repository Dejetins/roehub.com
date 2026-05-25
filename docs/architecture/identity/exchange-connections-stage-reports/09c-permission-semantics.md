# Stage 09C: Permission Semantics

Дата проверки: 2026-05-26.

Статус: accepted locally; direct-main delivery pending push, CI/deploy and
Mac Studio runtime evidence.

Scope ограничен permission semantics внутри `exchange-control`, API DTO
проекцией и отображением в `/settings`. Stage 09C не добавляет trading
execution, order placement, permission cleanup/backfill or public DELETE.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| 09B prerequisite | Stage 09B must be accepted before 09C starts. | Iteration ledger marks 09B as `accepted; direct-main delivered; CI/deploy and Mac Studio runtime evidence complete`; 09B report is accepted. | Accepted. | None. |
| Permission ownership | `effective_permissions` is computed only in `exchange-control`. | Binance/Bybit adapters compute requested/exchange/effective semantics; apps/api DTO/client and UI only pass/display fields. | Accepted. | Future execution must authorize from `effective_permissions`, not user-requested `permissions`. |
| DTO compatibility | Add explicit permission fields while preserving old `permissions` alias. | `ExchangeConnectionResponse` includes `requested_permissions`, `exchange_permissions`, `effective_permissions`, `permission_warnings`; `permissions` remains requested alias. | Accepted. | Old consumers can keep reading `permissions`; new consumers should switch to explicit fields. |
| Canonical mismatch | Requested trade + exchange readonly must become `permission_mismatch` with reason `requested_trade_but_exchange_readonly`. | Binance and Bybit truth-table tests cover mismatch status/reason and effective read. | Accepted. | None. |
| Invalid capability | Withdrawal/transfer and invalid credentials must not produce executable capability. | Tests cover `withdraw_or_transfer -> effective none` and invalid credentials -> `exchange_permissions=unknown`, `effective_permissions=none`. | Accepted. | None. |
| Browser display | UI must not show requested trade + readonly exchange validation as normal successful trade state. | Browser QA shows `permission_mismatch` rendered with `is-negative`, not `is-positive`, and visible permissions text `requested trade / exchange read / effective read`. | Accepted. | Production authenticated browser proof remains 09E. |
| Metrics | Permission mismatch metric has bounded labels and no user/connection/credential labels. | `exchange_permission_mismatch_total{exchange,requested,effective}` added; runtime test checks metric contract exists. | Accepted. | Metric increments only when validation returns canonical mismatch. |

## Permission Truth Table

| Requested | Exchange-observed | Validation status | Reason / warning | Effective |
|---|---|---|---|---|
| `read` | readonly/read | `valid_readonly` | `readonly_permission_detected` | `read` |
| `read` | trade/write | `valid_trade_enabled` | warning `exchange_permissions_exceed_requested` | `read` |
| `trade` | trade/write | `valid_trade_enabled` | exchange trade reason | `trade` |
| `trade` | readonly/read | `permission_mismatch` | `requested_trade_but_exchange_readonly` | `read` |
| `read` or `trade` | withdrawal/transfer | `invalid_permissions` | exchange-specific withdrawal/transfer reason | `none` |
| `read` or `trade` | invalid credentials | `invalid_credentials` | sanitized exchange rejection/failure reason | `none` |

## Implementation Evidence

| Surface | Change | Evidence |
|---|---|---|
| Validation contract | `permission_mismatch` added to canonical validation statuses while keeping `skipped_external_validation`. | `src/trading/contexts/exchange_control/application/validation.py`; focused validator tests. |
| Binance normalization | Adds requested/exchange/effective summary fields and mismatch policy without changing endpoint selection. | `normalize_binance_api_restrictions(..., requested_permissions=...)`; `test_binance_permission_truth_table`. |
| Bybit normalization | Same policy for Bybit `readOnly`, trade/write and wallet transfer permissions. | `normalize_bybit_api_key_info(..., requested_permissions=...)`; `test_bybit_permission_truth_table`. |
| Domain/repository | Connection view exposes explicit permission fields; in-memory and Postgres repositories preserve metadata through existing `permission_summary_json`. | `connections.py`; `postgres_connections.py`; no migration added. |
| Internal API | Exchange-control internal response includes explicit permission fields and mismatch metric counter. | `src/trading/contexts/exchange_control/adapters/inbound/http/app.py`; runtime tests. |
| apps/api facade | Client/parser and public DTO include explicit fields; route response maps bounded literals. | `apps/api/exchange_control_client.py`; `apps/api/dto/ui_account.py`; `apps/api/routes/ui_account.py`. |
| Settings UI | Permissions cell displays requested/exchange/effective; `permission_mismatch` is negative, not success. | `apps/web/dist/js/pages/settings.js`; web route test; browser QA. |

## Browser Evidence

| Check | Evidence | Result |
|---|---|---|
| Local browser target | `http://127.0.0.1:8766/settings` with mocked authenticated current user and same-origin API upstream. | Pass. |
| Active-only request | Playwright requests include `GET /api/ui/account/exchange-connections?status=active => 200`. | Pass. |
| Visible permission fields | Active row rendered `requested trade / exchange read / effective read`. | Pass. |
| Mismatch styling | DOM eval returned `validationClass=is-negative`, `successClass=false`, `negativeClass=true`. | Pass. |
| Console | `playwright-cli console warning` returned `Errors: 0, Warnings: 0`. | Pass. |
| Screenshot | `output/playwright/stage09c-settings-permission-mismatch.png`. | Captured local mismatch state. |
| Artifact grep | `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE|api_secret|passphrase|stage09c-session" logs output .playwright-cli 2>/dev/null || true`. | No matches. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds explicit permission fields and `permission_mismatch` status; keeps `permissions` as alias to `requested_permissions`. |
| Internal API | `compatible-change` | Adds permission fields to local-only exchange-control connection response. |
| Domain / port | `compatible-change` | Connection view/record now carries permission summary metadata; effective permissions remain exchange-control-owned. |
| Persistence | `compatible-change` | Uses existing `permission_summary_json`; no migration or destructive backfill. Existing rows fall back to requested alias, `exchange_permissions=unknown`, `effective_permissions=none`. |
| Browser-visible behavior | `compatible-change` | `/settings` displays requested/exchange/effective permissions and treats mismatch as negative/warning state. |
| Metrics / ops | `compatible-change` | Adds bounded `exchange_permission_mismatch_total{exchange,requested,effective}` without user/connection/credential labels. |
| Config / env | `none` | No new env vars or feature flags. |
| Trading execution | `none` | No order placement, execution authorization, order ledger or signal-to-execution path added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Focused pytest | Passed: `72 passed`, 3 known httpx cookie deprecation warnings. | `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_validation.py tests/unit/contexts/exchange_control/test_exchange_control_runtime.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py`. |
| Ruff focused | Passed. | `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web`. |
| Pyright focused | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`. |
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py` | Passed: `74 passed`, 3 known httpx cookie deprecation warnings. | Local run on 2026-05-26. |
| `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web` | Passed. | Local run on 2026-05-26. |
| `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed: `0 errors`. | Local run on 2026-05-26. |
| `python -m tools.docs.generate_docs_index --check` | Passed. | Docs index updated for the 09C report and checked on 2026-05-26. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Will be recorded after validation and direct-main delivery. |
| CI / deploy | Pending. | Will be recorded after push. |
| Mac Studio runtime | Pending. | Will be recorded after deploy/runtime follow-through. |

## Residual Risk And Stage 09D Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Old disabled `stage08_*` / `e2e_*` / `smoke_*` rows still exist. | 09D | Run controlled cleanup/backfill through supported archive semantics only; physical hard delete запрещен. |
| `permissions` alias remains for compatibility. | Future cleanup after adoption | Consumers should migrate to `requested_permissions`, `exchange_permissions`, and `effective_permissions` before alias removal is considered. |
| Production authenticated browser proof is deferred. | 09E | Prove create -> validate -> disable -> archive -> hidden default list on production after cleanup/backfill. |
