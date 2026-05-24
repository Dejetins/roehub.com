# Stage 7: Production Readiness

Дата проверки: 2026-05-24.

Статус: accepted for direct-main delivery; final direct-main push, CI/deploy
and post-push runtime evidence are recorded in the delivery section after this
report was first validated.

Stage 7 is the final production-readiness gate for Exchange Control v1 key
storage and validation. It does not change code, schemas, public API contracts,
runtime config, UI behavior, exchange execution, order placement, order ledger
or signal-to-execution behavior.

## Verdict

Exchange Control v1 key storage and validation are production-ready for
credential custody, read-only validation and `/settings` management only.
Trading execution remains explicitly out of scope; future execution work
заблокирован until a separate signal-to-execution design is accepted.

## Evidence Matrix

| Stage | Required evidence | Observed evidence | Command / artifact | Verdict | Blocker |
|---|---|---|---|---|---|
| Stage chain | Stage reports 00-06 present and accepted, including 03A, 03B and 03C. | Reports 00, 01, 02, 03A, 03B, 03C, 04, 05 and 06 are present; ledger marks 02-06 with direct-main and Mac Studio evidence; Stage 6 report is accepted with browser, CI/deploy and Mac Studio smoke evidence. | `rg --files docs/architecture/identity/exchange-connections-stage-reports`; ledger `Stage Status`; `06-settings-ui.md` | Pass | None |
| API/UI/storage/validation | Account facade, storage, validation status, metrics and audit evidence remain coherent. | Focused API/UI/web/exchange-control/migration tests passed; Stage 6 report confirms `/settings` uses backend connection status, `read` default permission, no Binance/Bybit passphrase facade and secret-safe actions. | `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control tests/unit/apps/migrations` -> `85 passed, 3 warnings` | Pass | None |
| Lint | Focused implementation and tests lint clean. | Ruff passed. | `uv run ruff check apps/api apps/web src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control` | Pass | None |
| Type check | Focused API/identity/exchange-control pyright clean. | Pyright returned `0 errors, 0 warnings, 0 informations`. | `uv run pyright apps/api src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control` | Pass | None |
| Docs index | Architecture index up to date after Stage 7 docs are added. | Initial pre-report check passed; post-report docs-index was regenerated and the final check passed. | `python -m tools.docs.generate_docs_index --check` | Pass | None |
| Runtime health | `exchange-control` ready on target runtime. | Mac Studio returned `status=ready`, `service=exchange-control`, `service_identity=exchange-control`; external exchange validation check was ready. | `ssh macstudio 'curl -fsS http://127.0.0.1:9205/health/ready'` | Pass | None |
| Internal API auth | Authenticated `/internal/v1/capabilities` reachable with service auth; missing auth denied. | Authenticated call returned `contract_version=internal-v1`, request id `stage-7-readiness`, capabilities including `exchange_connections.validate`; unauthenticated call returned HTTP `401 internal_auth_required`. | `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN`; `/internal/v1/capabilities`; `X-Roehub-Internal-Service: apps/api` | Pass | None |
| Metrics | Required exchange-control metrics exposed without secret-bearing labels. | Metrics include `exchange_control_active 1.0` and `exchange_connection_validation_total{exchange="none",reason="stage_2_no_real_exchange_calls",result="disabled"} 0.0`. | `ssh macstudio 'curl -fsS http://127.0.0.1:9205/metrics' \| rg 'exchange_control_active\|exchange_connection_validation_total'` | Pass | None |
| Prometheus | `up{job="exchange-control"}` is healthy on target runtime. | Prometheus returned `status=success`, `instance=127.0.0.1:9205`, `job=exchange-control`, value `1`. | `curl -fsSG http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=up{job="exchange-control"}'` on Mac Studio | Pass | None |
| Monit | `roehub_exchange_control` supervised and OK. | Monit summary returned `roehub_exchange_control OK Process`. | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary` on Mac Studio | Pass | None |
| Security | CSRF/recent-auth, no secret fields, audit redaction and no artifact leakage are accepted. | Focused tests cover `csrf_required`, `csrf_origin_mismatch`, `recent_auth_required`, secret-free responses/audit and passphrase rejection; artifact/log grep returned no matches. | `test_identity_exchange_keys_routes.py`; `test_ui_account_routes.py`; `rg ... logs output .playwright-cli` | Pass | None |
| No execution | No exchange execution, order placement or signal-to-execution implementation in scope. | No matches for execution/order placement literals in exchange-control/API/web paths. | `! rg -n "/order\|createOrder\|submit_order\|place_order\|exchange-execution" src/trading/contexts/exchange_control apps/api apps/web` | Pass | None |
| Direct-main preflight | Work starts and stays on `main`; fast-forward from `origin/main` succeeds. | `main`; `git pull --ff-only origin main` returned `Already up to date`; no stage branch or draft PR created. | `test "$(git branch --show-current)" = main`; `git pull --ff-only origin main`; `git status --short` | Pass | None |
| GitHub CLI | GitHub CLI installed and authenticated for direct-main CI/deploy follow-through. | `gh version 2.85.0`; authenticated account `Dejetins` with repo/workflow scopes. | `gh --version && gh auth status` | Pass | None |

## Security And Secrets

| Check | Evidence | Verdict |
|---|---|---|
| CSRF and recent-auth | Focused route tests assert fail-closed behavior for missing/cross-origin CSRF and stale recent auth before credential mutations. | Pass |
| Response shape | Focused route tests assert `api_secret`, `passphrase`, ciphertext, HMAC and raw secret material are absent from API responses. | Pass |
| Audit shape | Focused tests assert exchange credential audit metadata excludes secrets, ciphertext, HMAC, fingerprints and raw exchange error bodies. | Pass |
| Account facade | Stage 6 accepted `/settings` facade rejects Binance/Bybit `passphrase`, clears write-only password inputs and keeps `read` as the default permission. | Pass |
| Artifact/log grep | `rg -n 'STAGE7_SECRET\|STAGE6_BROWSER_SECRET\|STAGE6_SECRET\|STAGE6_BROWSER_PASSPHRASE\|TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE\|ROEHUB_TEST_BINANCE_READONLY_API_SECRET\|ROEHUB_TEST_BYBIT_READONLY_API_SECRET\|api_secret\|passphrase\|sk_live_\|AKIA[0-9A-Z]{16}' logs output .playwright-cli 2>/dev/null \|\| true` returned no matches. | Pass |
| Internal service token | `ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN` was sourced only from Mac Studio host-local env for the authenticated capabilities smoke; token value was not printed or committed. | Pass |

## Ops And Runtime

| Area | Evidence | Verdict |
|---|---|---|
| Mac Studio access | SSH alias `macstudio` resolved to `MacStudioDaniil`; `/opt/roehub/app` and host-local env file exist. | Pass |
| `exchange-control` readiness | `/health/ready` returned ready with service identity `exchange-control`. | Pass |
| Internal command API | `/internal/v1/capabilities` returned `contract_version=internal-v1`, `capabilities.read`, `exchange_credentials.encrypt`, `exchange_credentials.decrypt.exchange_control_only`, create/list/rotate/disable/validate capabilities and `no_implicit_retry`. | Pass |
| Auth denial | Missing `Authorization` header returned HTTP `401 internal_auth_required`. | Pass |
| Metrics | `/metrics` exposes `exchange_control_active` and `exchange_connection_validation_total`. | Pass |
| Prometheus | `up{job="exchange-control"}` returned `1`. | Pass |
| Monit | `roehub_exchange_control OK Process`. | Pass |
| Monitoring docs | `infra/macos/prometheus/prometheus.prod.yml`, `docs/runbooks/mac-studio-monitoring-plan.md` and `docs/runbooks/exchange-secret-management.md` already describe the accepted exchange-control target, Monit service, Transit custody and emergency disable boundaries; no stale current behavior was found in the reviewed continuity docs. | Pass |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | Stage 7 adds readiness documentation only. Public routes and request/response shapes are unchanged. |
| Persisted schema | `none` | No migration, table, column or backfill behavior changed. |
| Config / env | `none` | No runtime configuration keys or defaults changed. |
| Browser-visible behavior | `none` | No UI templates, JS or browser defaults changed. Stage 6 browser evidence remains the current accepted UI surface. |
| Runtime / ops | `none` | No launchd, Monit, Prometheus or service files changed. Runtime commands were evidence collection only. |
| Trading execution | `none` | No exchange-execution, order placement, order ledger or signal-to-execution code exists in this Stage 7 diff. |

## Rollback Notes

| Scenario | Rollback path | Data impact |
|---|---|---|
| Stage 7 report needs removal | Revert this report and the Stage 7 ledger/docs-index rows. | None; documentation-only. |
| A future runtime regression appears | Do not roll back Stage 7 docs as a functional mitigation; use the exchange-secret-management and Mac Studio monitoring runbooks to recover `exchange-control`, OpenBao/Transit, Prometheus or Monit first. | None from Stage 7. |
| Execution work is requested prematurely | Keep future execution work заблокирован and require a separate signal-to-execution architecture prompt before any exchange-execution/order work. | Prevents unsafe scope expansion. |

## Direct-Main Delivery Evidence

| Item | Evidence | Result |
|---|---|---|
| Pre-delivery validation | Branch `main`, fast-forward from `origin/main`, focused gates, docs index, no-order grep, secret grep and Mac Studio runtime evidence all passed before staging. | Accepted. |
| Stage 7 validation/report commit | `TBD after commit` | Pending until the scoped documentation commit is created. |
| Push | `git push origin main` | Pending after the scoped documentation commit. |
| CI/deploy | GitHub Actions for the pushed `main` revision | Pending after push. |
| Deploy/runtime | Stage 7 changes are docs-only; target runtime was already checked before delivery and must stay healthy after CI/deploy. | Pending post-push check. |

## Next Prompt

Use this only after a separate product decision accepts execution design scope:

```text
Design signal-to-execution v1 for Roehub as a separate architecture stage.
Do not reuse Exchange Control v1 key-storage readiness as permission to place
orders. Define strategy signal contracts, execution intent, risk gates, order
submit adapters, order ledger, reconciliation, kill switch, monitoring,
rollback and staged Mac Studio acceptance evidence before any implementation.
```
