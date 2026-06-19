# Stage 10: Strategy UI Status And Journal

Статус: `accepted`.

User required before start: nothing.

Stage `09` gate: accepted in `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; Stage `10` may start.

## Scope

Stage `10` completes the `/strategies` operational status/journal surface for paper and testnet strategies. The UI must show market, exchange, environment, producer state, allocation, readiness, latest signal, source event, intent, order/fill/reconciliation outcome, observed latency gap, and manual entry/exit controls without exposing secrets or raw provider payloads.

## Concrete File List Before Edits

The prompt listed `apps/web/templates` as a broad path. Before implementation, this stage narrows it to:

| Path | Planned action | Reason |
|---|---:|---|
| `apps/web/templates/pages/strategies.html` | modify | Render runtime status and richer execution outcome columns on the `/strategies` page. |
| `apps/web/templates/fragments/strategies/loading_state.html` | no planned edit | Existing loading fragment remains sufficient. |

Planned code/docs/test files:

| Path | Planned action | Reason |
|---|---:|---|
| `apps/api/dto/ui_strategies_dashboard.py` | modify | Add additive dashboard DTO fields for runtime status and richer outcome links. |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | modify | Populate runtime status, allocation, outcome fill/reconciliation, and observed latency data from existing read models. |
| `src/trading/contexts/live_execution/domain/execution_source.py` | modify | Add optional fields to the internal producer outcome link read model. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/execution_intent_repository.py` | modify | Populate source-event timestamp, fill count, and latest reconciliation summary from existing ledgers. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/execution_intent_repository.py` | modify | Keep the in-memory read model compatible with the additive link fields. |
| `apps/web/dist/js/pages/strategies.js` | modify | Render runtime status, latest signal/source event, observed gap, and expanded outcome rows. |
| `apps/web/dist/css/pages/strategies.css` | modify | Keep the added operational fields dense and responsive without overlap. |
| `apps/web/locales/en.json` | modify | Add display labels for the new `/strategies` status/journal fields. |
| `apps/web/locales/ru.json` | modify | Add Russian display labels for the new `/strategies` status/journal fields. |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | modify | Lock the additive DTO shape and degraded/ready status semantics. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/10-strategy-ui-status-journal.md` | create | Stage report and evidence log. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage handoff and acceptance status. |
| `docs/architecture/README.md` | check/update if generated index changes | Required docs index gate after Markdown change. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/10-strategy-ui-status-journal.md` | none | none | Stage `10` report and required pre-start/scope record. | none: documentation artifact only. |
| none | `apps/api/dto/ui_strategies_dashboard.py`; `apps/api/wiring/modules/ui_strategies_dashboard.py`; `src/trading/contexts/live_execution/domain/execution_source.py`; `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/execution_intent_repository.py`; `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/execution_intent_repository.py` | none | Add runtime status and richer status/journal read model fields from existing ledgers. | compatible-change: additive response/read-model fields, no persisted schema change. |
| none | `apps/web/templates/pages/strategies.html`; `apps/web/dist/js/pages/strategies.js`; `apps/web/dist/css/pages/strategies.css`; `apps/web/locales/en.json`; `apps/web/locales/ru.json` | none | Render work-focused `/strategies` status/journal, manual controls, and responsive dense layout. | compatible-change: browser-visible additive fields and columns. |
| none | `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | none | Focused DTO/read-model regression coverage. | none: test-only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; `docs/architecture/README.md` | none | Stage handoff/index after validation. | none: documentation handoff. |

Files outside prompt expected paths: CSS and locale files are necessary support for the visible `/strategies` UI changes; the live-execution read-model files are necessary to expose fill/reconciliation/source-event timing from existing execution ledgers without frontend guessing. No storage migration, secret material, or raw provider payload surface is planned.

## Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | compatible-change | `GET /api/ui/strategies/dashboard` gains additive response fields only. |
| Port contract | compatible-change | Internal `ExecutionProducerOutcomeLink` read model gains optional fields for existing ledger data. |
| DTO schema | compatible-change | Additive Pydantic response models/fields. |
| Persisted schema | none | Existing execution source/order/fill/reconciliation tables are queried; no migration. |
| Config/env schema | none | No config or default behavior change. |
| Request hash/cache/persistence identity | none | No cache key, idempotency, or persisted identity change. |
| Service-call auth/timeout/retry/error semantics | none | Existing authenticated dashboard route and repository calls remain unchanged. |
| External side-effect semantics | none | Manual entry/exit endpoints remain the existing idempotent action path; dashboard is read-only. |
| Logs/metrics/traces/audit/redaction | compatible-change | Stage docs/browser evidence must stay sanitized; no new logs or metrics planned. |
| Alerts/runbooks | none | No alert/runbook contract change. |
| Browser-visible behavior | compatible-change | Adds explicit environment, producer, latency, fill/reconciliation and status columns; blocked/unknown states remain visible. |

## Validation Log

| Check | Result | Evidence |
|---|---:|---|
| `node --check apps/web/dist/js/pages/strategies.js` | passed | Completed locally with no syntax errors. |
| `uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | passed | `5 passed`; focused additive dashboard DTO/read-model coverage. |
| `uv run ruff check apps tests` | passed | `All checks passed!`. |
| `uv run pyright apps tests` | passed | `0 errors, 0 warnings, 0 informations`; pyright emitted only its version notice. |
| `uv run ruff check .` | passed | Reran after CI type-check repair; `All checks passed!`. |
| `uv run pyright` | passed | Reran after CI type-check repair; `0 errors, 0 warnings, 0 informations`. |
| `uv run pytest -q tests/unit/apps` | passed | `328 passed, 3 warnings`; warnings are existing `httpx` deprecation warnings in web tests. |
| `python -m tools.docs.generate_docs_index --check` | passed | Initial check found the new report missing from the generated index; after `python -m tools.docs.generate_docs_index`, the check passed with `docs/architecture/README.md is up-to-date.` |
| Browser desktop/mobile `/strategies` QA | passed locally | Pinned Playwright CLI against a mocked authenticated SSR/API fixture. Paper and testnet strategy states loaded, runtime status and execution journal fields were visible, and mobile body width stayed `390/390`. |
| Network/console checks | passed locally | Browser run recorded `consoleErrors=[]`, `failedRequests=[]`, `badResponses=[]` for the local fixture page. |
| DOM/screenshot secret scan | passed locally | Rendered DOM scan after removing script/style/template payloads found `hitCount=0` for session, authorization, bearer, API key, secret, and passphrase value patterns. |
| GitHub CI | passed | Initial CI `27847946668` failed full-repo pyright on `_int_or_none(object)` in the new outcome-link mapper. Repaired in `cdde6cbd`; final CI `27848087142` passed. |
| Deploy/runtime smoke | passed | Deploy Backend `27848263337` succeeded; Mac Studio checkout synced to `cdde6cbdcf90cd4a0d7ae48abcfdf37ab7a7d1ad`; `/opt/roehub/app` smoke passed. |
| Production `/strategies` browser/API QA | passed | Pinned Playwright CLI against `https://roehub.com/strategies` with a temporary DB session for owner `a102f64d-...`; paper/testnet states loaded, dashboard API returned `200`, console/network errors were empty, mobile width stayed `390/390`, and rendered DOM secret scan had `hitCount=0`. |
| Smoke session cleanup | passed | Revoked the temporary proof sessions for owner `a102f64d-...`; final `fresh_active_sessions=0` for sessions created in the proof window. |

## Runtime / Browser Evidence

Local fixture browser proof used `http://127.0.0.1:18110/strategies` with authenticated web dependencies mocked in-process and a `/api/ui/strategies/dashboard` fixture containing:

- paper strategy: `paper` environment, Binance spot BTCUSDT, `$50` allocation, ready paper accounting, source event, filled execution, fill count, matched reconciliation, and observed gap;
- testnet strategy: `testnet` environment, Bybit futures BTCUSDT, `$50` allocation, account-config guard, cancelled execution, reconciliation state, and observed gap.

Screenshots:

| Artifact | Evidence |
|---|---|
| `output/playwright/stage10-local/stage10-strategies-desktop-local.png` | Desktop paper `/strategies` state. |
| `output/playwright/stage10-local/stage10-strategies-testnet-local.png` | Desktop selected testnet `/strategies?strategy_id=...` state. |
| `output/playwright/stage10-local/stage10-strategies-mobile-local.png` | Mobile paper `/strategies` state; runtime panel remains in flow and execution outcomes stay inside the table wrapper. |

Production browser/API proof used `https://roehub.com/strategies` after current-image deploy. The run used the pinned Playwright helper `~/.codex/skills/playwright/scripts/playwright_cli.sh` plus a sanitized direct DB-session bootstrap from host-local production DB credentials for the proof owner, not the standard Keycloak smoke path; cleanup revoked proof-window sessions and rechecked `fresh_active_sessions=0`.

| Artifact / check | Evidence |
|---|---|
| `output/playwright/stage10-prod/stage10-strategies-paper-prod.png` | Desktop paper strategy `52cab273-7c88-4549-865a-b853b1bffa28`; API summary: `environment=paper`, `producer_status=stopped`, `mainnet_available=false`, `execution_outcomes=2`, first gap status `observed`. |
| `output/playwright/stage10-prod/stage10-strategies-testnet-prod.png` | Desktop testnet strategy `2fc641c6-da50-465f-b9b6-2319d5962429`; API summary: `environment=testnet`, `producer_status=blocked`, `mainnet_available=false`; visible blocker `exchange_connection_not_found`. |
| `output/playwright/stage10-prod/stage10-strategies-mobile-prod.png` | Mobile paper strategy; runtime panel remains in flow, execution outcomes stay wrapped, body width `390/390`. |
| `output/playwright/stage10-prod/stage10-runtime-paper-prod.png`; `output/playwright/stage10-prod/stage10-runtime-testnet-prod.png` | Focused production runtime-panel captures for paper and testnet states. |
| API/network/console/DOM scan | `paperApi.status=200`, `testnetApi.status=200`, `consoleErrors=[]`, `failedRequests=[]`, `badResponses=[]`, secret scan `hitCount=0`. |

## Publish / Deploy

Direct `main` delivery; no temporary branch or PR was used.

| Step | Evidence |
|---|---|
| `gh --version` / auth | `gh version 2.85.0`; silent `gh auth status` exit check returned authenticated. |
| Scoped staging | Explicit path staging only for Stage `10` code/docs/tests; no output artifacts staged. |
| Commits | `7d32f68a2f4f78c0c870cf9cf0a359d24fbfd72f` (`Add strategy UI status journal`); `cdde6cbdcf90cd4a0d7ae48abcfdf37ab7a7d1ad` (`Fix strategy outcome link typing`). |
| `origin/main` | `refs/heads/main` resolved to `cdde6cbdcf90cd4a0d7ae48abcfdf37ab7a7d1ad`. |
| CI | `27847946668` failed full-repo pyright after the implementation commit; `cdde6cbd` repaired the type helper; final CI `27848087142` succeeded. |
| Backend/runtime deploy | Deploy Backend `27848263337` succeeded; Publish App Image `27848263331` and Deploy Web `27848263357`/`27848269732` were skipped by changed-file routing on the type-only repair commit. |
| Web image deploy | Manual `Publish App Image` `27848576348` built the current `cdde6cbd` image; manual `Deploy Web` `27848755965` deployed image tag `cdde6cbdcf90cd4a0d7ae48abcfdf37ab7a7d1ad` after route detection would otherwise skip web deploy. |
| Mac Studio checkout/smoke | Mac Studio checkout initially had two unrelated RL files already equivalent to `origin/main`; they were preserved in stash `codex-preserve-rl-stage04b-before-stage10-sync-20260619T210257Z`, then checkout fast-forwarded to `cdde6cbd`. `/opt/roehub/app` file parity for Stage `10` files matched the checkout, and `bash scripts/macos/smoke_prod.sh` exited `0`. |
| Branch cleanup | `N/A`; no temporary local or remote branch was created. |

## Handoff

Stage `10` is accepted. Stage `11` may start.

Next executor should keep the following boundaries:

- Stage `10` adds dashboard/UI read-model fields only; no persisted schema, config, auth, idempotency, or side-effect contract changed.
- Mainnet remains unavailable and is rendered as unavailable in the runtime panel.
- Testnet proof for strategy `2fc641c6-da50-465f-b9b6-2319d5962429` is intentionally blocked with `exchange_connection_not_found`; this is visible state, not styled as success.
- The manual image/web dispatch was required because the CI repair commit was type-only and route detection skipped web image rebuild/deploy; future UI stages should watch for the same two-commit pattern.
- Mac Studio has a preserved stash `codex-preserve-rl-stage04b-before-stage10-sync-20260619T210257Z` containing unrelated RL files that were already equivalent to `origin/main` before checkout sync.
