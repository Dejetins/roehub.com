# Stage 10: Strategy UI Status And Journal

Статус: `in_progress`.

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
| `uv run pytest -q tests/unit/apps` | passed | `328 passed, 3 warnings`; warnings are existing `httpx` deprecation warnings in web tests. |
| `python -m tools.docs.generate_docs_index --check` | passed | Initial check found the new report missing from the generated index; after `python -m tools.docs.generate_docs_index`, the check passed with `docs/architecture/README.md is up-to-date.` |
| Browser desktop/mobile `/strategies` QA | passed locally | Pinned Playwright CLI against a mocked authenticated SSR/API fixture. Paper and testnet strategy states loaded, runtime status and execution journal fields were visible, and mobile body width stayed `390/390`. |
| Network/console checks | passed locally | Browser run recorded `consoleErrors=[]`, `failedRequests=[]`, `badResponses=[]` for the local fixture page. |
| DOM/screenshot secret scan | passed locally | Rendered DOM scan after removing script/style/template payloads found `hitCount=0` for session, authorization, bearer, API key, secret, and passphrase value patterns. |

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

## Publish / Deploy

Pending. Stage must not be marked `accepted` until validated changes are delivered to `main`, `origin/main` contains the changes, Mac Studio sync/deploy smoke passes for code/runtime changes, and branch cleanup is recorded if a temporary branch is used.

## Handoff

Local implementation and validation are complete. Acceptance remains blocked only on publish/deploy/runtime proof:

- commit validated code/docs to `main`;
- push `origin main` with scoped staging only;
- wait for CI and deploy workflows;
- verify Mac Studio checkout/runtime sync and production smoke;
- run production browser/API evidence for `/strategies`;
- update this report and the stage ledger before final acceptance.
