---
doc: rl-trading-agent-platform-v1-stage-11-rl-tab-ui-skeleton
status: accepted
stage: 11
updated_at: 2026-07-02
---

# Stage 11: RL tab UI skeleton

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/11-rl-tab-ui-skeleton.md` |
| Prompt sha256 | `d25cff0c595bd8ff4f9805becf471e674338191f154abe4dae6b1dfa68790914` |
| Ledger state observed before work | `current_stage=11`; Stage `11` pending/current |
| Prerequisite verdict | accepted Stage `09B`; accepted Stage `10A`; Stage `11` may proceed |
| `.codex/agents/.context/promt_manager_state.yaml` | read; treated as stale prompt-generation state where it conflicts with current `.codex/AGENTS.md`, prompt and ledger |
| Browser/auth | browser QA required after implementation; smoke credentials must not be written into docs, traces, screenshots or logs |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, provider credential, or raw provider payload surface is in scope |

## Planned Concrete File List Before Edit

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | Static web asset/template contract for the RL/ML tab skeleton and disabled operator controls. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/11-rl-tab-ui-skeleton.md` | Stage report, file manifest, evidence and next-stage handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `apps/api/dto/ui_strategies_dashboard.py` | Add additive RL/ML tab DTOs and reusable `ml_agent_decision` source-event/outcome surface in the dashboard response. | `compatible-change` additive API/DTO response shape |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | Add fail-closed backend read model for model status, ticker slots, modes, risk placeholders, disabled operator controls and filtered `ml_agent_decision` outcomes. | `compatible-change` additive API/read-model behavior |
| `apps/web/templates/pages/strategies.html` | Add `/strategies` Classic/RL/ML tabs and RL/ML panel markup. | `compatible-change` browser-visible behavior |
| `apps/web/dist/js/pages/strategies.js` | Render RL/ML read model from backend state and keep operator controls disabled unless backend allows them. | `compatible-change` browser-visible behavior |
| `apps/web/dist/css/pages/strategies.css` | Add responsive layout and disabled-control styling for the RL/ML panel. | `compatible-change` browser-visible behavior |
| `apps/web/locales/en.json` | Add RL/ML tab copy. | `compatible-change` browser-visible copy |
| `apps/web/locales/ru.json` | Add RL/ML tab copy. | `compatible-change` browser-visible copy |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Prove additive API shape, blocked reasons, model/ticker/risk/mode surfaces, disabled operator controls, and `ml_agent_decision` outcome reuse. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `11` status/evidence and next-stage handoff after validation. | `compatible-change` docs/ledger |

Deleted: none planned.

Outside expected paths: none planned.

## Scope

Stage `11` добавил fail-closed RL/ML skeleton на существующую страницу `/strategies` без запуска model lifecycle side effects.

В scope вошло:

- Additive поле `rl_ml` в `GET /api/ui/strategies/dashboard`.
- Backend read model для model registry status, ticker slots, mode options, risk placeholder, disabled operator controls и `source_type=ml_agent_decision` outcomes.
- Browser-visible Classic/RL/ML tabbar и RL/ML tab panel.
- Disabled UI controls for retraining/rollback, driven only by backend state.
- Focused API and static web asset tests.

Вне scope осталось:

- retraining/rollback API endpoints or command execution;
- model promotion/activation mutation;
- production registry writes;
- paper/testnet/live/mainnet order intents;
- exchange SDK calls, provider credentials, raw provider payloads or new secret surfaces;
- Stage `12` entitlement mapping and server-side operator/admin authorization.

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisites | Ledger has `current_stage=11`; Stage `09B` and Stage `10A` are accepted. |
| Stage `10A` handoff | UI controls may expose retraining/rollback only as guarded lifecycle actions. No auto-promotion, no activation mutation and no exchange side effects are allowed. |
| Operator/admin guard | Current `CurrentUserPrincipal` has `user_id`, `paid_level` and optional `session_created_at`; no operator/admin guard primitive is present in the read-model path, so Stage `11` must render controls disabled/read-only. |
| Signal/outcome read model | Existing execution outcome DTO and service already carry a generic `source_type`; Stage `11` extends this reusable surface for `ml_agent_decision` instead of adding an RL-only journal. |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `compatible-change` | `GET /api/ui/strategies/dashboard` now returns additive required `rl_ml`; existing response fields are preserved. |
| Ports/internal service boundary | `compatible-change` | Dashboard query service composes additional read-model sections only; no external port signature or money-boundary call was added. |
| DTO schema | `compatible-change` | New additive RL/ML DTOs under `StrategyDashboardResponse.rl_ml`. |
| Persisted schema | `none` | No database migration or persistence write. |
| Config schema/defaults | `none` | No env/config default changed. |
| Request hashes/cache keys/identity | `none` | Existing dashboard route/query parameters and auth context are unchanged. |
| Service call auth/timeout/retry/error behavior | `none` | No new upstream service call; same dashboard route and current-user boundary. |
| External side effects/idempotency/unknown state | `none` | No exchange/provider call, order intent, training job, promotion, rollback, registry write or activation mutation. |
| Logs/metrics/traces/audit/report | `compatible-change` | Stage report/ledger evidence added; no secrets or raw provider payloads recorded. |
| Browser-visible behavior | `compatible-change` | `/strategies` adds Classic/RL/ML tabs; default remains Classic and RL/ML controls are disabled fail-closed. |
| Performance hot path | `none` | Dashboard read/render only; no model inference, ClickHouse scan or execution hot-path work was added. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index updated. |

## Quality Gates And Evidence

| Command / evidence | Result |
|---|---|
| `uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | `8 passed` after final CSS containment update. |
| `uv run ruff check apps/api/dto/ui_strategies_dashboard.py apps/api/wiring/modules/ui_strategies_dashboard.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | passed. |
| `uv run pyright apps/api/dto/ui_strategies_dashboard.py apps/api/wiring/modules/ui_strategies_dashboard.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | `0 errors, 0 warnings, 0 informations` with existing pyright version notice. |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed. |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | `0 errors, 0 warnings, 0 informations` with existing pyright version notice. |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `465 passed, 3 warnings`; warnings are existing `httpx` deprecation warnings in `tests/unit/apps/web/test_app_routes.py`. |
| Local browser harness at `http://127.0.0.1:64208/strategies` | Opened real `apps.web.main.app:create_app` page with sanitized mock dashboard JSON and fake current user; no credentials, provider payloads or exchange calls. |
| Playwright desktop proof | RL/ML tab activated; model status degraded/not_configured; artifact root `/opt/roehub/state/rl_trading/`; only `monitor_only` enabled; `paper`, `testnet`, `live` blocked; retraining/rollback buttons disabled; ticker slot row blocked with `stage12_entitlement_mapping_pending`; `ml_agent_decision` outcome table rendered. |
| Playwright desktop layout probe | `panelVisible=true`; operator controls disabled; `horizontalOverflow=0`; four RL cards rendered. |
| Playwright mobile probe at `390x844` after CSS containment fix | `bodyOverflow=0`, `panelOverflow=0`, `overflowingPills=[]`; wide RL tables use local horizontal scroll; retraining/rollback controls remain disabled. |
| Playwright console/network | `0` console errors/warnings; dashboard request returned `[200] OK`. |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after report/ledger/docs-index update. |

Browser proof boundary: `local_browser_runtime_harness`, not `post_main_production_runtime_proof`. The harness used mocked dashboard API payloads to verify browser-visible rendering and disabled controls only; it did not exercise production auth, production deploy, Mac Studio `/opt/roehub/app`, exchange connectivity or provider credentials.

## Review Mode

| Field | Value |
|---|---|
| Review mode | Cold self-review fallback. Independent review was not spawned for this local-only Stage `11` implementation. |
| Verdict | Accepted after fixing the browser-visible mobile overflow blocker found during Playwright QA. |
| Fixed blocker | Initial `390x844` probe showed page-level horizontal overflow from wide RL tables and long fail-closed status pills. CSS containment and scoped pill wrapping fixed it; final mobile probe showed `bodyOverflow=0`, `panelOverflow=0`, `overflowingPills=[]`. |
| Follow-up check | Re-ran focused tests, focused ruff, prompt-level ruff/pyright/pytest, docs index check, Playwright console/network/layout probes and `git diff --check`. |
| Residual risks | Stage acceptance evidence is local browser harness evidence, not production deploy/Mac Studio `/opt/roehub/app` proof. A later publish/deploy run must record its own `post_main_production_runtime_proof`. Stage `12` still owns entitlement mapping and operator/admin guard state before any RL/ML control can become enabled. |

## File Manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | Static contract for template hooks, backend-state-driven disabled controls, and responsive CSS containment. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/11-rl-tab-ui-skeleton.md` | Stage `11` report, evidence, file manifest and handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `apps/api/dto/ui_strategies_dashboard.py` | Add RL/ML dashboard DTOs and required `rl_ml` field on `StrategyDashboardResponse`. | `compatible-change` additive API/DTO response |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | Build fail-closed RL/ML read model, mode/risk/ticker/operator states and filtered `ml_agent_decision` outcomes. | `compatible-change` additive read-model behavior |
| `apps/web/templates/pages/strategies.html` | Add Classic/RL/ML tabbar and RL/ML panel markup. | `compatible-change` browser-visible behavior |
| `apps/web/dist/js/pages/strategies.js` | Render RL/ML state from backend response and keep operator actions disabled unless backend enables them. | `compatible-change` browser-visible behavior |
| `apps/web/dist/css/pages/strategies.css` | Add RL/ML responsive layout, disabled-control styling and mobile overflow containment. | `compatible-change` browser-visible behavior |
| `apps/web/locales/en.json` | Add English RL/ML tab copy. | `compatible-change` browser-visible copy |
| `apps/web/locales/ru.json` | Add Russian RL/ML tab copy. | `compatible-change` browser-visible copy |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Prove additive API shape, blocked reasons, disabled controls and reusable `ml_agent_decision` outcome filtering. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `11` accepted and open Stage `12`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index regenerated after adding Stage `11` report. | `compatible-change` docs index |

Deleted: none.

Outside expected path note: the prompt named `apps/web/static`, but this repo's established built asset paths for the page are `apps/web/dist/js/pages/strategies.js` and `apps/web/dist/css/pages/strategies.css`; locale copy lives in the existing `apps/web/locales/*.json` catalogs. No unrelated paths were changed.

## Blockers

No Stage `11` blocker remains.

Fail-closed runtime blockers intentionally exposed in UI:

- `rl_model_registry_read_model_not_configured`;
- `stage12_entitlement_mapping_pending`;
- `rl_execution_modes_pending_later_stage_gates`;
- `operator_admin_guard_not_available`.

These are not Stage `11` blockers because the requested skeleton renders them read-only/disabled; they are handoff constraints for later stages.

## Next-Stage Handoff

`current_stage` moves to Stage `12` Backend entitlements.

Stage `12` must map current identity plan codes `base|free|pro|ultra` to RL product labels/entitlements and keep ambiguous `base` fail-closed until product evidence says otherwise. Stage `12` should also provide the server-side entitlement and operator/admin guard state needed before any UI control can become enabled.

Stage `13` monitor-only inference remains blocked until Stage `12` is accepted. Stage `15`/`16` execution stages remain blocked by the classic strategy-producer paper/testnet prerequisites. Stage `11` acceptance did not write registry state, enable exchange execution or claim production runtime proof; any later `main` publish/deploy evidence is a separate `publish-ci-deploy` delivery record.
