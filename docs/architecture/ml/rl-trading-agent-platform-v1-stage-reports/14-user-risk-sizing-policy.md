---
doc: rl-trading-agent-platform-v1-stage-14-user-risk-sizing-policy
status: accepted
stage: 14
updated_at: 2026-07-03
---

# Stage 14: User Risk/Sizing Policy

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## 1. Result and stage status

Stage `14` implemented a local, additive RL user risk/sizing policy surface for `rl-trading-agent-platform-v1`.

What changed:

- Added owner/strategy/exchange/market/symbol-scoped RL risk policy validation, persistence and audit.
- Added additive `GET/PUT /strategies/{strategy_id}/rl-risk-policy` API surface.
- Added dashboard `/strategies` RL/ML risk configuration fields: sizing, loss/drawdown, confidence, validation reasons and synthetic exits.
- Represented TP/SL/trailing as platform-side synthetic exit rules that later create `close` intents.
- Preserved Stage `13` monitor-only behavior: monitor-only decisions remain `no_intent`, and this stage does not create paper/testnet/live/mainnet orders.

Prerequisite check:

| Required prerequisite | Observed ledger status | Result |
|---|---|---|
| Stage `13` Monitor-only inference producer | `accepted` | `passed` |

Prompt evidence:

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/14-user-risk-sizing-policy.md` |
| Prompt sha256 | `c7e35a094740138454cde52eb5008ade69a5a8c5ed77b7d3eac1ca44e4063f9a` |

The optional `.codex/agents/.context/promt_manager_state.yaml` still described the original Stage `02A` prompt-pack state and stale `github:yeet` delivery wording. It was read and ignored for execution facts because the live ledger and current repo instructions supersede it.

## 2. File manifest

### Created

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/risk_sizing_policy.py` | Stage `14` policy key/config/record, fail-closed validation, synthetic exits and decision preview helpers. | `compatible-change` additive domain contract |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/in_memory_risk_sizing_policy.py` | In-memory policy repository and audit event storage for tests/local wiring. | `compatible-change` additive adapter |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/postgres_risk_sizing_policy.py` | Postgres policy repository with scoped upsert and audit event write. | `compatible-change` additive adapter |
| `alembic/versions/20260703_0042_rl_risk_sizing_policy_v1.py` | Additive `rl_risk_sizing_policies` and `rl_risk_sizing_policy_audit_events` tables. | `compatible-change` additive persisted schema |
| `tests/unit/contexts/rl_trading/domain/test_risk_sizing_policy.py` | Domain validation, no-intent and synthetic-exit coverage. | `none` tests |
| `tests/unit/contexts/rl_trading/adapters/test_risk_sizing_policy_repositories.py` | Repository persistence/audit and fail-closed default coverage. | `none` tests |
| `tests/unit/apps/migrations/test_rl_risk_sizing_policy_sql.py` | Migration contract coverage for scoped policy and audit schema. | `none` tests |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/14-user-risk-sizing-policy.md` | This accepted stage report. | `compatible-change` docs |

### Modified

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/__init__.py` | Export additive Stage `14` policy types/services. | `compatible-change` additive Python export |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/__init__.py` | Export additive policy repositories. | `compatible-change` additive Python export |
| `apps/api/routes/strategies.py` | Add risk-policy DTOs/routes and apply saved-policy readiness blocking for non-monitor activation. | `compatible-change` additive API route; fail-closed activation behavior when a saved invalid policy exists |
| `apps/api/wiring/modules/strategy.py` | Wire `RlRiskSizingPolicyService` into strategy routes. | `compatible-change` internal wiring |
| `apps/api/dto/ui_strategies_dashboard.py` | Add dashboard risk config fields and synthetic-exit DTO. | `compatible-change` additive DTO fields |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | Load policy state into the RL/ML dashboard read model and fail closed when unavailable/missing. | `compatible-change` additive read model |
| `apps/web/templates/pages/strategies.html` | Add `/strategies` RL/ML risk field hooks. | `compatible-change` browser-visible additive UI |
| `apps/web/dist/js/pages/strategies.js` | Render risk sizing, thresholds, validation reasons and synthetic exits from backend state. | `compatible-change` browser-visible additive UI |
| `apps/web/dist/css/pages/strategies.css` | Keep RL/ML risk rows/headings responsive without clipping long policy identifiers. | `compatible-change` browser-visible styling |
| `apps/web/locales/en.json` | Add English labels for risk fields. | `compatible-change` additive locale keys |
| `apps/web/locales/ru.json` | Add Russian labels for risk fields. | `compatible-change` additive locale keys |
| `tests/unit/apps/api/test_strategies_routes.py` | Cover policy API persistence and invalid-policy activation block. | `none` tests |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Cover dashboard-ready risk config, validation reasons, notes and synthetic exits. | `none` tests |
| `tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | Cover risk UI hooks and rendering assets. | `none` tests |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `14` accepted and record handoff evidence. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Mechanical docs index regeneration after adding the Stage `14` report. | `compatible-change` docs index |

### Deleted

| Path | Reason | Contract impact |
|---|---|---|
| - | No files deleted. | `none` |

### Outside expected paths

| Path | Justification |
|---|---|
| `apps/api/wiring/modules/strategy.py` | Required production/test app construction dependency for the new route service. |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | `/strategies` reads the RL/ML risk config through this dashboard service. |
| `apps/web/dist/js/pages/strategies.js` | Existing `/strategies` page renders backend dashboard JSON through this asset. |
| `apps/web/dist/css/pages/strategies.css` | Browser QA required responsive risk rows without text clipping/overflow. |
| `apps/web/locales/en.json` | Existing template labels are locale-backed. |
| `apps/web/locales/ru.json` | Existing template labels are locale-backed. |

Ignored local artifacts:

| Path | State |
|---|---|
| `output/playwright/stage14-risk-desktop.png` | Ignored browser QA screenshot. |
| `output/playwright/stage14-risk-mobile.png` | Ignored browser QA screenshot. |

## 3. Evidence and quality gates

Focused checks:

| Command | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_risk_sizing_policy.py tests/unit/contexts/rl_trading/adapters/test_risk_sizing_policy_repositories.py tests/unit/apps/migrations/test_rl_risk_sizing_policy_sql.py tests/unit/apps/api/test_strategies_routes.py::test_stage14_rl_risk_policy_api_persists_valid_policy_and_synthetic_exits tests/unit/apps/api/test_strategies_routes.py::test_stage14_invalid_saved_policy_blocks_activation_without_exchange_submit tests/unit/apps/api/test_ui_strategy_dashboard_routes.py::test_strategy_dashboard_exposes_reference_panel_inventory_and_degraded_stats tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | `14 passed` |
| `uv run pytest -q tests/unit/apps/migrations/test_rl_risk_sizing_policy_sql.py tests/unit/contexts/rl_trading/adapters/test_risk_sizing_policy_repositories.py` | `3 passed` after scoped DB upsert/index repair |
| `uv run pytest -q tests/unit/apps/web/test_strategies_rl_ml_tab_asset.py` | `3 passed` after responsive CSS repair |

Prompt-level gates:

| Command | Result |
|---|---|
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | `0 errors, 0 warnings, 0 informations` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `490 passed, 8 skipped, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after `docs/architecture/README.md` regeneration |

Pytest skipped the existing optional Torch tests because the default environment does not include `torch`; this is the same optional ML dependency boundary used by prior stages.

Browser evidence:

| Evidence | Result |
|---|---|
| Local harness URL | `http://127.0.0.1:64214/strategies?strategy_id=00000000-0000-0000-0000-000000006101` |
| Harness boundary | `local_browser_runtime_harness`; mocked current-user and same-origin `/api/ui/strategies/dashboard` response generated by the real Stage `14` dashboard service. No production credentials, no exchange path, no paper/testnet/live/mainnet submit. |
| Dashboard request | `[GET] /api/ui/strategies/dashboard?... => [200] OK` |
| Desktop viewport | `1440x1000`, required Stage `14` risk values rendered, `docScrollWidth=1440`, `bodyScrollWidth=1440`, console `0` errors/warnings. |
| Mobile viewport | `390x844`, required Stage `14` risk values rendered, `docScrollWidth=390`, `bodyScrollWidth=390`, console `0` errors/warnings. |
| Screenshots | `output/playwright/stage14-risk-desktop.png`, `output/playwright/stage14-risk-mobile.png` |

Synthetic exit and no-submit proof:

| Surface | Evidence |
|---|---|
| Synthetic exits | Valid policy emits `take_profit`, `stop_loss`, `trailing_stop` with `platform_side=true` and `creates_intent_action=close`; dashboard renders `take_profit:5.00%:close / stop_loss:2.00%:close / trailing_stop:3.00%:close`. |
| Monitor-only decisions | Domain test verifies monitor-only returns `no_intent` with `monitor_only_no_intent`. |
| Invalid policy blocking | API test verifies a saved invalid policy blocks non-monitor activation with `rl_risk_policy_stop_loss_required`. |
| Exchange side effects | No exchange SDK, execution submit, paper order, testnet order, live order or Redis dispatch path was added or called. Dashboard notes include `stage14_no_exchange_submit`. |

## 4. Contract, safety, and delivery state

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `compatible-change` | Additive `GET/PUT /strategies/{strategy_id}/rl-risk-policy`; existing strategy APIs remain compatible. |
| DTO schema | `compatible-change` | Additive dashboard fields and synthetic-exit DTO. |
| Port/domain contract | `compatible-change` | Additive RL policy service/repository protocol and exports. |
| Persisted schema | `compatible-change` | Additive policy/audit tables only; no existing table is changed. |
| Config/defaults | `none` | No runtime config default changed. |
| Cache/request identity | `none` | No persisted cache key or request hash changed. |
| External side effects | `none` | No exchange submit, paper order, testnet order, live order, provider call or Redis dispatch side effect is introduced. |
| Browser-visible behavior | `compatible-change` | Additive RL/ML risk config rows on `/strategies`. |
| Performance | `none` for performance claims | No performance-sensitive path was changed and no throughput/latency claim is made. |

Safety notes:

- Policy is fail-closed when missing/invalid in the dashboard surface.
- A missing saved policy does not silently break existing classic live-profile flows; saved invalid RL policy blocks non-monitor activation.
- Platform synthetic exits are not native exchange OCO/TP/SL/trailing fields.
- Policy persistence is owner/strategy/exchange/market/symbol scoped and audited.
- Stage `14` reached `post_main_production_runtime_proof` only after direct `main` delivery, green GitHub workflows, `macstudio` checkout sync and production smoke. This proof does not approve paper/testnet/live/mainnet execution.

Delivery state: delivered to `main`.

Post-main delivery evidence:

| Surface | Evidence |
|---|---|
| Implementation commit | `e086eb9f7721fd83768885bd2e23dd16075389b8` on `main` / `origin/main`. |
| GitHub Actions | `CI` `28681177872` passed; `Deploy Backend` `28681255514` passed; `Publish App Image` `28681255516` passed; `Deploy Web` `28681255533` and follow-up `Deploy Web` `28681289201` passed. |
| `macstudio` checkout | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `e086eb9f7721fd83768885bd2e23dd16075389b8` and reported `## main...origin/main`. |
| Runtime tree parity | `/opt/roehub/app` matched the `macstudio` checkout for `risk_sizing_policy.py`, `postgres_risk_sizing_policy.py`, Alembic migration `20260703_0042_rl_risk_sizing_policy_v1.py`, `apps/api/routes/strategies.py`, and `apps/web/dist/js/pages/strategies.js`. Stage report docs are present in the checkout; they are not part of the runtime bundle. |
| Production smoke | `bash scripts/macos/smoke_prod.sh` passed from `/opt/roehub/app`. |
| Public edge auth smoke | Local public-edge curl flow using host-local `ROEHUB_SMOKE_E2E_PASSWORD` reached `final_path=https://roehub.com/strategies`, `current_user_status=200`, `strategies_status=200`, `dashboard_http=200`, `strategies_js_http=200`, Stage `14` HTML hooks present, `logout_status=204`, `after_logout_status=401`. The password, cookies and callback body were not printed or stored in repo artifacts. |
| Mac Studio public-edge caveat | The same optional auth flow from `macstudio` to `https://roehub.com:443` timed out; this was not used as delivery acceptance because GitHub `Deploy Web` edge smoke and the local public-edge auth smoke passed. |

## 5. Blockers and next-stage handoff

Stage `14` blockers: none for local acceptance.

Next-stage state:

| Stage | Allowed now | Reason |
|---|---|---|
| `15` Paper RL integration | `yes` | Prerequisite refresh on 2026-07-03 verified the classic strategy-producer ledger is completed and classic Stage `05`, `07`, and `09` are accepted. Stage `15` may start, but it has not been implemented or accepted yet. |
| `16` Testnet RL integration | `no` | Classic Stage `09` is accepted, but RL testnet remains blocked until Stage `15` paper integration is accepted. |
| Mainnet | `no` | Mainnet remains blocked until Stage `19` approval and Stage `20` prompt conditions. |

Handoff for Stage `15` now that prerequisites are open:

- Reuse `RlRiskSizingPolicyService` and saved policy records; do not duplicate sizing/risk rules in the paper executor.
- Convert synthetic exits into later platform `close` intents through the existing execution boundary only.
- Preserve the owner/strategy/exchange/market/symbol key as the audit and idempotency boundary.
- Treat `stage14_no_exchange_submit` as a safety invariant: Stage `15` may own paper behavior, but Stage `14` did not approve any exchange-side behavior.
