---
doc: rl-trading-agent-platform-v1-stage-12-backend-entitlements
status: accepted
stage: 12
updated_at: 2026-07-03
---

# Stage 12: Backend Entitlements

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/12-backend-entitlements.md` |
| Prompt sha256 | `e82eee4a2a9666b7313fb24661a8542d1028f63ca178391d6deaf06c09aa50e8` |
| Ledger state observed before work | `current_stage=12`; Stage `11` accepted; Stage `12` pending/current |
| Prerequisite verdict | accepted Stage `11`; Stage `12` may proceed |
| `.codex/agents/.context/promt_manager_state.yaml` | read; ignored as stale prompt-generation state because it still points to Stage `02A` and branch/PR delivery, while current `.codex/AGENTS.md`, prompt and ledger require `main`/local-only execution unless explicitly published |
| Browser/auth | browser QA required after implementation; smoke credentials must not be written into docs, traces, screenshots or logs |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, provider credential, or raw provider payload surface is in scope |

## Planned Concrete File List Before Edit

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/live_entitlements.py` | Shared backend contract for RL live ticker paid-level limits, override semantics, slot counting and stable blocked reasons. | `compatible-change` new internal RL domain contract |
| `src/trading/contexts/rl_trading/adapters/__init__.py` | Adapter package marker for RL entitlement persistence implementations. | `none` packaging only |
| `src/trading/contexts/rl_trading/adapters/outbound/__init__.py` | Outbound adapter package marker. | `none` packaging only |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/__init__.py` | Export in-memory and Postgres RL entitlement repositories. | `compatible-change` internal import surface |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/in_memory_live_entitlements.py` | Deterministic in-memory repository for tests/dev fallback. | `compatible-change` internal adapter |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/postgres_live_entitlements.py` | Postgres repository for owner-scoped override and activation checks. | `compatible-change` internal adapter |
| `alembic/versions/20260703_0041_rl_live_ticker_entitlements_v1.py` | Add optional RL override table and active live ticker activation table with partial uniqueness for active `(owner_user_id, exchange_name, market_type, symbol)`. | `compatible-change` additive persisted schema |
| `tests/unit/contexts/rl_trading/domain/test_live_entitlements.py` | Deterministic paid-level, override, mode-counting and producer/API agreement tests. | `none` test-only |
| `tests/unit/contexts/rl_trading/domain/test_live_entitlement_repositories.py` | Repository-level tests for active live slot counting, deactivation/replacement and override behavior. | `none` test-only |
| `tests/unit/apps/migrations/test_rl_live_ticker_entitlements_sql.py` | Migration contract test for additive override/activation tables and active-ticker uniqueness. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/12-backend-entitlements.md` | Stage report, evidence, file manifest and next-stage handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `apps/api/dto/ui_strategies_dashboard.py` | Extend RL ticker slot DTO with stable entitlement source. | `compatible-change` additive API/DTO response semantics |
| `apps/api/routes/strategies.py` | Add RL live ticker entitlement check to live-profile create/update/readiness paths without moving exchange submission or secret custody out of existing contexts. | `compatible-change` additive readiness blocking behavior for RL live quota path |
| `apps/api/wiring/modules/strategy.py` | Wire the RL entitlement service/repository in API composition. | `compatible-change` additive wiring |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | Replace Stage `11` hardcoded `stage12_entitlement_mapping_pending` with real paid-level/override limit, used count and stable blocked reasons. | `compatible-change` additive dashboard read-model behavior |
| `tests/unit/apps/api/test_strategies_routes.py` | Prove API live-profile entitlement blocking, mode-specific non-counting and slot replacement behavior. | `none` test-only |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Prove dashboard paid-level limit/used/blocked reason response. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `12` status/evidence and next-stage handoff after validation. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index update after adding Stage `12` report, if the generator reports drift. | `compatible-change` docs index |

Deleted: none planned.

Outside expected paths: `docs/architecture/README.md` only if generated by the docs-index gate; no frontend source edit was planned after verifying the existing Stage `11` renderer already displays backend slot counts and reasons.

## Scope

Stage `12` добавил fail-closed backend entitlement contract для RL live ticker slots.

В scope вошло:

- mapping identity plan codes to RL entitlement limits: `free=1`, `pro=5`, `ultra=20` with product label `Premium`;
- explicit fail-closed handling for `base` and unknown paid levels;
- additive per-owner override contract with product label `Enterprise`;
- counting only `mode=live` RL tickers against quota;
- API live-profile readiness blocking with stable reasons when a user has no live slots left;
- dashboard read-model values for `paid_level`, `product_label`, `entitlement_source`, `live_slots_used`, `live_slots_allowed` and blocked reason;
- additive Postgres schema for overrides and active ticker activations;
- local browser evidence that the existing `/strategies` RL/ML tab renders backend quota state.

Вне scope осталось:

- server-side operator/admin action enablement beyond entitlement state;
- monitor-only inference, `ml_agent_decision` production, Redis/live feature parity and runtime metrics, which remain Stage `13`;
- user risk/sizing and synthetic exits, which remain Stage `14`;
- paper/testnet/live/mainnet order execution;
- production DB apply, Mac Studio `/opt/roehub/app` deployment, CI delivery, GitHub publication or `post_main_production_runtime_proof`;
- exchange SDK calls, provider credentials, raw provider payloads, signed requests or new secret surfaces.

## Business-Readable Summary

Для пользователя это означает, что RL live ticker теперь ограничен понятной квотой по тарифу: Free - один live ticker, Pro - пять, Premium - двадцать. Enterprise может получить отдельный override. Если тариф внутренний `base`, неизвестный или entitlement storage недоступен, backend блокирует live activation и возвращает стабильную причину, а не разрешает торговый live режим по умолчанию.

Это не включает запуск модели, сигналы, paper/testnet/live ордера или mainnet. Stage `12` только решает вопрос: имеет ли пользователь право занять RL live ticker slot, и показывает это состояние в API/dashboard.

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisites | Ledger had `current_stage=12`; Stage `11` was accepted and unlocked only Stage `12`. |
| Identity source | API and dashboard use `CurrentUserPrincipal.paid_level` from the Roehub current-user/session boundary. Browser/Keycloak claims are not consumed directly for entitlement decisions. |
| Stage `11` UI state | Existing RL/ML panel already renders backend `rl_ml.ticker_slots` used/allowed counts and reasons; no frontend source change was needed. |
| Existing live-profile boundary | `strategy_live_profiles` remains the generic strategy profile table; Stage `12` adds readiness blocking and separate RL activation accounting without changing exchange/order submit boundaries. |
| Classic producer dependency | RL paper/testnet integration remains blocked by classic producer Stage `07`/`09` prerequisites; Stage `12` does not change that dependency. |

## Implemented Entitlement Contract

| Input / state | Result |
|---|---|
| `paid_level=free` | Product label `Free`; `live_slots_allowed=1`. |
| `paid_level=pro` | Product label `Pro`; `live_slots_allowed=5`. |
| `paid_level=ultra` | Product label `Premium`; `live_slots_allowed=20`. |
| `paid_level=base` | Product label `internal/base`; fail-closed reason `rl_live_ticker_paid_level_base_fail_closed`; `live_slots_allowed=0`. |
| Unknown/blank paid level | fail-closed reason `rl_live_ticker_paid_level_unknown_fail_closed`; `live_slots_allowed=0`. |
| Owner override | Product label `Enterprise`; override slot limit; `entitlement_source=override`. |
| `monitor_only`, `paper`, `testnet` profiles | Do not consume a live slot; readiness reason `rl_live_ticker_not_counted_for_mode` for entitlement accounting. |
| Eligible live profile | readiness reason `rl_live_ticker_entitlement_ready`; active live ticker activation is upserted. |
| Quota exceeded | profile readiness is persisted as `blocked` with reason `rl_live_ticker_quota_exceeded`. |
| Entitlement repository unavailable | API live checks fail closed with `rl_live_ticker_entitlement_repository_unavailable`. |
| Current principal unavailable | API live checks fail closed with `rl_live_ticker_current_principal_unavailable`. |

Business meaning: a Free user can run one active RL live ticker, Pro can run five, Premium can run twenty, and an Enterprise override can grant a custom limit. Ambiguous internal/base users and unknown plans cannot accidentally activate live RL tickers.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `compatible-change` | Live-profile create/update/readiness can now return an existing readiness-blocked response for RL live quota cases; dashboard response adds `entitlement_source` inside `rl_ml.ticker_slots`. |
| Ports/internal service boundary | `compatible-change` | New RL entitlement domain service and repositories are additive. Stage `13` can reuse `evaluate_rl_live_ticker_entitlement()` to re-check producer decisions. |
| DTO schema | `compatible-change` | Additive `entitlement_source` field on `StrategyRlMlTickerSlotsResponse`. |
| Persisted schema | `compatible-change` | Additive migration creates `rl_live_ticker_entitlement_overrides` and `rl_live_ticker_activations`; existing tables are not rewritten. |
| Config schema/defaults | `none` | Existing `STRATEGY_PG_DSN` selection is reused; no new required env var was added. |
| Request hashes/cache keys/identity | `compatible-change` | Entitlement decisions now depend on canonical `CurrentUserPrincipal.paid_level` and owner id for RL live profiles only. |
| Service call auth/timeout/retry/error behavior | `none` | No new external service call. Postgres access uses existing app DSN wiring; browser/Keycloak claims are not a new entitlement authority. |
| External side effects/idempotency/unknown state | `compatible-change` | Additive DB writes track RL live ticker activations; no order intent, exchange call, model load, promotion, paper/testnet/live execution or mainnet action is introduced. |
| Logs/metrics/traces/audit/report | `compatible-change` | Existing `record_live_strategy_profile_readiness` receives entitlement blocked reasons. Docs/screenshots contain no secrets, provider payloads or credentials. |
| Browser-visible behavior | `compatible-change` | `/strategies` RL/ML ticker-slot panel now shows backend paid-level quota values and blocked reasons through the existing Stage `11` renderer. |
| Performance hot path | `none` | Entitlement checks are API/dashboard readiness checks; no model inference, ClickHouse scan or execution hot-path work was added. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index updated. No runbook change was required because no operator workflow/deploy command changed. |

## Service Calls, Logging, Alerts

| Surface | Coverage |
|---|---|
| Business impact | Users see RL live ticker eligibility according to paid level. Free/Pro/Premium quota behavior is explicit, Enterprise override is supported, and ambiguous `base` is blocked rather than silently treated as Free. |
| Internal DB calls | Covered by repository tests and migration SQL contract. Postgres repository reads overrides and upserts/deactivates active live ticker activations; in-memory repository covers deterministic tests/dev fallback. |
| External service calls | `N/A`; no exchange, provider, Keycloak, Redis, ClickHouse, payment, notification or third-party API call was added. |
| Service auth/timeout/retry | `N/A` for external calls. DB access uses existing API Postgres DSN wiring. |
| Logging/redaction | No secrets, tokens, cookies, passphrases, ciphertext, raw provider payloads, HMACs, API keys, signed requests or checkpoint contents are logged or documented. |
| Alerts/monitoring/runbook | Existing live-profile readiness metric receives stable entitlement reasons. New alert/runbook coverage is `N/A` for Stage `12` because no runtime daemon, producer or exchange path was introduced; Stage `13` owns monitor-only runtime metrics. |

## Quality Gates And Evidence

| Command / evidence | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_live_entitlements.py tests/unit/contexts/rl_trading/domain/test_live_entitlement_repositories.py tests/unit/apps/migrations/test_rl_live_ticker_entitlements_sql.py tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | `31 passed in 1.87s` after one stale test assertion was corrected. |
| Focused `uv run ruff check ...` for changed backend/app/test paths | passed: `All checks passed!`. |
| Focused `uv run pyright ...` for changed backend/app/test paths | passed: `0 errors, 0 warnings, 0 informations` with the existing pyright version notice. |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed. |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed: `0 errors, 0 warnings, 0 informations` with the existing pyright version notice. |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | `467 passed, 8 skipped, 3 warnings in 13.28s`. Skips are known optional Torch skips; warnings are existing `httpx` cookie deprecation warnings in app/web tests. |
| Local browser harness at `http://127.0.0.1:64212/strategies` | Opened real `apps.web.main.app:create_app` page with sanitized mock dashboard JSON and fake current user; no credentials, provider payloads or exchange calls. |
| Playwright desktop proof | RL/ML tab activated; `rl_live_ticker_slots` state was degraded with reason `rl_live_ticker_quota_exceeded`; paid level `free`; product label `Free`; live slots `1 / 1`; slot row `ETHUSDT binance futures live blocked rl_live_ticker_quota_exceeded`; dashboard API response `200`; no console errors/warnings; no failed requests. |
| Playwright mobile proof at `390x844` | Same quota-blocked state rendered; `bodyOverflow=0`, `panelOverflow=0`; no console errors/warnings; no failed requests. |
| Browser screenshots | Local evidence only: `output/playwright/stage12-rl-entitlements-desktop.png`, `output/playwright/stage12-rl-entitlements-mobile.png`. These screenshots are generated artifacts and are not part of the git manifest. |

Browser proof boundary: `local_browser_runtime_harness`, not `post_main_production_runtime_proof`. The harness verified browser-visible rendering of sanitized backend entitlement state only; it did not exercise production auth, production deploy, Mac Studio `/opt/roehub/app`, exchange connectivity or provider credentials. A later `post_main_production_runtime_proof` must start from changed code on `main`, green GitHub Actions/CI for that revision, and completed deploy/sync to the target runtime before any Mac Studio `/opt/roehub/app` or production browser smoke can be reported as changed-code production proof.

## Review Mode

| Field | Value |
|---|---|
| Review mode | Cold self-review fallback. Independent review was not spawned for this local Stage `12` implementation. |
| Verdict | Accepted locally after focused/domain/API/migration/browser evidence and prompt-level backend gates passed. |
| Fixed blockers | Initial focused API/dashboard assertion expected the old Stage `11` placeholder; it was updated to the new Stage `12` entitlement reason and rerun successfully. No code blocker remained. |
| Follow-up check | Re-ran focused tests, focused ruff, focused pyright, prompt-level ruff/pyright/pytest, browser console/network/layout probes, docs-index check and `git diff --check`. |
| Residual risks | This is local working-tree acceptance, not GitHub delivery or production deployment. Additive schema exists but was not applied to production DB in this stage. Any future `post_main_production_runtime_proof` must require `main`, green GitHub Actions/CI, and deploy/sync evidence for the changed revision. Stage `13` must re-check entitlements before monitor-only runtime handoff and still cannot create order intents. |

## File Manifest

Created:

| Path | Reason | Contract impact |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/live_entitlements.py` | Shared entitlement domain contract, mapping, override semantics, mode-specific counting and stable reasons. | `compatible-change` internal domain surface |
| `src/trading/contexts/rl_trading/adapters/__init__.py` | Adapter package marker. | `none` packaging only |
| `src/trading/contexts/rl_trading/adapters/outbound/__init__.py` | Outbound adapter package marker. | `none` packaging only |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/__init__.py` | Export entitlement repositories. | `compatible-change` internal import surface |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/in_memory_live_entitlements.py` | Deterministic repository for tests/dev fallback. | `compatible-change` internal adapter |
| `src/trading/contexts/rl_trading/adapters/outbound/persistence/postgres_live_entitlements.py` | Postgres repository for override, active slot counting, activation upsert and deactivation. | `compatible-change` internal adapter |
| `alembic/versions/20260703_0041_rl_live_ticker_entitlements_v1.py` | Additive entitlement override and live activation schema. | `compatible-change` persisted schema |
| `tests/unit/contexts/rl_trading/domain/test_live_entitlements.py` | Domain contract tests for plan mapping, fail-closed cases, overrides and non-live mode semantics. | `none` test-only |
| `tests/unit/contexts/rl_trading/domain/test_live_entitlement_repositories.py` | Repository behavior tests for counting, replacement, deactivation and overrides. | `none` test-only |
| `tests/unit/apps/migrations/test_rl_live_ticker_entitlements_sql.py` | Migration SQL contract tests. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/12-backend-entitlements.md` | Stage `12` report, evidence, file manifest and handoff. | `compatible-change` docs/report |

Modified:

| Path | Reason | Contract impact |
|---|---|---|
| `apps/api/dto/ui_strategies_dashboard.py` | Add `entitlement_source` to RL ticker slot DTO. | `compatible-change` additive API/DTO response |
| `apps/api/routes/strategies.py` | Apply RL live ticker entitlements to live-profile create/update/readiness and persist blocked readiness reasons. | `compatible-change` additive readiness behavior |
| `apps/api/wiring/modules/strategy.py` | Build and inject RL live ticker entitlement service into strategy routes. | `compatible-change` additive wiring |
| `apps/api/wiring/modules/ui_strategies_dashboard.py` | Build real RL ticker slot read-model from entitlement service instead of Stage `11` placeholder. | `compatible-change` additive dashboard behavior |
| `tests/unit/apps/api/test_strategies_routes.py` | Cover free quota exceeded, non-live non-counting/release behavior and base fail-closed behavior. | `none` test-only |
| `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Cover dashboard entitlement state and no placeholder reason. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | Mark Stage `12` accepted and open Stage `13`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | Docs index regeneration if required by generator. | `compatible-change` docs index |

Deleted: none.

Outside expected path note: `docs/architecture/README.md` is the standard generated docs index and is expected by the repository docs gate after Markdown report/ledger changes. No `apps/web/*` source file was changed because Stage `11` already rendered backend-provided slot values and reasons.

## Blockers

No Stage `12` blocker remains.

Fail-closed constraints intentionally preserved:

- `base` and unknown paid levels cannot activate RL live tickers;
- unavailable entitlement repository blocks RL live readiness rather than allowing live activation;
- current-principal absence blocks RL live readiness;
- paper/testnet/live execution stages remain blocked by later RL and classic producer prerequisites;
- no production DB apply, deploy, GitHub delivery or Mac Studio changed-code runtime proof is claimed by this local Stage `12` implementation; `post_main_production_runtime_proof` remains gated on `main`, green GitHub Actions/CI and deploy/sync evidence.

## Next-Stage Handoff

`current_stage` moves to Stage `13` Monitor-only inference producer in the local ledger.

Stage `13` must verify Stage `10`, Stage `10A`, Stage `11` and Stage `12` are accepted, compute its own prompt hash, and re-check the Stage `12` entitlement contract before emitting any monitor-only `ml_agent_decision` events. Stage `13` is allowed to run only in monitor-only mode and must not create order intents.

Stage `14` remains blocked until Stage `13` is accepted. Stage `15`/`16` execution stages remain blocked by the classic strategy-producer paper/testnet prerequisites. Stage `12` acceptance did not write production registry state, enable exchange execution, apply production migrations or claim production runtime proof.
