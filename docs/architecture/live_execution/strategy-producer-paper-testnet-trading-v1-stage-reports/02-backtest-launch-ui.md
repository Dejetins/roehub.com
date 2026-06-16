# Stage 02 — Backtest-to-strategy launch UI

Дата: 2026-06-17

## Pre-Start

User required before start: nothing

Stage `01` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`: статус `accepted`, `Next stage allowed = yes`.

## Pre-Edit Scope

Ожидаемые broad paths сужены до конкретных файлов до implementation edits:

| Planned file | Planned action | Reason |
|---|---|---|
| `apps/api/routes/strategies.py` | modify | Добавить chained launch endpoint из backtest variant в strategy/profile/run setup. |
| `apps/api/wiring/modules/strategy.py` | modify | Подключить existing variant-create use case к strategy command router. |
| `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py` | modify | Записать sanitized launch config в provenance/hash без secrets. |
| `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py` | modify | Явный `testnet` profile mode и fail-closed readiness. |
| `src/trading/contexts/strategy/application/use_cases/run_strategy.py` | modify | Принять sanitized launch metadata в run row. |
| `src/trading/contexts/strategy/application/services/live_runner.py` | modify | Учесть `testnet` mode in runtime reason/capital guards. |
| `src/trading/contexts/strategy/domain/entities/live_strategy_profile.py` | modify | Расширить allowed mode enum на `testnet`. |
| `alembic/versions/20260617_0031_strategy_testnet_mode_v1.py` | create | Additive constraint expansion for `testnet` mode. |
| `tests/unit/apps/api/test_strategies_routes.py` | modify | API launch success and fail-closed cases. |
| `tests/unit/apps/migrations/test_strategy_testnet_mode_sql.py` | create | Migration contract for mode constraint expansion. |
| `tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | modify | Use-case metadata/profile/run coverage if needed. |
| `apps/web/templates/pages/backtests.html` | modify | Launch dialog fields for paper/testnet configuration. |
| `apps/web/dist/js/pages/backtests.js` | modify | Browser flow: configure launch, submit, redirect to `/strategies`. |
| `apps/web/dist/css/pages/backtests.css` | modify | Dialog/form layout and blocked reason display. |
| `apps/web/dist/js/pages/strategies.js` | modify | Display `testnet` mode without falling back to live wording. |
| `apps/web/locales/en.json` | modify | Launch UI copy. |
| `apps/web/locales/ru.json` | modify | Launch UI copy. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/02-backtest-launch-ui.md` | create/modify | Stage report, evidence, manifest. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify | Stage status/evidence/handoff. |
| `docs/architecture/README.md` | modify/check | Docs index after new report. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `alembic/versions/20260617_0031_strategy_testnet_mode_v1.py` | none | none | Additive SQL constraint expansion for `testnet` profile/signal mode. | compatible-change |
| `tests/unit/apps/migrations/test_strategy_testnet_mode_sql.py` | none | none | SQL text guard for the additive mode migration. | none |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/02-backtest-launch-ui.md` | none | none | Stage evidence/report. | none |
| none | `apps/api/routes/strategies.py` | none | Added `POST /api/strategies/launch-from-backtest-variant` with paper/testnet allowlist, `$50` default config, provenance, profile update, run start, and stable fail-closed error reasons. | compatible-change |
| none | `apps/api/wiring/modules/strategy.py` | none | Wired existing backtest variant reader/provenance repositories into strategy router composition. | compatible-change |
| none | `apps/api/common/errors.py` | none | Added canonical HTTP mappings for `strategy_launch.*` errors. | compatible-change |
| none | `apps/api/dto/ui_strategies_dashboard.py` | none | Allowed `testnet` in strategy dashboard mode DTOs. | compatible-change |
| none | `src/trading/contexts/strategy/domain/entities/live_strategy_profile.py` | none | Allowed `testnet` as a strategy live profile mode. | compatible-change |
| none | `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py` | none | Added deterministic `testnet` readiness reason. | compatible-change |
| none | `src/trading/contexts/strategy/application/use_cases/run_strategy.py` | none | Accepted sanitized launch metadata on run creation and included `testnet` in capital guards. | compatible-change |
| none | `src/trading/contexts/strategy/application/services/live_runner.py` | none | Included `testnet` in runtime guard reason classification. | compatible-change |
| none | `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py` | none | Added normalized launch config to request hash/provenance metadata without secrets. | compatible-change |
| none | `apps/web/templates/pages/backtests.html` | none | Replaced old variant strategy-create confirm with launch form for paper/testnet, BTCUSDT, market, allocation, sizing, risk, direction, and exchange connection id for testnet only. | compatible-change |
| none | `apps/web/dist/js/pages/backtests.js` | none | Added launch form state, payload submission, blocked reason rendering through existing error path, and redirect to `/strategies` after successful launch. | compatible-change |
| none | `apps/web/dist/css/pages/backtests.css` | none | Added compact launch form/button-group layout. | none |
| none | `apps/web/locales/en.json`, `apps/web/locales/ru.json` | none | Updated launch UI copy. | none |
| none | `tests/unit/apps/api/test_strategies_routes.py` | none | Added success and fail-closed API coverage for launch config/profile/run metadata. | none |
| none | `tests/unit/apps/web/test_app_routes.py` | none | Added SSR contract checks for launch endpoint/fields and no API secret prompt. | none |
| none | `docs/architecture/README.md` | none | Regenerated docs index after adding this report. | none |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Stage status/evidence handoff. | none |

## Contract Impact Draft

| Dimension | Classification | Notes |
|---|---|---|
| Public API / DTO | compatible-change | Additive launch endpoint and optional `testnet` mode expansion; existing endpoints remain. |
| Persistence | compatible-change | Additive check-constraint expansion; no destructive data migration. |
| Browser-visible behavior | compatible-change | Adds launch form and blocked reason display on `/backtests`. |
| External side effects | compatible-change | Launch remains `paper`/`testnet`; `mainnet` not accepted by the new endpoint. |
| Secrets/redaction | none | UI/API do not accept API secrets; exchange connection id only. |

## Evidence

### Local Gates

| Command | Result |
|---|---|
| `uv run pytest -q tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/migrations/test_strategy_testnet_mode_sql.py` | `10 passed` |
| `uv run pytest -q tests/unit/apps/web` | `39 passed, 3 warnings` (`httpx` cookie deprecation only) |
| `uv run pytest -q tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/strategy` | `302 passed, 3 warnings` |
| `uv run ruff check apps/api apps/web src/trading/contexts/strategy tests` | passed |
| `uv run pyright apps/api src/trading/contexts/strategy tests` | `0 errors` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

### Browser Runtime

Runtime target: local SSR app with real `/backtests` and `/strategies` templates/assets, mocked authenticated user, mocked API read models. No secrets or exchange credentials were used.

| Scenario | Evidence |
|---|---|
| Success path `/backtests` -> top variant -> launch -> `/strategies` | Playwright showed `GET /api/ui/backtests/workstation` `200`, variant summary/detail/readiness endpoints `200`, `POST /api/strategies/launch-from-backtest-variant` `200`, redirect to `/strategies`, and post-launch `GET /api/ui/strategies/dashboard` `200`. Console result: `0` errors, `0` warnings. |
| Launch payload | Request body: `job_id=11111111-1111-4111-8111-111111111111`, `variant_key=22222222-2222-4222-8222-222222222222`, `mode=paper`, `market_type=spot`, `symbol=BTCUSDT`, `capital_allocation_usd=50`, `entry_sizing=fixed_quote`, `risk_mode=single_position_cap`, `direction=long`; no secret fields. |
| Testnet UI branch | Screenshot artifact: `output/playwright/backtest-launch-ui-testnet-toggle-final.png`; selecting `testnet` reveals the exchange connection field and preserves `$50`/BTCUSDT defaults. |
| Blocked browser case | Screenshot artifact: `output/playwright/backtest-launch-ui-blocked-testnet.png`; testnet launch without exchange connection shows `HTTP 422: Invalid strategy launch configuration; exchange_connection_required` in the modal. The negative scenario has the expected Chromium resource error for the intercepted `422`; the success scenario is console-clean. |

### API / SQL

| Surface | Evidence |
|---|---|
| API success | `test_launch_from_backtest_variant_creates_profile_and_run_config` asserts created strategy/profile/run, `paper` mode, `BTCUSDT`, `$50`, `spot`, `fixed_quote`, `single_position_cap`, `long`, provenance, and run `metadata_json.launch_config`; no secret fields are accepted or persisted. |
| API blocked cases | `test_launch_from_backtest_variant_blocks_testnet_without_exchange` returns `strategy_launch.invalid_config` with reason `exchange_connection_required`; `test_launch_from_backtest_variant_blocks_invalid_sizing_and_min_notional` returns stable reasons for `invalid_entry_sizing` and `insufficient_allocation_min_notional`. |
| SQL migration contract | `tests/unit/apps/migrations/test_strategy_testnet_mode_sql.py` asserts the Alembic migration expands both `strategy_live_profiles_mode_chk` and `strategy_signals_mode_chk` to `mode IN ('monitor_only', 'paper', 'live', 'testnet')`. |
| Runtime SQL | Mac Studio production Postgres reports `alembic_version=20260617_0031`; both `strategy_live_profiles_mode_chk` and `strategy_signals_mode_chk` allow `monitor_only`, `paper`, `live`, `testnet`. |

## Delivery Status

| Item | Status |
|---|---|
| Local implementation | complete |
| Local gates | complete |
| Browser success and blocked evidence | complete |
| GitHub publish prerequisite | complete: `gh auth status` authenticated as `Dejetins` with `repo` and `workflow` scopes |
| Main delivery / origin main | complete: pushed `main` commit `762ef6cbdc95b8f0b969cdb20cef5e7dfb6300a0` (`Implement backtest launch UI`) |
| CI | complete: [CI run `27649915820`](https://github.com/Dejetins/roehub.com/actions/runs/27649915820) succeeded |
| Deploy Backend | complete: [Deploy Backend run `27650024744`](https://github.com/Dejetins/roehub.com/actions/runs/27650024744) succeeded |
| Publish App Image | complete: [Publish App Image run `27650024742`](https://github.com/Dejetins/roehub.com/actions/runs/27650024742) succeeded |
| Deploy Web | complete: [Deploy Web run `27650101271`](https://github.com/Dejetins/roehub.com/actions/runs/27650101271) succeeded |
| Mac Studio checkout sync | complete: `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded from `a7fb40f0` to `762ef6cbdc95b8f0b969cdb20cef5e7dfb6300a0`; `git status -sb` reports `## main...origin/main` |
| Mac Studio deploy smoke | complete: `scripts/macos/smoke_prod.sh` exited `0`; launchd services include `com.roehub.api`, `com.roehub.backtest-job-runner`, `com.roehub.exchange-execution`; app redirects unauthenticated `/` to `/login`; API unauthenticated call returns expected `401 missing_session_id`; Redis `PONG`; Tailscale backend state `Running` |
| Runtime SQL | complete: production DB has Alembic `20260617_0031` and `testnet` in both strategy mode constraints |

Stage `02` is accepted. `mainnet` remains unavailable through this launch path.

## Next Handoff

- Stage `03` may start from `main` commit `762ef6cbdc95b8f0b969cdb20cef5e7dfb6300a0` or later.
- Existing fail-closed branches to preserve: no mainnet, no secret prompt, testnet requires exchange connection id, spot short on testnet is rejected, allocation below `$10` is rejected.
