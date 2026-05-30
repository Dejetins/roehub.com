# Stage 06: Compatibility And Market-Data Readiness

Stage 06 adds a fail-closed strategy compatibility checker and market-data
readiness evidence before a strategy can be presented or run as ready.

Date: 2026-05-31.

Status: implemented locally; direct-main delivery and production runtime
evidence pending.

## Scope

Included:

- compatibility checks for the currently supported Stage 05 live evaluator
  subset;
- stable compatibility states `launchable`, `not_launchable`, and `degraded`;
- market-data readiness states `ready`, `missing`, `stale`, and `pending`;
- additive Postgres evidence tables for compatibility checks and market-data
  subscription requirements;
- Redis `XINFO STREAM` readiness probe for the existing market-data candle
  stream;
- fail-closed run/profile readiness integration;
- API read endpoints for strategy and backtest-variant readiness;
- bounded metrics for compatibility and market-data readiness results;
- `/backtests` and `/strategies` UI readiness display;
- focused migration, API, use-case, and web asset regression tests;
- this stage report and ledger update.

Out of scope:

- no mainnet or testnet order submit;
- no exchange SDK/API call, credential decrypt, signed payload, or exchange
  private response handling;
- no execution intent/request/source-event table or Redis execution stream;
- no account projection, position ownership, capital reservation, paper
  accounting, or reconciliation.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `05` accepted before Stage `06`. | Ledger status says Stage `05` accepted with direct-main delivery, CI/deploy and production runtime evidence complete. | Pass. |
| Work on `main`, no stage branch or PR. | Local checkout is on `main`; prompt requires direct-main delivery. | Pass. |
| Runtime acceptance is not tests-only. | Production API/DB/Redis/browser proof is still pending for this initial local implementation report. | Pending. |

## Files Changed

Code:

- `src/trading/contexts/strategy/application/use_cases/compatibility_readiness.py`
- `src/trading/contexts/strategy/application/ports/compatibility_readiness.py`
- `src/trading/contexts/strategy/application/ports/market_data_readiness.py`
- `src/trading/contexts/strategy/application/ports/repositories/compatibility_readiness_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/compatibility_readiness_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/compatibility_readiness_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_market_data_readiness.py`
- `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`
- `src/trading/contexts/strategy/application/use_cases/run_strategy.py`
- `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py`
- `apps/api/routes/backtests.py`
- `apps/api/routes/strategies.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/api/wiring/modules/strategy.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`
- `apps/api/dto/ui_strategies_dashboard.py`
- `apps/api/monitoring.py`
- strategy package export files under `src/trading/contexts/strategy/.../__init__.py`

Schema:

- `alembic/versions/20260531_0019_strategy_compatibility_market_data_readiness_v1.py`

UI:

- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/contexts/strategy/application/test_strategy_use_cases.py`
- `tests/unit/apps/migrations/test_strategy_compatibility_readiness_sql.py`

Docs:

- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/06-compatibility-market-data-readiness.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

`StrategyCompatibilityReadinessService` evaluates either an existing strategy or
a backtest variant snapshot converted through the same canonical
`StrategySpecV1` builder used by create-from-variant.

Compatibility states:

- `launchable`: supported Stage 05 `MA(fast,slow)` evaluator on `1m`;
- `degraded`: supported evaluator that requires market-data rollup or has a
  large warmup;
- `not_launchable`: unsupported schema/spec kind/evaluator.

Market-data states:

- `ready`: existing Redis market-data stream has a fresh last generated id;
- `missing`: Redis reports no such market-data stream key;
- `stale`: the stream exists but the last id timestamp is older than the
  freshness threshold;
- `pending`: the stream is empty or the readiness probe cannot currently reach
  Redis.

`RunStrategyUseCase` now checks compatibility/readiness before creating a run.
`LiveStrategyProfileService` also evaluates compatibility/readiness before
profile readiness, so monitor-only and paper profiles remain blocked while their
strategy/feed evidence is unsafe.

## Local Evidence

| Surface | Evidence | Result |
|---|---|---|
| Focused compatibility/migration use-case tests | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/apps/migrations/test_strategy_compatibility_readiness_sql.py` | `11 passed`. |
| Focused API/UI/use-case tests | `uv run pytest -q tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/web/test_backtest_ui_asset.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/apps/migrations/test_strategy_compatibility_readiness_sql.py` | `125 passed, 3 warnings` from existing httpx cookie deprecations. |
| Compile check | `uv run python -m compileall -q apps/api src/trading/contexts/strategy apps/web` | Passed after fixing an indentation issue found during implementation. |
| Locale JSON | `uv run python -m json.tool apps/web/locales/en.json >/dev/null && uv run python -m json.tool apps/web/locales/ru.json >/dev/null` | Passed. |
| Broad stage tests | `uv run pytest -q tests/unit/contexts/backtest tests/unit/contexts/strategy tests/unit/contexts/market_data tests/unit/apps` | `786 passed, 3 warnings` from existing httpx cookie deprecations. |
| Broad lint | `uv run ruff check src/trading/contexts/backtest src/trading/contexts/strategy src/trading/contexts/market_data apps tests` | Passed. |
| Broad type checking | `uv run pyright src/trading/contexts/backtest src/trading/contexts/strategy src/trading/contexts/market_data apps tests` | `0 errors, 0 warnings`. |

## Runtime Evidence

Local runtime probes before delivery:

- `pg_isready`: unavailable on this host;
- `redis-cli`: unavailable on this host;
- `docker`: unavailable on this host;
- relevant local DB/Redis env variables: not set in the current shell.

Production direct-main evidence is pending. Stage `06` is not accepted until the
deployed revision proves:

- API compatibility/readiness calls for `launchable`, `not_launchable`,
  `degraded`, owner-forbidden, and replay/read repeat cases;
- SQL rows in `strategy_variant_compatibility_checks` and
  `market_data_subscription_requirements`;
- Redis `XINFO`/`XRANGE` for ready/stale/missing market-data streams and absence
  of execution streams;
- `/backtests` and `/strategies` browser readiness display.

## Error Behavior

| Case | Code/state | Expected behavior |
|---|---|---|
| Unsupported evaluator | `not_launchable`, `unsupported_live_evaluator` | Run/profile readiness blocked; no run row or order side effect. |
| Non-`1m` supported evaluator | `degraded`, `timeframe_rollup_required` | Launch allowed only if market-data stream is ready; reason remains visible for later rollup stages. |
| Missing market-data stream | `missing`, `market_data_stream_missing` | Run/profile readiness blocked. |
| Empty/probe-unavailable stream | `pending`, `market_data_stream_empty` or `market_data_readiness_probe_unavailable` | Run/profile readiness blocked. |
| Stale market-data stream | `stale`, `market_data_stream_stale` | Run/profile readiness blocked. |
| Ready market-data stream | `ready`, `market_data_stream_ready` | Run can proceed if compatibility is not `not_launchable`. |

## Runtime Config

No new environment variables, YAML files, launchd jobs, Monit rules, or secret
settings were added.

The Redis readiness reader reuses the existing strategy live-worker Redis stream
configuration and reads the existing market-data candle stream. If the reader is
not configured or Redis is unavailable, readiness is `pending` and run/profile
readiness blocks fail closed.

## Monitoring

Added bounded API counters:

- `strategy_variant_compatibility_total{state,reason}`;
- `market_data_readiness_total{state,reason}`.

Metric labels are fixed state/reason codes and do not include user ids,
strategy ids, instrument keys, stream names, exchange payloads, credentials, or
raw provider responses.

## Logging And Redaction

No secrets, cookies, DSNs, Authorization headers, API keys, private keys,
passphrases, ciphertext, signed exchange payloads, raw exchange responses, or
raw idempotency keys are intentionally logged or persisted by Stage 06.

The new persistence stores owner/strategy/source ids, spec hash, instrument,
timeframe, bounded reason codes, stream metadata, and freshness timestamps.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds new readiness endpoints and dashboard fields; existing routes remain compatible. Run/profile readiness can now fail closed when checker dependencies are wired. |
| Port / boundary interfaces | `compatible-change` | Adds compatibility and market-data readiness ports plus optional use-case dependencies. |
| Persistence | `compatible-change` | Adds `strategy_variant_compatibility_checks` and `market_data_subscription_requirements`; no destructive migration. |
| Redis | `compatible-change` | Reads existing market-data streams only; no execution stream is created or consumed. |
| Config | `none` | Reuses existing strategy Redis runtime config. |
| Runtime / ops | `compatible-change` | Existing API and profile/run paths gain fail-closed readiness checks and bounded counters. No new process. |
| UI / browser | `compatible-change` | Adds readiness/feed fields to existing `/backtests` and `/strategies` surfaces. |
| Exchange/provider side effects | `none` | No credential decrypt, exchange read, exchange submit, or paper accounting. |
| Logs/metrics/redaction | `compatible-change` | Adds bounded metric labels and secret-safe readiness evidence. |
| Docs | `compatible-change` | Adds Stage 06 report and ledger state. |

## Rollback

Rollback path before delivery:

- revert Stage 06 code/UI/docs changes;
- drop local `20260531_0019` migration if it has only been applied locally.

Rollback path after delivery:

- revert the Stage 06 commit and redeploy backend/web;
- keep existing compatibility/readiness rows as inert audit evidence, or drop
  the additive tables only after confirming no later stage depends on them;
- no exchange/order reconciliation is required because Stage 06 has no external
  order or provider side effects.

## Handoff To Stage 07

Facts Stage 07 must preserve after Stage 06 acceptance:

- supported evaluator compatibility is intentionally narrow and reason-coded;
- feed readiness is a run/profile precondition and blocks on missing/stale/pending
  Redis evidence;
- `degraded` compatibility is visible and reason-coded but not a launch blocker
  when market data is ready;
- readiness evidence is durable in Postgres and API-visible for debugging;
- no execution stream, exchange submit, account projection, ownership lock, or
  capital reservation exists yet.
