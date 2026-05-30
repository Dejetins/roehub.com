# Stage 06: Compatibility And Market-Data Readiness

Stage 06 adds a fail-closed strategy compatibility checker and market-data
readiness evidence before a strategy can be presented or run as ready.

Date: 2026-05-31.

Status: accepted; direct-main delivered; CI/deploy and Mac Studio/public runtime
evidence complete.

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
| Runtime acceptance is not tests-only. | Mac Studio production proof recorded API states, SQL rows, Redis stream readiness, browser readiness display, metrics, no execution streams, and cleanup. | Pass. |

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
| Full publish lint | `uv run ruff check .` | Passed. |
| Full publish type checking | `uv run pyright` | `0 errors, 0 warnings, 0 informations`. |
| Full publish tests | `uv run pytest -q -ra` | `1018 passed, 3 warnings` from existing httpx cookie deprecations. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Diff whitespace | `git diff --check` | Passed. |

## Runtime Evidence

Local runtime probes before delivery:

- `pg_isready`: unavailable on the workstation;
- `redis-cli`: unavailable on the workstation;
- `docker`: unavailable on the workstation;
- relevant local DB/Redis env variables: not set in the workstation shell.

Commit and delivery:

- implementation commit: `2001e415 feat: add strategy compatibility readiness`;
- GitHub CI `26697234121`: success;
- Deploy Backend `26697309560`: success; DB bootstrap/migrations and backend
  smoke passed;
- Publish App Image `26697309566`: success with a non-fatal Docker cache
  reservation warning;
- Deploy Web `26697309568`: success; follow-up Deploy Web run `26697329959`
  initially hit a transient public-edge smoke failure and passed after rerun;
- Mac Studio `scripts/macos/smoke_prod.sh`: pass after deploy.

Controlled Stage 06 production smoke:

- smoke id: `stage06-ff9f51650345`;
- API strategy readiness states:
  - `launchable/ready`: `200`, reason `supported_live_evaluator` +
    `market_data_stream_ready`, `launch_blocked=false`;
  - `launchable/missing`: `200`, reason `market_data_stream_missing`,
    `launch_blocked=true`;
  - `launchable/stale`: `200`, age `600s`, reason
    `market_data_stream_stale`, `launch_blocked=true`;
  - `launchable/pending`: `200`, empty stream, reason
    `market_data_stream_empty`, `launch_blocked=true`;
  - `degraded/ready`: `200`, reason `timeframe_rollup_required` +
    `market_data_stream_ready_for_rollup`, `launch_blocked=false`;
  - `not_launchable/ready`: `200`, reason `unsupported_live_evaluator`,
    `launch_blocked=true`;
- run proof:
  - ready strategy `POST /strategies/{strategy_id}/run` returned `200
    starting`;
  - missing-feed strategy run returned `strategy_run.readiness_blocked` with
    reason `market_data_stream_missing`;
  - synthetic ready run was stopped and then manually moved from `stopping` to
    `stopped` as cleanup because the worker did not drain the synthetic stream
    within the smoke wait window;
- backtest variant proof:
  - controlled job `4d2fde8d-a8d9-4d2a-b688-31997d8dcd29` succeeded with `5`
    top variants;
  - selected variant
    `job_4d2fde8dcd29__dema_close_w10__risk_none__vh_cee9b06e`;
  - `GET /backtests/jobs/{job_id}/variants/{variant_key}/compatibility-readiness`
    returned `200`, `not_launchable`, `unsupported_live_evaluator`, and feed
    `ready`;
  - repeat read returned `200`;
  - foreign owner read returned `403` with
    `strategy_variant_launch.forbidden`.

Redis/runtime proof:

- ready stream `md.candles.1m.binance:spot:C06READY43E2E7`: `XINFO length=1`;
- stale stream `md.candles.1m.binance:spot:C06STALE43E2E7`: `XINFO length=1`,
  last id timestamp produced `age_seconds=600`;
- pending stream `md.candles.1m.binance:spot:C06PEND43E2E7`: `XINFO length=0`
  after controlled `XADD`/`XDEL`;
- missing stream `md.candles.1m.binance:spot:C06MISS43E2E7`: key absent;
- Redis scan for `*execution*` returned `0` keys.

Postgres proof for the smoke user:

- `strategy_variant_compatibility_checks`: `10` rows;
- `market_data_subscription_requirements`: `10` rows;
- recorded compatibility states include `launchable`, `not_launchable`, and
  `degraded`;
- recorded readiness states include `ready`, `missing`, `stale`, and `pending`;
- `to_regclass('public.execution_intents') = NULL`.

Metrics proof from `/metrics`:

- `strategy_variant_compatibility_total{state="launchable",reason="supported_live_evaluator"} 4`;
- `strategy_variant_compatibility_total{state="degraded",reason="timeframe_rollup_required"} 1`;
- `strategy_variant_compatibility_total{state="not_launchable",reason="unsupported_live_evaluator"} 3`;
- `market_data_readiness_total{state="ready",reason="market_data_stream_ready"} 2`;
- `market_data_readiness_total{state="missing",reason="market_data_stream_missing"} 1`;
- `market_data_readiness_total{state="stale",reason="market_data_stream_stale"} 1`;
- `market_data_readiness_total{state="pending",reason="market_data_stream_empty"} 1`;
- `market_data_readiness_total{state="ready",reason="market_data_stream_ready_for_rollup"} 3`.

Browser proof:

- Playwright against `https://roehub.com/strategies` with the temporary smoke
  session rendered `Compatibility launchable: supported_live_evaluator` and
  `Market data stale: market_data_stream_stale` after the controlled ready
  stream aged past the freshness threshold; rerun had `0` server `5xx`
  responses;
- Playwright against `https://roehub.com/backtests/{job_id}` rendered the
  selected variant detail with `READINESS not_launchable:
  unsupported_live_evaluator` and `FEED ready:
  market_data_stream_ready_for_rollup`; `0` failed requests and `0` console
  errors on the selected-variant proof;
- screenshots:
  - `output/playwright/stage06-strategies-readiness-rerun.png`;
  - `output/playwright/stage06-backtests-readiness-selected.png`.

Cleanup:

- temporary Stage 06 smoke sessions revoked: `2`;
- active Stage 06 smoke sessions after cleanup: `0`;
- active smoke runs after cleanup: `0`;
- durable smoke strategies, compatibility checks, market-data requirements, and
  the completed backtest job remain as inert audit evidence owned by the
  temporary smoke user.

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
