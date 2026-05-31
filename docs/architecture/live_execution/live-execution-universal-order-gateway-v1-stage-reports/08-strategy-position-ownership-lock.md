# Stage 08: Strategy Position Ownership Lock

Stage 08 introduces the durable lock that prevents two active strategies from
owning the same exchange position on the same user connection and instrument.
The implementation is additive and does not submit orders, create execution
intents, decrypt credentials in `apps/api`, or call exchange-control/exchange
SDKs from the strategy run hot path.

Date: 2026-05-31.

Status: accepted. Local schema, application behavior, metrics, UI conflict
rendering, tests, direct-main delivery, Mac Studio runtime proof and public
browser proof are complete.

## Scope

Included:

- `StrategyPositionOwnership` domain model in `live_execution`;
- `StrategyPositionOwnershipRepository` port plus in-memory and Postgres
  adapters;
- `StrategyPositionOwnershipService` with reserve, activate, releasing,
  release and stale-repair transitions;
- additive Alembic table `strategy_position_ownership`;
- partial unique invariant over
  `owner_user_id + exchange_connection_id + market_type + instrument_key` while
  state is `reserved`, `active`, `releasing` or `stale_requires_repair`;
- Strategy run start reserves ownership when the live profile has an
  `exchange_connection_id`;
- second strategy run on the same connection/instrument fails with
  `position_ownership_conflict`;
- stop/restart requests mark the ownership `releasing`;
- the live runner releases ownership when a run reaches terminal `stopped` or
  `failed`;
- restart successor runs reserve and activate ownership after the previous run
  is terminal;
- bounded metric `strategy_position_ownership_total{result,reason}`;
- `/strategies` command status now renders run/stop/restart API errors, including
  the ownership conflict reason.

Out of scope:

- no real exchange order submission;
- no Redis execution stream;
- no capital reservation or paper accounting;
- no portfolio allocator or shared-position policy;
- no hedge-mode ownership model;
- no exchange credential decrypt or raw exchange payload persistence.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused tests | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py::test_position_ownership_blocks_second_strategy_on_same_connection_instrument tests/unit/contexts/strategy/application/test_strategy_live_runner.py::test_live_runner_releases_position_ownership_after_terminal_stop tests/unit/apps/migrations/test_strategy_position_ownership_sql.py` | `3 passed`. |
| Broader focused tests | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/strategy/application tests/unit/apps/migrations/test_strategy_position_ownership_sql.py tests/unit/apps/migrations/test_exchange_account_projection_config_guard_sql.py tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | `40 passed`. |
| Ruff | `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy/application apps/api apps/worker tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/apps/migrations/test_strategy_position_ownership_sql.py` | Passed. |
| Pyright | `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy/application apps/api apps/worker tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/apps/migrations/test_strategy_position_ownership_sql.py` | `0 errors`. |
| Browser asset syntax | `node --check apps/web/dist/js/pages/strategies.js` | Passed. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |
| Required ruff | `uv run ruff check .` | Passed. |
| Required pyright | `uv run pyright` | `0 errors`. |
| Required tests | `uv run pytest -q -ra` | `1028 passed, 3 warnings`. |

## Delivery Evidence

Direct-main delivery:

- implementation commit: `ce1c52b2 Add strategy position ownership lock`;
- CI `26716333983`: success;
- Deploy Backend `26716376148`: success, including DB bootstrap/migrations and
  backend API smoke;
- Publish App Image `26716376147`: success, with a non-fatal Docker cache
  reservation annotation;
- Deploy Web `26716376154`: success;
- follow-up Deploy Web `26716401645`: success after app image publish.

## Runtime Evidence

Production proof used the smoke account's existing Bybit trade-ready mainnet
connection `8e3999ba-c35d-4bcc-8253-a12b1d458114` and the ready
`md.candles.1m.bybit:spot:BTCUSDT` market-data stream. The profile mode was
`monitor_only`; no order path, exchange submit or credential decrypt was used.

Accepted runtime proof:

| Surface | Evidence | Result |
|---|---|---|
| Migration | Mac Studio Postgres `to_regclass('public.strategy_position_ownership')` returned the table and initial row count was `0` before the first smoke. | Pass. |
| Profile/readiness | Two production strategies with the same `exchange_connection_id`, `market_type=spot`, and `instrument_key=bybit:spot:BTCUSDT` had profiles `ready/monitor_only_no_exchange_submit`; compatibility/readiness returned `launchable`, `supported_live_evaluator`, `ready`, `market_data_stream_ready`, `launch_blocked=false`. | Pass. |
| First owner | `POST /strategies/{first}/run` returned `200 starting`; DB ownership row for the run moved to `active`, reason `run_started`, same connection/instrument scope. | Pass. |
| Active conflict | `POST /strategies/{second}/run` returned HTTP `409` with code and reason `position_ownership_conflict`; error details reported existing state `active` and the existing strategy/run ids. | Pass. |
| Stale repair block | Diagnostic service transition marked the first ownership row `stale_requires_repair` with reason `stage08_smoke_stale_requires_repair`; the second run attempt still returned `409 position_ownership_conflict` with existing state `stale_requires_repair`. | Pass. |
| Stop/release | `POST /strategies/{first}/stop` returned `200 stopping` and ownership moved to `releasing`. A production live-runner `run_once` iteration reported `polled_runs=1`, `active_instruments=1`, `read_messages=0`, `acked_messages=0`, `failed_runs=0`, then SQL showed run `stopped` and ownership `released`, reason `run_stopped`. | Pass. |
| Final cleanup | Temporary Stage 08 smoke sessions were revoked; smoke active sessions `0`, active runs `0`, blocking ownership rows `0`. | Pass. |
| Metrics | API `/metrics` emitted `strategy_position_ownership_total` for `reserved/run_started`, `active/run_started`, `conflict/position_ownership_conflict`, and `releasing/run_stop_requested`. The long-running `:9203` metrics endpoint was not open in current Mac Studio ops state, so worker release was proven by one production live-runner iteration and SQL state. | Pass with ops note. |
| No execution side effects | Redis scan for `*execution*`, `*live_execution*`, and `exchange.execution*` returned `0` keys; `to_regclass('public.execution_intents')` returned `NULL`. | Pass. |
| Public browser | Playwright on `https://roehub.com/strategies` used a temporary smoke session, selected the second strategy and clicked `Run`. The action status rendered `Position ownership is already held by another active strategy run: position_ownership_conflict`; screenshot `output/playwright/stage08-prod-strategies-position-conflict.png`; failed network requests `0` after filtering favicon; console contained only the expected 409 resource error; DOM secret scan was `false`. | Pass. |

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Strategy run can now return HTTP `409` with code `position_ownership_conflict` when another active run owns the same connection/instrument. Existing successful run response is unchanged. |
| Persistence | compatible-change | Adds `strategy_position_ownership` only; no destructive migration. |
| Redis | none | No execution streams or consumers added. |
| Runtime/Ops | compatible-change | Adds bounded ownership metrics; no new supervised process. |
| UI/browser | compatible-change | `/strategies` command status displays action errors instead of swallowing run/stop/restart failures. |
| External side effects | none | No exchange read/write or order submit. |
| Rollback | compatible-change | Disable ownership wiring before live/paper activation if necessary; after active use, stop/release or repair affected ownership rows before disabling. |

## Handoff To Stage 09

Stage `09` must treat position ownership as necessary but not sufficient for
capital safety. It still needs durable capital reservation, paper orders/fills
and strategy-local accounting before paper/live execution can be accepted.
