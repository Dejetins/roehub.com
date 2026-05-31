# Stage 08: Strategy Position Ownership Lock

Stage 08 introduces the durable lock that prevents two active strategies from
owning the same exchange position on the same user connection and instrument.
The implementation is additive and does not submit orders, create execution
intents, decrypt credentials in `apps/api`, or call exchange-control/exchange
SDKs from the strategy run hot path.

Date: 2026-05-31.

Status: implemented locally. Local schema, application behavior, metrics, UI
conflict rendering and tests are complete. Production runtime acceptance,
direct-main delivery and browser proof are pending.

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
| Whitespace | `git diff --check` | Passed. |

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

## Runtime Acceptance Pending

Required before marking Stage `08` accepted:

- two production strategy/profile run attempts on the same
  connection/instrument prove first owner active and second
  `position_ownership_conflict`;
- terminal stop through the live runner proves ownership release;
- DB rows prove active, releasing, released and stale/repair behavior;
- `strategy_position_ownership_total` metrics are visible;
- `/strategies` Playwright proof shows the conflict reason and no unsafe override;
- no Redis execution stream, execution intent, exchange submit or credential
  decrypt path appears.

## Handoff To Stage 09

Stage `09` must treat position ownership as necessary but not sufficient for
capital safety. It still needs durable capital reservation, paper orders/fills
and strategy-local accounting before paper/live execution can be accepted.
