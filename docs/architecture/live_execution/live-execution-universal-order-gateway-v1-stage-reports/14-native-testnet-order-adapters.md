# Stage 14: Native Testnet Order Adapters

Stage 14 adds testnet-only native Binance/Bybit order adapters inside the
`exchange-execution` process boundary, plus durable testnet order/private-stream
ledgers, mainnet hard-blocks, exchange server-time guard, exchange-control
credential resolution and bounded metrics.

Date: 2026-05-31.

Status: implementation complete locally; runtime acceptance pending direct-main
deploy and testnet boundary proof.

## Scope

Included:

- `adapter_mode=testnet` for `exchange-execution`, with `disabled` preserved for
  dev/local safe operation;
- thin native HTTP adapters for Binance and Bybit testnet submit/status/cancel;
- hard-block of any non-testnet connection before submit;
- exchange-control credential resolution inside `exchange-execution` only,
  using Postgres connection metadata plus OpenBao Transit decrypt;
- durable `execution_orders` rows before submit and after submit/status/cancel;
- durable `exchange_private_stream_sessions` rows for Binance listen-key
  keepalive and Bybit private-session readiness skeleton;
- ack-after-durable-order-decision behavior for processed Redis dispatches;
- bounded metrics for testnet order outcomes, private-stream lifecycle and
  adapter latency;
- focused unit/app/migration tests.

Out of scope:

- no mainnet submit;
- no Strategy/UI/browser secret access;
- no CCXT dependency;
- no fill/reconciliation ledger;
- no canary/mainnet approval.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused ruff | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_testnet_order_adapters_sql.py alembic/versions/20260531_0027_testnet_order_adapters_v1.py` | Passed. |
| Focused pyright | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py` | `0 errors`. |
| Focused tests | `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_testnet_order_adapters_sql.py tests/unit/apps/migrations/test_exchange_execution_process_skeleton_sql.py` | `9 passed`. |
| Required ruff scope | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests` | Passed. |
| Required pyright scope | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests` | `0 errors`. |
| Required unit/apps scope | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | `333 passed, 3 warnings`. |
| Required integration path | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps tests/integration` | Blocked locally because `tests/integration` does not exist in this checkout; no tests ran for that literal path. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |
| Local testnet-mode smoke without runtime dependencies | `ROEHUB_ENV=prod STRATEGY_FAIL_FAST=false ROEHUB_EXCHANGE_EXECUTION_CONFIG=configs/prod/exchange_execution.yaml uv run python - <<'PY' ... TestClient(create_app()) ... PY` | `/health/ready` returned HTTP `200`, `adapter_mode=testnet`, `status=degraded`, `status_reason=testnet_adapter_dependency_missing`; `POST /internal/v1/run-once` returned HTTP `503`, `reason=ConnectionError` instead of crashing when local Redis was absent. |
| Full ruff | `uv run ruff check .` | Passed. |
| Full pyright | `uv run pyright` | `0 errors`. |
| Full local suite | `uv run pytest -q -ra` | `1086 passed, 3 warnings`. |
| Credential availability probe | Local environment boolean-only check | Workstation has no `STRATEGY_PG_DSN`, `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` or Redis secret env. |
| Mac Studio credential prerequisites | SSH boolean-only env check and sanitized SQL grouping | Mac Studio has `STRATEGY_PG_DSN`, `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`; current DB has active Binance futures testnet trading connections. No Bybit testnet trading connection is currently present. |

## Runtime Evidence

Pending direct-main deploy.

Required before acceptance:

- run Alembic through deploy and prove `execution_orders` plus
  `exchange_private_stream_sessions` exist;
- dispatch one controlled Binance futures testnet ops intent through
  `execution.requests.v1`;
- call `POST http://127.0.0.1:9206/internal/v1/run-once` or process loop and
  prove submit/status/cancel were persisted before Redis `XACK`;
- prove mainnet connection dispatch is hard-blocked as `mainnet_hard_block`;
- scrape `/metrics` for testnet order, private stream, latency and clock drift;
- run secret grep over code/docs/artifacts and avoid raw provider payloads.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | none | No public API or browser route changes. |
| Ports/DTO | compatible-change | Adds internal `ExchangeOrderAdapter`, credential resolver and order repository ports under `live_execution`. |
| Persistence | compatible-change | Adds `execution_orders` and `exchange_private_stream_sessions`; expands exchange-execution heartbeat/observation constraints to include `testnet`. |
| Redis | compatible-change | Valid dispatches in `testnet` mode are acked only after a durable order guard/submit/error decision. |
| Config | compatible-change | `configs/test` and `configs/prod` move `exchange_execution.process.adapter_mode` to `testnet`; `configs/dev` remains `disabled`. |
| Runtime/Ops | compatible-change | `exchange-execution` now requires Postgres/OpenBao credential resolution for ready testnet adapter status. |
| UI/browser | none | No visible page changed. |
| Metrics/logs/redaction | compatible-change | Adds bounded metrics without user, connection, token, signature, payload or secret labels. |
| External side effects | compatible-change | Testnet-only submit/status/cancel side effects are introduced behind mainnet hard-block and durable ledger guards. |

## Rollback

Set `exchange_execution.process.adapter_mode` back to `disabled` and restart
`com.roehub.exchange-execution`. Existing `execution_orders` and
`exchange_private_stream_sessions` rows are additive audit records and can
remain. If a Redis message was processed, the durable order row is the source
of truth for later reconciliation; do not blindly replay testnet submit after
an unknown adapter state.

## Handoff To Stage 15

Stage `15` must treat `execution_orders` as the first exchange-facing order
ledger, but not as full reconciliation. It still needs fill/status event
ingestion, durable reconciliation, fee/funding integration, unknown-state
repair and retention/backup proof before production readiness.
