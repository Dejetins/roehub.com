# Stage 14: Native Testnet Order Adapters

Stage 14 adds testnet-only native Binance/Bybit order adapters inside the
`exchange-execution` process boundary, plus durable testnet order/private-stream
ledgers, mainnet hard-blocks, exchange server-time guard, exchange-control
credential resolution and bounded metrics.

Date: 2026-05-31.

Status: blocked for acceptance. Implementation, CI and deploy are complete, but
the available Binance testnet trade-ready connections use placeholder
credentials that cannot be decrypted by the exchange-control Transit boundary.
Per the Stage 14 prompt, missing testnet credentials block live submit/status/
cancel acceptance rather than being downgraded to tests-only evidence.

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
- conditional durable mainnet hard-block rows in `execution_orders`, constrained
  to `guard_rejected/mainnet_hard_block` with no provider order id or submit/
  cancel timestamps;
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
| Mainnet guard migration repair | `uv run ruff check alembic/versions/20260531_0028_execution_order_mainnet_guard_rows_v1.py tests/unit/apps/migrations/test_testnet_order_adapters_sql.py` | Passed. |
| Mainnet guard migration test | `uv run pytest -q tests/unit/apps/migrations/test_testnet_order_adapters_sql.py` | `2 passed`. |
| Mainnet guard migration pyright | `uv run pyright tests/unit/apps/migrations/test_testnet_order_adapters_sql.py` | `0 errors`. |
| Mainnet guard migration whitespace | `git diff --check` | Passed. |
| Credential availability probe | Local environment boolean-only check | Workstation has no `STRATEGY_PG_DSN`, `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` or Redis secret env. |
| Mac Studio credential prerequisites | SSH boolean-only env check and sanitized SQL grouping | Mac Studio has `STRATEGY_PG_DSN`, `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`; current DB has active Binance futures testnet trading connections. No Bybit testnet trading connection is currently present. |

## Runtime Evidence

Direct-main deploy evidence:

- implementation commit `c2d3dd287692c461b85fb8b59bf4646b7db857bb`;
- mainnet hard-block guard-row repair commit
  `588106700f63ef3c2751c95a07b6ace06466349f`;
- CI for `c2d3dd28`: GitHub Actions run `26723973362`, success;
- deploy for `c2d3dd28`: Publish App Image `26724051166`, Deploy Backend
  `26724051155`, Deploy Web `26724051152`/`26724055221`, success;
- CI for `58810670`: GitHub Actions run `26724194507`, success;
- deploy for `58810670`: Publish App Image `26724237547`, Deploy Backend
  `26724237558`, Deploy Web `26724237553`, success.

Mac Studio runtime evidence:

- deployed runtime contains `apps/exchange_execution/adapters/native_http.py`
  and Stage 14 migrations;
- Alembic head is `20260531_0028`;
- `GET http://127.0.0.1:9206/health/ready` returned `status=ready`,
  `status_reason=all_dependencies_ready`, `adapter_mode=testnet`, with config,
  adapter, rate-limit, Redis, backpressure, DLQ, clock-drift and Postgres all
  ready;
- schema proof shows `execution_orders` and
  `exchange_private_stream_sessions` exist;
- sanitized connection grouping shows active Binance futures testnet trading
  connections, but both active Binance testnet credentials have
  `secret_cipher=stage03_no_decrypt_placeholder` and fail decrypt; there is no
  active Bybit testnet trading connection;
- controlled Binance futures testnet dispatch through
  `execution.requests.v1` returned `run-once` HTTP `200` with
  `guard_rejected_count=1`, `acked_count=1`; Postgres recorded
  `execution_orders.status=guard_rejected`,
  `status_reason=exchange_credential_decrypt_failed`,
  `exchange_order_id_present=false`, and the Redis observation used the same
  message id;
- controlled active mainnet dispatch through `execution.requests.v1` returned
  `run-once` HTTP `200` with `submitted_count=0`,
  `guard_rejected_count=1`, `acked_count=1`; Postgres recorded
  `environment=mainnet`, `status=guard_rejected`,
  `status_reason=mainnet_hard_block`, `exchange_order_id_present=false`,
  `submitted_at_present=false`, `cancelled_at_present=false`, and Redis was
  acked after the durable guard decision;
- `/metrics` shows `exchange_execution_ready{status="ready"} 1`,
  `exchange_execution_dependency_ready{dependency="adapter",reason="testnet_adapters_ready"} 1`,
  `exchange_execution_clock_drift_ms 0.128`,
  `exchange_execution_observations_total` for `mainnet_hard_block` and
  `exchange_credential_decrypt_failed`, plus matching
  `exchange_execution_ack_total` counters;
- `exchange_execution_testnet_order_total`,
  `exchange_execution_private_stream_total` and
  `exchange_execution_submit_latency_ms` are registered but have no samples
  because credential decrypt failed before private stream or order submit.

Acceptance blocker:

- Safe submit/status/cancel was not executed against Binance or Bybit testnet.
  The Stage 14 validation strategy explicitly says missing testnet credentials
  block acceptance for that adapter. This stage remains blocked until at least
  one active exchange-testnet connection has decryptable Transit-managed
  credentials.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | none | No public API or browser route changes. |
| Ports/DTO | compatible-change | Adds internal `ExchangeOrderAdapter`, credential resolver and order repository ports under `live_execution`. |
| Persistence | compatible-change | Adds `execution_orders` and `exchange_private_stream_sessions`; expands exchange-execution heartbeat/observation constraints to include `testnet`; permits `mainnet` rows only for durable `guard_rejected/mainnet_hard_block` evidence with no provider order id or submit/cancel timestamps. |
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

## Secrets And Redaction

No raw API keys, secrets, passphrases, signed payloads, OpenBao tokens,
ciphertext, cookies or raw provider responses were written to this report.
Runtime SQL and credential probes emitted only grouped connection state,
credential decrypt success/failure booleans and stable reason codes. Provider
order identifiers were not emitted because no testnet submit reached the
exchange boundary.

## Handoff To Stage 15

Stage `15` remains blocked until Stage `14` is unblocked with decryptable
exchange-testnet credentials and records real submit/status/cancel plus private
stream evidence. The next operator should provision or rotate a Binance and/or
Bybit testnet connection through exchange-control so the credential row uses
`exchange_control_transit_v1`, then rerun the same `exchange-execution`
dispatch probe. After that, Stage `15` can treat `execution_orders` as the
first exchange-facing order ledger, but not as full reconciliation.
