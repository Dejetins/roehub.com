# Stage 04: Run, Stop, Restart Lifecycle

Stage 04 hardens Strategy run control with a durable restart command, one-active-run enforcement through restart drain, worker pickup/drain behavior, and `/strategies` browser controls.

Date: 2026-05-31.

Status: accepted; direct-main delivered; CI/deploy and Mac Studio/public runtime evidence complete.

## Scope

Included:

- `RestartStrategyUseCase`;
- `POST /strategies/{strategy_id}/restart`;
- restart metadata persisted on the stopping run;
- live-runner drain behavior that stops the old run before creating a successor run;
- UI `Restart` control and refreshed run-control enablement;
- current-state Strategy docs for API, domain storage and live runner;
- regression coverage for restart conflicts, worker drain, route wiring and Postgres run metadata persistence.

Out of scope:

- no mainnet order submit;
- no testnet order submit;
- no paper order/fill/accounting ledger;
- no exchange credential decryption;
- no Redis execution dispatch stream.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `03` accepted before Stage `04`. | Ledger row for `03` is `accepted`; Stage 03 report records production API/SQL/metrics/browser proof. | Pass. |
| Work on `main`, no stage branch or PR. | Delivery used direct commits on `main`: `f1b5004a` and repair `586a2742`. | Pass. |
| Runtime acceptance is not tests-only. | Post-deploy production API/DB/worker/browser evidence is recorded below. | Pass. |

## Files Changed

Code:

- `apps/api/routes/strategies.py`
- `apps/api/wiring/modules/strategy.py`
- `src/trading/contexts/strategy/application/use_cases/restart_strategy.py`
- `src/trading/contexts/strategy/application/use_cases/__init__.py`
- `src/trading/contexts/strategy/application/__init__.py`
- `src/trading/contexts/strategy/__init__.py`
- `src/trading/contexts/strategy/application/services/live_runner.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py`

UI:

- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/apps/api/test_strategies_routes.py`
- `tests/unit/contexts/strategy/application/test_strategy_use_cases.py`
- `tests/unit/contexts/strategy/application/test_strategy_live_runner.py`
- `tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py`

Docs:

- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`
- `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/04-run-stop-restart-lifecycle.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

`RestartStrategyUseCase` requires an owned, non-deleted strategy and an active run. It rejects:

- restart without active run;
- duplicate restart while `metadata.restart.state == "pending_start"`;
- restart while a run is already stopping.

On accepted restart, the use case:

- writes `metadata.restart.operation_id`;
- sets restart state to `pending_start`;
- transitions the active run to `stopping`;
- appends `run_restart_requested`.

The live runner handles `stopping` runs by:

- transitioning the old run to `stopped`;
- changing restart metadata to `drained`;
- creating the successor run only after the old run is terminal;
- writing successor metadata with `restart.state=successor_started` and `previous_run_id`.

This keeps the one-active-run invariant intact during restart.

## Local Evidence

| Surface | Evidence | Result |
|---|---|---|
| Focused restart/API/runner/Postgres adapter tests | `uv run pytest -q tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/apps/api/test_strategies_routes.py` | `35 passed`. |
| Focused lint | `uv run ruff check src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py` | Passed. |
| Repository lint | `uv run ruff check .` | Passed. |
| Repository type checking | `uv run pyright` | `0 errors, 0 warnings, 0 informations`. |
| Repository tests | `uv run pytest -q -ra` | `1010 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed before report commit. |

The warnings are existing `httpx` per-request cookie deprecation warnings in web route tests.

## Runtime Evidence

Post-deploy Mac Studio smoke passed:

- `ssh macstudio 'cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh'`;
- launchd listed core Roehub services including API, exchange-control, backtest runner, market-data workers, exporters, Redis, Postgres, ClickHouse, Keycloak, OpenBao and Tailscale services;
- local API unauthenticated current-user returned `401`;
- metrics probes, ClickHouse, Redis `PONG`, Postgres schema check and Tailscale backend state passed.

The first production Stage 04 smoke exposed a real adapter bug before acceptance:

- `POST /strategies/{strategy_id}/run` returned `500 unexpected_error`;
- root cause: `PostgresStrategyRunRepository` passed Python dict metadata directly to `%(metadata_json)s::jsonb`;
- repair commit `586a2742` serializes `metadata_json` and adds regression coverage.

Accepted production API/DB/worker evidence after repair used a synthetic owner/session/strategy on Mac Studio with no printed cookie or DSN:

| Runtime boundary | Evidence | Result |
|---|---|---|
| Strategy create | `POST /strategies` | `201`, strategy `61221013-c0e3-4353-b4e4-a81f989c0159`. |
| Restart without active run | `POST /strategies/{id}/restart` | `409 conflict`, `Strategy has no active run to restart`. |
| Run start | `POST /strategies/{id}/run` | `200`, run `420868c1-9c71-471c-9cff-febf67e6a7fd`, state `starting`. |
| Duplicate run active conflict | second `POST /run` | `409 conflict`, `Strategy already has active run`. |
| Restart active run | `POST /restart` | `200`, same run state `stopping`, restart state `pending_start`, operation id present. |
| Duplicate restart | second `POST /restart` while pending | `409 conflict`, `Strategy restart is already pending`. |
| Run while restart stopping | `POST /run` while restart is pending | `409 conflict`, `Strategy already has active run`. |
| Worker drain restart | one production `StrategyLiveRunner.run_once()` with Postgres repos | `polled_runs=1`, `active_instruments=1`, `failed_runs=0`, stream read for `binance:spot:BTCUSDT`. |
| Run after successor active | `POST /run` after successor creation | `409 conflict`, `Strategy already has active run`. |
| Stop successor | `POST /stop` | `200`, successor `f8447578-87ec-4fc2-b24e-8a51ec418e0e`, state `stopping`. |
| Worker drain stop | one production `StrategyLiveRunner.run_once()` with Postgres repos | `polled_runs=1`, `active_instruments=1`, `failed_runs=0`. |
| Stop without active run | `POST /stop` after drain | `409 conflict`, `Strategy has no active run`. |

SQL proof for that strategy:

- `strategy_runs` old run `420868c1-9c71-471c-9cff-febf67e6a7fd`: `state=stopped`, `restart_state=drained`, `stopped_at` set;
- `strategy_runs` successor `f8447578-87ec-4fc2-b24e-8a51ec418e0e`: `state=stopped`, `restart_state=successor_started`, `previous_run_id=420868c1-9c71-471c-9cff-febf67e6a7fd`, `stopped_at` set;
- final active rows for the restart flow: `0`;
- `strategy_events`: one each for `strategy_created`, `run_started`, `run_restart_requested`, `run_stop_requested`.

Additional active-row browser target proof:

- synthetic strategy `f9412bca-5c0b-49ef-805a-6da5cc3f4c53`;
- `POST /run` created run `1aa247f9-1423-4d21-8fa7-b8de20bd2dc0` with state `starting`;
- SQL active rows for that strategy while browser proof ran: `1`;
- cleanup `POST /stop` returned `200`, worker drain had `polled_runs=1`, and active rows returned to `0`;
- the temporary session was revoked after browser proof.

Redis evidence:

- `redis-cli --raw SCAN 0 MATCH md.candles.1m.binance:spot:BTCUSDT COUNT 10`;
- returned cursor-only result for that specific stream key during the smoke;
- Stage 04 restart did not create or use execution dispatch streams.

Public browser proof used Playwright CLI against `https://roehub.com/strategies?strategy_id=f9412bca-5c0b-49ef-805a-6da5cc3f4c53` with a synthetic session cookie that was not printed or committed:

- selected strategy showed run state `starting`;
- `Run` button disabled;
- `Stop` and `Restart` buttons enabled;
- manual `Refresh` button triggered `GET https://roehub.com/api/ui/strategies/dashboard?refresh=manual&state=all&strategy_id=f9412bca-5c0b-49ef-805a-6da5cc3f4c53 => 200`;
- console after initial load and refresh: `0` errors, `0` warnings;
- screenshots and snapshots:
  - `output/playwright/stage04-strategies-active.png`
  - `output/playwright/stage04-strategies-active.snapshot.txt`
  - `output/playwright/stage04-strategies-after-refresh.png`
  - `output/playwright/stage04-strategies-after-refresh.snapshot.txt`
  - `output/playwright/stage04-strategies-after-refresh.requests.txt`

## Error Behavior

| Case | Code/state | Expected behavior |
|---|---|---|
| Restart without active run | `409 conflict` | No run row created. |
| Duplicate active run | `409 conflict` | Existing active run preserved. |
| Accepted restart | `200 stopping` | Existing run carries `restart.pending_start`. |
| Duplicate restart | `409 conflict` | Existing pending restart preserved. |
| Run while stopping/restart pending | `409 conflict` | One-active-run invariant preserved. |
| Stop successor | `200 stopping` | Worker drains successor to terminal state. |
| Stop without active run | `409 conflict` | No new run row created. |

## Runtime Config

No new environment variables, YAML files, feature flags, launchd jobs, Monit rules or kill switches were added.

Fail-closed defaults:

- if Strategy API is disabled, restart route is absent with the Strategy router;
- if Strategy Postgres storage is unavailable, run control fails closed rather than creating ambiguous in-memory production state;
- restart successor is created only after the previous run is terminal.

## Monitoring

No new metric family was added in this stage. Existing API request metrics and live-runner iteration counters cover the runtime path.

Stage 05+ should add strategy signal/evaluator metrics when candle processing begins to produce durable signal/no-signal outcomes.

## Logging And Redaction

No secrets, cookies, DSNs, Authorization headers, API keys, private keys, passphrases, ciphertext, signed exchange payloads, raw exchange provider responses or raw idempotency keys were intentionally printed or committed.

The synthetic browser session cookie was stored only in `/tmp/roehub_stage04_strategy_session.json` on Mac Studio for the duration of proof and then revoked.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds `POST /strategies/{strategy_id}/restart`; existing run/stop response shape unchanged. |
| Persistence | `compatible-change` | Uses existing `strategy_runs.metadata_json` and `strategy_events`; no new table or destructive migration. |
| Redis | `none` | Restart path does not add execution streams or Redis config. |
| Config | `none` | No env/YAML/default change. |
| Runtime / ops | `compatible-change` | Existing API and live-runner runtime path gains restart drain behavior; no new process. |
| UI / browser | `compatible-change` | Adds restart control and run-control enablement to existing `/strategies` dashboard. |
| Exchange/provider side effects | `none` | No credential decrypt, exchange read, exchange submit or paper accounting. |
| Docs | `compatible-change` | Updates Strategy docs, Stage 04 report and ledger. |

## Rollback

Rollback path:

- revert commits `586a2742` and `f1b5004a`;
- redeploy backend and web through the standard main workflows;
- existing `strategy_runs.metadata_json.restart` and `run_restart_requested` events are audit data and can remain inert;
- no schema rollback is required because Stage 04 adds no migration.

Operational fallback:

- keep using existing `run` and `stop`;
- do not expose/call `restart` until the reverted or repaired revision is redeployed.

## Handoff To Stage 05

Accepted facts for Stage 05:

- run/stop/restart API semantics are durable and conflict-tested;
- one-active-run invariant survived restart drain and successor creation in production DB;
- live runner can pick up and drain `stopping` runs against production Postgres;
- `/strategies` browser controls reflect active run state after refresh;
- no execution dispatch stream or exchange submit exists yet.
