# Stage 09: Capital Reservation And Paper Accounting

Stage 09 adds durable paper-mode capital reservation and paper accounting before
any real order-submit stage. The implementation is additive and does not create
execution intents, Redis execution dispatch, exchange orders, credential decrypt
in `apps/api`, or exchange SDK/API calls from the Strategy run hot path.

Date: 2026-05-31.

Status: accepted. Local schema/application/UI gates, direct-main delivery, Mac
Studio runtime proof, public browser proof, stage report and ledger update are
complete.

## Scope

Included:

- `CapitalReservation` and paper order/fill/accounting domain models in
  `live_execution`;
- `PaperAccountingRepository` port plus in-memory and Postgres adapters;
- `CapitalReservationPaperAccountingService` with reserve, release and
  paper-signal recording paths;
- additive Alembic tables `strategy_capital_reservations`, `paper_orders`,
  `paper_fills` and `strategy_paper_accounting`;
- Strategy run start reserves quote capital for paper/live profiles with an
  exchange connection and fails closed on missing, stale or insufficient account
  projection;
- Strategy live runner records paper order/fill/accounting from persisted paper
  signals and releases reserved capital when the run reaches terminal state;
- restart successor runs reserve capital before activation;
- explicit paper fee/funding/completeness fields:
  `paper_fixed_bps_10`, `spot_not_applicable`, `pnl_complete`;
- bounded metrics `strategy_capital_reservation_total{result,reason}` and
  `strategy_paper_accounting_total{result,reason}`;
- `/strategies` dashboard/API/UI paper accounting read-model with reserved
  budget, paper position, average entry, equity, PnL, fees, funding and
  completeness reason;
- HTTP `409` mapping for `strategy_run.capital_reservation_blocked`.

Out of scope:

- no real exchange order submission;
- no execution source event or execution intent creation;
- no Redis execution dispatch stream;
- no order-model/risk gate beyond capital reservation;
- no exchange credential decrypt in Strategy/API;
- no futures funding reconciliation beyond explicit paper completeness status.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused paper accounting tests | `uv run pytest -q tests/unit/contexts/live_execution/test_paper_accounting_service.py tests/unit/apps/migrations/test_capital_reservation_paper_accounting_sql.py` | `5 passed`. |
| Browser asset syntax | `node --check apps/web/dist/js/pages/strategies.js` | Passed. |
| Focused ruff | `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy/application apps/api apps/worker tests/unit/contexts/live_execution/test_paper_accounting_service.py tests/unit/apps/migrations/test_capital_reservation_paper_accounting_sql.py` | Passed. |
| Focused pyright | `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy/application apps/api apps/worker tests/unit/contexts/live_execution/test_paper_accounting_service.py tests/unit/apps/migrations/test_capital_reservation_paper_accounting_sql.py` | `0 errors`. |
| Strategy/live-execution regression | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/contexts/live_execution tests/unit/apps/migrations/test_capital_reservation_paper_accounting_sql.py` | `31 passed`. |
| API/web regression | `uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/web/test_backtest_ui_asset.py tests/unit/apps/web/test_app_routes.py` | `35 passed, 3 warnings`. |
| Required ruff | `uv run ruff check .` | Passed. |
| Required pyright | `uv run pyright` | `0 errors`. |
| Required tests before implementation publish | `uv run pytest -q -ra` | `1033 passed, 3 warnings`. |
| Error mapping regression | `uv run pytest -q tests/unit/apps/api/test_api_error_handlers.py tests/unit/contexts/live_execution/test_paper_accounting_service.py tests/unit/apps/migrations/test_capital_reservation_paper_accounting_sql.py` | `8 passed`. |
| Required tests after error mapping repair | `uv run pytest -q -ra` | `1034 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |

## Delivery Evidence

Direct-main delivery:

- implementation commit: `72e4cc1b Add capital reservation paper accounting`;
- HTTP mapping repair commit found by production smoke:
  `5642bb95 Map capital reservation blocks to conflict`;
- CI `26717573957`: success;
- Deploy Backend `26717617956`: success;
- Publish App Image `26717617942`: success, with a non-fatal Docker cache
  reservation annotation;
- Deploy Web `26717617959`: success;
- follow-up Deploy Web `26717638407`: success after app image publish;
- repair CI `26717822941`: success;
- repair Deploy Backend `26717866018`: success;
- repair Publish App Image `26717866012`: success;
- repair Deploy Web `26717866016` and follow-up `26717869924`: success.

## Runtime Evidence

Production proof used disposable authenticated smoke sessions, synthetic account
projection rows for the smoke user, and controlled Redis market-data candles.
The successful strategies used profile mode `paper`, fixed quote sizing `100`
and controlled `md.candles.1m.codexstage09*:spot:BTCUSDT` input. No real
exchange connection was called. A second equivalent smoke pass after the HTTP
mapping repair verified the final metric labels.

Accepted runtime proof:

| Surface | Evidence | Result |
|---|---|---|
| Migration | Mac Studio Postgres `to_regclass` found `strategy_capital_reservations`, `paper_orders`, `paper_fills` and `strategy_paper_accounting`. | Pass. |
| Blocked stale capital | `POST /strategies/{strategy}/run` with a 600-second-old account projection returned HTTP `409` and contained reason `capital_projection_stale`. Production smoke first found the correct reason mapped to HTTP `500`; commit `5642bb95` mapped `strategy_run.capital_reservation_blocked` to conflict and the same smoke passed. | Pass. |
| Blocked insufficient capital | `POST /strategies/{strategy}/run` with free USDT `10` and requested quote `25` returned HTTP `409` and contained reason `capital_insufficient_available_balance`. | Pass. |
| Reservation create | Successful `POST /strategies/{strategy}/run` returned run `98909d65-6230-4808-9b67-529ba17bb1ce`, state `starting`; SQL reservation row was `reserved`, requested/reserved `100.000000000000`, reason `capital_reserved`, fee model `paper_fixed_bps_10`, funding model `spot_not_applicable`, `pnl_complete=true`. | Pass. |
| Controlled paper signal | Production live-runner `run_once` first created the consumer group without reading preflight data, then four controlled candles produced report `polled_runs=1`, `active_instruments=1`, `read_messages=4`, `acked_messages=4`, `failed_runs=0`. | Pass. |
| Paper accounting rows | SQL showed one `paper_orders` row, one `paper_fills` row and one `strategy_paper_accounting` row for the run. The accounting row had reserved budget `100.000000000000`, position `5.000000000000`, average entry `20.000000000000`, equity `99.800000000000`, realized PnL `0`, unrealized PnL `-0.100000000000`, fee total `0.100000000000`, funding total `0`, fee model `paper_fixed_bps_10`, funding model `spot_not_applicable`, `pnl_complete=true`, completeness `paper_fee_fixed_bps_funding_not_applicable`. | Pass. |
| Signal source | `strategy_signals` recorded a paper `signal`, action `open`, side `buy`, reason `ma_fast_crossed_above_slow_paper_no_order_stage05`; Stage 09 converted that durable signal into paper order/fill/accounting without exchange submit. | Pass. |
| Idempotent replay | Additional live-runner `run_once` read `0`, acked `0`, and order/fill counts stayed `1 -> 1`. | Pass. |
| Stop/release | `POST /strategies/{strategy}/stop` returned state `stopping`; the next live-runner iteration released the reservation to `state=released`, reason `run_stopped`, `released_at=2026-05-31T16:24:26.326Z`; final active smoke runs `0`, active smoke reservations `0`. | Pass. |
| Dashboard API | `GET /ui/strategies/dashboard?strategy_id=6480052c-d6b3-4766-8f2f-80c45db89622` returned HTTP `200` and included paper accounting source `strategy_paper_accounting` with the same budget/position/equity/PnL/fee/funding/completeness fields; secret-token scan was false. | Pass. |
| Metrics | API `/metrics` emitted `strategy_capital_reservation_total` for reserved/rejected outcomes. The one-shot production live-runner process emitted local `strategy_capital_reservation_total{result="released"}`; paper accounting metric emitted `strategy_paper_accounting_total{result="filled",reason="paper_fee_fixed_bps_funding_not_applicable"}` in the runner process. | Pass. |
| No execution side effects | Redis scan for execution patterns returned `0` keys; `to_regclass('public.execution_intents')` returned `NULL`. | Pass. |
| Public browser | Playwright on `https://roehub.com/strategies?strategy_id=6480052c-d6b3-4766-8f2f-80c45db89622` with a temporary smoke session rendered the selected strategy, Live profile `paper`, Account readiness `fresh`, Paper accounting `complete`, Reserved budget `100.000000000000`, Paper position `5.000000000000`, Avg entry `20.000000000000`, Equity `99.800000000000`, PnL `0E-12 / -0.100000000000`, Fees/funding `0.100000000000 / 0E-12`, Fee/funding model `paper_fixed_bps_10 / spot_not_applicable`, and Latest signals with the paper buy signal. Dashboard network requests returned `200`; screenshot `output/playwright/stage09-strategies-paper-accounting.png`. Temporary browser session was revoked. | Pass. |

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Strategy run can now return HTTP `409` with code `strategy_run.capital_reservation_blocked` when capital projection is missing, stale, degraded, invalid or insufficient. `/ui/strategies/dashboard` includes additive paper accounting fields. |
| Persistence | compatible-change | Adds capital reservation and paper order/fill/accounting tables only; no destructive migration. |
| Redis | none | Reads existing market-data streams only; no execution stream or dispatch consumer added. |
| Runtime/Ops | compatible-change | Adds bounded capital and paper-accounting metrics; no new supervised process. |
| UI/browser | compatible-change | `/strategies` shows additive Paper accounting panel. |
| External side effects | none | No exchange order submit, exchange SDK/API call, credential decrypt or execution intent. |
| Rollback | compatible-change | Stop active paper runs, verify reservations released, then disable reservation/paper accounting wiring if required; audit rows remain inert. |

## Handoff To Stage 10

Stage `10` can rely on durable reservation and paper accounting for Strategy
paper mode, but no execution source event, execution intent, order-model risk
gate, Redis dispatch or exchange submit exists yet. Later stages must preserve
the Stage 09 fail-closed stale/insufficient capital behavior and must not treat
paper accounting as proof of real exchange fill.
