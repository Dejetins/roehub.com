# Stage 05: Live Signal Evaluator And StrategySignal Journal

Stage 05 runs Strategy live signal evaluation on controlled closed candles and
persists durable `StrategySignal`/no-signal journal rows without order
submission.

Date: 2026-05-31.

Status: accepted; direct-main delivered; CI/deploy and Mac Studio/public runtime
evidence complete.

## Scope

Included:

- additive `strategy_signals` Postgres journal;
- `StrategySignal` domain entity and repository ports/adapters;
- Stage 05 evaluator for supported `MA(fast,slow)` backtest-created specs;
- live-runner integration after closed rollup bucket processing;
- mode capture from `LiveStrategyProfile` with safe `monitor_only` fallback;
- bounded worker metric `strategy_signal_total{mode,action,outcome}`;
- latest signal journal projection in `/ui/strategies/dashboard`;
- `/strategies` latest signal journal panel;
- focused domain/runner/repository/API/migration regression tests;
- current strategy docs and live execution ledger updates.

Out of scope:

- no mainnet or testnet order submit;
- no `ExecutionIntent`, `ExecutionRequest` or `ExecutionSourceEvent` creation;
- no Redis execution dispatch stream;
- no paper order/fill/accounting ledger;
- no exchange credential decrypt, exchange SDK/API call or signed payload.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `04` accepted before Stage `05`. | Ledger status says Stage `04` accepted with direct-main delivery, CI/deploy and production runtime evidence complete. | Pass. |
| Work on `main`, no stage branch or PR. | Local checkout is `main...origin/main`; worktree was clean before implementation. | Pass. |
| Runtime acceptance is not tests-only. | Mac Studio production runtime proof recorded controlled Redis candle input, `strategy_signals` rows, API/browser signal journal, worker metric labels, realtime output streams, and absence of execution streams/messages. | Pass. |

## Files Changed

Code:

- `src/trading/contexts/strategy/domain/entities/strategy_signal.py`
- `src/trading/contexts/strategy/domain/entities/__init__.py`
- `src/trading/contexts/strategy/domain/__init__.py`
- `src/trading/contexts/strategy/application/services/signal_evaluator.py`
- `src/trading/contexts/strategy/application/services/live_runner.py`
- `src/trading/contexts/strategy/application/services/__init__.py`
- `src/trading/contexts/strategy/application/ports/repositories/strategy_signal_repository.py`
- `src/trading/contexts/strategy/application/ports/repositories/__init__.py`
- `src/trading/contexts/strategy/application/ports/__init__.py`
- `src/trading/contexts/strategy/application/__init__.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/strategy_signal_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/__init__.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_signal_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/__init__.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/__init__.py`
- `src/trading/contexts/strategy/adapters/outbound/__init__.py`
- `src/trading/contexts/strategy/adapters/__init__.py`
- `src/trading/contexts/strategy/__init__.py`
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
- `apps/api/wiring/modules/strategy.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`
- `apps/api/dto/ui_strategies_dashboard.py`

Schema:

- `alembic/versions/20260531_0018_strategy_signals_v1.py`

UI:

- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/dist/css/pages/strategies.css`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/contexts/strategy/application/test_strategy_live_runner.py`
- `tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py`
- `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py`
- `tests/unit/apps/migrations/test_strategy_signals_sql.py`

Docs:

- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/05-live-signal-evaluator-journal.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

`StrategyLiveRunner` now evaluates a signal only after a closed rollup bucket is
accepted and the run checkpoint/metadata path remains contiguous.

Supported Stage 05 evaluator subset:

- exactly one `MA` indicator;
- integer `params.fast` and `params.slow`;
- `fast > 0`, `slow > 0`, `fast < slow`;
- `signal_template == MA(fast,slow)`.

The evaluator persists:

- `warmup` while run warmup or evaluator close history is incomplete;
- `no_signal` once baseline is ready and no crossover occurred;
- `signal` with `open/buy` for fast MA crossing above slow MA;
- `signal` with `close/sell` for fast MA crossing below slow MA;
- `blocked` for unsupported live evaluator variants.

Signal rows capture `monitor_only|paper|live` mode from `strategy_live_profiles`.
If no profile is present, the default is `monitor_only`.

Stage 05 intentionally keeps `expected_order_json = {}` and does not create any
order intent, paper order, execution stream or exchange call.

## Local Evidence

| Surface | Evidence | Result |
|---|---|---|
| Focused runner/repository/API/migration tests | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/contexts/strategy/adapters/test_postgres_strategy_repositories.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/migrations/test_strategy_signals_sql.py` | `30 passed`. |
| Focused lint | `uv run ruff check src/trading/contexts/strategy apps tests/unit/contexts/strategy tests/unit/apps` | Passed. |
| Focused type checking | `uv run pyright src/trading/contexts/strategy apps tests/unit/contexts/strategy tests/unit/apps` | `0 errors, 0 warnings, 0 informations`. |
| Full publish gates | `uv run ruff check . && uv run pyright && uv run pytest -q -ra && uv run python -m tools.docs.generate_docs_index --check && git diff --check` | Ruff passed; Pyright `0 errors`; pytest `1015 passed, 3 warnings` (existing httpx cookie deprecations); docs index check passed; diff check passed. |

## Runtime Evidence

Commit and delivery:

- implementation commit: `17c56b36 feat: add live strategy signal journal`;
- GitHub CI `26696418272`: success;
- Deploy Backend `26696449721`: success; migration/bootstrap ran and backend
  smoke passed;
- Publish App Image `26696449720`: success with a non-fatal Docker cache
  reservation warning;
- Deploy Web `26696468892`: success; public edge smoke passed;
- Mac Studio `scripts/macos/smoke_prod.sh`: pass after deploy.

Controlled Stage 05 production smoke:

- smoke id: `stage05-8327e2ae4fc0`;
- controlled Redis stream:
  `md.candles.1m.codexstage0572d2ed:spot:BTCUSDT`;
- runner report from the live-runner service path:
  `polled_runs=3`, `active_instruments=1`, `read_messages=4`,
  `acked_messages=4`, `failed_runs=1`;
- Postgres `strategy_signals` rows for the smoke user: `9`;
- monitor-only strategy outcomes:
  `warmup`, `warmup`, `no_signal`, `signal`;
- latest monitor-only signal:
  `action=open`, `side=buy`,
  `reason_code=ma_fast_crossed_above_slow_monitor_only_no_intent`;
- paper profile signal:
  `action=open`, `side=buy`,
  `reason_code=ma_fast_crossed_above_slow_paper_no_order_stage05`;
- unsupported variant:
  `outcome=blocked`, `reason_code=unsupported_live_evaluator`, run state
  `failed`, `last_error=unsupported_live_evaluator`;
- `expected_order_json` proof: `0` rows for the smoke user had a non-empty
  expected-order payload;
- execution boundary proof:
  `to_regclass('public.execution_intents') = NULL`, Redis scan for
  `*execution*` returned `0` keys.

Redis/runtime proof:

- `XINFO STREAM` length for the controlled market-data stream: `4`;
- `XRANGE` sample count: `4`;
- consumer group pending count after ack: `0`;
- realtime output streams were written:
  metrics stream length `85`, events stream length `5`;
- bounded worker metric labels were emitted in-process:
  `strategy_signal_total{action="none",mode="monitor_only",outcome="warmup"} 2.0`,
  `strategy_signal_total{action="open",mode="monitor_only",outcome="signal"} 1.0`,
  `strategy_signal_total{action="none",mode="monitor_only",outcome="no_signal"} 1.0`.

API/browser proof:

- internal API
  `/ui/strategies/dashboard?strategy_id=2362df73-913a-4a64-8173-805285ef0315&state=all&refresh=manual`
  returned `signal_journal.source=strategy_signals`,
  `signal_journal.state=ready`, `signal_journal.items=4`;
- Playwright CLI against public `https://roehub.com/strategies` with the
  temporary smoke session rendered `rows=4`, `state=ready`, and visible
  `ma_fast_crossed_above_slow_monitor_only_no_intent`;
- screenshot:
  `output/playwright/stage05-strategies-signals.png`.

Cleanup:

- temporary smoke session was revoked;
- two synthetic active smoke runs were moved to `stopped`;
- post-cleanup active smoke runs: `0`;
- durable smoke strategies/signals remain as inert audit evidence owned by the
  temporary smoke user.

## Error Behavior

| Case | Code/state | Expected behavior |
|---|---|---|
| Unsupported evaluator variant | `StrategySignal.outcome=blocked`; run fails closed | No order, execution intent, Redis execution message or exchange call. |
| Warmup not satisfied | `StrategySignal.outcome=warmup` | Journal explains no trade decision. |
| No crossover | `StrategySignal.outcome=no_signal` | Journal explains normal no-signal outcome. |
| `monitor_only` signal | `signal` plus reason suffix `monitor_only_no_intent` | Signal recorded, no intent/order. |
| `paper` or `live` signal in Stage 05 | `signal` plus `*_no_order_stage05` reason suffix | Signal recorded, no paper/live side effect. |

## Runtime Config

No new environment variables, YAML files, launchd jobs, Monit rules or secret
settings were added.

Fail-closed behavior:

- unsupported evaluator specs write a blocked signal and fail the run;
- missing live profile falls back to `monitor_only`;
- `strategy_signals.expected_order_json` has a database check requiring `{}` in
  Stage 05.

## Monitoring

Added worker counter:

- `strategy_signal_total{mode,action,outcome}`

Labels are bounded to fixed enums and do not include user ids, strategy ids,
signal ids, instrument keys, raw reasons, exchange payloads or credentials.

## Logging And Redaction

No secrets, cookies, DSNs, Authorization headers, API keys, private keys,
passphrases, ciphertext, signed exchange payloads, raw exchange responses or raw
idempotency keys are intentionally logged or persisted by the new Stage 05 code.

`strategy_signals` stores strategy/run/profile ids, bounded reason codes,
reference price, source message id and an empty `expected_order_json` object.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds `signal_journal` to existing bounded dashboard response. Existing route and request contract unchanged. |
| Port / boundary interfaces | `compatible-change` | Adds `StrategySignalRepository` and optional live-runner profile/signal dependencies. |
| Persistence | `compatible-change` | Adds `strategy_signals` table and indexes only. No destructive migration. |
| Redis | `none` | Uses existing market-data input and existing strategy realtime output only; no execution streams. |
| Config | `none` | No env/YAML/default change. |
| Runtime / ops | `compatible-change` | Existing strategy live-runner records evaluator journal and bounded metric. No new process. |
| UI / browser | `compatible-change` | Adds latest signals panel to existing `/strategies` page. |
| Exchange/provider side effects | `none` | No credential decrypt, exchange read, exchange submit or paper accounting. |
| Logs/metrics/redaction | `compatible-change` | Adds bounded metric labels and secret-safe journal fields. |
| Docs | `compatible-change` | Updates Strategy docs, Stage 05 report and ledger. |

## Rollback

Rollback path before delivery:

- revert Stage 05 code/UI/docs changes;
- drop local `strategy_signals` migration if it has only been applied locally.

Rollback path after delivery:

- disable the strategy live-runner process or revert the Stage 05 commit and
  redeploy backend/web;
- existing `strategy_signals` rows are inert audit data and can remain;
- no exchange/order reconciliation is required because Stage 05 creates no
  external side effects.

## Handoff To Stage 06

Facts Stage 06 must preserve:

- Stage 05 evaluator support is intentionally narrow: one `MA(fast,slow)` spec
  and matching signal template;
- unsupported variants already fail closed with `unsupported_live_evaluator`;
- `strategy_signals` is an explanatory producer journal, not an order request;
- compatibility/readiness checks must prevent users from seeing unsupported
  variants as live-ready before run.
