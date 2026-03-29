# Backtest API v1 — `POST /backtests` (saved strategy + ad-hoc grid) (BKT-EPIC-07)

Фиксирует контракт BKT-EPIC-07: HTTP API v1 для синхронного (small-run) backtest запуска в двух режимах (saved/ad-hoc), с deterministic top-K ответом и unified deterministic 422 ошибками.

## Status

- Status: active v1 sync contract after R7-02 persisted `sync_inline` cutover.
- R7-02 storage note:
  - sync and background executions share one persisted-run storage family in Postgres,
  - successful `POST /backtests` now performs internal preflight, executes inline, and persists
    one terminal run row plus summary-only top rows in the same table family,
  - successful sync response now includes persisted run metadata:
    `run_id`, `state`, `execution_mode=sync_inline`, `engine_version`,
    `artifact_slot`, `artifact_slot_generation`, `artifact_asof_date`,
    `artifact_manifest_hash`,
  - deterministic validation failures remain canonical `422` and do not create persisted rows.
- Superseded by target-v2 contract:
  - `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
  - `docs/architecture/roadmap/base_refactor_plan.md`
  - `docs/architecture/backtest/backtest-v2-benchmarks.md`
- Historical scope kept here:
  - current `POST /backtests` sync behavior,
  - legacy `top_k_*` naming and response fields,
  - current manual split between sync and jobs endpoints.
- R1 target contract, enforced in backend validation:
  - allowed request TF: `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `1d`, `2d`, `3d`;
  - forbidden request TF in target contract: `1m`, `5m`;
  - `signals.v1.params` are `default-only`;
  - launch semantics move to auto-preflight + auto-fallback;
  - additive `top_n_default` / `top_n_max` naming is published via runtime defaults, while current request/response still keep `top_k`.

## Цель

- Дать UI один endpoint `POST /backtests`, который:
  - запускает staged backtest v1 синхронно (Stage A shortlist -> Stage B exact -> top-K),
  - поддерживает режим `saved` (по `strategy_id`) и `ad-hoc` (по template/grid),
  - возвращает достаточно данных, чтобы UI мог сохранить выбранный вариант как StrategySpec (как минимум: конкретные параметры индикаторов, signals, risk/sizing/execution),
  - возвращает deterministic ошибки (unified 422), не “плавающие” между версиями.

## Контекст

- Backtest v1 уже реализован как use-case и staged pipeline:
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- Варианты детерминированы, guards применяются в sync режиме:
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- Execution engine v1: close-fill + fee/slippage + sizing + SL/TP:
  - `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- Reporting v1: equity/trades + metrics table `|Metric|Value|`:
  - `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- Identity уже даёт authenticated principal через dependency:
  - `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- API уже имеет глобальные deterministic error handlers:
  - `apps/api/common/errors.py` (RoehubError + RequestValidationError -> deterministic 422)

## Scope

- Endpoint v1: `POST /backtests`.
- Режимы:
  - A) saved: body содержит `strategy_id` + `overrides`.
  - B) ad-hoc: body содержит `template` (grid) без сохранения Strategy.
- Auth: endpoint защищён; доступ только для authenticated user.
  - saved mode: ownership/deleted checks выполняются в backtest use-case (не в HTTP слое).
- Output policy v1:
  - для grid запуска возвращаем только top-K summary rows (default `top_k_default=300`,
    override из request),
  - ranking выбирается по 1-2 approved runtime metrics:
    `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
    `profit_factor`, `sharpe_trades`, `win_rate_pct`,
  - direction map фиксирован:
    `total_return_pct DESC`, `max_drawdown_pct ASC`,
    `return_over_max_drawdown DESC`, `profit_factor DESC`,
    `sharpe_trades DESC`, `win_rate_pct DESC`,
  - deterministic tie-break: `variant_key` (ASC),
  - `report` и `trades` в runtime summary response не materialize'ятся.
- Unified errors:
  - 422 payload детерминированный и единый через `RoehubError`.
- Reproducibility:
  - response включает `spec_hash` (saved) или `grid_request_hash` (ad-hoc) и `engine_params_hash`.

## Non-goals

- Async jobs/progress (Milestone 5).
- Public history/list/status endpoints for persisted runs.
- Дополнительные endpoints (get status/list).
- R7-03 не включает public `/backtests/runs*` history endpoints.
- R8-02 не включает full auto-fallback sync -> background orchestration.

## Ключевые решения

### 1) Один request envelope: `strategy_id` xor `template` (без explicit `mode`)

Request v1 не содержит поля `mode`. Режим определяется взаимоисключающими полями:

- saved mode: `strategy_id` задан, `template` отсутствует.
- ad-hoc mode: `template` задан, `strategy_id` отсутствует.

Причины:
- соответствует контракту BKT-EPIC-01 (use-case request DTO),
- снижает риск расхождений между API и application.

### 2) Signal params остаются в request shape, но работают как `default-only`

В template/saved mode поле `signal_grids` остаётся частью публичного request shape, но R1
фиксирует backend contract:

- `signals.v1.params` берутся из `configs/<env>/indicators.yaml`;
- request-level non-default overrides детерминированно отклоняются;
- практический UI flow: omit `signal_grids` и использовать runtime defaults как source of truth.

### 3) Typed API blocks -> canonical internal mappings

API принимает typed блоки (а не “произвольные dict”):

- `execution`: явные поля (`init_cash_quote`, `fee_pct`, `slippage_pct`, `fixed_quote`, `safe_profit_percent`)
- `risk_grid`: enable flags + axis specs (`explicit|range`) для SL/TP
- `signal_grids`: `indicator_id -> param_name -> axis spec`

На границе API эти блоки канонизируются в backtest application DTO:

- `RunBacktestTemplate.execution_params` (mapping),
- `RunBacktestTemplate.risk_grid` / `risk_params`,
- `RunBacktestTemplate.signal_grids`.

Причины:
- исключаем “плавающие” ключи (`fee` vs `fee_pct`) и делаем `variant_key`/report стабильными.

### 4) Ответ содержит variant payload для сохранения выбранного варианта

Для каждого варианта в top-K response содержит не только ключи и метрики, но и explicit payload:

- `indicator_selections`: список `{indicator_id, inputs, params}` (конкретные scalars),
- `signals`: `{indicator_id: {param: value}}`,
- `risk`: `{sl_enabled, sl_pct, tp_enabled, tp_pct}`,
- `execution`: effective execution scalars,
- `direction_mode`, `sizing_mode`.

Это позволяет UI:

- сохранить выбранный вариант как StrategySpec (минимум: индикаторы + params),
- сохранить/использовать risk/sizing/execution для следующего запуска или как defaults.

### 5) Reproducibility hashes: `spec_hash|grid_request_hash` + `engine_params_hash`

Response v1 включает:

- saved mode: `spec_hash` (детерминированный hash от saved StrategySpec payload),
- ad-hoc mode: `grid_request_hash` (детерминированный hash от canonical request payload),
- всегда: `engine_params_hash` (детерминированный hash от effective runtime settings, влияющих на результат).

Зачем:
- подтверждение воспроизводимости и защита от “тихих” изменений runtime defaults.

### 6) Успешный sync launch теперь always persisted (`execution_mode=sync_inline`)

- `POST /backtests` больше не является purely-ephemeral flow.
- После успешного inline execution backend сохраняет:
  - terminal row в `backtest_jobs`,
  - summary-only top rows в `backtest_job_top_variants`,
  - denormalized run metadata и artifact pin identity для будущих history/filter endpoints.
- Persisted top rows сохраняют только:
  - `payload_json`,
  - `summary_metrics_json`,
  - `best_tp_pct`,
  - `best_sl_pct`,
  - `report_table_md=NULL`,
  - `trades_json=NULL`.

### 7) Sync response остаётся summary-only; full report грузится on-demand

- По умолчанию `POST /backtests` возвращает ranking + payload summary без `report` body.
- Поля `rows/table_md/trades` загружаются on-demand через `POST /api/backtests/variant-report`.
- Runtime flag `backtest.reporting.eager_top_reports_enabled` остаётся только как compatibility
  knob для переходного wiring, но summary path R6-04 всё равно не строит `report`/`trades` тела.
- `top_trades_n` остаётся параметром downstream detail/report flows и не включает eager
  materialization в summary response.
- Этот summary-only shape совпадает с R7-01 persisted top-row contract:
  в storage сохраняются только `payload_json`, `summary_metrics_json`, `best_tp_pct`,
  `best_sl_pct`; `report/trades` не становятся частью persisted summary rows.

### 8) Sync cancellation: disconnect + hard deadline (кооперативно, без kill)

С 2026-02-25 sync route реализован как `async` и запускает compute в thread через `asyncio.to_thread(...)`.

Пока thread выполняет use-case, route:

- периодически проверяет `request.is_disconnected()`;
- при disconnect помечает `BacktestRunControlV1` как cancelled (`reason=client_disconnected`);
- дополнительно использует hard deadline (`BacktestRunControlV1(deadline_seconds=...)`),
  где значение берётся из runtime config `backtest.sync.sync_deadline_seconds`
  (`configs/<env>/backtest.yaml`) и прокидывается через
  `apps/api/wiring/modules/backtest.py -> build_backtests_router(...)`.

Отмена реализована кооперативно: staged loops проверяют token/checkpoint и прекращают вычисление без принудительного завершения thread/process.

### 9) Sync half-budgets, jobs full-budgets

С 2026-02-25 wiring применяет разные guard budgets:

- Sync (`RunBacktestUseCase`) получает половинные лимиты:
  - `floor(backtest.guards.max_variants_per_compute / 2)`;
  - `floor(backtest.guards.max_compute_bytes_total / 2)`.
- Jobs path сохраняет полные лимиты из `backtest.guards.*`.

HTTP response schema при этом не меняется.

### 10) CPU knob через Numba threads

С 2026-02-25 в `backtest.yaml` добавлен `backtest.cpu.max_numba_threads`.

- Значение валидируется fail-fast на старте.
- В sync и jobs перед run attempt вызывается `numba.set_num_threads(...)`.
- Это текущий runtime CPU knob v1 (без новых native dependencies).

## Endpoint v1: `POST /backtests`

### Request (v1)

Common fields:

- `time_range`: `{start, end}` (UTC, half-open `[start, end)`).
- `warmup_bars?`, `top_k?`, `preselect?`, `top_trades_n?`, `ranking?`.

Mode selection:

- `strategy_id` (saved) xor `template` (ad-hoc).

Saved mode:

- `strategy_id: uuid`
- `overrides?`:
  - `direction_mode?`, `sizing_mode?`
  - `execution?`
  - `risk_grid?`
  - `signal_grids?` (shape preserved, non-default values rejected by `signals.v1.params=default-only`)

Ad-hoc mode:

- `template`:
  - `instrument_id: {market_id, symbol}`
  - `timeframe` (`15m|30m|1h|2h|4h|6h|8h|1d|2d|3d`; `1m`/`5m` rejected)
  - `indicator_grids[]` (grid specs)
  - `direction_mode?`, `sizing_mode?`
  - `execution?`, `risk_grid?`, `signal_grids?` (`default-only`; omission uses server defaults)

Axis spec shape (reused across grid specs):

- explicit: `{ "mode": "explicit", "values": [ ... ] }`
- range: `{ "mode": "range", "start": 1.0, "stop_incl": 5.0, "step": 0.1 }`

Percent units:

- `fee_pct=0.075` means `0.075%`.
- `slippage_pct=0.01` means `0.01%`.
- `sl_pct=3.0` means `3%`.

Ranking override:

- `ranking.primary_metric` и `ranking.secondary_metric?` используют approved literals:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`,
  `sharpe_trades`, `win_rate_pct`.
- `secondary_metric` не может дублировать `primary_metric`.
- Если ranking не задан, применяются runtime defaults:
  `primary_metric=total_return_pct`, `secondary_metric=null`.

### Response (v1)

Response содержит:

- `schema_version=1`
- `mode: "saved"|"template"`
- `instrument_id`, `timeframe`, `strategy_id?`
- `warmup_bars`, `top_k`, `preselect`, `top_trades_n`
- persisted sync run metadata:
  - `run_id`
  - `state`
  - `execution_mode` (`sync_inline`)
  - `engine_version`
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_asof_date`
  - `artifact_manifest_hash`
- reproducibility hashes:
  - `spec_hash?` or `grid_request_hash?`
  - `engine_params_hash`
- `variants[]` (length `<= top_k`), отсортировано:
  - primary/secondary: по выбранному ranking contract
  - tie-break: `variant_key` asc

Каждый `variants[i]` содержит:

- `variant_index`, `variant_key`, `indicator_variant_key`
- `total_return_pct`
- `report`:
  - всегда `null` в runtime summary response R6-04
  - полный `rows + table_md + optional trades` загружается только через
    `POST /api/backtests/variant-report`
- `payload` (explicit parameters for saving):
  - `indicator_selections[]`
  - `signals`
  - `risk`
  - `execution`
  - `direction_mode`, `sizing_mode`

## Ошибки и статус-коды

- `401` — unauthenticated (identity dependency).
- `422` — `RoehubError(code="validation_error")`:
  - invalid payload
  - guards/preflight budget exceeded
  - invalid time range / no market data
- `404` — `RoehubError(code="not_found")`:
  - saved strategy missing or deleted
- `403` — `RoehubError(code="forbidden")`:
  - saved strategy принадлежит другому user
- `409` — `RoehubError(code="conflict")`:
  - request mode conflict (если будет нарушен контракт)

Порядок validation errors детерминирован (см. `apps/api/common/errors.py`).

## Wiring / Composition

FastAPI wiring v1:

- `apps/api/routes/backtests.py` — thin route: DTO mapping -> use-case call -> response mapping.
- `apps/api/wiring/modules/backtest.py` — composition:
  - `CandleFeed` (reuse indicators `MarketDataCandleFeed`),
  - `IndicatorCompute` (reuse indicators compute adapter),
  - `BacktestStrategyReader` adapter (ACL over StrategyRepository),
  - `BacktestJobRepository` over the unified Postgres storage family for persisted
    `sync_inline` writes,
  - `BacktestGridDefaultsProvider` (reads `configs/<env>/indicators.yaml` defaults),
  - `BacktestRuntimeConfig` from `configs/<env>/backtest.yaml`.

Fail-fast:

- модуль backtest загружается и валидируется на старте (как strategy/identity/indicators).

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-4-epics-v1.md` — BKT-EPIC-07.
- `docs/architecture/roadmap/base_milestone_plan.md` — UX/flow и hashes.
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — request DTO и ownership rule.
- `docs/architecture/api/api-errors-and-422-payload-v1.md` — unified errors contract.

API:
- `apps/api/routes/backtests.py`
- `apps/api/common/errors.py`
- `apps/api/main/app.py`

Backtest:
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/dto/run_backtest.py`
- `src/trading/contexts/backtest/application/ports/strategy_reader.py`

Strategy:
- `src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py`

## Как проверить

После реализации EPIC-07:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: большие payload’ы (top-K * report rows + trades). Митигатор: trades только для top N.
- Риск: несоответствие saved Strategy индикаторного payload формату backtest grid. Митигатор: явный ACL mapper + строгая deterministic validation с 422.
