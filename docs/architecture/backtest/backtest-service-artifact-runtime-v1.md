# Backtest Service Artifact Runtime v1

Документ фиксирует целевую архитектуру artifact-backed backtest service, его публичный API, job flow, lazy trades detail и benchmark-gate по итерациям.

## Статус

Proposed target architecture.

Этот документ проектируется поверх:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`;
- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`;
- текущего trusted artifact publisher/precompute scope в `backtest_artifacts`;
- текущей DDD / ports-and-adapters структуры репозитория.

Старые backtest runtime docs и удаленный runtime compute path не восстанавливаются.

## Цель

Построить production backtest service для сайта и публичного API, который:

- всегда запускает backtest как persisted job;
- читает precomputed `.npy` artifacts (`prices`, `signals`, `mappings`, `hit_times/15m`);
- принимает координаты рынка, расчетный период, timeframe, набор индикаторов с `source` и `window` grid;
- поддерживает `no-risk` и `tp/sl grid` режимы;
- возвращает persisted top N summary;
- лениво пересчитывает trades для выбранного `variant_key` по запросу UI/API;
- развивается итерационно, где каждая итерация проходит benchmark на `Mac Studio`;
- считается завершенной только если каждый pipeline segment держит не ниже 90% baseline по скорости, памяти и CPU-метрикам.

## Контекст

Текущий backtest runtime официально сброшен: активной доверенной частью считаются artifact publisher/precompute, strict manifests, `.npy` layout, signal rules, hit-times precompute, path/current-pointer adapters и job storage guard.

API и Web сейчас не монтируют backtest routes/pages. Новый сервис должен стартовать поверх artifacts и не тащить legacy runtime kernels/scorers/shortlists.

`hit_times` target model: `hit_times/15m`. Семантически таблицы обслуживают риск-исполнение через precomputed hit-time data, а request выбирает подмножество из заранее опубликованного достаточно широкого TP/SL grid.

## Охват

- Публичный пользовательский API для jobs, top results, variant summary и lazy trades.
- UI consumption через тот же публичный API.
- Application service pipeline для artifact-backed runtime.
- Job storage and result summary contract.
- Lazy trades cache на 1-2 дня.
- Benchmark workspace и обязательный формат evidence per iteration.
- Target module boundaries and dependency direction.

## Что не входит

- Separate internal API. На v1 он не нужен и не планируется.
- Восстановление удаленных legacy docs `deep-research-report.md` и `backtest-core-refactor-prompt-pack-v1.md`.
- Хранение старых artifact versions как immutable historical snapshots.
- Portfolio/multi-instrument backtest.
- Live execution/order routing.
- ML/inference path.
- Browser UI implementation details beyond API consumption contract.

## Ключевые решения

### 1) Backtest всегда создается как job

`POST /backtests/jobs` создает persisted job. Даже если job фактически завершается быстро, внешний контракт остается job-based.

Последствия:

- UI и API используют один путь запуска;
- результат всегда попадает в history;
- progress/cancel/list/top работают одинаково для коротких и длинных расчетов;
- нет отдельного sync endpoint, который расходится с background execution.

### 2) UI использует тот же публичный API

Backtest UI не получает отдельный internal API. Web SSR/HTMX/JS ходит через same-origin `/api/*` proxy к тем же authenticated routes, что и внешний API.

Риск:

- публичный API должен быть достаточно удобным для UI, иначе UI начнет требовать private shortcuts.

Митигация:

- добавить публичные runtime-defaults/preflight/read endpoints вместо internal API.

### 3) Artifact versions не хранятся, но historical prefix должен быть immutable

Старые версии artifacts не сохраняются как отдельные immutable snapshots. Дизайн исходит из инварианта:

- published historical prefix не переписывается;
- tail только дополняется или безопасно обновляет еще незафиксированный хвост;
- для одного и того же `{coordinates, timeframe, [start, end), request params}` historical результат остается детерминированным.

Job все равно сохраняет artifact identity/watermark metadata:

- `artifact_slot`;
- `artifact_slot_generation`;
- `artifact_manifest_hash`;
- `artifact_asof_date`.

В v1 эти поля нужны не для восстановления старой версии artifact store, а для:

- audit/evidence;
- cache key для lazy trades;
- диагностики, каким published state пользовался job;
- защиты от неявного drift в benchmark/reporting.

Риск:

- если publisher когда-либо перепишет historical prefix, старые jobs и lazy trades могут стать невоспроизводимыми.

Инвариант release gate:

- любое изменение publisher, которое может переписать historical prefix, считается contract change и требует отдельного migration/compatibility design.

### 4) `risk.mode` вместо отдельного execution profile

Термин `execution profile` не нужен как публичный v1-концепт. Пользовательский контракт проще:

```json
{
  "risk": {"mode": "none"}
}
```

или:

```json
{
  "risk": {
    "mode": "tp_sl_grid",
    "tp": {"start_pct": 0.5, "stop_pct": 5.0, "step_pct": 0.5},
    "sl": {"start_pct": 0.5, "stop_pct": 3.0, "step_pct": 0.5}
  }
}
```

`risk.mode = "none"` означает:

- вход по consensus signal;
- выход по signal reversal/exit или `close_on_end`;
- `best_tp_pct = null`;
- `best_sl_pct = null`;
- fees/slippage/sizing все равно применяются.

`risk.mode = "tp_sl_grid"` означает:

- Stage B выбирает лучший TP/SL cell из request grid;
- request grid обязан быть покрыт published `hit_times/15m` grid;
- если request grid не покрыт artifacts, API возвращает deterministic 422.

### 5) Sizing modes входят в публичный request

Поддерживаемые v1 sizing modes:

- `all_in`: вся доступная quote-сумма в сделку;
- `fixed_quote`: фиксированная quote-сумма;
- `fixed_equity_pct`: процент текущего equity;
- `fixed_equity_pct_min_quote`: процент equity, но не меньше заданной quote-суммы;
- `fixed_equity_pct_max_quote`: процент equity, но не больше заданной quote-суммы.

`profit_lock` проектируется как надстройка над sizing:

```json
{
  "profit_lock": {
    "enabled": true,
    "safe_profit_percent": 30.0
  }
}
```

Benchmark и tests должны покрывать все sizing modes.

### 6) `variant_key` человекочитаемый и уникальный внутри каждого запуска

Нужны два идентификатора:

- `variant_hash`: стабильный SHA-256 от canonical JSON параметров варианта без `job_id`;
- `variant_key`: публичный человекочитаемый ключ, уникальный для конкретного job.

Target format:

```text
job_<job_short>__<readable_slug>__vh_<variant_hash_short>
```

Пример:

```text
job_f7d2c378__dema_close_w192__hma_hlc3_w64__risk_none__vh_a13f09c2
```

Последствия:

- одинаковая комбинация параметров в двух jobs имеет одинаковый `variant_hash`, но разный `variant_key`;
- `variant_key` уникален даже если два пользователя запускают одинаковые параметры в одну секунду;
- UI может показывать readable slug, а backend использует canonical hash для validation/cache.

### 7) Persisted top N summary, trades считаются лениво

Job completion сохраняет только top N summary.

Lazy trades endpoint:

```http
POST /backtests/jobs/{job_id}/variants/{variant_key}/trades
```

Поведение:

- проверяет ownership;
- читает persisted job request snapshot и selected variant params;
- читает current artifact data по historical-prefix invariant и stored artifact metadata;
- пересчитывает trades только для одного варианта;
- возвращает trades + chart overlay payload;
- может сохранить cache на 1-2 дня.

Cache recommendation:

- metadata в Postgres;
- payload в локальном object/file cache под `/opt/roehub/state/backtest/trades_cache`;
- TTL default: 48h;
- cache key включает `job_id`, `variant_key`, `variant_hash`, `request_hash`, `engine_params_hash`, `artifact_manifest_hash`.

Postgres-only JSONB допустим для малых payloads, но v1 должен избегать неограниченного раздувания основной БД.

## Публичный API v1

### `POST /backtests/jobs`

Создает job.

Request shape:

```json
{
  "coordinates": {
    "exchange": "binance",
    "market_type": "spot",
    "symbol": "BTCUSDT"
  },
  "timeframe": "15m",
  "time_range": {
    "start": "2020-01-11T20:08:00Z",
    "end": "2026-04-11T20:08:00Z"
  },
  "indicators": [
    {
      "indicator_id": "ma.dema",
      "sources": ["close", "high", "hlc3"],
      "window": {"start": 5, "stop": 200, "step": 1}
    }
  ],
  "risk": {"mode": "none"},
  "execution": {
    "direction_mode": "long-short",
    "fee_rate": 0.00075,
    "slippage_rate": 0.0001,
    "initial_cash_quote": 10000.0,
    "sizing": {
      "mode": "fixed_equity_pct",
      "equity_pct": 10.0
    },
    "profit_lock": {
      "enabled": false
    },
    "close_on_end": true
  },
  "ranking": {
    "primary_metric": "total_return_pct",
    "direction": "desc"
  },
  "top_n": 100
}
```

Defaults:

- missing execution fields use runtime config defaults;
- missing ranking uses `total_return_pct desc`;
- missing `top_n` uses config default;
- missing `risk` should be rejected unless product decides `risk.mode = "none"` as default.

### `GET /backtests/jobs/{job_id}`

Returns status, progress, hashes, selected artifact metadata, and terminal summary counts.

### `GET /backtests/jobs/{job_id}/top`

Returns persisted top N summary rows.

Each row includes:

- `rank`;
- `variant_key`;
- `variant_hash`;
- `summary_metrics`;
- `best_tp_pct`;
- `best_sl_pct`;
- compact readable params;
- links/actions for lazy detail.

### `GET /backtests/jobs/{job_id}/variants/{variant_key}`

Returns one persisted variant summary and full parameter decomposition.

### `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`

Computes or returns cached lazy trades for one variant.

## Runtime pipeline

### Stage 0: request normalization

- validate coordinates;
- validate timeframe excludes `1m` and `5m`;
- validate `[start, end)`;
- validate indicator ids against `configs/prod/indicators.yaml` loaded through registry/defaults;
- materialize requested `source` and `window` row selections;
- validate request TP/SL grid is covered by published artifact grid when `risk.mode = "tp_sl_grid"`;
- apply execution defaults;
- compute canonical request/config hashes.

### Stage 1: job creation

- create `backtest_jobs` row with canonical request snapshot;
- record artifact metadata/watermark for audit and cache identity;
- state starts as `queued`.

### Stage 2: artifact load

- resolve artifact root from config;
- read strict `current.yaml` and slot `manifest.yaml`;
- load `.npy` via `np.load(..., mmap_mode="r")`;
- load `prices/<tf>`, `prices/1m`, `mappings/<tf>`, requested `signals/<tf>/<indicator_id>`, and `hit_times/15m` only when risk mode requires it;
- slice `[start, end)` on the signal timeframe and corresponding execution window.

### Stage 3: row pools

- map `{indicator_id, source, window}` to artifact row ids using manifest grid contract;
- copy only selected rows into contiguous working arrays when kernels need contiguous memory;
- preserve source-major/window-minor ordering from artifacts.

### Stage 4: prefilter and combo planning

Notebook-derived patterns:

- single-indicator row prefilter;
- combo proxy prefilter;
- bounded combo chunks;
- trade-list-first exact scoring;
- fast-vs-reference self-check on bounded subset.

### Stage 5A: no-risk exact scoring

For `risk.mode = "none"`:

- build compact trade list from consensus signals;
- execute no-risk scoring with fees/slippage/sizing/profit-lock;
- rank by selected metric;
- persist top N summary.

### Stage 5B: TP/SL exact scoring

For `risk.mode = "tp_sl_grid"`:

- build compact trade list from consensus signals;
- use `hit_times/15m` and request TP/SL subset;
- run fast monotone TP/SL search;
- run bounded self-check against reference path per benchmark/test policy;
- persist best TP/SL cell per variant in top N summary.

### Stage 6: lazy trades

- recompute exact trades for one `variant_key`;
- cache result for 48h;
- return chart-ready payload.

## Целевая структура модулей

Planned target structure:

- `src/trading/contexts/backtest/domain/` — job aggregate, request/value objects, variant identity, execution/sizing value objects.
- `src/trading/contexts/backtest/application/use_cases/` — create job, read status/top, read variant, compute lazy trades.
- `src/trading/contexts/backtest/application/services/v2/` — artifact-backed runtime pipeline, row selection, combo planning, scoring orchestration.
- `src/trading/contexts/backtest/application/ports/` — artifact loaders, job repositories, cache storage, metrics, current user.
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/` — strict artifact readers over `backtest_artifacts` contracts.
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/` — jobs/top/cache metadata repositories.
- `apps/api/routes/backtests.py` — public API routes.
- `apps/api/wiring/modules/backtest.py` — composition root.
- `apps/web/templates/backtests_*.html` and `apps/web/dist/backtest_ui.js` — UI integration in later iteration.

Dependency direction:

```text
apps/api routes
  -> backtest application use cases
    -> backtest domain/value objects
    -> application ports
      <- outbound adapters: artifacts_fs, postgres, cache_fs, metrics
```

`backtest` may consume `backtest_artifacts` contracts through an adapter/ACL, but runtime orchestration should live in `backtest`, not in publisher/precompute code.

## Benchmark policy

Benchmark execution is allowed only on `Mac Studio`.

Baseline sources:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`;
- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`.

Each implementation iteration must record:

- code version / branch / commit;
- artifact config and artifact root;
- notebook baseline used;
- request fixture;
- service warmup metrics;
- per-stage runtime metrics;
- speed ratio vs baseline;
- peak RSS / memory delta;
- CPU time, process CPU percent, thread count, effective Numba threads, system load;
- pass/fail against 90% threshold;
- correctness/parity result.

Warmup policy:

- service warmup is a first-class measured pipeline segment;
- user-facing runtime benchmark is measured after warmup;
- both warmup and warm runtime must stay within accepted 90% envelope for their segment.

Required benchmark segments:

1. `service_warmup`
2. `artifact_manifest_load`
3. `artifact_array_mmap_load`
4. `time_range_slice`
5. `signal_row_selection`
6. `stage_a_prefilter`
7. `combo_prefilter`
8. `no_risk_exact_scoring` or `tp_sl_exact_scoring`
9. `persist_top_n`
10. `lazy_trades_compute`
11. `lazy_trades_cache_hit`

Benchmark records live under:

- `docs/architecture/backtest/benchmark_iterations/`

## Операционные аспекты

Metrics:

- `backtest_jobs_created_total{risk_mode}`
- `backtest_jobs_completed_total{status,risk_mode}`
- `backtest_job_duration_seconds{risk_mode}`
- `backtest_stage_duration_seconds{stage,risk_mode}`
- `backtest_stage_cpu_seconds_total{stage,risk_mode}`
- `backtest_stage_peak_rss_bytes{stage,risk_mode}`
- `backtest_lazy_trades_requests_total{cache_status}`
- `backtest_lazy_trades_duration_seconds{cache_status}`
- `backtest_artifact_runtime_load_duration_seconds{family}`
- `backtest_artifact_runtime_manifest_hash_info`

Failure behavior:

- invalid request returns deterministic 422;
- missing artifact family returns deterministic runtime failure on job;
- request TP/SL not covered by published grid returns deterministic 422 before job execution if possible;
- lazy trades cache failure must not fail the trades response if recompute succeeds;
- benchmark failure blocks the current iteration from being considered complete.

## План внедрения

### Iteration 0: docs and benchmark harness

- finalize this architecture document;
- create benchmark iteration workspace;
- extract notebook fixtures into reproducible benchmark inputs;
- define Mac Studio benchmark command contract.

Exit criteria:

- benchmark folder exists;
- baseline notebook scenarios are named;
- service stages and metrics are fixed.

### Iteration 1: artifact runtime load

- implement strict artifact context resolver;
- implement mmap loaders for `prices`, `signals`, `mappings`, `hit_times/15m`;
- implement `[start, end)` slicing and row selection.

Benchmark gate:

- `artifact_manifest_load`, `artifact_array_mmap_load`, `time_range_slice`, `signal_row_selection`.

### Iteration 2: no-risk job path

- implement job create/status/top;
- implement `risk.mode = "none"` scorer;
- persist top N summary.

Benchmark gate:

- `service_warmup`, `stage_a_prefilter`, `combo_prefilter`, `no_risk_exact_scoring`, `persist_top_n`.

### Iteration 3: TP/SL grid path

- validate request TP/SL subset against artifact grid;
- implement hit-times backed Stage B scoring;
- persist `best_tp_pct`, `best_sl_pct`.

Benchmark gate:

- `tp_sl_exact_scoring` vs notebook-derived baseline.

### Iteration 4: lazy trades detail

- implement variant lookup;
- implement lazy trades recompute;
- implement 48h cache;
- return chart-ready payload.

Benchmark gate:

- `lazy_trades_compute`, `lazy_trades_cache_hit`.

### Iteration 5: UI integration

- use same public API;
- show job progress/top N;
- implement `show trades`;
- render trades on candle chart.

Verification:

- browser-visible QA through runtime browser surface.

## Как проверить

Static checks after docs updates:

```bash
python -m tools.docs.generate_docs_index --check
```

Implementation-phase checks will be added per iteration. Benchmark checks must run on `Mac Studio`, not on local non-production-equivalent hosts.

## Риски и открытые вопросы

- Risk: no immutable artifact version retention means reproducibility depends on strict historical-prefix immutability.
- Risk: `tests/notebook_tests/new_engine/02...` currently has no executed notebook outputs; baseline extraction must execute it on Mac Studio and record results.
- Risk: `hit_times/15m` contract must be reconciled across docs, contracts, publisher and notebooks before implementation uses it.
- Risk: full `configs/prod/indicators.yaml` catalog expansion can create large combo spaces; prefilter and guards are not optional.
- Question: exact chart payload shape for UI candles/trades overlay remains a UI iteration decision.
