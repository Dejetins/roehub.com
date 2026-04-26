# Backtest Service Artifact Runtime v1

Companion/reference copy for the artifact-backed backtest service architecture.
Canonical implementation source is
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`; if the
two documents diverge, the Russian document wins for implementation.

## Статус

Reference target architecture for planning implementation. Runtime service еще
не реализован; notebook и benchmark evidence ниже определяют production
prototype, которому сервис обязан соответствовать. The canonical target contract
for implementation is the Russian document:
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`.

Этот документ проектируется поверх:

- canonical notebook prototype:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
- canonical benchmark evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- human-readable benchmark summary:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`;
- текущего trusted artifact publisher/precompute scope в `backtest_artifacts`;
- текущей DDD / ports-and-adapters структуры репозитория.

Этот документ является companion/reference copy для нового `Backtest Service
Artifact Runtime v1`; implementation source of truth остается в `.ru.md`.

Старые backtest runtime docs и удаленный runtime compute path не восстанавливаются.
Roadmap-формулировки, которые противоречат canonical `.ru.md` runtime document, считаются
historical/compatibility context для v1, а не source of truth. Это включает:

- `POST /backtests` как основной create endpoint вместо `POST /backtests/jobs`;
- `runs` vocabulary вместо `jobs` vocabulary;
- any hit-times wording that contradicts target `hit_times/15m`;
- public `execution profile` vocabulary вместо `risk.mode`;
- старые benchmark notebooks под `tests/notebook_tests/new_engine/*` как
  canonical algorithm source.

`docs/architecture/roadmap/backtest-refactor-final-plan-v2.md` упоминается старым
roadmap как source of truth, но отсутствует в текущем рабочем дереве. Пока он не
восстановлен отдельным ADR/документом, он не переопределяет этот v1 contract.
Существующие code/schema элементы со старым словарем рассматриваются как
compatibility inputs: adapters may translate them, but public v1 contract stays
defined by the canonical `.ru.md` runtime document.

## Цель

Построить production backtest service для сайта и публичного API, который:

- всегда запускает backtest как persisted job;
- читает precomputed `.npy` artifacts (`prices`, `signals`, `mappings`, `hit_times/15m`);
- принимает координаты рынка, расчетный период, timeframe, набор индикаторов с `source` и `window` grid;
- поддерживает `no-risk` и `tp/sl grid` режимы;
- возвращает persisted top N summary;
- лениво пересчитывает trades для выбранного `variant_key` по запросу UI/API;
- развивается итерационно, где каждая итерация проходит benchmark на `Mac Studio`;
- считается завершенной только если каждый pipeline segment держит не ниже 90%
  target baseline по скорости, памяти и CPU-метрикам из canonical benchmark
  evidence `2026-04-26_engine_test_btcusdt_15m`.

## Контекст

Текущий backtest runtime официально сброшен: активной доверенной частью считаются artifact publisher/precompute, strict manifests, `.npy` layout, signal rules, hit-times precompute, path/current-pointer adapters и job storage guard.

Новый сервис должен стартовать поверх artifacts и не тащить legacy runtime kernels/scorers/shortlists. Если в репозитории остаются routes, templates, worker wiring или runtime modules со старым словарем, они должны быть явно классифицированы как `active`, `compatibility-only` или `obsolete` перед использованием в реализации.

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
    "tp": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
    "sl": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5}
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

- `variant_key`: public route/UI key, человекочитаемый и уникальный внутри конкретного job;
- `variant_hash`: стабильный SHA-256 от canonical JSON параметров варианта без `job_id`;
- `indicator_variant_hash`: стабильный SHA-256 от indicator/source/window части, если нужен отдельный ключ для indicator-only identity.

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

Persistence transition:

- current/legacy DB/domain code может ожидать 64-char SHA в поле `variant_key`;
- пока schema не мигрирована, persistence adapter должен хранить canonical `variant_hash`
  в legacy SHA-only field и отдавать наружу public `variant_key`;
- target storage должен иметь отдельное поле для public route key, например
  `public_variant_key`, либо read-model mapping, который однозначно восстанавливает его
  из `{job_id, readable_slug, variant_hash}`;
- public API response всегда использует `variant_key` как readable route key и
  `variant_hash` как stable identity hash.

Adapter invariant:

- inbound route accepts only public `variant_key`;
- application layer resolves it to exactly one persisted top-N row inside `job_id`;
- cache and lazy recompute keys include both public `variant_key` and `variant_hash`;
- direct lookup by raw storage SHA is not a public v1 contract.

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

Cache topology:

- v1 assumes single API/worker host or sticky local cache semantics;
- cache miss is normal and must trigger deterministic recompute for one variant;
- local cache failure must not fail the response if recompute succeeds;
- cache storage is an outbound port so it can be replaced by shared object storage;
- if API/worker deployment becomes multi-host without sticky routing, shared object
  storage becomes required before production scale-out.

## Публичный API v1

### `POST /backtests/jobs`

Создает job.

Headers:

- optional `Idempotency-Key`;
- without `Idempotency-Key` every valid request creates a new job;
- with `Idempotency-Key`, retrying the same canonical request for the same user within
  configured TTL returns the original job;
- reusing the same key with a different canonical request returns deterministic `409`.

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
    "direction_mode": "long_short_reversal",
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

Response:

- `201 Created` for a new job;
- `200 OK` for idempotent replay of an existing job;
- payload includes `job_id`, `state`, `request_hash`, selected artifact metadata,
  links for `status`, `top`, `cancel`, and `runtime-defaults`.

### `GET /backtests/jobs`

Returns authenticated user's job history using keyset pagination.

Query params:

- `state`;
- `risk_mode`;
- `created_before`;
- `limit`;
- `cursor`.

Rows include `job_id`, `state`, progress snapshot, coordinates, timeframe, time range,
ranking metric, requested `top_n`, created/started/finished timestamps and links.

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

### `POST /backtests/jobs/{job_id}/cancel`

Requests cancellation. The endpoint is idempotent:

- active jobs move toward `cancelled`;
- already terminal jobs return current terminal state;
- cancellation does not delete persisted top-N summary already committed.

### `GET /backtests/runtime-defaults`

Returns public defaults and limits used to build valid requests:

- supported timeframes;
- supported and maximum indicator arity (`backtest.max_indicator_arity = 10`);
- available ranking metrics;
- execution defaults;
- sizing mode defaults and required fields;
- direction modes;
- default and max `top_n`;
- guardrail limits used by preflight.

### `POST /backtests/preflight`

Validates and normalizes a request without creating a job.

Returns:

- normalized effective request;
- `request_hash`;
- selected artifact metadata;
- estimated indicator rows, combinations and TP/SL cells;
- estimated cost class;
- blocking validation errors or warnings.

Preflight is advisory for UX and API clients. `POST /backtests/jobs` repeats all
validation and remains the authoritative create path.

### Error catalog

| HTTP | Code | Retryable | Meaning |
|---:|---|---|---|
| 400 | `backtest.invalid_json` | no | Payload is not a JSON object or cannot be parsed. |
| 401 | `auth.required` | yes after auth | User is not authenticated. |
| 403 | `backtest.forbidden` | no | User does not own the job or variant. |
| 404 | `backtest.not_found` | no | Job or variant is not visible to the user. |
| 409 | `backtest.idempotency_key_conflict` | no | Same `Idempotency-Key` was reused with different canonical request. |
| 409 | `backtest.job_not_cancellable` | no | Cancellation is not valid for the current state. |
| 422 | `backtest.invalid_request` | no | Coordinates, timeframe, range, indicator grid or execution settings are invalid. |
| 422 | `backtest.tp_sl_grid_not_covered` | no | Requested TP/SL grid is not covered by published `hit_times/15m`. |
| 422 | `backtest.request_too_expensive` | no | Request exceeds configured cost/combination limits. |
| 429 | `backtest.rate_limited` | yes | User or service quota is exhausted. |
| 503 | `backtest.artifacts_unavailable` | yes | Required artifact family/current pointer is unavailable. |
| 503 | `backtest.queue_saturated` | yes | Worker queue cannot accept more jobs within limits. |

## Runtime pipeline

Runtime has two stage vocabularies:

- lifecycle stages: request/job/queue states visible to API and persistence;
- benchmark stages: notebook-compatible measured compute stages from canonical
  `btcusdt_15m_research_engine.ipynb`.

The production implementation must preserve the algorithmic semantics of the
notebook and expose benchmark records with the same timer names as
`benchmark_results.json`. Old five-stage vocabulary with `count_trades` is
superseded: trade counting remains inside self-check/reference paths and exact
scoring outputs, but it is not a production pipeline stage.

### Algorithm scope and backends

Canonical prototype validates:

- coordinates: `binance` / `spot` / `BTCUSDT`;
- timeframe: `15m`;
- period semantics: `[start, end)` by 15m `open_time`;
- target public arity: 1..10 indicators for both `risk.mode = "none"` and
  `risk.mode = "tp_sl_grid"`;
- canonical production acceptance benchmark covers arity 1..7; if the service
  reaches 90% of the target for arity 1..7, the algorithm transfer to production
  runtime is considered successful;
- arity 8..10 are allowed by request validation but controlled by cost guardrails;
  a separate Mac Studio benchmark iteration for them is needed only before
  expanding production budget tiers, not for v1 completion;
- benchmark directions: `long_only`, `long_short_reversal`;
- risk modes: `none`, `tp_sl_grid`;
- TP/SL target grid: `2.0..25.0` inclusive, step `0.5`
  (`47 x 47 = 2209` cells).

Backend registry:

| Backend | Risk mode | Arity | Role |
|---|---|---:|---|
| `event_segments_2_no_risk` | `none` | 2 | Default specialized two-indicator no-risk backend. |
| `streaming_2_no_risk` | `none` | 2 | Fallback/parity backend, not default production target. |
| `event_segments_n_no_risk` | `none` | 1, 3..10 | Generic no-risk backend. |
| `event_segments_n_tp_sl_15m_grid` | `tp_sl_grid` | 1..10 | Generic risk-on backend backed by `hit_times/15m`. |

Direction semantics:

- `long_only`: raw consensus `+1` opens/holds long; raw `0` or `-1`
  closes an open long; short trades are never opened;
- `long_short_reversal`: raw consensus `+1` opens/holds long; raw `-1`
  opens/holds short; opposite signal closes and reverses.

### Lifecycle stage: request normalization

- validate coordinates;
- validate timeframe excludes `1m` and `5m`;
- validate `[start, end)`;
- validate indicator ids against `configs/prod/indicators.yaml` loaded through
  registry/defaults;
- validate `source` and `window` ranges, then materialize row selections;
- validate `direction_mode`, `sizing`, `profit_lock`, fees, slippage,
  `initial_cash_quote` and `close_on_end`;
- validate request TP/SL grid is covered by published artifact grid when
  `risk.mode = "tp_sl_grid"`;
- apply execution defaults;
- compute canonical `request_hash`, result-affecting config hash and estimated
  request cost.

Notebook method equivalents:

- `validate_request_indicators`;
- `canonical_json_hash`;
- `row_ids_for_sources`.

### Lifecycle stage: job creation

- create `backtest_jobs` row with canonical request snapshot;
- record artifact metadata/watermark for audit and cache identity;
- state starts as `queued`;
- worker pins the resolved artifact root and manifest hashes for the job.

### Benchmark stage: `sample_warmup` / `service_warmup` / `numba_warmup`

Warmup is measured separately from user-facing runtime. Canonical benchmark uses
sample warmup, not a full dry-run:

- warmup rows per indicator: `min(2, rows_per_indicator)`;
- same arity, risk mode, direction mode and backend as the measured run;
- JIT compilation and first-touch array costs are attributed to warmup;
- measured `total_without_warmup` excludes warmup.

The service must record warmup metrics and compare them to the canonical target,
but warmup is not added to `total_without_warmup`.

### Benchmark stage: `load_hit_times`

Risk-on only. Notebook method: `load_tp_sl_hit_times_15m`.

This stage:

- reads `hit_times/15m/manifest.yaml`;
- reads `tp_values.f32.npy`, `sl_values.f32.npy`;
- maps requested TP/SL percentages to artifact indexes;
- loads selected rows from `long_tp.u32.npy`, `long_sl.u32.npy`,
  `short_tp.u32.npy`, `short_sl.u32.npy`;
- copies the selected subset into contiguous arrays for kernels;
- precomputes fee-adjusted log factors for long and short TP/SL outcomes;
- records `hit_times_manifest_hash`.

### Benchmark stage: `tp_sl_grid_validation`

Risk-on only. This is timed separately from hit-time array loading.

Validation:

- request values are interpreted as percentages and converted to decimal levels;
- every requested TP and SL level must match exactly one published artifact value
  using bounded float tolerance;
- missing levels fail deterministically before compute with
  `422 backtest.tp_sl_grid_not_covered`;
- target benchmark grid is `2.0..25.0` inclusive with `0.5` step.

### Benchmark stage: `prepare_pools`

Notebook methods:

- `extract_signal_rows`;
- `prefilter_indicator_rows`;
- `fused_row_prefilter_stats`;
- `topk_fraction_idx`;
- `build_signal_segments`;
- `fill_signal_segments_i8`;
- `prepare_indicator_pool`;
- `prepare_indicator_pools`.

This stage:

- resolves artifact root through trusted configuration/current pointer;
- reads slot `manifest.yaml` and requested indicator manifests;
- loads `.npy` arrays with `np.load(..., mmap_mode="r")`;
- loads `prices/<tf>`, `prices/1m`, `mappings/<tf>` and requested
  `signals/<tf>/<indicator_id>/signals.i8.npy`;
- slices 15m bars by `[start, end)` using `open_time`;
- derives 15m return intervals from close prices;
- derives 15m-to-1m execution mapping for no-risk mode:
  signal at 15m bar `t` enters at the next 15m bar open mapped to 1m;
- copies only requested signal rows into contiguous `int8` matrices;
- applies row prefilter per indicator:
  - `nonzero`: number of non-zero signal intervals;
  - `proxy`: dot-product-like directional return proxy;
  - `change_count`: number of signal change points;
  - `adjusted = proxy - fee_rate * nonzero`;
  - keeps top fraction after `min_nonzero`;
- builds per-row metadata `{indicator_id, row_id, source, window}`;
- builds compressed signal segments:
  `starts`, `ends`, `values`, `counts`, `change_count`;
- returns indicator pools with `trade_T`, `eval_T`, `segments`, row ids,
  scores and metadata.

### Benchmark stage: `build_exact_context`

Notebook method: `build_segment_stack`.

This stage prepares arity-first segment arrays used by generic exact kernels:

- `starts[arity, max_rows, max_segments]`;
- `ends[arity, max_rows, max_segments]`;
- `values[arity, max_rows, max_segments]`;
- `counts[arity, max_rows]`.

For no-risk arity 2 with the specialized backend, the service may read segments
directly from each pool and `build_exact_context` can be near zero. For generic
no-risk and all TP/SL risk-on runs, this stage is required.

### Benchmark stage: `build_proxy_context`

Notebook methods:

- `build_eval_stack`;
- `build_combo_proxy_cache_two`;
- `gather_combo_proxy_cache_two`.

This stage exists only when combo prefilter is active:

- active when `combo_top_frac < 1.0` or `combo_min_confirm > 1`;
- for arity 2, builds matrix-backed confirm/proxy lookup tables using
  `eval_T` and 15m returns;
- for generic N, packs `eval_T` into `eval_stack[arity, max_rows, n_intervals]`;
- in the canonical target benchmark combo prefilter is configured as pass-through
  (`combo_top_frac = 1.0`, `combo_min_confirm = 1`), so this stage is expected
  to be near zero but still must be recorded.

### Benchmark stage: `combo_iteration`

Notebook method: `iter_combo_chunks`.

This stage:

- builds deterministic Cartesian product over filtered local row pools;
- preserves indicator order from the normalized request;
- emits chunks as `{indicator_id: int32[K]}`;
- uses bounded chunk size (`4096` in canonical benchmark);
- records `cartesian_combinations`, `combo_chunks_processed` and
  `exact_candidates_evaluated`.

### Benchmark stage: `proxy_filter`

Notebook methods:

- `proxy_prefilter_combos_chunk_two`;
- `proxy_prefilter_combos_chunk_n`;
- `topk_fraction_idx`.

When combo prefilter is inactive, this stage selects the full chunk and records
near-zero time. When active, it:

- computes consensus confirmation count per combo;
- computes cheap directional proxy score from 15m returns;
- applies `combo_min_confirm`;
- keeps top fraction by proxy score;
- passes only selected combos to exact scoring.

### Benchmark stage: `self_check`

Self-check is part of benchmark evidence and must fail fast on parity drift.
Canonical benchmark uses `self_check_n = 2`.

No-risk methods:

- `build_trade_list_for_indicator_rows_slow`;
- `evaluate_no_risk_reference_rows_slow`;
- `run_fast_vs_reference_self_check_two`.

TP/SL methods:

- `build_trade_list_15m_for_indicator_rows_slow`;
- `evaluate_tp_sl_reference_trade_list_direct`;
- `evaluate_tp_sl_reference_rows_slow`;
- `run_tp_sl_self_check`.

Checks:

- backend `trade_count` equals slow reference;
- no-risk `total_return_pct` differs by at most `1e-4`;
- TP/SL best return differs by at most `5e-5`;
- TP/SL best cell indexes are valid;
- TP/SL best TP/SL cell must match reference unless the return difference is
  numerically immaterial.

### Benchmark stage: `exact_scoring`

This is the dominant hot path. It dispatches by `risk.mode`.

For `risk.mode = "none"`:

- default backend is `event_segments_2_no_risk` for arity 2;
- generic backend is `event_segments_n_no_risk` for arity 1 and 3..10;
- `streaming_2_no_risk` exists only as fallback/parity comparator;
- segment intersections produce raw consensus direction;
- `apply_direction_mode` maps raw direction to `long_only` or
  `long_short_reversal`;
- entries use the next 15m signal bar open mapped to 1m execution index;
- exits use signal close/reversal mapped to 1m open, or final 1m close when
  `close_on_end = true`;
- `apply_no_risk_trade_to_state` updates cash/equity state without allocating
  a full trade list in the hot path;
- summary metrics include `total_return_pct`, `max_drawdown_pct`,
  `return_over_max_drawdown`, `profit_factor`, `trade_count`, `sharpe_trades`,
  `win_rate_pct`, `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`.

For `risk.mode = "tp_sl_grid"`:

- backend is `event_segments_n_tp_sl_15m_grid` for arity 1..10;
- `exact_scoring` and `tp_sl_exact_scoring` record the same hot-path elapsed time;
- entries and signal exits are represented as absolute 15m bar indexes;
- TP/SL hit tables are `hit_times/15m`;
- scoring uses log-return accumulation for numerical stability;
- for each candidate trade, `tp_sl_apply_trade_to_diff` writes contribution
  ranges into three difference buffers:
  - `row_diff` for TP-only ranges;
  - `col_diff` for SL-only ranges;
  - `rect_diff` for signal/final-close fallback rectangles;
- prefix sums materialize the full TP/SL grid contribution for one combo;
- the best cell is the max log-return cell converted back to
  `total_return_pct = (exp(best_log) - 1) * 100`;
- if TP and SL hit on the same bar, SL wins the tie: TP requires
  `t_tp < t_sl`, SL accepts `t_sl <= t_tp`;
- persisted summary must include the same full metric set as no-risk
  (`total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`) plus
  `best_tp_pct` and `best_sl_pct` for the selected best cell.

### Benchmark stage: `heap_update`

Notebook uses Python `heapq` to keep top K.

This stage:

- ranks by selected metric, default `total_return_pct desc`;
- builds a deterministic heap key from score and original row ids;
- keeps only top N in memory;
- attaches compact per-indicator metadata;
- produces deterministic ordering for persisted top-N rows.

### Benchmark stage: `top_result_proxy_fill`

This stage is not lazy trades.

It runs only when final top rows did not receive proxy metadata from active combo
prefilter. It recomputes `confirm_count` and `proxy_score` for top rows using
`proxy_for_indicator_rows`. It must not be mapped to the UI/API
`show trades` endpoint.

### Runtime total without warmup

Every benchmark record must expose notebook-compatible timer names:

| Timer | Required | Notes |
|---|---:|---|
| `sample_warmup` / `service_warmup` / `numba_warmup` | yes | Measured separately, excluded from total. |
| `load_hit_times` | risk-on only | Hit-time subset loading. |
| `tp_sl_grid_validation` | risk-on only | Request grid coverage. |
| `prepare_pools` | yes | Artifact load, slicing, row selection, row prefilter, segment build. |
| `build_exact_context` | yes | Arity-first segment context where required. |
| `build_proxy_context` | yes | May be near zero when proxy prefilter is pass-through. |
| `combo_iteration` | yes | Cartesian chunk generation. |
| `proxy_filter` | yes | Pass-through or active combo pruning. |
| `self_check` | benchmark/test yes | Bounded parity check. |
| `exact_scoring` | yes | No-risk or TP/SL exact scorer. |
| `tp_sl_exact_scoring` | risk-on only | Alias/subsegment of `exact_scoring` for risk-on. |
| `heap_update` | yes | Top-N heap maintenance. |
| `top_result_proxy_fill` | no-risk yes | Top-row proxy metadata fill. |
| `total_without_warmup` | yes | User-facing measured runtime after warmup. |
| `persist_top_n_io` | service only | DB write overhead; not part of notebook baseline. |

### Lazy trades

- recompute exact trades for one `variant_key`;
- cache result for 48h;
- return chart-ready payload.

Lazy trades is not part of `total_without_warmup`; it has its own benchmark gate:
`lazy_trades_compute` and `lazy_trades_cache_hit`.

### Progress mapping

Persisted job state may stay coarse (`stage_a`, `stage_b`, `finalizing`) while API
progress exposes the finer pipeline stage.

| API `progress.pipeline_stage` | Persisted stage | Notes |
|---|---|---|
| `queued` | `stage_a` | Job exists but worker has not started. |
| `service_warmup` | `stage_a` | Sample/JIT warmup before measured runtime. |
| `load_hit_times` | `stage_b` | Risk-on hit-times subset load. |
| `tp_sl_grid_validation` | `stage_b` | Risk-on grid coverage validation. |
| `prepare_pools` | `stage_a` | Artifact load, slicing, row selection, prefilter and segment build. |
| `build_exact_context` | `stage_a` / `stage_b` | Segment stack for exact kernels. |
| `build_proxy_context` | `stage_a` | Optional combo proxy context. |
| `combo_iteration` | `stage_a` / `stage_b` | Cartesian chunk planning. |
| `proxy_filter` | `stage_a` / `stage_b` | Optional combo pruning. |
| `self_check` | `stage_a` / `stage_b` | Benchmark/test parity check. |
| `exact_scoring` with `risk.mode = "none"` | `stage_a` | No Stage B risk grid. |
| `exact_scoring` / `tp_sl_exact_scoring` with `risk.mode = "tp_sl_grid"` | `stage_b` | Hit-times backed TP/SL scoring. |
| `heap_update` | `finalizing` | Ranking and top-N assembly. |
| `top_result_proxy_fill` | `finalizing` | Top-row proxy metadata, not lazy trades. |
| `persist_top_n_io` | `finalizing` | Service-only DB write overhead. |
| `succeeded`, `failed`, `cancelled` | terminal state | Terminal state wins over stage. |

Contract:

- API consumers must use `state` for lifecycle and `progress.pipeline_stage` for UI detail;
- persisted stage is an implementation/read-model compatibility field;
- benchmark records use canonical notebook timer names, not legacy persisted stage names.

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

Canonical benchmark sources:

- canonical algorithm:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
- target numeric evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- target human-readable summary:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`.

The JSON evidence is the source of truth for numeric target values. The summary
is for review convenience only and must not be manually edited independently.

Canonical benchmark identity:

- host: `macstudio`;
- period: `[2020-01-11T20:08:00Z, 2026-04-11T20:08:00Z)`;
- rows per indicator: `6`;
- warmup rows per indicator: `2`;
- canonical production acceptance arities: `1..7`;
- target public request arities: `1..10` for both no-risk and TP/SL; arity 8..10
  are allowed only within cost guardrails and are not part of the mandatory 90%
  acceptance benchmark for v1 completion;
- direction modes: `long_only`, `long_short_reversal`;
- risk modes: `none`, `tp_sl_grid`;
- TP/SL grid: `2.0..25.0` inclusive with `0.5` step;
- TP/SL cells per combo: `2209`;
- runs: `28` (`7 arities x 2 risk modes x 2 direction modes`);
- request hash:
  `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`;
- artifact manifest hash:
  `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`;
- hit-times manifest hash:
  `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`.

Each implementation iteration must record:

- code version / branch / commit;
- artifact config and artifact root;
- artifact slot and `artifact_manifest_hash`;
- notebook baseline used;
- notebook baseline output path or captured metrics;
- request fixture;
- canonical `request_hash` and result-affecting config hash;
- service warmup metrics;
- canonical notebook timer metrics without warmup;
- service-only overhead metrics;
- speed ratio vs baseline;
- absolute latency budget result;
- peak RSS / memory delta;
- CPU time, process CPU percent, thread count, effective Numba threads, system load;
- pass/fail against 90% threshold;
- correctness/parity result.

Warmup policy:

- `service_warmup`, `numba_warmup` and `sample_warmup` are first-class measured
  segments;
- canonical benchmark uses sample warmup on `min(2, rows_per_indicator)` rows per
  indicator for the same arity/risk/direction/backend;
- user-facing runtime benchmark is measured after warmup;
- warmup and warm runtime must both stay within accepted 90% envelope for their
  matching segment.

Required benchmark segments:

1. `service_warmup`
2. `numba_warmup`
3. `sample_warmup`
4. `total_without_warmup`
5. `load_hit_times` for `risk.mode = "tp_sl_grid"`
6. `tp_sl_grid_validation` for `risk.mode = "tp_sl_grid"`
7. `prepare_pools`
8. `build_exact_context`
9. `build_proxy_context`
10. `combo_iteration`
11. `proxy_filter`
12. `self_check`
13. `exact_scoring`
14. `tp_sl_exact_scoring` for `risk.mode = "tp_sl_grid"`
15. `heap_update`
16. `top_result_proxy_fill`
17. `persist_top_n_io`
18. `lazy_trades_compute`
19. `lazy_trades_cache_hit`

Acceptance comparison:

- stages 1-16 are compared with notebook-compatible target values by tuple
  `{arity, risk_mode, direction_mode, backend}`;
- for arity 1..7, the target source is
  `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- arity 8..10 do not block v1 completion if arity 1..7 passes the 90% threshold;
  before raising production budgets for broad arity 8..10 workloads, create a
  follow-up Mac Studio benchmark iteration;
- `persist_top_n_io`, `lazy_trades_compute` and
  `lazy_trades_cache_hit` use service-specific absolute budgets plus regression
  comparison after their own baseline exists;
- an implementation may record lower-level subsegments, but pass/fail must include
  every canonical timer exposed by the notebook;
- latency target passes when service wall time is no worse than the canonical
  target divided by `0.90` for the same segment;
- memory target passes when service peak RSS / RSS delta is no worse than the
  canonical target divided by `0.90`, unless a stricter absolute budget is set;
- CPU target passes when service CPU time is no worse than the canonical target
  divided by `0.90`, with process CPU percent and thread count recorded for
  diagnosis.

Canonical target table excerpt:

| arity | risk | direction | backend | combos | exact target | total target |
|---:|---|---|---|---:|---:|---:|
| 1 | `none` | `long_only` | `event_segments_1_no_risk` | 6 | `0.003169s` | `0.132407s` |
| 1 | `tp_sl_grid` | `long_only` | `event_segments_1_tp_sl_15m_grid` | 6 | `0.003561s` | `0.937853s` |
| 7 | `none` | `long_only` | `event_segments_7_no_risk` | 279936 | `139.585680s` | `140.746091s` |
| 7 | `tp_sl_grid` | `long_only` | `event_segments_7_tp_sl_15m_grid` | 279936 | `146.899213s` | `147.415075s` |
| 1 | `none` | `long_short_reversal` | `event_segments_1_no_risk` | 6 | `0.002300s` | `0.166162s` |
| 1 | `tp_sl_grid` | `long_short_reversal` | `event_segments_1_tp_sl_15m_grid` | 6 | `0.008307s` | `1.956840s` |
| 7 | `none` | `long_short_reversal` | `event_segments_7_no_risk` | 279936 | `136.112667s` | `137.263877s` |
| 7 | `tp_sl_grid` | `long_short_reversal` | `event_segments_7_tp_sl_15m_grid` | 279936 | `140.994417s` | `141.519415s` |

The full target table, per-stage timers, runtime metrics and result hashes live
only in `benchmark_results.json`.

Benchmark records live under:

- `docs/architecture/backtest/benchmark_iterations/`

Stage completion rule:

- every implementation stage ends with a benchmark record in
  `docs/architecture/backtest/benchmark_iterations/<date>_<stage>/`;
- a stage is not complete until its benchmark record includes code version,
  request hash, artifact hashes, canonical timers, CPU/RSS metrics and
  correctness/parity evidence;
- the next stage must not be treated as accepted until the previous stage passes
  its benchmark gate.

## Test matrix

Implementation is not complete until functional and benchmark coverage includes:

- benchmark matrix for production acceptance:
  `arity 1..7 x risk.mode none/tp_sl_grid x direction_mode long_only/long_short_reversal`;
- service-level correctness smoke for arity 8..10 on small row pools to confirm
  contract support without including those arities in the mandatory v1
  performance gate;
- `risk.mode = "none"`;
- `risk.mode = "tp_sl_grid"` with request TP/SL subset covered by `hit_times/15m`;
- TP/SL benchmark grid `2.0..25.0` inclusive, step `0.5`;
- every sizing mode: `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- `profit_lock` disabled and enabled;
- every supported `direction_mode` from runtime defaults:
  `long_only`, `long_short_reversal`;
- `close_on_end = true` and `close_on_end = false`, where `close_on_end = false`
  is covered by service-level correctness tests rather than a required notebook
  benchmark;
- full persisted summary metrics for both no-risk and TP/SL risk-on variants;
- public API contract tests for create/status/list/top/variant/trades/cancel/defaults/preflight;
- idempotency tests for replay and key conflict;
- ownership/authz tests for job, top and lazy trades reads;
- golden parity tests against notebook-derived fixtures;
- failure injection for missing artifacts, stale current pointer, TP/SL grid not covered,
  request too expensive, queue saturation and lazy cache failure;
- cache identity tests covering `job_id`, `variant_key`, `variant_hash`,
  `request_hash`, `engine_params_hash` and `artifact_manifest_hash`.

Current canonical `sizing_smoke` evidence has compiled parity for `all_in` and
`fixed_quote`; equity-percent sizing modes are reference-only in the notebook
evidence. The service implementation is the first compiled parity point for
`fixed_equity_pct`, `fixed_equity_pct_min_quote` and
`fixed_equity_pct_max_quote`, and must record service-level parity evidence for
those modes before v1 is considered functionally complete.

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
- `backtest_request_cost_estimate{risk_mode,cost_class}`
- `backtest_requests_rejected_total{reason}`
- `backtest_jobs_cancel_requested_total{state}`

Security and access:

- all endpoints require authenticated user identity;
- every job, top row, variant and lazy trades response is scoped by owner;
- API never accepts artifact paths from request payloads;
- artifact root, cache root and runtime config come only from trusted config;
- file/object cache permissions must prevent cross-user direct filesystem access;
- request payloads and logs must not include secrets.

Resource guardrails:

| Guardrail | Config key | v1 default | What it limits | Failure |
|---|---|---:|---|---|
| Active jobs per user | `backtest.max_active_jobs_per_user` | `1` | How many jobs one user may run in `running/warming/scoring` at the same time. One arity 7 benchmark already loads CPU at roughly one heavy-worker level. | `429 backtest.rate_limited` |
| Queued jobs per user | `backtest.max_queued_jobs_per_user` | `3` | How many jobs one user may keep queued beyond the active job. | `429 backtest.rate_limited` |
| Global active jobs | `backtest.max_active_jobs_global` | `1` | How many heavy jobs the service runs concurrently. Increase to `2+` only after a separate concurrency benchmark; otherwise the 90% latency/CPU target becomes unstable. | `503 backtest.queue_saturated` |
| `top_n` | `backtest.max_top_n` | `100` | How many summary rows are persisted and returned per job. Larger values increase heap work, payload size and DB write cost. | `422 backtest.request_too_expensive` |
| Indicator arity | `backtest.max_indicator_arity` | `10` | Maximum indicators in one request. Production acceptance benchmark is mandatory for arity 1..7; arity 8..10 is allowed only when the other cost guardrails pass. | `422 backtest.request_too_expensive` |
| Indicator rows after source/window expansion | `backtest.max_indicator_rows` | `1000` | Total signal rows after expanding all `source` and `window` ranges, before row prefilter. Example: 5 sources x 200 windows = 1000 rows already consumes the default budget. | `422 backtest.request_too_expensive` |
| Candidate combinations after row prefilter | `backtest.max_candidate_combinations` | `300000` | Combinations before exact scoring after row prefilter. Default covers the canonical arity 7 fixture (`6^7 = 279936`) and rejects requests like `20^5 = 3200000`. | `422 backtest.request_too_expensive` |
| TP/SL cells | `backtest.max_tp_sl_cells` | `2209` | Request TP/SL grid size. Default equals the canonical `47 x 47` grid for `2.0..25.0` step `0.5`. | `422 backtest.request_too_expensive` |
| Lazy trades requests per user window | `backtest.lazy_trades_rate_limit` | `30 / 10 min` | Lazy trades detail requests per user sliding window. | `429 backtest.rate_limited` |
| Job queue wait | `backtest.job_queue_timeout_seconds` | `300` | Maximum time a job may wait in queue before terminal failure. | terminal job failure |
| Job wall time | `backtest.job_wall_timeout_seconds` | `900` | Maximum job wall-clock runtime. Requests estimated to exceed this budget must be rejected by preflight. | terminal job failure |
| Lazy trades wall time | `backtest.lazy_trades_timeout_seconds` | `30` | Maximum lazy trade recompute time for one `variant_key`. | `503` retryable |

The default tier intentionally stays close to the canonical benchmark workload.
Paid or admin tiers may expand `max_top_n`, `max_indicator_rows`,
`max_candidate_combinations`, `max_active_jobs_global` and
`job_wall_timeout_seconds`, but only after a separate `Mac Studio` benchmark
record for that tier.

Production rollout is blocked if these config keys are unset or if preflight cannot
explain which guardrail rejected a request and how the user can narrow that
request.

Failure behavior:

- invalid request returns deterministic 422;
- missing artifact family returns deterministic runtime failure on job;
- request TP/SL not covered by published grid returns deterministic 422 before job execution if possible;
- lazy trades cache failure must not fail the trades response if recompute succeeds;
- benchmark failure blocks the current iteration from being considered complete.

## План внедрения

Rule for all iterations:

- every iteration has an explicit benchmark/evidence gate;
- benchmark records are written before the iteration is marked complete;
- a later stage may be prototyped locally, but it must not be considered accepted
  until all earlier stage gates have passed.

### Iteration 0: docs and benchmark harness

- finalize this architecture document against canonical notebook prototype;
- mark conflicting roadmap/runtime docs as superseded or compatibility-only where needed;
- keep `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`
  as canonical algorithm source;
- keep `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
  as canonical target values;
- define Mac Studio benchmark command contract for future iteration records.

Exit criteria:

- benchmark folder exists and contains JSON + Markdown evidence;
- canonical notebook and benchmark evidence are named;
- canonical timer names and service overhead stages are fixed;
- source-of-truth status is explicit in backtest docs;
- `variant_key`/`variant_hash` and progress mapping contracts are documented.

Benchmark gate:

- `docs/architecture/README.md` index is up to date;
- canonical benchmark evidence paths are readable;
- future benchmark record template contains commit, request hash, artifact hashes,
  stage timers, CPU/RSS metrics and correctness result fields.

### Iteration 1: request normalization and artifact context

- implement strict artifact context resolver;
- implement request normalization for coordinates/timeframe/period/indicator grids;
- implement execution defaults and validation for `direction_mode`, `sizing`,
  `profit_lock`, fees, slippage and `close_on_end`;
- implement canonical `request_hash` and result-affecting config hash;
- implement cost estimate for rows, combinations and TP/SL cells;
- expose `POST /backtests/preflight` and `GET /backtests/runtime-defaults`
  as API shell.

Benchmark gate:

- request normalization and preflight smoke benchmark;
- artifact current/root resolution timing;
- parity check that request hash matches canonical fixture hash where applicable;
- failure evidence for invalid indicator/source/window and request-too-expensive.

### Iteration 2: artifact arrays and `prepare_pools`

- implement mmap loaders for `prices`, `signals`, `mappings`;
- implement `[start, end)` slicing by 15m `open_time`;
- implement 15m return interval derivation;
- implement 15m-to-1m execution mapping for no-risk;
- implement signal row extraction and source/window row mapping;
- implement row prefilter with `fused_row_prefilter_stats`;
- implement compressed signal segments with `build_signal_segments`;
- expose `prepare_pools` timing.

Benchmark gate:

- `prepare_pools` vs canonical notebook target for arity 1..7 fixture;
- optional subsegments: `artifact_manifest_load`, `artifact_array_mmap_load`,
  `time_range_slice`, `signal_row_selection`, `row_prefilter`,
  `segment_build`;
- row metadata/order hash equals notebook fixture;
- stage record written before moving to combo planning.

### Iteration 3: combo planning contexts

- implement backend registry for `event_segments_2_no_risk`,
  `event_segments_n_no_risk`, `streaming_2_no_risk` and
  `event_segments_n_tp_sl_15m_grid`;
- implement `build_exact_context`;
- implement `build_proxy_context`;
- implement deterministic `combo_iteration`;
- implement pass-through and active `proxy_filter`.

Benchmark gate:

- `build_exact_context`;
- `build_proxy_context`;
- `combo_iteration`;
- `proxy_filter`;
- deterministic combo ordering and candidate-count evidence;
- active and inactive proxy-filter fixture evidence;
- stage record written before exact scoring.

### Iteration 4: no-risk exact scoring and top-N

- implement `event_segments_2_no_risk`;
- implement `event_segments_n_no_risk` for arity 1..10;
- implement `streaming_2_no_risk` fallback/parity comparator;
- implement no-risk self-check against generic slow reference;
- implement full no-risk summary metric set;
- implement `heap_update` and `top_result_proxy_fill`;
- implement persisted top-N summary for no-risk;
- map public `variant_key` to storage `variant_hash` safely.

Benchmark gate:

- `service_warmup`;
- `self_check`;
- `exact_scoring` for no-risk;
- `heap_update`;
- `top_result_proxy_fill`;
- arity 1..7 target comparison against current canonical evidence;
- service-level correctness smoke for arity 8..10 on small row pools;
- persisted top-N summary hash/parity evidence.

### Iteration 5: TP/SL grid loading and validation

- validate request TP/SL subset against artifact grid;
- implement `load_hit_times` and `tp_sl_grid_validation`;
- implement hit-times manifest hashing;
- implement requested subset materialization for long/short TP/SL arrays;
- implement deterministic 422 for grid-not-covered failure.

Benchmark gate:

- `load_hit_times`;
- `tp_sl_grid_validation`;
- request grid coverage success and failure evidence;
- target grid `2.0..25.0` step `0.5` evidence;
- stage record written before TP/SL exact scoring.

### Iteration 6: TP/SL exact scoring and full metrics

- implement `event_segments_n_tp_sl_15m_grid` for arity 1..10;
- implement TP/SL self-check against slow direct reference;
- implement TP/SL full persisted summary metric set:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`,
  `best_tp_pct`, `best_sl_pct`;
- persist risk-on top-N summary.

Benchmark gate:

- `build_exact_context`;
- `combo_iteration`;
- `self_check`;
- `exact_scoring` / `tp_sl_exact_scoring` for TP/SL grid vs canonical target;
- `heap_update`;
- arity 1..7 target comparison against current canonical evidence;
- service-level correctness smoke for arity 8..10 on small row pools;
- full metric-set correctness evidence for selected best TP/SL cell.

### Iteration 7: job orchestration and persistence

- implement job create/status/top/list/cancel contracts;
- implement idempotency and request guardrails;
- persist canonical request snapshot, artifact metadata and top-N rows;
- expose progress using canonical pipeline stage names;
- implement ownership/authz checks.

Benchmark gate:

- `persist_top_n_io`;
- end-to-end job benchmark for no-risk and TP/SL with current canonical fixtures;
- API contract tests for create/status/list/top/cancel/defaults/preflight;
- idempotency replay/conflict evidence;
- authz/ownership failure evidence.

### Iteration 8: execution/sizing completion

- implement all public sizing modes in service compiled path:
  `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- implement `profit_lock` parity for every sizing mode;
- implement `close_on_end = false`;
- verify no-risk and TP/SL semantics remain stable across execution settings.

Benchmark and correctness gate:

- sizing smoke vs canonical notebook evidence;
- service compiled parity for equity-percent modes, which are reference-only in
  current notebook evidence;
- service-level correctness tests for `close_on_end = true/false`;
- regression check that canonical arity/risk/direction benchmark remains within
  target envelope.

### Iteration 9: lazy trades detail

- implement variant lookup;
- implement lazy trades recompute;
- implement 48h cache;
- return chart-ready payload.

Benchmark gate:

- `lazy_trades_compute`, `lazy_trades_cache_hit`.

Verification:

- cache miss and cache hit;
- cache failure with successful recompute;
- ownership failure;
- variant key/hash mismatch failure.

### Iteration 10: UI integration

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

## Риски, пояснения и открытые решения

Не риск: artifact reproducibility. Artifact store для backtest считается
практически immutable на уровне опубликованных файлов. Даже если publisher
дописывает новый хвост, benchmark и jobs читают `[start, end)` внутри historical
prefix, который не изменяется. Инвариант v1: published historical prefix never
rewrites. Поэтому отсутствие отдельного immutable version retention не является
самостоятельным production blocker для текущей модели.

Риск: production request может быть шире canonical benchmark fixture.
Canonical target измерен на `rows_per_indicator = 6`. Это означает: у каждого
indicator после отбора в benchmark остается 6 signal rows, и combo count растет
как произведение rows по indicators. Простой пример: `7` indicators x `6` rows =
`6^7 = 279936` combos, это покрыто benchmark. Но `5` indicators x `20` rows =
`20^5 = 3200000` combos, то есть больше чем в 11 раз. Решения:

- default public tier держит `max_indicator_rows = 1000` и
  `max_candidate_combinations = 300000`;
- `POST /backtests/preflight` заранее считает cost и объясняет, что нужно сузить:
  меньше sources, уже window range, больше step или меньше indicators;
- row prefilter обязателен и должен сокращать широкие source/window ranges до
  ограниченного числа rows перед exact scoring;
- отдельные paid/admin tiers могут иметь более высокий budget, но только после
  отдельного Mac Studio benchmark для такого tier.

Не риск для v1 completion: arity 8..10. Production acceptance benchmark
обязателен для arity 1..7. Если сервис достигает 90% target по arity 1..7,
перенос алгоритма в production runtime считается успешным. Arity 8..10 остаются
разрешенной формой request, но проходят только при соблюдении cost guardrails.
Отдельный benchmark для arity 8..10 нужен перед расширением budgets, а не для
закрытия v1.

Риск/implementation decision: full metrics для TP/SL best-cell summary. Данных
хватает: hit-times, prices, signals и выбранная best TP/SL cell есть. Не хватает
доказанного production-способа посчитать полный metric set дешево. Fast TP/SL path
быстро находит best cell по return через difference buffers и log-return sums. Но
`max_drawdown_pct`, `profit_factor`, `sharpe_trades`, `win_rate_pct`,
`avg_trade_ret_pct`, `avg_trade_exec_bars` и `exposure_pct` требуют пройти сделки
для выбранной cell и восстановить equity/trade stats. Решения:

- предпочтительно: hot path ранжирует combos по return, а полный metric set
  пересчитывается только для persisted top-N variants по их selected best cell;
- если пользователь ранжирует по метрике не `total_return_pct`, нужен отдельный
  compiled path или bounded shortlist, иначе придется считать много trade stats;
- Iteration 6 должна записать CPU/RSS cost именно этого second-pass metrics step.

Риск: equity-percent sizing modes пока reference-only в notebook evidence.
`all_in` и `fixed_quote` уже имеют compiled parity smoke. Для
`fixed_equity_pct`, `fixed_equity_pct_min_quote` и
`fixed_equity_pct_max_quote` текущий notebook evidence является reference-only.
Пример: при `initial_cash_quote = 10000` и `fixed_equity_pct = 10%` первая сделка
использует `1000` quote, но после прибыли/убытка следующая сделка должна считать
10% уже от нового equity; min/max modes дополнительно clamp-ят quote size. Решения:

- основной путь: реализовать эти sizing modes в compiled service path и сравнить
  с notebook/reference fixtures как first compiled parity point;
- дополнительный путь: сначала добавить compiled parity в notebook, но это
  задержит service implementation;
- fallback с отключением equity-percent modes в публичном API не подходит для v1,
  потому что эти modes входят в публичный request contract.

Риск: полный catalog `configs/prod/indicators.yaml` может создать большие combo
spaces. Пользователь может выбрать несколько indicators, все sources и широкий
window range. Даже если каждый indicator валиден сам по себе, Cartesian product
может стать слишком дорогим. Пример: `4` indicators, у каждого после expansion
`50` rows, дают `50^4 = 6250000` combos. Решения:

- `runtime-defaults` должен показывать limits до запуска;
- `preflight` должен считать rows, combos и TP/SL cells до создания job;
- row prefilter и combo proxy prefilter обязательны, а не performance luxury;
- requests выше default budget получают `422 backtest.request_too_expensive` с
  понятной подсказкой, какой параметр сузить.

Риск: `variant_key` может конфликтовать с текущим storage identity. Public v1
хочет readable route key, например
`job_f7d2c378__dema_close_w192__risk_none__vh_a13f09c2`. Legacy/current DB code
может ожидать, что `variant_key` это 64-char SHA. Если положить readable key в
SHA-only column, можно сломать validation, indexes или lazy-trades lookup. Решения:

- preferred: добавить отдельные persisted fields `public_variant_key` и
  `variant_hash`, где `variant_hash` остается stable SHA-256;
- transition: adapter принимает public `variant_key`, резолвит его в top-N row
  внутри `job_id`, а в legacy SHA-only field временно хранит `variant_hash`;
- rejected: сделать public key только hash-ем, потому что UI/debuggability хуже и
  это противоречит readable route contract.

Риск: lazy trades cache зависит от deployment topology. В single-host v1 local
file/object cache на 48h достаточен: cache miss просто запускает deterministic
recompute для одного variant. В multi-host deployment запрос `show trades` может
попасть на другой API/worker host, где local cache файла нет. Это не ломает
correctness, но может давать лишний recompute и нестабильную latency. Решения:

- v1 default: single API/worker host или sticky routing + local cache;
- scale-out trigger: перед несколькими API/worker hosts включить shared object
  storage для lazy trades payload;
- Postgres JSONB допустим только для малых payloads и metadata, но не как
  безлимитное хранилище всех trades.

Docs cleanup: `docs/architecture/roadmap/*.md` не содержат прямых упоминаний
старой hit-times модели. Canonical docs должны использовать `hit_times/15m`.
Исторический deep-research report может сохранять старый анализ как архивный
контекст, но он не является source of truth для реализации.

Отложенное решение без production risk: точная форма chart payload для UI
candles/trades overlay будет спроектирована в UI iteration.

Решение по guardrails: v1 default values зафиксированы в таблице `Resource
guardrails` выше. Перед production rollout эти значения должны попасть в runtime
config и быть видимы через `GET /backtests/runtime-defaults` и
`POST /backtests/preflight`.
