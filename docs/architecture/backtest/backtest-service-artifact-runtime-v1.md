# Backtest Service Artifact Runtime v1

Документ фиксирует целевую архитектуру artifact-backed backtest service, его публичный API, job flow, lazy trades detail и benchmark-gate по итерациям.

## Статус

Proposed target architecture.

Этот документ проектируется поверх:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`;
- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`;
- текущего trusted artifact publisher/precompute scope в `backtest_artifacts`;
- текущей DDD / ports-and-adapters структуры репозитория.

Этот документ является каноническим target contract для нового `Backtest Service
Artifact Runtime v1`.

Старые backtest runtime docs и удаленный runtime compute path не восстанавливаются.
Roadmap-формулировки, которые противоречат этому документу, считаются
historical/compatibility context для v1, а не source of truth. Это включает:

- `POST /backtests` как основной create endpoint вместо `POST /backtests/jobs`;
- `runs` vocabulary вместо `jobs` vocabulary;
- `hit_times/1m` вместо target `hit_times/15m`;
- public `execution profile` vocabulary вместо `risk.mode`.

`docs/architecture/roadmap/backtest-refactor-final-plan-v2.md` упоминается старым
roadmap как source of truth, но отсутствует в текущем рабочем дереве. Пока он не
восстановлен отдельным ADR/документом, он не переопределяет этот v1 contract.
Существующие code/schema элементы со старым словарем рассматриваются как
compatibility inputs: adapters may translate them, but public v1 contract stays
defined here.

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
- benchmark stages: notebook-compatible measured compute stages after warmup.

### Lifecycle stage: request normalization

- validate coordinates;
- validate timeframe excludes `1m` and `5m`;
- validate `[start, end)`;
- validate indicator ids against `configs/prod/indicators.yaml` loaded through registry/defaults;
- materialize requested `source` and `window` row selections;
- validate request TP/SL grid is covered by published artifact grid when `risk.mode = "tp_sl_grid"`;
- apply execution defaults;
- compute canonical request/config hashes and estimated request cost.

### Lifecycle stage: job creation

- create `backtest_jobs` row with canonical request snapshot;
- record artifact metadata/watermark for audit and cache identity;
- state starts as `queued`.

### Benchmark stage 1: `prepare_indicator_pools`

Notebook-compatible meaning:

- resolve artifact root;
- read strict `current.yaml` and slot `manifest.yaml`;
- load `.npy` via `np.load(..., mmap_mode="r")`;
- load `prices/<tf>`, `prices/1m`, `mappings/<tf>`, requested `signals/<tf>/<indicator_id>`, and `hit_times/15m` only when risk mode requires it;
- slice `[start, end)` on the signal timeframe and corresponding execution window;
- map `{indicator_id, source, window}` to artifact row ids using manifest grid contract;
- copy only selected rows into contiguous working arrays when kernels need contiguous memory;
- preserve source-major/window-minor ordering from artifacts.

Reference notebook example:

```text
prepare indicator pools    0.964s
```

### Benchmark stage 2: `combo_proxy_prefilter`

Notebook-compatible meaning:

- single-indicator row prefilter;
- combo proxy prefilter;
- bounded combo chunk planning;
- deterministic combo ordering for downstream exact scoring.

Reference notebook example:

```text
combo proxy prefilter      0.595s
```

### Benchmark stage 3: `count_trades`

Notebook-compatible meaning:

- build compact trade-count evidence for surviving combos;
- reject combos with invalid/empty trade count;
- keep count arrays compact enough for exact scoring.

Reference notebook example:

```text
count trades               0.637s
```

### Benchmark stage 4: `exact_scoring`

For `risk.mode = "none"`:

- build compact trade list from consensus signals;
- execute no-risk scoring with fees/slippage/sizing/profit-lock;
- apply `direction_mode` and `close_on_end`.

For `risk.mode = "tp_sl_grid"`:

- build compact trade list from consensus signals;
- use `hit_times/15m` and request TP/SL subset;
- run fast monotone TP/SL search;
- run bounded self-check against reference path per benchmark/test policy;
- select best TP/SL cell per variant.

Reference notebook example:

```text
exact scoring              0.775s
```

### Benchmark stage 5: `heap_top_k_python_work`

Notebook-compatible meaning:

- rank by selected metric;
- maintain heap/top-K shortlist;
- assemble persisted top-N summary rows in deterministic order.

Reference notebook example:

```text
heap/top-K Python work     0.026s
```

Persistence writes are measured separately as service overhead because notebook
baseline does not include network/DB IO. The service still records
`persist_top_n_io` with an absolute budget.

### Runtime total without warmup

Every benchmark record must show the canonical notebook-compatible runtime table:

| Stage | Reference example |
|---|---:|
| `total_without_warmup` | `3.012s` |
| `prepare_indicator_pools` | `0.964s` |
| `combo_proxy_prefilter` | `0.595s` |
| `count_trades` | `0.637s` |
| `exact_scoring` | `0.775s` |
| `heap_top_k_python_work` | `0.026s` |

Implementation-specific subsegments are allowed, but they must roll up to these
five stages for acceptance comparison.

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
| `prepare_indicator_pools` | `stage_a` | Artifact load, slicing and row selection. |
| `combo_proxy_prefilter` | `stage_a` | Prefilter and combo planning. |
| `count_trades` | `stage_a` | Trade-count pass before exact scoring. |
| `exact_scoring` with `risk.mode = "none"` | `stage_a` | No Stage B risk grid; transition to `finalizing` happens before heap/top-K persistence. |
| `exact_scoring` with `risk.mode = "tp_sl_grid"` | `stage_b` | Hit-times backed TP/SL scoring. |
| `heap_top_k_python_work` | `finalizing` | Ranking, top-K assembly and summary persistence. |
| `persist_top_n_io` | `finalizing` | Service-only DB write overhead. |
| `succeeded`, `failed`, `cancelled` | terminal state | Terminal state wins over stage. |

Contract:

- API consumers must use `state` for lifecycle and `progress.pipeline_stage` for UI detail;
- persisted stage is an implementation/read-model compatibility field;
- benchmark records use `progress.pipeline_stage` names, not legacy persisted stage names.

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
- artifact slot and `artifact_manifest_hash`;
- notebook baseline used;
- notebook baseline output path or captured metrics;
- request fixture;
- canonical `request_hash` and result-affecting config hash;
- service warmup metrics;
- canonical five-stage runtime metrics without warmup;
- service-only overhead metrics;
- speed ratio vs baseline;
- absolute latency budget result;
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
2. `total_without_warmup`
3. `prepare_indicator_pools`
4. `combo_proxy_prefilter`
5. `count_trades`
6. `exact_scoring`
7. `heap_top_k_python_work`
8. `persist_top_n_io`
9. `lazy_trades_compute`
10. `lazy_trades_cache_hit`

Acceptance comparison:

- stages 2-7 are compared with notebook-compatible baseline stages;
- `service_warmup`, `persist_top_n_io`, `lazy_trades_compute` and
  `lazy_trades_cache_hit` use service-specific absolute budgets plus regression
  comparison after their own baseline exists;
- an implementation may record lower-level subsegments, but pass/fail is decided
  on canonical rollups.

Initial notebook-style baseline example:

```text
total                       3.012s
prepare indicator pools     0.964s
combo proxy prefilter       0.595s
count trades                0.637s
exact scoring               0.775s
heap/top-K Python work      0.026s
```

Benchmark records live under:

- `docs/architecture/backtest/benchmark_iterations/`

## Test matrix

Implementation is not complete until functional and benchmark coverage includes:

- `risk.mode = "none"`;
- `risk.mode = "tp_sl_grid"` with request TP/SL subset covered by `hit_times/15m`;
- every sizing mode: `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- `profit_lock` disabled and enabled;
- every supported `direction_mode` from runtime defaults;
- `close_on_end = true` and `close_on_end = false`;
- public API contract tests for create/status/list/top/variant/trades/cancel/defaults/preflight;
- idempotency tests for replay and key conflict;
- ownership/authz tests for job, top and lazy trades reads;
- golden parity tests against notebook-derived fixtures;
- failure injection for missing artifacts, stale current pointer, TP/SL grid not covered,
  request too expensive, queue saturation and lazy cache failure;
- cache identity tests covering `job_id`, `variant_key`, `variant_hash`,
  `request_hash`, `engine_params_hash` and `artifact_manifest_hash`.

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

| Guardrail | Config key | Failure |
|---|---|---|
| Active jobs per user | `backtest.max_active_jobs_per_user` | `429 backtest.rate_limited` |
| Queued jobs per user | `backtest.max_queued_jobs_per_user` | `429 backtest.rate_limited` |
| Global active jobs | `backtest.max_active_jobs_global` | `503 backtest.queue_saturated` |
| `top_n` | `backtest.max_top_n` | `422 backtest.request_too_expensive` |
| Indicator rows after source/window expansion | `backtest.max_indicator_rows` | `422 backtest.request_too_expensive` |
| Candidate combinations before prefilter | `backtest.max_candidate_combinations` | `422 backtest.request_too_expensive` |
| TP/SL cells | `backtest.max_tp_sl_cells` | `422 backtest.request_too_expensive` |
| Lazy trades requests per user window | `backtest.lazy_trades_rate_limit` | `429 backtest.rate_limited` |
| Job queue wait | `backtest.job_queue_timeout_seconds` | terminal job failure |
| Job wall time | `backtest.job_wall_timeout_seconds` | terminal job failure |
| Lazy trades wall time | `backtest.lazy_trades_timeout_seconds` | `503` retryable |

Production rollout is blocked if these config keys are unset or if preflight cannot
explain which guardrail rejected a request.

Failure behavior:

- invalid request returns deterministic 422;
- missing artifact family returns deterministic runtime failure on job;
- request TP/SL not covered by published grid returns deterministic 422 before job execution if possible;
- lazy trades cache failure must not fail the trades response if recompute succeeds;
- benchmark failure blocks the current iteration from being considered complete.

## План внедрения

### Iteration 0: docs and benchmark harness

- finalize this architecture document;
- mark conflicting roadmap/runtime docs as superseded or compatibility-only where needed;
- create benchmark iteration workspace;
- extract notebook fixtures into reproducible benchmark inputs with five-stage output;
- define Mac Studio benchmark command contract.

Exit criteria:

- benchmark folder exists;
- baseline notebook scenarios are named;
- five canonical benchmark stages and service overhead stages are fixed;
- source-of-truth status is explicit in backtest docs;
- `variant_key`/`variant_hash` and progress mapping contracts are documented.

### Iteration 1: `prepare_indicator_pools`

- implement strict artifact context resolver;
- implement mmap loaders for `prices`, `signals`, `mappings`, `hit_times/15m`;
- implement `[start, end)` slicing and row selection;
- expose subsegment timings that roll up into `prepare_indicator_pools`.

Benchmark gate:

- `prepare_indicator_pools` vs notebook baseline;
- optional subsegments: `artifact_manifest_load`, `artifact_array_mmap_load`,
  `time_range_slice`, `signal_row_selection`.

### Iteration 2: no-risk compute and persisted top-N

- implement job create/status/top;
- implement list/cancel/defaults/preflight contracts as minimal API shell;
- implement idempotency and request guardrails;
- implement `risk.mode = "none"` scorer;
- persist top N summary;
- map public `variant_key` to storage `variant_hash` safely.

Benchmark gate:

- `service_warmup`;
- `combo_proxy_prefilter`;
- `count_trades`;
- `exact_scoring` for no-risk;
- `heap_top_k_python_work`;
- `persist_top_n_io`.

### Iteration 3: TP/SL grid path

- validate request TP/SL subset against artifact grid;
- implement hit-times backed Stage B scoring;
- persist `best_tp_pct`, `best_sl_pct`.

Benchmark gate:

- five-stage table with `exact_scoring` for TP/SL grid vs notebook-derived baseline;
- request grid coverage validation and failure path.

### Iteration 4: lazy trades detail

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
- Risk: current persistence/domain code may still treat `variant_key` as a SHA-only storage key; adapters or schema migration must protect the public readable `variant_key` contract.
- Risk: multi-host API/worker deployment requires shared lazy trades cache before production scale-out.
- Risk: roadmap docs still contain superseded `runs`, `POST /backtests`, `hit_times/1m` and `execution profile` vocabulary; implementation must not use those as v1 source of truth.
- Question: exact chart payload shape for UI candles/trades overlay remains a UI iteration decision.
- Question: exact numeric defaults for resource guardrails must be set in runtime config before production rollout.
