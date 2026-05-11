# Бектест-сервис на артефактах v1

Документ фиксирует целевую архитектуру бектест-сервиса на артефактах, его публичный API, поток выполнения задач, ленивую детализацию сделок и гейт бенчмарка по итерациям.

Русская версия переводит человекочитаемое описание. Идентификаторы API routes,
config keys, metric names, timer names, backend ids, file paths и значения
контрактных полей сохраняются в исходном написании, чтобы не разорвать связь с
кодом, notebook и benchmark evidence.

## Статус

Каноническая целевая архитектура для планирования реализации. Runtime-сервис
реализуется итерационно; Iteration 7 принята для public jobs API,
summary-only top-N persistence и service-only job orchestration evidence.
Notebook и benchmark evidence ниже определяют production-прототип, которому
сервис обязан соответствовать.

Этот документ проектируется поверх:

- канонический notebook-прототип:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
- канонические benchmark evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- человекочитаемое benchmark-резюме:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`;
- текущего доверенного artifact publisher/precompute scope в `backtest_artifacts`;
- текущей DDD / ports-and-adapters структуры репозитория.

Этот документ является каноническим целевым контрактом для нового `Backtest Service
Artifact Runtime v1`.

Старые документы backtest runtime и удаленный runtime compute path не восстанавливаются.
Roadmap-формулировки, которые противоречат этому документу, считаются
историческим/compatibility-контекстом для v1, а не источником истины. Это включает:

- `POST /backtests` как основной create endpoint вместо `POST /backtests/jobs`;
- словарь `runs` вместо словаря `jobs`;
- любые hit-times формулировки, которые противоречат target `hit_times/15m`;
- публичный словарь `execution profile` вместо `risk.mode`;
- старые benchmark notebooks под `tests/notebook_tests/new_engine/*` как
  канонический источник алгоритма.

`docs/architecture/roadmap/backtest-refactor-final-plan-v2.md` упоминается старым
roadmap как источник истины, но отсутствует в текущем рабочем дереве. Пока он не
восстановлен отдельным ADR/документом, он не переопределяет этот v1 contract.
Существующие code/schema элементы со старым словарем рассматриваются как
compatibility-входы: адаптеры могут транслировать их, но публичный v1 contract
остается определенным здесь.

## Цель

Построить продакшн-сервис бектестов для сайта и публичного API, который:

- всегда запускает backtest как persisted job;
- читает precomputed `.npy` artifacts (`prices`, `signals`, `mappings`, `hit_times/15m`);
- принимает координаты рынка, расчетный период, timeframe и набор индикаторов с `source` и `window` grid;
- поддерживает `no-risk` и `tp/sl grid` режимы;
- возвращает persisted top N summary;
- лениво пересчитывает trades для выбранного `variant_key` по запросу UI/API;
- развивается итерационно, где каждая итерация проходит benchmark на `Mac Studio`;
- считается завершенным только если каждый pipeline segment держит не ниже 90%
  целевого baseline по скорости, памяти и CPU-метрикам из канонических benchmark
  evidence `2026-04-26_engine_test_btcusdt_15m`.

## Контекст

Текущий backtest runtime официально сброшен: активной доверенной частью считаются artifact publisher/precompute, строгие manifests, `.npy` layout, signal rules, hit-times precompute, path/current-pointer adapters и job storage guard.

Новый сервис должен стартовать поверх artifacts и не тащить legacy runtime kernels/scorers/shortlists. Если в репозитории остаются routes, templates, worker wiring или runtime modules со старым словарем, они должны быть явно классифицированы как `active`, `compatibility-only` или `obsolete` перед использованием в реализации.

Целевая модель `hit_times`: `hit_times/15m`. Семантически таблицы обслуживают риск-исполнение через precomputed hit-time data, а request выбирает подмножество из заранее опубликованного достаточно широкого TP/SL grid.

## Охват

- Публичный пользовательский API для jobs, top results, variant summary и lazy trades.
- Использование UI через тот же публичный API.
- Application service pipeline для runtime на артефактах.
- Контракт хранения jobs и result summary.
- Cache для lazy trades на 1-2 дня.
- Рабочая папка benchmark и обязательный формат evidence по каждой итерации.
- Целевые границы модулей и направление зависимостей.

## Что не входит

- Отдельный internal API. На v1 он не нужен и не планируется.
- Восстановление удаленных legacy docs `deep-research-report.md` и `backtest-core-refactor-prompt-pack-v1.md`.
- Хранение старых artifact versions как immutable historical snapshots.
- Portfolio/multi-instrument backtest.
- Live execution/order routing.
- ML/inference path.
- Детали реализации browser UI за пределами API consumption contract.

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

Job все равно сохраняет metadata для artifact identity/watermark:

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

- Stage B выбирает лучшую TP/SL cell из request grid;
- request grid обязан быть покрыт published grid `hit_times/15m`;
- если request grid не покрыт artifacts, API возвращает детерминированный 422.

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

Бенчмарки и тесты должны покрывать все sizing modes.

### 6) `variant_key` человекочитаемый и уникальный внутри каждого запуска

Нужны два идентификатора:

- `variant_key`: публичный route/UI key, человекочитаемый и уникальный внутри конкретного job;
- `variant_hash`: стабильный SHA-256 от canonical JSON параметров варианта без `job_id`;
- `indicator_variant_hash`: стабильный SHA-256 от indicator/source/window части, если нужен отдельный ключ для indicator-only identity.

Целевой формат:

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

Переходная модель persistence:

- текущий/legacy DB/domain code может ожидать 64-символьный SHA в поле `variant_key`;
- пока schema не мигрирована, persistence adapter должен хранить canonical `variant_hash`
  в legacy SHA-only field и отдавать наружу public `variant_key`;
- целевое storage должно иметь отдельное поле для public route key, например
  `public_variant_key`, либо read-model mapping, который однозначно восстанавливает его
  из `{job_id, readable_slug, variant_hash}`;
- public API response всегда использует `variant_key` как readable route key и
  `variant_hash` как stable identity hash.

Инвариант адаптера:

- inbound route принимает только public `variant_key`;
- application layer разрешает его ровно в одну persisted top-N row внутри `job_id`;
- cache и lazy recompute keys включают и public `variant_key`, и `variant_hash`;
- прямой lookup по raw storage SHA не является публичным v1 contract.

### 7) Persisted top N summary, trades считаются лениво

Job completion сохраняет только top N summary.

Endpoint для lazy trades:

```http
POST /backtests/jobs/{job_id}/variants/{variant_key}/trades
```

Поведение:

- проверяет ownership;
- читает persisted job request snapshot и selected variant params;
- читает current artifact data по historical-prefix invariant и stored artifact metadata;
- cache hit возвращает trades + chart overlay payload;
- cache miss в production не должен выполнять тяжелый recompute внутри API request
  path; он создает или переиспользует materialization task, который исполняет
  `backtest-job-runner`;
- может сохранить cache на 1-2 дня.

Рекомендация по cache:

- metadata в Postgres;
- payload в локальном object/file cache под `/opt/roehub/state/backtest/trades_cache`;
- TTL по умолчанию: 48h;
- cache key включает `job_id`, `variant_key`, `variant_hash`, `request_hash`, `engine_params_hash`, `artifact_manifest_hash`.

Postgres-only JSONB допустим для малых payloads, но v1 должен избегать неограниченного раздувания основной БД.

Топология cache:

- v1 исходит из single API/worker host или sticky local cache semantics;
- cache miss является нормальным состоянием и должен запускать deterministic
  materialization для одного варианта через runner, а не тяжелый API request path;
- отказ local cache не должен ломать response, если materialization/recompute
  успешен;
- cache storage является outbound port, чтобы его можно было заменить на shared object storage;
- если API/worker deployment становится multi-host без sticky routing, shared object
  storage становится обязательным до production scale-out.

## Публичный API v1

### `POST /backtests/jobs`

Создает job.

Заголовки:

- опциональный `Idempotency-Key`;
- без `Idempotency-Key` каждый валидный request создает новый job;
- с `Idempotency-Key` повтор того же canonical request для того же пользователя в пределах
  настроенного TTL возвращает исходный job;
- повторное использование того же ключа с другим canonical request возвращает deterministic `409`.

Форма request:

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

Значения по умолчанию:

- пропущенные execution fields берутся из runtime config defaults;
- пропущенный ranking использует `total_return_pct desc`;
- пропущенный `top_n` использует config default;
- пропущенный `risk` должен отклоняться, если product явно не выберет `risk.mode = "none"` как default.

Важно: публичный `top_n` и canonical benchmark `top_k` — разные величины.

- `top_n` — продуктовый/API контракт: сколько summary rows нужно сохранить и
  вернуть для job. v1 default/max: `100`.
- `benchmark_top_k` — размер heap/top results в каноническом benchmark. Текущий
  notebook target вызывает `run_benchmark_matrix(..., top_k=5)`, поэтому все
  canonical timer targets для `heap_update` и `top_result_proxy_fill`
  относятся к `5` финальным top rows, а не к публичному `top_n = 100`.
- если service benchmark сравнивается с canonical notebook target, runner должен
  явно использовать `benchmark_top_k = 5` и записывать рядом `request.top_n`.
- production budget для `top_n = 100` измеряется отдельно как service-specific
  overhead и не должен смешиваться с notebook-compatible timer comparison.

Ответ:

- `201 Created` для нового job;
- `200 OK` для idempotent replay существующего job;
- payload включает `job_id`, `state`, `request_hash`, selected artifact metadata,
  links для `status`, `top`, `cancel` и `runtime-defaults`.

### `GET /backtests/jobs`

Возвращает историю jobs аутентифицированного пользователя через keyset pagination.

Query-параметры:

- `state`;
- `risk_mode`;
- `created_before`;
- `limit`;
- `cursor`.

Rows включают `job_id`, `state`, progress snapshot, coordinates, timeframe, time range,
ranking metric, requested `top_n`, timestamps `created/started/finished` и links.

### `GET /backtests/jobs/{job_id}`

Возвращает status, progress, hashes, selected artifact metadata и terminal summary counts.

### `GET /backtests/jobs/{job_id}/top`

Возвращает persisted top N summary rows.

Каждая row включает:

- `rank`;
- `variant_key`;
- `variant_hash`;
- `summary_metrics`;
- `best_tp_pct`;
- `best_sl_pct`;
- compact readable params;
- links/actions для lazy detail.

### `GET /backtests/jobs/{job_id}/variants/{variant_key}`

Возвращает один persisted variant summary и полный разбор параметров.

### `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`

Возвращает cached lazy trades для одного варианта или создает materialization task
для cache miss.

Production target:

- cache hit: `200` с trades/detail payload;
- cache miss: `202` со статусом materialization task, `retry_after_seconds` и
  ссылкой/контрактом для повторного чтения;
- sync recompute внутри API process допустим только как transitional non-production
  fallback с явным feature flag.

### `POST /backtests/jobs/{job_id}/cancel`

Запрашивает отмену. Endpoint идемпотентен:

- active jobs переходят к `cancelled`;
- already terminal jobs возвращают current terminal state;
- cancellation не удаляет уже committed persisted top-N summary.

### `GET /backtests/runtime-defaults`

Возвращает публичные defaults и limits для построения валидных requests:

- supported timeframes;
- supported и maximum indicator arity (`backtest.max_indicator_arity = 10`);
- available ranking metrics;
- execution defaults;
- sizing mode defaults и required fields;
- direction modes;
- default и max `top_n`;
- guardrail limits, используемые preflight.

### `POST /backtests/preflight`

Валидирует и нормализует request без создания job.

Возвращает:

- normalized effective request;
- `request_hash`;
- selected artifact metadata;
- estimated indicator rows, combinations и TP/SL cells;
- estimated cost class;
- blocking validation errors или warnings.

Preflight является advisory-проверкой для UX и API clients. `POST /backtests/jobs`
повторяет всю validation и остается authoritative create path.

### Каталог ошибок

| HTTP | Code | Retryable | Значение |
|---:|---|---|---|
| 400 | `backtest.invalid_json` | no | Payload не является JSON object или не может быть распарсен. |
| 401 | `auth.required` | yes after auth | Пользователь не аутентифицирован. |
| 403 | `backtest.forbidden` | no | Пользователь не владеет job или variant. |
| 404 | `backtest.not_found` | no | Job или variant не виден пользователю. |
| 409 | `backtest.idempotency_key_conflict` | no | Тот же `Idempotency-Key` повторно использован с другим canonical request. |
| 409 | `backtest.job_not_cancellable` | no | Cancellation недопустима для текущего state. |
| 422 | `backtest.invalid_request` | no | Coordinates, timeframe, range, indicator grid или execution settings невалидны. |
| 422 | `backtest.tp_sl_grid_not_covered` | no | Requested TP/SL grid не покрыт published `hit_times/15m`. |
| 422 | `backtest.request_too_expensive` | no | Request превышает настроенные cost/combination limits. |
| 429 | `backtest.rate_limited` | yes | User или service quota исчерпана. |
| 503 | `backtest.artifacts_unavailable` | yes | Required artifact family/current pointer недоступен. |
| 503 | `backtest.queue_saturated` | yes | Worker queue не может принять больше jobs в пределах limits. |

## Runtime-пайплайн

Runtime использует два словаря стадий:

- lifecycle stages: состояния request/job/queue, видимые API и persistence;
- benchmark stages: notebook-compatible измеряемые compute stages из канонического
  `btcusdt_15m_research_engine.ipynb`.

Продакшн-реализация должна сохранить алгоритмическую семантику notebook и
экспортировать записи бенчмарков с теми же timer names, что и
`benchmark_results.json`. Старый пятистадийный словарь с `count_trades` считается
замененным: подсчет сделок остается внутри self-check/reference paths и outputs
exact scoring, но не является production pipeline stage.

### Охват алгоритма и backends

Канонический prototype валидирует:

- coordinates: `binance` / `spot` / `BTCUSDT`;
- timeframe: `15m`;
- period semantics: `[start, end)` по 15m `open_time`;
- целевая public arity: 1..10 indicators для `risk.mode = "none"` и
  `risk.mode = "tp_sl_grid"`;
- canonical production acceptance benchmark покрывает arity 1..7; если сервис
  достигает 90% target по arity 1..7, перенос алгоритма в production runtime
  считается успешным;
- arity 8..10 разрешены контрактом request validation, но контролируются cost
  guardrails; отдельная Mac Studio benchmark iteration для них нужна только перед
  расширением production budget tiers, а не для завершения v1;
- benchmark directions: `long_only`, `long_short_reversal`;
- risk modes: `none`, `tp_sl_grid`;
- целевой TP/SL grid: `2.0..25.0` inclusive, step `0.5`
  (`47 x 47 = 2209` cells).

Реестр backends:

| Backend | Risk mode | Arity | Роль |
|---|---|---:|---|
| `event_segments_2_no_risk` | `none` | 2 | Default specialized two-indicator no-risk backend. |
| `streaming_2_no_risk` | `none` | 2 | Fallback/parity backend, не default production target. |
| `event_segments_n_no_risk` | `none` | 1, 3..10 | Generic no-risk backend. |
| `event_segments_n_tp_sl_15m_grid` | `tp_sl_grid` | 1..10 | Generic risk-on backend на основе `hit_times/15m`. |

Семантика direction:

- `long_only`: raw consensus `+1` открывает/держит long; raw `0` или `-1`
  закрывает открытый long; short trades никогда не открываются;
- `long_short_reversal`: raw consensus `+1` открывает/держит long; raw `-1`
  открывает/держит short; противоположный signal закрывает и разворачивает позицию.

### Lifecycle stage: нормализация request

- валидирует coordinates;
- валидирует, что timeframe исключает `1m` и `5m`;
- валидирует `[start, end)`;
- валидирует indicator ids по `configs/prod/indicators.yaml`, загруженному через
  registry/defaults;
- валидирует ranges `source` и `window`, затем materialize row selections;
- валидирует `direction_mode`, `sizing`, `profit_lock`, fees, slippage,
  `initial_cash_quote` and `close_on_end`;
- валидирует, что request TP/SL grid покрыт published artifact grid, когда
  `risk.mode = "tp_sl_grid"`;
- применяет execution defaults;
- рассчитывает canonical `request_hash`, result-affecting config hash и estimated
  request cost.

Эквиваленты методов notebook:

- `validate_request_indicators`;
- `canonical_json_hash`;
- `row_ids_for_sources`.

### Lifecycle stage: создание job

- создает row `backtest_jobs` с canonical request snapshot;
- записывает artifact metadata/watermark для audit и cache identity;
- state начинается как `queued`;
- worker фиксирует resolved artifact root и manifest hashes для job.

### Измеряемая стадия бенчмарка: `sample_warmup` / `service_warmup` / `numba_warmup`

Warmup измеряется отдельно от user-facing runtime. Канонический бенчмарк использует
sample warmup, а не полный dry-run:

- warmup rows per indicator: `min(2, rows_per_indicator)`;
- те же arity, risk mode, direction mode и backend, что у измеряемого run;
- JIT compilation и first-touch array costs относятся к warmup;
- measured `total_without_warmup` исключает warmup.

Сервис должен записывать warmup metrics и сравнивать их с canonical target, но
warmup не добавляется в `total_without_warmup`.

### Измеряемая стадия бенчмарка: `load_hit_times`

Только risk-on. Метод notebook: `load_tp_sl_hit_times_15m`.

Эта stage:

- читает `hit_times/15m/manifest.yaml`;
- читает `tp_values.f32.npy`, `sl_values.f32.npy`;
- сопоставляет requested TP/SL percentages с artifact indexes;
- загружает selected rows из `long_tp.u32.npy`, `long_sl.u32.npy`,
  `short_tp.u32.npy`, `short_sl.u32.npy`;
- копирует selected subset в contiguous arrays для kernels;
- precomputes fee-adjusted log factors для long и short TP/SL outcomes;
- записывает `hit_times_manifest_hash`.

### Измеряемая стадия бенчмарка: `tp_sl_grid_validation`

Только risk-on. Эта часть измеряется отдельно от загрузки hit-time arrays.

Валидация:

- request values интерпретируются как проценты и переводятся в decimal levels;
- каждый requested TP и SL level должен совпадать ровно с одним published artifact value
  с bounded float tolerance;
- missing levels детерминированно падают до compute с
  `422 backtest.tp_sl_grid_not_covered`;
- target benchmark grid: `2.0..25.0` inclusive, step `0.5`.

### Измеряемые стадии бенчмарка: artifact подготовка и `prepare_pools_core`

Методы notebook:

- `extract_signal_rows`;
- `prefilter_indicator_rows`;
- `fused_row_prefilter_stats`;
- `topk_fraction_idx`;
- `build_signal_segments`;
- `fill_signal_segments_i8`;
- `prepare_indicator_pool`;
- `prepare_indicator_pools`.

Stage contract разделяет notebook-compatible compute и service overhead.
Canonical notebook `prepare_pools` timer замеряет уже прогретые/opened arrays и
оборачивает только `prepare_indicator_pools(...)`. Поэтому 90% comparison с
canonical notebook prepare_pools применяется только к `prepare_pools_core`.

Service overhead stages измеряются отдельно и не сравниваются с canonical
notebook prepare_pools:

- `artifact_context_resolve`: trusted configuration/current pointer, slot
  `manifest.yaml`, manifest hash validation и typed runtime context;
- `artifact_array_open`: manifest-backed dtype/shape validation и открытие
  `.npy` handles через `np.load(..., mmap_mode="r")` для `prices/<tf>`,
  `prices/1m`, `mappings/<tf>` и requested
  `signals/<tf>/<indicator_id>/signals.i8.npy`;
- `request_slice_prepare`: slices 15m bars по `[start, end)` через `open_time`,
  derives 15m return intervals из close prices и derives 15m-to-1m execution
  mapping для no-risk mode:
  signal на 15m bar `t` входит на open следующего 15m bar, mapped to 1m.

Notebook-compatible `prepare_pools_core`:

- копирует только requested signal rows в contiguous `int8` matrices;
- применяет row prefilter per indicator:
  - `nonzero`: количество non-zero signal intervals;
  - `proxy`: dot-product-like directional return proxy;
  - `change_count`: количество signal change points;
  - `adjusted = proxy - fee_rate * nonzero`;
  - оставляет top fraction после `min_nonzero`;
- строит per-row metadata `{indicator_id, row_id, source, window}`;
- строит compressed signal segments:
  `starts`, `ends`, `values`, `counts`, `change_count`;
- возвращает indicator pools с `trade_T`, `eval_T`, `segments`, row ids,
  scores и metadata.

`prepare_pools_total` является aggregate service telemetry:
`artifact_context_resolve + artifact_array_open + request_slice_prepare +
prepare_pools_core`. Он нужен для production observability и service overhead
budgeting, но не является прямой notebook-compatible comparison metric.

### Измеряемая стадия бенчмарка: `build_exact_context`

Метод notebook: `build_segment_stack`.

Эта stage готовит arity-first segment arrays для generic exact kernels:

- `starts[arity, max_rows, max_segments]`;
- `ends[arity, max_rows, max_segments]`;
- `values[arity, max_rows, max_segments]`;
- `counts[arity, max_rows]`.

Для no-risk arity 2 со specialized backend сервис может читать segments напрямую
из каждого pool, и `build_exact_context` может быть близок к нулю. Для generic
no-risk и всех TP/SL risk-on runs эта stage обязательна.

### Измеряемая стадия бенчмарка: `build_proxy_context`

Методы notebook:

- `build_eval_stack`;
- `build_combo_proxy_cache_two`;
- `gather_combo_proxy_cache_two`.

Эта stage существует только когда combo prefilter активен:

- активна, когда `combo_top_frac < 1.0` или `combo_min_confirm > 1`;
- для arity 2 строит matrix-backed confirm/proxy lookup tables через
  `eval_T` и 15m returns;
- для generic N упаковывает `eval_T` в `eval_stack[arity, max_rows, n_intervals]`;
- в canonical target benchmark combo prefilter настроен как pass-through
  (`combo_top_frac = 1.0`, `combo_min_confirm = 1`), поэтому stage ожидается
  близкой к нулю, но все равно должна записываться.

### Измеряемая стадия бенчмарка: `combo_iteration`

Метод notebook: `iter_combo_chunks`.

Эта stage:

- строит deterministic Cartesian product по filtered local row pools;
- сохраняет indicator order из normalized request;
- emits chunks как `{indicator_id: int32[K]}`;
- использует bounded chunk size (`4096` в canonical benchmark);
- записывает `cartesian_combinations`, `combo_chunks_processed` и
  `exact_candidates_evaluated`.

### Измеряемая стадия бенчмарка: `proxy_filter`

Методы notebook:

- `proxy_prefilter_combos_chunk_two`;
- `proxy_prefilter_combos_chunk_n`;
- `topk_fraction_idx`.

Когда combo prefilter неактивен, эта stage выбирает весь chunk и записывает
near-zero time. Когда активен, она:

- рассчитывает consensus confirmation count per combo;
- рассчитывает cheap directional proxy score из 15m returns;
- применяет `combo_min_confirm`;
- оставляет top fraction по proxy score;
- передает только selected combos в exact scoring.

### Измеряемая стадия бенчмарка: `self_check`

Self-check является частью benchmark evidence и должен fail fast при parity drift.
Канонический benchmark использует `self_check_n = 2`.

Методы no-risk:

- `build_trade_list_for_indicator_rows_slow`;
- `evaluate_no_risk_reference_rows_slow`;
- `run_fast_vs_reference_self_check_two`.

Методы TP/SL:

- `build_trade_list_15m_for_indicator_rows_slow`;
- `evaluate_tp_sl_reference_trade_list_direct`;
- `evaluate_tp_sl_reference_rows_slow`;
- `run_tp_sl_self_check`.

Проверки:

- backend `trade_count` равен slow reference;
- no-risk `total_return_pct` отличается не больше чем на `1e-4`;
- TP/SL best return отличается не больше чем на `5e-5`;
- TP/SL best cell indexes валидны;
- TP/SL best TP/SL cell должна совпадать с reference, если return difference
  не является численно незначимой.

### Измеряемая стадия бенчмарка: `exact_scoring`

Это доминирующий hot path. Dispatch выполняется по `risk.mode`.

Для `risk.mode = "none"`:

- default backend: `event_segments_2_no_risk` для arity 2;
- generic backend: `event_segments_n_no_risk` для arity 1 и 3..10;
- `streaming_2_no_risk` существует только как fallback/parity comparator;
- segment intersections создают raw consensus direction;
- `apply_direction_mode` отображает raw direction в `long_only` или
  `long_short_reversal`;
- entries используют open следующего 15m signal bar, mapped to 1m execution index;
- exits используют signal close/reversal, mapped to 1m open, либо final 1m close, когда
  `close_on_end = true`;
- `apply_no_risk_trade_to_state` обновляет cash/equity state без allocation
  полного trade list в hot path;
- summary metrics включают `total_return_pct`, `max_drawdown_pct`,
  `return_over_max_drawdown`, `profit_factor`, `trade_count`, `sharpe_trades`,
  `win_rate_pct`, `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`.

Для `risk.mode = "tp_sl_grid"`:

- backend: `event_segments_n_tp_sl_15m_grid` для arity 1..10;
- `exact_scoring` и `tp_sl_exact_scoring` записывают один и тот же hot-path elapsed time;
- entries и signal exits представлены как absolute 15m bar indexes;
- TP/SL hit tables: `hit_times/15m`;
- scoring использует log-return accumulation для numerical stability;
- для каждой candidate trade `tp_sl_apply_trade_to_diff` записывает contribution
  ranges в три difference buffers:
  - `row_diff` для TP-only ranges;
  - `col_diff` для SL-only ranges;
  - `rect_diff` для signal/final-close fallback rectangles;
- prefix sums materialize полный TP/SL grid contribution для одной combo;
- best cell — это max log-return cell, converted back to
  `total_return_pct = (exp(best_log) - 1) * 100`;
- если TP и SL hit на одном bar, SL wins the tie: TP требует
  `t_tp < t_sl`, SL допускает `t_sl <= t_tp`;
- final persisted summary должен включать тот же full metric set, что и no-risk
  (`total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`) плюс
  `best_tp_pct` и `best_sl_pct` для выбранной best cell;
- текущий notebook benchmark hot path для TP/SL выбирает best cell по return и
  сохраняет `trade_count`, `best_tp_pct`, `best_sl_pct`; full metric set для
  selected best cell в service может считаться отдельным
  `tp_sl_full_metrics_second_pass`, чтобы не раздувать `exact_scoring` boundary.

### Измеряемая стадия бенчмарка: `heap_update`

Notebook использует Python `heapq`, чтобы держать top K.

Для canonical acceptance эта stage должна воспроизводить именно notebook
boundary:

- full benchmark run использует `benchmark_top_k = 5`;
- sample warmup использует `top_k = 1`;
- `request.top_n = 100` из публичного fixture не применяется к canonical
  comparison этой stage;
- если service дополнительно прогоняет product mode с `top_n = 100`, этот
  результат записывается как отдельный service-specific budget, а не как
  замена canonical `heap_update`.

Эта stage:

- ranks по selected metric, default `total_return_pct desc`;
- строит deterministic heap key из score и original row ids;
- держит только `top_k` rows в heap;
- добавляет compact per-indicator metadata только для rows, которые реально
  удержаны в heap и нужны для notebook-compatible `top_results`;
- для no-risk heap item содержит full summary metrics, `_local_indices` и
  `_proxy_pending`, как в notebook;
- для TP/SL heap item содержит `total_return_pct`, `best_tp_pct`,
  `best_sl_pct`, `trade_count` и deterministic ordinal tie-breaker, как в
  notebook benchmark.

Notebook-compatible в этой stage означает одинаковые ranking, tie-break,
cardinality и final `top_results`, а не обязательное повторение лишних Python
allocations для rejected candidates. Service implementation должна быть
low-allocation:

- canonical benchmark path `total_return_pct desc` должен идти через прямое
  чтение `scores.total_return_pct[index]`, без generic string dispatch в tight
  loop;
- сначала рассчитываются только `rank_score`, original row ids и heap key;
- full `item` dict, full summary metrics и per-indicator metadata materialization
  выполняются только если heap еще не заполнен или candidate заменяет текущий
  worst heap item;
- generic ranking по другим метрикам может существовать, но benchmark evidence
  для default `total_return_pct desc` не должен платить за polymorphic dispatch;
- metadata conversion из domain objects в mapping не должна выполняться для
  candidates, которые не попали в final heap.

Запрещено включать в `heap_update`:

- генерацию public `variant_key`;
- расчет `variant_hash` / `indicator_variant_hash`;
- сборку API DTO / persisted row objects;
- validation под legacy SHA-only storage schema;
- запись в БД или object storage.

### Измеряемая стадия бенчмарка: `top_result_proxy_fill`

Эта stage не является lazy trades.

Она выполняется только когда final top rows не получили proxy metadata из active
combo prefilter. Она пересчитывает `confirm_count` и `proxy_score` для top rows
через `proxy_for_indicator_rows`. Она не должна маппиться на UI/API endpoint
`show trades`.

Notebook-compatible boundary:

- вход stage — уже заполненный heap размера `top_k`;
- stage сортирует heap descending по heap key;
- для каждого final top row, где `_proxy_pending = true`, строит `eval_rows`
  из `indicator_pools[indicator_id]["eval_T"][local_index]`;
- вызывает `proxy_for_indicator_rows(...)` ровно для final top rows, а не для
  всех evaluated candidates;
- `proxy_for_indicator_rows(...)` должен повторять текущий notebook dispatch:
  для `len(eval_rows) == 2` используется compiled scalar fast path
  `proxy_for_two_rows(...)`, а generic consensus copy/mask path используется для
  arity 1 и arity 3..10;
- мутирует compact summary item полями `confirm_count` и `proxy_score`;
- удаляет notebook-internal `_local_indices` и `_proxy_pending`;
- возвращает `top_results` длиной не больше `benchmark_top_k`.

Benchmark по `top_result_proxy_fill` должен сравниваться per tuple
`{arity, direction_mode}`. Median pass не достаточен: отдельный fail arity 2
означает, что notebook arity-2 fast path не перенесен корректно.

Запрещено включать в `top_result_proxy_fill`:

- lazy trades recompute;
- повторный exact scoring;
- расчет public/storage identity;
- сборку persisted rows;
- batch-fill proxy metadata для `request.top_n = 100`, если benchmark
  сравнивается с canonical `top_k = 5`.

Если production runtime хочет сохранять `top_n = 100`, это отдельный режим:
сначала должна пройти canonical parity на `top_k = 5`, затем измеряется
service-specific `top_n = 100` overhead.

### Service-only стадия: `top_result_assembly`

Эта stage существует только в production service и не имеет notebook baseline.

Она принимает notebook-compatible top rows после `heap_update` /
`top_result_proxy_fill` и строит persisted/API shape:

- canonical variant params;
- readable public `variant_key`;
- stable `variant_hash`;
- optional `indicator_variant_hash`;
- DTO/read-model rows для `GET /backtests/jobs/{job_id}/top`;
- mapping public identity to storage identity.

`top_result_assembly` измеряется с CPU/RSS evidence и regression baseline после
первой реализации, но не участвует в 90% comparison с notebook timers.

### Service-only стадия: `tp_sl_full_metrics_second_pass`

Текущий notebook target для TP/SL exact scoring оптимизирован под поиск best
TP/SL cell. Он не доказывает, что весь no-risk-like metric set для выбранной cell
посчитан бесплатно.

Production service должен вернуть full summary metrics и для risk-on variants,
поэтому после выбора top rows и best cell может потребоваться второй bounded
проход по сделкам:

- вход: final top rows, selected `best_tp_idx` / `best_sl_idx`, indicator row
  combination и pinned execution context;
- расчет: восстановить trade/equity stats для selected cell теми же execution,
  direction, sizing, fees, slippage и `close_on_end` settings;
- выход: `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`,
  `sharpe_trades`, `win_rate_pct`, `avg_trade_ret_pct`,
  `avg_trade_exec_bars`, `exposure_pct` плюс уже выбранные
  `total_return_pct`, `trade_count`, `best_tp_pct`, `best_sl_pct`;
- cardinality: только persisted/in-memory top rows, а не все evaluated combos.

Эта stage service-only: она измеряется отдельно и не ухудшает canonical
`exact_scoring` / `tp_sl_exact_scoring` comparison. Если позже notebook будет
расширен и начнет считать эти метрики внутри TP/SL hot path, canonical baseline
нужно перезаписать отдельной benchmark iteration.

### Runtime total без warmup

Каждый benchmark record должен экспонировать notebook-compatible timer names:

| Timer | Required | Примечания |
|---|---:|---|
| `sample_warmup` / `service_warmup` / `numba_warmup` | yes | Измеряется отдельно, исключается из total. |
| `load_hit_times` | risk-on only | Загрузка hit-time subset. |
| `tp_sl_grid_validation` | risk-on only | Проверка coverage request grid. |
| `prepare_pools` | yes | Artifact load, slicing, row selection, row prefilter, segment build. |
| `build_exact_context` | yes | Arity-first segment context, где требуется. |
| `build_proxy_context` | yes | Может быть близок к нулю, когда proxy prefilter является pass-through. |
| `combo_iteration` | yes | Генерация Cartesian chunks. |
| `proxy_filter` | yes | Pass-through или active combo pruning. |
| `self_check` | benchmark/test yes | Ограниченная parity check. |
| `exact_scoring` | yes | No-risk или TP/SL exact scorer. |
| `tp_sl_exact_scoring` | risk-on only | Alias/subsegment of `exact_scoring` for risk-on. |
| `heap_update` | yes | Top-N heap maintenance. |
| `top_result_proxy_fill` | no-risk yes | Заполнение top-row proxy metadata. |
| `top_result_assembly` | service only | Public/storage identity, DTO/read-model assembly; не часть notebook baseline. |
| `tp_sl_full_metrics_second_pass` | service only | Full metric set для выбранной best TP/SL cell; не часть текущего notebook baseline. |
| `total_without_warmup` | yes | Notebook-compatible measured runtime после warmup. |
| `service_total_without_warmup` | service only | Полный service runtime после warmup до доступности terminal result, включая service-only overhead. |
| `persist_top_n_io` | service only | DB write overhead; не часть notebook baseline. |

`total_without_warmup` для 90% comparison должен совпадать по boundary с
notebook runtime и исключать `top_result_assembly`,
`tp_sl_full_metrics_second_pass`, `persist_top_n_io`,
`lazy_trades_compute` и другие service-only stages. Для user-facing SLA сервис
дополнительно записывает `service_total_without_warmup`, где эти расходы уже
видны отдельно и в сумме.

Memory cleanup начинается после того, как summary result стал доступен
пользователю или persistence layer. Cleanup не является canonical benchmark
stage, не меняет порядок notebook stages и не сравнивается по правилу `>= 90%`
до появления собственного accepted service baseline. При этом worker обязан
освободить per-job heavy objects до того, как slot считается свободным для
следующего heavy job.

### Lazy trades

- recompute exact trades для одного `variant_key`;
- cache result на 48h;
- cache hit возвращает chart-ready payload из API;
- cache miss материализуется через `backtest-job-runner`, чтобы Web UI detail view
  не блокировал API process.

Lazy trades не является частью `total_without_warmup`; у него свой benchmark gate:
`lazy_trades_compute` и `lazy_trades_cache_hit`.

### Маппинг progress

Persisted job state может оставаться coarse (`stage_a`, `stage_b`, `finalizing`),
а API progress экспонирует более детальную pipeline stage.

| API `progress.pipeline_stage` | Persisted stage | Примечания |
|---|---|---|
| `queued` | `stage_a` | Job существует, но worker еще не стартовал. |
| `service_warmup` | `stage_a` | Sample/JIT warmup перед measured runtime. |
| `load_hit_times` | `stage_b` | Risk-on загрузка hit-times subset. |
| `tp_sl_grid_validation` | `stage_b` | Risk-on validation grid coverage. |
| `artifact_context_resolve` | `stage_a` | Service overhead: current pointer, manifest identity и typed context. |
| `artifact_array_open` | `stage_a` | Service overhead: mmap open через `np.load(..., mmap_mode="r")` и manifest validation. |
| `request_slice_prepare` | `stage_a` | Service overhead: `[start, end)` slice, returns и execution mapping. |
| `prepare_pools_core` | `stage_a` | Notebook-compatible row selection, prefilter и segment build. |
| `prepare_pools_total` | `stage_a` | Aggregate service telemetry, не notebook comparison target. |
| `build_exact_context` | `stage_a` / `stage_b` | Segment stack для exact kernels. |
| `build_proxy_context` | `stage_a` | Optional combo proxy context. |
| `combo_iteration` | `stage_a` / `stage_b` | Cartesian chunk planning. |
| `proxy_filter` | `stage_a` / `stage_b` | Optional combo pruning. |
| `self_check` | `stage_a` / `stage_b` | Benchmark/test parity check. |
| `exact_scoring` with `risk.mode = "none"` | `stage_a` | Нет Stage B risk grid. |
| `exact_scoring` / `tp_sl_exact_scoring` with `risk.mode = "tp_sl_grid"` | `stage_b` | TP/SL scoring на hit-times. |
| `heap_update` | `finalizing` | Ranking и top-N assembly. |
| `top_result_proxy_fill` | `finalizing` | Top-row proxy metadata, не lazy trades. |
| `top_result_assembly` | `finalizing` | Service-only public/storage identity и DTO/read-model assembly. |
| `tp_sl_full_metrics_second_pass` | `finalizing` | Service-only TP/SL metrics recompute для selected best cells. |
| `persist_top_n_io` | `finalizing` | Service-only DB write overhead. |
| `succeeded`, `failed`, `cancelled` | terminal state | Terminal state приоритетнее stage. |

Контракт:

- API consumers должны использовать `state` для lifecycle и `progress.pipeline_stage` для UI detail;
- persisted stage является implementation/read-model compatibility field;
- записи бенчмарков используют canonical notebook timer names, а не legacy persisted stage names.

## Целевая структура модулей

Планируемая целевая структура:

- `src/trading/contexts/backtest/domain/` — job aggregate, request/value objects, variant identity, execution/sizing value objects.
- `src/trading/contexts/backtest/application/use_cases/` — create job, read status/top, read variant, compute lazy trades.
- `src/trading/contexts/backtest/application/services/v2/` — runtime pipeline на артефактах, row selection, combo planning, scoring orchestration.
- `src/trading/contexts/backtest/application/ports/` — artifact loaders, job repositories, cache storage, metrics, current user.
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/` — строгие artifact readers поверх contracts `backtest_artifacts`.
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/` — jobs/top/cache metadata repositories.
- `apps/api/routes/backtests.py` — public API routes.
- `apps/api/wiring/modules/backtest.py` — composition root.
- `apps/web/templates/backtests_*.html` и `apps/web/dist/backtest_ui.js` — UI integration в более поздней итерации.

Направление зависимостей:

```text
apps/api routes
  -> backtest application use cases
    -> backtest domain/value objects
    -> application ports
      <- outbound adapters: artifacts_fs, postgres, cache_fs, metrics
```

`backtest` может потреблять contracts `backtest_artifacts` через adapter/ACL, но runtime orchestration должен жить в `backtest`, а не в publisher/precompute code.

## Политика бенчмарков

Запуск бенчмарков разрешен только на `Mac Studio`.

Канонические benchmark sources:

- canonical algorithm:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
- target numeric evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- target human-readable summary:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`.

JSON evidence является источником истины для numeric target values. Summary нужен
только для удобства review и не должен вручную редактироваться отдельно.

Особенность текущего canonical evidence: JSON request сохраняет публичный
`top_n = 100`, но benchmark entry point вызывает
`run_benchmark_matrix(..., top_k=5)`. Поэтому comparison по `heap_update`,
`top_result_proxy_fill`, `total_without_warmup` и result hashes должен учитывать
фактический `top_results_count = 5`. Реализация, которая внутри measured
notebook-compatible stages строит 100 rows, сравнивается с неправильной
нагрузкой и не может быть принята как доказательство parity.

Известный непринятый benchmark record:

- [`2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk`](benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/)
  является failure record, а не accepted baseline;
- этот record подтверждает, что `exact_scoring` может пройти `14 / 14`, но
  `heap_update` fail `13 / 14`, `top_result_proxy_fill` fail для arity 2, а
  `total_without_warmup` был сравнен с `service_total_without_warmup`, то есть с
  неверной boundary;
- следующие Iteration 4 prompts и benchmark runner должны использовать этот
  record как чеклист против регрессий, а не как target values.

Идентичность канонического benchmark:

- host: `macstudio`;
- period: `[2020-01-11T20:08:00Z, 2026-04-11T20:08:00Z)`;
- rows per indicator: `6`;
- warmup rows per indicator: `2`;
- public request `top_n`: `100`;
- canonical benchmark `top_k`: `5` для measured runs, `1` для sample warmup;
- canonical production acceptance arities: `1..7`;
- target public request arities: `1..10` для no-risk и TP/SL; arity 8..10
  разрешены только в пределах cost guardrails и не входят в обязательный 90%
  acceptance benchmark для завершения v1;
- direction modes: `long_only`, `long_short_reversal`;
- risk modes: `none`, `tp_sl_grid`;
- TP/SL grid: `2.0..25.0` inclusive, step `0.5`;
- TP/SL cells per combo: `2209`;
- runs: `28` (`7 arities x 2 risk modes x 2 direction modes`);
- фактический `top_results_count`: `5` в каждом canonical measured run;
- request hash:
  `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`;
- artifact manifest hash:
  `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`;
- hit-times manifest hash:
  `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`.

Каждая implementation iteration должна записывать:

- code version / branch / commit;
- artifact config и artifact root;
- artifact slot и `artifact_manifest_hash`;
- использованный notebook baseline;
- notebook baseline output path или captured metrics;
- request fixture;
- public `request.top_n`, `benchmark_top_k`, фактический
  `top_results_count` и heap capacity, использованный runner-ом;
- canonical `request_hash` и result-affecting config hash;
- service warmup metrics;
- canonical notebook timer metrics без warmup;
- service-only overhead metrics (`artifact_context_resolve`, `artifact_array_open`,
  `request_slice_prepare`, `prepare_pools_total`, `top_result_assembly`,
  `tp_sl_full_metrics_second_pass`, `service_total_without_warmup`);
- memory cleanup evidence как отдельная service hygiene check, а не как
  canonical benchmark stage;
- speed ratio vs baseline;
- absolute latency budget result;
- peak RSS / memory delta;
- CPU time, process CPU percent, thread count, effective Numba threads, system load;
- pass/fail against 90% threshold;
- correctness/parity result.

Политика warmup:

- `service_warmup`, `numba_warmup` и `sample_warmup` являются first-class measured
  segments;
- canonical benchmark использует sample warmup на `min(2, rows_per_indicator)` rows
  per indicator для того же arity/risk/direction/backend;
- notebook-compatible runtime benchmark (`total_without_warmup`) измеряется после
  warmup;
- user-facing service runtime (`service_total_without_warmup`) измеряется после
  warmup отдельно и включает service-only overhead;
- warmup и warm runtime оба должны оставаться в accepted 90% envelope для
  соответствующего segment.

Canonical notebook-compatible benchmark stages сохраняют порядок и состав,
экспонируемый текущим `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.
Runner может писать дополнительные service-only telemetry fields, но не должен
добавлять их в этот ordered stage list.

1. `service_warmup`
2. `numba_warmup`
3. `sample_warmup`
4. `total_without_warmup`
5. `load_hit_times` для `risk.mode = "tp_sl_grid"`
6. `tp_sl_grid_validation` для `risk.mode = "tp_sl_grid"`
7. `prepare_pools_core` как service alias для notebook `prepare_pools`
8. `build_exact_context`
9. `build_proxy_context`
10. `combo_iteration`
11. `proxy_filter`
12. `self_check`
13. `exact_scoring`
14. `tp_sl_exact_scoring` для `risk.mode = "tp_sl_grid"`
15. `heap_update`
16. `top_result_proxy_fill`

Canonical JSON также может содержать `total` как historical/no-risk alias для
`total_without_warmup`. Это не отдельная stage и не добавляет новый comparison
gate.

Service-only telemetry fields не являются canonical stages и не имеют notebook
target values:

- `artifact_context_resolve`;
- `artifact_array_open`;
- `request_slice_prepare`;
- `prepare_pools_total`;
- `service_total_without_warmup`;
- `top_result_assembly`;
- `tp_sl_full_metrics_second_pass`;
- `persist_top_n_io`;
- `lazy_trades_compute`;
- `lazy_trades_cache_hit`.

Сравнение acceptance:

- notebook-compatible stages сравниваются с target values по tuple
  `{arity, risk_mode, direction_mode, backend}`;
- benchmark runner обязан записывать два разных поля:
  - `total_without_warmup`: сумма только notebook-compatible stages;
  - `service_total_without_warmup`: user-facing service runtime до доступности
    terminal result, включая service-only overhead;
- для Iteration 4 no-risk `total_without_warmup` считается как
  `prepare_pools_core + build_exact_context + build_proxy_context +
  combo_iteration + proxy_filter + self_check + exact_scoring + heap_update +
  top_result_proxy_fill`;
- `artifact_context_resolve`, `artifact_array_open`, `request_slice_prepare`,
  `prepare_pools_total`, `top_result_assembly`, `persist_top_n_io`,
  memory cleanup и любые DTO/storage/cache assembly steps не должны попадать в
  `total_without_warmup`;
- в benchmark evidence `backend` может быть display/logical name
  (`event_segments_1_no_risk`, `event_segments_7_no_risk`), а runtime registry
  может использовать общий implementation id (`event_segments_n_no_risk`) для
  arity 1 и 3..10; acceptance runner обязан записывать оба поля и сравнивать
  правильный tuple `{arity, risk_mode, direction_mode, exact_engine,
  implementation}`;
- для Iteration 2 canonical notebook prepare_pools сравнивается только с
  `prepare_pools_core`;
- `artifact_context_resolve`, `artifact_array_open`, `request_slice_prepare`,
  `prepare_pools_total`, `top_result_assembly`,
  `tp_sl_full_metrics_second_pass`, `service_total_without_warmup`,
  `persist_top_n_io`,
  `lazy_trades_compute` и `lazy_trades_cache_hit` являются service
  overhead/telemetry stages; они
  измеряются с CPU/RSS evidence, но не сравниваются с canonical notebook timer
  targets;
- `heap_update` и `top_result_proxy_fill` сравниваются с canonical notebook target
  только при том же `benchmark_top_k = 5`; product run с `top_n = 100`
  требует отдельного service-specific budget record;
- для arity 1..7 target source:
  `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- arity 8..10 не блокируют v1 completion, если arity 1..7 проходят 90%
  threshold; перед увеличением production budgets для широких arity 8..10
  workloads нужно создать follow-up Mac Studio benchmark iteration;
- `persist_top_n_io`, `lazy_trades_compute` и
  `lazy_trades_cache_hit` используют service-specific absolute budgets плюс regression
  comparison после появления собственного baseline;
- memory cleanup evidence использует service-specific retained RSS / recycle
  checks; это не canonical stage и не участвует в `>= 90%` comparison до
  появления собственного accepted service baseline;
- implementation может записывать lower-level subsegments, но pass/fail должен включать
  каждый canonical timer, экспонируемый notebook;
- latency target проходит, когда service wall time не хуже canonical target,
  деленного на `0.90`, для того же segment;
- memory target проходит, когда service peak RSS / RSS delta не хуже canonical target,
  деленного на `0.90`, если не установлен более строгий absolute budget;
- CPU target проходит, когда service CPU time не хуже canonical target, деленного на
  `0.90`, а process CPU percent и thread count записаны для диагностики.

Фрагмент canonical target table:

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

Полная target table, per-stage timers, runtime metrics и result hashes находятся
только в `benchmark_results.json`.

Записи бенчмарков хранятся в:

- `docs/architecture/backtest/benchmark_iterations/`

Правило завершения stage:

- каждая implementation stage завершается benchmark record в
  `docs/architecture/backtest/benchmark_iterations/<date>_<stage>/`;
- stage не считается complete, пока ее benchmark record не включает code version,
  request hash, artifact hashes, canonical timers, CPU/RSS metrics и
  correctness/parity evidence;
- следующая stage не должна считаться accepted, пока предыдущая stage не пройдет
  свой benchmark gate.

## Матрица тестов

Реализация не считается complete, пока functional и benchmark coverage не включают:

- benchmark matrix для production acceptance:
  `arity 1..7 x risk.mode none/tp_sl_grid x direction_mode long_only/long_short_reversal`;
- service-level correctness smoke для arity 8..10 на малых row pools, чтобы
  подтвердить contract support без включения этих arities в обязательный
  performance gate v1;
- `risk.mode = "none"`;
- `risk.mode = "tp_sl_grid"` с request TP/SL subset, покрытым `hit_times/15m`;
- TP/SL benchmark grid `2.0..25.0` inclusive, step `0.5`;
- каждый sizing mode: `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- `profit_lock` disabled и enabled;
- каждый supported `direction_mode` из runtime defaults:
  `long_only`, `long_short_reversal`;
- `close_on_end = true` и `close_on_end = false`, где `close_on_end = false`
  покрывается service-level correctness tests, а не обязательным notebook
  benchmark;
- full persisted summary metrics для no-risk и TP/SL risk-on variants;
- public API contract tests для create/status/list/top/variant/trades/cancel/defaults/preflight;
- idempotency tests для replay и key conflict;
- ownership/authz tests для чтения job, top и lazy trades;
- golden parity tests против notebook-derived fixtures;
- failure injection для missing artifacts, stale current pointer, TP/SL grid not covered,
  request too expensive, queue saturation и lazy cache failure;
- cache identity tests, покрывающие `job_id`, `variant_key`, `variant_hash`,
  `request_hash`, `engine_params_hash` и `artifact_manifest_hash`.

Текущие canonical `sizing_smoke` evidence имеют compiled parity для `all_in` и
`fixed_quote`; equity-percent sizing modes в notebook evidence являются
reference-only. Iteration 8 service implementation является first compiled
parity point для
`fixed_equity_pct`, `fixed_equity_pct_min_quote` и
`fixed_equity_pct_max_quote`; service-level parity evidence записана в
[`2026-05-02_iteration_8_execution_sizing_completion`](benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/).

## Операционные аспекты

Метрики:

- `backtest_jobs_created_total{risk_mode}`
- `backtest_jobs_completed_total{status,risk_mode}`
- `backtest_job_duration_seconds{risk_mode}`
- `backtest_stage_duration_seconds{stage,risk_mode}`
- `backtest_stage_cpu_seconds_total{stage,risk_mode}`
- `backtest_stage_peak_rss_bytes{stage,risk_mode}`
- `backtest_lazy_trades_requests_total{cache_status}`
- `backtest_lazy_trades_duration_seconds{cache_status}`
- `backtest_job_cleanup_duration_seconds{risk_mode}`
- `backtest_job_retained_rss_bytes{risk_mode}`
- `backtest_worker_recycles_total{reason}`
- `backtest_artifact_runtime_load_duration_seconds{family}`
- `backtest_artifact_runtime_manifest_hash_info`
- `backtest_request_cost_estimate{risk_mode,cost_class}`
- `backtest_requests_rejected_total{reason}`
- `backtest_jobs_cancel_requested_total{state}`

Security и доступ:

- все endpoints требуют authenticated user identity;
- каждый job, top row, variant и lazy trades response scoped by owner;
- API никогда не принимает artifact paths из request payloads;
- artifact root, cache root и runtime config приходят только из trusted config;
- file/object cache permissions должны предотвращать cross-user direct filesystem access;
- request payloads и logs не должны включать secrets.

Ресурсные guardrails:

| Guardrail | Config key | v1 default | Что ограничивает | Failure |
|---|---|---:|---|---|
| Active jobs per user | `backtest.max_active_jobs_per_user` | `1` | Сколько jobs одного пользователя могут одновременно быть в `running/warming/scoring`. Один arity 7 benchmark уже загружает CPU примерно на уровень одного heavy worker. | `429 backtest.rate_limited` |
| Queued jobs per user | `backtest.max_queued_jobs_per_user` | `3` | Сколько jobs пользователь может держать в очереди сверх active job. | `429 backtest.rate_limited` |
| Global active jobs | `backtest.max_active_jobs_global` | `1` | Сколько heavy jobs весь сервис исполняет параллельно. Увеличивать до `2+` можно только после отдельного concurrency benchmark, иначе 90% latency/CPU target станет нестабильным. | `503 backtest.queue_saturated` |
| `top_n` | `backtest.max_top_n` | `100` | Сколько summary rows сохраняется и возвращается по job. Большие значения увеличивают heap work, payload и DB write. | `422 backtest.request_too_expensive` |
| Indicator arity | `backtest.max_indicator_arity` | `10` | Максимальное число indicators в request. Production acceptance benchmark обязателен для arity 1..7; arity 8..10 допускаются только при прохождении остальных cost guardrails. | `422 backtest.request_too_expensive` |
| Indicator rows after source/window expansion | `backtest.max_indicator_rows` | `1000` | Суммарное число signal rows после раскрытия всех `source` и `window` ranges, до row prefilter. Например 5 sources x 200 windows = 1000 rows уже весь default budget. | `422 backtest.request_too_expensive` |
| Candidate combinations after row prefilter | `backtest.max_candidate_combinations` | `300000` | Число combinations перед exact scoring после row prefilter. Default покрывает canonical arity 7 fixture (`6^7 = 279936`) и режет requests вроде `20^5 = 3200000`. | `422 backtest.request_too_expensive` |
| TP/SL cells | `backtest.max_tp_sl_cells` | `2209` | Размер request TP/SL grid. Default равен canonical grid `47 x 47` для `2.0..25.0` step `0.5`. | `422 backtest.request_too_expensive` |
| Lazy trades requests per user window | `backtest.lazy_trades_rate_limit` | `30 / 10 min` | Сколько lazy trades detail запросов пользователь может сделать за sliding window. | `429 backtest.rate_limited` |
| Job queue wait | `backtest.job_queue_timeout_seconds` | `300` | Максимальное ожидание job в очереди до terminal failure. | terminal job failure |
| Job wall time | `backtest.job_wall_timeout_seconds` | `900` | Максимальное wall-clock время исполнения job. Requests, которые по estimate не помещаются в этот budget, должны отсеиваться preflight. | terminal job failure |
| Lazy trades wall time | `backtest.lazy_trades_timeout_seconds` | `30` | Максимальное время ленивого пересчета сделок по одному `variant_key`. | `503` retryable |
| Worker retained RSS recycle | `backtest.worker_recycle_retained_rss_mb` | `256` | Если после cleanup boundary worker удерживает больше configured RSS delta относительно baseline, worker должен быть recycled до следующего heavy job. | worker recycle |

Default tier intentionally близок к canonical benchmark workload. Для платных или
админских tiers можно расширять `max_top_n`, `max_indicator_rows`,
`max_candidate_combinations`, `max_active_jobs_global` и
`job_wall_timeout_seconds`, но только после отдельного benchmark record на
`Mac Studio` для этого tier.

Продакшн-выкатка блокируется, если эти config keys не заданы или если preflight
не может объяснить, какой guardrail отклонил request и как пользователю сузить
request.

Поведение при failure:

- invalid request возвращает deterministic 422;
- missing artifact family возвращает deterministic runtime failure на job;
- request TP/SL, не покрытый published grid, по возможности возвращает deterministic 422 до job execution;
- lazy trades cache failure не должен ломать trades response, если recompute успешен;
- benchmark failure блокирует текущую iteration от статуса complete.

### Жизненный цикл памяти и cleanup

Backtest job должен исполняться как bounded memory scope. Summary result может
оставаться в памяти только в compact форме, достаточной для response/persistence;
все heavy per-job objects должны освобождаться после доступности terminal result.

Обязательные правила:

- artifact `.npy` handles могут удерживаться только bounded artifact runtime cache;
  per-job slices, contiguous copies, score arrays, segment stacks, hit-time
  subsets, combo buffers, heaps больше final top rows и self-check reference data
  не должны жить дольше job;
- application result DTO не должен содержать references на prepared pools,
  proxy/exact contexts, score arrays, hit-times arrays или full evaluated
  candidates;
- worker обязан иметь `try/finally` cleanup boundary вокруг scoring path:
  удалить strong references на heavy objects, очистить per-job containers и
  зафиксировать cleanup telemetry;
- `gc.collect()` допустим как fallback cleanup step после удаления references,
  но основная гарантия должна строиться на отсутствии retained references;
- если Python/macOS allocator не возвращает RSS операционной системе, service
  должен использовать process worker recycle по
  `backtest.worker_recycle_retained_rss_mb` до следующего heavy job;
- benchmark evidence должен записывать `rss_before`, `rss_peak`,
  `rss_after_cleanup`, `retained_rss_delta`, cleanup duration и факт recycle.

Acceptance по memory cleanup для каждой compute iteration:

- один run проверяет, что returned result содержит только compact top rows и
  scalar telemetry;
- repeated-run smoke выполняет один и тот же heavy request минимум 3 раза подряд
  в одном worker lifecycle и доказывает отсутствие монотонного роста retained RSS;
- если retained RSS превышает configured recycle threshold, iteration может быть
  принята только при доказанном worker recycle до следующего heavy job.

## План внедрения

Правило для всех итераций:

- каждая iteration имеет explicit benchmark/evidence gate;
- benchmark records записываются до того, как iteration помечается complete;
- более поздняя stage может прототипироваться локально, но она не должна считаться
  accepted, пока все предыдущие stage gates не пройдены.
- implementation prompts должны явно разделять notebook-compatible measured
  stages и service-only stages;
- код, который собирает API DTOs, storage identity, persisted rows, cache keys или
  DB writes, не должен попадать в timings notebook-compatible stages;
- когда публичный request parameter отличается от canonical benchmark parameter
  (`top_n` vs `benchmark_top_k`), benchmark record обязан записывать оба значения
  и фактическую cardinality обработанных rows.
- после доступности terminal result каждая compute iteration должна иметь
  memory cleanup evidence: heavy per-job references освобождены, retained RSS
  не растет монотонно на repeated runs или worker recycle сработал до следующего
  heavy job.

Текущий статус принятых итераций:

| Итерация | Статус | Принятые benchmark records |
|---:|---|---|
| 0 | `pass` | [`2026-04-26_iteration_0_docs_benchmark_harness`](benchmark_iterations/2026-04-26_iteration_0_docs_benchmark_harness/) |
| 1 | `pass` | [`2026-04-26_iteration_1_request_normalization_artifact_context`](benchmark_iterations/2026-04-26_iteration_1_request_normalization_artifact_context/) |
| 2 | `pass` | [`2026-04-26_iteration_2_prepare_pools`](benchmark_iterations/2026-04-26_iteration_2_prepare_pools/) |
| 3 | `pass` | [`2026-04-27_iteration_3_combo_planning_contexts`](benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/) |
| 4 | `pass` | [`4.1 no-risk boundary`](benchmark_iterations/2026-05-01_iteration_4_1_no_risk_boundary/), [`4.2 exact/self-check`](benchmark_iterations/2026-05-01_iteration_4_2_exact_scoring_self_check/), [`4.3 heap corrective`](benchmark_iterations/2026-05-01_iteration_4_3_heap_update_corrective/), [`4.4 proxy fill`](benchmark_iterations/2026-05-01_iteration_4_4_top_result_proxy_fill/), [`4.5 shape/hash parity`](benchmark_iterations/2026-05-01_iteration_4_5_result_shape_hash_parity/), [`4.6 accounting`](benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/), [`4.7 memory cleanup`](benchmark_iterations/2026-05-01_iteration_4_7_memory_cleanup/) |
| 5 | `pass` | [`5 TP/SL hit-times loading/grid validation`](benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/) |
| 6 | `pass` | [`6 TP/SL exact scoring/full metrics`](benchmark_iterations/2026-05-01_iteration_6_tp_sl_exact_scoring_full_metrics/) |
| 7 | `pass` | [`7 job orchestration/persistence`](benchmark_iterations/2026-05-01_iteration_7_job_orchestration_persistence/) |
| 8 | `pass` | [`8 execution/sizing completion`](benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/) |
| 9 | `pass` | [`9 lazy trades detail`](benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/) |
| 10 | `pass` | Browser QA evidence: [`10 UI integration`](benchmark_iterations/2026-05-02_iteration_10_ui_integration/browser_qa_summary.md) |

Итерация 6 принята с явным manual stage override: пользователь допустил
нарушения `0.9` ratio для микросекундных стадий как неразличимые на уровне
service latency. Raw ratios сохранены в benchmark evidence. Iteration 10
закрыта через browser-visible QA, а не через compute benchmark: это UI
integration gate поверх уже принятого public API/lazy trades runtime.

### Итерация 0: документы и benchmark harness

- финализировать этот architecture document относительно canonical notebook prototype;
- пометить conflicting roadmap/runtime docs как superseded или compatibility-only там, где нужно;
- сохранить `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`
  как canonical algorithm source;
- сохранить `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
  как canonical target values;
- определить Mac Studio benchmark command contract для будущих iteration records.

Критерии выхода:

- benchmark folder существует и содержит JSON + Markdown evidence;
- canonical notebook и benchmark evidence названы;
- canonical timer names и service overhead stages зафиксированы;
- source-of-truth status явно указан в backtest docs;
- contracts `variant_key`/`variant_hash` и progress mapping задокументированы.

Гейт бенчмарка:

- index `docs/architecture/README.md` актуален;
- paths canonical benchmark evidence читаются;
- future benchmark record template содержит commit, request hash, artifact hashes,
  stage timers, CPU/RSS metrics и correctness result fields.

### Итерация 1: нормализация request и artifact context

- реализовать strict artifact context resolver;
- реализовать request normalization для coordinates/timeframe/period/indicator grids;
- реализовать execution defaults и validation для `direction_mode`, `sizing`,
  `profit_lock`, fees, slippage и `close_on_end`;
- реализовать canonical `request_hash` и result-affecting config hash;
- реализовать cost estimate для rows, combinations и TP/SL cells;
- expose `POST /backtests/preflight` и `GET /backtests/runtime-defaults`
  как API shell.

Гейт бенчмарка:

- request normalization и preflight smoke benchmark;
- timing artifact current/root resolution;
- parity check, что request hash совпадает с canonical fixture hash, где применимо;
- failure evidence для invalid indicator/source/window и request-too-expensive.

Принятый benchmark record:
[`2026-04-26_iteration_1_request_normalization_artifact_context`](benchmark_iterations/2026-04-26_iteration_1_request_normalization_artifact_context/).

### Итерация 2: artifact arrays и `prepare_pools`

- реализовать mmap loaders для `prices`, `signals`, `mappings`;
- реализовать `[start, end)` slicing по 15m `open_time`;
- реализовать derivation 15m return intervals;
- реализовать 15m-to-1m execution mapping для no-risk;
- реализовать signal row extraction и source/window row mapping;
- реализовать row prefilter через `fused_row_prefilter_stats`;
- реализовать compressed signal segments через `build_signal_segments`;
- expose timings `artifact_context_resolve`, `artifact_array_open`,
  `request_slice_prepare`, `prepare_pools_core`, `prepare_pools_total`.

Гейт бенчмарка:

- `prepare_pools_core` vs canonical notebook prepare_pools target для arity
  1..7 fixture;
- service overhead измеряется отдельно:
  `artifact_context_resolve`, `artifact_array_open`,
  `request_slice_prepare`, `prepare_pools_total`;
- compatibility subsegments остаются доступными для historical evidence:
  `artifact_manifest_load`, `artifact_array_mmap_load`, `time_range_slice`,
  `signal_row_selection`, `row_prefilter`, `segment_build`;
- row metadata/order hash равен notebook fixture;
- stage record записан до перехода к combo planning.

Принятый benchmark record:
[`2026-04-26_iteration_2_prepare_pools`](benchmark_iterations/2026-04-26_iteration_2_prepare_pools/).
В Итерации 2 принятая stage — `prepare_pools_core`; исторический strict-total
fail по `prepare_pools_total` сохранен в evidence как `stage_boundary_mismatch`.

### Итерация 3: combo planning contexts

- реализовать backend registry для `event_segments_2_no_risk`,
  `event_segments_n_no_risk`, `streaming_2_no_risk` and
  `event_segments_n_tp_sl_15m_grid`;
- реализовать `build_exact_context`;
- реализовать `build_proxy_context`;
- реализовать deterministic `combo_iteration`;
- реализовать pass-through и active `proxy_filter`.

Гейт бенчмарка:

- `build_exact_context`;
- `build_proxy_context`;
- `combo_iteration`;
- `proxy_filter`;
- deterministic combo ordering и candidate-count evidence;
- active и inactive proxy-filter fixture evidence;
- stage record записан до exact scoring.

Принятый benchmark record:
[`2026-04-27_iteration_3_combo_planning_contexts`](benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/).

### Итерация 4: no-risk exact scoring и notebook-compatible top-K

Цель iteration: перенести no-risk compiled exact scoring и top-K output contract
из canonical notebook без service-only assembly внутри measured stages.

Известная неуспешная попытка:

- [`2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk`](benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/)
  не принимается как успешное evidence;
- подтверждено как корректное: semantic metrics parity `14 / 14`, proxy
  metadata parity `14 / 14`, `exact_scoring` latency `14 / 14`;
- подтвержденные разрывы: object-heavy `heap_update`, отсутствующий arity-2
  `proxy_for_two_rows` fast path в `top_result_proxy_fill`, strict hash drift
  arity 1/2 и неверная benchmark boundary для `total_without_warmup`.

#### Итерация 4.1: no-risk execution context и DTO boundary

Реализация:

- создать минимальные internal DTOs/config для no-risk scoring result,
  telemetry и price/execution context;
- result object хранит только compact `top_results`, scalar telemetry и
  self-check summary;
- result object не хранит references на prepared pools, segment stacks, score
  arrays, combo chunks или hit-time arrays;
- `request.top_n` сохраняется в telemetry как public input, но не управляет
  canonical heap capacity.

Измерение:

- smoke benchmark записывает `request.top_n`, `benchmark_top_k`,
  `top_results_count`, heap capacity и отсутствие heavy references в result;
- memory cleanup evidence фиксирует, что после удаления локальных heavy references
  worker не удерживает per-job arrays через result DTO.

#### Итерация 4.2: exact scoring kernels и self-check

Реализация:

- реализовать `event_segments_2_no_risk`;
- реализовать `event_segments_n_no_risk` для arity 1..10;
- реализовать `streaming_2_no_risk` как fallback/parity comparator, а не default
  production path;
- реализовать no-risk self-check против generic slow reference;
- реализовать full no-risk summary metric set:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`.

Измерение:

- `service_warmup` / `sample_warmup` отдельно от measured runtime;
- `self_check` отдельно от `exact_scoring`;
- `exact_scoring` сравнивается per tuple `{arity, direction_mode, backend}` с
  canonical target для arity 1..7;
- arity 8..10 покрываются service-level correctness smoke на малых row pools, но
  не блокируют v1 acceptance, если arity 1..7 проходят 90% target.

#### Итерация 4.3: low-allocation `heap_update`

Реализация:

- canonical measured heap capacity: `benchmark_top_k = 5`;
- sample warmup heap capacity: `top_k = 1`;
- public `request.top_n = 100` не участвует в measured heap work;
- default benchmark ranking `total_return_pct desc` идет через прямое чтение
  score array, без generic ranking dispatch в hot loop;
- для каждого candidate сначала строится только rank score и deterministic
  heap key `(rank_score, original_row_ids)`;
- full summary `item`, `_local_indices`, `_proxy_pending` и per-indicator
  metadata materialize только если candidate реально входит в heap или заменяет
  worst heap row;
- metadata conversion в mapping не выполняется для rejected candidates.

Измерение:

- `heap_update` timer включает только candidate top-K maintenance и materialized
  compact items для retained heap rows;
- `heap_update` не включает public `variant_key`, `variant_hash`, API DTO,
  persisted row construction или DB/object storage writes;
- acceptance требует `target_heap_update / service_heap_update >= 0.9` для всех
  arity 1..7 x direction modes.

#### Итерация 4.4: notebook-compatible `top_result_proxy_fill`

Реализация:

- вход stage — heap размера `benchmark_top_k`;
- stage сортирует heap descending по heap key;
- proxy recompute выполняется только для final top rows с
  `_proxy_pending = true`;
- `proxy_for_indicator_rows(...)` повторяет current notebook dispatch:
  `len(eval_rows) == 2` вызывает compiled `proxy_for_two_rows(...)`, остальные
  arity используют generic consensus path;
- stage удаляет `_local_indices` и `_proxy_pending`;
- stage возвращает `top_results` длиной не больше `benchmark_top_k`.

Измерение:

- `top_result_proxy_fill` сравнивается per tuple `{arity, direction_mode}`;
- arity-2 pass обязателен отдельно, потому что именно он имеет special fast path
  в notebook;
- stage не включает lazy trades, exact scoring, identity/hash assembly,
  persisted rows или product `top_n = 100` proxy fill.

#### Итерация 4.5: canonical result shape и hash/parity

Реализация:

- output shape должен совпадать с notebook-compatible top results;
- все floats, ints, ordering, metric names и metadata shape должны проходить
  tolerance parity;
- strict result hash должен использовать canonical serialization/float
  normalization, совместимую с notebook evidence;
- float representation drift может быть зафиксирован как non-semantic finding
  только временно и не должен скрывать metric или top-row identity mismatch.

Измерение:

- semantic metric parity для всех top rows;
- proxy metadata parity для всех top rows;
- strict result hash parity или явно documented waiver с причиной;
- top row identity/order parity по deterministic heap key.

#### Итерация 4.6: benchmark runner accounting

Реализация:

- runner записывает `request.top_n = 100`, `benchmark_top_k = 5`,
  `sample_warmup_top_k = 1`, `top_results_count = 5` и heap capacity;
- runner строит `total_without_warmup` только из notebook-compatible measured
  stages:
  `prepare_pools_core + build_exact_context + build_proxy_context +
  combo_iteration + proxy_filter + self_check + exact_scoring + heap_update +
  top_result_proxy_fill`;
- runner отдельно записывает `service_total_without_warmup`, куда входят
  service-only overhead stages до доступности terminal result.

Измерение:

- canonical target сравнивается только с `total_without_warmup`;
- `service_total_without_warmup`, `artifact_context_resolve`,
  `artifact_array_open`, `request_slice_prepare`, `prepare_pools_total`,
  `top_result_assembly` и `persist_top_n_io` записываются
  как service telemetry / service-specific budget, но не участвуют в 90%
  notebook timer comparison;
- benchmark summary должен явно показывать оба ratio, чтобы следующие agents не
  сравнивали разные процессы.

#### Итерация 4.7: memory cleanup после no-risk run

Реализация:

- scoring service использует bounded job scope и `try/finally` cleanup boundary;
- после доступности terminal result освобождаются strong references на prepared
  pools, combo planning context, score arrays, segment stacks, self-check
  reference objects, combo buffers и temporary heaps больше final top rows;
- artifact mmap/cache handles могут оставаться только в bounded artifact runtime
  cache, а не в per-job result;
- если retained RSS после cleanup превышает configured threshold, worker должен
  recycle до следующего heavy job.

Проверка:

- cleanup duration как service hygiene metric, а не canonical benchmark stage;
- `rss_before`, `rss_peak`, `rss_after_cleanup`, `retained_rss_delta`;
- repeated-run smoke минимум 3 раза подряд на одном worker lifecycle;
- условие pass: нет монотонного retained RSS growth или доказан worker recycle перед
  следующим heavy job.

Не входит в Iteration 4 acceptance:

- persisted top-N rows;
- public `variant_key`;
- `variant_hash` / `indicator_variant_hash`;
- API DTO/read-model assembly;
- product `top_n = 100` performance gate.

Эти задачи принадлежат `top_result_assembly` / `persist_top_n_io` и закрываются
в Iteration 7. Если они прототипируются раньше, их timings должны быть
service-only и не должны попадать в `heap_update`, `top_result_proxy_fill` или
`total_without_warmup` notebook comparison.

Гейт бенчмарка:

- `service_warmup`;
- `self_check`;
- `exact_scoring` для no-risk;
- `heap_update` при `benchmark_top_k = 5`;
- `top_result_proxy_fill` при `benchmark_top_k = 5`;
- `total_without_warmup` по notebook-compatible formula из Итерации 4.6;
- `service_total_without_warmup` только как service-specific telemetry;
- memory cleanup smoke evidence без добавления нового canonical benchmark stage;
- arity 1..7 target comparison против current canonical evidence;
- evidence fields: `request.top_n = 100`, `benchmark_top_k = 5`,
  `top_results_count = 5`, heap capacity, exact backend display name и
  implementation id;
- result hash/parity для notebook-compatible top results;
- service-level correctness smoke для arity 8..10 на малых row pools;
- evidence, что public/storage identity work не попадает внутрь measured
  notebook-compatible stages.

Принятые benchmark records:

- [`4.1 no-risk boundary`](benchmark_iterations/2026-05-01_iteration_4_1_no_risk_boundary/)
  принят как service hygiene smoke для compact DTO/result boundary и отсутствия
  retained heavy references.
- [`4.2 exact/self-check`](benchmark_iterations/2026-05-01_iteration_4_2_exact_scoring_self_check/)
  принят для no-risk `self_check` и `exact_scoring`: `14 / 14` canonical
  no-risk rows прошли с artifact policy `historical_prefix_compatible`.
- [`4.3 heap corrective`](benchmark_iterations/2026-05-01_iteration_4_3_heap_update_corrective/)
  принят для low-allocation `heap_update`: `14 / 14` rows прошли и top identity
  совпал. Более ранний failed 4.3 evidence удален из active benchmark tree
  после этого corrective pass.
- [`4.4 proxy fill`](benchmark_iterations/2026-05-01_iteration_4_4_top_result_proxy_fill/)
  принят для notebook-compatible `top_result_proxy_fill`: `14 / 14` rows прошли
  с top identity и proxy metadata parity.
- [`4.5 shape/hash parity`](benchmark_iterations/2026-05-01_iteration_4_5_result_shape_hash_parity/)
  принят для canonical result shape, ordering, semantic metric parity, proxy
  metadata parity и strict result hash: `14 / 14`.
- [`4.6 benchmark runner accounting`](benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/)
  принят для local и Mac Studio validation canonical stage aliases,
  accounting `total_without_warmup` и отделения
  `service_total_without_warmup`.
- [`4.7 memory cleanup`](benchmark_iterations/2026-05-01_iteration_4_7_memory_cleanup/)
  принят как service hygiene evidence для compact results и non-monotonic
  retained RSS growth на repeated no-risk runs.
- [`5 TP/SL hit-times loading/grid validation`](benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/)
  принят для `load_hit_times` и `tp_sl_grid_validation`: `14 / 14` canonical
  risk-on rows прошли с target grid `2.0..25.0` step `0.5`, deterministic
  `backtest.tp_sl_grid_not_covered` failure evidence и artifact policy
  `historical_prefix_compatible`.

### Итерация 5: загрузка и validation TP/SL grid

Текущий статус: `pass`;
[`benchmark record`](benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/)
принят на Mac Studio.

- валидировать request TP/SL subset against artifact grid;
- реализовать `load_hit_times` и `tp_sl_grid_validation`;
- реализовать hit-times manifest hashing;
- реализовать requested subset materialization для long/short TP/SL arrays;
- реализовать deterministic 422 для grid-not-covered failure;
- гарантировать, что requested hit-time subset живет только в bounded job scope и
  освобождается через cleanup boundary, если scoring дальше падает.

Гейт бенчмарка:

- `load_hit_times`;
- `tp_sl_grid_validation`;
- request grid coverage success и failure evidence;
- target grid `2.0..25.0` step `0.5` evidence;
- stage record записан до TP/SL exact scoring;
- cleanup evidence для failed validation / failed load path без retained heavy
  arrays.

### Итерация 6: TP/SL exact scoring и full metrics

Текущий статус: `pass` с manual stage override в
[`2026-05-01_iteration_6_tp_sl_exact_scoring_full_metrics`](benchmark_iterations/2026-05-01_iteration_6_tp_sl_exact_scoring_full_metrics/).
Raw benchmark ratios сохранены в evidence; override принят пользователем для
микросекундных stage-ratio misses.

- реализовать `event_segments_n_tp_sl_15m_grid` для arity 1..10;
- реализовать TP/SL self-check против slow direct reference;
- реализовать TP/SL full summary metric set:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`,
  `best_tp_pct`, `best_sl_pct`;
- реализовать notebook-compatible `heap_update` для risk-on top rows с
  `benchmark_top_k = 5`;
- применить те же low-allocation правила `heap_update`, что в Итерации 4.3:
  сначала heap key и admission check, затем materialization только для retained
  heap rows;
- реализовать service-only `tp_sl_full_metrics_second_pass`, если full metric set
  считается после выбора best TP/SL cell.

Не входит в Iteration 6 acceptance:

- persisted risk-on top-N rows;
- public/storage identity assembly;
- API read-model shape.

Эти задачи закрываются в Iteration 7 как service-only
`top_result_assembly` / `persist_top_n_io`.

Гейт бенчмарка:

- `build_exact_context`;
- `combo_iteration`;
- `self_check`;
- `exact_scoring` / `tp_sl_exact_scoring` для TP/SL grid vs canonical target;
- `heap_update` при `benchmark_top_k = 5`;
- risk-on `total_without_warmup` сравнивается только как notebook-compatible sum:
  `load_hit_times + tp_sl_grid_validation + prepare_pools_core +
  build_exact_context + build_proxy_context + combo_iteration + proxy_filter +
  self_check + exact_scoring + heap_update`;
- `tp_sl_full_metrics_second_pass` CPU/RSS/latency evidence как service-only
  budget, если этот step уже реализован;
- `service_total_without_warmup` и memory cleanup evidence записываются отдельно и не
  сравниваются с canonical notebook target;
- arity 1..7 target comparison против current canonical evidence;
- service-level correctness smoke для arity 8..10 на малых row pools;
- full metric-set correctness evidence для selected best TP/SL cell;
- repeated-run memory cleanup smoke для hit-time subsets, TP/SL diff buffers,
  score arrays и selected top rows.

### Итерация 7: job orchestration и persistence

Текущий статус: `pass`;
[`benchmark record`](benchmark_iterations/2026-05-01_iteration_7_job_orchestration_persistence/)
принят на Mac Studio для public/storage variant identity mapping,
summary-only top-N rows, `top_result_assembly`, `persist_top_n_io` и
`service_total_without_warmup`.

- реализовать `top_result_assembly` для no-risk и TP/SL top rows:
  public `variant_key`, stable `variant_hash`, optional
  `indicator_variant_hash`, canonical variant params и API/read-model DTOs;
- безопасно map public `variant_key` to storage `variant_hash`;
- реализовать job create/status/top/list/cancel contracts;
- реализовать idempotency и request guardrails;
- persist canonical request snapshot, artifact metadata и top-N rows;
- expose progress через canonical pipeline stage names;
- реализовать ownership/authz checks;
- реализовать production cleanup boundary в worker orchestration: cleanup должен
  выполняться после terminal persistence / доступности result и до освобождения
  worker slot для следующего heavy job.

Гейт бенчмарка:

- `top_result_assembly`;
- `persist_top_n_io`;
- end-to-end job benchmark для no-risk и TP/SL с current canonical fixtures;
- persisted top-N summary hash/parity evidence;
- API contract tests для create/status/list/top/cancel/defaults/preflight;
- idempotency replay/conflict evidence;
- authz/ownership failure evidence;
- repeated end-to-end cleanup evidence: после чтения результатов job worker не
  удерживает heavy compute objects, а превышение retained RSS threshold приводит
  к worker recycle.

### Итерация 8: завершение execution/sizing

- реализовать все public sizing modes в service compiled path:
  `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- реализовать `profit_lock` parity для каждого sizing mode;
- реализовать `close_on_end = false`;
- verify, что no-risk и TP/SL semantics остаются stable across execution settings.

Статус: `pass`, accepted Mac Studio evidence:
[`2026-05-02_iteration_8_execution_sizing_completion`](benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/).

Гейт бенчмарка и корректности:

- sizing smoke vs canonical notebook evidence;
- service compiled parity для equity-percent modes, которые являются reference-only
  в current notebook evidence;
- service-level correctness tests для `close_on_end = true/false`;
- regression check, что canonical arity/risk/direction benchmark остается внутри
  target envelope.

### Итерация 9: lazy trades detail

- реализовать variant lookup;
- реализовать lazy trades recompute;
- реализовать 48h cache;
- возвращать chart-ready payload.

Статус: `pass`, accepted Mac Studio evidence:
[`2026-05-02_iteration_9_lazy_trades_detail`](benchmark_iterations/2026-05-02_iteration_9_lazy_trades_detail/).

Гейт бенчмарка:

- `lazy_trades_compute`, `lazy_trades_cache_hit`.

Проверка:

- cache miss и cache hit;
- cache failure with successful recompute;
- ownership failure;
- variant key/hash mismatch failure.

### Итерация 10: UI integration

- использовать тот же public API;
- показывать job progress/top N;
- реализовать `show trades`;
- render trades на candle chart.

Статус: `pass`, локальная browser QA evidence:
[`2026-05-02_iteration_10_ui_integration`](benchmark_iterations/2026-05-02_iteration_10_ui_integration/browser_qa_summary.md).

Проверка:

- browser-visible QA через runtime browser surface.

## Как проверить

Static checks после docs updates:

```bash
python -m tools.docs.generate_docs_index --check
```

Проверки этапа реализации будут добавляться по итерациям. Benchmark checks должны запускаться на `Mac Studio`, а не на локальных non-production-equivalent hosts.

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
  пересчитывается только для final top rows по их selected best cell; после
  Iteration 7 эти rows становятся persisted top-N variants;
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
file/object cache на 48h достаточен: cache miss запускает deterministic
materialization для одного variant через runner. В multi-host deployment запрос
`show trades` может
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
