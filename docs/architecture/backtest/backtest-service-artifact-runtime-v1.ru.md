# Бектест-сервис на артефактах v1

Документ фиксирует целевую архитектуру бектест-сервиса на артефактах, его публичный API, поток выполнения задач, ленивую детализацию сделок и гейт бенчмарка по итерациям.

Русская версия переводит человекочитаемое описание. Идентификаторы API routes,
config keys, metric names, timer names, backend ids, file paths и значения
контрактных полей сохраняются в исходном написании, чтобы не разорвать связь с
кодом, notebook и benchmark evidence.

## Статус

Каноническая целевая архитектура для планирования реализации. Runtime-сервис
еще не реализован; notebook и benchmark evidence ниже определяют production-
прототип, которому сервис обязан соответствовать.

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
- пересчитывает trades только для одного варианта;
- возвращает trades + chart overlay payload;
- может сохранить cache на 1-2 дня.

Рекомендация по cache:

- metadata в Postgres;
- payload в локальном object/file cache под `/opt/roehub/state/backtest/trades_cache`;
- TTL по умолчанию: 48h;
- cache key включает `job_id`, `variant_key`, `variant_hash`, `request_hash`, `engine_params_hash`, `artifact_manifest_hash`.

Postgres-only JSONB допустим для малых payloads, но v1 должен избегать неограниченного раздувания основной БД.

Топология cache:

- v1 исходит из single API/worker host или sticky local cache semantics;
- cache miss является нормальным состоянием и должен запускать deterministic recompute для одного варианта;
- отказ local cache не должен ломать response, если recompute успешен;
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

Рассчитывает или возвращает cached lazy trades для одного варианта.

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

### Измеряемая стадия бенчмарка: `prepare_pools`

Методы notebook:

- `extract_signal_rows`;
- `prefilter_indicator_rows`;
- `fused_row_prefilter_stats`;
- `topk_fraction_idx`;
- `build_signal_segments`;
- `fill_signal_segments_i8`;
- `prepare_indicator_pool`;
- `prepare_indicator_pools`.

Эта stage:

- resolves artifact root через trusted configuration/current pointer;
- читает slot `manifest.yaml` и requested indicator manifests;
- загружает `.npy` arrays через `np.load(..., mmap_mode="r")`;
- загружает `prices/<tf>`, `prices/1m`, `mappings/<tf>` и requested
  `signals/<tf>/<indicator_id>/signals.i8.npy`;
- slices 15m bars по `[start, end)` через `open_time`;
- derives 15m return intervals из close prices;
- derives 15m-to-1m execution mapping для no-risk mode:
  signal на 15m bar `t` входит на open следующего 15m bar, mapped to 1m;
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
- persisted summary должен включать тот же full metric set, что и no-risk
  (`total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`) плюс
  `best_tp_pct` и `best_sl_pct` для выбранной best cell.

### Измеряемая стадия бенчмарка: `heap_update`

Notebook использует Python `heapq`, чтобы держать top K.

Эта stage:

- ranks по selected metric, default `total_return_pct desc`;
- строит deterministic heap key из score и original row ids;
- держит только top N в памяти;
- добавляет compact per-indicator metadata;
- производит deterministic ordering для persisted top-N rows.

### Измеряемая стадия бенчмарка: `top_result_proxy_fill`

Эта stage не является lazy trades.

Она выполняется только когда final top rows не получили proxy metadata из active
combo prefilter. Она пересчитывает `confirm_count` и `proxy_score` для top rows
через `proxy_for_indicator_rows`. Она не должна маппиться на UI/API endpoint
`show trades`.

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
| `total_without_warmup` | yes | User-facing measured runtime после warmup. |
| `persist_top_n_io` | service only | DB write overhead; не часть notebook baseline. |

### Lazy trades

- recompute exact trades для одного `variant_key`;
- cache result на 48h;
- возвращает chart-ready payload.

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
| `prepare_pools` | `stage_a` | Artifact load, slicing, row selection, prefilter и segment build. |
| `build_exact_context` | `stage_a` / `stage_b` | Segment stack для exact kernels. |
| `build_proxy_context` | `stage_a` | Optional combo proxy context. |
| `combo_iteration` | `stage_a` / `stage_b` | Cartesian chunk planning. |
| `proxy_filter` | `stage_a` / `stage_b` | Optional combo pruning. |
| `self_check` | `stage_a` / `stage_b` | Benchmark/test parity check. |
| `exact_scoring` with `risk.mode = "none"` | `stage_a` | Нет Stage B risk grid. |
| `exact_scoring` / `tp_sl_exact_scoring` with `risk.mode = "tp_sl_grid"` | `stage_b` | TP/SL scoring на hit-times. |
| `heap_update` | `finalizing` | Ranking и top-N assembly. |
| `top_result_proxy_fill` | `finalizing` | Top-row proxy metadata, не lazy trades. |
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

Идентичность канонического benchmark:

- host: `macstudio`;
- period: `[2020-01-11T20:08:00Z, 2026-04-11T20:08:00Z)`;
- rows per indicator: `6`;
- warmup rows per indicator: `2`;
- canonical production acceptance arities: `1..7`;
- target public request arities: `1..10` для no-risk и TP/SL; arity 8..10
  разрешены только в пределах cost guardrails и не входят в обязательный 90%
  acceptance benchmark для завершения v1;
- direction modes: `long_only`, `long_short_reversal`;
- risk modes: `none`, `tp_sl_grid`;
- TP/SL grid: `2.0..25.0` inclusive, step `0.5`;
- TP/SL cells per combo: `2209`;
- runs: `28` (`7 arities x 2 risk modes x 2 direction modes`);
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
- canonical `request_hash` и result-affecting config hash;
- service warmup metrics;
- canonical notebook timer metrics без warmup;
- service-only overhead metrics;
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
- user-facing runtime benchmark измеряется после warmup;
- warmup и warm runtime оба должны оставаться в accepted 90% envelope для
  соответствующего segment.

Обязательные benchmark segments:

1. `service_warmup`
2. `numba_warmup`
3. `sample_warmup`
4. `total_without_warmup`
5. `load_hit_times` для `risk.mode = "tp_sl_grid"`
6. `tp_sl_grid_validation` для `risk.mode = "tp_sl_grid"`
7. `prepare_pools`
8. `build_exact_context`
9. `build_proxy_context`
10. `combo_iteration`
11. `proxy_filter`
12. `self_check`
13. `exact_scoring`
14. `tp_sl_exact_scoring` для `risk.mode = "tp_sl_grid"`
15. `heap_update`
16. `top_result_proxy_fill`
17. `persist_top_n_io`
18. `lazy_trades_compute`
19. `lazy_trades_cache_hit`

Сравнение acceptance:

- stages 1-16 сравниваются с notebook-compatible target values по tuple
  `{arity, risk_mode, direction_mode, backend}`;
- для arity 1..7 target source:
  `2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
- arity 8..10 не блокируют v1 completion, если arity 1..7 проходят 90%
  threshold; перед увеличением production budgets для широких arity 8..10
  workloads нужно создать follow-up Mac Studio benchmark iteration;
- `persist_top_n_io`, `lazy_trades_compute` и
  `lazy_trades_cache_hit` используют service-specific absolute budgets плюс regression
  comparison после появления собственного baseline;
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
reference-only. Service implementation является first compiled parity point для
`fixed_equity_pct`, `fixed_equity_pct_min_quote` и
`fixed_equity_pct_max_quote` и должна записать service-level parity evidence для
этих modes до того, как v1 можно считать functionally complete.

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

## План внедрения

Правило для всех итераций:

- каждая iteration имеет explicit benchmark/evidence gate;
- benchmark records записываются до того, как iteration помечается complete;
- более поздняя stage может прототипироваться локально, но она не должна считаться
  accepted, пока все предыдущие stage gates не пройдены.

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

### Итерация 2: artifact arrays и `prepare_pools`

- реализовать mmap loaders для `prices`, `signals`, `mappings`;
- реализовать `[start, end)` slicing по 15m `open_time`;
- реализовать derivation 15m return intervals;
- реализовать 15m-to-1m execution mapping для no-risk;
- реализовать signal row extraction и source/window row mapping;
- реализовать row prefilter через `fused_row_prefilter_stats`;
- реализовать compressed signal segments через `build_signal_segments`;
- expose timing `prepare_pools`.

Гейт бенчмарка:

- `prepare_pools` vs canonical notebook target для arity 1..7 fixture;
- optional subsegments: `artifact_manifest_load`, `artifact_array_mmap_load`,
  `time_range_slice`, `signal_row_selection`, `row_prefilter`,
  `segment_build`;
- row metadata/order hash равен notebook fixture;
- stage record записан до перехода к combo planning.

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

### Итерация 4: no-risk exact scoring и top-N

- реализовать `event_segments_2_no_risk`;
- реализовать `event_segments_n_no_risk` для arity 1..10;
- реализовать `streaming_2_no_risk` fallback/parity comparator;
- реализовать no-risk self-check против generic slow reference;
- реализовать full no-risk summary metric set;
- реализовать `heap_update` и `top_result_proxy_fill`;
- реализовать persisted top-N summary для no-risk;
- безопасно map public `variant_key` to storage `variant_hash`.

Гейт бенчмарка:

- `service_warmup`;
- `self_check`;
- `exact_scoring` для no-risk;
- `heap_update`;
- `top_result_proxy_fill`;
- arity 1..7 target comparison против current canonical evidence;
- service-level correctness smoke для arity 8..10 на малых row pools;
- persisted top-N summary hash/parity evidence.

### Итерация 5: загрузка и validation TP/SL grid

- валидировать request TP/SL subset against artifact grid;
- реализовать `load_hit_times` и `tp_sl_grid_validation`;
- реализовать hit-times manifest hashing;
- реализовать requested subset materialization для long/short TP/SL arrays;
- реализовать deterministic 422 для grid-not-covered failure.

Гейт бенчмарка:

- `load_hit_times`;
- `tp_sl_grid_validation`;
- request grid coverage success и failure evidence;
- target grid `2.0..25.0` step `0.5` evidence;
- stage record записан до TP/SL exact scoring.

### Итерация 6: TP/SL exact scoring и full metrics

- реализовать `event_segments_n_tp_sl_15m_grid` для arity 1..10;
- реализовать TP/SL self-check против slow direct reference;
- реализовать TP/SL full persisted summary metric set:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
  `profit_factor`, `trade_count`, `sharpe_trades`, `win_rate_pct`,
  `avg_trade_ret_pct`, `avg_trade_exec_bars`, `exposure_pct`,
  `best_tp_pct`, `best_sl_pct`;
- persist risk-on top-N summary.

Гейт бенчмарка:

- `build_exact_context`;
- `combo_iteration`;
- `self_check`;
- `exact_scoring` / `tp_sl_exact_scoring` для TP/SL grid vs canonical target;
- `heap_update`;
- arity 1..7 target comparison против current canonical evidence;
- service-level correctness smoke для arity 8..10 на малых row pools;
- full metric-set correctness evidence для selected best TP/SL cell.

### Итерация 7: job orchestration и persistence

- реализовать job create/status/top/list/cancel contracts;
- реализовать idempotency и request guardrails;
- persist canonical request snapshot, artifact metadata и top-N rows;
- expose progress через canonical pipeline stage names;
- реализовать ownership/authz checks.

Гейт бенчмарка:

- `persist_top_n_io`;
- end-to-end job benchmark для no-risk и TP/SL с current canonical fixtures;
- API contract tests для create/status/list/top/cancel/defaults/preflight;
- idempotency replay/conflict evidence;
- authz/ownership failure evidence.

### Итерация 8: завершение execution/sizing

- реализовать все public sizing modes в service compiled path:
  `all_in`, `fixed_quote`, `fixed_equity_pct`,
  `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote`;
- реализовать `profit_lock` parity для каждого sizing mode;
- реализовать `close_on_end = false`;
- verify, что no-risk и TP/SL semantics остаются stable across execution settings.

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
