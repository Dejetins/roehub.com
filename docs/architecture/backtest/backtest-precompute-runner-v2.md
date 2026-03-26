# Backtest Precompute Runner V2 (R2-03 / R2-04 / R3-01)

Статус: `Milestone R2 / EPIC R2-03 + R2-04`, `Milestone R3 / EPIC R3-01`

Документ фиксирует контракт precompute/publish слоя, который:

- строит inactive slot в `artifacts/backtest/v2`;
- пишет strict manifests для root / signals / hit_times;
- выполняет fail-fast validation до switch `current.yaml`;
- оставляет runtime только fixed metadata reads без schema inference и без hash recomputation.

Основные документы:

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

## Config Inputs (R2-04)

Precompute/publish слой читает strict `configs/<env>/backtest_artifacts.yaml` contract.

Из него берутся:

- `artifact_root` для path builder / loader wiring;
- `validation_plan` для config-driven `ArtifactSlotValidationSpecV2`;
- `hit_times_grid` как source-of-truth TP/SL levels contract;
- `slot_policy`, `publish_schedule`, `lookback_policy`, `validation_budgets` как fail-fast
  validated pipeline settings.

R2-04 intentionally keeps these settings отдельно от `configs/<env>/backtest.yaml`, чтобы
runtime request defaults и artifact pipeline knobs не смешивались в одном контракте.

## Область ответственности

Precompute runner v2 обязан:

- писать файлы только в inactive slot;
- использовать deterministic paths из R2-01;
- писать root `manifest.yaml`;
- писать per-indicator `signals/<tf>/<indicator_id>/manifest.yaml`;
- писать `hit_times/1m/manifest.yaml`;
- указывать в manifests fixed runtime metadata:
  - `dtype`
  - `shape`
  - `axis_order`
  - `sha256`
  - `provenance`
  - `slot_generation`
  - `timeline` coverage
- завершать publish только после whole-slot validation.

### R3-01 canonical `1m` export

На этапе R3-01 precompute runner получает отдельную обязанность:

- материализовать canonical source-of-truth export только для `prices/1m/*` в inactive slot;
- писать:
  - `prices/1m/open_time.i64.npy`
  - `prices/1m/close_time.i64.npy`
  - `prices/1m/ohlcv.f32.npy`
- использовать source table `market_data.canonical_candles_1m` через existing
  `CanonicalCandleReader` contract;
- поддерживать deterministic tail update по
  `backtest_artifacts.lookback_policy.price_tail_bars_1m`;
- никогда не мутировать active slot in place.

Tail update semantics для R3-01:

- если inactive slot ещё не содержит valid `prices/1m`, выполняется full build по заданному
  `TimeRange [start, end)`;
- если `prices/1m` уже существует в inactive slot, runner переиспользует prefix внутри requested
  range и reread'ит только tail overlap длиной `price_tail_bars_1m`;
- merge policy фиксирована как `prefix + reread_tail`, без best-effort dedup/coercion;
- identical source candles + identical config/request должны давать byte-stable `.npy` и
  `manifest.yaml`.

Precompute runner v2 не должен:

- мутировать active slot in place;
- discover'ить содержимое через directory scanning;
- делать dynamic schema discovery;
- переносить expensive hash validation в runtime hot path.

## Manifest outputs

### Root `manifest.yaml`

Root manifest обязан фиксировать:

- `identity` (`exchange`, `market_type`, `symbol`);
- `slot`, `slot_generation`, `asof_date`;
- `prices[]` с metadata для `open_time`, `close_time`, `ohlcv`;
- `mappings[]` с metadata для `bar_open_1m_idx`, `bar_close_1m_idx`;
- `signals.manifests[]` с `manifest_path` и `manifest_sha256`;
- `hit_times.manifest_path` и `manifest_sha256`;
- `signal_encoding`:
  - `dtype: int8`
  - `axis_order: [variant, time]`
  - `value_set: [-1, 0, 1]`
- `provenance`.

R3-01 placeholder strategy до materialization следующих stage:

- `prices[]` содержит свежий strict section для `1m` и может сохранять уже существующие non-`1m`
  sections, если они были подготовлены в том же inactive slot более поздним stage;
- `mappings[]` может оставаться пустым до R3-03;
- `signals` фиксируется как explicit empty catalog
  (`supported_timeframes=[]`, `supported_indicator_ids=[]`, `manifests=[]`) до R4;
- `hit_times` обязан оставаться explicit fixed-path reference
  `hit_times/1m/manifest.yaml`, но до R5 допускается placeholder
  `manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"`;
- `signal_encoding` остаётся fixed even when `signals.manifests` is empty.

### Per-indicator signal manifest

Каждый `signals/<tf>/<indicator_id>/manifest.yaml` обязан фиксировать:

- `indicator_id`, `timeframe`;
- `signals.path`, `dtype`, `shape`, `axis_order`, `sha256`;
- `rows_count`;
- `timeline` coverage;
- `signal_value_set: [-1, 0, 1]`;
- `grid.variant_key_version: 1`;
- `grid.variant_keys_sha256`;
- `grid.signals_v1_params_defaults`;
- `provenance`.

### `hit_times/1m/manifest.yaml`

`hit_times/1m/manifest.yaml` обязан фиксировать:

- `timeline_bar_count`;
- `sentinel_index`;
- `tp_values` и `sl_values`;
- `tables.long_tp|long_sl|short_tp|short_sl`;
- `monotonicity: non_decreasing_by_level`;
- `provenance`.

## Validator responsibilities

Whole-slot validator обязан идти в фиксированном порядке:

1. root manifest schema + root contract;
2. price arrays;
3. mapping arrays;
4. signal manifest refs + signal manifests + `signals.i8.npy`;
5. hit-times manifest ref + hit-times manifest + `tp/sl` grids + tables.

Для каждого artifact family validator обязан проверять:

- exact required keys / no unsupported drift;
- expected path literal;
- file existence;
- file `sha256`;
- `dtype`;
- `shape`;
- `axis_order`.

Дополнительно:

- prices:
  - `open_time` strict monotonicity
  - `close_time` strict monotonicity
  - `close_time > open_time`
  - timeline coverage metadata
- mappings:
  - non-decreasing indexes
  - `bar_open_1m_idx <= bar_close_1m_idx`
  - mapping bounds относительно `1m`
- signals:
  - signal value set `{-1,0,1}`
  - `shape=[V,T_tf]`
  - timeline equality с root price coverage
- hit_times:
  - `tp/sl` grids strictly increasing
  - tables bounded by sentinel
  - hit-time monotonicity by level.

## Publish interaction

R3-01 сам по себе не делает slot publish-ready.
Если `validation_plan` всё ещё требует `mappings`, `signals` или real `hit_times`, whole-slot
validation обязана fail-fast и pointer switch не выполняется до соответствующих later epics.

Runner обязан работать только в порядке:

1. resolve `current.yaml`;
2. precheck inactive slot pin guard;
3. rebuild inactive slot;
4. validate whole slot по strict manifests и validation plan, полученному из
   `backtest_artifacts.validation_plan`;
5. atomically switch `current.yaml`.

Если validation вернула хотя бы одну error diagnostic:

- publish завершается без pointer switch;
- `current.yaml` остаётся прежним;
- оператор получает stable `code/message/diagnostics`.

## Runtime contract after publish

После успешного publish runtime может:

- читать root manifest один раз;
- использовать fixed `dtype/shape/axis_order`;
- выбирать ровно нужные signal manifests и `signals.i8.npy`;
- читать `hit_times/1m` без recompute metadata.

Runtime не должен:

- повторно считать `sha256`;
- вычислять `shape` или `axis_order` по соглашениям из имени файла;
- сканировать slot для discovery.
