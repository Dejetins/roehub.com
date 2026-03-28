# Backtest Precompute Runner V2 (R2-03 / R2-04 / R3-01 / R3-02 / R3-03 / R3-04 / R4-01 / R4-02 / R4-03 / R5-01)

Статус: `Milestone R2 / EPIC R2-03 + R2-04`, `Milestone R3 / EPIC R3-01 + R3-02 + R3-03 + R3-04`, `Milestone R4 / EPIC R4-01 + R4-02 + R4-03`, `Milestone R5 / EPIC R5-01`

Документ фиксирует контракт precompute/publish слоя, который:

- строит inactive slot в `artifacts/backtest/v2`;
- пишет strict manifests для root / signals / hit_times;
- выполняет fail-fast validation до switch `current.yaml`;
- оставляет runtime только fixed metadata reads без schema inference и без hash recomputation.

Основные документы:

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
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

### R4-01 / R4-02 / R4-03 signal boundary

На этапе R4-01 precompute layer получил explicit signal-rules engine contract.
На этапе R4-02 этот contract стал source-of-truth для real signal artifact materialization.

Это означает:

- indicator outputs детерминированно преобразуются в compact `int8` signals `{-1,0,1}`;
- `inputs.source` трактуется явно для rule families, где price сравнивается с indicator output;
- `signals.v1.params` остаются strictly `default-only` и берутся только из
  `configs/<env>/indicators.yaml`;
- для каждого explicit target из `backtest_artifacts.validation_plan.signal_artifacts`
  runner обязан писать:
  - `signals/<tf>/<indicator_id>/signals.i8.npy`
  - `signals/<tf>/<indicator_id>/manifest.yaml`
- root manifest обязан публиковать real `signals.supported_timeframes`,
  `signals.supported_indicator_ids` и `signals.manifests`;
- после R4-03 rebuild обязан выводить bounded per-target signal window из
  `lookback_policy.signal_tail_bars_1m`, а затем materialize'ить только
  `prefix + rebuilt_tail` по time axis;
- prefix reuse разрешён только при strict reuse-check:
  - target уже перечислен в root `signals.manifests`
  - existing `manifest.yaml` и `signals.i8.npy` существуют
  - `rows_count`, `timeline`, `variant_key_version`, `variant_keys_sha256`,
    `signals.v1.params = default-only` и file `sha256` не дрейфуют
- missing target files могут переводить target в deterministic full rebuild, но manifest/data
  drift при reuse attempt обязан fail-fast с stable diagnostics;
- R4-04 propagation `source` в runtime payloads теперь закрывается downstream-контрактами:
  `GET /backtests/runtime-defaults`, persisted jobs `/top` payloads и explicit
  `variant-report` payloads.

### R5-01 `hit_times/1m` boundary

На этапе R5-01 precompute runner materialize'ит strict `hit_times/1m` family из уже
artifact-backed `prices/1m.ohlcv`.

Это означает:

- `backtest_artifacts.hit_times_grid` становится source-of-truth для `tp_values` и `sl_values`;
- runner обязан писать real files:
  - `hit_times/1m/tp_values.f32.npy`
  - `hit_times/1m/sl_values.f32.npy`
  - `hit_times/1m/long_tp.u32.npy`
  - `hit_times/1m/long_sl.u32.npy`
  - `hit_times/1m/short_tp.u32.npy`
  - `hit_times/1m/short_sl.u32.npy`
  - `hit_times/1m/manifest.yaml`
- `sentinel_index` обязан равняться `timeline_bar_count`, а таблицы обязаны оставаться
  bounded-by-sentinel и monotone by level;
- root manifest больше не должен публиковать placeholder hash для `hit_times`, если slot построен
  этим R5-01 path;
- runtime читает `hit_times/1m` только по strict manifest metadata, без recompute и discovery.
- на этом boundary ответственность precompute слоя заканчивается: `signal timeline`,
  `execution timeline`, `compact trade list`, `fast TP/SL grid search`,
  `exact replay of best TP/SL cell` и `metrics over compact trades` описываются отдельно в
  `docs/architecture/backtest/backtest-runtime-kernels-v2.md`;
- precompute runner materialize'ит inputs для `signal_tf + 1m_risk`, но не становится notebook
  orchestration layer.

### R3-01 / R3-02 prices stage

На этапах R3-01 / R3-02 precompute runner получает отдельную обязанность:

- материализовать canonical source-of-truth export для `prices/1m/*` в inactive slot;
- затем построить из materialized `prices/1m/*` только разрешённые request TF:
  - `15m`
  - `30m`
  - `1h`
  - `2h`
  - `4h`
  - `6h`
  - `8h`
  - `1d`
  - `2d`
  - `3d`
- писать для каждого TF:
  - `prices/<tf>/open_time.i64.npy`
  - `prices/<tf>/close_time.i64.npy`
  - `prices/<tf>/ohlcv.f32.npy`
- использовать source table `market_data.canonical_candles_1m` только для canonical `1m`
  export через existing `CanonicalCandleReader` contract;
- строить rollup только из artifact-backed `prices/1m`, без ClickHouse reads на runtime hot path;
- поддерживать deterministic tail update по
  `backtest_artifacts.lookback_policy.price_tail_bars_1m`;
- строить `mappings/<tf>/bar_open_1m_idx.u32.npy` и
  `mappings/<tf>/bar_close_1m_idx.u32.npy` только из artifact-backed `prices/1m` и
  `prices/<tf>`;
- поддерживать deterministic tail update для mappings по
  `backtest_artifacts.lookback_policy.mapping_tail_bars_1m`;
- никогда не мутировать active slot in place.

Tail update semantics для R3-01 / R3-02 / R3-03:

- если inactive slot ещё не содержит valid `prices/1m`, выполняется full build по заданному
  `TimeRange [start, end)`, затем full rollup для всех allowed request TF;
- если `prices/1m` уже существует в inactive slot, runner переиспользует prefix внутри requested
  range и reread'ит только tail overlap длиной `price_tail_bars_1m`;
- для rolled `prices/<tf>` prefix reuse считается от bucket, в который попадает reread-tail start;
- для `mappings/<tf>` prefix reuse считается до последнего request-TF бара, чей `close_time`
  остаётся строго левее первого `1m` bar open, попавшего в mapping-tail window;
- mapping rebuild обязан сохранять `dtype=uint32`, `shape=[T_tf]`,
  `bar_open_1m_idx <= bar_close_1m_idx` и exact price correspondence;
- merge policy фиксирована как `prefix + rebuilt_tail`, без best-effort dedup/coercion;
- identical source candles + identical config/request должны давать byte-stable `.npy` и
  `manifest.yaml`.

### R4-03 signal tail-update semantics

Signal rebuild для explicit configured targets обязан быть локальным и deterministic:

- source-of-truth для bounded signal tail rebuild:
  - `lookback_policy.signal_tail_bars_1m`
  - target timeframe duration
  - finite compute context, выведенный из materialized grid axes
  - finite lag/default-only context из `signals.v1.params`
- effective tail window считается в target bars и используется только для explicit configured
  `(timeframe, indicator_id)` targets;
- compute window строится локально внутри precompute runner internals без filesystem discovery;
- merge policy фиксирована как `prefix + rebuilt_tail`, без hidden dedup/coercion;
- merged matrix обязана оставаться strict:
  - `dtype: int8`
  - `shape: [V, T_tf]`
  - `axis_order: [variant, time]`
  - value set `{-1,0,1}`
- per-indicator manifest после merge обязан обновлять:
  - `rows_count`
  - `timeline`
  - `signals.sha256`
  - provenance inputs с `lookback_policy.signal_tail_bars_1m`,
    `effective_target_tail_bars` и `rebuild_strategy = prefix + rebuilt_tail`
- root `signals` catalog обязан оставаться deterministic:
  - `signals.supported_timeframes` deduplicated in canonical timeframe order
  - `signals.supported_indicator_ids` in lexical order
  - `signals.manifests` ordered by `(timeframe, indicator_id)`
- identical source candles + identical config/request + identical generated timestamp должны
  давать byte-stable `signals/<tf>/<indicator_id>/signals.i8.npy` и related manifests.

Rollup contract для R3-02:

- bucket boundaries считаются только через `Timeframe.bucket_open/bucket_close`;
- materialize'ятся только fully covered epoch-aligned buckets;
- partial leading/trailing buckets детерминированно отбрасываются;
- `open_time` / `close_time` пишутся отдельно от `ohlcv`;
- root manifest обязан содержать metadata и coverage для `1m` и всех rolled request TF.

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
- `signals.supported_timeframes`;
- `signals.supported_indicator_ids`;
- `signals.manifests[]` с `manifest_path` и `manifest_sha256`;
- `hit_times.manifest_path` и `manifest_sha256`;
- `signal_encoding`:
  - `dtype: int8`
  - `axis_order: [variant, time]`
  - `value_set: [-1, 0, 1]`
- `provenance`.

R3-01 / R3-02 / R3-03 placeholder strategy до materialization следующих stage:

- `prices[]` содержит свежие strict sections для `1m` и всех allowed request TF;
- `mappings[]` может оставаться пустым до R3-03;
- `signals` фиксируется как explicit empty catalog
  (`supported_timeframes=[]`, `supported_indicator_ids=[]`, `manifests=[]`) до R4-02;
- `hit_times` обязан оставаться explicit fixed-path reference
  `hit_times/1m/manifest.yaml`, но до R5-01 допускается placeholder
  `manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"`;
- `signal_encoding` остаётся fixed even when `signals.manifests` is empty.

R3-03 mapping contract:

- `mappings[]` больше не placeholder и обязан содержать non-empty strict sections для:
  - `15m`
  - `30m`
  - `1h`
  - `2h`
  - `4h`
  - `6h`
  - `8h`
  - `1d`
  - `2d`
  - `3d`
- для каждого section обязательны metadata:
  - `path`
  - `dtype`
  - `shape`
  - `axis_order`
  - `sha256`
- validator обязан подтверждать:
  - bounds within `[0, T_1m)`
  - monotonicity
  - `bar_open_1m_idx <= bar_close_1m_idx`
  - `prices/1m.open_time[bar_open_1m_idx] == prices/<tf>.open_time`
  - `prices/1m.close_time[bar_close_1m_idx] == prices/<tf>.close_time`

R4-02 replaces the root-manifest signal placeholder for explicit configured targets:

- `signals.supported_timeframes` must equal the deduplicated ordered timeframes from
  `signals.manifests`;
- `signals.supported_indicator_ids` must equal the lexical ordered indicator ids from
  `signals.manifests`;
- `signals.manifests` must be ordered deterministically by timeframe contract then
  `indicator_id`;
- `signals.manifests` remains explicit configured-target metadata; directory scanning is not a
  supported source of truth;
- root manifest keeps `hit_times/1m` as an explicit placeholder reference only until R5-01.

### Per-indicator signal manifest

Каждый `signals/<tf>/<indicator_id>/manifest.yaml` обязан фиксировать:

- `indicator_id`, `timeframe`;
- `signals.path = signals/<tf>/<indicator_id>/signals.i8.npy`;
- `signals.dtype = int8`;
- `signals.shape = [V, T_tf]`;
- `signals.axis_order = [variant, time]`;
- `signals.sha256`;
- `rows_count = V`;
- `timeline` coverage, совпадающий с root `prices/<tf>.coverage`;
- `signal_value_set: [-1, 0, 1]`;
- `grid.variant_key_version: 1`;
- `grid.variant_keys_sha256`;
- `grid.signals_v1_params_defaults` из strict `signals.v1.params = default-only`;
- `provenance`.

R4-03 provenance additions for per-indicator signal manifests:

- `inputs_sha256` must include `lookback_policy.signal_tail_bars_1m`;
- `inputs_sha256` must include the effective target tail budget derived for the target;
- `inputs_sha256` must include `rebuild_strategy = prefix + rebuilt_tail`.

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
  - exact correspondence с materialized `prices/1m` и `prices/<tf>`
- signals:
  - signal value set `{-1,0,1}`
  - `shape=[V,T_tf]`
  - timeline equality с root price coverage
  - deterministic root catalog ordering and hash/path correspondence
- hit_times:
  - `tp/sl` grids strictly increasing
  - tables bounded by sentinel
  - hit-time monotonicity by level.

## Publish interaction

R3-01 / R3-02 / R3-03 сами по себе не делают slot publish-ready.
R3-04 делает publish-ready только stage `prices + mappings`, если validation scope выбран явно и
config-driven:

- `price_timeframes` и `mapping_timeframes` берутся из `backtest_artifacts.validation_plan`;
- `signal_artifacts = ()`;
- `require_hit_times_manifest = false`.

После R4-02 full validation spec уже может требовать real `signals` и успешно проходить, если
root catalog и per-indicator manifests materialized для explicit configured targets.
После R5-01 full validation spec может также требовать real `hit_times/1m`, если slot построен
через актуальный precompute runner path. Отдельный R3-04 prices+mappings stage helper по-прежнему
должен оставаться explicit и выставлять `require_hit_times_manifest = false`.

Runner обязан работать только в порядке:

1. resolve `current.yaml`;
2. precheck inactive slot pin guard;
3. rebuild inactive slot;
4. validate whole slot по strict manifests и explicit validation spec, полученному из
   `backtest_artifacts.validation_plan`;
5. atomically switch `current.yaml`.

Для R3-04 рекомендуется отдельный config-driven derivation:

- взять `price_timeframes` из `validation_plan`;
- взять `mapping_timeframes` из `validation_plan`;
- принудительно выставить `signal_artifacts = ()`;
- принудительно выставить `require_hit_times_manifest = false`.

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
