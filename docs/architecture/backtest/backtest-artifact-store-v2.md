# Backtest Artifact Store V2 (R2-01 / R2-02 / R2-03 / R2-04)

Статус: `Milestone R2 / EPIC R2-01 + R2-02 + R2-03 + R2-04`

Документ фиксирует:

- R2-01: deterministic layout/path contract для `artifacts/backtest/v2`;
- R2-02: strict `current.yaml` contract, publish sequence `build inactive slot -> validate whole slot -> atomically switch current.yaml`, slot pinning и publish guard.
- R2-03: strict manifest schemas, fail-fast slot validators, fixed runtime metadata from manifests.
- R2-04: strict `configs/<env>/backtest_artifacts.yaml` contract для artifact root,
  validation plan, hit-times grid и publish/runtime boundary.

Основные источники:

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`

## Канонический root

Корневой каталог artifact store v2:

```text
artifacts/backtest/v2
```

Координаты symbol root:

- `exchange`
- `market_type`
- `symbol`

Канонический symbol root:

```text
artifacts/backtest/v2/<exchange>/<market_type>/<symbol>/
```

## Artifact Runtime Config Contract (R2-04)

Artifact pipeline configuration больше не живёт в `configs/<env>/backtest.yaml`.

Источник правды для artifact runtime/precompute settings:

- `configs/dev/backtest_artifacts.yaml`
- `configs/test/backtest_artifacts.yaml`
- `configs/prod/backtest_artifacts.yaml`

Path resolution precedence:

1. `ROEHUB_BACKTEST_ARTIFACTS_CONFIG`
2. `configs/<ROEHUB_ENV>/backtest_artifacts.yaml`

Strict top-level contract:

- `version`
- `backtest_artifacts`

Strict `backtest_artifacts` sections:

- `artifact_root`
- `validation_plan`
- `hit_times_grid`
- `slot_policy`
- `publish_schedule`
- `lookback_policy`
- `validation_budgets`

R2-04 guarantees:

- `artifact_root` becomes source-of-truth root for `BacktestArtifactPathBuilderV2`;
- publish validation plan is translated from frozen config into
  `ArtifactSlotValidationSpecV2`;
- missing keys, extra keys, duplicate YAML keys, duplicate sequence items and invalid literals
  are rejected fail-fast with stable diagnostics;
- config ordering is canonicalized deterministically for timeframes, signal targets, hit-times
  grids and slot list.

## Каноническое дерево

```text
artifacts/backtest/v2/
  <exchange>/
    <market_type>/
      <symbol>/
        current.yaml
        slot_a/
          manifest.yaml
          prices/
            1m/
              open_time.i64.npy
              close_time.i64.npy
              ohlcv.f32.npy
            15m/
            30m/
            1h/
            2h/
            4h/
            6h/
            8h/
            1d/
            2d/
            3d/
          signals/
            15m/
              <indicator_id>/
                signals.i8.npy
                manifest.yaml
            30m/
            1h/
            2h/
            4h/
            6h/
            8h/
            1d/
            2d/
            3d/
          mappings/
            15m/
              bar_open_1m_idx.u32.npy
              bar_close_1m_idx.u32.npy
            30m/
            1h/
            2h/
            4h/
            6h/
            8h/
            1d/
            2d/
            3d/
          hit_times/
            1m/
              manifest.yaml
              tp_values.f32.npy
              sl_values.f32.npy
              long_tp.u32.npy
              long_sl.u32.npy
              short_tp.u32.npy
              short_sl.u32.npy
        slot_b/
          ...
```

## Naming contract

| Scope | Path |
| --- | --- |
| Symbol root pointer | `<root>/<exchange>/<market_type>/<symbol>/current.yaml` |
| Slot root | `<root>/<exchange>/<market_type>/<symbol>/slot_a/` or `slot_b/` |
| Slot manifest | `<slot>/manifest.yaml` |
| Prices | `<slot>/prices/<tf>/open_time.i64.npy` |
| Prices | `<slot>/prices/<tf>/close_time.i64.npy` |
| Prices | `<slot>/prices/<tf>/ohlcv.f32.npy` |
| Signals | `<slot>/signals/<tf>/<indicator_id>/signals.i8.npy` |
| Signals | `<slot>/signals/<tf>/<indicator_id>/manifest.yaml` |
| Mappings | `<slot>/mappings/<tf>/bar_open_1m_idx.u32.npy` |
| Mappings | `<slot>/mappings/<tf>/bar_close_1m_idx.u32.npy` |
| Hit times | `<slot>/hit_times/1m/manifest.yaml` |
| Hit times | `<slot>/hit_times/1m/tp_values.f32.npy` |
| Hit times | `<slot>/hit_times/1m/sl_values.f32.npy` |
| Hit times | `<slot>/hit_times/1m/long_tp.u32.npy` |
| Hit times | `<slot>/hit_times/1m/long_sl.u32.npy` |
| Hit times | `<slot>/hit_times/1m/short_tp.u32.npy` |
| Hit times | `<slot>/hit_times/1m/short_sl.u32.npy` |

## R3-01 / R3-02 stage boundary

До R3-04/R4/R5 store contract уже требует strict root `manifest.yaml`, но цены materialize'ятся
поэтапно:

- R3-01 materializes canonical `prices/1m`;
- R3-02 materializes `prices/<tf>` для всех allowed request TF из artifact-backed `prices/1m`.

На этом этапе допустимо и ожидаемо:

- `prices/1m/*` присутствуют как real artifact files с strict metadata;
- после R3-02 присутствуют real artifact files и strict manifest metadata для:
  - `prices/15m/*`
  - `prices/30m/*`
  - `prices/1h/*`
  - `prices/2h/*`
  - `prices/4h/*`
  - `prices/6h/*`
  - `prices/8h/*`
  - `prices/1d/*`
  - `prices/2d/*`
  - `prices/3d/*`
- `mappings[]` в root manifest остаётся пустым до R3-03;
- `signals` в root manifest остаётся explicit empty catalog до R4;
- `hit_times` в root manifest остаётся explicit fixed-path reference
  `hit_times/1m/manifest.yaml`, но до R5 может использовать placeholder
  `manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"`.

Это не relax schema.
Это explicit stage placeholder contract, который later epics обязаны заменить на real manifests и
validated files перед publish.

## Fixed literals and ordering

Слоты фиксированы и упорядочены:

1. `slot_a`
2. `slot_b`

Допустимые TF для `prices/<tf>`:

- `1m`
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

Допустимые TF для `signals/<tf>` и `mappings/<tf>`:

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

Для `hit_times` допустим только фиксированный путь:

- `hit_times/1m/manifest.yaml`

## Validation contract

Path builder и loader обязаны fail-fast reject'ить:

- любой slot, отличный от `slot_a` или `slot_b`;
- пустые `exchange`, `market_type`, `symbol`;
- пустой `indicator_id`;
- токены с whitespace;
- токены с `/`, `\`, `..`, `.` или NUL;
- недопустимые TF вне фиксированных списков выше.

Нормализация токенов не допускается.
Runtime не должен silently trim/clean values перед построением путей.

## Loader contract

Loader работает только по explicit deterministic paths:

- читает `current.yaml` по точному symbol-root path;
- читает slot `manifest.yaml` по точному slot-root path;
- резолвит пути `prices`, `signals`, `mappings`, `hit_times` напрямую из координат и literals;
- не использует `os.listdir`, `Path.iterdir`, `glob`, `rglob`, `Path.walk` как обязательный шаг hot path.

## Strict `current.yaml` contract (R2-02)

`current.yaml` обязан содержать ровно эти поля:

- `schema_version`
- `active_slot`
- `slot_generation`
- `asof_date`
- `manifest_sha256`
- `published_at_utc`

Runtime-проверки:

- `schema_version` должен быть поддержан (`1`);
- `active_slot` должен быть `slot_a | slot_b`;
- `slot_generation` должен быть positive int;
- `asof_date` должен быть strict date literal `YYYY-MM-DD`;
- `manifest_sha256` должен быть 64-char lowercase hex;
- `published_at_utc` должен быть strict UTC timestamp literal `YYYY-MM-DDTHH:MM:SSZ`;
- missing keys и extra keys запрещены.

Канонический пример:

```yaml
schema_version: 1
active_slot: slot_a
slot_generation: 42
asof_date: "2026-03-24"
manifest_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
published_at_utc: "2026-03-24T02:00:00Z"
```

## Publish contract (R2-02)

Publish всегда работает только через два фиксированных слота и не мутирует активный слот in place.

Последовательность всегда одна:

1. `build inactive slot`
2. `validate whole slot`
3. `atomically switch current.yaml`

Правила:

- inactive slot определяется только как противоположный `active_slot`;
- publish не делает hidden scanning;
- validation получает explicit deterministic paths/timeframes/indicator ids;
- pointer payload сериализуется в фиксированном порядке:
  - `schema_version`
  - `active_slot`
  - `slot_generation`
  - `asof_date`
  - `manifest_sha256`
  - `published_at_utc`
- switch выполняется через temp-file write + atomic rename/replace;
- partial overwrite `current.yaml` запрещён.

## Slot pinning contract (R2-02)

Каждый active background run/job обязан хранить immutable pin metadata:

- `artifact_slot`
- `artifact_slot_generation`
- `artifact_manifest_hash`
- `artifact_asof_date`

Pin metadata фиксируется при create-time из strict `current.yaml` и не изменяется до terminal state.

Назначение pin metadata:

- гарантировать reproducibility во время `queued/running`;
- дать publish guard детерминированный способ проверить, можно ли rebuild/publish inactive slot;
- отделить job identity от последующих publish switch'ей.

## Publish guard (R2-02)

Перед publish runtime обязан проверить inactive slot на блокировку:

- если inactive slot pinned хотя бы одним active background run (`queued | running`), publish fail-fast прерывается;
- blocking decision опирается на persisted `(artifact_slot, artifact_manifest_hash)` и instrument identity;
- безопасный rebuild/publish допускается только когда blocking active run count = `0`.

Это правило нужно именно перед шагом `build inactive slot`, потому что rebuild inactive slot иначе затрёт dataset, который уже pinned запущенной job.

## Strict manifest schemas (R2-03)

R2-03 добавляет три фиксированных строгих schema contract:

- root slot `manifest.yaml`;
- per-indicator `signals/<tf>/<indicator_id>/manifest.yaml`;
- `hit_times/1m/manifest.yaml`.

Все manifest contracts используют:

- explicit `schema_version` и `manifest_kind`;
- exact required keys, extra keys запрещены;
- explicit `dtype`, `shape`, `axis_order`, `sha256`, `provenance`;
- explicit `slot_generation` и `asof_date`;
- slot-relative deterministic `path` literals;
- fail-fast validation без best-effort coercion.

### Root `manifest.yaml`

Root manifest обязан содержать:

- `identity` (`exchange`, `market_type`, `symbol`);
- `slot`, `slot_generation`, `asof_date`;
- список `prices` с metadata для `open_time`, `close_time`, `ohlcv`;
- список `mappings` с metadata для `bar_open_1m_idx`, `bar_close_1m_idx`;
- `signals`:
  - `supported_timeframes`
  - `supported_indicator_ids`
  - `manifests[]` c `timeframe`, `indicator_id`, `manifest_path`, `manifest_sha256`
- `hit_times` c `manifest_path`, `manifest_sha256`;
- `signal_encoding` c fixed contract:
  - `dtype: int8`
  - `axis_order: [variant, time]`
  - `value_set: [-1, 0, 1]`
- `provenance`.

Канонический shape section для price arrays:

- `open_time`: `dtype=int64`, `shape=[T_tf]`, `axis_order=[time]`
- `close_time`: `dtype=int64`, `shape=[T_tf]`, `axis_order=[time]`
- `ohlcv`: `dtype=float32`, `shape=[T_tf, 5]`, `axis_order=[time, field]`

### Per-indicator signal manifest

Per-indicator `signals/<tf>/<indicator_id>/manifest.yaml` обязан содержать:

- `indicator_id`, `timeframe`, `slot`, `slot_generation`, `asof_date`;
- `signals` metadata:
  - `path: signals/<tf>/<indicator_id>/signals.i8.npy`
  - `dtype: int8`
  - `shape: [V, T_tf]`
  - `axis_order: [variant, time]`
  - `sha256`
- `rows_count`;
- `timeline` coverage;
- `signal_value_set: [-1, 0, 1]`;
- `grid`:
  - `variant_key_version: 1`
  - `variant_keys_sha256`
  - `signals_v1.params defaults`
- `provenance`.

### `hit_times/1m/manifest.yaml`

`hit_times/1m/manifest.yaml` обязан содержать:

- `timeframe: 1m`, `slot`, `slot_generation`, `asof_date`;
- `timeline_bar_count`;
- `sentinel_index` (равен `timeline_bar_count`);
- `tp_values` и `sl_values`:
  - `dtype: float32`
  - `shape: [N_levels]`
  - `axis_order: [level]`
  - `sha256`
- `tables.long_tp|long_sl|short_tp|short_sl`:
  - `dtype: uint32`
  - `shape: [N_levels, T_1m]`
  - `axis_order: [level, time]`
  - `sha256`
  - `monotonicity: non_decreasing_by_level`
- `provenance`.

## Slot-wide validator contract (R2-03)

Перед `current.yaml` switch publish validator обязан детерминированно проверить весь inactive slot:

- root/signal/hit-times manifests читаются только по explicit paths;
- strict schema version / manifest kind / required keys;
- `sha256` каждого referenced file;
- `dtype` contract;
- `shape` contract;
- `axis_order` contract;
- monotonic `open_time` и `close_time`;
- signal value set `{-1,0,1}`;
- mapping bounds относительно `1m` timeline;
- hit-time monotonicity;
- `provenance` field presence/format.

Диагностики должны быть:

- structured;
- deterministic by artifact order;
- stable for the same invalid slot contents.

Runtime hot path не делает эти проверки повторно.
Runtime читает fixed metadata из manifests и не recompute'ит schema facts на месте.

## Кодовый контракт

R2-01/R2-02 реализованы следующими модулями:

- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py`
- `src/trading/contexts/backtest/application/services/v2/contracts.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py`
- `src/trading/contexts/backtest/domain/entities/backtest_job.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `docs/runbooks/backtest-artifacts-rebuild.md`

Unit-тесты:

- `tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_backtest_artifact_path_builder_v2.py`
- `tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_current_pointer_writer_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py`
