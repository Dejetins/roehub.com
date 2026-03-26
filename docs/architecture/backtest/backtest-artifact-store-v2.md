# Backtest Artifact Store V2 (R2-01 / R2-02)

Статус: `Milestone R2 / EPIC R2-01 + R2-02`

Документ фиксирует:

- R2-01: deterministic layout/path contract для `artifacts/backtest/v2`;
- R2-02: strict `current.yaml` contract, publish sequence `build inactive slot -> validate whole slot -> atomically switch current.yaml`, slot pinning и publish guard.

Что не входит в этот документ:

- R2-03: manifest schema/hash/dtype/shape validators;
- R2-04: `configs/<env>/backtest_artifacts.yaml` loader/validator contract.

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
