# Runbook — Backtest Artifacts Rebuild / Publish (R2-02)

Этот runbook фиксирует безопасную operational-процедуру для rebuild/publish артефактов в `artifacts/backtest/v2` с учётом strict `current.yaml`, slot pinning и publish guard.

Основные документы:

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

## Предусловия

- symbol root уже существует и содержит `current.yaml`, `slot_a/`, `slot_b/`;
- опубликованный active slot не мутируется in place;
- у оператора есть explicit validation plan для `prices`, `signals`, `mappings`, `hit_times`;
- background jobs пишут pin metadata:
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_manifest_hash`
  - `artifact_asof_date`

## Обязательная publish sequence

Порядок всегда один и тот же:

1. `build inactive slot`
2. `validate whole slot`
3. `atomically switch current.yaml`

Другой порядок запрещён.

## Шаг 1. Resolve pointer и inactive slot

- прочитать strict `current.yaml`;
- проверить обязательные поля:
  - `schema_version`
  - `active_slot`
  - `slot_generation`
  - `asof_date`
  - `manifest_sha256`
  - `published_at_utc`
- определить inactive slot только как противоположный текущему:
  - `slot_a -> slot_b`
  - `slot_b -> slot_a`

Если `current.yaml` невалиден, publish/rebuild прекращается сразу.

## Шаг 2. Pin guard перед rebuild

До rebuild inactive slot нужно проверить active pins:

- найти active background jobs (`queued | running`) для того же инструмента;
- сравнить persisted `artifact_slot` + `artifact_manifest_hash` с текущей inactive identity;
- если найден хотя бы один active pin, publish блокируется.

Ожидаемое сообщение класса ошибок:

- `inactive_slot_pinned`

Операционное значение:

- inactive slot нельзя rebuild'ить, пока на нём ещё висят активные runs;
- active slot contents и inactive slot contents не должны быть перезаписаны вручную в обход guard.

## Шаг 3. build inactive slot

- пересобрать только inactive slot;
- писать файлы по explicit deterministic paths;
- не использовать directory scanning как способ discover'ить содержимое;
- не изменять active slot contents.

Минимально ожидаемые пути:

- `<slot>/manifest.yaml`
- `<slot>/prices/<tf>/open_time.i64.npy`
- `<slot>/prices/<tf>/close_time.i64.npy`
- `<slot>/prices/<tf>/ohlcv.f32.npy`
- `<slot>/signals/<tf>/<indicator_id>/manifest.yaml`
- `<slot>/signals/<tf>/<indicator_id>/signals.i8.npy`
- `<slot>/mappings/<tf>/bar_open_1m_idx.u32.npy`
- `<slot>/mappings/<tf>/bar_close_1m_idx.u32.npy`
- `<slot>/hit_times/1m/manifest.yaml`

## Шаг 4. validate whole slot

Перед switch нужно explicit проверить весь inactive slot:

- `manifest.yaml` существует и читается;
- все expected `prices/*`, `signals/*`, `mappings/*`, `hit_times/*` присутствуют;
- validation идёт только по явным path targets/timeframes/indicator ids;
- hidden scanning / best-effort discovery не допускаются.

Если любой required path отсутствует, publish останавливается без изменения `current.yaml`.

## Шаг 5. atomically switch current.yaml

После успешной validation:

- вычислить новый `manifest_sha256` для inactive slot `manifest.yaml`;
- увеличить `slot_generation`;
- подготовить новый strict payload:

```yaml
schema_version: 1
active_slot: slot_b
slot_generation: 43
asof_date: "2026-03-26"
manifest_sha256: "..."
published_at_utc: "2026-03-26T03:04:05Z"
```

- записать payload в temp file в той же директории;
- выполнить atomic rename/replace;
- не делать partial overwrite существующего `current.yaml`.

## Что считается ошибкой

- missing/extra field в `current.yaml`;
- unsupported `schema_version`;
- invalid `active_slot`, `slot_generation`, `asof_date`, `manifest_sha256`, `published_at_utc`;
- попытка rebuild inactive slot при active pin;
- missing explicit validation path;
- любой non-atomic pointer write.

## Rollback / recovery notes

- если switch не произошёл, source-of-truth остаётся прежний `current.yaml`, rollback не нужен;
- если validation упала, исправлять только inactive slot и повторять sequence сначала;
- если pin guard сработал, нужно дождаться terminal state блокирующих jobs или явно отменить их по штатной процедуре;
- ручное редактирование `current.yaml` или manual cleanup slot contents вне процедуры запрещены.

## После изменения документации

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
