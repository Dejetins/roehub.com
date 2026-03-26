# Runbook — Backtest Artifacts Rebuild / Publish (R2-02 / R2-03 / R2-04)

Этот runbook фиксирует безопасную operational-процедуру для rebuild/publish артефактов в
`artifacts/backtest/v2` с учётом strict `current.yaml`, slot pinning, config-driven validation
plan и strict manifest validation.

Основные документы:

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

## Предусловия

- artifact pipeline contract загружен из strict `configs/<env>/backtest_artifacts.yaml`;
- symbol root уже существует и содержит `current.yaml`, `slot_a/`, `slot_b/`;
- опубликованный active slot не мутируется in place;
- у оператора есть explicit validation plan для `prices`, `signals`, `mappings`, `hit_times`,
  полученный из `backtest_artifacts.validation_plan`;
- background jobs пишут pin metadata:
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_manifest_hash`
  - `artifact_asof_date`

## Artifact Config Source Of Truth

Path resolution precedence:

1. `ROEHUB_BACKTEST_ARTIFACTS_CONFIG`
2. `configs/<ROEHUB_ENV>/backtest_artifacts.yaml`

Обязательные секции config contract:

- `artifact_root`
- `validation_plan`
- `hit_times_grid`
- `slot_policy`
- `publish_schedule`
- `lookback_policy`
- `validation_budgets`

Fail-fast loader обязан reject'ить:

- missing/extra keys;
- duplicate YAML keys;
- invalid timeframe / slot / indicator literals;
- duplicate sequence items;
- non-positive lookbacks / validation budgets.

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
- использовать `artifact_root` из `backtest_artifacts.yaml`, а не default literal в коде;
- писать файлы по explicit deterministic paths;
- не использовать directory scanning как способ discover'ить содержимое;
- не изменять active slot contents.

R3-01 / R3-02 / R3-03 exception boundary:

- до R3-04/R4/R5 rebuild может materialize `prices/1m/*` и rolled `prices/<tf>/*` для всех
  allowed request TF;
- после R3-03 rebuild также materialize'ит `mappings/<tf>/bar_open_1m_idx.u32.npy` и
  `mappings/<tf>/bar_close_1m_idx.u32.npy` для всех allowed request TF;
- в таком случае root `manifest.yaml` всё равно остаётся strict:
  - `mappings` уже должны содержать real non-empty sections после R3-03;
  - `signals.supported_timeframes: []`
  - `signals.supported_indicator_ids: []`
  - `signals.manifests: []`
  - `hit_times.manifest_path: "hit_times/1m/manifest.yaml"`
  - `hit_times.manifest_sha256: "0000000000000000000000000000000000000000000000000000000000000000"`
- после R3-04 такой slot можно публиковать только через explicit prices+mappings stage spec:
  - `price_timeframes` из `backtest_artifacts.validation_plan`
  - `mapping_timeframes` из `backtest_artifacts.validation_plan`
  - `signal_artifacts: []`
  - `require_hit_times_manifest: false`
- если использовать полный validation plan, такой slot по-прежнему не должен публиковаться, пока
  later-stage plan требует real `signals/hit_times`.

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

- root `manifest.yaml` schema валиден;
- `signals/<tf>/<indicator_id>/manifest.yaml` schema валидны;
- `hit_times/1m/manifest.yaml` schema валиден;
- все referenced `prices/*`, `signals/*`, `mappings/*`, `hit_times/*` присутствуют;
- `sha256` совпадает с содержимым файлов;
- `dtype`, `shape`, `axis_order` совпадают с manifest contracts;
- `open_time` / `close_time` monotonic;
- signal value set ограничен `{-1,0,1}`;
- mapping bounds валидны относительно `1m`;
- `prices/1m.open_time[bar_open_1m_idx] == prices/<tf>.open_time`;
- `prices/1m.close_time[bar_close_1m_idx] == prices/<tf>.close_time`;
- hit-time monotonicity выполняется;
- validation plan берётся из `backtest_artifacts.validation_plan` и переводится в
  `ArtifactSlotValidationSpecV2`;
- для R3-04 prices+mappings stage validation spec выводится явно из того же
  `backtest_artifacts.validation_plan`, но с
  `signal_artifacts=[]` и `require_hit_times_manifest=false`;
- validation идёт только по явным path targets/timeframes/indicator ids;
- hidden scanning / best-effort discovery не допускаются.

Для R3-01 / R3-02 / R3-03 rebuild-only stage оператор должен явно понимать:

- `prices/1m` можно пересобирать и tail-update'ить по
  `lookback_policy.price_tail_bars_1m` без pointer switch;
- rolled `prices/<tf>` пересчитываются из materialized `prices/1m` и переиспользуют unaffected
  prefix до bucket, в который попадает reread-tail start;
- `mappings/<tf>` пересчитываются только из materialized `prices/1m` и `prices/<tf>` и
  переиспользуют unaffected prefix до последнего request-TF бара, чей `close_time` остаётся
  левее mapping-tail window;
- mapping tail rebuild bounded by `lookback_policy.mapping_tail_bars_1m`;
- rebuild обязан валидировать epoch-aligned bucket boundaries и full-bucket coverage для rolled TF;
- rebuild обязан валидировать mapping monotonicity, bounds и strict price correspondence;
- publish разрешён для R3-04 только если explicit stage spec ограничен `prices+mappings`;
- publish остаётся запрещённым, если active validation plan still expects later-stage artifacts.

Если есть хотя бы один validator diagnostic, publish останавливается без изменения `current.yaml`.

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
- missing/extra field в root/signal/hit-times manifests;
- unsupported `schema_version`;
- invalid `active_slot`, `slot_generation`, `asof_date`, `manifest_sha256`, `published_at_utc`;
- invalid `dtype`, `shape`, `axis_order`, `sha256`, `provenance`;
- invalid signal value set `{-1,0,1}`;
- invalid mapping bounds;
- invalid hit-time monotonicity;
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
