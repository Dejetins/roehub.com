# Runbook — Backtest Artifacts Rebuild / Publish (R2-02 / R2-03 / R2-04 / R4-02 / R4-03 / R5-01 / R6-01)

Этот runbook фиксирует безопасную operational-процедуру для rebuild/publish артефактов в
`artifacts/backtest/v2` с учётом strict `current.yaml`, slot pinning, config-driven validation
plan, strict manifest validation и R6-01 runtime bootstrap boundary.

## Status

- Status: active operational runbook for the delivered artifact-backed runtime.
- Canonical scope:
  - rebuild/publish for the same slot family consumed by sync launch, claimed worker, and
    run-scoped lazy detail;
  - publish guards consider only active `background_auto` and `background_manual_legacy` rows.
- Compatibility note:
  - this runbook documents rebuild/publish only and does not revive legacy runtime paths;
  - broader perf closure and extra troubleshooting expansion remain R10-03 scope.

Основные документы:

- `docs/architecture/backtest/README.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

## Scheduled service contract

- Artifact rebuild/publish is executed by a dedicated service on Mac Studio native backend.
- Scheduled mode is anchored to `Europe/Moscow` and runs daily at `03:05`.
- Scheduled universe source-of-truth is `market_data.ref_instruments`; the service processes all
  enabled+tradable pairs from its latest snapshot.
- Manual mode may rebuild one explicit symbol root or a bounded subset, but still uses the same
  inactive-slot build and whole-slot validation contract.
- Service execution must be serialized with a host-level lock so overlapping rebuild/publish runs
  do not target the same inactive slot concurrently.

Minimal service metrics:

- `backtest_artifact_publish_runs_total{status}`
- `backtest_artifact_publish_duration_seconds`
- `backtest_artifact_publish_symbols_total{status}`
- `backtest_artifact_publish_blocked_total{reason}`
- `backtest_artifact_publish_last_success_unixtime`
- `backtest_artifact_tail_rebuild_bars_total{stage}`

R12 implementation-facing code surface near this runbook:

- `ArtifactPrecomputeCoordinatorV2` owns stage order and structured progress logs.
- `ArtifactTimeframeSessionV2` owns explicit open/close lifecycle for one `current_timeframe`.
- `ArtifactPrecomputeProgressEventV2` and `ArtifactPrecomputeStageResultV2` define the payload
  shape behind the structured logs and final `stage_results` completion summary.

## Предусловия

- artifact pipeline contract загружен из strict `configs/<env>/backtest_artifacts.yaml`;
- prod `artifact_root` указывает на стабильный host data path вне repo checkout;
- symbol root может либо уже содержать `current.yaml`, `slot_a/`, `slot_b/`, либо отсутствовать
  целиком в bootstrap case;
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

Artifact-precompute indicators config precedence:

1. `ROEHUB_INDICATORS_CONFIG`
2. sibling `indicators.yaml` of the explicitly selected artifact config
   example: `configs/prod/backtest_artifacts.yaml` -> `configs/prod/indicators.yaml`
3. `configs/<ROEHUB_ENV>/indicators.yaml`
4. final default `configs/dev/indicators.yaml`

Обязательные секции config contract:

- `artifact_root`
- `validation_plan`
- `hit_times_grid`
- `slot_policy`
- `publish_schedule`
- `lookback_policy`
- `validation_budgets`

R12 execution-policy contract:

- `execution_policy` is part of the checked-in `configs/dev|test|prod/backtest_artifacts.yaml`;
- mandatory keys:
  - `max_open_timeframe_sessions`
  - `signal_worker_processes`
  - `signal_worker_memory_budget_bytes`
  - `signal_chunk_rows_min`
  - `signal_chunk_rows_max`
- этот section меняет только execution model и resource bounds;
- layout/manifests/public runtime contracts не меняются.

`validation_plan.signal_artifacts` contract:

- explicit list of `{timeframe, indicator_id}` items, or
- machine-readable literal `all_supported_v1` for full signal registry expansion across all
  artifact signal timeframes.

Ключевые budget keys для `hit_times`:

- `validation_budgets.max_hit_times_cells` для steady-state incremental rebuild;
- `validation_budgets.max_hit_times_cells_full_rebuild` для bootstrap пустого symbol root и для
  ручного `--full-rebuild`.

Отдельно от этого:

- public/runtime indicators memory guard продолжает жить в `configs/<env>/indicators.yaml ->
  compute.numba.max_compute_bytes_total`;
- artifact rebuild/publish intentionally does **not** use this ceiling during signal materialization;
- manual CLI и scheduled publisher поднимают dedicated offline compute wiring with
  effectively-unbounded total compute budget, потому что slot build остаётся batch/offline flow и
  уже ограничен stage-specific artifact contracts (`max_signal_rows_per_artifact`,
  `max_hit_times_cells*`, whole-slot validation).
- R13-01 additionally narrows the heaviest non-`ma.*` defaults in
  `configs/<env>/indicators.yaml` so one `all_supported_v1` publish keeps full indicator coverage
  without operationally oversized signal matrices.
- Canonical `inputs.source` catalogs remain required for source-aware narrowed families:
  `close`, `hlc3`, `ohlc4`, `low`, `high`, `open`.
- Zero-axis signal targets `structure.candle_stats`, `volatility.tr`, `volume.ad_line`,
  `volume.obv` intentionally may keep `compute_defaults(...) = None` in YAML; precompute derives
  their deterministic single-variant grid from hard definitions, but missing defaults for
  axis-bearing indicators remain a fail-fast misconfiguration.

Fail-fast loader обязан reject'ить:

- missing/extra keys;
- duplicate YAML keys;
- invalid timeframe / slot / indicator literals;
- duplicate sequence items;
- non-positive lookbacks / validation budgets.

## Manual CLI entrypoint

Manual publish uses the same shared orchestration as the future scheduler service and keeps the
same strict sequence:

1. `build inactive slot`
2. `validate whole slot`
3. `atomically switch current.yaml`

Default manual mode stays incremental-ready for one explicit symbol root:

```bash
uv run python -m apps.cli.main.main backtest-artifact-publish \
  --config configs/prod/backtest_artifacts.yaml \
  --exchange binance \
  --market-type spot \
  --symbol BTCUSDT
```

Operational note:

- explicit `--config configs/prod/backtest_artifacts.yaml` is now sufficient to force the matching
  `configs/prod/indicators.yaml` for artifact precompute, even when `ROEHUB_ENV` is unset;
- `ROEHUB_INDICATORS_CONFIG` still wins over the artifact-config-derived path and should be used
  only for deliberate operator overrides.

Explicit deterministic full rebuild for one target:

```bash
uv run python -m apps.cli.main.main backtest-artifact-publish \
  --config configs/prod/backtest_artifacts.yaml \
  --exchange binance \
  --market-type spot \
  --symbol BTCUSDT \
  --full-rebuild
```

CLI result contract returns deterministic diagnostics with:

- target coordinates;
- `publish_mode` = `bootstrap` | `incremental` | `full_rebuild`;
- old/new slot identity and slot generation;
- whole-slot validation summary.

Для ручного операционного прогона удобнее сразу писать лог в файл:

```bash
uv run python -m apps.cli.main.main backtest-artifact-publish \
  --config configs/prod/backtest_artifacts.yaml \
  --exchange binance \
  --market-type spot \
  --symbol BTCUSDT \
  --full-rebuild \
  2>&1 | tee /tmp/backtest-artifact-publish-BTCUSDT.log
```

Manual progress checks:

```bash
rg "event=artifact_precompute_(stage_started|stage_finished|chunk_started|chunk_finished|completed|failed)" /tmp/backtest-artifact-publish-BTCUSDT.log
tail -f /tmp/backtest-artifact-publish-BTCUSDT.log
```

## R12 progress model: как читать длинный publish

Prometheus для `backtest-artifact-publisher` отвечает на вопрос "здоров ли цикл и остаётся ли
tail bounded". Structured logs отвечают на вопрос "что runner делает прямо сейчас".

Минимальные поля, которые оператор должен ожидать в structured progress logs:

- `event=artifact_precompute_stage_started|artifact_precompute_stage_finished`
- `stage`
- `current_timeframe`
- `current_indicator_id`
- `chunk_index`
- `chunk_count`
- `row_start_inclusive`
- `row_end_exclusive`
- `chunk_rows`
- `completed_chunks_total`
- `completed_indicators_total`

R12 additive completion detail:

- final `event=artifact_precompute_finished` now carries `details.stage_results` in deterministic
  execution order;
- each `timeframe_session` summary now carries `completed_chunks_total` and
  `completed_indicators_total`;
- this does not change CLI/scheduler publish result shape, but gives operators and future metrics
  adapters one canonical per-stage summary stream.

Интерпретация:

- если stage перешёл в `timeframe_session`, то одновременно должен быть открыт только один
  `current_timeframe`;
- если один и тот же `current_timeframe` держится долго, это нормально для bootstrap full build,
  но требует chunk progress (`chunk_index` / `chunk_count`);
- отсутствие chunk progress при растущем memory pressure означает, что executor drift'нул обратно
  к giant in-memory behavior и должен считаться regression;
- `reused_prefix_bars >> rewritten_tail_bars` ожидаемо для daily rebuild;
- `reused_prefix_bars = 0` сразу на нескольких стадиях нормально только для bootstrap или
  deterministic full rebuild fallback.

### Operator checklist

Полный bootstrap:

1. Ожидайте `canonical_prices`, затем `hit_times`, затем по очереди
   `current_timeframe=15m`, `30m`, `1h`, ... в canonical order.
2. Ожидайте крупные `rewritten_tail_bars` и почти нулевой reuse.
3. Следите, чтобы не было нескольких одновременно открытых timeframe sessions.

Daily tail rebuild:

1. Ожидайте bounded `rewritten_tail_bars` по `prices`, `mappings`, `signals`, `hit_times`.
2. Ожидайте заметный `reused_prefix_bars`.
3. Если one-off target перешёл в full rebuild fallback, это должно быть видно либо по diagnostics,
   либо по резкому росту `rewritten_tail_bars` только в одной стадии.

Operational note:

- `http://127.0.0.1:9203/metrics` отражает long-running scheduled service;
- manual CLI run не увеличивает scheduler Prometheus counters и наблюдается через shell log/stdout.
- если rebuild падает с `ComputeBudgetExceeded`, это означает, что используется не artifact
  precompute wiring, а обычный public indicators/runtime compute path; для delivered R11 publish
  contract это считается wiring regression.

Operational note:

- manual CLI publishes one explicit symbol root;
- the full enabled+tradable universe is handled by the dedicated scheduled publisher service at
  `03:05 Europe/Moscow`.

Path contract by environment:

- prod uses `/opt/roehub/state/backtest_artifacts/v2`;
- dev/test may keep repo-local `artifacts/backtest/v2`.

## Обязательная publish sequence

Порядок всегда один и тот же:

1. `build inactive slot`
2. `validate whole slot`
3. `atomically switch current.yaml`

Другой порядок запрещён.

## Bootstrap exception

Если для symbol root ещё нет valid `current.yaml` и published slot:

- создать symbol root и два canonical slot roots;
- выбрать `slot_a` как initial bootstrap target;
- выполнить full build для этого symbol root;
- выполнить whole-slot validation;
- создать initial `current.yaml` с `active_slot=slot_a`;
- дальше перейти к обычной steady-state модели `inactive slot -> validate -> switch`.

Ручное создание "пустого active slot" вне publish contract запрещено.

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
- учитывать только `execution_mode in ('background_auto', 'background_manual_legacy')`;
- сравнить persisted `artifact_slot` + `artifact_manifest_hash` с текущей inactive identity;
- если найден хотя бы один active pin, publish блокируется.

Ожидаемое сообщение класса ошибок:

- `inactive_slot_pinned`

Операционное значение:

- inactive slot нельзя rebuild'ить, пока на нём ещё висят активные runs;
- `queued -> cancelled` снимает блокировку сразу, потому что run становится terminal;
- `running` с `cancel_requested_at` всё ещё блокирует rebuild до terminal transition;
- active slot contents и inactive slot contents не должны быть перезаписаны вручную в обход guard.

## Шаг 3. build inactive slot

- пересобрать только inactive slot;
- использовать `artifact_root` из `backtest_artifacts.yaml`, а не default literal в коде;
- писать файлы по explicit deterministic paths;
- не использовать directory scanning как способ discover'ить содержимое;
- не изменять active slot contents.
- canonical `prices/1m` source read выполняется через `market_data.canonical_candles_1m FINAL`
  в columnar precompute path, чтобы bootstrap был устойчив к историческим дублям в ClickHouse.
- R12 canonical execution model for steady-state signal materialization is `timeframe-scoped
  execution`:
  - в canonical scope materialize'ить `canonical_prices` и `hit_times/1m`;
  - затем детерминированно открыть `rolled_prices` ровно для одного target timeframe;
  - открыть один `current_timeframe`;
  - materialize all mappings/signals for that timeframe;
  - eagerly flush `signals/<tf>/<indicator_id>/signals.i8.npy` through `np.memmap`;
  - закрыть timeframe session before opening the next one.

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

R4-01 / R4-02 / R4-03 clarification:

- наличие explicit signal-rules engine contract само по себе ещё не гарантирует publish-ready
  `signals`, но после R4-02 rebuild обязан materialize'ить real signal artifacts для explicit
  targets из `validation_plan.signal_artifacts`;
- rebuild/publish sequence после R4-02 должна ожидать:
  - `signals/<tf>/<indicator_id>/signals.i8.npy`
  - `signals/<tf>/<indicator_id>/manifest.yaml`
  - real root `signals.supported_timeframes`
  - real root `signals.supported_indicator_ids`
  - real root `signals.manifests`
- `signals.v1.params` для signal engine остаются `default-only`, поэтому signal-param grid
  expansion вручную не включается;
- signal matrices обязаны использовать `dtype=int8`, `shape=[V, T_tf]`,
  `axis_order=[variant, time]`, value set `{-1,0,1}`;
- после R4-03 rebuild обязан:
  - брать bounded signal window из `lookback_policy.signal_tail_bars_1m`
  - вычислять effective target tail budget детерминированно для каждого explicit
    `(timeframe, indicator_id)` target
  - сохранять unchanged prefix и rewrite'ить только overlapping tail как
    `prefix + rebuilt_tail`
- missing existing signal target files могут переводить target в deterministic full rebuild, но
  drift в existing manifest/data при reuse attempt обязан останавливать rebuild fail-fast;
- R4-04 runtime `source` integration downstream уже потребляет этот artifact contract через
  runtime defaults, jobs `/top` payloads и `variant-report`;
- R5-01 materialize'ит real `hit_times/1m`, поэтому rebuild теперь должен писать strict
  hit-times arrays и manifest в inactive slot по тем же deterministic paths.

Steady-state rebuild policy after the first successful publish:

- `prices` используют bounded reread/rewrite по
  `lookback_policy.price_tail_bars_1m`;
- `mappings` используют bounded reread/rewrite по
  `lookback_policy.mapping_tail_bars_1m`;
- `signals` используют bounded reread/rewrite по
  `lookback_policy.signal_tail_bars_1m`;
- если `validation_plan.signal_artifacts = all_supported_v1`, rebuild/publish обязан materialize'ить
  весь signal-capable registry для каждого symbol root, а не operator-curated subset;
- `hit_times/1m` используют bounded reread/rewrite по
  `lookback_policy.hit_times_tail_bars_1m`;
- `hit_times/1m` budget выбирается по режиму:
  - bootstrap / `--full-rebuild` -> `validation_budgets.max_hit_times_cells_full_rebuild`
  - steady-state incremental -> `validation_budgets.max_hit_times_cells`
- если existing files/manifest reuse prerequisites нарушены для конкретного stage или symbol root,
  rebuild переключается на deterministic full rebuild для этого symbol root.

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
- `signals.supported_timeframes` и `signals.supported_indicator_ids` совпадают с
  `signals.manifests`;
- `signals.manifests` ordered deterministically by `(timeframe, indicator_id)`;
- `signals/<tf>/<indicator_id>.timeline` совпадает с root `prices/<tf>.coverage`;
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
- publish после R5-01 может включать real `signals` и real `hit_times`, если полный validation
  plan этого требует;
- отдельный R3-04 prices+mappings stage helper по-прежнему обязан оставаться explicit и держать
  `require_hit_times_manifest=false`;
- signal tail rebuild bounded by `lookback_policy.signal_tail_bars_1m`, но operator всё равно
  обязан проверять final merged contracts:
  - `rows_count`
  - `timeline`
  - `signals.manifests`
  - `signals/<tf>/<indicator_id>/signals.i8.npy`
  - `signals/<tf>/<indicator_id>/manifest.yaml`
- `hit_times/1m` должны использовать bounded rebuild по
  `lookback_policy.hit_times_tail_bars_1m`; если reuse невозможен, rebuild переключается в
  deterministic full rebuild для этого symbol root;
- после R11-03 operator diagnostics обязаны читать stage-level rebuild stats из shared publish
  result:
  - `prices.reused_prefix_bars` / `prices.rewritten_tail_bars`
  - `mappings.reused_prefix_bars` / `mappings.rewritten_tail_bars`
  - `signals.reused_prefix_bars` / `signals.rewritten_tail_bars`
  - `hit_times.reused_prefix_bars` / `hit_times.rewritten_tail_bars`
- proof strategy for repeated daily runs is explicit:
  - unchanged prefix arrays must stay byte-stable;
  - only the bounded suffix addressed by `prefix + rebuilt_tail` may change;
  - warmup-heavy signal targets must still match a deterministic full rebuild even when naive
    `effective_target_tail_bars` alone is smaller than the required `warmup`;
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

После switch оператор или scheduled service обязаны проверить observability invariants:

- `backtest_artifact_publish_runs_total{status="succeeded"}` вырос;
- `backtest_artifact_publish_last_success_unixtime` обновился;
- `backtest_artifact_publish_symbols_total{status="failed"}` остаётся `0` или соответствует
  известным degraded symbols;
- `backtest_artifact_publish_blocked_total{reason=~"lock_held|inactive_slot_pinned|validation_failed"}` не растёт неожиданно;
- `backtest_artifact_tail_rebuild_bars_total{stage}` показывает ожидаемый bounded tail profile, а
  не скрытый full rebuild без причины.
- per-stage `reused_prefix_bars` / `rewritten_tail_bars` из publish diagnostics согласованы с этим
  profile и объясняют, почему `hit_times/1m` или long-window `signals.i8.npy` могли перейти в
  full rebuild.

## R6-01 runtime bootstrap checks

После publish оператор должен исходить из одного invariants set:

- sync runtime стартует только из active `current.yaml`;
- background runtime стартует только из persisted pin metadata:
  - `artifact_slot`
  - `slot_generation`
  - `artifact_asof_date`
  - `artifact_manifest_hash`
- оба path обязаны сходиться в один `slot-pinned context`;
- runtime loaders читают только explicit paths из manifests:
  - `prices/<tf>`
  - `signals/<tf>/<indicator_id>/signals.i8.npy`
  - `mappings/<tf>/bar_open_1m_idx.u32.npy`
  - `mappings/<tf>/bar_close_1m_idx.u32.npy`
  - `hit_times/1m/manifest.yaml`
- runtime не должен делать directory scanning и не должен recompute'ить `manifest_sha256` в hot
  path.

Минимальная verification sequence после изменений в loader/context слое:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_resolver_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
```

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

## R10-03 closure checks

После rebuild/publish оператор должен выполнить closure-oriented verification именно в этом
порядке:

1. pointer и manifests:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py
```

2. loaders / slot-pinned bootstrap:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_resolver_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py
```

3. perf closure smoke:

```bash
uv run pytest -q \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

Ожидаемый результат:

- published runtime проходит с `0 CH calls on hot path`;
- published runtime проходит с `0 IndicatorCompute.compute(...) calls on hot path`;
- current slot продолжает быть пригоден и для sync launch, и для `execution_mode in
  ('background_auto', 'background_manual_legacy')`.

Стоп-условия:

- любой validator diagnostic;
- `inactive_slot_pinned`;
- perf smoke failure;
- mismatch между `current.yaml` и root `manifest.yaml`.

При любом из этих stop conditions переходить к
`docs/runbooks/backtest-rollout-rollback.md`, а не выполнять ручные правки slot contents.

## После изменения документации

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
