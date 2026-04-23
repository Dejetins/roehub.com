# Base Refactor Plan: Backtest Engine v2 Milestone / EPIC Map

Этот документ не формулирует новую архитектуру с нуля.
Он раскладывает уже утверждённую архитектуру из
`docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
в исполнимый roadmap по milestone и EPIC'ам, в логическом порядке внедрения.

Если между этим документом и
`docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
есть расхождение, источником истины считается final plan v2.

Референсы:
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/README.md`
- `tests/notebook_tests/06_backtest_compute.ipynb`
- `tests/notebook_tests/05_hit_time_grid.ipynb`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md` (historical / compatibility reference)
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`
- `configs/prod/indicators.yaml`

---

## Зафиксированные решения

- Engine v2 является целевой production-архитектурой backtest.
- bounded context `backtest` сохраняется; v2 внедряется новыми модулями внутри него.
- Приоритет внедрения: сначала sync path, затем background path.
- Artifact store публикуется через два слота: `slot_a` / `slot_b` + pointer file `current.yaml`.
- Published slot не переписывается in-place.
- Если неактивный слот ещё pinned активными background run, новый publish не стартует.
- Production rebuild/publish выполняется отдельным artifact precompute/publish service на Mac
  Studio, а не inline внутри API или `backtest-job-runner`.
- Scheduled service использует `market_data.ref_instruments` как source-of-truth universe и
  ежедневно в `03:05 Europe/Moscow` проходит по всем enabled+tradable trading pairs.
- Сервис обязан публиковать Prometheus metrics и structured logs для publish health/freshness,
  instrument coverage, partial rebuild progress и pin/lock/validation failures.
- Первый publish для symbol root допускает bootstrap full build без существующего
  `current.yaml`; subsequent runs должны использовать bounded incremental rebuild для `prices`,
  `mappings`, `signals`, `hit_times` с deterministic fallback на full rebuild при reuse/config
  drift.
- Backtest request TF ограничены списком:
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
- `1m` и `5m` запрещены как request timeframe.
- `1m` остаётся внутренней базой для:
  - source prices,
  - `1m hit-times`,
  - mappings request TF -> `1m`.
- Risk semantics фиксируются как `1m hit-time only`.
- Signals хранятся как `int8` с кодировкой `{-1, 0, 1}`.
- Layout signal matrices фиксируется как `[V, T_tf]`.
- `signals.v1.params` добавляются в `configs/*/indicators.yaml`, но в initial v2 работают только как default-only.
- `POST /backtests` становится create-and-execute persisted run endpoint.
- Отдельный ручной `Estimate preflight` убирается из пользовательского launch flow.
- Если sync budgets не проходят, но full background budgets проходят, run автоматически переводится в background execution.
- Все пользовательские запуски попадают в `Backtest history`.
- Persisted результат хранится только как summary-only `top N`.
- Trades/report bodies не входят в persisted top results.
- Detail page варианта пересчитывает ровно один вариант лениво, по pinned artifact slot исходного run.
- Физически переиспользуется существующее семейство PG таблиц:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- Пользователь может выбирать несколько `inputs.source` через UI; доступные значения берутся из runtime defaults.

---

## Принцип декомпозиции

Изначальный delivery plan покрывает 11 milestone `R0..R10`.
Post-R10 follow-up work `R11..R12` остаётся additive: оно operationalizes dedicated artifact
publisher и синхронизирует post-R11 execution model без reopening stable runtime/output
contracts.

Базовый план делится на 11 milestone:

1. зафиксировать контракты, baseline docs и benchmark/parity baseline;
2. очистить scope индикаторов и runtime-конфигов;
3. зафиксировать artifact store, manifests и publish semantics;
4. построить precompute pipeline для prices и mappings;
5. построить precompute pipeline для signals;
6. построить `1m hit-times` и перенести алгоритмические kernel-правила из notebook;
7. реализовать runtime kernels v2;
8. обобщить persisted run storage и перевести sync API на новый contract;
9. перевести background execution;
10. перевести web UX;
11. удалить legacy path, закрыть документы и runbooks.

Такой порядок выбран, чтобы:

- сначала зафиксировать все неизменяемые контракты;
- затем убрать из scope то, что делает storage/runtime непрактичными;
- затем построить данные;
- затем подключать runtime к API и UX;
- только после этого удалять legacy.

---

## Порядок внедрения (рекомендуемый)

1. Milestone R0 — Базовые контракты, baseline docs и benchmark baseline.
2. Milestone R1 — Чистка indicator zoo и runtime defaults/config.
3. Milestone R2 — Artifact store contracts + `slot_a/slot_b` publish model.
4. Milestone R3 — Precompute prices и `tf -> 1m` mappings.
5. Milestone R4 — Precompute signals для всего remaining zoo.
6. Milestone R5 — `1m hit-times` + перенос notebook kernel semantics в production docs/tests.
7. Milestone R6 — Runtime kernels v2.
8. Milestone R7 — Persisted run storage generalization + sync API cutover.
9. Milestone R8 — Background execution cutover.
10. Milestone R9 — Web UI/history/detail/strategy-save flows.
11. Milestone R10 — Legacy cleanup, финальная синхронизация docs/runbooks/benchmarks.

Post-R10 additive follow-up:

12. Milestone R11 — Artifact publisher operationalization on Mac Studio
    (bootstrap, scheduler, bounded tail rebuild proofs).
13. Milestone R12 — Docs-first execution-model sync for stage-oriented, timeframe-scoped
    precompute.

Нельзя перепрыгивать через зависимые этапы.
Например:

- нельзя делать history/detail UX до фиксации persisted run storage contract;
- нельзя переводить sync runtime на artifacts до появления manifest/validator/load contracts;
- нельзя запускать background auto-fallback, пока worker не умеет читать pinned artifact slot.

---

## Milestone R0 — Baseline docs, contracts и baseline measurements

Цель: зафиксировать утверждённую архитектуру как source-of-truth, убрать конкурирующие трактовки и подготовить минимальный parity/perf baseline для дальнейших шагов.

### EPIC R0-01 — Зафиксировать source-of-truth документы

**Цель:** синхронизировать roadmap, final plan и связанные backtest docs, чтобы команда работала по одному набору решений.

**Scope:**
- зафиксировать `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md` как final architecture baseline;
- обновить этот `base_refactor_plan.md` до milestone/epic roadmap;
- явно пометить старые предположения в смежных docs как superseded, если они противоречат final plan;
- зафиксировать, что новые milestone не вводят альтернативную архитектуру.

**Non-goals:**
- реализация кода;
- детальный UI copywriting.

**DoD:**
- есть один final architecture doc и один execution roadmap;
- старые conflicting assumptions не остаются неявными.

**Основные документы:**
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R0-02 — Benchmark/parity baseline до начала cutover

**Цель:** зафиксировать измеримую baseline-картину, чтобы далее оценивать реальный эффект рефакторинга.

**Scope:**
- подготовить representative datasets и сценарии запуска для:
  - sync small-run,
  - large-run,
  - background-run;
- зафиксировать baseline perf v1:
  - wall-clock,
  - CPU time,
  - memory footprint,
  - количество CH calls в hot path;
- зафиксировать parity fixtures:
  - Stage A без риска,
  - Stage B legacy close-fill,
  - будущий v2 `1m risk execution`.

**Non-goals:**
- обещание полной parity между v1 и v2 там, где меняется execution semantics;
- оптимизация baseline v1.

**DoD:**
- есть baseline doc/fixtures;
- последующие milestone могут ссылаться на измеряемый baseline.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `tests/notebook_tests/06_backtest_compute.ipynb`
- `tests/notebook_tests/05_hit_time_grid.ipynb`

**Основные пути:**
- `tests/**`
- `src/trading/contexts/backtest/**`

---

### EPIC R0-03 — Зафиксировать runtime/config contracts

**Цель:** на раннем этапе заморозить основные runtime knobs, чтобы следующие milestone не меняли форму API и артефактов.

**Scope:**
- зафиксировать allowed request TF;
- зафиксировать ranking metrics и sortable summary columns;
- зафиксировать `top_n_default` / `top_n_max` как runtime-config knobs;
- зафиксировать `signals.v1.params: default-only`;
- зафиксировать auto-preflight и auto-fallback semantics.

**Non-goals:**
- реализация web UI;
- ввод новых ranking metrics, которых нет в approved final plan.

**DoD:**
- runtime/config contract зафиксирован в docs;
- последующие milestone не переопределяют эти правила.

**Основные документы:**
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`

---

### R0 artifact matrix

| EPIC | Что должно существовать после R0 |
|---|---|
| R0-01 | source-of-truth roadmap + final plan + explicit superseded/status markers в конфликтующих v1 docs |
| R0-02 | `docs/architecture/backtest/README.md` + deterministic benchmark/parity fixtures в `tests/perf_smoke/contexts/backtest/**` |
| R0-03 | frozen runtime/config contract в `configs/<env>/backtest.yaml`, runtime loader, `/backtests/runtime-defaults`, unit tests |

---

## Milestone R1 — Scope cleanup: indicators, timeframes, defaults

Цель: убрать из продукта всё, что делает artifact-backed runtime непрактичным или конфликтует с утверждённой v2-моделью.

### EPIC R1-01 — Полное удаление 11 тяжёлых индикаторов

**Цель:** физически убрать из системы индикаторы, дающие основной combinatorial explosion.

**Scope:**
- удалить 11 индикаторов из:
  - `configs/*/indicators.yaml`,
  - registry/definitions,
  - compute kernels,
  - signal docs,
  - API/UI выбора,
  - тестов,
  - архитектурных документов;
- убедиться, что runtime defaults и API deterministic reject'ят эти indicator_id.

**Non-goals:**
- мягкий режим “не используем, но оставляем в коде”;
- отдельный legacy compatibility layer для удалённых индикаторов.

**DoD:**
- перечисленные indicator_id больше не поддерживаются ни в config, ни в runtime;
- docs и tests не содержат их как valid choice.

**Основные файлы:**
- `configs/prod/indicators.yaml`
- `src/trading/contexts/indicators/domain/definitions/momentum.py`
- `src/trading/contexts/indicators/domain/definitions/trend.py`
- `src/trading/contexts/indicators/domain/definitions/volatility.py`
- `src/trading/contexts/indicators/domain/definitions/volume.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py`
- `src/trading/contexts/indicators/adapters/outbound/registry/yaml_indicator_registry.py`
- `src/trading/contexts/indicators/domain/definitions/__init__.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

**Основные документы:**
- `docs/architecture/indicators/indicators_formula.yaml`
- `docs/architecture/indicators/indicators-trend.md`
- `docs/architecture/indicators/indicators-momentum.md`
- `docs/architecture/indicators/indicators-volatility.md`
- `docs/architecture/indicators/indicators-volume.md`

---

### EPIC R1-02 — `signals.v1.params` schema и default-only enforcement

**Цель:** привести config/schema к новой signal-модели без signal-grid explosion.

**Scope:**
- добавить `signals.v1.params` в `configs/*/indicators.yaml` для поддержанных indicator_id;
- зафиксировать default values;
- обновить defaults provider и validators так, чтобы non-default signal params deterministic reject'ились;
- синхронизировать signal docs с новым config shape.

**Non-goals:**
- full signal params grid;
- поддержка произвольных signal-range overrides в initial v2.

**DoD:**
- `signals.v1.params` присутствуют в schema/config;
- request с non-default signal params детерминированно отклоняется.

**Основные файлы:**
- `configs/prod/indicators.yaml`
- `src/trading/contexts/backtest/adapters/outbound/defaults/indicators_yaml_defaults_provider.py`

**Основные документы:**
- `docs/architecture/backtest/README.md`

---

### EPIC R1-03 — Ограничение request TF и runtime defaults

**Цель:** зафиксировать продуктовый/технический contract для request timeframes, ranking metrics, source choices и configurable `top N`.

**Scope:**
- исключить `1m` и `5m` из backtest request defaults и UI defaults;
- зафиксировать allowed request TF;
- добавить/обновить runtime defaults contract для:
  - ranking metrics,
  - sortable columns,
  - `top_n_default`,
  - `top_n_max`,
  - allowed `inputs.source` per indicator;
- убедиться, что API/runtime валидируют запрещённые TF.

**Non-goals:**
- изменение market-data ingestion;
- поддержка UI hardcoded options.

**DoD:**
- runtime defaults полностью описывают launch form;
- запрещённые TF deterministic reject'ятся на backend.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`

---

## Milestone R2 — Artifact store contracts: slots, manifests, validators

Цель: зафиксировать filesystem layout и publish semantics, чтобы и precompute, и runtime читали один и тот же контракт.

### EPIC R2-01 — Path builder и slot layout

**Цель:** ввести детерминированную координатную систему для artifact store.

**Scope:**
- путь строится по координатам:
  - `exchange`
  - `market_type`
  - `symbol`
- внутри symbol root:
  - `current.yaml`
  - `slot_a/`
  - `slot_b/`
- внутри слота:
  - `manifest.yaml`
  - `prices/<tf>/...`
  - `signals/<tf>/<indicator_id>/...`
  - `mappings/<tf>/...`
  - `hit_times/1m/...`
- реализовать path builder и loader contract без directory scanning как обязательного шага hot path.

**Non-goals:**
- хранение ежедневной истории slot version на диске;
- archive/compression policy.

**DoD:**
- layout и path contracts описаны и реализуемы без неоднозначности.

**Основные пути:**
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/**`
- `src/trading/contexts/backtest/application/services/v2/**`

**Основные документы:**
- `docs/architecture/backtest/README.md`

---

### EPIC R2-02 — `current.yaml`, slot publishing и pinning contract

**Цель:** обеспечить безопасную публикацию новых артефактов без in-place mutations.

**Scope:**
- зафиксировать `current.yaml` contract:
  - `active_slot`
  - `slot_generation`
  - `asof_date`
  - `manifest_sha256`
  - `published_at_utc`
- зафиксировать publish sequence:
  1. build inactive slot
  2. validate whole slot
  3. atomically switch `current.yaml`
- зафиксировать правило:
  - если inactive slot ещё pinned активным background run, publish не стартует;
- зафиксировать slot pinning metadata, которую runtime/job storage должен хранить.

**Non-goals:**
- retention policy более чем на два слота;
- автоматическая очистка historical snapshots, которых здесь нет по дизайну.

**DoD:**
- publish semantics описаны без двусмысленности;
- background run имеет стабильную slot identity.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`

---

### EPIC R2-03 — Manifest schemas и validators

**Цель:** ввести fail-fast проверки целостности слота.

**Scope:**
- root `manifest.yaml`;
- per-indicator signal `manifest.yaml`;
- `hit_times/1m/manifest.yaml`;
- validators на:
  - dtype
  - shape
  - monotonic time arrays
  - signal value set `{-1,0,1}`
  - axis order contract
  - mapping bounds
  - hit-time monotonicity
  - hash/provenance fields

**Non-goals:**
- expensive full-file hashing на hot path runtime;
- dynamic schema discovery в runtime.

**DoD:**
- слот можно целиком провалидировать перед publish;
- runtime получает fixed metadata, а не вычисляет её на месте.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R2-04 — Artifact runtime/precompute config

**Цель:** отделить artifact pipeline configuration от общего runtime request config.

**Scope:**
- завести/обновить `configs/<env>/backtest_artifacts.yaml`;
- зафиксировать в нём:
  - artifact root
  - allowed artifact TF
  - TP/SL grid для `1m hit-times`
  - slot policy
  - publish schedule
  - lookback policy
  - validation budgets
- `publish_schedule` должен быть executable operational contract, а не narrative-only заметкой:
  initial production schedule фиксируется как daily `03:05 Europe/Moscow`;
- загрузчик/валидатор должен fail-fast валидировать эти поля.

**Non-goals:**
- перенос всех backtest runtime knobs в `backtest_artifacts.yaml`;
- скрытая магия через env без явного config contract.

**DoD:**
- artifact pipeline настраивается отдельным конфигом;
- config shape документирован и валидируется.

**Основные файлы:**
- `configs/dev/backtest_artifacts.yaml`
- `configs/test/backtest_artifacts.yaml`
- `configs/prod/backtest_artifacts.yaml`

---

## Milestone R3 — Precompute prices и mappings

Цель: материализовать price arrays для всех разрешённых TF и связи request TF -> `1m`, не трогая runtime hot path.

### EPIC R3-01 — Canonical `1m` export в inactive slot

**Цель:** построить источник правды для всего precompute pipeline.

**Scope:**
- читать `market_data.canonical_candles_1m` через существующие reader/ACL;
- материализовать:
  - `prices/1m/open_time.i64.npy`
  - `prices/1m/close_time.i64.npy`
  - `prices/1m/ohlcv.f32.npy`
- поддержать tail update c lookback там, где это нужно pipeline;
- не смешивать timestamps и float OHLCV в одном homogeneous массиве.

**Non-goals:**
- использование CH в runtime;
- хранение `prices_1m_columns.npy` legacy-формата как нового стандарта.

**DoD:**
- inactive slot умеет получить корректный `1m` price base;
- root manifest отражает coverage.

**Основные файлы:**
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py`
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py`

---

### EPIC R3-02 — Rollup allowed TF prices

**Цель:** заранее материализовать цены всех разрешённых request TF.

**Scope:**
- строить из `1m` только разрешённые TF:
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
- для каждого TF сохранять:
  - `open_time.i64.npy`
  - `close_time.i64.npy`
  - `ohlcv.f32.npy`
- валидировать rollup coverage и boundary alignment.

**Non-goals:**
- runtime rollup при backtest request;
- хранение `1m`/`5m` как request-level runtime datasets.

**DoD:**
- runtime может загружать prices нужного TF без CH и без rollup.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R3-03 — Build `tf -> 1m` mappings

**Цель:** быстро переводить bars request timeframe в minute execution space.

**Scope:**
- для каждого разрешённого request TF строить:
  - `bar_open_1m_idx.u32.npy`
  - `bar_close_1m_idx.u32.npy`
- валидировать:
  - bounds
  - monotonicity
  - соответствие price arrays

**Non-goals:**
- runtime binary search по timestamps на каждом trade;
- отдельные mappings для неразрешённых TF.

**DoD:**
- runtime Stage B может перейти от request bar к `1m` за O(1).

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R3-04 — Slot build/publish flow для prices+mappings

**Цель:** довести price stage до publish-ready формы.

**Scope:**
- build в inactive slot;
- validation before publish;
- запись root manifest;
- атомарный switch `current.yaml`;
- smoke tests на loader/publisher.

**Non-goals:**
- сигналы;
- hit-times.

**DoD:**
- можно опубликовать слот, содержащий только validated prices+mappings stage.
- bootstrap path для symbol root без `current.yaml` описан отдельно и не требует ручного
  "создания пустого active slot" вне artifact contract.

**Основные пути:**
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py`

---

## Milestone R4 — Precompute signals

Цель: ежедневно материализовать дискретные сигналовые матрицы для всего remaining zoo индикаторов на всех разрешённых TF.

### EPIC R4-01 — Signal rules engine v2-aligned

**Цель:** привести production signal rule application к contract из `backtest-signals-from-indicators-v1.md`.

**Scope:**
- использовать signal families и semantics из docs;
- поддержать `inputs.source` как явный axis;
- поддержать `signals.v1.params` только в default-only режиме;
- не хранить raw indicator values как artifact.

**Non-goals:**
- signal-grid expansion beyond defaults;
- runtime signal derivation на hot path.

**DoD:**
- любой поддержанный indicator output превращается в `{-1,0,1}` по фиксированным правилам.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/indicators/indicators_formula.yaml`

---

### EPIC R4-02 — Export `signals.i8.npy` per timeframe + indicator

**Цель:** получить mmap-friendly signal artifacts в layout `[V, T_tf]`.

**Scope:**
- для каждого `timeframe + indicator_id` сохранять отдельный `signals.i8.npy`;
- shape фиксируется как `[V, T_tf]`;
- хранить per-indicator `manifest.yaml` с:
  - `indicator_id`
  - `timeframe`
  - `grid`
  - deterministic axis order
  - rows count
  - timeline coverage
  - sha/dtype/shape

**Non-goals:**
- один общий giant signal file на весь instrument;
- shape `[T, V]`.

**DoD:**
- runtime может загрузить ровно нужные row ranges без полного чтения всех variants.

**Основные пути:**
- `src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py`
- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/**`

---

### EPIC R4-03 — Tail update logic для signals

**Цель:** поддержать ежедневное обновление signals без полного пересчёта всей истории.

**Scope:**
- для каждого `indicator_id + timeframe` хранить и использовать lookback policy;
- пересчитывать хвост на достаточной длине для warmup/lag semantics;
- перезаписывать пересекающийся tail segment в inactive slot;
- обновлять signal manifest coverage metadata.

**Non-goals:**
- in-place mutation active slot;
- обязательный full historical rebuild на каждый daily update.

**DoD:**
- daily signal rebuild ограничен хвостом и валидно покрывает warmup semantics.

**Основные документы:**
- `docs/architecture/backtest/README.md`

---

### EPIC R4-04 — Source axis coverage и runtime defaults integration

**Цель:** сделать `inputs.source` частью и precompute coverage, и launch UX.

**Scope:**
- гарантировать, что signal row order учитывает `inputs.source`;
- runtime defaults должны возвращать per-indicator allowed source values;
- summary/detail payloads должны уметь сохранять explicit source selection как часть variant payload.

**Non-goals:**
- hardcoded UI списки `close/open/...`;
- implicit default source без явного payload.

**DoD:**
- source selection проходит сквозь config -> precompute -> runtime -> persisted summary rows -> detail page.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

---

## Milestone R5 — `1m hit-times` и notebook-algorithm extraction

Цель: вынести из research notebooks production-ready risk primitives и зафиксировать их как runtime contract.

### EPIC R5-01 — Перенос `1m hit-time` grid compute из notebook

**Цель:** материализовать единый `1m hit-times` слой для всех backtest run.

**Scope:**
- использовать notebook ideas из `tests/notebook_tests/05_hit_time_grid.ipynb`;
- строить:
  - `tp_values.f32.npy`
  - `sl_values.f32.npy`
  - `long_tp.u32.npy`
  - `long_sl.u32.npy`
  - `short_tp.u32.npy`
  - `short_sl.u32.npy`
- sentinel и monotonicity semantics должны быть явно описаны и валидированы.

**Non-goals:**
- отдельные hit-times для каждого request TF;
- runtime расчёт hit-times на лету.

**DoD:**
- published slot содержит валидный `hit_times/1m`;
- runtime может использовать его без дополнительной подготовки.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R5-02 — Документировать перенос алгоритма из `06_backtest_compute.ipynb`

**Цель:** чётко зафиксировать, что именно переносится из notebook в production kernels.

**Scope:**
- описать:
  - signal aggregation
  - compact trade list
  - `1m` risk exits
  - fast TP/SL grid search
  - exact replay best TP/SL cell
  - metric calculation over compact trades
- отдельно зафиксировать, что notebook не переносится как literal orchestration script;
- выделить generic kernels от notebook-specific research details.

**Non-goals:**
- порт notebook как есть;
- перенос notebook pair-specific эвристик в production runtime без отдельного решения.

**DoD:**
- docs описывают new algorithm понятно и без противоречий.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

---

### EPIC R5-03 — Golden fixtures для новой execution semantics

**Цель:** закрепить корректность Stage B в новых правилах `signal_tf + 1m risk`.

**Scope:**
- подготовить golden fixtures на:
  - entry mapping request TF -> `1m`
  - TP/SL earliest hit
  - earliest signal-exit mapping
  - tie-break rules
  - exact best-cell replay
- фиксировать expected metrics для representative cases.

**Non-goals:**
- обещание полного совпадения с legacy close-fill;
- fuzzy manual notebook comparison без формальных expected outputs.

**DoD:**
- Stage B kernels можно тестировать deterministic fixtures.

**Основные пути:**
- `tests/**`
- `src/trading/contexts/backtest/application/services/v2/**`

---

## Milestone R6 — Runtime kernels v2

Цель: реализовать сам artifact-backed runtime без CH/IndicatorCompute в hot path.

### EPIC R6-01 — Artifact loaders и slot-pinned runtime context

**Цель:** построить слой загрузки артефактов, пригодный для sync и background run.

**Scope:**
- resolver `current.yaml`;
- loader root manifest;
- loaders:
  - price arrays
  - signal matrices
  - mappings
  - `1m hit-times`
- runtime context должен pin'ить:
  - `artifact_slot`
  - `slot_generation`
  - `artifact_asof_date`
  - `artifact_manifest_hash`

**Non-goals:**
- runtime filesystem scanning;
- hash recomputation на hot path.

**DoD:**
- sync/background run стартуют с одинакового slot-pinned context.

**Основные файлы:**
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py`
- `src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py`

---

### EPIC R6-02 — Stage A kernels

**Цель:** заменить variant-by-variant Python loop на batch-oriented signal aggregation и trade compaction.

**Scope:**
- загрузка subset row ranges из signal matrices;
- сборка final signal на request timeframe;
- compact trade list без risk exits;
- exact no-risk metrics для shortlist/ranking stage;
- chunked variant processing.

**Non-goals:**
- materialize trades/report для всех variants;
- bar-by-bar legacy replay.

**DoD:**
- Stage A работает по artifacts-only inputs и даёт deterministic shortlist.

**Основные файлы:**
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`

---

### EPIC R6-03 — Stage B kernels

**Цель:** реализовать новую risk execution semantics через `1m hit-times`.

**Scope:**
- mapping request timeframe entries в `1m`;
- earliest TP/SL hits через precomputed hit-time tables;
- signal-exit mapping request TF -> `1m`;
- actual exit = earliest of TP/SL/signal-exit;
- fast search по TP/SL grid;
- exact replay only для best cell.

**Non-goals:**
- old close-fill engine;
- runtime compute of TP/SL hit-times.

**DoD:**
- Stage B не использует legacy execution engine;
- notebooks-derived algorithm rules реализуемы в production kernels.

**Основные файлы:**
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`

---

### EPIC R6-04 — Ranking metrics и top-N materialization

**Цель:** завершить runtime layer продуктовым ranking contract.

**Scope:**
- поддержать ranking metric selection:
  - `total_return_pct`
  - `max_drawdown_pct`
  - `return_over_max_drawdown`
  - `profit_factor`
  - `sharpe_trades`
  - `win_rate_pct`
- materialize только summary rows top-N;
- не materialize trades/report bodies в runtime summary result;
- добавить deterministic tie-break.

**Non-goals:**
- recompute top-N при локальной UI sorting;
- отдельный persisted detail result.

**DoD:**
- runtime выдаёт summary-only table top-N, пригодную для persisted storage.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

## Milestone R7 — Persisted run storage и sync API cutover

Цель: перевести `POST /backtests` на новую persisted run модель и общий storage contract.

### EPIC R7-01 — Generalize PG jobs storage into persisted run storage

**Цель:** переиспользовать существующие job tables как единое хранилище всех backtest run.

**Scope:**
- логически обобщить:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- добавить/зафиксировать поля:
  - `execution_mode`
  - `market_id`
  - `symbol`
  - `timeframe`
  - `requested_top_n`
  - `ranking_primary_metric`
  - `ranking_secondary_metric`
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_manifest_hash`
- top rows хранят:
  - `rank`
  - `variant_key`
  - `variant_index`
  - `payload_json`
  - `summary_metrics_json`
  - `best_tp_pct`
  - `best_sl_pct`
- `report_table_md` и `trades_json` удаляются из persisted results contract или оставляются постоянно `NULL` на переходный период.

**Non-goals:**
- второй parallel persistence stack для sync results;
- persisted detail result storage.

**DoD:**
- inline и background run используют одно и то же storage семейство.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

**Основные пути:**
- `alembic/versions/*`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/**`
- `src/trading/contexts/backtest/application/ports/**`

---

### EPIC R7-02 — `POST /backtests` becomes create-and-execute persisted run

**Цель:** убрать ephemeral sync run и встроить auto-preflight в единый run flow.

**Scope:**
- `POST /backtests`:
  - всегда делает internal preflight;
  - всегда создаёт persisted run record;
  - если sync budgets проходят, исполняет run inline;
  - если full budgets не проходят, возвращает deterministic `422`;
- response metadata должна включать:
  - `run_id`
  - `state`
  - `execution_mode`
  - `engine_version`
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_asof_date`
  - `artifact_manifest_hash`

**Non-goals:**
- отдельная пользовательская кнопка preflight;
- сохранение trades/report bodies в sync response как persisted contract.

**DoD:**
- sync launch flow соответствует утверждённому persisted run contract.

**Основные файлы:**
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `apps/api/routes/**`
- `apps/api/wiring/modules/**`

**Основные документы:**
- `docs/architecture/backtest/README.md`

---

### EPIC R7-03 — History/top/detail runtime API

**Цель:** ввести единый public contract поверх persisted run storage.

**Scope:**
- endpoints:
  - `GET /backtests/runs`
  - `GET /backtests/runs/{run_id}`
  - `GET /backtests/runs/{run_id}/top`
  - `POST /backtests/runs/{run_id}/cancel`
- runtime defaults endpoint должен отдавать:
  - allowed TF
  - supported indicator ids
  - `signals.v1.params: default-only`
  - ranking metrics
  - sortable summary columns
  - `top_n_default`
  - `top_n_max`
  - available `inputs.source`

**Non-goals:**
- public UX через `/backtests/jobs*` как финальный контракт;
- realtime transport beyond polling.

**DoD:**
- history и run retrieval API покрывают весь product flow summary-level результатов.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R7-04 — Lazy single-variant detail API

**Цель:** отделить summary-level persisted results от тяжёлого detail расчёта.

**Scope:**
- detail endpoint/page flow получает:
  - persisted `run_id`
  - variant identity/payload;
- backend заново считает один вариант по:
  - pinned artifact slot исходного run
  - explicit variant params
  - original range/request semantics
- отдаёт:
  - detailed stats
  - equity/price chart series
  - trades list
  - графическую разметку сделок

**Non-goals:**
- persisted storage detail results;
- full top-N recompute при открытии detail page.

**DoD:**
- detail flow существует отдельно от summary persistence.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

---

## Milestone R8 — Background execution cutover

Цель: перевести worker и auto-fallback semantics на тот же artifact-backed runtime.

### EPIC R8-01 — Worker uses slot-pinned runtime v2

**Цель:** сделать background run execution воспроизводимым и идентичным sync path по алгоритму.

**Scope:**
- worker использует тот же runtime facade, что и sync;
- pinned fields сохраняются в job/run record;
- worker читает artifacts только через slot-pinned identity;
- в hot path worker нет CH и `IndicatorCompute.compute(...)`.

**Non-goals:**
- separate worker-only runtime;
- legacy Stage A/B loop в v2 worker path.

**DoD:**
- background run и sync run используют один и тот же runtime contract.

**Основные файлы:**
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `apps/worker/backtest_job_runner/**`

**Основные документы:**
- `docs/architecture/backtest/README.md` — canonical worker contract
- `docs/architecture/backtest/README.md` — historical / compatibility context

---

### EPIC R8-02 — Auto fallback `sync -> background`

**Цель:** завершить единый launch contract, где пользователь не выбирает режим вручную при превышении sync budgets.

**Scope:**
- если sync budgets fail, но full budgets pass:
  - API создаёт background run;
  - response явно показывает `execution_mode=background_auto`;
  - history entry появляется сразу;
- cancel/status/progress semantics едины для всех background run.

**Non-goals:**
- hidden fallback без явной индикации в API/UI;
- ручное дублирование launch flow отдельным endpoints contract как product default.

**DoD:**
- auto-fallback работает end-to-end и не выглядит как неявная магия.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`

---

### EPIC R8-03 — Background safety, cancel и slot publish interaction

**Цель:** закрепить эксплуатационные гарантии при параллельной работе publish и background run.

**Scope:**
- active background run pin'ит slot;
- publish блокируется, если inactive slot занят pinned run;
- cancel остаётся best-effort, но не нарушает slot safety;
- progress/history list корректно отражают `queued/running/succeeded/failed/cancelled`.

**Non-goals:**
- точный cursor-resume Stage B beyond already approved semantics;
- многопоточная обработка одного run несколькими worker'ами.

**DoD:**
- long-running background run не ломается из-за publish cycle;
- publish policy согласована с worker behavior.

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`

---

## Milestone R9 — Web UI: launch, history, summary table, detail page

Цель: привести web UX к новой persisted run модели и убрать старую ручную preflight-схему.

### EPIC R9-01 — Launch form v2

**Цель:** обновить страницу запуска backtest под новый backend contract.

**Scope:**
- убрать обязательный manual `Estimate preflight`;
- форма выбирает:
  - instrument
  - timeframe
  - indicators
  - multiple `inputs.source`
  - ranking metric
  - desired `top N` в пределах runtime config
- форма работает через new runtime defaults contract.

**Non-goals:**
- отдельная user-facing кнопка preflight;
- hardcoded indicator source lists.

**DoD:**
- launch form соответствует backend contract и не требует manual preflight step.

**Основные документы:**
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

**Основные пути:**
- `apps/web/**`

---

### EPIC R9-02 — Summary table и Backtest history

**Цель:** дать пользователю единый способ открывать старые run без пересчёта grid.

**Scope:**
- вкладка `Backtest history`;
- run summary page;
- одна summary table:
  - `top N`
  - локальная пересортировка по summary columns
  - без trades внутри таблицы
- history entries должны показывать:
  - status
  - execution mode
  - key metadata
  - возможность открыть old run.

**Non-goals:**
- сохранение trades/report bodies внутри history rows;
- новый расчёт top-N при каждой сортировке колонок.

**DoD:**
- пользователь может открыть старый run и увидеть persisted summary-only results.

**Основные документы:**
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/architecture/backtest/README.md`

---

### EPIC R9-03 — Variant detail page + save variant via existing strategy flow

**Цель:** отделить summary UX от тяжелого per-variant detail и не вводить лишнее storage.

**Scope:**
- отдельная страница варианта;
- ленивый one-variant recompute;
- показ:
  - chart
  - trades
  - detailed metrics
- save from summary/detail через existing Strategy persistence flow, а не через новый отдельный “favorites storage” в этом milestone.

**Non-goals:**
- persisted detail result;
- новая отдельная БД-сущность избранного, не зафиксированная в текущей документации.

**DoD:**
- variant detail page работает без пересчёта всей grid;
- save variant reuses existing strategy flow.

**Основные документы:**
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`

---

## Milestone R10 — Legacy cleanup, финальные docs, tests и runbooks

Цель: завершить миграцию, убрать противоречивые legacy path и довести документацию до консистентного состояния.

### EPIC R10-01 — Убрать legacy hot path из production usage

**Цель:** исключить runtime-зависимости, от которых v2 должен был избавить систему.

**Scope:**
- убрать/закрыть production path через legacy candle/scoring/execution pipeline;
- сохранить `grid_builder_v1.py` только как grid-expansion слой там, где он ещё нужен;
- оставить controlled legacy fallback только на переходный период, если это явно нужно rollout'у;
- затем удалить fallback.

**Non-goals:**
- бесконечное сосуществование двух production engines;
- скрытый silent fallback на v1.

**DoD:**
- v2 является единственным production path для покрытого scope.

**Основные файлы:**
- `src/trading/contexts/backtest/application/services/grid_builder_v1.py`
- artifact-backed runtime modules в `src/trading/contexts/backtest_artifacts/application/services/v2/`

---

### EPIC R10-02 — Финальная синхронизация docs

**Цель:** обновить весь связанный doc set под реальную v2-систему.

**Closure note (R10-02):**
- canonical v2 docs должны описывать active artifact-backed runtime, runs-first UX и
  `summary-only` persisted results без параллельных трактовок;
- legacy `/backtests/jobs*` и `POST /api/backtests/variant-report` допускаются только как
  `compatibility alias`;
- после R10-02 в handoff остаётся только R10-03 perf/runbook closure, а не дополнительный
  runtime cutover.

**Scope:**
- создать:
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
  - `docs/runbooks/backtest-artifacts-rebuild.md`
- обновить:
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md` (historical / compatibility reference)
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
  - `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`
  - `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

**Non-goals:**
- оставлять v1 docs без пометки superseded там, где контракт уже изменён;
- поддерживать несколько разных описаний одной и той же runtime semantics.

**DoD:**
- docs не противоречат фактической реализации и друг другу.

---

### EPIC R10-03 — Tests, perf gates и runbooks

**Цель:** закрыть миграцию не только кодом, но и проверками/эксплуатацией.

**Closure note (R10-03 handoff):**
- сюда входят perf gate closure, benchmark expansion и дополнительные runbooks;
- сюда не входят новые изменения публичного runtime/API surface, уже зафиксированного в R10-02.

**Scope:**
- unit/integration/golden tests для:
  - artifact validators
  - slot publish/pinning
  - signal loaders
  - Stage A/Stage B kernels
  - persisted run storage
  - history/detail API
- perf gates:
  - `0` CH calls на hot path
  - `0` `IndicatorCompute.compute(...)` calls на hot path
  - измеримый speedup против baseline
- runbooks:
  - artifact rebuild
  - background run troubleshooting
  - rollout / rollback guidance
- observability closure:
  - метрики daily artifact service
  - freshness/last-success checks
  - instrument coverage и skipped/failed symbol counters
  - pin-block / lock-contention / validation-failure counters

**Non-goals:**
- абстрактная “надеемся, что быстро” без perf measurements;
- перенос operational knowledge только в головы команды.

**DoD:**
- migration считается завершённой только после прохождения test/perf/runbook closure.

**Closure matrix (R10-03):**

| Scope item | Canonical closure evidence |
|---|---|
| artifact validators / slot publish / pinning | `tests/unit/contexts/backtest/application/services/v2/test_artifact_manifest_validator_v2.py`, `tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_publisher_v2.py` |
| signal/mapping/hit-times loaders | `tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py`, `tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py` |
| Stage A / Stage B kernels | `tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py`, `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py`, `tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py`, `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py` |
| persisted run storage / history/detail compatibility | `tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py`, `tests/unit/apps/api/test_backtests_routes.py` |
| perf gates and measurable speedup reference | `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py` |
| artifact rebuild / background troubleshooting / rollout-rollback | `docs/runbooks/backtest-artifacts-rebuild.md`, `docs/runbooks/backtest-job-runner.md`, `docs/runbooks/backtest-rollout-rollback.md` |

**Основные документы:**
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`

---

## Milestone R12 — Post-R11 precompute execution-model sync

Цель: documentation-first закрепить корректную post-R11 архитектуру artifact precompute без
изменения artifact output contract.

### EPIC R12-00 — Stage-oriented, timeframe-scoped precompute docs

**Цель:** убрать implicit tensor-first narrative и заменить его на operator-friendly execution
model, который follow-up code epics смогут реализовать без догадок.

**Scope:**
- явно разделить:
  - stable artifact layout/manifests/public runtime contracts;
  - changed offline precompute execution model;
  - unchanged public/runtime `indicators` compute semantics;
- зафиксировать pipeline в порядке:
  - load canonical `1m` once
  - materialize `prices/1m`
  - build `hit_times/1m`
  - derive one target TF `rolled_prices` at a time
  - open one timeframe session at a time
  - build `mappings/<tf>` и `signals/<tf>/<indicator_id>` for that timeframe
  - close timeframe session
  - finalize manifests and publish `current.yaml`;
- документировать strict `execution_policy` contract для
  `configs/<env>/backtest_artifacts.yaml`;
- документировать `ChunkPlanner`, memory ownership, worker model и progress observability для Mac
  Studio.

**Non-goals:**
- менять layout `artifacts/backtest/v2`;
- менять manifest schemas;
- менять public `/backtests*` или `/indicators*` contracts;
- возвращать giant in-memory dense tensor model как recommended future state.

**DoD:**
- roadmap, canonical architecture docs и runbooks описывают одну и ту же stage-oriented
  `timeframe-scoped execution` model;
- follow-up implementation получает strict contracts для `execution_policy` и `ChunkPlanner`;
- operator docs объясняют, как читать `current_timeframe`, stage progress и chunk progress на
  длинном bootstrap run.

**Основные документы:**
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`

---

## Что не входит в initial v2

Ниже перечислено то, что в этом roadmap не считается частью обязательного initial scope:

- full signal-grid expansion для `signals.v1.params`;
- возврат удалённых тяжёлых индикаторов;
- request TF `1m` и `5m`;
- хранение ежедневной длинной истории artifact snapshots поверх `slot_a/slot_b`;
- persisted storage для detail page trades/equity;
- новый отдельный persistence layer для “favorites”, не опирающийся на уже существующий strategy flow;
- архивирование/сжатие active runtime artifacts как обязательная часть initial v2;
- generic platform jobs framework вне контекста backtest.

---

## Итоговая логика исполнения плана

План должен выполняться не “по файлам”, а по цепочке зависимостей:

1. сначала заморозить решения и метрики успеха;
2. затем вычистить scope;
3. затем построить artifact contracts;
4. затем построить сами artifacts;
5. затем реализовать runtime;
6. затем перевести storage/API;
7. затем перевести worker и web UX;
8. только потом удалять legacy.

Именно в таком порядке milestone не конфликтуют друг с другом и не требуют постоянного пересмотра уже принятых решений.
