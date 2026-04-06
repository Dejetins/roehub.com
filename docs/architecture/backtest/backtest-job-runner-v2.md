# Backtest Job Runner v2 -- claimed background worker для persisted runs

## Status

- Статус: canonical architecture document для фактического `backtest-job-runner` поверх
  backtest v2 runtime.
- Этот документ является основным source of truth для background execution path persisted runs.
- Документ `docs/architecture/backtest/backtest-job-runner-worker-v1.md` считается
  historical/compatibility document и не является больше canonical описанием worker-а.
- Публичный lifecycle vocabulary и persisted run storage остаются совместимыми с runs/history
  contracts, а runtime internals работают через artifact-backed v2.

## Цель

`backtest-job-runner` это отдельный долгоживущий worker-процесс, который забирает queued
persisted runs из Postgres и исполняет их через общий backtest v2 runtime.

Worker существует для того, чтобы:

- тяжёлые, но валидные, запуски могли автоматически уходить в `background_auto`;
- такие запуски оставались видимыми в `Backtest history` как persisted runs;
- API и UI не блокировались тяжёлым background execution;
- sync path и background path использовали одну общую runtime orchestration surface и один
  canonical exact scorer.

## Scope

Этот документ покрывает:

- роль worker-а в backtest v2 architecture;
- startup и fail-fast wiring;
- claim, lease, heartbeat, cancel и reclaim semantics;
- slot-pinned artifact bootstrap для claimed runs;
- shared runtime planning и execution profile contract;
- persisted progress и summary-only results contract;
- observability;
- production deployment contract;
- compatibility boundaries.

Этот документ не покрывает:

- artifact rebuild/publish pipeline;
- sync-inline response assembly;
- lazy variant detail/trades recomputation;
- browser UI rendering, кроме тех частей, которые завязаны на persisted progress contract;
- детальную rollout matrix adaptive selector policy.

## Основные архитектурные решения

### 1. `backtest-job-runner` не является отдельным backtest engine

`backtest-job-runner` это background execution host для того же canonical v2 runtime, который
используется sync path.

Worker не должен:

- создавать второй orchestration surface;
- вводить отдельный scoring engine;
- иметь собственную policy matrix выбора runtime profile;
- иметь отдельный источник истины для progress semantics или rollout semantics.

### 2. Один claimed background path

Canonical background launch mode для новой системы это `background_auto`.

`background_manual_legacy` сохраняется только как compatibility-only literal:

- новые запуски не должны создаваться с `background_manual_legacy`;
- worker обязан продолжать корректно исполнять уже существующие persisted rows с этим literal;
- в новой продуктовой и архитектурной документации основной background path описывается через
  `background_auto`.

### 3. Один worker process = один claim loop = одна claimed job одновременно

Один экземпляр `backtest-job-runner`:

- является одним долгоживущим процессом;
- держит один последовательный claim loop;
- в каждый момент времени обрабатывает не более одной claimed job.

Горизонтальное масштабирование очереди достигается не через усложнение claim loop, а через
несколько независимых worker processes.

### 4. Queue concurrency и runtime parallelism это разные вещи

Параллельность обработки нескольких независимых background runs должна задаваться отдельно от
внутренней вычислительной параллельности одного run.

Для этого canonical operational knob это:

- `backtest.jobs.worker_processes`

Этот параметр задаёт число независимых worker processes, которые одновременно claim-ят
background jobs.

Важно:

- `worker_processes` отвечает за queue concurrency;
- runtime parallelism одного run остаётся responsibility shared v2 runtime и resolved
  `ExecutionProfile`;
- нельзя смешивать queue concurrency knob и intra-run compute knob в один и тот же параметр.

### 5. Service-manager-agnostic архитектура

Этот документ описывает service contract worker-а, а не конкретный supervisor.

Архитектурный контракт требует, чтобы worker был:

- долгоживущим supervised service;
- автоматически перезапускаемым при падении;
- масштабируемым до `N = worker_processes` экземпляров;
- оснащённым уникальной instance identity;
- оснащённым logs и metrics.

Конкретная реализация через `launchd`, `systemd`, контейнеры или другой supervisor описывается
в runbook/ops документации, но не в этом архитектурном документе.

## Роль worker-а в системе

В production-системе существует одна canonical persisted run storage family и одна runtime
orchestration surface:

- API создаёт persisted run record;
- sync-compatible runs могут завершаться inline;
- heavy-but-valid runs классифицируются в `background_auto`;
- `backtest-job-runner` claim-ит queued persisted runs и исполняет их;
- runs history/detail API читает то же самое persisted storage;
- progress, ETA и top snapshots видны через единый persisted run contract.

Worker является частью production backtest contract. Это не факультативный вспомогательный
процесс.

## Startup и fail-fast wiring

На старте worker обязан:

- загрузить runtime config из `backtest.yaml`;
- загрузить artifact runtime config;
- потребовать `STRATEGY_PG_DSN`;
- собрать Postgres repositories для jobs, leases и results;
- собрать canonical request decoder для persisted `request_json`;
- собрать defaults provider и compatibility-only estimate helpers;
- собрать artifact loader и `ArtifactSlotResolverV2`;
- собрать `BacktestArtifactRuntimePlannerV2` из startup-loaded `execution_profiles` и
  `adaptive_selector_policy`;
- поднять metrics endpoint;
- проверить, что worker может войти в claim loop без ленивого исправления критических
  зависимостей.

Startup является fail-fast. Отсутствующий runtime config, неверный artifact config, пустой DSN,
невозможность собрать artifact/runtime dependencies или invalid worker cardinality должны
останавливать процесс до запуска claim loop.

## Конфигурационный контракт

Минимальный operational contract worker-а включает:

- `backtest.jobs.enabled`
- `backtest.jobs.worker_processes`
- `backtest.jobs.claim_poll_seconds`
- `backtest.jobs.lease_seconds`
- `backtest.jobs.heartbeat_seconds`
- `backtest.jobs.snapshot_seconds`
- `backtest.jobs.snapshot_variants_step`

Нормативные правила:

- если `backtest.jobs.enabled=false`, worker может завершаться со статусом disabled и кодом `0`;
- если `backtest.jobs.enabled=true`, то `worker_processes` должен быть `>= 1`;
- queue concurrency определяется только через `worker_processes`;
- service manager обязан materialize ровно `worker_processes` независимых экземпляров worker-а;
- каждый экземпляр должен иметь уникальный `locked_by` и уникальную runtime identity.

Если operational environment временно materialize-ит несколько workers внешним способом, без
явного config knob, это может считаться переходным состоянием, но canonical contract этого
документа остаётся `backtest.jobs.worker_processes`.

## Claim, lease и reclaim

Worker loop работает так:

1. poll через `claim_next(now, locked_by, lease_seconds)`;
2. если job нет, sleep на `claim_poll_seconds`;
3. если job claimed, запускается один deterministic attempt;
4. во время обработки worker продлевает lease через heartbeat;
5. при потере lease текущий attempt немедленно прекращается.

Обязательные инварианты:

- claim атомарный и использует row-lock semantics;
- в каждый момент времени только один worker может владеть claimed job;
- `locked_by` это стабильная identity worker-а, например `<hostname>-<pid>-<instance_index>`;
- queued jobs могут быть отменены сразу на уровне storage;
- running jobs отменяются на границах батчей;
- reclaim может перезапустить attempt с начала;
- job не должна застревать в `running` навсегда после смерти worker-а;
- worker, потерявший lease, не должен продолжать писать progress, snapshots или terminal state.

## Source of truth для claimed runs

Claimed worker читает только persisted run payloads:

- `job.request_json` является request source of truth;
- snapshot payload saved-mode используется там, где нужно восстановить effective template
  semantics;
- worker не должен перечитывать live saved strategy state для уже созданной job;
- worker не должен принимать runtime-решения на основе текущего browser state или UI state.

Это гарантирует, что claimed run остаётся воспроизводимым, даже если strategy storage изменился
после запуска.

## Контракт slot-pinned artifacts

Каждый claimed background run обязан уже нести pinned artifact identity в persisted job row:

- `artifact_slot`;
- `artifact_slot_generation`;
- `artifact_asof_date`;
- `artifact_manifest_hash`.

До старта runtime work worker обязан разрешить slot-pinned context через
`resolve_pinned_context(...)`.

Обязательные правила:

- отсутствие pin metadata является deterministic failure;
- отсутствие pinned artifacts является deterministic failure;
- drift pinned artifacts является deterministic failure;
- fallback на live `current.yaml` discovery для claimed runs запрещён;
- fallback на legacy runtime при провале pinned artifact bootstrap запрещён.

Worker исполняет только slot-pinned run. Он не участвует в publish/rebuild decision path и не
выполняет rebuild artifacts внутри себя.

## Shared runtime planning и ExecutionProfile

Worker не владеет отдельной planning policy. Он делегирует runtime planning в shared v2 planner
stack.

Для каждого claimed run worker:

- валидирует effective request/template contract;
- применяет supported request timeframes;
- применяет default-only rules для signal overrides;
- разрешает runtime plan через `BacktestArtifactRuntimePlannerV2`;
- использует startup-loaded `execution_profiles` и `adaptive_selector_policy`;
- исполняет выбранный profile через shared artifact-backed runtime services.

Это означает:

- `background_auto` это launch classification, а не отдельный scoring engine;
- `exact_small`, `exact_parallel`, `hybrid_conservative` и `hybrid_family` это runtime profile
  decisions внутри claimed path;
- browser и public API по-прежнему не выбирают `execution_profile_mode` напрямую;
- worker не принимает rollout-решения сам, а obeys shared adaptive selector policy.

## Граница rollout policy

Детальная rollout matrix для adaptive selector не дублируется в этом документе.

Этот документ фиксирует только границу ответственности:

- worker использует shared `execution_profiles`;
- worker использует shared `adaptive_selector_policy`;
- worker не владеет отдельными фазами rollout;
- worker не определяет самостоятельно `shadow`, `opt_in` или `active`;
- worker исполняет runtime plan, уже разрешённый общим planner/policy layer.

Подробные rollout rules, phase literals, benchmark gates и promotion criteria должны жить в
selector/runtime docs и config, а не в worker architecture doc.

## Flow исполнения по стадиям

Claimed execution использует одну общую stage model.

### Stage A

Worker строит deterministic shortlist базовых вариантов через artifact-backed Stage A services и
сохраняет Stage A shortlist metadata для observability и дальнейшей диагностики.

### Stage B

Worker расширяет retained candidates по risk dimensions и считает их через shared artifact-backed
Stage B runtime. Текущие best rows сохраняются как summary-only snapshots по snapshot cadence.

### Finalizing

Finalizing завершает persisted summary rows и terminal state transition.

Persisted background rows остаются summary-only:

- worker не materialize full report bodies как часть persisted run contract;
- worker не materialize trades payloads как часть persisted run contract;
- detail/trades concern остаётся отдельно от claimed background execution.

## Persistence и progress/ETA contract

Canonical persisted run storage остаётся на family таблиц `backtest_jobs`,
`backtest_job_top_variants` и `backtest_job_stage_a_shortlist`.

Worker отвечает за сохранение:

- lifecycle state и timestamps;
- имени текущей stage;
- `processed_units` и `total_units`;
- lease и heartbeat fields;
- failure/cancel payloads, если они есть;
- summary-only top rows и Stage A shortlist snapshots.

Worker не отвечает за финальный browser ETA.

Worker обязан сохранять deterministic progress counters, на которых позже строится read model:

- `stage`;
- `processed_units`;
- `total_units`;
- timestamps и heartbeat data.

Пользовательские `progress_percent` и `eta_seconds` вычисляются в runs history layer на основе:

- worker counters;
- execution-profile semantics;
- throughput estimate;
- benchmark-backed fallback, если throughput ещё не является defensible.

Таким образом, worker и history layer делят ответственность явно:

- worker пишет факты исполнения;
- read model считает user-facing progress и ETA.

## Observability

Worker публикует process-level metrics и structured logs.

Минимальная metrics surface:

- общее число claimed jobs;
- число succeeded jobs;
- число failed jobs;
- число cancelled jobs;
- число lease-lost events;
- длительность job;
- длительность stage;
- gauge активных claimed jobs.

Operational contract для metrics:

- каждый экземпляр worker-а должен иметь наблюдаемую metrics surface;
- deployment target обязан обеспечивать уникальный binding metrics endpoint или эквивалентную
  aggregation model;
- отсутствие видимой metrics surface у живого worker-а считается operational defect.

Structured logs должны позволять ответить на вопросы:

- был ли worker включён или выключен на startup;
- какая job была claimed каким worker identity;
- на какой stage сейчас находится run;
- терял ли worker lease;
- failure возник из-за pin drift, runtime error или cancellation;
- сколько экземпляров worker-а реально активно в production.

## Production deployment contract

Если `backtest.jobs.enabled=true`, production обязан обеспечивать постоянно работающий supervised
worker service.

Это означает:

- deployment обязан установить или обновить все worker services;
- deployment обязан перезапустить worker services;
- deployment обязан проверить, что поднялось нужное число экземпляров;
- deployment обязан проверить, что экземпляры реально живы и способны находиться в claim loop.

Если `backtest.jobs.enabled=true`, а постоянный worker service не установлен или не запущен,
такой deploy считается некорректным, даже если API успешно поднялся.

Рекомендуемый post-deploy smoke для worker-а должен быть service-level, а не request-level:

- проверить регистрацию service instances у supervisor;
- проверить, что процессы запущены;
- проверить, что каждый экземпляр публикует logs и metrics;
- проверить, что worker не завершился сразу со статусом disabled/error;
- проверить соответствие числа живых worker processes значению `worker_processes`.

Создание тестовой production job как обязательной части каждого deploy smoke не требуется и не
должно считаться canonical методом проверки.

## Масштабируемость

Архитектура worker-а должна быть готова к дальнейшему масштабированию:

- на одном host через большее число supervised worker processes;
- на нескольких host через ту же storage-based claim/lease model;
- через смену service manager без переписывания worker contract;
- без появления второго queue coordinator вне Postgres storage contract.

Scaling boundary остаётся простой:

- очередь масштабируется числом worker processes;
- correctness обеспечивается storage-level claim/lease;
- runtime semantics одного run определяются shared planner и shared kernels.

## Compatibility

`background_manual_legacy` остаётся поддерживаемым только как compatibility input для уже
существующих persisted rows.

Нормативные правила compatibility:

- worker обязан уметь claim-ить и исполнять такие строки;
- новые product flows не должны документироваться через `background_manual_legacy`;
- новые launch contracts не должны производить этот literal;
- в canonical архитектурном описании worker-а основной background path считается только через
  `background_auto`.

## Non-goals

Этот документ не вводит:

- новую публичную launch API surface;
- право browser-а выбирать `execution_profile_mode`;
- отдельный policy layer внутри worker-а;
- отдельный engine для background runs;
- смешение benchmark evidence anchor и active runtime default;
- platform-specific dependency на `launchd` или другой supervisor в архитектурном контракте.

## Связанные документы

- [Final Backtest Refactor Plan v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-refactor-final-plan-v2.md)
- [План доработки и ускорения backtest runtime v1](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md)
- [Backtest Jobs v1 -- Job-Runner Worker (claim/lease + streaming batches + cancel) (BKT-EPIC-10)](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)
- [Backtest Runs History API v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [Backtest API v1 — `POST /backtests` (saved strategy + ad-hoc grid) (BKT-EPIC-07)](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [Ранбук backtest job runner](/Users/daniildegtyarev/Projects/roehub.com/docs/runbooks/backtest-job-runner.md)
- [run_backtest_job_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py)
- [backtest_job_runner.py](/Users/daniildegtyarev/Projects/roehub.com/apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py)
