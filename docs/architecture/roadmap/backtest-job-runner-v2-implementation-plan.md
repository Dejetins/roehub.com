---
title: План внедрения backtest-job-runner v2
version: 1
status: draft
owner: backtest
---

# План внедрения backtest-job-runner v2

Статус: proposed implementation roadmap for canonical claimed background worker  
Дата фиксации: 2026-04-06  
Область: `backtest`, `apps/worker`, runtime config, deploy/runtime ops, documentation

## 1. Зачем нужен этот документ

В репозитории уже существует фактический `backtest-job-runner`, который:

- claim-ит queued persisted runs из Postgres;
- использует slot-pinned artifact-backed runtime;
- пишет progress, snapshots и terminal states;
- является реальным execution path для `background_auto` persisted runs.

Но вокруг worker-а остались structural gaps:

- canonical architecture doc до сих пор не выделен отдельно от historical v1 документа;
- в config и коде queue concurrency всё ещё выражена через `parallel_workers`, что конфликтует
  с будущим разделением queue concurrency и intra-run runtime parallelism;
- production deploy/bootstrap path не зафиксирован как обязательная часть backtest contract;
- docs и ops contracts не описывают масштабирование worker-а как first-class service surface;
- `background_manual_legacy` ещё недостаточно чётко отделён как compatibility-only literal.

Цель этого документа: описать реалистичный пошаговый план, как довести `backtest-job-runner`
до canonical v2 background worker без расширения публичного launch API и без появления второго
runtime engine.

## 2. Что считаем уже реализованным baseline

Вне scope этого roadmap:

- persisted run storage уже существует и используется как для sync, так и для background flows;
- claimed worker уже использует slot-pinned artifact-backed runtime;
- `background_auto` уже является canonical launch outcome для heavy-but-valid runs;
- runs history/status/top contracts уже существуют;
- один shared exact scorer и один shared runtime orchestration surface уже являются canonical
  архитектурой.

Основные reference документы:

- [Final Backtest Refactor Plan v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-refactor-final-plan-v2.md)
- [План доработки и ускорения backtest runtime v1](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md)
- [Backtest Job Runner v2 -- claimed background worker для persisted runs](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- [Backtest Runs History API v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [Ранбук backtest job runner](/Users/daniildegtyarev/Projects/roehub.com/docs/runbooks/backtest-job-runner.md)

## 3. Ключевые решения этого roadmap

### 3.1 Новый canonical doc

Canonical architecture doc для worker-а это отдельный файл:

- `docs/architecture/backtest/backtest-job-runner-v2.md`

`docs/architecture/backtest/backtest-job-runner-worker-v1.md` остаётся historical /
compatibility document и не является больше основным source of truth.

### 3.2 Canonical background launch mode

Canonical background launch mode это:

- `background_auto`

`background_manual_legacy` остаётся только compatibility-only literal:

- worker обязан исполнять уже существующие persisted rows с этим literal;
- новые launch flows не должны производить этот literal;
- product/UI/docs не должны описывать его как основной background path.

### 3.3 Queue concurrency задаётся отдельным конфигом

Canonical knob для числа независимых background workers это:

- `backtest.jobs.worker_processes`

Он задаёт число независимых worker processes и должен быть отделён от любой intra-run
parallelism semantics внутри runtime.

### 3.4 Worker architecture остаётся service-manager-agnostic

Архитектурный контракт worker-а не зашивается в `launchd`.

Документируется только service contract:

- долгоживущий supervised service;
- auto-restart;
- `N` экземпляров по конфигу;
- уникальная instance identity;
- logs и metrics per instance.

Конкретная реализация для Mac Studio через `launchd` фиксируется в runbooks и ops docs.

### 3.5 Deploy correctness

Если `backtest.jobs.enabled=true`, deploy без живого worker service считается некорректным.

Это должно быть закреплено:

- в deploy scripts;
- в bootstrap/reload scripts;
- в service-level smoke checks;
- в ops/runbook docs.

### 3.6 Rollout policy owner остаётся единым

Worker не владеет отдельной rollout policy.

Подробная policy surface остаётся в shared runtime docs/config:

- `execution_profiles`
- `adaptive_selector_policy`

Worker doc фиксирует только границу ответственности: worker obeys shared planner/policy layer.

## 4. Почему внедрение начинается с Milestone A

Порядок внедрения в этом документе начинается с `Milestone A`, а не с кодовых/ops изменений,
потому что сначала должен быть зафиксирован единый vocabulary и canonical source of truth.

Причина простая:

- если сначала менять config/deploy/runtime, а потом договариваться о canonical contract,
  получится несколько конкурирующих описаний одной и той же системы;
- `background_auto`, `background_manual_legacy`, `worker_processes`, deploy correctness и
  service-model boundary должны быть закреплены в документации до начала operational migration;
- только после этого безопасно менять config schema, worker entrypoint и deployment surface.

То есть `Milestone A` это не “необязательная бумажная стадия”, а foundation для всех следующих
изменений.

## 5. Целевая архитектура

Итоговая operational схема должна выглядеть так:

```text
POST /backtests
  -> request normalization
  -> launch classification
  -> persisted run creation
  -> execution_mode=background_auto
  -> queued row in backtest_jobs

supervised worker fleet
  -> N = backtest.jobs.worker_processes
  -> each worker owns one claim loop
  -> claim_next(..., locked_by, lease_seconds)
  -> resolve pinned artifact context
  -> shared runtime planning
  -> Stage A / Stage B / finalizing
  -> persisted progress + summary-only top rows

history/detail APIs
  -> read persisted run state
  -> derive progress/ETA for UX
```

Ключевые правила:

1. Один worker process обрабатывает максимум одну claimed job одновременно.
2. Масштабирование background queue идёт через число worker processes.
3. Runtime semantics одного run определяются shared planner и shared kernels.
4. Worker не должен создавать второй orchestration surface.
5. Deploy должен считать worker частью обязательного production contract.

## 6. Milestones и EPICs

Ниже предложен порядок внедрения. Он намеренно разделяет:

- foundation docs и terminology;
- config/schema migration;
- worker process model;
- production supervisor/deploy;
- shared runtime boundary;
- runbook/test closure.

### Milestone A. Foundation: canonical docs и terminology

Это первый milestone. Его задача: сначала зафиксировать единый source of truth для worker-а,
чтобы последующие config и deploy изменения опирались на уже утверждённый vocabulary.

#### EPIC A1. Canonical worker doc

Что делаем простыми словами:

- создаём отдельный canonical architecture doc для worker-а;
- старый v1-документ переводим в historical/compatibility reference.

Что должно быть реализовано:

- создать `docs/architecture/backtest/backtest-job-runner-v2.md`;
- обновить `docs/architecture/backtest/backtest-job-runner-worker-v1.md`;
- обновить ссылки в активных docs и индексах.

Документы:

- создать:
  - `docs/architecture/backtest/backtest-job-runner-v2.md`
- обновить:
  - `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
  - `docs/architecture/backtest/README.md`
  - `docs/runbooks/backtest-job-runner.md`
  - `docs/architecture/README.md` через docs index generation

#### EPIC A2. Canonical background vocabulary

Что делаем простыми словами:

- фиксируем `background_auto` как canonical background path;
- фиксируем `background_manual_legacy` как compatibility-only literal.

Что должно быть реализовано:

- launch docs описывают canonical background path только через `background_auto`;
- active docs помечают `background_manual_legacy` как compatibility-only literal;
- browser/public API contracts не возвращают этот literal в активный launch vocabulary.

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-runs-history-v2.md`
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
  - `docs/architecture/backtest/README.md`

#### EPIC A3. Roadmap synchronization

Что делаем простыми словами:

- делаем новый worker doc видимым для active roadmap surface.

Что должно быть реализовано:

- active roadmap docs должны ссылаться на `backtest-job-runner-v2.md` как на canonical worker
  doc;
- historical roadmap docs могут сохранять ссылки на v1-документ как на historical reference.

Документы:

- обновить:
  - `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
  - при необходимости: `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`

### Milestone B. Config schema: queue concurrency и worker cardinality

Это первый кодовый milestone. Он закрепляет отдельный runtime knob для числа background
workers и убирает смешение queue concurrency с intra-run parallelism.

#### EPIC B1. `worker_processes` как canonical config key

Что делаем простыми словами:

- добавляем explicit `backtest.jobs.worker_processes`;
- нормализуем typed runtime contract вокруг него.

Что должно быть реализовано:

- runtime config loader читает `worker_processes` как canonical key;
- typed runtime object наружу экспонирует только `worker_processes`;
- при `backtest.jobs.enabled=true` значение должно быть `>= 1`.

Кодовые точки:

- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`

Тесты:

- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py`

#### EPIC B2. Переходный alias для `parallel_workers`

Что делаем простыми словами:

- обеспечиваем additive migration path;
- не оставляем двусмысленность между старым и новым literal.

Что должно быть реализовано:

- если `worker_processes` отсутствует, а `parallel_workers` есть, loader может принять его как
  deprecated alias;
- если заданы оба ключа, loader должен fail-fast требовать один источник истины;
- docs и env configs переходят на новый literal;
- deprecated alias помечается как временный compatibility path.

Кодовые точки:

- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`

Тесты:

- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py`

#### EPIC B3. Runtime hash и startup invariants

Что делаем простыми словами:

- делаем worker cardinality частью строгого startup/runtime contract.

Что должно быть реализовано:

- новый canonical key участвует в runtime validation;
- новый canonical key участвует в runtime hash там, где это необходимо по текущему contract;
- loader и tests fail-fast защищают недопустимые значения.

Кодовые точки:

- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`

Тесты:

- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py`

### Milestone C. Worker process model и instance identity

Этот milestone делает worker fleet first-class operational surface.

#### EPIC C1. Instance-aware startup

Что делаем простыми словами:

- каждый worker instance получает уникальную identity и runtime surface.

Что должно быть реализовано:

- worker entrypoint поддерживает instance-aware startup;
- `locked_by` содержит уникальную worker identity;
- лог identity различает экземпляры одного и того же worker service.

Кодовые точки:

- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`

Тесты:

- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`

#### EPIC C2. Metrics surface per instance

Что делаем простыми словами:

- обеспечиваем наблюдаемость каждого экземпляра worker-а.

Что должно быть реализовано:

- metrics endpoint каждого экземпляра worker-а различим;
- deployment/service manager может однозначно адресовать каждый instance;
- docs фиксируют required metrics visibility для worker fleet.

Кодовые точки:

- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`

Тесты:

- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`

#### EPIC C3. Single-claim-loop invariant

Что делаем простыми словами:

- сохраняем простую process model: один процесс, одна claimed job за раз.

Что должно быть реализовано:

- BacktestJobRunnerApp остаётся single-claim-loop app;
- worker fleet concurrency достигается только количеством processes;
- код и docs явно не обещают parallel processing нескольких claimed jobs внутри одного process.

Кодовые точки:

- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

Тесты:

- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`

### Milestone D. Production supervisor, bootstrap и deploy

Этот milestone закрывает operational gap: при enabled jobs production обязан materialize worker
fleet как обязательную часть backtest contract.

#### EPIC D1. Mac Studio launchd materialization

Что делаем простыми словами:

- делаем generated/templated worker plists вместо отсутствующего или ручного fleet management.

Что должно быть реализовано:

- есть template или deterministic generator для per-instance worker services;
- label naming поддерживает масштабирование;
- bootstrap/reload scripts умеют materialize worker fleet из config.

Создать:

- `infra/macos/launchd/com.roehub.backtest-job-runner.plist.template`
- helper script для materialization per-instance plists, например:
  - `scripts/macos/render_backtest_job_runner_launchd.py`

Обновить:

- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/reload_launchd_services.sh`

#### EPIC D2. Deploy workflow для worker fleet

Что делаем простыми словами:

- deploy pipeline должен ставить и перезапускать worker fleet, а не только API.

Что должно быть реализовано:

- `.github/workflows/deploy-backend.yml` устанавливает worker services;
- `.github/workflows/deploy-backend.yml` перезапускает worker services;
- `.github/workflows/deploy-backend.yml` проверяет число живых worker instances.

Кодовые точки:

- `.github/workflows/deploy-backend.yml`

#### EPIC D3. Deploy correctness и service-level smoke

Что делаем простыми словами:

- зелёный deploy без живого worker fleet невозможен;
- проверяем не synthetic production backtest run, а состояние supervised services.

Что должно быть реализовано:

- deploy падает, если worker services не установлены или не поднялись;
- smoke проверяет:
  - service registration у supervisor;
  - живые процессы;
  - metrics endpoint на каждом instance;
  - отсутствие immediate disabled/error exit;
  - соответствие числа service instances значению `worker_processes`.

Документы:

- обновить:
  - `docs/runbooks/backtest-job-runner.md`
  - `docs/runbooks/mac-studio-native-backend-operations.md`

### Milestone E. Shared runtime boundary и compatibility closure

Этот milestone закрепляет, что worker не превращается во второй policy/runtime surface.

#### EPIC E1. Worker obeys shared planner/policy layer

Что делаем простыми словами:

- не создаём второй policy surface внутри worker-а;
- фиксируем shared ownership для execution profile и adaptive selector policy.

Что должно быть реализовано:

- worker code не получает отдельного policy surface вне shared runtime config;
- worker docs фиксируют только policy boundary;
- rollout rules продолжают жить в selector/runtime docs.

Кодовые точки:

- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`

Тесты:

- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`
- `tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py`

Документы:

- `docs/architecture/backtest/backtest-job-runner-v2.md`
- `docs/architecture/backtest/backtest-adaptive-selector-v1.md`

#### EPIC E2. Compatibility-only path для `background_manual_legacy`

Что делаем простыми словами:

- не ломаем старые persisted rows;
- убираем двусмысленность из новой документации и launch flow.

Что должно быть реализовано:

- worker продолжает исполнять `background_manual_legacy`;
- launch docs и API docs описывают canonical background path только через `background_auto`;
- active docs помечают `background_manual_legacy` как compatibility-only literal;
- тесты сохраняют coverage на compatibility processing path.

Кодовые точки:

- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

Документы:

- `docs/architecture/backtest/backtest-runs-history-v2.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

### Milestone F. Runbooks, tests и docs closure

Этот milestone закрывает migration handoff и делает worker fleet reviewable и поддерживаемым.

#### EPIC F1. Runbook alignment

Что делаем простыми словами:

- runbooks описывают фактическую service model для worker fleet на текущем deployment target.

Что должно быть реализовано:

- runbook описывает `worker_processes`;
- runbook описывает per-instance service shape;
- runbook описывает service-level smoke и диагностику fleet health;
- ops docs явно перечисляют worker services как production backtest dependency.

Документы:

- обновить:
  - `docs/runbooks/backtest-job-runner.md`
  - `docs/runbooks/mac-studio-native-backend-operations.md`

#### EPIC F2. Deterministic test closure

Что делаем простыми словами:

- докрываем config, wiring и deploy-sensitive invariants тестами.

Тестовые точки:

- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py`
- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`
- `tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py`

#### EPIC F3. Docs index closure

Что делаем простыми словами:

- закрепляем новый canonical worker surface в общей архитектурной навигации.

Что должно быть реализовано:

- docs index сгенерирован;
- `docs/architecture/README.md` указывает на новый canonical worker doc;
- новые roadmap/docs видны из стандартной навигации по репозиторию.

## 7. Какие файлы должны быть созданы или обновлены

### Создать

- `docs/architecture/backtest/backtest-job-runner-v2.md`
- `docs/architecture/roadmap/backtest-job-runner-v2-implementation-plan.md`
- `infra/macos/launchd/com.roehub.backtest-job-runner.plist.template`
- `scripts/macos/render_backtest_job_runner_launchd.py`

### Обновить

- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/backtest-runs-history-v2.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/backtest/backtest-adaptive-selector-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
- `docs/architecture/README.md`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`
- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/reload_launchd_services.sh`
- `.github/workflows/deploy-backend.yml`
- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py`
- `tests/unit/apps/worker/backtest_job_runner/wiring/modules/test_backtest_job_runner.py`
- `tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py`

## 8. Anti-patterns, которых надо избежать

- Не переиспользовать `parallel_workers` как canonical queue-concurrency knob.
- Не смешивать queue concurrency и Stage B/internal runtime parallelism.
- Не делать worker architecture doc зависимым от `launchd`.
- Не считать deploy успешным только потому, что поднялся API.
- Не возвращать `background_manual_legacy` в активную launch vocabulary.
- Не позволять browser-у выбирать `execution_profile_mode`.
- Не добавлять worker-specific planner или отдельный exact scorer.
- Не создавать synthetic production backtest run как обязательный deploy smoke.

## 9. Что считаем успехом

Успехом считается состояние, в котором:

- существует отдельный canonical doc `backtest-job-runner-v2.md`;
- все активные docs ссылаются на новый canonical worker document;
- `backtest.jobs.worker_processes` является canonical queue-concurrency contract;
- production deploy materialize-ит и проверяет worker fleet при enabled jobs;
- число живых worker instances соответствует config;
- `background_auto` документирован как canonical background path;
- `background_manual_legacy` остаётся только compatibility-only literal;
- worker продолжает использовать shared runtime planner/policy layer без отдельной policy
  surface;
- ops/runbook docs описывают фактическую service model на текущем deployment target;
- архитектура остаётся масштабируемой на несколько процессов и несколько host-ов.

## 10. Проверка согласованности

После изменения `.md` файлов запускать:

- `python -m tools.docs.generate_docs_index`
- `python -m tools.docs.generate_docs_index --check`
