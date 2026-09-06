# Бектест: текущая реализация и контракты

Статус: active; сверено с репозиторием 2026-09-06.

Artifact-backed runtime реализован: API создаёт persisted jobs, отдельный worker
исполняет расчёт и lazy trades materialization, Web UI показывает результаты.
Прежняя запись о runtime-compute reset больше не описывает текущий код.

## Текущие источники

- [Контракт artifact runtime v1](backtest-service-artifact-runtime-v1.ru.md).
- [API routes](../../../apps/api/routes/backtests.py).
- [Backtest application services](../../../src/trading/contexts/backtest/application/services/).
- [Artifact publisher/precompute](../../../src/trading/contexts/backtest_artifacts/).
- [Job worker](../../../apps/worker/backtest_job_runner/).
- [Installation service manifest](../../../configs/installation/runtime-service-manifest.json).
- [Self-hosted эксплуатация](../../runbooks/offline-release-installation.md).

Наличие реализации и тестов не подтверждает развёртывание. Репозиторий не выбирает
действующий production host; запуск и runtime proof требуют конкретной установки.

## Исторические материалы

- [План ввода job runner](backtest-job-runner-production-plan-v1.md).
- [Начальный prompt pack Iteration 0/1](backtest-service-implementation-prompt-pack-iteration-0-1.md).
- [Benchmark records](benchmark_iterations/README.md).
- [Результаты неудачных ускорений](backtest-compute-acceleration-negative-results-v1.md).

Планы, ledgers и прежние требования к Mac Studio сохраняют историю решений и
измерений. Они не выбирают текущую задачу или deployment target. Новый benchmark
фиксирует revision, окружение, fixture, hashes и сопоставимый baseline; старые
числа нельзя переносить на другое оборудование без повторного измерения.
