# Backtest Architecture Docs (v1)

Краткий индекс и rollout-заметки для актуального backtest-контракта.

## Основные контракты

- Sync API: `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- Reporting metrics/table: `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- Jobs API: `docs/architecture/backtest/backtest-jobs-api-v1.md`
- Jobs worker: `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- Perf optimization plan: `docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md`
- Artifact store v2 layout/publish/pinning/validator/config contract: `docs/architecture/backtest/backtest-artifact-store-v2.md`
- Precompute runner v2 manifest/validator/config-driven publish contract: `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- Artifact rebuild/publish runbook: `docs/runbooks/backtest-artifacts-rebuild.md`

## Актуальная политика rollout

- Ranking order в sync/jobs фиксирован:
  - primary metric `total_return_pct` (DESC),
  - tie-break `variant_key` (ASC).
- Детальные отчёты (`rows/table_md/trades`) загружаются по explicit `variant-report`.
- Runtime flag `backtest.reporting.eager_top_reports_enabled` оставлен для legacy sync fallback;
  целевой режим v1: lazy-only (`false`).
- Artifact pipeline settings живут отдельно в `configs/<env>/backtest_artifacts.yaml`; runtime
  request defaults остаются в `configs/<env>/backtest.yaml`.

## Проверка согласованности

- После изменения `.md` файлов запускать:
  - `python -m tools.docs.generate_docs_index`
