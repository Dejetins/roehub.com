# Backtest Architecture Docs

Краткий индекс и rollout-заметки для актуального backtest-контракта.

## Основные контракты

- Runtime kernels v2 contract for `signal_tf + 1m_risk`, Stage A / Stage B boundaries and
  notebook-derived transfer scope: `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- Notebook transfer reference and function-level semantics anchors:
  `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- Sync API: `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- Reporting metrics/table: `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- Jobs API: `docs/architecture/backtest/backtest-jobs-api-v1.md`
- Jobs worker: `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- Perf optimization plan: `docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md`
- Artifact store v2 layout/publish/pinning/validator/config contract: `docs/architecture/backtest/backtest-artifact-store-v2.md`
- Precompute runner v2 manifest/validator/config-driven publish contract, включая R3-01 canonical `1m` export, R3-02 rolled request TF prices, R3-03 `mappings/<tf>`, R3-04 publish-ready prices+mappings stage, R4-02 real `signals/<tf>/<indicator_id>` artifacts, R4-03 bounded `prefix + rebuilt_tail` signal rebuild и R5-01 real `hit_times/1m`: `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- Signal rules catalog and R4-01 semantic source-of-truth: `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
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
- R3-04 может publish'ить validated slot с `prices+mappings`, если validation spec явно выведен
  из `backtest_artifacts.validation_plan` и фиксирует `signal_artifacts=[]`,
  `require_hit_times_manifest=false`.
- R4-01 добавляет explicit v2 signal-rules engine contract с `inputs.source` semantics и
  `signals.v1.params = default-only`.
- R4-02 materialize'ит real `signals/<tf>/<indicator_id>/signals.i8.npy`, strict per-indicator
  manifests и root `signals.*` catalog для explicit configured targets.
- R4-03 переводит signal rebuild на deterministic bounded tail-update через
  `lookback_policy.signal_tail_bars_1m` и merge policy `prefix + rebuilt_tail`.
- R5-01 materialize'ит real `hit_times/1m`, поэтому full validation spec уже может требовать
  `require_hit_times_manifest=true` для актуального runner path.
- R5-02 фиксирует единый contract path:
  - `docs/architecture/backtest/backtest-runtime-kernels-v2.md` описывает production kernels,
  - `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md` сохраняет
    reference-only notebook walkthrough.
- Отдельный R3-04 prices+mappings publish helper остаётся stage-specific и по-прежнему выводит
  `signal_artifacts=[]` и `require_hit_times_manifest=false`.
- R4-04 runtime `source` integration в текущем репозитории проходит через runtime defaults, jobs
  `/top` payloads и explicit `variant-report` payloads, хотя отдельные history/detail v2 docs из
  roadmap пока отсутствуют.

## Проверка согласованности

- После изменения `.md` файлов запускать:
  - `python -m tools.docs.generate_docs_index`
