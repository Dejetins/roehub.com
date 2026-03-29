# Backtest Architecture Docs

Краткий индекс и rollout-заметки для актуального backtest-контракта.

## Основные контракты

- R6-01 runtime-side loader/context layer:
  `src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py`,
  `src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py`,
  `src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py`
- R6-02 Stage A artifact-backed kernels and shortlist bridge:
  `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`,
  `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`,
  `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- R6-03 Stage B artifact-backed risk kernels and scorer bridge:
  `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`,
  `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`,
  `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- Runtime kernels v2 contract for `signal_tf + 1m_risk`, Stage A / Stage B boundaries and
  notebook-derived transfer scope: `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- Notebook transfer reference and function-level semantics anchors:
  `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- R5-03 Stage B golden fixture baseline for `signal_tf + 1m_risk`:
  `tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json`
- R5-03 executable fixture tests:
  `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`
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

- Ranking order в sync/jobs конфигурируется по approved runtime set:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`,
  `sharpe_trades`, `win_rate_pct`.
- Direction map фиксирован:
  `total_return_pct DESC`, `max_drawdown_pct ASC`, `return_over_max_drawdown DESC`,
  `profit_factor DESC`, `sharpe_trades DESC`, `win_rate_pct DESC`.
- Deterministic tie-break зафиксирован отдельно по стадиям:
  - Stage A shortlist: `base_variant_key ASC`;
  - Stage B/final rows: `variant_key ASC`.
- Детальные отчёты (`rows/table_md/trades`) загружаются по explicit `variant-report`.
- Runtime flag `backtest.reporting.eager_top_reports_enabled` оставлен только как compatibility
  knob; sync/jobs runtime summary paths остаются `summary-only` и не materialize'ят report/trades.
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
- R5-03 фиксирует executable validation baseline для Stage B semantics через:
  - unit fixture catalog `stage_b_golden_fixtures_v2.json`,
  - perf manifest `r5_stage_b_golden_cases.json`,
  - notebook-independent tests `test_stage_b_golden_fixtures_v2.py`.
- R6-01 фиксирует shared `slot-pinned context` bootstrap:
  - sync run читает active identity из `current.yaml`,
  - background run использует persisted `artifact_slot`, `slot_generation`,
    `artifact_asof_date`, `artifact_manifest_hash`,
  - runtime loaders читают arrays только по explicit manifest-driven paths через
    `np.load(..., mmap_mode='r')` и `allow_pickle=False`.
- R6-02 добавляет Stage A kernels и additive shortlist bridge:
  - `signal_aggregator_kernel.py` агрегирует `final_signal` по strict consensus AND policy;
  - `trade_compactor_kernel.py` строит compact trades и no-risk shortlist metrics без Stage B
    risk exits;
  - `stage_a_shortlist_builder_v2.py` использует `artifacts-only inputs`, subset row loading и
    `chunked variant processing`;
  - sync/jobs paths подключают v2 Stage A path только при наличии валидного pinned artifact
    context.
- R6-03 добавляет Stage B kernels и additive scorer bridge:
  - `risk_exit_kernel_1m.py` резолвит one-trade exits по compact trades и shipped `1m hit-times`;
  - `metrics_kernel.py` считает deterministic ranking/summary payloads по exact replay;
  - `artifact_backed_stage_b_scorer_v2.py` использует fast TP/SL search и ограничивает exact
    replay winning cell / explicit retained variants;
  - sync/jobs/detail используют artifact-backed Stage B scoring как production hot path при
    pinned artifact context; legacy close-fill scorer остаётся только compatibility-only module и
    не используется в active production orchestration.
- R6-04 закрывает runtime ranking/top-N materialization:
  - accepted ranking literals совпадают между DTO/API/runtime defaults;
  - Stage A/Stage B tie-break остаётся explicit и stable;
  - sync/jobs runtime summary rows не строят `report`/`trades` тела.
- R8-03 закрепляет background safety contract:
  - publish guard блокируется только по active background runs
    (`queued|running` + `background_auto|background_manual_legacy`);
  - `queued -> cancelled` снимает guard сразу;
  - `running` с `cancel_requested_at` остаётся blocking до terminal state;
  - runs history/status сохраняют один и тот же lifecycle vocabulary для обоих background modes.
- R9-01 закрепляет launch UI contract:
  - `/backtests` больше не требует manual `Estimate preflight`;
  - browser launch всегда идёт через `POST /api/backtests`;
  - request timeframes, ranking metrics, `top_n_default` / `top_n_max`,
    `supported_indicator_ids`, `source_values_by_indicator_id` читаются из runtime defaults;
  - user-facing `top_n` маппится в request `top_k`;
  - `202 Accepted` + `execution_mode=background_auto` показываются пользователю явно.
- R9-02 закрепляет history/summary UX:
  - primary navigation теперь использует `/backtests/history` и `/backtests/runs/{run_id}`;
  - history page грузит `GET /api/backtests/runs` с opaque `next_cursor`;
  - run summary page грузит `GET /api/backtests/runs/{run_id}` и
    `GET /api/backtests/runs/{run_id}/top`;
  - local sort использует только runtime-approved `contracts.summary.sortable_columns`,
    переставляет loaded rows in-browser и не триггерит server recompute;
  - `/backtests/jobs*` остаются compatibility alias на переходный период.
- R9-03 закрепляет persisted variant detail/save UX:
  - dedicated detail page живёт по `/backtests/runs/{run_id}/variants/{variant_key}`;
  - detail page находит exact summary row через `/top`, затем вызывает только
    `POST /api/backtests/runs/{run_id}/variant-report` с `variant + include_trades`;
  - summary page остаётся summary-only, но добавляет row actions `Open detail` и
    `Save as Strategy`;
  - save flow переиспользует existing strategy builder prefill transport через
    `sessionStorage` и `/strategies/new?prefill=...`;
  - `/backtests/jobs*` остаются compatibility alias и не становятся primary UX.
- R10-01 закрепляет production hot-path cutover:
  - `/backtests` sync launch, claimed worker execution и run-scoped lazy detail используют только
    artifact-backed v2 runtime orchestration;
  - active production path больше не зависит от `candle_timeline_builder.py`,
    `grid_builder_v1.py`, `staged_core_runner_v1.py`, `staged_runner_v1.py`;
  - silent legacy fallback запрещён; `background_manual_legacy` остаётся только совместимым
    persisted/public literal.
- После R10-01 вне scope остаются R10-02 docs synchronization и R10-03 perf/runbook closure.
- Отдельный R3-04 prices+mappings publish helper остаётся stage-specific и по-прежнему выводит
  `signal_artifacts=[]` и `require_hit_times_manifest=false`.
- R4-04 runtime `source` integration в текущем репозитории проходит через runtime defaults, jobs
  `/top` payloads и explicit `variant-report` payloads, хотя отдельные history/detail v2 docs из
  roadmap пока отсутствуют.

## Проверка согласованности

- После изменения `.md` файлов запускать:
  - `python -m tools.docs.generate_docs_index`
  - `python -m tools.docs.generate_docs_index --check`
