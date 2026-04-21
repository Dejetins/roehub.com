# Backtest Core Refactor Prompt-Pack v1 (`NR2` / `RG-TTR`)

Статус: proposed executable prompt-pack after `deep-research-report` audit  
Дата: 2026-04-21

Связанные документы:
- `docs/architecture/backtest/deep-research-report.md`
- `docs/architecture/backtest/README.md`
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb` (`RG-TTR` anchor)
- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb` (`NR2` anchor)
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`
- `docs/architecture/backtest/README.md`

## 1. Цель prompt-pack

Собрать практическую цепочку промптов для доведения ядра backtest до состояния, где canonical benchmark-классы (`NR2`, `RG-TTR`) проходят текущие blocking-gates и выходят на **stretch-цель около 90% по скорости и памяти** относительно notebook anchor.

Notebook anchors для этой цели:
- `NR2`: `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`
- `RG-TTR`: `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`

Под «около 90%» в этом пакете фиксируем как stretch-gate:
- `wall_clock_ratio <= 1.10`
- `peak_rss_ratio <= 1.10`

При этом базовые репозиторные gates из `backtest_notebook_parity_benchmark_corpus_v1.json` остаются обязательными и не ослабляются.

## 2. Аудит покрытия `deep-research-report.md`

## Вердикт

`docs/architecture/backtest/deep-research-report.md` покрывает большую часть стратегического направления, но **не полностью закрывает practical execution scope** для цели ~90%.

### Что покрыто хорошо

- Корректно зафиксированы blocking классы: `NR2`, `RG-TTR`.
- Корректно зафиксировано текущее benchmark-состояние (closure open).
- Корректно выделена необходимость explicit runtime-shape evidence.
- Корректно обозначены риски broad corrective waves и необходимость узких kernel-фаз.

### Что не покрыто полностью (критично для реализации)

1. Не выделен отдельным root-cause факт, что no-risk path заранее инициализирует тяжёлый Stage B scorer.

- В `run_backtest.py` и `run_backtest_job_runner_v1.py` scorer создаётся до решения terminal-path.
- В `artifact_backed_stage_b_scorer_v2.py` конструктор сразу грузит `hit_times/1m` и ребейзит их в локальные массивы.
- Для canonical no-risk это лишняя загрузка памяти и CPU до старта Stage A.

2. Не выделен как отдельный corrective item факт unconditional sync payload override.

- `backtest_runs_api_v1.py::_with_sync_inline_redesigned_engine_request_payload(...)` всегда добавляет `execution_profile_mode=exact_no_risk_parity`.
- Это допустимо для canonical `NR2`, но потенциально конфликтно для risk-grid benchmark flows (`RG-TTR`) и требует явного conditional rule.

3. Нет отдельного prompt-шага для доказуемого перехода от repo-gates к stretch-gates (~90%).

- В отчете есть acceptance на `1.18/1.35`, но нет формализованного плана перехода к `1.10/1.10`.

4. Не зафиксирован explicit stop-line порядок для case, когда `RG-TTR` остаётся `bypassed_no_risk` после routing fixes.

- Нужен отдельный prompt на provenance trace request/config/template before any next optimization.

## 3. Подтверждённые факты (по коду и corpus)

1. Sync wrapper injects parity profile:
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- helper `_with_sync_inline_redesigned_engine_request_payload(...)` пишет `execution_profile_mode=exact_no_risk_parity`.

2. Scorer инициализируется до runtime branch selection:
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

3. Artifact Stage B scorer eagerly loads heavy arrays в constructor:
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- внутри `__init__`: `load_hit_times_arrays(...)` + `slice_hit_times_to_execution_window_v2(...)`.

4. No-risk terminal path уже существует в runtime core:
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `run_stage_b_or_finalize_no_risk(...)` при no-risk уходит в `_finalize_no_risk_stage_a_v2(...)`.

5. Stage A уже публикует no-risk metrics для shortlist rows:
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- no-risk rows строятся с `no_risk_metrics`.

6. Benchmark authority (фиксированные текущие blockers):
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `NR2` live capture: `capture_status=missing`, notes с датой `2026-04-18`, ratios `6.469x / 14.499x`.
- `RG-TTR` live capture: `capture_status=captured`, но `stage_b_execution_mode=bypassed_no_risk` (expected `in_process`).

## 4. Подтверждённое направление реализации

Порядок работ, который даёт наименьший риск и максимальный шанс быстро выйти к ~90%:

1. Сначала убрать заведомо лишний memory/CPU overhead в no-risk пути (`NR2`):
- decouple no-risk from eager Stage B scorer/hit-times load.

2. Затем починить truth-routing для `RG-TTR`:
- conditional sync profile injection + provenance trace request/config/template.

3. Только после этого делать точечные Stage A tuning шаги:
- pair-block size, retained frontier caps, chunk lifecycle.

4. В конце — closure по benchmark authority:
- обновление live capture evidence и docs sync.

## 5. Prompt-Pack (исполняемая цепочка)

## P0. Baseline Freeze + Trace Surface

**Цель:** зафиксировать baseline до новых правок и сделать обязательные trace-поля для root-cause диагностики.

**Primary files:**
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `src/trading/contexts/backtest/application/services/v2/notebook_parity_benchmark_corpus_v2.py`

**DoD:**
- baseline snapshot зафиксирован и не спорит с текущим fixture;
- trace payload включает: `stage_b_execution_mode`, `stage_b_process_fallback_threshold`,
  `exact_replay_count`, `max_python_processes_seen`.

**Checks:**
```bash
uv run pytest -q tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
```

## P1. No-Risk Scorer Decoupling (Critical)

**Цель:** для no-risk terminal path не создавать тяжёлый artifact Stage B scorer до Stage A.

**Primary files:**
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`

**DoD:**
- no-risk ветка не вызывает eager hit-times load;
- Stage A no-risk финализация работает на `no_risk_metrics` без поведенческого регресса;
- sync и worker используют одинаковую policy.

**Checks:**
```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py -k "nr2 or no_risk"
```

## P2. Lazy Hit-Times Loading for Risk Path Only

**Цель:** перенести загрузку/ребейз `hit_times` из scorer constructor в lazy path, который вызывается только при risk Stage B.

**Primary files:**
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`

**DoD:**
- при no-risk execution нет materialization `StageBHitTimesSliceV2`;
- risk-grid Stage B behavior unchanged.

**Checks:**
```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py
```

## P3. RG-TTR Request/Template Provenance Fix

**Цель:** перестать применять `exact_no_risk_parity` override к non-NR2 sync launches.

**Primary files:**
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py`

**DoD:**
- override `execution_profile_mode=exact_no_risk_parity` остаётся только для canonical NR2 class;
- risk-grid canonical shape не коллапсирует в no-risk из-за transport override.

**Checks:**
```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py \
  tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py -k "rg_ttr or execution_profile_mode"
```

## P4. RG-TTR Runtime Shape Guard

**Цель:** сделать невозможным незаметный drift `RG-TTR -> bypassed_no_risk` без явного test failure.

**Primary files:**
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`

**DoD:**
- gate изолированно валит на `stage_b_execution_mode` и/или `exact_replay_count`, если drift повторяется;
- нет ложноположительных прохождений при неверном runtime shape.

**Checks:**
```bash
uv run pytest -q tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py -k "rg_ttr"
```

## P5. Conditional Stage A Tuning (Only If Needed)

**Запускать только если после P1-P4 всё ещё провал по wall/RSS.**

**Цель:** точечный Stage A tuning без изменения контрактов ранжирования.

**Primary files:**
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`

**DoD:**
- shortlist identity и tie-break unchanged;
- measurable drop wall/RSS на NR2 benchmark flow.

**Checks:**
```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_signal_aggregator_kernel_v2.py
```

## P6. Live Capture Refresh + Corpus Update

**Цель:** получить новые benchmark-host capture для `NR2` и `RG-TTR` на equal-thread-budget и обновить corpus.

**Primary files:**
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `docs/architecture/backtest/README.md`

**DoD:**
- `NR2` capture больше не `missing`;
- `RG-TTR` capture отражает truthful `stage_b_execution_mode`;
- notes содержат дату/host/slot/thread-budget + notebook anchor (`01_run_322...` / `02_run_f7d2...`).

## P7. Closure + Stretch Gate Pass

**Цель:** закрыть контур по repo-gates и зафиксировать stretch result около 90%.

**Primary files:**
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

**DoD:**
- repo-gates pass (`1.18/1.35` + runtime shape);
- отдельно зафиксирован stretch status (`1.10/1.10`) с датой и capture_id;
- если stretch не достигнут, явно указаны remaining blockers и следующий corrective prompt.

## 6. Gate policy для этого prompt-pack

### Обязательные repo-gates (cannot relax)

- `NR2`:
  - `wall_clock_ratio <= 1.18`
  - `peak_rss_ratio <= 1.35`
  - `max_python_processes_seen <= 1`
  - `stage_b_execution_mode = bypassed_no_risk`
  - `stage_b_process_fallback_threshold = none`

- `RG-TTR`:
  - `wall_clock_ratio <= 1.18`
  - `max_python_processes_seen <= 1`
  - `stage_b_execution_mode = in_process`
  - `stage_b_process_fallback_threshold = none`
  - `exact_replay_count <= 64`

### Stretch-gates для цели «около 90%»

- `NR2`: `wall_clock_ratio <= 1.10`, `peak_rss_ratio <= 1.10`
- `RG-TTR`: `wall_clock_ratio <= 1.10`, `peak_rss_ratio <= 1.10` (если в capture есть notebook RSS)

## 7. Stop-line правила

1. Если после `P1+P2` нет заметного падения RSS на `NR2`, не переходить к broad refactor; сначала снять memory profile по моменту scorer init.
2. Если после `P3` `RG-TTR` всё ещё `bypassed_no_risk`, остановиться и сделать provenance trace request/config/template (до новых kernel-оптимизаций).
3. Любое ухудшение top-1 parity или tie-order — немедленный rollback последнего prompt.

## 8. Completion status

- Текущий документ: **готов к исполнению** как новая prompt-pack цепочка.
- `deep-research-report.md`: **частично достаточен**, но требует этого prompt-pack как operational layer для достижения цели ~90%.
