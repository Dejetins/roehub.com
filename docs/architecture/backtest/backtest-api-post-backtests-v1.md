# Backtest API v1 — `POST /backtests` (saved strategy + ad-hoc grid) (BKT-EPIC-07)

Фиксирует контракт BKT-EPIC-07/R8-02: HTTP API v1 для deterministic launch flow в двух режимах
(`saved` / `ad-hoc`) с explicit `sync_inline` / `background_auto` semantics и unified
deterministic `422` ошибками.

## Status

- Status: active v1 launch contract after R8-02 explicit auto-fallback cutover.
- Master-plan note:
  - this document records the current shipped public launch contract;
  - remaining parity/product closure authority belongs to
    `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`.
- Compatibility note:
  - `POST /backtests` remains the active public launch route;
  - `background_auto` is the canonical background path for new queued runs created by
    `POST /backtests`;
  - `/backtests/jobs*` stays a `compatibility alias` for legacy background flows;
  - `background_manual_legacy` remains a `compatibility-only` literal for already persisted
    queued/running rows and read compatibility surfaces, not an active launch choice for new
    runs;
  - request/response naming keeps `top_k`, while `top_n_default` and `top_n_max` stay additive
    runtime-defaults literals.
- R7-03 follow-up note:
  - persisted runs created by `POST /backtests` are now publicly readable through
    `docs/architecture/backtest/backtest-runs-history-v2.md`;
  - legacy `/backtests/jobs*` remains a `compatibility alias` over the same storage family.
- R7-04 follow-up note:
  - preferred public lazy detail endpoint is now
    `POST /backtests/runs/{run_id}/variant-report`,
    which reconstructs original run context from persisted storage and pinned artifact fields;
  - legacy `POST /backtests/variant-report` remains behavior-compatible compatibility path for
    clients that still send full run envelope in request body.
- R8-01 background follow-up note:
  - external `execution_mode=background_manual_legacy` remains unchanged only for already
    persisted queued/running rows and public history payloads as a `compatibility-only` literal,
  - claimed worker execution behind that mode now reuses the same slot-pinned Stage A / Stage B
    artifact runtime contract as sync path and no longer depends on ClickHouse or
    `IndicatorCompute.compute(...)`,
  - summary-only persistence contract remains unchanged:
    `report_table_md=NULL`, `trades_json=NULL`.
- R10-01 production cutover note:
  - `POST /backtests` sync branch, claimed background execution, and run-scoped lazy
    `POST /backtests/runs/{run_id}/variant-report` now share one artifact-backed production hot
    path built from `BacktestArtifactTimelineBuilderV2`,
    `BacktestArtifactRuntimePlannerV2`, `BacktestStageAShortlistBuilderV2`,
    `BacktestArtifactRuntimeRunnerV2`, and `BacktestArtifactBackedStageBScorerV2`,
  - silent fallback to legacy `candle_timeline_builder.py`, `grid_builder_v1.py`,
    `staged_core_runner_v1.py`, `staged_runner_v1.py`, and close-fill Stage B wiring is no longer
    allowed in production launch/history/detail flows,
  - compatibility literals remain unchanged:
    `sync_inline`, `background_auto`, `background_manual_legacy`, `execution_mode`,
    `engine_version`.
- R8-02 launch orchestration note:
  - `POST /backtests` теперь пробует `sync_inline` только в sync half-budgets,
  - при canonical guard overflow backend выполняет explicit full-budget preflight; если он
    проходит, создаётся queued persisted run с `execution_mode=background_auto` как canonical
    background path,
  - fallback branch отвечает `202 Accepted` и явно возвращает `run_id`, `state=queued`,
    `execution_mode=background_auto`, `engine_version` и artifact pin metadata,
  - active launch docs stay centered on `background_auto`; if history/status surfaces still show
    `background_manual_legacy`, that literal is preserved only for already persisted legacy rows,
  - если full budgets тоже не проходят, backend возвращает canonical deterministic `422`
    и не создаёт persisted run row.
- R7-02 storage note:
  - sync and background executions share one persisted-run storage family in Postgres,
  - successful `POST /backtests` now performs internal preflight, executes inline, and persists
    one terminal run row plus summary-only top rows in the same table family; canonical
    `exact_no_risk_parity` sync runs also persist one internal `backtest_job_stage_a_shortlist`
    snapshot in the same atomic write,
  - successful sync response now includes persisted run metadata:
    `run_id`, `state`, `execution_mode=sync_inline`, `engine_version`,
    `artifact_slot`, `artifact_slot_generation`, `artifact_asof_date`,
    `artifact_manifest_hash`,
  - deterministic validation failures remain canonical `422` and do not create persisted rows.
- Superseded by target-v2 contract:
  - `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
  - `docs/architecture/roadmap/base_refactor_plan.md`
  - `docs/architecture/backtest/backtest-v2-benchmarks.md`
- Historical scope kept here:
  - current `POST /backtests` launch behavior,
  - legacy `top_k_*` naming and response fields,
  - current manual split between sync and jobs endpoints.
- A2/A3 redesign anchoring note:
  - target redesign vocabulary is now anchored in
    `docs/architecture/backtest/backtest-engine-vnext.md`;
  - this v1 document still records the shipped request/response shape, but active launch wording
    keeps only the current public fields and the summary-only launch rule;
  - the active full-detail path is
    `POST /backtests/runs/{run_id}/variant-report`, with full trades available only on-demand for
    an explicitly requested variant.
- R9-01 web-launch note:
  - `/backtests` browser UX now launches only through `POST /api/backtests`,
  - user-facing `top_n` input is sourced from runtime defaults and mapped explicitly to request
    `top_k`,
  - runtime defaults drive request timeframes, ranking metrics, supported indicators, and
    `inputs.source` catalogs,
  - explicit `202 Accepted` + `execution_mode=background_auto` is surfaced in UI as queued launch
    metadata, not as a silent mode switch.
- R1 target contract, enforced in backend validation:
  - allowed request TF: `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `1d`, `2d`, `3d`;
  - forbidden request TF in target contract: `1m`, `5m`;
  - `signals.v1.params` are `default-only`;
  - launch semantics move to auto-preflight + auto-fallback;
  - additive `top_n_default` / `top_n_max` naming is published via runtime defaults, while current request/response still keep `top_k`.
- A1 additive execution-profile note:
  - runtime-defaults now also publish `contracts.execution.default_execution_profile` and
    ordered `contracts.execution.available_execution_profiles`,
  - source config for that discovery surface lives in `backtest.execution_profiles`,
  - `POST /backtests` launch branches remain unchanged in A1:
    no new profile-based routing, no change to `sync_inline` vs `background_auto`, and no
    hybrid activation yet.
- A2 additive persisted-progress note:
  - persisted `/backtests/runs*` status/history payloads now add
    `progress_percent`, `eta_seconds`, and `execution_profile_mode` for browser rendering;
  - this does not change `POST /backtests` transport: launch remains request/response only,
    without SSE/WebSocket/streaming progress;
  - sync-inline launch page therefore still cannot show true in-flight progress before the HTTP
    response returns; real progress/ETA appears only after opening the persisted run page by
    `run_id`.
- B3 exact-classification note:
  - `POST /backtests` now classifies valid exact requests deterministically into
    `exact_small`, `exact_parallel`, or earlier queued `background_auto`;
  - classification uses planner cost evidence only:
    `stage_a_variants_total`, `stage_b_variants_total`, `estimated_memory_bytes`,
    and startup-validated execution-profile launch budgets;
  - heavy-but-valid exact requests are queued instead of being hard-rejected, while true full-budget
    guard violations still return canonical deterministic `422`;
  - effective `execution_profile_mode` is persisted into additive unified-run metadata for later
    `/backtests/runs*` progress/history rendering.
- D2 hybrid rollout note:
  - `hybrid_conservative` remains an internal-only approximate runtime profile;
  - current shipped sync wrapper may still server-pin that profile for the approximate shortlist
    path in the pre-master-plan snapshot;
  - `v2` master-plan treats this as current-state truth to be removed by the parity-first cutover,
    not as the target closure condition;
  - public `POST /backtests` still does not expose a profile selector;
  - internal-only execution-profile metadata must live in additive persisted fields outside
    `request_json`, while legacy `request_json.execution_profile_mode` remains compatibility-only
    for already stored rows and stays out of request-hash semantics.
- F1 sync cutover note:
  - current shipped sync `POST /backtests` path still records the hybrid-era approximate wrapper
    behavior where internal `execution_profile_mode=hybrid_conservative` may be forced for that
    snapshot of the cutover;
  - `v2` master-plan D0-D2 explicitly requires canonical no-risk parity closure to move away from
    that server-owned hybrid pinning;
  - public request/response transport remains unchanged:
    no new public launch fields, launch stays `summary-only`, and persisted top rows stay
    `summary-only`;
  - claimed worker/background cutover remains deferred to the separate F2 milestone.

## Цель

- Дать UI один endpoint `POST /backtests`, который:
  - запускает staged backtest v1 inline, если sync budgets проходят,
  - автоматически переводит launch в queued background run, если sync budgets не проходят, но
    full budgets проходят,
  - поддерживает режим `saved` (по `strategy_id`) и `ad-hoc` (по template/grid),
  - возвращает достаточно данных, чтобы UI мог сохранить выбранный вариант как StrategySpec (как минимум: конкретные параметры индикаторов, signals, risk/sizing/execution),
  - возвращает deterministic ошибки (unified 422), не “плавающие” между версиями.

## Контекст

- Public API contract остаётся v1, но active production runtime для launch/detail теперь идёт
  через artifact-backed v2 orchestration:
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_timeline_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- B3 runtime-profile routing uses one explicit typed `ExecutionProfile` surface in
  `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py` as the shared
  source of truth for:
  - exact launch-budget classification,
  - runtime-defaults discovery payloads,
  - persisted-run `execution_profile_mode`,
  - deterministic progress/ETA stage weights in `/backtests/runs*`.
- Current exact runtime-enabled launch profiles are:
  - `exact_small`
  - `exact_parallel`
- current shipped sync wrapper may still use `hybrid_conservative` as a server-owned internal
  launch profile for the approximate shortlist path; `v2` master-plan treats this as temporary
  current-state truth rather than the accepted parity closure target.
- `hybrid_family` remains future rollout surface only.
- Варианты детерминированы, guards применяются в sync режиме:
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- Execution engine v1: close-fill + fee/slippage + sizing + SL/TP:
  - `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- Reporting v1: equity/trades + metrics table `|Metric|Value|`:
  - `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- Identity уже даёт authenticated principal через dependency:
  - `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- API уже имеет глобальные deterministic error handlers:
  - `apps/api/common/errors.py` (RoehubError + RequestValidationError -> deterministic 422)

## Scope

- Endpoint v1: `POST /backtests`.
- Режимы:
  - A) saved: body содержит `strategy_id` + `overrides`.
  - B) ad-hoc: body содержит `template` (grid) без сохранения Strategy.
- Auth: endpoint защищён; доступ только для authenticated user.
  - saved mode: ownership/deleted checks выполняются в backtest use-case (не в HTTP слое).
- Output policy v1:
  - для grid запуска возвращаем только top-K summary rows (default `top_k_default=300`,
    override из request),
  - ranking выбирается по одному approved runtime metric с deterministic tie-break:
    `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`,
    `profit_factor`, `sharpe_trades`, `win_rate_pct`,
  - direction map фиксирован:
    `total_return_pct DESC`, `max_drawdown_pct ASC`,
    `return_over_max_drawdown DESC`, `profit_factor DESC`,
    `sharpe_trades DESC`, `win_rate_pct DESC`,
  - deterministic tie-break: `variant_key` (ASC),
  - `report` и `trades` в runtime summary response не materialize'ятся.
- A2/A3 target note:
  - current public redesign keeps only `primary_metric` plus the shipped `top_k` / `preselect`
    launch inputs;
  - removed legacy launch knobs stay out of active request/response/runtime-defaults wording;
  - full trades remain on-demand only; default launch and persisted top rows stay `summary-only`.
- Unified errors:
  - 422 payload детерминированный и единый через `RoehubError`.
- Reproducibility:
  - response включает `spec_hash` (saved) или `grid_request_hash` (ad-hoc) и `engine_params_hash`.

## Non-goals

- Async jobs/progress (Milestone 5).
- Public history/list/status endpoint design inside this document.
- Дополнительные endpoints (get status/list).
- R7-03 history retrieval semantics documented separately in
  `docs/architecture/backtest/backtest-runs-history-v2.md`.

## Ключевые решения

### 1) Один request envelope: `strategy_id` xor `template` (без explicit `mode`)

Request v1 не содержит поля `mode`. Режим определяется взаимоисключающими полями:

- saved mode: `strategy_id` задан, `template` отсутствует.
- ad-hoc mode: `template` задан, `strategy_id` отсутствует.

Причины:
- соответствует контракту BKT-EPIC-01 (use-case request DTO),
- снижает риск расхождений между API и application.

### 2) Signal params остаются в request shape, но работают как `default-only`

В template/saved mode поле `signal_grids` остаётся частью публичного request shape, но R1
фиксирует backend contract:

- `signals.v1.params` берутся из `configs/<env>/indicators.yaml`;
- request-level non-default overrides детерминированно отклоняются;
- практический UI flow: omit `signal_grids` и использовать runtime defaults как source of truth.

### 3) Typed API blocks -> canonical internal mappings

API принимает typed блоки (а не “произвольные dict”):

- `execution`: явные поля (`init_cash_quote`, `fee_pct`, `slippage_pct`, `fixed_quote`, `safe_profit_percent`)
- `risk_grid`: enable flags + axis specs (`explicit|range`) для SL/TP
- `signal_grids`: `indicator_id -> param_name -> axis spec`

На границе API эти блоки канонизируются в backtest application DTO:

- `RunBacktestTemplate.execution_params` (mapping),
- `RunBacktestTemplate.risk_grid` / `risk_params`,
- `RunBacktestTemplate.signal_grids`.

Причины:
- исключаем “плавающие” ключи (`fee` vs `fee_pct`) и делаем `variant_key`/report стабильными.

### 4) Ответ содержит variant payload для сохранения выбранного варианта

Для каждого варианта в top-K response содержит не только ключи и метрики, но и explicit payload:

- `indicator_selections`: список `{indicator_id, inputs, params}` (конкретные scalars),
- `signals`: `{indicator_id: {param: value}}`,
- `risk`: `{sl_enabled, sl_pct, tp_enabled, tp_pct}`,
- `execution`: effective execution scalars,
- `direction_mode`, `sizing_mode`.

Это позволяет UI:

- сохранить выбранный вариант как StrategySpec (минимум: индикаторы + params),
- сохранить/использовать risk/sizing/execution для следующего запуска или как defaults.

### 5) Reproducibility hashes: `spec_hash|grid_request_hash` + `engine_params_hash`

Response v1 включает:

- saved mode: `spec_hash` (детерминированный hash от saved StrategySpec payload),
- ad-hoc mode: `grid_request_hash` (детерминированный hash от canonical request payload),
- всегда: `engine_params_hash` (детерминированный hash от effective runtime settings, влияющих на результат).

Persisted-only launch/read metadata such as `execution_profile_mode` must be stored in additive
fields like `execution_profile_mode_hint` / `effective_execution_profile_mode`, while any legacy
`request_json` fallback must stay out of `grid_request_hash` while exact result semantics remain
unchanged.

Зачем:
- подтверждение воспроизводимости и защита от “тихих” изменений runtime defaults.

R9-01 launch UX note:

- browser может читать `contracts.summary.top_n_default` / `top_n_max`, но request/response schema
  endpoint-а остаётся `top_k`;
- mapping `top_n -> top_k` должен быть явным и детерминированно тестируемым на стороне UI.

### 6) Успешный sync launch теперь always persisted (`execution_mode=sync_inline`)

- `POST /backtests` больше не является purely-ephemeral flow.
- После успешного inline execution backend сохраняет:
  - terminal row в `backtest_jobs`,
  - summary-only top rows в `backtest_job_top_variants`,
  - для canonical `exact_no_risk_parity` runs: internal-only shortlist snapshot в
    `backtest_job_stage_a_shortlist`,
  - denormalized run metadata и artifact pin identity для будущих history/filter endpoints.
- Persisted top rows сохраняют только:
  - `payload_json`,
  - `summary_metrics_json`,
  - `best_tp_pct`,
  - `best_sl_pct`,
  - `report_table_md=NULL`,
  - `trades_json=NULL`.
- Публичный `POST /backtests` response при этом остаётся `summary-only` и не раскрывает
  parity internals.

### 6A) `POST /backtests` теперь имеет три deterministic launch branch

- Branch A: sync budgets проходят
  - backend исполняет inline compute,
  - valid exact request детерминированно получает effective profile
    `exact_small` или `exact_parallel` до execution,
  - persist'ит terminal row с `execution_mode=sync_inline`,
  - возвращает `200 OK` и ranked `variants[]`.
- Branch B: sync exact launch budgets не проходят, но full budgets/worker contract проходят
  - backend не делает hidden mode switch,
  - planner-aware exact classification может отправить heavy-but-valid request в этот branch
    ещё до legacy overflow-style fallback reject,
  - выполняет full-budget preflight и создаёт queued row с `execution_mode=background_auto` как
    canonical background path,
  - persist'ит effective `execution_profile_mode` в additive metadata
    (`execution_profile_mode_hint` / `effective_execution_profile_mode`), чтобы history/progress
    read path не терял реальный selected profile без засорения `request_json`,
  - возвращает `202 Accepted`,
  - `variants[]` остаётся пустым, потому что ranking summary ещё не materialize'ился.
- Branch C: full budgets тоже не проходят
  - backend возвращает canonical deterministic `422`,
  - persisted run row не создаётся.

### 7) Sync response остаётся summary-only; detail lives on-demand

- По умолчанию `POST /backtests` возвращает ranking + payload summary без `report` body.
- Поля `rows/table_md/trades` загружаются on-demand через
  `POST /api/backtests/runs/{run_id}/variant-report`.
- Legacy `POST /api/backtests/variant-report` остаётся compatibility-only endpoint для старых
  клиентов, которые всё ещё отправляют full run envelope.
- Runtime flag `backtest.reporting.eager_top_reports_enabled` остаётся только как compatibility
  knob для переходного wiring, но summary path R6-04 всё равно не строит `report`/`trades` тела.
- Этот summary-only shape совпадает с R7-01 persisted top-row contract:
  в storage сохраняются только `payload_json`, `summary_metrics_json`, `best_tp_pct`,
  `best_sl_pct`; `report/trades` не становятся частью persisted summary rows.

### 8) Sync cancellation: disconnect + hard deadline (кооперативно, без kill)

С 2026-02-25 sync route реализован как `async` и запускает compute в thread через `asyncio.to_thread(...)`.

Пока thread выполняет use-case, route:

- периодически проверяет `request.is_disconnected()`;
- при disconnect помечает `BacktestRunControlV1` как cancelled (`reason=client_disconnected`);
- дополнительно использует hard deadline (`BacktestRunControlV1(deadline_seconds=...)`),
  где значение берётся из runtime config `backtest.sync.sync_deadline_seconds`
  (`configs/<env>/backtest.yaml`) и прокидывается через
  `apps/api/wiring/modules/backtest.py -> build_backtests_router(...)`.

Отмена реализована кооперативно: staged loops проверяют token/checkpoint и прекращают вычисление без принудительного завершения thread/process.

### 9) Sync half-budgets, jobs full-budgets

С 2026-02-25 wiring применяет разные guard budgets:

- Sync (`RunBacktestUseCase`) получает половинные лимиты:
  - `floor(backtest.guards.max_variants_per_compute / 2)`;
  - `floor(backtest.guards.max_compute_bytes_total / 2)`.
- Jobs path сохраняет полные лимиты из `backtest.guards.*`.
- C 2026-03-29 `POST /backtests` использует это различие как deterministic launch policy:
  - half-budgets -> `sync_inline`,
  - half overflow + full pass -> `background_auto`,
  - full overflow -> canonical `422`.
- C 2026-04-04 B3 добавляет deterministic exact-profile classification поверх этого split:
  - planner сначала считает `stage_a_variants_total`, `stage_b_variants_total`,
    `estimated_memory_bytes`;
  - затем startup-configured `backtest.execution_profiles.profiles[*].launch_budget`
    классифицирует request в `exact_small` или `exact_parallel`;
  - если request остаётся exact-valid, но уже не помещается в sync exact budgets,
    launch branch отвечает explicit `202 Accepted` + `execution_mode=background_auto`,
    а не hard reject;
  - public launch response shape не меняется, но effective `execution_profile_mode`
    сохраняется для persisted `/backtests/runs*` contract.

### 10) CPU knob через Numba threads

С 2026-02-25 в `backtest.yaml` добавлен `backtest.cpu.max_numba_threads`.

- Значение валидируется fail-fast на старте.
- В sync и jobs перед run attempt вызывается `numba.set_num_threads(...)`.
- Это текущий runtime CPU knob v1 (без новых native dependencies).

## Endpoint v1: `POST /backtests`

### Request (v1)

Common fields:

- `time_range`: `{start, end}` (UTC, half-open `[start, end)`).
- current active launch fields: `top_k?`, `preselect?`, `ranking?`.
- target redesign keeps `top_k`/top-N semantics and `ranking.primary_metric`, while removed legacy
  launch knobs stay outside the public launch surface.

Mode selection:

- `strategy_id` (saved) xor `template` (ad-hoc).

Saved mode:

- `strategy_id: uuid`
- `overrides?`:
  - `direction_mode?`, `sizing_mode?`
  - `execution?`
  - `risk_grid?`
  - `signal_grids?` (shape preserved, non-default values rejected by `signals.v1.params=default-only`)

Ad-hoc mode:

- `template`:
  - `instrument_id: {market_id, symbol}`
  - `timeframe` (`15m|30m|1h|2h|4h|6h|8h|1d|2d|3d`; `1m`/`5m` rejected)
  - `indicator_grids[]` (grid specs)
  - `direction_mode?`, `sizing_mode?`
  - `execution?`, `risk_grid?`, `signal_grids?` (`default-only`; omission uses server defaults)

Axis spec shape (reused across grid specs):

- explicit: `{ "mode": "explicit", "values": [ ... ] }`
- range: `{ "mode": "range", "start": 1.0, "stop_incl": 5.0, "step": 0.1 }`

Percent units:

- `fee_pct=0.075` means `0.075%`.
- `slippage_pct=0.01` means `0.01%`.
- `sl_pct=3.0` means `3%`.

Ranking override:

- current public surface accepts only `ranking.primary_metric` with approved literals:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`,
  `sharpe_trades`, `win_rate_pct`.
- Если ranking не задан, применяются runtime defaults:
  `primary_metric=total_return_pct`.
- deterministic tie-break остается внутренним runtime behavior и не добавляет новых public fields.

### Response (v1)

Response содержит:

- `schema_version=1`
- `mode: "saved"|"template"`
- `instrument_id`, `timeframe`, `strategy_id?`
- `top_k`, `preselect`
- persisted launch metadata:
  - `run_id`
  - `state`
  - `execution_mode` (`sync_inline|background_auto`)
  - `engine_version`
  - `artifact_slot`
  - `artifact_slot_generation`
  - `artifact_asof_date`
  - `artifact_manifest_hash`
- reproducibility hashes:
  - `spec_hash?` or `grid_request_hash?`
  - `engine_params_hash`
- `variants[]`
  - для `execution_mode=sync_inline`: length `<= top_k`, отсортировано по выбранному ranking
    contract и tie-break `variant_key ASC`,
  - для `execution_mode=background_auto`: пустой список до materialized background result.
- `execution_profile_mode` не добавляется в public launch response, чтобы сохранить backward
  compatibility shape; effective profile instead persists in unified run storage and читается через
  additive `/backtests/runs*` fields.
- launch response stays summary-only: runtime-owned warmup derivation and eager-detail controls do
  not appear in public request/response payloads.
- legacy persisted payloads MAY still carry extra internal-only read-compatibility fields outside
  the active launch contract.

Каждый `variants[i]` содержит:

- `variant_index`, `variant_key`, `indicator_variant_key`
- `total_return_pct`
- `report`:
  - всегда `null` в runtime summary response R6-04
  - полный `rows + table_md + optional trades` загружается только через
    `POST /api/backtests/variant-report`
    или preferred run-scoped endpoint `POST /api/backtests/runs/{run_id}/variant-report`
- `payload` (explicit parameters for saving):
  - `indicator_selections[]`
  - `signals`
  - `risk`
  - `execution`
  - `direction_mode`, `sizing_mode`

## Lazy detail compatibility after R7-04

- Legacy compatibility endpoint:
  - `POST /api/backtests/variant-report`
  - request body содержит full run envelope (`time_range`, `strategy_id xor template`,
    `overrides?`) + explicit `variant` and optional `include_trades?`
- Preferred public endpoint:
  - `POST /api/backtests/runs/{run_id}/variant-report`
  - path содержит persisted `run_id`
  - request body содержит только:
    - `variant`
    - `include_trades?`
- Общие invariants:
  - пересчитывается ровно один selected variant;
  - detail/report/trades payloads не сохраняются в persisted summary rows;
  - full top-N recompute не запускается.

## Ошибки и статус-коды

- `200 OK` — sync branch (`execution_mode=sync_inline`) с inline-ranked `variants[]`.
- `202 Accepted` — explicit queued fallback branch (`execution_mode=background_auto`).
- `401` — unauthenticated (identity dependency).
- `422` — `RoehubError(code="validation_error")`:
  - invalid payload
  - sync/full guards or preflight budget exceeded
  - invalid time range / no market data
  - full-budget branch does not create persisted run rows on this path
- `404` — `RoehubError(code="not_found")`:
  - saved strategy missing or deleted
- `403` — `RoehubError(code="forbidden")`:
  - saved strategy принадлежит другому user
- `409` — `RoehubError(code="conflict")`:
  - request mode conflict (если будет нарушен контракт)

Порядок validation errors детерминирован (см. `apps/api/common/errors.py`).

## Wiring / Composition

FastAPI wiring v1:

- `apps/api/routes/backtests.py` — thin route: DTO mapping -> use-case call -> `200`/`202`
  status selection -> response mapping.
- `apps/api/wiring/modules/backtest.py` — composition:
  - `CandleFeed` (reuse indicators `MarketDataCandleFeed`),
  - `IndicatorCompute` (reuse indicators compute adapter),
  - `BacktestStrategyReader` adapter (ACL over StrategyRepository),
  - `BacktestJobRepository` over the unified Postgres storage family for persisted
    `sync_inline` writes and queued `background_auto` fallback rows,
  - `BacktestGridDefaultsProvider` (reads `configs/<env>/indicators.yaml` defaults),
  - `BacktestRuntimeConfig` from `configs/<env>/backtest.yaml`,
  - application-layer launch orchestrator, который композиционно использует:
    - sync persisted inline use-case,
    - full-budget preflight use-case,
    - jobs create use-case с explicit `background_auto`.

Fail-fast:

- модуль backtest загружается и валидируется на старте (как strategy/identity/indicators).

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-4-epics-v1.md` — BKT-EPIC-07.
- `docs/architecture/roadmap/base_milestone_plan.md` — UX/flow и hashes.
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — request DTO и ownership rule.
- `docs/architecture/api/api-errors-and-422-payload-v1.md` — unified errors contract.

API:
- `apps/api/routes/backtests.py`
- `apps/api/common/errors.py`
- `apps/api/main/app.py`

Backtest:
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/dto/run_backtest.py`
- `src/trading/contexts/backtest/application/ports/strategy_reader.py`

Strategy:
- `src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py`

## Как проверить

После реализации EPIC-07:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: большие detail payload’ы при variant-report запросах. Митигатор: launch/top rows остаются
  `summary-only`, а full trades materialize'ятся только on-demand для выбранного варианта.
- Риск: несоответствие saved Strategy индикаторного payload формату backtest grid. Митигатор: явный ACL mapper + строгая deterministic validation с 422.
