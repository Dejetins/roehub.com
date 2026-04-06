# Backtest Runs History API v2

Документ фиксирует публичный контракт EPIC R7-03: owner-scoped history/status/top/cancel API
поверх unified persisted run storage (`backtest_jobs*`) после R7-02 cutover.

## Status

- Status: active public runs contract after R7-03.
- Unified storage note:
  - public `/backtests/runs*` читает то же семейство PG таблиц, что и legacy `/backtests/jobs*`;
  - `backtest_jobs.job_id` остается storage identity, но наружу используется vocabulary `run_id`;
  - `GET /backtests/runs/{run_id}/top` читает только summary rows из
    `backtest_job_top_variants`;
  - legacy `/backtests/jobs*` остается `compatibility alias` на период миграции.
- Summary-only note:
  - persisted top rows contract fields: `payload_json`, `summary_metrics_json`,
    `best_tp_pct`, `best_sl_pct`;
  - `report_table_md` и `trades_json` не входят в public runs contract и остаются `NULL`-only.
  - storage mapper for `GET /backtests/runs/{run_id}/top` normalizes persisted
    `updated_at` from PostgreSQL `timestamptz` to UTC before creating
    `BacktestJobTopVariant`, so read-path behavior does not depend on session timezone.
- Compatibility note:
  - `background_auto` is the canonical background path for new queued runs that enter
    `/backtests/runs*`;
  - `background_manual_legacy` remains a `compatibility-only` `execution_mode` literal for
    already persisted rows and compatibility aliases, and those rows remain supported.
- R7-04 additive note:
  - public lazy detail endpoint `POST /backtests/runs/{run_id}/variant-report` пересчитывает
    ровно один выбранный вариант по persisted `run_id` и explicit `variant` payload;
  - endpoint восстанавливает original request semantics из `request_json` и использует только
    persisted `artifact_slot`, `artifact_slot_generation`, `artifact_manifest_hash`,
    `artifact_asof_date` для pinned runtime context;
  - detail/report/trades payloads не сохраняются в PG и не участвуют в `/top`.
- R8-02 additive note:
  - `POST /backtests` теперь может создавать queued run с `execution_mode=background_auto`
    и отвечать `202 Accepted` вместо скрытого mode switch;
  - такие runs появляются в `/backtests/runs*` сразу после launch и используют тот же storage,
    owner policy, status, `/top`, `/cancel` и history semantics, что и уже persisted
    `background_manual_legacy` compatibility rows.
- R8-03 additive note:
  - public history/status продолжает показывать lifecycle literals
    `queued|running|succeeded|failed|cancelled` без отдельного vocabulary для background modes;
  - `POST /backtests/runs/{run_id}/cancel` для `running` rows возвращает всё ещё `running`, но с
    заполненным `cancel_requested_at`, пока worker не доведёт run до terminal state;
  - это поведение одинаково для `background_auto` и `background_manual_legacy`.
- R9-02 additive note:
  - web primary UX теперь использует `/backtests/history` и `/backtests/runs/{run_id}` поверх
    этого public API;
  - persisted run summary page загружает `/top` summary rows и разрешает только local resort по
    runtime-approved `contracts.summary.sortable_columns`;
  - local resort не декодирует `next_cursor`, не пересчитывает top-N и не вызывает server-side
    recompute.
- R9-03 additive note:
  - persisted summary rows получили browser-side actions:
    - `/backtests/runs/{run_id}/variants/{variant_key}`
    - `Save as Strategy` через existing `/strategies/new?prefill=...` flow;
  - web detail page восстанавливает exact selected row через `/top`, затем вызывает только
    run-scoped `POST /backtests/runs/{run_id}/variant-report`
    (`POST /api/backtests/runs/{run_id}/variant-report` на browser/API boundary);
  - detail/save UX не добавляет новых persisted report/trades storage surfaces.
- A2 additive progress note:
  - public `/backtests/runs*` payloads теперь добавляют
    `progress_percent`, `eta_seconds`, `execution_profile_mode`;
  - `progress_percent` считается детерминированно из
    `stage + processed_units + total_units + execution-profile progress_weights`,
    а не из client-side spinner/эвристики;
  - `eta_seconds` использует precedence `current throughput -> benchmark fallback -> null`:
    сначала текущий run timeline, затем startup-loaded benchmark corpus, и только потом
    `null`, если ни один источник не даёт defensible estimate;
  - read path берёт `execution_profile_mode` из persisted `request_json`, если B3 profile-aware
    launch уже сохранил effective profile;
  - для старых rows или unrelated legacy rows fallback остаётся configured default exact profile,
    чтобы public contract был backward compatible.
- B3 additive exact-profile note:
  - `POST /backtests` now persists effective `execution_profile_mode` for both
    `sync_inline` and earlier queued `background_auto` exact runs;
  - `/backtests/runs*` progress/ETA semantics now read stage weights from the same execution-profile
    catalog used by launch classification, without a second out-of-band mapping table.

## Цель

- Дать UI/public clients единый vocabulary `runs/history`, а не `jobs`.
- Зафиксировать deterministic owner policy для persisted history:
  - missing run -> `404 not_found`;
  - foreign existing run -> `403 forbidden`.
- Зафиксировать public summary-level contract без detail/report payloads до R7-04.

## Контекст

- R7-01 обобщил PG jobs storage в canonical persisted run storage:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- R7-02 перевел `POST /backtests` на persisted `sync_inline` flow.
- R8-02 добавил explicit `background_auto` launch branch в тот же persisted-run storage.
- B3 сделал этот branch profile-aware:
  - medium exact runs can persist as `execution_mode=sync_inline` + `execution_profile_mode=exact_parallel`;
  - heavy-but-valid exact runs can be queued earlier with `execution_mode=background_auto`
    while keeping exact semantics and persisted effective profile metadata.
- R7-03 поверх этого storage добавляет public history/status/top/cancel contract.

## Ключевые решения

### 1) Public vocabulary: `run_id` и `/backtests/runs*`

Внешний contract использует:

- `GET /backtests/runs`
- `GET /backtests/runs/{run_id}`
- `GET /backtests/runs/{run_id}/top`
- `POST /backtests/runs/{run_id}/variant-report`
- `POST /backtests/runs/{run_id}/cancel`

Storage/domain внутри bounded context может сохранять `job_*` naming, но API наружу использует
только `run_*`.

### 2) История и cursor semantics остаются детерминированными

Для `GET /backtests/runs`:

- ordering фиксирован: `created_at DESC, job_id DESC`;
- cursor payload совпадает с legacy jobs contract:
  `base64url(canonical_json({created_at, job_id}))`;
- `state` filter literals:
  - `queued`
  - `running`
  - `succeeded`
  - `failed`
  - `cancelled`

### 3) History/status payload показывает key metadata, но не внутренние hashes

Public runs payload не должен протекать внутренними reproducibility hashes из legacy jobs API.

Минимальные public metadata fields:

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
- `artifact_asof_date`

Hashes (`request_hash`, `engine_params_hash`, `backtest_runtime_config_hash`, `spec_hash`) остаются
частью legacy `/backtests/jobs*`, но не требуются public `/backtests/runs*`.

`execution_mode` после R8-02 может быть:

- `sync_inline`
- `background_auto` as the canonical background path for new queued runs
- `background_manual_legacy` as a `compatibility-only` literal for already persisted rows

### 4) `/top` остается summary-only

`GET /backtests/runs/{run_id}/top`:

- читает rows в deterministic order `rank ASC, variant_key ASC`;
- возвращает только summary payload:
  - `payload`
  - `summary_metrics_json`
  - `best_tp_pct`
  - `best_sl_pct`
- не возвращает:
  - `report_table_md`
  - `trades`
  - `equity`
- new history/summary web UX может локально пересортировывать уже загруженные rows по approved
  summary columns, но server payload первого render-а остаётся canonical ordering source.
- summary page может добавлять only-browser actions (`Open detail`, `Save as Strategy`), но не
  materialize'ит detail/report/trades inline.

### 5) Cancel endpoint idempotent

`POST /backtests/runs/{run_id}/cancel`:

- `queued -> cancelled` сразу;
- `running -> cancel_requested_at` (best-effort);
- terminal states возвращаются без изменений;
- ответ всегда содержит status snapshot, а не `204`.
- repeated cancel для уже помеченного `running` не должен перетирать первый
  `cancel_requested_at`.

### 6) Legacy `/backtests/jobs*` остается compatibility alias

На период миграции legacy endpoints продолжают работать поверх тех же use-case/repository rules.

| Legacy endpoint | Public endpoint | Notes |
| --- | --- | --- |
| `GET /backtests/jobs` | `GET /backtests/runs` | same keyset ordering and cursor semantics |
| `GET /backtests/jobs/{job_id}` | `GET /backtests/runs/{run_id}` | same owner policy; public payload hides hashes |
| `GET /backtests/jobs/{job_id}/top` | `GET /backtests/runs/{run_id}/top` | same deterministic summary rows; legacy payload may carry extra compatibility fields |
| `POST /backtests/jobs/{job_id}/cancel` | `POST /backtests/runs/{run_id}/cancel` | same idempotent cancel semantics |

## Endpoint contracts

### 1) `GET /backtests/runs`

Request params:

- `state` optional enum:
  - `queued`
  - `running`
  - `succeeded`
  - `failed`
  - `cancelled`
- `limit` optional, default `50`, max `250`
- `cursor` optional opaque base64url string

Response (`200 OK`):

- `items[]` ordered by `created_at DESC, run_id DESC`
- `next_cursor` opaque string or `null`

Каждый item содержит:

- `run_id`
- `mode`
- `state`
- `stage`
- `created_at`, `updated_at`, `started_at`, `finished_at`, `cancel_requested_at`
- `processed_units`, `total_units`
- `progress_percent`
- `eta_seconds` (`throughput first`, `benchmark fallback second`, `null last`)
- `execution_mode`
- `execution_profile_mode`
- `market_id`, `symbol`, `timeframe`
- `requested_top_n`
- `ranking_primary_metric`, `ranking_secondary_metric`
- `artifact_slot`, `artifact_slot_generation`, `artifact_manifest_hash`, `artifact_asof_date`

### 2) `GET /backtests/runs/{run_id}`

Response (`200 OK`):

- полный status/progress snapshot run;
- key metadata из history list;
- additive progress fields:
  - `progress_percent`
  - `eta_seconds`
  - `execution_profile_mode`
- для `failed` дополнительно:
  - `last_error`
  - `last_error_json` (`code/message/details`)

### 3) `GET /backtests/runs/{run_id}/top?limit=...`

Request params:

- `limit` optional; default = `backtest.jobs.top_k_persisted_default`
- validation:
  - `limit > 0`
  - `limit <= backtest.jobs.top_k_persisted_default`

Response (`200 OK`):

- `run_id`
- `state`
- `execution_mode`
- `items[]`

Каждый item содержит:

- `rank`
- `variant_key`
- `indicator_variant_key`
- `variant_index`
- `total_return_pct`
- `payload`
- `summary_metrics_json`
- `best_tp_pct`
- `best_sl_pct`

### 4) `POST /backtests/runs/{run_id}/cancel`

Response (`200 OK`):

- status snapshot после попытки cancel

### 5) `POST /backtests/runs/{run_id}/variant-report`

Request body:

- `variant` required:
  - explicit selected variant payload from summary row:
    - `indicator_selections`
    - `signal_params`
    - `risk_params`
    - `execution_params`
    - `direction_mode`
    - `sizing_mode`
- `include_trades` optional bool, default `false`

Request invariants:

- client не присылает `time_range`, `strategy_id`, `template`, `overrides` или любой другой
  full run envelope;
- backend восстанавливает original request context из persisted `request_json`;
- saved-mode runs восстанавливают effective template из `spec_payload_json + overrides`;
- runtime context pin'ится строго по persisted
  `artifact_slot/artifact_slot_generation/artifact_manifest_hash/artifact_asof_date`;
- endpoint пересчитывает ровно один выбранный вариант и не запускает full top-N recompute.

Response (`200 OK`):

- `rows`
- `table_md`
- `trades` optional (when `include_trades=true`)

## Инварианты

- Все operations owner-only.
- Missing run -> `404 not_found`.
- Foreign existing run -> `403 forbidden`.
- History ordering фиксирован: `created_at DESC, job_id DESC`.
- Top ordering фиксирован: `rank ASC, variant_key ASC`.
- Public payload не хранит и не отдает persisted detail/report bodies.
- `POST /backtests/runs/{run_id}/variant-report` сохраняет explicit owner policy:
  missing run -> `404`, foreign existing run -> `403`.

## Связанные файлы

- `apps/api/routes/backtest_runs.py`
- `apps/api/dto/backtest_runs.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`
- `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `tests/unit/apps/api/test_backtest_runs_routes.py`
- `tests/unit/apps/api/test_backtest_runs_dto.py`
- `tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_history_api_v1.py`
