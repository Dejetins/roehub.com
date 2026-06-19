---
doc: rl-trading-agent-platform-v1-stage-04b-binance-futures-history-backfill
stage: "04B"
status: in_progress
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-19"
---

# Stage 04B: Binance Futures Historical Backfill And Coverage

Статус: `in_progress`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `04B` стартовал managed resumable/background backfill для accepted Stage `04A` universe. Это start-only/in-progress evidence: ClickHouse ingestion доказан через rows/high-watermark deltas и продолжающийся resumable coordinator, но полный historical load и full coverage report еще не завершены. Stage `04C` остается blocked до completed coverage manifest.

## Scope

Входит:

- проверить prerequisite Stage `04A`;
- записать prompt path/hash и concrete planned file list до правок;
- выбрать safe explicit REST range path для `binance:futures`;
- создать dry-run/resume manifest для accepted Stage `04A` symbols и source windows;
- запустить bounded managed/background backfill на Mac Studio;
- доказать в ClickHouse, что historical rows начали появляться;
- обновить stage report и ledger как `in_progress`.

Не входит:

- изменение Stage `04A` whitelist/universe;
- feature slabs, session extraction, dataset refresh manifest, model training;
- private Binance endpoints, exchange/account actions, paper/testnet/live/mainnet submit;
- browser/UI/API/schema changes;
- direct GitHub branch/PR/main delivery.

## Методология Анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`, data-quality/backfill coverage gate. |
| Тип задачи | Historical market-data fill start proof and resumable coverage planning. |
| Выбранная методология | Source-window planning + current metadata recheck + chunked resumable execution + ClickHouse read-back delta. |
| Простое объяснение метода | Берем accepted symbols из Stage `04A`, повторно проверяем, что они still current Binance USD-M Futures USDT perpetuals, строим half-open UTC chunks только внутри allowed source windows and symbol history lower bounds, затем пишем через existing REST/raw writer path. |
| Бизнес-язык | Stage доказывает, что Roehub начал загружать реальные минутные Binance Futures candles для будущего RL dataset, не синтезируя gaps и не включая delisted/non-accepted symbols. |
| Единица анализа | `binance:futures` symbol + dataset source window + chunk. |
| Основные метрики | Planned symbols/chunks/minutes, completed chunks, rows read/written, first/last candle, missing minutes, duplicates, `volume_quote`, `trades_count`, zero-volume rows, `vwap` computability. |
| Прокси-метрики | Start proof uses row-count and first/high-watermark movement in ClickHouse before full completion. |
| Период анализа | `hf_period_rebuild_current_trading` and `post_hf_extension_current_trading` source windows. |
| Основные фильтры | Accepted Stage `04A` symbols only; current `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`; `market_id=2`. |
| Риски интерпретации | Start proof is not full coverage. Full residual-gap decision remains open until all chunks complete and coverage report is generated. |
| Проверки перед выводом | Focused unit tests/ruff, Mac Studio dry-run manifest, background PID/log/resume manifest, ClickHouse row/high-watermark delta. |
| Вопросы до расчетов | Нет; Stage `04A` accepted and public market-data REST plus host-local ClickHouse environment were available. |

## File Manifest

Planned concrete file list before implementation edits:

- `scripts/rl_trading/stage04b_binance_futures_history_backfill.py`
- `tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/stage04b_binance_futures_history_backfill.py` | - | - | Stage `04B` operator runner with `plan`, `execute`, and `coverage` modes. | `compatible-change` opt-in operator script; writes market-data rows only when explicitly run |
| `tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | - | - | Focused tests for source-window planning, stale symbol exclusion, and coverage counters. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md` | - | - | Stage `04B` report. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `04B` `in_progress` start proof and keep Stage `04C` blocked. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Generated docs index only if required by docs check. | `compatible-change` docs index only |

Outside expected paths: none for Stage `04B`. The checkout already had unrelated local changes before this stage; they were not reverted or swept into this work.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` |
| Prompt sha256 | `899a79b0664852cc576b7e343d820c9b4a8d2143fd8452e158d1c604085d3c25` |
| Ledger state before implementation | Stage `04A` accepted; `current_stage=04B`; Stage `04B` pending |
| Required prerequisite | Stage `04A` accepted |
| Delivery state | Report/ledger changes are `local-only`; no branch, PR, merge or app deploy was performed in this run. Local checkout is on `80327d15`; Mac Studio checkout was behind at `fe956e0a`, so the Stage `04B` runner/test files were copied there as untracked runtime tooling for proof only. |
| Large/runtime artifacts | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/`; no raw candle dumps, raw provider payloads, secrets or credentials are committed. |

## Selected Backfill Path

Selected path: `scripts/rl_trading/stage04b_binance_futures_history_backfill.py execute` wraps existing market-data ports:

- `RestCandleIngestSource` for public Binance Futures `/fapi/v1/klines`;
- `RestFillRange1mUseCase` for explicit 1m half-open ranges;
- `ClickHouseRawKlineWriter` for `raw_binance_klines_1m`;
- `ClickHouseCanonicalCandleIndexReader` for dedup/read-back guardrails;
- existing ClickHouse materialization path for canonical rows.

Coordinator mode:

| Mode | Status | Safety model |
|---|---|---|
| Sequential `--workers 1` | Default / backward compatible | Preserves original one-chunk-at-a-time behavior. |
| Sharded parallel `--workers N` | Opt-in operator mode; Stage `04B` runtime uses `--workers 4` | Uses `ThreadPoolExecutor` over the same `RestFillRange1mUseCase`; ClickHouse access uses `ThreadLocalClickHouseConnectGateway`; scheduler permits at most one active chunk per `symbol` shard, so overlapping source windows for the same symbol are not processed concurrently. |

The parallel coordinator does not implement a separate candle downloader. Each worker builds the same REST source, raw writer, canonical index reader, and `RestFillRange1mUseCase` wiring used by the sequential runner.

Rejected alternatives:

| Alternative | Decision |
|---|---|
| `apps/cli/commands/backfill_1m.py` | Rejected for Stage `04B`: current CLI is parquet/file oriented and requires `--parquet`; it is not the direct public REST historical bootstrap runner. |
| `apps/cli/commands/rest_catchup_1m.py` | Rejected for initial bootstrap: it raises on missing canonical seed and is intended for seeded tail/gap maintenance. |
| Scheduler queue only | Not used as the stage entrypoint because Stage `04B` needs a durable per-symbol resume manifest, start-proof paths, and bounded operator command. The runner still reuses the same fill use case and raw writer path. |

## Parallel Coordinator Restart

On `2026-06-19`, the initial sequential long run was stopped and replaced with the opt-in sharded coordinator because the observed runtime had only one active chunk at a time.

| Field | Value |
|---|---|
| Sequential PID stopped | `4579` via `SIGTERM` |
| Sequential baseline sample | `2026-06-19T08:15:34Z` to `2026-06-19T08:16:34Z`: `10` completed chunks/minute, `97,985` rows read/minute, `0` rows written/minute in that sample |
| First parallel command | `execute --workers 4 --max-runtime-seconds 21600 --delay-s 0.2` |
| First parallel PID | `18406` |
| Launch correction | First restart attempt PID `18177` did not start because non-login `zsh -s` lacked `uv` in `PATH`; final command uses `/opt/homebrew/bin/uv`. |
| First parallel manifest proof | `coordinator=sharded_parallel`, `workers=4`, `running=4`, active shards `AAVEUSDT`, `ACEUSDT`, `ACHUSDT`, `ADAUSDT` |
| First parallel result | Too aggressive: after ~60s, all four active shards hit `HTTP request failed after retries url=https://fapi.binance.com/fapi/v1/klines`; process stopped fail-closed with `4` failed chunks. This is treated as a rate/backoff ceiling, not accepted speed evidence. |
| Current parallel command | `execute --workers 2 --max-chunk-attempts 3 --failure-backoff-s 60 --max-runtime-seconds 21600 --delay-s 0.2` |
| Current parallel PID | `18688` |
| Current safety adjustment | Bounded retry/backoff is explicit for this run. Default CLI remains fail-fast with `--max-chunk-attempts 1`. |
| Contract impact | `compatible-change`: opt-in operator concurrency only; public API, DB schema, DTOs, and `RestFillRange1mUseCase` contract unchanged. |

## Dry-Run Manifest

Runtime plan/resume manifest:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json`

| Metric | Value |
|---|---:|
| Manifest sha256 | `1dc3b304a628b3700c4fa4c9aa05a0d0aba6e4e87337fc14e53f12439507e89b` |
| Chunk list sha256 | `23c4ddc56a92b8a70693ffed42696da45d6457eff58927a803eccb0d46fca9c4` |
| Manifest size | `14M` |
| Active Stage `04A` symbols | `215` |
| Stale Stage `04A` symbols | `0` |
| Planned chunks | `44,946` |
| Expected minute-symbol-window rows | `451,395,870` |
| Latest Binance Futures candle snapshot | `2026-06-18T23:34:00Z` |
| Post-HF source end | `2026-06-18T23:35:00Z` |

Dataset source windows:

| Dataset version | Source start | Source end |
|---|---|---|
| `hf_period_rebuild_current_trading` | `2020-01-13T22:30:00Z` | `2025-06-01T01:00:00Z` |
| `post_hf_extension_current_trading` | `2025-05-31T22:30:00Z` | `2026-06-18T23:35:00Z` |

The first pending chunk before execution was:

| Field | Value |
|---|---|
| `chunk_id` | `e51cc95d3f89532319beabac` |
| `symbol` | `1000BONKUSDT` |
| `dataset_version` | `hf_period_rebuild_current_trading` |
| `start_utc` | `2023-11-22T14:00:00Z` |
| `end_utc` | `2023-11-29T14:00:00Z` |
| `expected_minutes` | `10,080` |

## Start-Proof Evidence

Managed/background command state:

| Field | Value |
|---|---|
| PID file | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill.pid` |
| Initial observed PID | `76736` |
| Current observed PID | `18688` |
| Current process state | `running` with `coordinator=sharded_parallel`, `workers=2`, `max_chunk_attempts=3` |
| Initial background self-stop bound | `--max-runtime-seconds 600`; current restart bound is `--max-runtime-seconds 21600` |
| Resume manifest | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json` |
| JSONL log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_run.jsonl` |
| Nohup log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_nohup.log` |
| Latest `ingest_id` | `b3847ad9-b9c5-439a-b6f5-14c5b216456e` |

Resume manifest after bounded observation:

| Metric | Value |
|---|---:|
| Execution status | `paused_with_pending_chunks` |
| Chunks completed | `6` |
| Current failed chunks | `0` |
| Pending chunks | `44,940` |
| Rows read | `60,480` |
| Rows written | `60,480` |
| Batches written | `12` |
| Started at | `2026-06-18T23:36:12Z` |
| Updated at | `2026-06-18T23:40:40Z` |

One transient `RuntimeError` event occurred on chunk `6346bcfab2c11d38e852cd3b`; a bounded `--max-chunks 1` retry completed that same chunk successfully. The manifest currently has `0` chunks with `status=failed`.

Current sharded run initial observation after successful restart:

| Metric | Value |
|---|---:|
| Observed at | `2026-06-19T08:25:18Z` |
| PID | `18688` |
| Coordinator | `sharded_parallel` |
| Workers | `2` |
| Max chunk attempts | `3` |
| Chunks completed | `1,647` |
| Running chunks | `2` |
| Pending chunks | `43,297` |
| Current failed chunks | `0` |
| Running shards | `AAVEUSDT`, `ACEUSDT` |

Short runtime performance sample:

| Mode | Window | Completed chunks/min | Rows read/min | Rows written/min | Failed chunks | Retries |
|---|---|---:|---:|---:|---:|---:|
| Sequential baseline, PID `4579` | `2026-06-19T08:15:34Z` to `2026-06-19T08:16:34Z` | `10` | `97,985` | `0` | `0` | `0` |
| Sharded parallel, PID `18688`, `workers=2` | `2026-06-19T08:25:32Z` to `2026-06-19T08:26:33Z` | `16` | `161,280` | `0` | `0` | `0` |

Interpretation: the accepted current operator setting is `workers=2`, which showed about `+60%` chunks/minute and `+65%` rows-read/minute on a short runtime sample without retries or failed chunks. The `workers=4` sample is rejected as unsafe for the current external REST/runtime environment because it produced simultaneous HTTP retry exhaustion.

Latest runtime recheck:

| Metric | Value |
|---|---:|
| Observed at | `2026-06-19T08:30:03Z` |
| PID | `18688` |
| Execution status | `running` |
| Chunks completed | `1,700` |
| Running chunks | `2` |
| Pending chunks | `43,244` |
| Failed chunks | `0` |
| Retried chunks | `1` |
| Rows read | `17,055,362` |
| Rows written | `1,131,623` |

The single transient retry recovered after the configured `60s` backoff; the process resumed with two active shards and no terminal failed chunks.

Completed chunks observed in log:

| Symbol | Dataset version | Range | Rows read | Rows written |
|---|---|---|---:|---:|
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-11-22T14:00:00Z, 2023-11-29T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-11-29T14:00:00Z, 2023-12-06T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-06T14:00:00Z, 2023-12-13T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-13T14:00:00Z, 2023-12-20T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-20T14:00:00Z, 2023-12-27T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-27T14:00:00Z, 2024-01-03T14:00:00Z)` | `10,080` | `10,080` |

ClickHouse row/high-watermark evidence:

| Symbol | Before rows | Before first | Before last | After rows | After first | After last |
|---|---:|---|---|---:|---|---|
| `1000BONKUSDT` | `192` | `2026-06-18 20:23:00.000` | `2026-06-18 23:34:00.000` | `60,677` | `2023-11-22 14:00:00.000` | `2026-06-18 23:39:00.000` |
| `BTCUSDT` | `3,564,337` | `2019-09-08 17:57:00.000` | `2026-06-18 23:34:00.000` | `3,564,339` | `2019-09-08 17:57:00.000` | `2026-06-18 23:36:00.000` |

Interpretation:

- Historical ingestion started for the first accepted Stage `04A` symbol.
- The first candle for `1000BONKUSDT` moved from current-tail-only `2026-06-18T20:23:00Z` back to `2023-11-22T14:00:00Z`.
- Stage `04B` is not accepted yet because `44,940` chunks remain pending and the full coverage report is not available.

## Coverage Status

Full coverage report was intentionally not used as acceptance evidence in this run. The prompt requires full coverage only after backfill completion; current evidence is start-only.

Follow-up command after the background job finishes or pauses:

```bash
cd /Users/daniildegtyarev/Projects/roehub.com
set -a; source /Users/daniildegtyarev/.config/roehub/roehub.env; set +a
uv run python scripts/rl_trading/stage04b_binance_futures_history_backfill.py coverage \
  --manifest /opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json \
  --output-json /opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_coverage_report.json
```

Coverage acceptance criteria for the follow-up:

- every active Stage `04A` symbol/window has `missing_minutes=0`;
- `duplicate_rows=0`;
- `volume_quote_rows == physical_rows`;
- `trades_count_rows == physical_rows`;
- `vwap_computable_rows + zero_volume_rows == physical_rows`;
- residual gaps are explicit and no candles are synthesized.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API changed. |
| Port contract | `none` | Existing `RestCandleIngestSource`, `RestFillRange1mUseCase`, `RawKlineWriter`, and canonical index reader contracts are reused unchanged. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No schema migration or table shape change. |
| Persisted rows / identity semantics | `compatible-change` | Adds raw/canonical market-data rows for accepted `binance:futures` symbols through existing append/materialization path; chunk IDs and resume status are deterministic runtime artifacts. |
| Config schema/defaults | `none` | No config schema/default changed. Host-local env was used only to connect to ClickHouse on Mac Studio. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Public Binance Futures REST calls use existing market-data config timeouts/retries; sharded operator run uses `--workers 2`; no private/account endpoint and no exchange-side effect. |
| External side-effect idempotency/unknown-state | `compatible-change` | Unknown-state retries are guarded by resume manifest plus canonical read-back dedup. Interrupted execution resumes pending chunks only. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds sanitized runtime JSON manifest/logs and docs summaries; no raw provider payloads or secrets in docs/git. |
| Alert/runbook semantics | `none` | No alert routing or service manager config changed. |
| Browser-visible behavior | `none` | Browser/runtime UI verification disabled by prompt and not relevant. |
| Performance/hot path | `none` | Operator backfill throughput changed, but the job is offline; no live inference/execution hot path changed. |

## Service-Call And Runtime Coverage

| Surface | Coverage |
|---|---|
| Public REST call | Covered: Binance USD-M Futures public `/fapi/v1/klines` through existing `RestCandleIngestSource`; no raw provider payloads are persisted in docs/git. |
| Private/auth exchange call | `N/A`: no Binance private/account endpoint, no order endpoint, no signed request, no exchange credential. |
| ClickHouse write | Covered: existing `ClickHouseRawKlineWriter` writes raw 1m rows; canonical materialization is existing market-data pipeline. No schema change. |
| Timeout/retry/backoff | Covered by existing `configs/prod/market_data.yaml` REST settings loaded by the same market-data source; this stage did not change defaults. |
| Rate/batch bounding | Covered: source requests remain 1,000-minute Binance windows; runner chunks source windows into seven-day chunks; raw insert batch size default is `10,000`; current sharded run uses `--workers 4`, one active chunk per symbol, `--delay-s 0.2`, and `--max-runtime-seconds 21600`. |
| Idempotency / retry after unknown state | Covered: chunk resume manifest records pending/running/completed/failed state; `RestFillRange1mUseCase` reads canonical minutes and skips existing rows before writing. |
| Redaction boundary | Covered: docs include counts, hashes, paths, PID and sanitized UUID only; secrets/env values and raw provider payloads are not copied into docs/git. |
| Alerts / runbook | `N/A`: no Monit/launchd/systemd, alert routing, or operational paging rule changed. |
| Browser / UI | `N/A`: no browser-visible behavior. |
| Mainnet / trading | `N/A`: no paper/testnet/live/mainnet submit and no exchange account state mutation. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` | passed; hash recorded above |
| `uv run pytest -q tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed: `7 passed`; Mac Studio passed: `7 passed` |
| `uv run pytest -q tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py tests/unit/contexts/market_data/application/use_cases/test_backfill_1m_candles.py tests/unit/contexts/market_data/application/use_cases/test_rest_catchup_1m.py` | passed: `16 passed` |
| `uv run ruff check scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed; Mac Studio passed |
| `uv run pyright scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed: `0 errors` |
| `uv run ruff check apps/cli src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data tests/unit/apps/cli tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | passed |
| Mac Studio dry-run manifest generation | passed; `215` active symbols, `0` stale symbols, `44,946` chunks |
| Mac Studio bounded background start proof | passed; PID `76736` launched, final manifest `paused_with_pending_chunks`, `6` chunks completed, `60,480` rows written, ClickHouse first/high-watermark moved |
| Mac Studio sharded restart proof | passed; PID `18688` running with `coordinator=sharded_parallel`, `workers=2`, `running=2`, `failed=0` |
| Mac Studio short performance sample | passed; sequential `10` chunks/min and `97,985` rows_read/min vs sharded `workers=2` `16` chunks/min and `161,280` rows_read/min; `0` failures/retries in accepted sample |
| CLI-specific pytest gate | not run; no CLI path changed |
| `uv run pyright scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | passed: `0 errors` |
| `python -m tools.docs.generate_docs_index && python -m tools.docs.generate_docs_index --check` | passed; generated index was unchanged and up to date |

## Blockers And Handoff

Stage `04B` remains `in_progress`.

Stage `04C` is not allowed yet because:

- the background historical backfill has not completed;
- full per-symbol/source-window coverage report has not been generated;
- residual gaps, duplicates, and feature-field coverage are still unknown for the complete source-window set.

Next operator handoff:

1. Let the managed Stage `04B` job stop on its `--max-runtime-seconds 600` bound or resume it with the same `execute --manifest ...` command.
2. Inspect `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json` for `execution.status` and pending chunks.
3. After all chunks complete, run the `coverage` command above.
4. Accept Stage `04B` only if the coverage report is clean; otherwise keep Stage `04B` `in_progress` or `blocked` with exact residual gaps.
5. Start Stage `04C` only after accepted coverage exists for at least one dataset refresh version.
