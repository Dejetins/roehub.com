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

Stage `04B` стартовал managed resumable/background backfill для accepted Stage `04A` universe. Это start-only evidence: ClickHouse ingestion доказан через новые rows/high-watermark deltas, но полный historical load и full coverage report еще не завершены. Stage `04C` остается blocked до completed coverage manifest.

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
| Delivery state | Repository changes are `local-only`; no branch, PR, merge, main delivery or app deploy. Runtime start proof ran on Mac Studio from copied local script/test files. |
| Large/runtime artifacts | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/`; no raw candle dumps, raw provider payloads, secrets or credentials are committed. |

## Selected Backfill Path

Selected path: `scripts/rl_trading/stage04b_binance_futures_history_backfill.py execute` wraps existing market-data ports:

- `RestCandleIngestSource` for public Binance Futures `/fapi/v1/klines`;
- `RestFillRange1mUseCase` for explicit 1m half-open ranges;
- `ClickHouseRawKlineWriter` for `raw_binance_klines_1m`;
- `ClickHouseCanonicalCandleIndexReader` for dedup/read-back guardrails;
- existing ClickHouse materialization path for canonical rows.

Rejected alternatives:

| Alternative | Decision |
|---|---|
| `apps/cli/commands/backfill_1m.py` | Rejected for Stage `04B`: current CLI is parquet/file oriented and requires `--parquet`; it is not the direct public REST historical bootstrap runner. |
| `apps/cli/commands/rest_catchup_1m.py` | Rejected for initial bootstrap: it raises on missing canonical seed and is intended for seeded tail/gap maintenance. |
| Scheduler queue only | Not used as the stage entrypoint because Stage `04B` needs a durable per-symbol resume manifest, start-proof paths, and bounded operator command. The runner still reuses the same fill use case and raw writer path. |

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
| Observed PID | `76736` |
| Process state at observation | `running`, elapsed `00:44` |
| Self-stop bound | `--max-runtime-seconds 600` |
| Resume manifest | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json` |
| JSONL log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_run.jsonl` |
| Nohup log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_nohup.log` |
| `ingest_id` | `f882a124-8ed1-403b-948c-a2e61f7aaa8e` |

Resume manifest after bounded observation:

| Metric | Value |
|---|---:|
| Execution status | `running` |
| Chunks completed | `5` |
| Chunks failed | `0` |
| Rows read | `50,400` |
| Rows written | `50,400` |
| Batches written | `10` |
| Started at | `2026-06-18T23:36:12Z` |
| Updated at | `2026-06-18T23:36:39Z` |

Completed chunks observed in log:

| Symbol | Dataset version | Range | Rows read | Rows written |
|---|---|---|---:|---:|
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-11-22T14:00:00Z, 2023-11-29T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-11-29T14:00:00Z, 2023-12-06T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-06T14:00:00Z, 2023-12-13T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-13T14:00:00Z, 2023-12-20T14:00:00Z)` | `10,080` | `10,080` |
| `1000BONKUSDT` | `hf_period_rebuild_current_trading` | `[2023-12-20T14:00:00Z, 2023-12-27T14:00:00Z)` | `10,080` | `10,080` |

ClickHouse row/high-watermark evidence:

| Symbol | Before rows | Before first | Before last | After rows | After first | After last |
|---|---:|---|---|---:|---|---|
| `1000BONKUSDT` | `192` | `2026-06-18 20:23:00.000` | `2026-06-18 23:34:00.000` | `50,594` | `2023-11-22 14:00:00.000` | `2026-06-18 23:36:00.000` |
| `BTCUSDT` | `3,564,337` | `2019-09-08 17:57:00.000` | `2026-06-18 23:34:00.000` | `3,564,339` | `2019-09-08 17:57:00.000` | `2026-06-18 23:36:00.000` |

Interpretation:

- Historical ingestion started for the first accepted Stage `04A` symbol.
- The first candle for `1000BONKUSDT` moved from current-tail-only `2026-06-18T20:23:00Z` back to `2023-11-22T14:00:00Z`.
- Stage `04B` is not accepted yet because `44,946` planned chunks are not completed and the full coverage report is not available.

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
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Public Binance Futures REST calls use existing market-data config timeouts/retries; no private/account endpoint and no exchange-side effect. |
| External side-effect idempotency/unknown-state | `compatible-change` | Unknown-state retries are guarded by resume manifest plus canonical read-back dedup. Interrupted execution resumes pending chunks only. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds sanitized runtime JSON manifest/logs and docs summaries; no raw provider payloads or secrets in docs/git. |
| Alert/runbook semantics | `none` | No alert routing or service manager config changed. |
| Browser-visible behavior | `none` | Browser/runtime UI verification disabled by prompt and not relevant. |
| Performance/hot path | `none` | Operator backfill job is offline; no live inference/execution hot path changed. |

## Service-Call And Runtime Coverage

| Surface | Coverage |
|---|---|
| Public REST call | Covered: Binance USD-M Futures public `/fapi/v1/klines` through existing `RestCandleIngestSource`; no raw provider payloads are persisted in docs/git. |
| Private/auth exchange call | `N/A`: no Binance private/account endpoint, no order endpoint, no signed request, no exchange credential. |
| ClickHouse write | Covered: existing `ClickHouseRawKlineWriter` writes raw 1m rows; canonical materialization is existing market-data pipeline. No schema change. |
| Timeout/retry/backoff | Covered by existing `configs/prod/market_data.yaml` REST settings loaded by the same market-data source; this stage did not change defaults. |
| Rate/batch bounding | Covered: source requests remain 1,000-minute Binance windows; runner chunks source windows into seven-day chunks; raw insert batch size default is `10,000`; background run used `--max-runtime-seconds 600`. |
| Idempotency / retry after unknown state | Covered: chunk resume manifest records pending/running/completed/failed state; `RestFillRange1mUseCase` reads canonical minutes and skips existing rows before writing. |
| Redaction boundary | Covered: docs include counts, hashes, paths, PID and sanitized UUID only; secrets/env values and raw provider payloads are not copied into docs/git. |
| Alerts / runbook | `N/A`: no Monit/launchd/systemd, alert routing, or operational paging rule changed. |
| Browser / UI | `N/A`: no browser-visible behavior. |
| Mainnet / trading | `N/A`: no paper/testnet/live/mainnet submit and no exchange account state mutation. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` | passed; hash recorded above |
| `uv run pytest -q tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed: `3 passed`; Mac Studio passed: `3 passed` |
| `uv run ruff check scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed; Mac Studio passed |
| Mac Studio dry-run manifest generation | passed; `215` active symbols, `0` stale symbols, `44,946` chunks |
| Mac Studio bounded background start proof | passed; PID `76736`, `5` chunks completed, `50,400` rows written, ClickHouse first/high-watermark moved |
| Full prompt REST/backfill pytest suite | pending; only new runner tests were run because no existing use-case behavior was changed |
| `python -m tools.docs.generate_docs_index --check` | pending after this report/ledger update |

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
