---
doc: rl-trading-agent-platform-v1-stage-04b-binance-futures-history-backfill
stage: "04B"
status: in_progress
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-21"
---

# Stage 04B: Binance Futures Historical Backfill And Coverage

Статус: `in_progress`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `04B` был скорректирован после 2026-06-21 review: старый completed backfill на `215` Stage `04A` symbols остается reusable partial progress, но не является финальной целевой вселенной. Актуальная цель Stage `04B` - все текущие Binance USD-M Futures symbols с `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`.

Live metadata snapshot на Mac Studio подтвердил `528` current symbols. Missing supplement к старому Stage `04A` set: `313` symbols. Whitelist/ref/enrichment синхронизированы до `528/528`, новый full-current repair manifest создан, и managed resumable backfill запущен поверх того же `RestFillRange1mUseCase`.

## Scope

Входит:

- проверить Stage `04A` как accepted prerequisite, но не использовать его как финальную вселенную;
- зафиксировать prompt path/hash и planned file list;
- расширить `configs/prod/whitelist.csv` для всех live Binance Futures USDT perpetual symbols;
- выполнить whitelist -> `ref_instruments` sync и Binance Futures enrichment на Mac Studio;
- построить full-current dry-run/resume manifest с reuse старого completed chunk state;
- запустить managed parallel/sharded background backfill через existing `RestFillRange1mUseCase`;
- доказать старт ingestion и оставить Stage `04B` `in_progress` до full coverage.

Не входит:

- feature slabs, session extraction, dataset refresh manifest, model training;
- private Binance endpoints, account/order state, paper/testnet/live/mainnet submit;
- browser/UI/API/schema changes;
- accepted Stage `04B` coverage claim before all chunks complete.

## Methodology

| Поле | Значение |
|---|---|
| Тип задачи | Historical market-data repair/supplement backfill with coverage gate. |
| Метод | Live `exchangeInfo` universe -> whitelist/ref/enrichment sync -> per-symbol source-window manifest -> resumable chunk execution -> later full coverage report. |
| Целевая вселенная | Binance USD-M Futures `TRADING` + `PERPETUAL` + `quoteAsset=USDT`, live count wins at run time. |
| Source lower bound | Existing Stage `04A` source bounds for old `215`; for supplement use `max(2020-01-13T22:30:00Z, exchangeInfo.onboardDate)`. |
| Execution path | Existing `RestCandleIngestSource`, `RestFillRange1mUseCase`, `ClickHouseRawKlineWriter`, `ClickHouseCanonicalCandleIndexReader`. |
| Safety | Half-open UTC ranges, deterministic chunk ids, previous completed chunk reuse only when chunk id and market/symbol/window identity match. |
| Acceptance risk | Current evidence is start/progress only; full residual-gap decision waits for coverage report after all chunks complete. |

## File Manifest

Planned concrete file list before implementation edits:

- `scripts/rl_trading/stage04b_binance_futures_history_backfill.py`
- `tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py`
- `configs/prod/whitelist.csv`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| - | `scripts/rl_trading/stage04b_binance_futures_history_backfill.py` | - | Plan all current futures USDT perpetual symbols and merge completed state from previous manifest. | `compatible-change` opt-in operator script |
| - | `tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | - | Cover supplement planning and previous completed chunk reuse. | `compatible-change` test-only |
| - | `configs/prod/whitelist.csv` | - | Add missing `market_id=2` enabled rows for the 313-symbol supplement. | `compatible-change` production config contents; CSV schema unchanged |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md` | - | Replace stale 215-symbol report with full-current repair evidence. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Keep Stage `04B` `in_progress` and record full-current repair start. | `compatible-change` docs/ledger only |

The checkout already had unrelated local changes before this stage; they were not reverted or swept into this work.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` |
| Prompt sha256 | `5a0d768f18f789e7c72b694f70cd14763fffb95a323807cb4a8038566d74cc13` |
| Ledger state before repair | Stage `04A` accepted; Stage `04B` `in_progress`; old 215-symbol manifest completed but full-current supplement pending |
| Delivery state | Code/config changes are local checkout changes and were also copied to Mac Studio checkout/runtime for operator proof; no branch, PR, merge, or main delivery was performed in this run. |
| Runtime state root | `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/` |

## Universe And Ref Sync

Live metadata check on Mac Studio:

| Metric | Value |
|---|---:|
| Current Binance Futures USDT perpetual symbols | `528` |
| Current symbols sha256 | `a921fb1faea04411d3a07f596dafb804ba8be85ec7a8686e92dce37a30bce85c` |
| Stage `04A` old accepted count | `215` |
| Stage `04A` still active now | `215` |
| Stage `04A` stale now | `0` |
| Supplement symbols | `313` |
| Supplement symbols sha256 | `7e6ba956e5df0ac022922423cf27d09001ed03ab7bf8c2d0ff09244bc63c7cdc` |

Whitelist/ref/enrichment evidence after applying the supplement:

| Metric | Value |
|---|---:|
| `configs/prod/whitelist.csv` rows | `591` |
| Added `market_id=2` rows | `313` |
| `ref_instruments` `market_id=2`, `status=ENABLED` | `528` |
| `ref_instruments` enabled/tradable | `528` |
| Enriched core rows (`base_asset`, `quote_asset`, `price_step`, `qty_step`) | `528` |
| `min_notional` rows | `528` |

## Backfill Path

Selected command remains the dedicated Stage `04B` operator runner, but it does not implement a separate candle downloader. Each worker constructs the same existing market-data path:

- `RestCandleIngestSource` for public Binance Futures `/fapi/v1/klines`;
- `RestFillRange1mUseCase` for explicit 1m half-open ranges;
- `ClickHouseRawKlineWriter` for `raw_binance_klines_1m`;
- `ClickHouseCanonicalCandleIndexReader` for canonical skip/read-back checks.

Coordinator mode:

| Mode | Status | Safety model |
|---|---|---|
| Sequential `--workers 1` | Default / backward compatible | One chunk at a time. |
| Sharded parallel `--workers N` | Opt-in operator mode; current repair run uses `--workers 20` | `ThreadPoolExecutor` over the same use case, thread-local ClickHouse clients, retry/backoff, deterministic manifest. |

## Full-Current Repair Manifest

New runtime plan/resume manifest:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest.json`

| Metric | Value |
|---|---:|
| Symbols planned | `528` |
| Symbols reused from Stage `04A` | `215` |
| Supplement symbols planned | `313` |
| Stale Stage `04A` symbols | `0` |
| Chunks total | `62,048` |
| Completed chunks reused from previous manifest | `44,731` |
| Pending chunks after reuse | `17,317` |
| Chunk list sha256 | `2fd928d4148a3ec22b98dffcb357f2c57e34800a34e01c98bf11655eb762c57c` |
| Reuse rule | Copy only `completed` chunks with matching chunk id and identical market/symbol/window identity. |

Previous 215-symbol manifest retained as partial progress:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_history_backfill/stage04b_backfill_resume_manifest.json`

| Metric | Value |
|---|---:|
| Old manifest execution status | `completed` |
| Old chunks completed | `44,946/44,946` |
| Old rows read | `194,707,550` |
| Old rows written | `107,557,928` |

## Start-Proof Evidence

Managed repair job:

| Field | Value |
|---|---|
| PID file | `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair.pid` |
| Initial PID | `19909` (`workers=12`) |
| Current PID | `20101` (`workers=20`) |
| Manifest | `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest.json` |
| JSONL log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_run.jsonl` |
| Nohup log | `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_nohup.log` |
| Command bound | `--max-runtime-seconds 28800`, `--max-chunk-attempts 3`, `--failure-backoff-s 60`, `--skip-covered-chunks` |

Observed progress after restart:

| Metric | Value |
|---|---:|
| Observed at | `2026-06-21T15:18:10Z` |
| Execution status | `running` |
| Coordinator | `sharded_parallel` |
| Workers | `20` |
| Chunks completed | `45,007` |
| Running chunks | `18` |
| Pending chunks | `17,023` |
| Failed chunks | `0` |
| Retried chunks | `1` |
| Skipped-covered chunks | `6` |
| Rows read | `196,528,160` |
| Rows written | `109,625,765` |
| Current process CPU sample | about `108%` for PID `20101` |

The first stable `workers=20` sample showed `86` completed-or-skipped chunk events over `110s`, about `46.9 chunks/min`; `80` of those were real `chunk_completed` events and `6` were `chunk_skipped_covered`. Raising from `12` to `20` workers did not reach the requested `80 chunks/min`, because one Binance HTTP retry exhaustion triggered a `60s` backoff; the coordinator recovered automatically and resumed with no terminal failed chunks.

Example real supplement completions observed in the JSONL log:

| Symbol | Dataset version | Range | Rows read | Rows written |
|---|---|---|---:|---:|
| `1000000BOBUSDT` | `post_hf_extension_current_trading` | `[2025-08-28T08:30:00Z, 2025-09-04T08:30:00Z)` | `10,080` | `10,080` |
| `1000000BOBUSDT` | `post_hf_extension_current_trading` | `[2025-09-04T08:30:00Z, 2025-09-11T08:30:00Z)` | `10,080` | `10,080` |
| `1000000MOGUSDT` | `hf_period_rebuild_current_trading` | `[2025-05-29T12:30:00Z, 2025-06-01T01:00:00Z)` | `3,630` | `3,630` |
| `1000CATUSDT` | `hf_period_rebuild_current_trading` | `[2025-03-03T12:30:00Z, 2025-03-10T12:30:00Z)` | `10,080` | `10,080` |

## Coverage Status

Full coverage report is not available yet and must not be inferred from start-proof evidence.

Follow-up command after the background job finishes or pauses:

```bash
cd /opt/roehub/app
set -a; source /Users/daniildegtyarev/.config/roehub/roehub.env; set +a
.venv/bin/python scripts/rl_trading/stage04b_binance_futures_history_backfill.py coverage \
  --manifest /opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest.json \
  --output-json /opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_coverage_report.json
```

Acceptance criteria:

- every planned symbol/window has `missing_minutes=0`;
- `duplicate_rows=0`;
- `volume_quote_rows == physical_rows`;
- `trades_count_rows == physical_rows`;
- `vwap_computable_rows + zero_volume_rows == physical_rows`;
- residual gaps are explicit and no candles are synthesized.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API changed. |
| Port contract | `none` | Existing market-data ports and `RestFillRange1mUseCase` are reused unchanged. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or table shape change. |
| Persisted rows / identity semantics | `compatible-change` | Adds raw/canonical market-data rows and `ref_instruments` rows for live Binance Futures symbols via existing append/materialization paths. |
| Config schema/defaults | `compatible-change` | `configs/prod/whitelist.csv` contents changed; CSV schema unchanged. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Public Binance REST only; no private/account/order endpoint. Operator run uses bounded retries/backoff. |
| External side-effect idempotency/unknown-state | `compatible-change` | Resume manifest plus canonical skip/read-back support interrupted/retried chunks. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds sanitized runtime JSON manifest/log evidence; no raw provider payloads or secrets in docs/git. |
| Browser-visible behavior | `none` | No browser/UI behavior changed. |
| Performance/hot path | `none` | Offline operator backfill only; live execution/inference hot paths unchanged. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` | passed; `5a0d768f18f789e7c72b694f70cd14763fffb95a323807cb4a8038566d74cc13` |
| `uv run pytest -q tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed: `10 passed`; Mac Studio checkout passed: `10 passed` |
| `uv run ruff check scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed; Mac Studio checkout passed |
| `uv run pyright scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed; Mac Studio checkout passed: `0 errors` |
| Mac Studio live universe check | passed; `528` current symbols, `313` supplement |
| Mac Studio whitelist/ref/enrichment sync | passed; `528/528` enabled/tradable/enriched for `market_id=2` |
| Mac Studio full-current manifest generation | passed; `62,048` chunks, `44,731` reused completed, `17,317` pending |
| Mac Studio managed backfill start proof | passed; PID `20101` running, `workers=20`, `failed=0`, rows written increasing |

## Blockers And Handoff

Stage `04B` remains `in_progress`.

Stage `04C` is not allowed yet because:

- the full-current historical backfill has not completed;
- full per-symbol/source-window coverage report has not been generated;
- residual gaps, duplicates, and feature-field coverage are still unknown for the complete `528`-symbol source-window set.

Next operator handoff:

1. Continue monitoring `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest.json`.
2. If the process stops with pending chunks, resume with the same `execute --manifest ...` command.
3. After all chunks complete, run the `coverage` command above.
4. Accept Stage `04B` only if the coverage report is clean; otherwise keep Stage `04B` `in_progress` or `blocked` with exact residual gaps.
5. Start Stage `04C` only after accepted coverage exists for the full current Binance Futures USDT perpetual universe.
