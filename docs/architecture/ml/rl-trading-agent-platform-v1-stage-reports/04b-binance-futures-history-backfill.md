---
doc: rl-trading-agent-platform-v1-stage-04b-binance-futures-history-backfill
stage: "04B"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 04B: Binance Futures Historical Backfill And Coverage

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `04B` был скорректирован после 2026-06-21 review: старый completed backfill на `215` Stage `04A` symbols остается reusable partial progress, но не является финальной целевой вселенной. Актуальная цель Stage `04B` - все текущие Binance USD-M Futures symbols с `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`.

Live metadata snapshot на Mac Studio подтвердил `528` current symbols. Missing supplement к старому Stage `04A` set: `313` symbols. Whitelist/ref/enrichment синхронизированы до `528/528`, full-current repair manifest завершил все `62,048/62,048` chunks. Follow-up repair подтвердил, что old residual missing minutes были ложным pre-history prefix из-за `exchangeInfo.onboardDate`, а не repairable candle loss: Binance public REST `klines limit=1` для `ICPUSDT` при start `2021-07-30T07:00:00Z` возвращает first candle `2022-09-27T02:30:00Z`. Planner и scheduler history-start policy теперь используют `max(stage04a_lower_bound, exchangeInfo.onboardDate, first_returned_binance_futures_1m_kline)`, canonical duplicates удалены существующим partition `OPTIMIZE ... FINAL` runbook path, и новый full coverage report clean: `coverage_status=accepted_coverage`, `0/886` windows blocked. Stage `04C` открыт.

## Scope

Входит:

- проверить Stage `04A` как accepted prerequisite, но не использовать его как финальную вселенную;
- зафиксировать prompt path/hash и planned file list;
- расширить `configs/prod/whitelist.csv` для всех live Binance Futures USDT perpetual symbols;
- выполнить whitelist -> `ref_instruments` sync и Binance Futures enrichment на Mac Studio;
- построить full-current dry-run/resume manifest с reuse старого completed chunk state;
- запустить managed parallel/sharded background backfill через existing `RestFillRange1mUseCase`;
- доказать старт ingestion, дождаться completed manifest state без повторного backfill и выполнить full coverage report;
- принять Stage `04B` только при clean coverage или оставить blocked/in_progress с точными residual gaps.

Не входит:

- feature slabs, session extraction, dataset refresh manifest, model training;
- private Binance endpoints, account/order state, paper/testnet/live/mainnet submit;
- browser/UI/API/schema changes;
- accepted Stage `04B` coverage claim before all chunks complete.

## Methodology

| Поле | Значение |
|---|---|
| Тип задачи | Historical market-data repair/supplement backfill with coverage gate. |
| Метод | Live `exchangeInfo` universe -> whitelist/ref/enrichment sync -> per-symbol source-window manifest -> resumable chunk execution -> full coverage report. |
| Целевая вселенная | Binance USD-M Futures `TRADING` + `PERPETUAL` + `quoteAsset=USDT`, live count wins at run time. |
| Source lower bound | For every current symbol use `max(required_source_start, Stage 04A lower bound when present, exchangeInfo.onboardDate, first returned Binance Futures 1m kline)`. |
| Execution path | Existing `RestCandleIngestSource`, `RestFillRange1mUseCase`, `ClickHouseRawKlineWriter`, `ClickHouseCanonicalCandleIndexReader`. |
| Safety | Half-open UTC ranges, deterministic chunk ids, previous completed chunk reuse only when chunk id and market/symbol/window identity match. |
| Acceptance result | First-kline-aware coverage is clean; Stage `04B` accepted and Stage `04C` opened. |

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
| - | `src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_history_start_source.py` | - | Confirm Binance Futures history start with first returned 1m kline so scheduler/rest maintenance does not retry unavailable pre-history prefixes. | `compatible-change` runtime planning semantics |
| - | `tests/unit/contexts/market_data/adapters/test_rest_instrument_history_start_source.py` | - | Regression coverage for Binance Futures first-kline history-start confirmation. | `compatible-change` test-only |
| - | `scripts/ops/optimize_canonical_partitions.sh` | - | Make the existing optimal canonical dedup script compatible with macOS bash and fail hard on duplicate-scan query errors. | `compatible-change` operator script |
| - | `configs/prod/whitelist.csv` | - | Add missing `market_id=2` enabled rows for the 313-symbol supplement. | `compatible-change` production config contents; CSV schema unchanged |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md` | - | Replace stale 215-symbol report with full-current repair evidence. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `04B` runtime evidence, accepted coverage status, and Stage `04C` handoff. | `compatible-change` docs/ledger only |

The checkout already had unrelated local changes before this stage; they were not reverted or swept into this work.

Continuation concrete file list for the 2026-06-22 coverage pass:

- `src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_history_start_source.py`
- `tests/unit/contexts/market_data/adapters/test_rest_instrument_history_start_source.py`
- `scripts/ops/optimize_canonical_partitions.sh`
- `scripts/rl_trading/stage04b_binance_futures_history_backfill.py`
- `tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` |
| Prompt sha256 | `15f6c9c6b3761a26d20335509fc7a41e03f302be7ab8b8c1afcd682e24f766e6` |
| Ledger state before repair | Stage `04A` accepted; Stage `04B` `in_progress`; old 215-symbol manifest completed but full-current supplement pending |
| Delivery state | Code/config changes are local checkout changes and the Stage `04B` repair scripts were copied to Mac Studio checkout for operator proof; no branch, PR, merge, or main delivery was performed in this run. |
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
| Manifest file sha256 | `8e8031f530e3f25c5b32b410743e72f648edeba3e3b19d20acdcf59135b3406f` |
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
| Observed PID during start proof | `20101` (`workers=20`) |
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
| Observed process CPU sample | about `108%` for PID `20101` |

The first stable `workers=20` sample showed `86` completed-or-skipped chunk events over `110s`, about `46.9 chunks/min`; `80` of those were real `chunk_completed` events and `6` were `chunk_skipped_covered`. Raising from `12` to `20` workers did not reach the requested `80 chunks/min`, because one Binance HTTP retry exhaustion triggered a `60s` backoff; the coordinator recovered automatically and resumed with no terminal failed chunks.

Example real supplement completions observed in the JSONL log:

| Symbol | Dataset version | Range | Rows read | Rows written |
|---|---|---|---:|---:|
| `1000000BOBUSDT` | `post_hf_extension_current_trading` | `[2025-08-28T08:30:00Z, 2025-09-04T08:30:00Z)` | `10,080` | `10,080` |
| `1000000BOBUSDT` | `post_hf_extension_current_trading` | `[2025-09-04T08:30:00Z, 2025-09-11T08:30:00Z)` | `10,080` | `10,080` |
| `1000000MOGUSDT` | `hf_period_rebuild_current_trading` | `[2025-05-29T12:30:00Z, 2025-06-01T01:00:00Z)` | `3,630` | `3,630` |
| `1000CATUSDT` | `hf_period_rebuild_current_trading` | `[2025-03-03T12:30:00Z, 2025-03-10T12:30:00Z)` | `10,080` | `10,080` |

## Completion Evidence

Manifest readback on Mac Studio:

| Metric | Value |
|---|---:|
| Observed at | `2026-06-22T19:33Z` |
| Execution status | `completed` |
| Finished at | `2026-06-22T01:12:46Z` |
| Chunk statuses | `62,048 completed / 62,048 total` |
| Rows read | `333,924,388` |
| Rows written | `242,824,220` |
| JSONL run log sha256 | `8059c025c4e4df70c8fcee3c14564426cfd927fd3ea480534404ba6e89421813` |

The PID file still existed, but the referenced process was no longer running. No backfill was restarted during the 2026-06-22 continuation.

## Root Cause And Data Repair

The superseded coverage report was:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_coverage_report.json`

It had `coverage_status=residual_gaps`, `113/886` blocked windows, `1,062,322` missing minutes and `65,700` duplicate rows. That report is retained as diagnostic evidence only; it is no longer the acceptance artifact.

Root-cause hypothesis was confirmed with public Binance REST and canonical readback:

- `exchangeInfo.onboardDate` is not always the first returned Binance Futures 1m candle.
- Example: `ICPUSDT` `onboardDate=2021-07-30T07:00:00Z`, but `GET /fapi/v1/klines?symbol=ICPUSDT&interval=1m&startTime=2021-07-30T07:00:00Z&limit=5` returns first candle `2022-09-27T02:30:00Z`.
- Therefore the old coverage counted unavailable pre-history prefixes as missing minutes.
- Stage `04B` planner and `RestInstrumentHistoryStartSource` now confirm Binance Futures history start with the first returned 1m kline and use `max(stage04a_lower_bound, exchangeInfo.onboardDate, first_returned_binance_futures_1m_kline)`.
- No synthetic candles were created.

Duplicate repair used the existing optimal canonical partition script:

- `scripts/ops/optimize_canonical_partitions.sh` now works on macOS bash without `mapfile` and fails hard if duplicate-partition scan fails.
- The script was run with `ENV_FILE=/tmp/roehub-no-env CH_PORT=9000` because the repo env file uses ClickHouse HTTP port `8123`, while this shell script calls native `clickhouse client`.
- `ZEREBROUSDT` was optimized through `INSTRUMENT_KEY=binance:futures:ZEREBROUSDT`, clearing `37` duplicate partitions.
- `IOTAUSDT`, `MEWUSDT`, `ZORAUSDT`, `RSRUSDT`, `USDCUSDT`, and `MAVIAUSDT` were optimized through `INSTRUMENT_KEY_COL=symbol`, clearing `7` additional duplicate partitions.
- Post-check query for all seven residual duplicate symbols returned no rows with `dup_rows > 0`.

First-kline-aware repair manifest:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest_first_kline.json`

| Metric | Value |
|---|---:|
| Manifest file sha256 | `1dcdbe29ae76f1a57d36a2bfc2ef56b1419a7fee461530387adde0ab87cbeb3b` |
| Symbols planned | `528` |
| Symbols reused from Stage `04A` | `215` |
| Supplement symbols planned | `313` |
| First-kline confirmed symbols | `528` |
| First-kline confirmed symbols sha256 | `a921fb1faea04411d3a07f596dafb804ba8be85ec7a8686e92dce37a30bce85c` |
| First-kline starts sha256 | `20ebea283ec7eb56ee12528a8f3da3a62f0764c0eb6f6eef03ff56af3147c5e4` |
| Chunks total | `61,949` |
| Completed chunks reused from previous manifest | `45,067` |
| Pending chunks after reuse | `16,882` |
| Expected minutes total | `618,533,473` |
| Chunk list sha256 | `298f0d87498cd75b2d55aa3a94c05eaa1683da566cba0f32d9eb5c4faf969726` |

Focused residual probe:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_first_kline_focused_residual_probe.json`

| Metric | Value |
|---|---:|
| Probe file sha256 | `6bf9ee0ec771fcc8507480ef7bc06ec87682553d3d5f72f695a86deeb2714bad` |
| Old blocked windows rechecked | `113` |
| Windows still blocked | `0` |
| Missing minutes | `0` |
| Duplicate rows | `0` |

Examples after first-kline repair:

| Symbol | Old problem | Accepted safe start | Missing minutes after repair | First candle |
|---|---|---|---:|---|
| `ICPUSDT` | `610,290` pre-history minutes | `2022-09-27T02:30:00Z` | `0` | `2022-09-27T02:30:00Z` |
| `SKLUSDT` | `152,675` pre-history minutes | `2020-12-08T07:35:00Z` | `0` | `2020-12-08T07:35:00Z` |
| `GTCUSDT` | `126,510` pre-history minutes | `2021-06-11T03:30:00Z` | `0` | `2021-06-11T03:30:00Z` |
| `MASKUSDT` | `60,270` pre-history minutes | `2021-08-27T03:30:00Z` | `0` | `2021-08-27T03:30:00Z` |

## Coverage Status

Accepted coverage report:

`/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_coverage_report_first_kline.json`

Execution note: the first-kline-aware JSON above was generated by a bounded read-only parallel equivalent using the same `schema_version=1`, `coverage_entry_from_row` semantics, and acceptance predicates as the Stage `04B` runner. The latest candle endpoint was kept at the original Stage `04B` manifest value `2026-06-21T15:09:00Z` to avoid mixing historical repair with later live tail maintenance.

| Metric | Value |
|---|---:|
| Coverage status | `accepted_coverage` |
| Coverage file sha256 | `aaf77389ba76ca001adfc3bfe4fc06efa0487dc407b47ea734228dca4c0738eb` |
| Entries sha256 | `25f9b5ccc744ebf873364a39442a6564ee8fb9e37d0fd4f3ab0767900c580608` |
| Collected at | `2026-06-22T21:27:22Z` |
| Latest Binance Futures candle in manifest | `2026-06-21T15:09:00Z` |
| Windows checked | `886` |
| Windows blocked | `0` |
| Expected minutes | `618,533,473` |
| Distinct minutes | `618,533,473` |
| Missing minutes | `0` |
| Duplicate rows | `0` |
| Zero-volume rows | `3,254,044` |

Coverage predicate result:

| Predicate | Status | Evidence |
|---|---|---|
| `missing_minutes=0` for every window | clean | `0` blocked windows; total `0`. |
| `duplicate_rows=0` for every window | clean | `0` blocked windows; total `0`. |
| `volume_quote_rows == physical_rows` | clean | `0` blocked windows. |
| `trades_count_rows == physical_rows` | clean | `0` blocked windows. |
| `vwap_computable_rows + zero_volume_rows == physical_rows` | clean | `0` blocked windows. |

## Business Impact

- Stage `04B` now provides accepted full-current Binance Futures USDT perpetual candle coverage for dataset refresh.
- This changes only data readiness for the RL training pipeline; no user-facing product surface, entitlement, trading mode, paper/testnet/live order flow, or mainnet capability is enabled.
- Stage `04C` may now build dataset refresh manifests from accepted source-window evidence instead of residual-gap diagnostics.

## Conditional Service-Call Coverage

| Surface | Coverage |
|---|---|
| Public Binance REST | Used only public `exchangeInfo` and `klines` endpoints; no private/account/order endpoint. |
| ClickHouse writes | Market-data writes went through existing raw/canonical ingestion path; dedup used existing canonical partition `OPTIMIZE ... FINAL` operator script. |
| WS/tail maintenance | No WS runtime was changed by this stage, but `RestInstrumentHistoryStartSource` now prevents scheduler/rest maintenance from retrying unavailable Binance Futures pre-history prefixes. |
| Auth/secrets | No credentials or private exchange keys were required for public Binance REST; ClickHouse access stayed local to Mac Studio. |
| Retry/unknown state | Backfill remains resumable by manifest; dedup was partition-scoped and post-checked; coverage is read-only and repeatable. |

## Logging And Redaction

- Runtime JSON/log artifacts stay under `/opt/roehub/state/rl_trading/`.
- Docs contain only sanitized paths, hashes, counts and timestamps.
- No raw provider payloads, credentials, tokens, cookies, passwords, signed requests or private account data are recorded.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API changed. |
| Port contract | `compatible-change` | `RestInstrumentHistoryStartSource` keeps the same interface but tightens Binance Futures history-start semantics to first returned 1m kline. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or table shape change. |
| Persisted rows / identity semantics | `compatible-change` | Existing raw/canonical data retained; duplicate canonical partitions were merged; first-kline manifest changes accepted source-window starts, not row identity. |
| Config schema/defaults | `compatible-change` | `configs/prod/whitelist.csv` contents changed; CSV schema unchanged. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Public Binance REST first-kline probe added for Binance Futures history-start resolution; existing timeout/retry config is reused. |
| External side-effect idempotency/unknown-state | `compatible-change` | Resume manifest plus canonical skip/read-back support interrupted/retried chunks. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds sanitized runtime JSON manifest/log evidence; no raw provider payloads or secrets in docs/git. |
| Browser-visible behavior | `none` | No browser/UI behavior changed. |
| Performance/hot path | `none` | Offline operator backfill only; live execution/inference hot paths unchanged. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` | passed; `15f6c9c6b3761a26d20335509fc7a41e03f302be7ab8b8c1afcd682e24f766e6` |
| `uv run pytest -q tests/unit/contexts/market_data/adapters/test_rest_instrument_history_start_source.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed: `18 passed` |
| `uv run ruff check src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_history_start_source.py scripts/rl_trading/stage04b_binance_futures_history_backfill.py tests/unit/contexts/market_data/adapters/test_rest_instrument_history_start_source.py tests/unit/scripts/rl_trading/test_stage04b_binance_futures_history_backfill.py` | local passed |
| `uv run pyright src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_history_start_source.py scripts/rl_trading/stage04b_binance_futures_history_backfill.py` | local passed: `0 errors` |
| `bash -n scripts/ops/optimize_canonical_partitions.sh` | local passed |
| Mac Studio live universe check | passed; `528` current symbols, `313` supplement |
| Mac Studio whitelist/ref/enrichment sync | passed; `528/528` enabled/tradable/enriched for `market_id=2` |
| Mac Studio full-current manifest generation | passed; `62,048` chunks, `44,731` reused completed, `17,317` pending |
| Mac Studio managed backfill start proof | passed; PID `20101` running, `workers=20`, `failed=0`, rows written increasing |
| Mac Studio full-current manifest completion readback | passed; `62,048/62,048` chunks completed, finished `2026-06-22T01:12:46Z` |
| Mac Studio duplicate repair | passed; existing optimized partition script cleared all seven duplicate residual symbols; post-check returned no `dup_rows > 0`. |
| Mac Studio first-kline focused residual probe | passed; `113` old blocked windows checked, `0` still blocked, `missing_minutes_total=0`, `duplicate_rows_total=0`. |
| Mac Studio first-kline full coverage report | passed; `coverage_status=accepted_coverage`, `0/886` windows blocked, `missing_minutes_total=0`, `duplicate_rows_total=0`. |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Blockers And Handoff

Stage `04B` is `accepted`.

Stage `04C` is opened because:

- full-current first-kline-aware coverage is clean (`0/886` blocked windows);
- `missing_minutes_total=0`;
- `duplicate_rows_total=0`;
- `volume_quote`, `trades_count`, and VWAP computability predicates are clean;
- unavailable Binance pre-history prefixes are excluded by confirmed first returned 1m kline, not synthesized.

Stage `04C` handoff:

1. Use `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_repair_manifest_first_kline.json` as the accepted source-window manifest.
2. Use `/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair/stage04b_full_current_coverage_report_first_kline.json` as the accepted coverage artifact.
3. Build dataset refresh manifests only for `binance:futures` USDT perpetual source windows proven above.
4. Keep later live tail maintenance separate from this Stage `04B` historical acceptance endpoint (`2026-06-21T15:09:00Z`).
