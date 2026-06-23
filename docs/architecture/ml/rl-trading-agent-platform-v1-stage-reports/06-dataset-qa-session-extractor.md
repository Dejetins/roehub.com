---
doc: rl-trading-agent-platform-v1-stage-06-dataset-qa-session-extractor
stage: "06"
status: blocked
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 06: Dataset QA And Session Extractor

Status: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `06` started after checking the ledger: Stage `05` is `accepted`, `current_stage=06`, and the Stage `05` report explicitly says the local checkout may continue Stage `06` even though Stage `05` delivery state is still `local-only`.

The Stage `06` code path is implemented and a bounded Mac Studio ClickHouse-backed sample is accepted, but the stage is **not accepted** because the required full sessionized train/validation/test/backtest datasets were not materialized. Stage `07` remains blocked.

## Scope

In scope:

- record prompt path/hash and planned file list before implementation edits;
- implement the additive Stage `06` `binance:futures` session extraction and QA contract;
- define high-volatility selection, stride/overlap, `pre_signal_len=90`, `post_signal_len=60`, split policy, embargo and lifecycle checks;
- emit machine-readable gap, lifecycle, overlap, embargo and leakage evidence for sessionized artifacts;
- keep runtime datasets under `/opt/roehub/state/rl_trading/`.

Out of scope:

- model training, checkpoint creation, registry writes or calibration packs;
- Binance spot, Bybit spot or Bybit futures training/evaluation datasets;
- exchange/account/order/provider side effects;
- browser/UI/API changes;
- mainnet, paper or testnet execution.

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/sessionized_dataset.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `scripts/rl_trading/stage06_dataset_qa_session_extractor.py`
- `tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Final file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/sessionized_dataset.py` | - | - | Pure Stage `06` session extraction, split-window, leakage, gap and manifest helper contract. | `compatible-change` additive domain helper |
| `scripts/rl_trading/stage06_dataset_qa_session_extractor.py` | - | - | Opt-in CLI to consume Stage `04C` manifest plus either Stage `05` raw slabs or explicit ClickHouse reads and write runtime sessionized artifacts. | `compatible-change` operator helper |
| `tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | - | - | Focused deterministic tests for split parsing, source gating, past-only high-volatility scoring, overlap/embargo/leakage reports and manifest payloads. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md` | - | - | Stage `06` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export Stage `06` helper surface for tests and later stages. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `06` evidence, blocker and Stage `07` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding Stage `06` report. | `compatible-change` docs index only |

Outside expected paths: none.

Pre-existing local-only Stage `05` files remain in the working tree and are required context for this checkout, but they are not counted as new Stage `06` touches.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md` |
| Prompt sha256 | `2e9b44c51bc31d7d5e8601730ac6208a4f9f451e95949ef963878c05720176b0` |
| Ledger state before implementation | Stage `05` accepted; `current_stage=06`; Stage `06` pending |
| Required prerequisite | Stage `05` accepted |
| Stage `05` caveat | Delivery state is `local-only`; Stage `05` sample artifact is not a full accepted training dataset. |
| Delivery state | `local-only`; no branch, PR, main delivery or deploy. |

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisite | Ledger records Stage `05` as `accepted`; Stage `06` was current and pending before this run. |
| Feature contract | Stage `02B` accepted feature hash `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` and channel order `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`. |
| Stage `04C` manifest | Accepted input path is `/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest/stage04c_dataset_refresh_manifest.json` with sha256 `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344`. |
| Market scope | `binance:futures` only is trainable in v1; `binance:spot`, `bybit:spot`, `bybit:futures` stay blocked as `blocked_not_training_source_v1`. |
| Compact context note | `.codex/agents/.context/promt_manager_state.yaml` is topical but stale on delivery policy; current `.codex/AGENTS.md`, ledger and Stage `06` prompt override it. |

## Методология

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`, dataset QA / leakage certification for an ML training artifact. |
| Тип задачи | Build high-volatility session samples and certify split, gap, lifecycle and leakage rules before training. |
| Выбранная методология | Vectorized feature-window QA with deterministic time split, per-symbol lifecycle bounds, fixed embargo, machine-readable rejected-window reasons and artifact hashes. |
| Единица анализа | `dataset_version + split + binance:futures symbol + signal_ts_open`. |
| Основные метрики | Session count, rejected count by reason, gap count, overlap pairs, cross-split overlap violations, embargo violations, lifecycle violations, feature-window hashes. |
| Риски интерпретации | The exact external article extractor rule is not claimed; Stage `06` records a deterministic high-volatility proxy and does not claim model profitability. |

## Session Extraction Policy

| Area | V1 policy |
|---|---|
| Market branch | `binance:futures` only; other branches fail closed with `blocked_not_training_source_v1`. |
| Source of universe/splits | Accepted Stage `04C` manifest only; no rediscovery of universe or backfill scope inside Stage `06`. |
| Window shape | `full_seq_len=150`, `pre_signal_len=90`, `post_signal_len=60`, feature shape `(150, 7)`, dtype `float32`. |
| Signal timestamp semantics | `signal_ts_open` is the first post-signal candle open; the score window ends at `signal_ts_open`, so scoring does not use post-signal rows. |
| High-volatility proxy | `pre_signal_realized_volatility_plus_range_v1`: realized close volatility plus average high/low range ratio over the 90 pre-signal rows. |
| Default selection | 30-minute stride, top 1% by score per symbol/split, capped at 64 sessions per symbol/split. CLI exposes explicit overrides for bounded evidence runs. |
| Overlap | Allowed inside a split; reported as `within_split_overlap_pairs`. Cross-split overlap blocks the leakage report. |
| Embargo | Split-boundary embargo is at least 150 minutes. Violations block the leakage report. |
| Lifecycle/gap | Sessions are built only inside Stage `04C` safe source windows; minute gaps in source slabs block materialization. |
| Keys | Session key includes `exchange_name`, `market_type`, `symbol`, `instrument_key`, `signal_ts_open`, `split`, `feature_contract_hash`. |

## Runtime Evidence

Accepted bounded Mac Studio sample:

| Field | Value |
|---|---|
| Evidence label | `target_host_non_production_sample_pre_main` |
| Host | `MacStudioDaniil` |
| Remote checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Runtime artifact root | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1_sample` |
| Command scope | `BTCUSDT`, `post_hf_extension_current_trading`, `post_hf_extension`, `max_minutes_per_source_window=240`, ClickHouse-backed, `max_sessions_per_symbol_split=2` |
| Manifest path | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1_sample/stage06_sessionized_manifest.json` |
| Manifest status | `accepted` |
| Manifest file sha256 | `6561c1882f54bf4054dc36d94d69b7b1f282c2ce18299d183489f91514c59d92` |
| Manifest deterministic rebuild hash | `dcf1bdfba43b1e9b40275a4dd860286a1f481899ae5a82c4896f4546152b2ad9` |
| Split artifacts / sessions | `1` split artifact / `2` sessions |
| Split artifact rebuild hash | `049362b15a1cbdb0361d2de29394ef34349093ce37280e38885cab128f4ac3f3` |
| Session shape | `[150, 7]` |
| Source rows | `240` rows from `2025-05-31T22:30:00Z` through `2025-06-01T02:29:00Z` |
| Gap report | `gap_count=0`, `missing_minutes=0`, `status=accepted` |
| Leakage report | `status=accepted`, `cross_split_overlap_violations=0`, `embargo_violations=0`, `lookahead_violations=0`, `lifecycle_violations=0`, `within_split_overlap_pairs=1` |
| Session signals | `2025-06-01T00:00:00Z`, `2025-06-01T00:30:00Z` |
| Feature array sha256 | `b29e246b3be02f5e362758b7333a2512ade711cd8e9ac5a71f5c476e5109e40a` |
| Signal-time array sha256 | `058f8b0974f986f82217d08eb8a347997b72ce7349be152c21bafb4aa24c556d` |
| Metadata sha256 | `37d4baf78b13c48326d2b8aebfeba13a652037c53ee435718a65749726a10157` |

Implementation note: the first Mac Studio sample exposed a timezone boundary bug in the new Stage `06` ClickHouse path. The query was corrected to filter by `toUnixTimestamp64Milli(ts_open)` integer UTC bounds; the accepted sample then read the expected `2025-05-31T22:30:00Z` source start.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response payload or web behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, table or database schema changed. |
| Config schema/defaults | `none` | No env/YAML/default changed; the CLI reads existing ClickHouse env only when explicitly asked to use ClickHouse. |
| Request hash / cache key / persistence identity | `none` | No existing runtime identity, cache key or request hash changed. |
| Sessionized artifact/QA contract | `compatible-change` | New additive Stage `06` schema version `1` for sessionized split artifacts, leakage/gap reports, deterministic rebuild hashes and safe runtime manifest summaries. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Adds an opt-in read-only ClickHouse CLI path; no provider/private exchange call and no retry loop. Missing rows/features fail closed. |
| External side effects / unknown state | `none` | No exchange/account/order/provider side effect. Runtime artifact writes are local files under `/opt/roehub/state/rl_trading/`. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized report/ledger/runtime manifest hashes and counts; no secrets, raw provider payloads or model checkpoints. |
| Alerts/runbook semantics | `none` | No alert, Monit, launchd, scheduler or runbook behavior changed. |
| Browser-visible behavior | `none` | No UI changed; browser verification is N/A for this stage. |
| Performance hot path | `none` | Offline dataset materialization only; live inference/execution hot paths are unchanged. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index are updated only. |

## Business Impact

- Stage `06` now defines the deterministic sessionized dataset contract that Stage `07` training must consume.
- The implementation prevents accidental training on spot/Bybit branches and prevents session overlap or look-ahead leakage from being hidden in later model metrics.
- Business/user-facing trading behavior does not change: no strategy launch, entitlement, order, paper/testnet/live or mainnet capability is enabled.
- Full ML progression is still blocked until the complete accepted train/validation/test/backtest datasets are materialized and recorded.

## Conditional Service-Call Coverage

| Surface | Coverage |
|---|---|
| Public/provider calls | `N/A`; Stage `06` does not call Binance, Bybit or private exchange endpoints. |
| ClickHouse | Covered for the opt-in `--from-clickhouse` path: read-only `SELECT ... FINAL` against `market_data.canonical_candles_1m`; no writes. |
| Runtime artifact writes | Writes `.npy` session arrays and sanitized JSON manifests under `/opt/roehub/state/rl_trading/datasets/`. |
| Auth/secrets | Existing host-local ClickHouse env is used on Mac Studio; no credential values are printed, copied or committed. |
| Timeout/retry/error behavior | No retry loop is added. Missing rows, null fields, non-monotonic minutes or leakage violations fail closed. |
| Idempotency/unknown state | Re-running with the same manifest/window/input rows rewrites the same output paths and records deterministic rebuild hashes. |
| Browser/UI | `N/A`; prompt disabled browser verification and no browser-visible behavior changed. |

## Logging, Redaction, Alerts, Runbook

- Runtime manifests contain symbol names, timestamps, row counts, selected session metadata and hashes only.
- No raw provider payloads, tokens, cookies, credentials, signed requests, private account data or model checkpoint contents are written into docs or committed files.
- No logs, metrics, alert routes, Monit/launchd settings, scheduler intervals or runbook actions changed.
- Full artifact retention/backup remains owned by later Stage `09B`; Stage `06` does not add cleanup policy.

## Mac Studio Proof Boundary

| Boundary label | Status | Evidence / rule |
|---|---|---|
| `target_host_readiness_pre_main` | collected | SSH reached `MacStudioDaniil`; remote git commands were run only in `/Users/daniildegtyarev/Projects/roehub.com`; accepted Stage `04C` manifest path/hash exists under `/opt/roehub/state/rl_trading/`. |
| `read_only_existing_runtime_smoke` | N/A | No existing production service or browser/runtime smoke was needed; Stage `06` does not change `/opt/roehub/app` behavior. |
| `target_host_non_production_sample_pre_main` | collected | Temporary scoped Stage `05`+`06` diff was applied to the Mac Studio git checkout, focused tests and bounded dataset sample ran, and the diff was reversed. Remote checkout returned clean. |
| `post_main_production_runtime_proof` | not collected | Requires the target revision on `main`, green GitHub Actions/CI, deploy or verified sync into `/opt/roehub/app` when service/runtime code is affected, and then runtime smoke. This stage report does not claim that proof. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md` | passed; `2e9b44c51bc31d7d5e8601730ac6208a4f9f451e95949ef963878c05720176b0` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed; `6 passed` |
| `uv run ruff check src/trading/contexts/rl_trading/domain/sessionized_dataset.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage06_dataset_qa_session_extractor.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed |
| `uv run pyright src/trading/contexts/rl_trading/domain/sessionized_dataset.py scripts/rl_trading/stage06_dataset_qa_session_extractor.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed; `0 errors` |
| CLI fail-closed smoke: `uv run python scripts/rl_trading/stage06_dataset_qa_session_extractor.py --exchange bybit --market-type spot --symbol BTCUSDT` | passed; exited `2` with `reason=blocked_not_training_source_v1` |
| Mac Studio focused tests from temporary scoped Stage `05`+`06` diff | passed; `11 passed` |
| Mac Studio focused ruff from temporary scoped Stage `05`+`06` diff | passed |
| Mac Studio focused pyright from temporary scoped Stage `05`+`06` diff | passed; `0 errors` |
| Mac Studio bounded ClickHouse-backed Stage `06` sample | passed for bounded sample; manifest status `accepted`, `2` sessions |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `383 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after `uv run python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md` |

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `06` report, stage ledger update, file manifest, contract-impact table, Mac Studio proof-boundary wording, service-call/redaction/alert coverage, quality-gate evidence and Stage `07` handoff.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Added the explicit blocked acceptance state, full-dataset blocker, business impact, conditional service-call coverage, logging/redaction, alert/runbook N/A coverage, exact runtime sample path/hash/counts, proof-boundary labels and docs-index gate placeholder.
Local follow-up check: completed
Residual risks: Stage `06` remains blocked for acceptance until the complete train/validation/test/backtest sessionized datasets are materialized; repository changes remain `local-only`; `post_main_production_runtime_proof` was not collected.

## Blockers And Handoff

Stage `06` is `blocked`.

Implemented and verified locally:

- additive sessionized dataset domain contract;
- opt-in Stage `06` CLI;
- deterministic tests for split parsing, source gating, past-only high-volatility scoring, session shape, overlap/embargo/leakage and manifest payloads;
- bounded Mac Studio `BTCUSDT` sample under `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1_sample`.

Acceptance blocker:

- full accepted `binance:futures` sessionized train/validation/test/backtest datasets were not materialized and therefore do not yet have final hashes, counts and split/leakage/gap reports.

Stage `07` is **not allowed**.

Next action:

- run the Stage `06` CLI against the complete accepted Stage `04C`/Stage `05` source universe, preferably after the Stage `05` local-only diff is published or a full Stage `05` raw manifest is available in the target checkout;
- record final full-dataset manifest hashes/counts and leakage/gap/lifecycle reports;
- only then update this report and ledger from `blocked` to `accepted`.
