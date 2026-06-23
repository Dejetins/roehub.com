---
doc: rl-trading-agent-platform-v1-stage-06-dataset-qa-session-extractor
stage: "06"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 06: Dataset QA And Session Extractor

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `06` started after checking the ledger: Stage `05` is `accepted`, delivered to `origin/main`, and fully materialized under `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1`.

Stage `06` is accepted after full Mac Studio materialization from the delivered Stage `05` raw manifest. The accepted full sessionized dataset has final hashes, counts, split artifacts, gap evidence and leakage evidence. Stage `07` may start; no training, checkpoint, registry, exchange, paper/testnet/live or mainnet behavior was started by Stage `06`.

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
| `src/trading/contexts/rl_trading/domain/sessionized_dataset.py` | - | - | Pure Stage `06` session extraction, vectorized high-volatility selection, split-window embargo, leakage, gap and manifest helper contract. | `compatible-change` additive domain helper |
| `scripts/rl_trading/stage06_dataset_qa_session_extractor.py` | - | - | Opt-in CLI to consume Stage `04C` manifest plus either Stage `05` raw slabs or explicit ClickHouse reads and write runtime sessionized artifacts. It records effective embargo-adjusted split windows. | `compatible-change` operator helper |
| `tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | - | - | Focused deterministic tests for split parsing, source gating, past-only high-volatility scoring, overlap/embargo/leakage reports, split embargo and manifest payloads. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md` | - | - | Stage `06` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export Stage `06` helper surface for tests and later stages. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `06` full materialization evidence and Stage `07` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding Stage `06` report. | `compatible-change` docs index only |

Outside expected paths: none.

Stage `06` delivery also included follow-up repair commits after the initial user push: package exports, vectorized selector performance, and split-embargo enforcement for full acceptance.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md` |
| Prompt sha256 | `2e9b44c51bc31d7d5e8601730ac6208a4f9f451e95949ef963878c05720176b0` |
| Ledger state before implementation | Stage `05` accepted; `current_stage=06`; Stage `06` pending |
| Required prerequisite | Stage `05` accepted |
| Stage `05` input manifest | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1/stage05_raw_feature_manifest.json` |
| Stage `05` input sha256 | `393461747be00aff457858473637d978791525cb629e60199c1e74c1148807f1` |
| Stage `06` implementation commits | `328a9a3a` initial Stage `06` files; `cfc8cde6` package exports/docs index; `7359d721` vectorized selector; `bcdc2473` split embargo |
| Delivery state | `delivered-to-main`; Mac Studio checkout clean on `bcdc247320cf6a7cffad643673ad5b91bd34d39c`; no `/opt/roehub/app` service deploy needed. |

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisite | Ledger records Stage `05` as `accepted`; Stage `05` full raw manifest exists and is accepted. |
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
| Embargo | Split-boundary embargo is 150 minutes. Effective right-hand split signal starts are shifted to `previous.signal_end + 150m`; violations block the leakage report. |
| Lifecycle/gap | Sessions are built only inside Stage `04C` safe source windows; minute gaps in source slabs block materialization. |
| Keys | Session key includes `exchange_name`, `market_type`, `symbol`, `instrument_key`, `signal_ts_open`, `split`, `feature_contract_hash`. |

## Runtime Evidence

Accepted full Mac Studio materialization:

| Field | Value |
|---|---|
| Evidence label | `post_main_dataset_materialization_proof` |
| Host | `MacStudioDaniil` |
| Remote checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Remote checkout commit | `bcdc247320cf6a7cffad643673ad5b91bd34d39c` |
| Remote checkout state | `## main...origin/main` |
| Runtime artifact root | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1` |
| Manifest path | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` |
| Manifest status | `accepted` |
| Manifest file sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Manifest deterministic rebuild hash | `a28084ac5dfe6533446ed3da45bfca955e36f0451ae6d00ff4cac55ea9582b56` |
| Leakage report path | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_leakage_report.json` |
| Leakage report sha256 | `cbe1424bab47b4907cdee4b4585d107a449650dcde9f8b39b06d4f867e2e370a` |
| Input Stage `04C` manifest sha256 | `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344` |
| Input Stage `05` raw manifest | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1/stage05_raw_feature_manifest.json` |
| Build scope | `full_selected_windows`, `all_symbols=true`, `from_clickhouse=false` |
| Dataset versions | `hf_period_rebuild_current_trading`, `post_hf_extension_current_trading` |
| Selected symbols | `528` |
| Split artifact count | `1,656` |
| Total sessions | `83,772` |
| Artifact directory size / files | `426M` / `6,626` files |
| Runtime | `112.25s` wall, max RSS `2,281,766,912` bytes |
| Gap report summary | `1,656/1,656` split artifacts accepted; `gap_count_total=0`, `missing_minutes_total=0` |
| Leakage report summary | `status=accepted`, `cross_split_overlap_violations=0`, `embargo_violations=0`, `lookahead_violations=0`, `lifecycle_violations=0`, `within_split_overlap_pairs=60,145` |
| Rejected windows | `304`, all `lifecycle_no_signal_overlap_for_split` |
| Safety flags | `contains_model_checkpoint=false`, `contains_raw_provider_payloads=false`, `contains_secrets=false`, `exchange_side_effects=false`, `market_data_writes=false`, `score_uses_post_signal_rows=false` |

Accepted full split/session distribution:

| Dataset version / split | Split artifacts | Sessions |
|---|---:|---:|
| `hf_period_rebuild_current_trading:train` | `220` | `13,381` |
| `hf_period_rebuild_current_trading:validation` | `250` | `10,249` |
| `hf_period_rebuild_current_trading:test` | `300` | `12,346` |
| `hf_period_rebuild_current_trading:backtest` | `358` | `14,731` |
| `post_hf_extension_current_trading:post_hf_extension` | `528` | `33,065` |

Effective split embargo windows:

| Dataset version | Split | Effective signal window |
|---|---|---|
| `hf_period_rebuild_current_trading` | `train` | `[2020-01-14T00:00:00Z, 2024-08-31T00:00:00Z)` |
| `hf_period_rebuild_current_trading` | `validation` | `[2024-09-01T00:00:00Z, 2024-12-01T00:00:00Z)` |
| `hf_period_rebuild_current_trading` | `test` | `[2024-12-01T02:30:00Z, 2025-03-01T00:00:00Z)` |
| `hf_period_rebuild_current_trading` | `backtest` | `[2025-03-01T02:30:00Z, 2025-06-01T00:00:00Z)` |
| `post_hf_extension_current_trading` | `post_hf_extension` | `[2025-06-01T00:00:00Z, 2026-06-21T14:10:00Z)` |

Full-run repair note: the first full raw-manifest run proved full-scale materialization performance but returned `status=blocked` because Stage `04C` `validation->test` and `test->backtest` signal windows are contiguous. Stage `06` then added effective split embargo enforcement (`right.signal_start >= previous.signal_end + 150m`) and the accepted rerun cleared embargo and cross-split overlap violations.

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

- Stage `06` now provides the accepted deterministic sessionized dataset contract that Stage `07` training must consume.
- The implementation prevents accidental training on spot/Bybit branches and prevents session overlap or look-ahead leakage from being hidden in later model metrics.
- Business/user-facing trading behavior does not change: no strategy launch, entitlement, order, paper/testnet/live or mainnet capability is enabled.
- Stage `07` can start from the accepted Stage `06` manifest; later paper/testnet/live paths remain gated by their own stages and classic producer prerequisites.

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
| `target_host_non_production_sample_pre_main` | collected historically | Temporary scoped Stage `05`+`06` diff was applied to the Mac Studio git checkout, focused tests and bounded dataset sample ran, and the diff was reversed. This is retained as historical debugging evidence only. |
| `post_main_dataset_materialization_proof` | collected | Mac Studio git checkout was clean at `bcdc247320cf6a7cffad643673ad5b91bd34d39c` on `main...origin/main`; full Stage `06` raw-manifest materialization wrote accepted artifacts under `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1`. |
| `post_main_production_runtime_proof` | N/A | Stage `06` changes no `/opt/roehub/app` service/runtime/browser surface. The relevant proof is dataset materialization from the clean Mac Studio git checkout, not service deploy/reload. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md` | passed; `2e9b44c51bc31d7d5e8601730ac6208a4f9f451e95949ef963878c05720176b0` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed after split-embargo repair; `7 passed` |
| `uv run ruff check src/trading/contexts/rl_trading/domain/sessionized_dataset.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage06_dataset_qa_session_extractor.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed |
| `uv run pyright src/trading/contexts/rl_trading/domain/sessionized_dataset.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage06_dataset_qa_session_extractor.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed; `0 errors` |
| CLI fail-closed smoke: `uv run python scripts/rl_trading/stage06_dataset_qa_session_extractor.py --exchange bybit --market-type spot --symbol BTCUSDT` | passed; exited `2` with `reason=blocked_not_training_source_v1` |
| Mac Studio focused tests after `bcdc2473` sync | passed; `7 passed` |
| Mac Studio focused ruff after `bcdc2473` sync | passed |
| Mac Studio focused pyright after `bcdc2473` sync | passed; `0 errors` |
| Mac Studio bounded ClickHouse-backed Stage `06` sample | passed for bounded sample; manifest status `accepted`, `2` sessions |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed after split-embargo repair; `385 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed |
| GitHub Actions run `28043055395` for `7359d721` | passed; status `completed`, conclusion `success` |
| GitHub Actions run `28043371282` for `bcdc2473` | passed; status `completed`, conclusion `success` |
| GitHub deploy/image workflows for `bcdc2473` | passed; Deploy Backend `28043668302`, Publish App Image `28043667093`, Deploy Web `28043665654`/`28043681387` |
| Mac Studio first full Stage `06` raw-manifest run | blocked as designed; `83,772` sessions written but leakage report had `2` embargo and `3` cross-split overlap violations before effective split embargo repair |
| Mac Studio accepted full Stage `06` raw-manifest run at `bcdc2473` | passed; `status=accepted`, `split_artifact_count=1,656`, `total_sessions=83,772`, manifest sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `06` report, stage ledger update, file manifest, contract-impact table, Mac Studio proof-boundary wording, service-call/redaction/alert coverage, quality-gate evidence, full-materialization manifest evidence and Stage `07` handoff.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Replaced blocked/sample-only wording with accepted full-materialization evidence; recorded package-export, vectorized-selector and split-embargo repairs; preserved no-training/no-exchange/no-service-deploy boundaries; recorded final hashes, counts, split distribution, gap/leakage reports and Stage `07` handoff.
Local follow-up check: completed
Residual risks: Stage `09B` still owns backup/restore policy for large artifacts; Stage `07` must prove training/resource behavior before any model, checkpoint, registry, paper/testnet/live or activation claim.

## Blockers And Handoff

Stage `06` is `accepted`.

Implemented and verified:

- additive sessionized dataset domain contract;
- opt-in Stage `06` CLI;
- deterministic tests for split parsing, source gating, past-only high-volatility scoring, session shape, overlap/embargo/leakage and manifest payloads;
- vectorized full-scale high-volatility selection;
- effective 150-minute split embargo;
- bounded Mac Studio `BTCUSDT` sample under `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1_sample`;
- full Mac Studio accepted sessionized dataset under `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1`.

No Stage `06` acceptance blocker remains.

Stage `07` may start from:

`/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json`

Stage `07` must still not treat Stage `06` acceptance as a model-quality, profitability, backtest, registry, paper/testnet/live, exchange or activation approval.

Next action:

- start Stage `07` D3QN/PER training runner only from the accepted Stage `06` manifest and keep all training/checkpoint/registry/resource claims inside Stage `07`.
