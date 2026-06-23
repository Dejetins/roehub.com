---
doc: rl-trading-agent-platform-v1-stage-05-roehub-dataset-builder-v1
stage: "05"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-23"
---

# Stage 05: Roehub Dataset Builder v1

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `05` стартовал после проверки ledger: Stage `04C` имеет статус `accepted`, `current_stage=05`, а accepted Stage `04C` manifest path/hash зафиксирован для единственного входа Stage `05`.

## Scope

Входит:

- зафиксировать prompt path/hash и planned file list до implementation edits;
- построить raw `binance:futures` feature slabs и sanitized manifests из accepted Stage `04C` refresh manifest;
- записать deterministic rebuild hash, manifest schema и feature stats;
- явно заблокировать `binance:spot`, `bybit:spot` и `bybit:futures` как `blocked_not_training_source_v1`;
- создать golden fixture, доказывающий parity offline/live-equivalent feature vector через общий Stage `02B` feature builder.

Не входит:

- accepted sessionized train/validation/test/backtest datasets;
- model training, checkpoints, registry writes or calibration packs;
- exchange/account/order/provider side effects;
- browser/UI/API changes;
- rediscovering universe/backfill scope or changing Stage `04C` manifest semantics.

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/raw_feature_dataset.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `scripts/rl_trading/stage05_roehub_dataset_builder.py`
- `tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/05-roehub-dataset-builder-v1.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Initial contract impact: `compatible-change`; Stage `05` adds an opt-in raw dataset artifact/manifest contract and runtime CLI under `/opt/roehub/state/rl_trading/`, without changing public API, exchange execution, persisted schema, config schema, browser-visible behavior, or market-data writer behavior.

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/raw_feature_dataset.py` | - | - | Pure Stage `05` raw feature slab, manifest, stats and golden parity helper. | `compatible-change` additive domain helper |
| `scripts/rl_trading/stage05_roehub_dataset_builder.py` | - | - | Opt-in CLI to consume accepted Stage `04C` manifest and write runtime raw feature artifacts. | `compatible-change` operator helper |
| `tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | - | - | Focused deterministic coverage for slab shape/order/hash, source gating and parity fixtures. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/05-roehub-dataset-builder-v1.md` | - | - | Stage `05` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export Stage `05` helper surface for tests and later stages. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `05` evidence, blockers and Stage `06` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding Stage `05` report. | `compatible-change` docs index only |

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/05-roehub-dataset-builder-v1.md` |
| Prompt sha256 | `7a8d14125b6950edd1150af53ea75a8ce2d952cfae86ea6cc1215b0702704e61` |
| Ledger state before implementation | Stage `04C` accepted; `current_stage=05`; Stage `05` pending |
| Required prerequisite | Stage `04C` accepted |
| Stage `04C` input manifest | `/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest/stage04c_dataset_refresh_manifest.json` |
| Stage `04C` input sha256 | `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344` |
| Delivery state | `local-only`; no branch, PR, main delivery or deploy. Mac Studio evidence is a bounded pre-main target-host sample run from the temporary scoped Stage `05` diff, not production deploy proof. |

## Observed State

| Area | Evidence summary |
|---|---|
| Stage prerequisite | Ledger records Stage `04C` as `accepted`; Stage `05` is the current stage. |
| Feature contract | Stage `02B` accepted feature hash `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` and channel order `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`. |
| Stage `04C` manifest | Accepted input path/hash are fixed; Stage `05` must consume this manifest and must not rediscover universe or backfill scope. |
| Market scope | `binance:futures` only is trainable in v1; `binance:spot`, `bybit:spot`, `bybit:futures` are blocked as `blocked_not_training_source_v1`. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response payload or web behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed; the existing market-data 5-column array DTO remains unchanged. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, table or database schema changed. |
| Config schema/defaults | `none` | No env/YAML/default changed; the CLI reads existing ClickHouse env the same way existing operator CLIs do. |
| Request hash / cache key / persistence identity | `none` | No existing runtime identity, cache key or request hash changed. |
| Raw feature artifact/manifest contract | `compatible-change` | New additive Stage `05` schema version `1` for raw feature slabs, deterministic rebuild hash, feature stats and parity fixture. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Adds an opt-in read-only ClickHouse CLI path; no provider/private exchange call and no write/retry behavior. Missing rows/features fail closed. |
| External side effects / unknown state | `none` | No exchange/account/order/provider side effect. Runtime artifact writes are local files under `/opt/roehub/state/rl_trading/`. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized report/ledger/runtime manifest hashes and counts; no secrets, raw provider payloads or model checkpoints. |
| Alerts/runbook semantics | `none` | No alert, Monit, launchd, scheduler or runbook behavior changed. |
| Browser-visible behavior | `none` | No UI changed; browser verification is N/A for this stage. |
| Performance hot path | `none` | Offline dataset materialization only; live inference/execution hot paths are unchanged. |
| Docs/runbooks | `compatible-change` | Stage report, ledger and docs index are updated only. |

## Mac Studio Proof Boundary

| Boundary label | Status | Evidence / rule |
|---|---|---|
| `target_host_readiness_pre_main` | collected | SSH reached `MacStudioDaniil`; remote git commands were run only in `/Users/daniildegtyarev/Projects/roehub.com`; accepted Stage `04C` manifest path/hash exists under `/opt/roehub/state/rl_trading/`. |
| `read_only_existing_runtime_smoke` | N/A | No existing production service or browser/runtime smoke was needed; Stage `05` does not change `/opt/roehub/app` behavior. |
| `target_host_non_production_sample_pre_main` | collected | Temporary scoped Stage `05` diff was applied to the Mac Studio git checkout, a bounded dataset sample was written under `/opt/roehub/state/rl_trading/datasets/`, and the diff was reversed. This is not production deploy proof. |
| `post_main_production_runtime_proof` | not collected | Requires the target revision on `main`, green GitHub Actions/CI, deploy or verified sync into `/opt/roehub/app` when service/runtime code is affected, and then runtime smoke. This stage report does not claim that proof. |

## Business Impact

- Stage `05` gives Stage `06` deterministic raw feature slabs and manifests from the accepted Stage `04C` source of truth, so session extraction no longer chooses universe/window scope itself.
- Operators get a bounded CLI that can build full selected windows or a small sample without opening exchange, account, order, browser or model-training surfaces.
- `binance:spot`, `bybit:spot` and `bybit:futures` remain explicitly blocked for v1 training as `blocked_not_training_source_v1`.

## Conditional Service-Call Coverage

| Surface | Coverage |
|---|---|
| Public/provider calls | `N/A`; Stage `05` does not call Binance, Bybit or private exchange endpoints. |
| ClickHouse | Read-only `SELECT ... FINAL` against `market_data.canonical_candles_1m` through existing ClickHouse settings; no writes. |
| Runtime artifact writes | Writes `.npy` slabs and sanitized JSON manifests under `/opt/roehub/state/rl_trading/datasets/`. |
| Auth/secrets | Existing host-local ClickHouse env is used on Mac Studio; no credential values are printed, copied or committed. |
| Timeout/retry/error behavior | No retry loop is added. Missing candles, null `volume_quote`, null `trades_count`, invalid OHLC, non-monotonic time or parity mismatch fail closed. |
| Idempotency/unknown state | Re-running with the same manifest/window/input rows rewrites the same output paths and records deterministic rebuild hashes. |
| Browser/UI | `N/A`; prompt disabled browser runtime verification and no browser-visible behavior changed. |

## Logging, Redaction, Alerts, Runbook

- Runtime manifests contain symbol names, timestamps, row counts, feature stats and hashes only; no raw provider payloads, tokens, cookies, credentials, signed requests or checkpoint contents.
- The Mac Studio sample command used the host-local env file through shell sourcing but did not echo environment values.
- No logs, metrics, alert routes, Monit/launchd settings, scheduler intervals or runbook actions changed.
- Full artifact retention/backup remains owned by later Stage `09B`; Stage `05` does not add cleanup policy.

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/05-roehub-dataset-builder-v1.md` | passed; `7a8d14125b6950edd1150af53ea75a8ce2d952cfae86ea6cc1215b0702704e61` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed; `5 passed` |
| `uv run ruff check src/trading/contexts/rl_trading/domain/raw_feature_dataset.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage05_roehub_dataset_builder.py tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed |
| `uv run pyright src/trading/contexts/rl_trading/domain/raw_feature_dataset.py scripts/rl_trading/stage05_roehub_dataset_builder.py tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading` | passed; `31 passed` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `377 passed, 3 warnings` |
| CLI fail-closed smoke: `uv run python scripts/rl_trading/stage05_roehub_dataset_builder.py --exchange bybit --market-type spot --symbol BTCUSDT` | passed; exited `2` with `reason=blocked_not_training_source_v1` |
| Mac Studio focused test from temporary scoped Stage `05` diff | passed; `5 passed` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after `uv run python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md` |

## Evidence

Mac Studio Stage `04C` input manifest readback:

| Field | Value |
|---|---|
| Host | `MacStudioDaniil` |
| Manifest path | `/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest/stage04c_dataset_refresh_manifest.json` |
| Manifest sha256 | `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344` |

Bounded target-host sample evidence:

| Field | Value |
|---|---|
| Evidence label | `target_host_non_production_sample_pre_main` |
| Code path | Temporary scoped Stage `05` diff applied to `/Users/daniildegtyarev/Projects/roehub.com`, then reversed; remote checkout returned clean. |
| Command scope | `BTCUSDT`, `post_hf_extension_current_trading`, first `5` minutes, `chunk_minutes=5` |
| Output root | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1_sample` |
| Manifest status | `accepted` |
| Manifest file sha256 | `163d4d09cb1cba86894878bd98cad4d87cecfa928603db61b6f024f07d0ae7a3` |
| Manifest deterministic rebuild hash | `e8ce5fe0699f70a682ae8232fee3c10c3e3fe41479c3591b51915a8544a8a25b` |
| Slab count / rows | `1` slab / `5` rows |
| Slab rebuild hash | `6c4d7a87a56b38cc9e0136c0b9504f81586dadae4e317ce279d777e4d8f76e4e` |
| `features.f32.npy` sha256 | `29513496948a106662642e08f3abf3b2ab3da75beceb9dde6d959746cf72e3a7` |
| `open_time_ms.i64.npy` sha256 | `9c9fe67e0586934c532a8d5e8932ed643ba47c485725e13142fd6a439a12af04` |
| `close_time_ms.i64.npy` sha256 | `20439e8e635dd207fad5d4da712051ff01b20b92661a6f6265155b8536279545` |
| Feature stats | `7` features recorded in sanitized slab manifest |
| Golden parity fixture | `3` samples, `max_abs_diff=0.0`, feature hash `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` |

Implementation notes:

- Raw slab materialization uses Stage `02B` channel order and dtype `float32`.
- VWAP parity bug found by the first Mac Studio sample was fixed: vectorized VWAP now matches the shared feature builder after `float32` materialization.
- The existing `CanonicalCandleBatch1m` / backtest artifact 5-column OHLCV contract was not changed; Stage `05` keeps the 7-channel RL feature slab contract inside `rl_trading`.

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `05` report, stage ledger update, file manifest, contract-impact table, Mac Studio proof-boundary wording, service-call/redaction/alert coverage, quality-gate evidence and Stage `06` handoff.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Added the missing cold-head receipt required by the artifact gate; no additional Blocker or High artifact findings remained after checking file manifest, proof-boundary labels, validation depth, service-call coverage, redaction and docs-index evidence.
Local follow-up check: completed
Residual risks: Delivery remains `local-only`; `post_main_production_runtime_proof` was not collected; full all-symbol raw slab materialization is supported by the CLI but not executed in this stage.

## Blockers And Handoff

Stage `05` is `accepted` with delivery state `local-only`.

No blocker remains for Stage `06` to consume the Stage `05` raw feature builder and manifests in this checkout. Stage `06` must still create accepted sessionized train/validation/test/backtest datasets and prove split/leak/overlap policy; it must not treat the Stage `05` sample artifact as a full accepted training dataset.

Handoff caveats:

- The Mac Studio sample is `target_host_non_production_sample_pre_main`, not `post_main_production_runtime_proof` and not a service deploy.
- Repository changes are not delivered to `origin/main` in this stage report. A separate publish step is required before another clean checkout can run Stage `06` without carrying this local diff.
- Full all-symbol raw slab materialization is supported by the CLI with `--all-symbols`, but Stage `05` acceptance evidence used a bounded sample to avoid launching a multi-hundred-symbol build inside this agent turn.
