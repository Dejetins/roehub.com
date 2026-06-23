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

Stage `05` завершен как delivered clean-main state, а не sample-only/local-only: код доставлен в `origin/main`, CI зеленый, Mac Studio checkout чистый на commit `1d08bf4a53c94757f3ef4be4f85e54e0403786b3`, и полный raw feature dataset manifest записан под `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1`.

## Scope

Входит:

- зафиксировать prompt path/hash и planned file list до implementation edits;
- построить raw `binance:futures` feature slabs и sanitized manifests из accepted Stage `04C` refresh manifest;
- записать deterministic rebuild hash, manifest schema и feature stats;
- явно заблокировать `binance:spot`, `bybit:spot` и `bybit:futures` как `blocked_not_training_source_v1`;
- создать golden fixture, доказывающий parity offline/live-equivalent feature vector через общий Stage `02B` feature builder;
- доставить Stage `05` code/docs в `origin/main` и материализовать полный all-symbol raw dataset artifact на Mac Studio.

Не входит:

- accepted sessionized train/validation/test/backtest datasets;
- model training, checkpoints, registry writes or calibration packs;
- exchange/account/order/provider side effects;
- browser/UI/API changes;
- rediscovering universe/backfill scope or changing Stage `04C` manifest semantics;
- accepting Stage `06` or starting Stage `07`.

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
| `scripts/rl_trading/stage05_roehub_dataset_builder.py` | - | - | Opt-in CLI to consume accepted Stage `04C` manifest and write runtime raw feature artifacts; later made resume-safe for long materialization. | `compatible-change` operator helper |
| `tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | - | - | Focused deterministic coverage for slab shape/order/hash, source gating, parity fixtures and resume validation. | `compatible-change` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/05-roehub-dataset-builder-v1.md` | - | - | Stage `05` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export Stage `05` helper surface while keeping Stage `06` local-only exports out of the clean Stage `05` publish. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `05` full materialization evidence, blockers and Stage `06` handoff. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding Stage `05` report; later repaired to exclude untracked Stage `06` report from Stage `05` publish. | `compatible-change` docs index only |

Out of scope and intentionally not published as Stage `05`: local Stage `06` files under `06-dataset-qa-session-extractor.md`, `stage06_dataset_qa_session_extractor.py`, `sessionized_dataset.py`, and `test_sessionized_dataset.py`.

## Prompt And Delivery Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/05-roehub-dataset-builder-v1.md` |
| Prompt sha256 | `7a8d14125b6950edd1150af53ea75a8ce2d952cfae86ea6cc1215b0702704e61` |
| Ledger state before implementation | Stage `04C` accepted; `current_stage=05`; Stage `05` pending |
| Required prerequisite | Stage `04C` accepted |
| Stage `04C` input manifest | `/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest/stage04c_dataset_refresh_manifest.json` |
| Stage `04C` input sha256 | `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344` |
| Implementation commit | `6b46c3bae6b7ee0843d5a2a7d77891f5713cde40` (`Deliver RL Stage 05 raw feature builder`) |
| Docs index repair commit | `48c9e6b1` (`Fix RL Stage 05 docs index scope`) |
| Resume/full-build commit | `1d08bf4a53c94757f3ef4be4f85e54e0403786b3` (`Make RL Stage 05 builder resumable`) |
| Delivery state | `delivered-to-main`; `origin/main` at `1d08bf4a53c94757f3ef4be4f85e54e0403786b3` before full Mac Studio materialization |

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
| `target_host_non_production_sample_pre_main` | collected historically | Temporary scoped Stage `05` diff produced the earlier `BTCUSDT` 5-row sample. This evidence is retained as historical debugging/proof only and is not the final acceptance boundary. |
| `post_main_dataset_materialization_proof` | collected | Mac Studio git checkout was clean at `1d08bf4a53c94757f3ef4be4f85e54e0403786b3` on `main...origin/main`; full Stage `05` raw feature dataset manifest was written under `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1`. |
| `post_main_production_runtime_proof` | N/A | Stage `05` changed no `/opt/roehub/app` service/runtime/browser surface. The relevant post-main proof for this stage is dataset materialization from the clean Mac Studio git checkout, not service deploy/reload. |

## Runtime Artifact Evidence

| Field | Value |
|---|---|
| Host | `MacStudioDaniil` |
| Mac Studio checkout commit | `1d08bf4a53c94757f3ef4be4f85e54e0403786b3` |
| Mac Studio checkout state | `## main...origin/main` with no local changes reported |
| Output root | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1` |
| Manifest path | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1/stage05_raw_feature_manifest.json` |
| Manifest file sha256 | `393461747be00aff457858473637d978791525cb629e60199c1e74c1148807f1` |
| Manifest deterministic rebuild hash | `4c2c99524c8c3d4ff60ae10e03be926ea1c4734d0bdda1b8c986e67d55d4890b` |
| Manifest status | `accepted` |
| Build scope | `full_selected_windows`, `all_symbols=true` |
| Selected symbol count | `528` |
| Slab count | `886` |
| Total rows | `618,533,473` |
| `hf_period_rebuild_current_trading` | `358` slabs / `358` symbols / `361,453,025` rows |
| `post_hf_extension_current_trading` | `528` slabs / `528` symbols / `257,080,448` rows |
| Stage `04C` dependency sha256 | `9e633516cbc4aa4a711802b586e942a0a20638a4789ca6d19792fe7c78040344` |
| Feature contract hash | `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` |
| Golden parity fixture | `3` samples, `max_abs_diff=0.0`, symbol `1000000MOGUSDT`, dataset version `hf_period_rebuild_current_trading` |
| Artifact safety flags | `contains_sessionized_training_dataset=false`, `contains_raw_provider_payloads=false`, `contains_secrets=false`, `exchange_side_effects=false`, `market_data_writes=false` |
| Final artifact directory size | `25,966` MiB at completion probe |
| ClickHouse health after materialization | `/ping` returned `Ok.` |

The first full run was interrupted after `283` slab manifests by a transient ClickHouse HTTP `127.0.0.1:8123` connection refusal. ClickHouse returned healthy on `/ping` immediately after the interruption. Stage `05` therefore added `--resume-existing-slabs`, with fail-closed validation of existing slab manifests/files before reuse, then completed the full materialization from the same accepted Stage `04C` manifest.

## Business Impact

- Stage `06` now has a full Roehub-native raw feature source of truth for all accepted Stage `04C` `binance:futures` windows, not a local sample.
- Session extraction can consume deterministic slab manifests, row counts and rebuild hashes instead of rediscovering universe, lifecycle windows or feature semantics.
- Operators can rerun or resume Stage `05` without opening exchange/provider/account/order surfaces, and stale partial slabs fail closed instead of silently entering the training pipeline.

## Conditional Service-Call Coverage

| Surface | Coverage |
|---|---|
| Public/provider calls | `N/A`; Stage `05` does not call Binance, Bybit or private exchange endpoints. |
| ClickHouse | Read-only `SELECT ... FINAL` against `market_data.canonical_candles_1m` through existing ClickHouse settings; no writes. |
| Runtime artifact writes | Writes `.npy` slabs and sanitized JSON manifests under `/opt/roehub/state/rl_trading/datasets/`. |
| Auth/secrets | Existing host-local ClickHouse env is used on Mac Studio; no credential values are printed, copied or committed. |
| Timeout/retry/error behavior | The builder itself does not blindly retry ClickHouse reads. Resume mode reuses only previously accepted slab manifests whose schema, source window, row count, files and feature contract match the current request; stale or corrupt slabs fail closed. |
| Idempotency/unknown state | Re-running with the same manifest/window/input rows rewrites or validates the same output paths and records deterministic rebuild hashes. |
| Browser/UI | `N/A`; prompt disabled browser runtime verification and no browser-visible behavior changed. |

## Logging, Redaction, Alerts, Runbook

- Runtime manifests contain symbol names, timestamps, row counts, feature stats and hashes only; no raw provider payloads, tokens, cookies, credentials, signed requests or checkpoint contents.
- The Mac Studio commands used the host-local env file through shell sourcing but did not echo environment values.
- No logs, metrics, alert routes, Monit/launchd settings, scheduler intervals or runbook actions changed.
- Full artifact retention/backup remains owned by later Stage `09B`; Stage `05` does not add cleanup policy.

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/05-roehub-dataset-builder-v1.md` | passed; `7a8d14125b6950edd1150af53ea75a8ce2d952cfae86ea6cc1215b0702704e61` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed after resume patch; `6 passed` |
| `uv run ruff check scripts/rl_trading/stage05_roehub_dataset_builder.py tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed |
| `uv run pyright scripts/rl_trading/stage05_roehub_dataset_builder.py tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps --ignore=tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | passed; `378 passed, 3 warnings`; ignored only the untracked local Stage `06` test to mirror a clean Stage `05` checkout |
| Mac Studio `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_raw_feature_dataset.py` at `1d08bf4a` | passed; `6 passed` |
| CLI fail-closed smoke: `uv run python scripts/rl_trading/stage05_roehub_dataset_builder.py --exchange bybit --market-type spot --symbol BTCUSDT` | passed; exited `2` with `reason=blocked_not_training_source_v1` |
| GitHub Actions run `28034733779` for `1d08bf4a` | passed; status `completed`, conclusion `success` |
| Mac Studio full materialization command with `--all-symbols --resume-existing-slabs` | passed; `status=accepted`, `slab_count=886`, `total_rows=618533473` |

## Historical Sample Evidence

The earlier bounded Mac Studio sample remains useful as a debugging trace, but it is no longer the acceptance boundary.

| Field | Value |
|---|---|
| Evidence label | `target_host_non_production_sample_pre_main` |
| Command scope | `BTCUSDT`, `post_hf_extension_current_trading`, first `5` minutes, `chunk_minutes=5` |
| Output root | `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1_sample` |
| Manifest file sha256 | `163d4d09cb1cba86894878bd98cad4d87cecfa928603db61b6f024f07d0ae7a3` |
| Manifest deterministic rebuild hash | `e8ce5fe0699f70a682ae8232fee3c10c3e3fe41479c3591b51915a8544a8a25b` |
| Slab count / rows | `1` slab / `5` rows |
| Golden parity fixture | `3` samples, `max_abs_diff=0.0`, feature hash `d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9` |

## Cold Self-Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `05` report, stage ledger update, file manifest, contract-impact table, Mac Studio proof-boundary wording, service-call/redaction/alert coverage, quality-gate evidence, full-materialization manifest evidence and Stage `06` handoff.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: Replaced stale `local-only` / sample-only wording with delivered clean-main full-materialization evidence; removed the Stage `06` blocker clause that said the Stage `05` full raw manifest was unavailable in a delivered clean checkout; recorded the ClickHouse interruption and fail-closed resume mitigation; preserved Stage `06`/`07` boundaries.
Local follow-up check: completed
Residual risks: Stage `06` full sessionized train/validation/test/backtest datasets are still not materialized or accepted; no `/opt/roehub/app` service deploy was needed or performed because Stage `05` has no service/browser/runtime surface; Stage `09B` still owns backup/restore policy for large artifacts.

## Blockers And Handoff

Stage `05` is `accepted`, delivered to `origin/main`, and fully materialized under `/opt/roehub/state/rl_trading/datasets/stage05_raw_feature_dataset_v1`.

No blocker remains for Stage `06` to consume the Stage `05` raw feature builder and full raw manifest. Stage `06` itself remains `blocked` until full accepted `binance:futures` train/validation/test/backtest sessionized datasets are materialized and their split/leak/overlap/gap/lifecycle evidence is recorded.

Stage `07` must not start from the raw Stage `05` manifest or the bounded Stage `06` sample. Training, checkpoints, registry writes, calibration, provider/exchange side effects and activation remain out of scope until the required later stages are accepted.
