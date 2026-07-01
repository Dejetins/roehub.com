---
doc: rl-trading-agent-platform-v1-stage-08i2-exhaustive-methodology-discrepancy-audit
stage: "08I2"
status: blocked
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-07-02"
---

# Stage 08I2: exhaustive methodology discrepancy audit

Статус: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08I2` выполнен как audit/report-only stage. Новое обучение, `Optuna`, exchange calls, browser/auth smoke, secrets, `/opt/roehub/app` production claim и изменения Python-кода не выполнялись. Доказательная граница: `target_host_non_production_forensic_pre_main` через Mac Studio runtime artifacts под `/opt/roehub/state/rl_trading/` плюс локальная source/code/docs проверка.

## Gate

| Gate | Result |
|---|---|
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08i2-exhaustive-methodology-discrepancy-audit.md` |
| Prompt sha256 | `9fc7085453f3ff8752236b4e0fbedbc1769f17fe530abfffd1fe8f97e7c23fca` |
| Ledger before work | `current_stage=08I2`; `08I` blocked; `08J`/`08K`/`09` pending/no |
| Upstream source pin | `YuriyKolesnikov/rl-trading-binance@f71130903f8237351164f4b875494185465bf1ea` |
| Matrix artifact | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i2_exhaustive_methodology_discrepancy_audit_v1/stage08i2_methodology_discrepancy_matrix.json` |
| Matrix artifact sha256 | `abe3a0c8ba42d6b453e2166bf3a9089aba4bfc6e6e07656708829990bba81c30` |
| Matrix status | `blocked`; row counts: `gap=7`, `blocked_by_prior_gap=1` |
| `stage09_allowed` | `false` |
| `next_stage_allowed` | `false` |

## Upstream Source Hashes

Source files were read from the pinned upstream commit without vendoring or checkout.

| File | sha256 |
|---|---|
| `configs/alpha.py` | `c8f0348379ed4deaf7dc306bbab039203e22e4039321ab294caedd2f5f698f9e` |
| `config.py` | `65bfc4b8fa0722defe75ecf38dbb0ce92c53d5edc2e96b8b5fe0d849fc6219d6` |
| `utils.py` | `38d00c1bbdafa0201f219e530544c70ac47dc0d143b503b158d52c8c96db2f25` |
| `trading_environment.py` | `c38154ee416f1fb3de59c2f7085092d0237216c7854e70ba89863d9676920c8c` |
| `model.py` | `042f406b0c35222bb79d659883d935454b12f42f4551daa06dc95e3a08a396cc` |
| `agent.py` | `49ef8faaba845eb31207704fae23a73a9f784af0a4b6aef9323fd8be769e2fab` |
| `replay_buffer.py` | `4c0f806232408a4f4fe6d71ca4627d5dd81b1fe18589aead72b070295444ce68` |
| `train.py` | `890a3557a3378dee7ed7aae8f65db2a29b1da2c890d02fa9dd5d5286824eeca5` |
| `test_agent.py` | `02ae41899a8168ce1fe3175f2353f50f380a301c90e3f581261be95d3f82be0f` |
| `backtest_engine.py` | `d05e426fdad3acb24df4c74fce17536d584e56a0b9e528160c5cb9762e179892` |
| `optimize_cfg.py` | `f6b2c542958cdce4c1cec6096cdae619304f67740b79098e136bf8dbfbe646dd` |
| `baseline_cnn_classifier.py` | `cf790ae8b0057393c27fc377ccfbac66d8d3f35653295403139ee8628eb71865` |

## Runtime Inputs

| Artifact | Path | sha256 / hash |
|---|---|---|
| Stage `06` session manifest | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Stage `06` leakage report | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_leakage_report.json` | `cbe1424bab47b4907cdee4b4585d107a449650dcde9f8b39b06d4f867e2e370a` |
| Stage `08F` native evaluation manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/stage08f_roehub_native_c130ca5e_9934fa9be9f1a0b5c14a/stage08f_evaluation_manifest.json` | `6854055bd89f82446bb9952a3e98b66d9a20c0f5d72c74848be38caaf325581c` |
| Stage `08G` dual-branch summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/stage08g_dual_branch_cpu_full_20260626T123500Z/stage08g_dual_branch_cpu_run_summary.json` | `84bf7c09d5f9654a2b695657b55b6dbe5d0a210407eec1ac238141152882f4be` |
| Stage `08H` diagnostics summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/diagnostics/stage08h_dataset_diagnostics_summary.json` | `9a0fe21114dfc25cf3fb2c2f183f5a8cf8bc2faf398ad9295fc1d71ca8cae338` |
| Stage `08H` 90/60 run summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/dual_branch_runs/stage08h_dual_branch_mps_90_60_full_20260626T215849Z/stage08h_dual_branch_cpu_run_summary.json` | file sha `97ad82d6afd737c3d9a183e7c27f94457d36a1273aa4d0b9d65c811e01372e5b`; internal `summary_hash=f4820678327b78137522418e1e4b7e105c702ccb6f3e3fc52b57176a6b3dc82b` |
| Stage `08H` corrected native recheck | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/optuna/stage08h_roehub_native_9f6e307f_7cbdd825ddd9f8aacb88/manual_final_holdout_rechecks_20260629/manual_final_holdout_recheck_summary.json` | file sha `30d8455361519b798bc4d86b82c908d24f1d9d44226677e9c6e449dba42563b9`; internal `summary_hash=01bb66fbf69a0ee7871b12600af0fb5a752ceaa559442f5425d8c6848b0a1f46` |
| Stage `08I` trace manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/stage08i_trace_manifest.json` | `6e33daa8bf4b857d9aef3db3bdf2ccf93fab20f90a7d400c3ff2ea1d764ad13d` |

## Methodology Discrepancy Matrix

| Surface | Source expected behavior | Roehub current behavior | Status | Severity | Repair required | Recheck required | Next stage allowed |
|---|---|---|---|---|---|---|---|
| `session_extractor_policy` | Article/upstream branch depends on the source article session selector; upstream repo itself evaluates provided HF splits. | Stage `06` uses `pre_signal_realized_volatility_plus_range_v1`: `pre_signal_len=90`, `post_signal_len=60`, top `1%`, cap `64`, 30m stride, 150m embargo. The article selector is not materialized. | `gap` | High | Materialize article-selector dataset in a repair path only after evaluator parity blocker is repaired or explicitly superseded. | Re-run leakage/embargo/lifecycle/rejected-window proof and matrix row. | no |
| `dataset_geometry_and_distribution` | Dataset geometry should match the intended methodology branch before native quality conclusions. | Stage `06` current selector creates much larger and differently distributed native splits than HF: backtest `14,731` vs HF `3,186`, test `12,346` vs HF `3,400`; volatility buckets and session density differ materially. | `gap` | High | Rebuild/select article-methodology dataset or document an explicit superseding dataset contract. | Counts by split/ticker/month, volatility buckets, session density, lifecycle exclusions, ratios. | no |
| `past_only_signal_strength` | Past-only features should show enough signal after leakage-free selection before spending on more RL training. | Oracle opportunities exist, but native past-only supervised sanity is weak: native balanced accuracy is `0.3428400835` for `30/10` and `0.3677879955` for `90/60`, below recent-return balanced baselines `0.5866836232` and `0.6095069448`. | `gap` | High | Re-evaluate after evaluator parity and article selector; do not infer RL failure or success from current native selector alone. | Repeat supervised/oracle diagnostics on repaired dataset and fixed evaluator. | no |
| `reward_sparsity_and_semantics` | `trading_environment.py` training reward is realized PnL over initial balance minus flat hold penalty; `backtest_step()` reports reward `0.0` and PnL in `info`. | Roehub training reward is compatible, but Stage `08I` trace exposes training reward in a backtest trace field where upstream reports `0.0`. Reward is sparse: trade-step proxy `0.2` for `30/10`, `0.0333333333` for `90/60`, while dense proxy is about `0.53`-`0.56`. | `gap` | Medium | Separate `training_reward` from `backtest_reporting_reward`; do not silently redesign reward. | Trace parity after field-semantics fix; reward diagnostics after dataset/evaluator repair. | no |
| `action_q_policy_distribution` | `backtest_engine.py` filters unmasked Q advantages and lets environment/backtest validity semantics handle invalid/no-op/last-step behavior. | Roehub current evaluation masks invalid Q before `FilteredBacktestPolicy`; Stage `08F` shows pathological short bias: raw selected `open_short=51517` vs `open_long=23`; effective `open_short=5176` vs `open_long=7`. | `gap` | High | Align or explicitly supersede mask/filter/order semantics, then reassess action distribution. | Re-run Q/action distribution, invalid-action accounting and first-diff traces. | no |
| `optuna_and_calibration_overfit` | `optimize_cfg.py` searches thresholds/risk parameters and selects from a multi-objective Pareto set; position fraction and max sessions are not active search variables in source. | Roehub corrected zero-trade selection, separates calibration/final holdout and requires trade-sufficient trials, but native final holdout still fails: Stage `08G` native PnL `-145.16434371` vs baseline `95274.46982886`; corrected Stage `08H` trial `82` final PnL `-229005.38413725` vs baseline `481012.90631972`. | `gap` | High | Keep final holdout locked; tune only after data/evaluator gaps are closed. | Calibration/final split evidence, baseline comparison and trade-count gate after repair. | no |
| `sanity_baselines` | Sanity baselines must remain hard blockers for candidate/research-save acceptance. | Baselines correctly block current native branch. Stage `08F` simple recent-return baseline `125328.99619872` beats filtered candidate `-31754.48132078`; Stage `08G` and `08H` native holdouts also lose to baselines. | `gap` | High | Preserve baseline gate; no Stage `09` until candidate clears it under repaired methodology. | Repeat baseline scorecard after repaired dataset/evaluator/training path. | no |
| `full_evaluator_backtest_parity` | Full dynamic parity requires source-derived scheduling, sizing, filtering, balance and reward-report semantics. | Stage `08I` already found a material first diff: upstream rolling `open_sessions` and shared `balance * position_fraction`; Roehub exact `signal_time` group cap and independent per-session sizing. Dynamic deeper comparison is blocked by this prior material gap. | `blocked_by_prior_gap` | High | Repair or explicitly supersede Stage `08I` evaluator/session parity before any article dataset or training continuation. | Repeat source-derived traces and full evaluator parity matrix after repair. | no |

## Dataset Geometry Snapshot

| Branch | Split | Sessions | Ratio | Symbols | Median sessions/symbol | Period | Vol q33 | Vol q66 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| HF | train | `24086` | `0.7515367094` | `309` | `55` | `2020-01..2024-08` | `0.0084209970` | `0.0112167272` |
| HF | validation | `1377` | `0.0429654591` | `280` | `4` | `2024-09..2024-11` | `0.0082108954` | `0.0106677973` |
| HF | test | `3400` | `0.1060875534` | `362` | `6` | `2024-12..2025-02` | `0.0093747314` | `0.0124759844` |
| HF | backtest | `3186` | `0.0994102780` | `321` | `5` | `2025-03..2025-05` | `0.0089085385` | `0.0118065333` |
| Stage `06` | train | `13381` | `0.2638886150` | `220` | `64` | `2020-01..2024-08` | `0.0216192691` | `0.0287741016` |
| Stage `06` | validation | `10249` | `0.2021219950` | `250` | `44` | `2024-09..2024-11` | `0.0096906225` | `0.0125580168` |
| Stage `06` | test | `12346` | `0.2434772319` | `300` | `44` | `2024-12..2025-02` | `0.0123164513` | `0.0169894745` |
| Stage `06` | backtest | `14731` | `0.2905121581` | `358` | `45` | `2025-03..2025-05` | `0.0098288365` | `0.0141629242` |
| Stage `06` | post_hf_extension | `33065` | n/a | `528` | `64` | `2025-06..2026-06` | `0.0176294419` | `0.0288671584` |

## Signal And Reward Diagnostics

| Branch/profile | Backtest sessions | Oracle positive ratio | Oracle mean best net return | Reward trade-step proxy | Dense proxy | Ridge accuracy | Ridge balanced | Majority accuracy | Recent-return balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HF `30/10` | `3186` | `1.0` | `0.0348674254` | `0.2` | `0.5529190207` | `0.5734463277` | `0.5728238959` | `0.4962335217` | `0.4274078278` |
| HF `90/60` | `3186` | `1.0` | `0.0739193702` | `0.0333333333` | `0.5513287299` | `0.5787821720` | `0.5780207427` | `0.4849340866` | `0.4066547156` |
| Stage `06` native `30/10` | `14731` | `0.9961984930` | `0.0157745443` | `0.2` | `0.5313253833` | `0.5193130134` | `0.3428400835` | `0.5096734777` | `0.5866836232` |
| Stage `06` native `90/60` | `14731` | `0.9969452176` | `0.0426501432` | `0.0333333333` | `0.5584275274` | `0.5536623447` | `0.3677879955` | `0.5127961442` | `0.6095069448` |

## Data Quality And Validation Read

Stage `06` existing leakage report remains internally accepted for its current selector: `selected_session_count=83772`, `cross_split_overlap_violations_count=0`, `embargo_violations_count=0`, `lookahead_violations_count=0`, `lifecycle_violations_count=0`, `within_split_overlap_pairs=60145`, `rejected_windows_count=304`. This does not accept the methodology branch for current corrective work, because the article selector and evaluator parity are still missing.

The correct quality label for `08I2` is therefore `pass_with_blocking_methodology_gaps`: available artifacts are readable and hash-checked, but they prove that the current native methodology surface is not ready for `08J`, `08K`, `09`, or more training.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API | `none` | No API route, DTO or service behavior changed. |
| Ports | `none` | No port implementation or interface changed. |
| Persisted schema | `none` | No migration or persisted table changed. |
| Config/defaults | `none` | No runtime config changed. |
| Request hash / cache identity | `none` | No cache key or request identity changed. |
| Browser-visible behavior | `none` | Browser/auth was out of scope and not run. |
| Exchange side effects | `none` | No private/auth exchange calls and no orders. |
| Runtime artifacts | `compatible-change` | Added sanitized non-production matrix artifact under `/opt/roehub/state/rl_trading/`; no production runtime mutation. |
| Documentation/gates | `compatible-change` | Stage ledger now records `08I2` as blocked and keeps downstream stages closed. |
| Performance | `none` | No performance claim or benchmark change. |

## Business And Ops Coverage

| Surface | Coverage |
|---|---|
| Business impact | Research/docs gate only. No user-facing product behavior, tariff, strategy control, registry activation, paper/testnet/live trading capability, or mainnet readiness changed. |
| Conditional service calls | `N/A` for Roehub API, browser/auth, private exchange endpoints, order submit, ClickHouse writes, Monit/launchd/service reload and `/opt/roehub/app`. The only remote work was Mac Studio filesystem artifact reads and one sanitized matrix artifact write under `/opt/roehub/state/rl_trading/`; upstream source was read from pinned public GitHub raw files. |
| Logging and redaction | Report and matrix contain paths, counts, statuses, hashes and sanitized metrics only. No secrets, tokens, cookies, credentials, raw private provider payloads, account identifiers, HMACs or API keys were read into docs. |
| Alerts, monitoring and runbook | `N/A`; no production runtime, scheduler, alert route, notification provider, runbook action or incident workflow changed. |

## File Manifest

| Path | Change | Reason | Contract impact |
|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md` | created | Stage `08I2` report and matrix summary. | `compatible-change` docs |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i2_exhaustive_methodology_discrepancy_audit_v1/stage08i2_methodology_discrepancy_matrix.json` | created outside git | Sanitized runtime evidence matrix. | `compatible-change` non-production artifact |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08I2` blocked and keep downstream stages closed. | `compatible-change` docs/gate |
| `docs/architecture/README.md` | modified after docs-index regeneration | Include the new stage report in the architecture index. | `compatible-change` docs index |

## Quality Gates

| Check | Result |
|---|---|
| Prompt hash recorded | passed |
| Ledger prerequisite check | passed: `08I2` was current and executable; downstream was blocked |
| Upstream source hash/read check | passed |
| Mac Studio artifact read/hash check | passed |
| Matrix artifact write/read/hash check | passed |
| Browser/auth smoke | `N/A` |
| Python gates | `N/A`; no Python code changed |
| Docs index | passed after regeneration: `uv run python -m tools.docs.generate_docs_index --check` |

## Cold Review

Mode: `cold self-review fallback`. Independent subagent review was not used because the current tool policy does not allow spawning subagents unless the user explicitly asks.

Verdict: `blocked`, and the block is intentional. The review checked that all mandatory rows are present, statuses are from the allowed set, each row has source/current/evidence/repair/recheck/next-stage decision, secrets are absent, browser/auth is not claimed, and `08J`/`08K`/`09` remain closed.

Residual risks: dynamic full evaluator parity is not complete because Stage `08I` found a prior material scheduling/sizing blocker; no new training was run; article-selector behavior is not yet materialized.

## Handoff

No existing next prompt is currently allowed. `08J`, `08K` and `09` remain blocked. The next executable work must be a repair or superseding forensic prompt for the `08I`/`08I2` blockers: rolling `open_sessions`, shared `balance * position_fraction` sizing, Q-mask/filter ordering, reward trace field semantics, and then a repeat of the source-derived evaluator/session parity plus `08I2` matrix recheck. Only after those blockers are accepted can `08J` article session extractor work be considered.
