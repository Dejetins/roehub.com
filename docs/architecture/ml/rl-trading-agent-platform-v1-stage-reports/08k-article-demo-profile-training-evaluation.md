---
doc: rl-trading-agent-platform-v1-stage-08k-article-demo-profile-training-evaluation
status: blocked
stage: 08K
updated_at: 2026-07-02
---

# Stage 08K: article demo-profile training/evaluation

Статус: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08K` выполнил полный `30/10` article/demo-profile workflow: HF-original control branch и Roehub-native article-selector branch прошли full training, `Optuna` calibration на calibration split и untouched final holdout evaluation. Stage `09` не открыт.

Короткий итог: HF-control branch passed branch-level research gate. Roehub-native article-selector branch дала положительный final holdout PnL и обошла best sanity baseline, но strict native gate заблокировал candidate из-за concentration/stability risk: `single_group_dominates_final_result` и `ticker_stability_obviously_broken`. Поэтому итоговый `stage09_allowed=false`, а следующий разрешенный prompt — `08L`.

Доказательная граница текущего stage: `target_host_non_production_training_and_evaluation_pre_main`. Это pre-main Mac Studio non-production training/evaluation artifact evidence only, from `/Users/daniildegtyarev/Projects/roehub.com` plus artifacts under `/opt/roehub/state/rl_trading/`.

This evidence must not be reported as changed-code production proof. A later `post_main_production_runtime_proof` would require all of the following in order: target revision delivered to `main`, green CI/GitHub Actions for that revision, deploy or verified sync into the production runtime tree such as `/opt/roehub/app`, and then a smoke from that production runtime tree. Stage `08K` did not do those steps and makes no production-runtime claim. Browser/auth, exchange/provider side effects, registry promotion, paper/testnet/live/mainnet trading and secret reads were out of scope.

## Gate

| Check | Result |
|---|---|
| Ledger before work | `current_stage=08K`; `08I3` accepted; `08I4` accepted; `08J` accepted; `09` blocked |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08k-article-demo-profile-training-evaluation.md` |
| Prompt sha256 | `e8edba1fcd92155a59f0bee5da472a5fb58ea5f41575d9915490863d759f36ee` |
| Article dataset | Stage `08J` manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| Training/evaluation profile | `agent_history_len=30`, `agent_session_len=10`, `episodes=55000`, `env_steps=550000`, `trials=100` per branch |
| Calibration/final isolation | `Optuna` used calibration split only; final split evaluated after selection |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider side effects | `N/A`; no orders, no registry promotion, no live/paper/testnet execution |
| Changed-code production proof | `not collected`; later proof requires `main`, green CI/GitHub Actions, deploy/verified sync into `/opt/roehub/app` or equivalent production runtime tree, then smoke from that tree |
| Overall verdict | `blocked`; native strict gate failed, `stage09_allowed=false` |

## Methodology

| Field | Value |
|---|---|
| Уровень глубины | `integration/offline`: full branch training, calibration, final holdout and orchestrator summary |
| Тип задачи | Candidate-quality gate after accepted evaluator/action/reward repair and article-selector dataset materialization |
| Выбранная методология | Run HF-original control and Roehub-native article-selector branch under the source/demo `30/10` profile; tune only backtest/risk parameters on calibration split; decide on final holdout with strict native gate |
| Единица анализа | One materialized session plus grouped filtered backtest decision on calibration/final split |
| Основные метрики | Final PnL after costs, best sanity baseline, closed trades, monthly/ticker/volatility dominance, positive group ratios, action balance |
| Проверка качества данных | Stage `08J` accepted manifest, selector id, split manifests and calibration/final split isolation recorded in candidate/evaluation artifacts |
| Риски интерпретации | Positive native PnL is not sufficient for Stage `09`; strict stability gate must also pass to avoid accepting a concentrated research artifact |

## Runtime Artifacts

| Artifact | Value |
|---|---|
| Dual-branch summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/stage08k_dual_branch_cpu_run_summary.json` |
| Dual-branch summary file sha256 | `70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d` |
| Dual-branch summary hash | `9193103e37009cbecb0eb6851a76e2efcea90208a63129ec59daa267cf9ae836` |
| Overall `stage09_allowed` | `false` |
| HF candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage08k_article_demo_profile_training_evaluation_v1/hf_original/stage08c_hf_original_2829f520faca1e8cd60f/hf_original_candidate_manifest.json` |
| HF candidate manifest sha256 | `2376f871d69d3c319e73140e9d27ab8fed7144e29378f0472ade247c233d12c5` |
| HF `Optuna` summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_hf_original_2376f871_261ef7a00f6a9e6044b8/stage08k_optuna_summary.json` |
| HF `Optuna` summary file sha256 | `4245ea8c3190698f948f71fb482d50878cfd375d580ab056ddb7def8ccb96120` |
| HF final evaluation manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_hf_original_2376f871_261ef7a00f6a9e6044b8/final_holdout_6c6343328c9c/stage08d_evaluation_manifest.json` |
| HF final evaluation manifest sha256 | `7fd968fe2f20a08ce3f74915cad5eeea55dfae936063d6f40adc7e45a520f381` |
| Native candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage08k_article_demo_profile_training_evaluation_v1/roehub_native/stage08e_roehub_native_fd7c614b_7500ec1bd322437afd18/roehub_native_candidate_manifest.json` |
| Native candidate manifest sha256 | `03fd26aa9cbf3ee4d4d3f50e62301408dccfa443e10a2cf9875014b064b444cc` |
| Native `Optuna` summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/stage08k_optuna_summary.json` |
| Native `Optuna` summary file sha256 | `8585d4342dab24850cd077e5287de5faab251e848f18eb044f70cc410ebf6cec` |
| Native final evaluation manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/final_holdout_b2adb7da3abc/stage08f_evaluation_manifest.json` |
| Native final evaluation manifest sha256 | `c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07` |

Mac Studio command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && PYTHONUNBUFFERED=1 uv run python scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py --stage-label 08K --device-policy mps_preferred_cpu_fallback --generated-at-utc 2026-07-02T12:00:00Z"'
```

Mac Studio result:

```json
{"run_dir": "/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e", "run_id": "stage08k_dual_branch_cpu_76f51186c00ecb54255e", "status": "completed", "summary_path": "/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/stage08k_dual_branch_cpu_run_summary.json", "summary_sha256": "70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d"}
```

## Branch Results

| Branch | Training | `Optuna` status | Best trial | Closed trades | Final PnL after costs | Best sanity baseline | Strict gate | `stage09_allowed` |
|---|---|---|---:|---:|---:|---:|---|---|
| `hf_original_control_30_10` | `55000/55000` episodes, `550000/550000` env steps, `device=mps` | `accepted_for_research` | `13` | `328` | `10034.10614479` | N/A for overall native gate | no blockers | `true` branch-level only |
| `roehub_native_article_selector_30_10` | `55000/55000` episodes, `550000/550000` env steps, `device=mps` | `completed` | `39` | `316` | `12502.65333026` | `0.0` | `blocked` | `false` |

HF-control confirms the source/demo path can still produce a positive final holdout under this local workflow, but it is not the platform-quality gate for Stage `09`. Stage `09` depends on Roehub-native acceptance.

## Native Strict Gate

| Gate | Value | Result |
|---|---:|---|
| Candidate final PnL after costs | `12502.65333026` | passed |
| Best sanity baseline PnL after costs | `0.0` | passed |
| Closed trades | `316` | passed; minimum `100` |
| Monthly dominance | `0.5181999915221556` (`2025-05`, `3` groups) | passed; limit `0.8` |
| Ticker dominance | `0.056308441568191724` (`BROCCOLIF3BUSDT`, `291` groups) | passed; limit `0.8` |
| Volatility-bucket dominance | `0.954610281973835` (`high`, `3` groups) | blocked; limit `0.8` |
| Monthly positive group ratio | `1.0` | passed; minimum `0.25` |
| Ticker positive group ratio | `0.24054982817869416` | blocked; minimum `0.25` |
| Open-side dominance | `0.8006329113924051` (`open_long=253`, `open_short=63`) | passed; limit `0.95` |

Blockers:

- `single_group_dominates_final_result`
- `ticker_stability_obviously_broken`

The candidate is therefore useful blocked research evidence, not an accepted registry/promotion input.

## Decision

| Field | Value |
|---|---|
| `08K` status | `blocked` |
| HF branch | branch-level `accepted_for_research`; not sufficient for Stage `09` |
| Native branch | positive final holdout but strict stability gate blocked |
| `stage09_allowed` | `false` |
| `08L_allowed` | `true` |
| Next prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08l-reward-warm-start-research.md` |

## Business Impact

`08K` changes the failure mode from "native cannot make positive final holdout PnL" to "native can make positive final holdout PnL, but the result is not stable enough to accept." That is progress for research diagnosis, not a production or registry unlock.

The next useful work is Stage `08L`: reward/warm-start/contextual-bandit research, with the strict rule that it cannot silently replace the frozen reward/action contract or open Stage `09` without a later accepted candidate scorecard.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API/UI route or DTO changed. |
| Port contract | `none` | No application port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration/table/storage schema changed. |
| Config schema/defaults | `compatible-change` | Additive optional CLI stage/dataset selector support for offline research commands. |
| Request hash / cache key / persistence identity | `compatible-change` | Adds offline `08K` runtime artifact identity and branch/run hashes; no production request/cache identity changed. |
| Runtime artifacts | `compatible-change` | Adds sanitized non-production training/evaluation artifacts under `/opt/roehub/state/rl_trading/`. |
| Benchmark / rollout gate | `compatible-change` | Ledger advances from `08K` to `08L`; `stage09_allowed=false` remains explicit. |
| Browser-visible behavior | `none` | Browser/auth scope is `N/A`. |
| Performance hot path | `none` | Offline training/evaluation scripts only; no API/live inference hot path changed. |

## Conditional Operational Coverage

| Surface | Coverage |
|---|---|
| Service calls | `N/A`; no Roehub API, worker, queue, Redis, ClickHouse write, external provider, exchange SDK or browser service call was added or changed. |
| Timeout / retry / idempotency | `N/A`; no retry loop or side-effecting operation was introduced. |
| Unknown external side-effect state | `N/A`; no exchange/provider submit or money-moving call occurred. |
| Secrets and redaction | No secrets, tokens, cookies, credentials, raw provider payloads, account identifiers, HMACs or API keys were read or written. |
| Alerts / monitoring / runbook | `N/A`; no production runtime, scheduler, alert route, notification provider, incident workflow or runbook action changed. |
| Browser/auth | `N/A`; browser-visible behavior and authenticated UI were out of scope. |
| Mac Studio path contract | Runtime artifacts are under `/opt/roehub/state/rl_trading/`; no git command was run under `/opt/roehub/app`. |

## File Manifest

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `scripts/rl_trading/stage08e_roehub_native_full_training_run.py` | modified | Add opt-in `08J` sessionized manifest support for native training. | `compatible-change` additive CLI/data-source support |
| `scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py` | modified | Add opt-in `08J` sessionized manifest support for native evaluation. | `compatible-change` additive CLI/data-source support |
| `scripts/rl_trading/stage08g_cpu_optuna_calibration.py` | modified | Add `08K` stage label/default roots, article-selector dataset routing and strict native final gate. | `compatible-change` additive offline research path |
| `scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py` | modified | Add `08K` dual-branch orchestration, branch labels and native strict-gate handoff. | `compatible-change` additive offline research path |
| `scripts/rl_trading/stage08j_article_session_extractor_dataset.py` | modified | Persist split source `manifest_stage` and `selector_id` for downstream `08K` provenance. | `compatible-change` artifact metadata |
| `src/trading/contexts/rl_trading/domain/roehub_native_training.py` | modified | Accept Stage `08J` dataset dependency and record dataset-specific safety flags. | `compatible-change` additive offline dataset dependency |
| `src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py` | modified | Surface dynamic Stage `06`/`08J` dataset provenance in evaluation manifests. | `compatible-change` additive artifact metadata |
| `tests/unit/scripts/rl_trading/test_stage08g_cpu_optuna_calibration.py` | modified | Cover strict native final gate pass/fail behavior. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage08g_dual_branch_cpu_training_evaluation.py` | modified | Cover `08K` dry-run defaults to article-selector dataset. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md` | created | Stage `08K` blocked report. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08K` blocked, advance `current_stage` to `08L`, keep `09` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative with blocked `08K` and `08L` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding Stage `08K` report. | `compatible-change` docs index |

Existing Stage `08J` files in the same dirty worktree remain part of the previous accepted handoff and were not reverted.

Runtime artifact manifest:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/training_runs/stage08k_article_demo_profile_training_evaluation_v1/hf_original/stage08c_hf_original_2829f520faca1e8cd60f/` | created outside git; HF-control training run completed |
| `/opt/roehub/state/rl_trading/training_runs/stage08k_article_demo_profile_training_evaluation_v1/roehub_native/stage08e_roehub_native_fd7c614b_7500ec1bd322437afd18/` | created outside git; native article-selector training run completed |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_hf_original_2376f871_261ef7a00f6a9e6044b8/` | created outside git; HF-control `Optuna` and final holdout |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/` | created outside git; native `Optuna` and final holdout |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/` | created outside git; dual-branch orchestration summary |

## Quality Gates

| Gate | Result |
|---|---|
| Focused local pytest | passed: `12 passed in 2.21s` |
| Focused local ruff | passed |
| Focused local pyright | passed |
| `py_compile` on changed Python files | passed |
| Broad local RL gates | passed: `uv run ruff check ...`; `uv run pyright ...`; `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` -> `114 passed in 3.83s` |
| Remote Mac Studio focused dry-run | passed; dry-run summary status `planned` |
| Remote Mac Studio focused ruff | passed |
| Remote Mac Studio focused pytest | passed: `11 passed in 0.11s` |
| Remote Mac Studio focused pyright | passed |
| Mac Studio pre-main full training/evaluation artifact run | completed; dual-branch summary sha256 `70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d`; overall `stage09_allowed=false` |
| Docs index | passed: `uv run python -m tools.docs.generate_docs_index --check` |

## Review

Cold self-review fallback used; no independent reviewer/subagent was available in this run.

Verdict: `blocked` is correct. Fixed blockers before report: none after the pre-main artifact run; the candidate is intentionally rejected by strict stability gates. Residual risk: this is non-production artifact evidence from a scoped Mac Studio checkout sync, not a committed `main`/production proof. A later publish must preserve unrelated dirty Stage `08J` work and stage only scoped files.
