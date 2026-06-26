# Stage 08G — dual-branch CPU Optuna training/evaluation

## Статус

| Поле | Значение |
|---|---|
| `stage` | `08G` |
| `status` | `blocked` |
| `created_at` | `2026-06-26` |
| `completed_at` | `2026-06-26` |
| `proof_boundary` | `target_host_non_production_training_and_evaluation_pre_main` |
| `stage09_allowed` | `false` |
| `owner` | `Roehub agents / research executor` |

Stage `08G` был открыт после пользовательской проверки статьи и исходного репозитория. Цель этапа была не принять старый `08F`, а проверить полный workflow в более честной форме: обучение модели, затем backtest-configuration optimization через `Optuna`, затем независимый final holdout.

Итог: полный sequential CPU run на `macstudio` завершился, но Stage `08G` заблокирован. HF-original ветка после `Optuna` стала положительной и получила branch-level `stage09_allowed=true`, но Roehub-native ветка после `Optuna` осталась отрицательной на финальном Stage `06` `backtest`. Поэтому общий `stage09_allowed=false`.

## Почему Нужен Был Этот Stage

`08F` проверил Roehub-native candidate на фиксированных порогах и показал провал:

- filtered native backtest PnL after costs: `-31754.48132078`;
- best sanity baseline: `125328.99619872`;
- positive session ratio: `0.38011242`;
- simulator/accounting parity: passed;
- runtime status: `blocked`.

Это было полезное diagnostic evidence, но оно отвечало только на вопрос: "работает ли текущий candidate на текущих фиксированных настройках?". Оно не доказывало, что полный workflow из статьи после `Optuna` тоже не работает.

В статье [Habr `934258`](https://habr.com/ru/articles/934258/) финальный backtest показан в workflow, где `Optuna` используется для подбора backtest configuration. В upstream `configs/alpha.py` значение `max_parallel_sessions=2` является конфигурационным default, а в `optimize_cfg.py` подбор `max_sessions` есть только как закомментированная строка. Поэтому `max_parallel_sessions=2` не считается доказанным optimum для Roehub dataset.

## Методологические Границы

| Решение | Статус |
|---|---|
| Device policy | `cpu_only_deterministic` |
| Execution order | sequential на `macstudio`: `hf_original` training -> `hf_original` `Optuna` -> `roehub_native` training -> `roehub_native` `Optuna` |
| `Optuna` trials | `100` на каждую branch |
| `Optuna` jobs | `1` |
| Search space | `long_thr`, `short_thr`, `close_thr`, `use_rm`, `stop_loss`, `take_profit`, `trail`, `max_sigma only for ensemble_q_filter` |
| `max_parallel_sessions` | `2`, source default from upstream `configs/alpha.py`, not optimized in this stage |
| `position_fraction` | `0.5`, source default, not optimized in this stage |
| Final split optimized by `Optuna` | `false` |

Важно: `Optuna` калибровала параметры на calibration split, а финальный verdict считался на отдельном final split.

## Операторский Запуск

Entry point:

```bash
uv run --extra rl-ml python scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py
```

Full run:

| Поле | Значение |
|---|---|
| `run_id` | `stage08g_dual_branch_cpu_full_20260626T123500Z` |
| summary path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/stage08g_dual_branch_cpu_full_20260626T123500Z/stage08g_dual_branch_cpu_run_summary.json` |
| summary file sha256 | `84bf7c09d5f9654a2b695657b55b6dbe5d0a210407eec1ac238141152882f4be` |
| `summary_hash` | `2f567ea42ee877fc89434334e4a905e3b1e6444cc5c63a21f21c252655771736` |
| `status` | `completed` |
| `stage09_allowed` | `false` |
| branch order | `hf_original`, `roehub_native` |

## Branch Results

| Branch | Training status | `Optuna` status | Branch `stage09_allowed` | Final holdout PnL after costs | Best sanity baseline | Verdict |
|---|---|---|---:|---:|---:|---|
| `hf_original` | `completed` | `accepted_for_research` | `true` | `2914.76906569` | `4508.37753925` | Positive after `Optuna`, but still below best sanity baseline. This is HF methodology evidence only. |
| `roehub_native` | `completed` | `completed` | `false` | `-145.16434371` | `95274.46982886` | Blocked: final native holdout is negative and does not beat sanity baseline. |

Простой смысл: на исходном HF dataset метод после `Optuna` смог получить положительный результат, но на нашем Stage `06` Roehub-native dataset финальный holdout остался отрицательным. Для Stage `09` важна именно Roehub-native ветка, поэтому Stage `09` не открыт.

## Training Artifacts

| Branch | Candidate manifest | File sha256 | Manifest hash | Evidence |
|---|---|---|---|---|
| `hf_original` | `/opt/roehub/state/rl_trading/training_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/hf_original/stage08c_hf_original_0d26983f2749996fa0a4/hf_original_candidate_manifest.json` | `e6eb9e6e0e6702fb4a1010784619cb6c4592a482e81a1c467c027a4a6180bf13` | `0fca40763b959591b7cfb6e4548fdda5ae9129f4c6ed23bd66de80d5ab073adb` | `55000/55000` episodes, `550000/550000` env steps, `device=cpu`, `training_used_environment_rollout=true` |
| `roehub_native` | `/opt/roehub/state/rl_trading/training_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/roehub_native/stage08e_roehub_native_61995c61_6de2d626ed5461c98325/roehub_native_candidate_manifest.json` | `55e868f35244c2d39d16c183108cffc936584e9cf0fb15aec3fada6b447e1297` | `053f0ff4befd727e1e68d49822a2db132df2b53561cb2f173e2ffef838bc335f` | `55000/55000` episodes, `550000/550000` env steps, `device=cpu`, `training_used_environment_rollout=true` |

## Optuna Artifacts

| Branch | Summary | File sha256 | Summary hash | Trials | Best trial | Best params |
|---|---|---|---|---:|---:|---|
| `hf_original` | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/branch_evaluations/stage08g_hf_original_e6eb9e6e_5ed04031c3031aad82bd/stage08g_optuna_summary.json` | `1d3695810609f0deb02f136ae870382db5d89139ff06d236e2a7561c42ddd3b7` | `e1122efb344778806d239ecf9fcf7a005a9fb4235e44978e20dd4380f57ef728` | `100/100` | `93` | `long_thr=0.029660751817924526`, `short_thr=0.01944060532382035`, `close_thr=0.01470327676293068`, `use_rm=false` |
| `roehub_native` | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/branch_evaluations/stage08g_roehub_native_55e868f3_ee53c1929ff0468ed80c/stage08g_optuna_summary.json` | `51c2a435f7785172c687761bef63eb5c4464de61d716016f98b3c37efdfaf6cd` | `ef8ab2ec04ce614ef41734a77ae3c6c49a0beb39ccc3807a5d55153996394db7` | `100/100` | `1` | `long_thr=0.015074446258474733`, `short_thr=0.024814836489377153`, `close_thr=0.0028087759081231105`, `use_rm=true`, `stop_loss=0.014622090070405885`, `take_profit=0.021425899595750336`, `trail=0.004079567040352733` |

## Calibration И Final Holdout Splits

| Branch | Calibration split | Final split | Split rule |
|---|---|---|---|
| `hf_original` | HF `test_data.npz`, `3400` sessions, sha256 `ff72d998fbf7d507b3db46e543aae324bece368a50ad043c057217ec2c744b1b` | HF `backtest_data.npz`, `3186` sessions, sha256 `dce732fda8fe1d33e92617d12f0defa3e202013617b91bb34df4d0b65aa023ee` | `calibration_split_used_for_optuna=true`, `final_split_optimized_by_optuna=false` |
| `roehub_native` | Stage `06` `test`, `12346` sessions, `300` split artifacts, manifest sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` | Stage `06` `backtest`, `14731` sessions, `358` split artifacts, manifest sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` | `calibration_split_used_for_optuna=true`, `final_split_optimized_by_optuna=false` |

## Final Holdout Scorecards

| Branch | Final manifest | Evaluation hash | Closed trades | Net PnL after costs | Return after costs | Win rate |
|---|---|---|---:|---:|---:|---:|
| `hf_original` | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/branch_evaluations/stage08g_hf_original_e6eb9e6e_5ed04031c3031aad82bd/final_holdout_0b4b93d5e7cb/stage08d_evaluation_manifest.json` | `e4bc6052a8ee9bfd19c8d6e2c2d8ba3488bf820a6b3fc840ef072fc38e5aa9c0` | `19` | `2914.76906569` | `0.01063396` | `0.84210526` |
| `roehub_native` | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/full/branch_evaluations/stage08g_roehub_native_55e868f3_ee53c1929ff0468ed80c/final_holdout_5eb5b3f7c894/stage08f_evaluation_manifest.json` | `37a0b728a72aff9a310615bde7be59817873e93d6b4abdbd7090d0241c0c4b95` | `3` | `-145.16434371` | `-0.00028138` | `0.33333333` |

Native final manifest verdict:

- `native_research_verdict.status=blocked`;
- `research_candidate_save_allowed=false`;
- `stage09_handoff.allowed=false`;
- blocker: `candidate_non_positive_native_backtest_pnl`;
- `simulator_accounting_parity_passed=true`;
- best sanity baseline net PnL after costs: `95274.46982886`.

## Acceptance Gate

| Gate | Result |
|---|---|
| Both branch runs finish with durable status files and candidate manifests | passed |
| `stage08g_dual_branch_cpu_run_summary.json` records both branch outcomes and `stage09_allowed` | passed |
| `Optuna` study artifacts are written and hash-recorded | passed |
| Source default or override decision for `max_parallel_sessions` is explicit | passed: `source_default=2`, `used_value=2`, `optimized_in_this_stage=false` |
| Final native holdout PnL after costs is positive | failed: `-145.16434371` |
| Final native candidate does not materially lose to sanity baselines | failed: best sanity baseline `95274.46982886` |
| Stage `06` manifest hash and split evidence are recorded | passed |
| `stage09_allowed=true` only from final accepted manifest, not calibration trials | passed: overall `stage09_allowed=false` |

Result: Stage `08G` is `blocked`; Stage `09` remains blocked.

## Бизнес И Операционная Граница

This was offline research work only. It did not create registry activation, UI activation, paper/testnet/live orders, exchange intents, secrets access, billing behavior or mainnet behavior.

| Area | Impact |
|---|---|
| User-visible product | none |
| Exchange/live execution | none |
| Runtime services | none |
| Service calls | N/A; stage ran offline training/evaluation CLIs only and did not call Roehub API, exchange APIs, Redis execution streams or live services. |
| Secrets | none |
| Alerts/monitoring/runbook | N/A; no deployed runtime, scheduler, API route, worker daemon, Redis stream, exchange integration or user-facing mode was activated. Runtime evidence is limited to local files under `/opt/roehub/state/rl_trading/`. |
| Cost/risk | Mac Studio CPU time and local disk usage under `/opt/roehub/state/rl_trading/` |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08g-dual-branch-cpu-optuna-training-evaluation.md` | - | - | Stage `08G` final blocked report for completed CPU-only dual-branch Optuna workflow. | `compatible-change` docs/report |
| `scripts/rl_trading/stage08g_cpu_optuna_calibration.py` | - | - | Source-search-space `Optuna` calibration and final holdout CLI for one trained branch. | `compatible-change` additive opt-in CLI |
| `scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py` | - | - | Sequential CPU orchestrator for HF-original training, HF `Optuna`, Roehub-native training and native `Optuna`. | `compatible-change` additive opt-in CLI |
| `tests/unit/scripts/rl_trading/test_stage08g_dual_branch_cpu_training_evaluation.py` | - | - | Unit coverage for sequential CPU orchestration, fresh candidate sha256 handoff and source-like `100` trial default. | `none` test-only |
| - | `pyproject.toml` | - | Add optional `Optuna` dependency to `rl-ml`. | `compatible-change` optional dependency |
| - | `uv.lock` | - | Lock optional `Optuna` dependency. | `compatible-change` lockfile |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Insert Stage `08G`, correct post-`08F` Optuna and `max_parallel_sessions` assumptions, keep Stage `09` blocked. | `compatible-change` architecture plan |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record completed `08G` result as blocked and keep Stage `09` blocked. | `compatible-change` docs/ledger |

Runtime artifact manifest is intentionally summarized with paths and hashes only. Full datasets, checkpoints, logs and study DB artifacts remain under `/opt/roehub/state/rl_trading/` and are not committed.

## Проверка Перед Финалом

Review mode: `cold_self_review_fallback`.

Verdict:

- blocker fixed in execution path: Mac Studio SSH recovered, bounded real-data smoke passed, and full sequential CPU run completed;
- blocker fixed in code path: optional `Optuna`, risk-management overrides and sequential CPU orchestrator are present and covered by focused checks;
- source-methodology correction recorded: `Optuna` is required for a fair post-article workflow check, and `max_parallel_sessions=2` is only a source default;
- final research blocker remains: Roehub-native final holdout after `Optuna` is negative and below sanity baseline;
- Stage `09` remains blocked.
