---
doc: rl-trading-agent-platform-v1-stage-08f-roehub-native-backtest-evaluation
stage: "08F"
status: blocked
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-25"
---

# Stage 08F: Roehub-Native Backtest Evaluation

Status: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08F` started after checking the ledger: Stage `08E` is `accepted`,
provides a completed `roehub_native_candidate`, and `current_stage=08F`.
Browser/auth QA is `N/A` for this offline evaluation stage; the Roehub smoke
Keycloak username and host-local password source were not used.

This is `target_host_non_production_evaluation_pre_main` evidence only. No
production `/opt/roehub/app` sync, service reload, browser/auth proof, registry
write, promotion, activation, exchange side effect, paper/testnet/live run, or
mainnet submit was performed.

Do not treat the Mac Studio code snapshot as
`post_main_production_runtime_proof`. A future
`post_main_production_runtime_proof` can be recorded only after the changed code
is committed to `main`/`origin/main`, GitHub Actions/CI are green for that
commit, deployment/sync updates the production runtime tree on Mac Studio, and
the changed code is proven from that deployed runtime. None of those steps
occurred in Stage `08F`.

The candidate lifecycle and simulator/accounting parity completed, but the
Roehub-native grouped filtered backtest is negative after costs. Stage `09`
remains blocked.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/25a48e26-d8c0-4782-b65e-9d57e6f24c10/pasted-text.txt` |
| Prompt sha256 | `57ff876d2c187166174ae4998c95240b9c15077f8b089bb0089f19f8b843e6e8` |
| Repo prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08f-roehub-native-backtest-evaluation.md` |
| Repo prompt sha256 | `57ff876d2c187166174ae4998c95240b9c15077f8b089bb0089f19f8b843e6e8` |
| Previous-stage gate | passed: Stage `08E` is `accepted`, `current_stage=08F`, and `roehub_native_candidate` is present |
| Candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/roehub_native_candidate_manifest.json` |
| Candidate manifest sha256 | `c130ca5ede6f0e6f1d57e7940b385a52dbfab616bca0b01b2771f6de46613cdc` |
| Candidate manifest logical hash | `f22fbb9348ba616e33927e81f8c52f22d30cd487b8c84c68362272f3b6b7e53c` |
| Evaluation checkpoint | `best.pth` by default |
| Best checkpoint sha256 | `86896683503335e99a15d78c8e37e30e7bef673e7a92704f46b64d570821d3bc` |
| Train-only normalization stats hash | `8bb7e4d04b4b6a6e4035834b96c8460b2485e0525f2af2acb04e2d85ada3e247` |
| Stage `06` manifest | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` |
| Stage `06` manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py` | - | - | Stage `08F` evaluator: validates `08E` native manifest/checkpoint, loads train-only stats, runs raw diagnostics, grouped filtered native backtest, sanity baselines, parity fixture and research verdict. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage08f_roehub_native_backtest_evaluation.py` | - | - | Opt-in operator CLI for accepted Stage `06` test/backtest split artifacts and completed `08E` candidate. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_roehub_native_evaluation.py` | - | - | Focused domain coverage for native scorecards, random baseline, volatility buckets, parity fixture and safety flags. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08f_roehub_native_evaluation.py` | - | - | Tiny CLI smoke over fixture Stage `06` split artifacts and sanitized manifest output. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08f-roehub-native-backtest-evaluation.md` | - | - | This blocked Stage `08F` report. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/hf_original_evaluation.py` | - | Add optional volatility-score bucket aggregation to the shared scorecard shape used by `08F`; `08D` callers remain compatible when metadata is absent. | `compatible-change` additive scorecard field |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08F` evaluator identifiers and helpers. | `compatible-change` additive Python export |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08F` blocked, keep `current_stage=08F`, and keep Stage `09` blocked. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index |

Outside expected paths: none in git.

Runtime artifacts (`proof_boundary=target_host_non_production_evaluation_pre_main`):

| Path | Host | Reason | sha256 / state |
|---|---|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/code_snapshot/` | Mac Studio | Non-production code snapshot used to run current local Stage `08F` code without mutating `/Users/daniildegtyarev/Projects/roehub.com` or `/opt/roehub/app`. | source hashes include `hf_original_evaluation.py` `e7de975c124de96caf88c9ee7d985982ed28f893e2ea23b4d2ac65e7a807d291`, `roehub_native_evaluation.py` `ba17747449cba8139220e683b07ec9ac2f5308fb1739de524e0662bef6e91f15`, `__init__.py` `e72626623286c18fab1e4c5132fb56e5a13ccc3fa16c6cc254dc0a6816415bd0`, CLI `544382128c4ed081a70539bc5e11a41d9a4c0f2829bbd877edd10861c9d170e5` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/stage08f_roehub_native_c130ca5e_9934fa9be9f1a0b5c14a/stage08f_evaluation_manifest.json` | Mac Studio | Full Stage `06` native test/backtest evaluation manifest. | sha256 `6854055bd89f82446bb9952a3e98b66d9a20c0f5d72c74848be38caaf325581c`; evaluation hash `1a068069d91f642d4cf10e8f545380943a6ff697ac1f7589cf0160b2bea9c1b8`; status `blocked` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/stage08f_roehub_native_c130ca5e_9934fa9be9f1a0b5c14a/scorecards.json` | Mac Studio | Full scorecards by split/surface, ticker, month and volatility bucket. | sha256 `864e97a44d83728535e5983d441e2630bfe45c0d5fd8184e13be3a246e32c73e` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/stage08f_roehub_native_c130ca5e_9934fa9be9f1a0b5c14a/filtered_backtest_balance_curve.json` | Mac Studio | Filtered grouped backtest balance curve. | sha256 `c1406e388ac26a323bab69ebbd1b583d08b93307b51eb0e9dfa732885b137234` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08f_roehub_native_backtest_evaluation_v1/stage08f_roehub_native_c130ca5e_9934fa9be9f1a0b5c14a/simulator_accounting_parity_fixture.json` | Mac Studio | Deterministic simulator/accounting parity fixture. | sha256 `a74d7cbfc342b544c68b739a683e9e6b22e91fc78285c57aeae6b59b5f0cba5c`; passed |

Delivery state: `local-only` implementation plus
`target_host_non_production_evaluation_pre_main` managed evaluation. No branch,
commit, PR, deploy, production runtime sync, browser/auth proof, registry write,
promotion, activation or exchange side effect was performed by Stage `08F`.

## Методология Оценки

| Поле | Значение |
|---|---|
| Уровень глубины | `standard_analysis` |
| Тип задачи | ML/backtest evaluation gate for an offline Roehub-native research candidate |
| Единица анализа | Stage `06` Roehub-native session keyed by symbol and `signal_time_ms` |
| Основные метрики | net PnL after costs, return %, win-rate, closed trades, action distribution, rejection/skipped counts, drawdown, ticker/month/volatility stability |
| Baselines | `hold`, `no_trade`, `deterministic_random_valid_action`, `simple_recent_return_threshold` |
| Acceptance surface | `roehub_native_candidate_filtered_backtest`; raw argmax is diagnostic only |
| Проверка утечки | Evaluation uses Stage `08E` train-only normalization stats; `test` and `backtest` splits do not recompute normalization |
| Статус вывода | `не подтверждено`: lifecycle/parity work, but research candidate is blocked by negative native grouped filtered backtest PnL |

## Roehub-Native Evaluation Evidence

Raw argmax test diagnostics (`Stage 06 test`, diagnostic only):

| Metric | Value |
|---|---:|
| Sessions | `12,346` |
| Decisions | `123,460` |
| Closed trades | `17,916` |
| Win rate | `0.54046662` |
| Net PnL after costs | `-236,596.74518072` |
| Positive session ratio | `0.45537016` |
| Action counts | hold `87,628`, open_long `26`, open_short `17,890`, close `17,916` |
| Runtime | wall `29.08065704s`, `4,245.43365102` decisions/sec, RSS `503.890625 MiB` |

Filtered grouped backtest (`Stage 06 backtest`, acceptance surface):

| Metric | Value |
|---|---:|
| Source sessions | `14,731` |
| Selected sessions after timestamp grouping / `max_parallel_sessions=2` | `5,159` |
| Skipped sessions due parallel cap | `9,572` |
| Signal-time groups | `3,253` |
| Decisions | `51,590` |
| Closed trades | `5,183` |
| Win rate | `0.44665252` |
| Net PnL after costs | `-31,754.48132078` |
| Positive session ratio | `0.38011242` |
| Max drawdown | recorded in scorecard artifact |
| Runtime | wall `112.98587642s`, `456.60574256` decisions/sec, RSS `586.6875 MiB` |

Action filter:

| Field | Value |
|---|---|
| Selection strategy | `advantage_based_filter` |
| Thresholds | long `0.012695`, short `0.009902`, close `0.001141`, ensemble sigma `0.01` |
| Rejection counts | `weak_advantage_threshold=36,774` |
| Raw argmax action counts | hold `9`, open_long `23`, open_short `51,517`, close `41` |
| Requested filtered action counts | hold `4,450`, open_long `22`, open_short `41,928`, close `5,190` |
| Effective action counts | hold `41,224`, open_long `7`, open_short `5,176`, close `5,183` |

Baselines on the same grouped backtest selection:

| Policy | Net PnL after costs | Closed trades | Win rate | Positive session ratio |
|---|---:|---:|---:|---:|
| `hold` | `0.0` | `0` | `0.0` | `0.0` |
| `no_trade` | `0.0` | `0` | `0.0` | `0.0` |
| `deterministic_random_valid_action` | `-59,997.4667888` | `10,058` | `0.45893816` | `0.39193642` |
| `simple_recent_return_threshold` | `125,328.99619872` | `8,784` | `0.58891166` | `0.60767591` |

Period stability for filtered grouped backtest:

| Period | Sessions | Net PnL after costs | Closed trades | Win rate |
|---|---:|---:|---:|---:|
| `2025-03` | `1,621` | `-13,582.26744187` | `1,629` | `0.44137508` |
| `2025-04` | `1,943` | `-12,164.50208894` | `1,947` | `0.43708269` |
| `2025-05` | `1,595` | `-6,007.71178997` | `1,607` | `0.46359676` |

Volatility-bucket stability for filtered grouped backtest:

| Bucket | Sessions | Net PnL after costs | Closed trades | Win rate |
|---|---:|---:|---:|---:|
| `high` | `1,720` | `-9,003.26798302` | `1,740` | `0.47988506` |
| `low` | `1,720` | `-11,542.65027377` | `1,722` | `0.41579559` |
| `medium` | `1,719` | `-11,208.563064` | `1,721` | `0.44392795` |

Ticker stability examples:

| Group | Symbol | Sessions | Net PnL after costs | Closed trades | Win rate |
|---|---|---:|---:|---:|---:|
| Worst | `ORCAUSDT` | `31` | `-1,118.61657525` | `31` | `0.35483871` |
| Worst | `BROCCOLI714USDT` | `30` | `-1,111.60325132` | `31` | `0.4516129` |
| Worst | `ACTUSDT` | `32` | `-1,041.87209594` | `32` | `0.34375` |
| Best | `TUTUSDT` | `16` | `876.85159013` | `18` | `0.61111111` |
| Best | `TNSRUSDT` | `14` | `468.02005645` | `15` | `0.66666667` |
| Best | `LAYERUSDT` | `34` | `456.77204545` | `34` | `0.67647059` |

Simulator/accounting parity:

| Check | Value |
|---|---|
| Source | `apply_training_reward_step_v1` |
| Expected open fee | `-0.1` |
| Expected close net PnL | `9.88011` |
| Expected total net PnL | `9.78011` |
| Observed total net PnL | `9.78011` |
| Passed | `true` |

## Business And Ops Impact

Business impact: the completed native candidate is rejected and no model is
registered, promoted, activated, exposed to users, or used for paper/testnet/live
trading. Stage `09` remains blocked, so there is no product or revenue-impacting
runtime change from Stage `08F`.

Alerts/monitoring/runbook coverage: `N/A` for this offline non-production
evaluation. Stage `08F` does not add or modify services, workers, alerts,
dashboards, runbooks, Redis streams, database writes, exchange calls, browser
flows, or operator controls. The only operational output is the sanitized
evaluation artifact set under `/opt/roehub/state/rl_trading/`.

## Research Candidate Decision

Verdict: `blocked`.

| Check | Result |
|---|---|
| `best.pth` used by default | passed |
| Train-only normalization reused from `08E` | passed |
| Stage `06` held-out `test`/`backtest` splits used | passed |
| Raw argmax kept diagnostic-only | passed |
| Grouped filtered backtest with timestamp grouping and `max_parallel_sessions` | passed |
| Action filter / Q-cache / skipped-signal mechanics | passed |
| Simulator/accounting parity | passed |
| Candidate positive after costs | blocked: `-31,754.48132078` |
| Candidate beats best sanity baseline | warning: simple threshold baseline is `125,328.99619872` |
| Session-level stability | warning: positive session ratio is `0.38011242` |

Blockers:

- `candidate_non_positive_native_backtest_pnl`

Warnings carried forward:

- `candidate_does_not_clear_best_sanity_baseline`;
- `low_positive_session_ratio`;
- `stage08d_warning_register_carried_forward`;
- no stronger `90/60` or larger-profile training;
- no multiple-seed study;
- no Optuna/tuned backtest calibration in this stage.

Because Stage `08F` is blocked, no `research_candidate` is saved and Stage `09`
must not start.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol or service boundary changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `compatible-change` | Additive Python evaluation config and additive opt-in CLI flags. Existing runtime defaults remain unchanged. |
| Request hash / cache key / persistence identity | `none` | No request/cache/persistence identity changed. The Q-value cache is in-memory inside the offline evaluator only. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, DB, Redis, registry, paper/testnet/live or mainnet side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized Stage `08F` scorecards and evaluation manifests under `/opt/roehub/state/rl_trading/`. |
| Benchmark / rollout gates | `compatible-change` | Stage `08F` is blocked; Stage `09` remains blocked. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline evaluation only; no API or live inference hot path changed. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `08E` is `accepted`, `current_stage=08F`, and `roehub_native_candidate` is present |
| Prompt hash | passed; `57ff876d2c187166174ae4998c95240b9c15077f8b089bb0089f19f8b843e6e8` |
| Focused local ruff | passed |
| Focused local tests | passed; `2 passed` for new Stage `08F` domain/CLI tests |
| Focused local pyright | passed; `0 errors` |
| Prompt-level local ruff | passed; `uv run ruff check src/trading/contexts/rl_trading apps tests` |
| Prompt-level local pyright | passed; `0 errors` |
| Prompt-level local tests | passed; `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` -> `416 passed, 3 warnings` |
| Mac Studio snapshot focused tests | passed after snapshot metadata/source repair; `2 passed` |
| Initial Mac Studio full launch | blocked before evaluation by too many open memory-mapped Stage `06` files; loader changed to eager `.npy` reads and focused CLI smoke reran cleanly |
| Mac Studio full native evaluation/backtest | completed as `blocked`; manifest sha256 `6854055bd89f82446bb9952a3e98b66d9a20c0f5d72c74848be38caaf325581c`; evaluation hash `1a068069d91f642d4cf10e8f545380943a6ff697ac1f7589cf0160b2bea9c1b8` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed; docs index is up-to-date |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used
because the available multi-agent tool contract requires an explicit user
request before spawning subagents.

Review scope: Stage `08F` implementation/report, ledger update, file/runtime
manifest, proof-boundary/browser-auth wording, contract impact, quality gates,
scorecard/verdict, and Stage `09` handoff.

Review instructions:
`architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: release for blocked Stage `08F` result.

Blockers fixed: replaced pending docs-index/cold-head rows with final passed
evidence; report and ledger both keep `current_stage=08F` and Stage `09`
blocked; no research candidate saved.

Local follow-up check: completed; ruff, pyright, unit gate, and docs index check
passed after report/ledger updates.

Residual risks: stronger training/calibration/multiple seeds remain future
work; no production runtime proof; no registry/promotion activation.

## Residual Risks

- Stage `08F` does not prove native model quality; it rejects the completed
  native candidate on the research-save gate.
- Stronger `90/60` or larger-profile training, multiple seeds and
  Optuna/tuned backtest calibration remain required before any future
  promotion-grade review, but they were not run inside Stage `08F`.
- Futures funding, mark/index, leverage-tier and liquidation realism remain
  research-only approximation concerns until later accepted metadata stages.
- This remains non-production target-host evidence from a code snapshot, not
  `post_main_production_runtime_proof`.

## 09 Handoff

Stage `09` is not allowed.

Repair path: do not register, promote, activate, paper/testnet/live trade, or
mainnet submit this candidate. A future prompt must either supersede the native
training candidate with a stronger accepted training/calibration path outside
Stage `08F`, or explicitly redefine the research gate, then rerun Stage `08F`
against a completed `roehub_native_candidate` manifest. Until an `08F`
evaluation is accepted, the ledger must remain at `current_stage=08F` and Stage
`09` stays blocked.
