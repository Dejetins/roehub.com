---
doc: rl-trading-agent-platform-v1-stage-08a-upstream-methodology-parity-audit
stage: "08A"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08A: Upstream Methodology Parity Audit

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08A` started after checking the ledger: Stage `08` is `blocked`, its rejection evidence records `methodology_parity_not_yet_ported`, and `current_stage=08A`. Browser/auth QA is `N/A` for this docs/review stage; no Roehub smoke Keycloak username or host-local password source was used.

This stage freezes what "fully port the original methodology" means before implementation resumes. It does not implement code, train a model, evaluate a candidate, register a model, promote/activate anything, or touch paper/testnet/live/mainnet execution.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08a-upstream-methodology-parity-audit.md` |
| Prompt sha256 | `f62fec8437c642a11a7872610d1bc24bca8bd7fee4f193dac6006a0c947673e4` |
| Upstream repo | `https://github.com/YuriyKolesnikov/rl-trading-binance` |
| Upstream observed SHA | `f71130903f8237351164f4b875494185465bf1ea` |
| Upstream files read | `README.md`, `config.py`, `configs/alpha.py`, `utils.py`, `model.py`, `agent.py`, `replay_buffer.py`, `trading_environment.py`, `train.py`, `test_agent.py`, `backtest_engine.py`, `optimize_cfg.py`, `baseline_cnn_classifier.py` |
| Article cross-check | `https://habr.com/ru/articles/934258/`; used only to confirm repo-methodology claims. Repo code is authoritative where article/readme wording is broader. |
| External code vendored | none |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

Planned concrete file list before edits:

- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `.codex/agents/generated/rl-trading-agent-platform-v1/08c-original-hf-full-training-run.md`
- `.codex/agents/generated/rl-trading-agent-platform-v1/08e-roehub-native-full-training-run.md`
- `.codex/agents/generated/rl-trading-agent-platform-v1/08f-roehub-native-backtest-evaluation.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Final manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md` | - | - | Stage `08A` parity report, source matrix, gaps, accepted deviations, and downstream checklist. | `compatible-change` docs/report only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Tighten the frozen upstream methodology acceptance surface in the plan. | `compatible-change` docs/plan only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08A` accepted, advance `current_stage` to `08B`, and keep Stage `09` blocked until `08F`. | `compatible-change` docs/ledger only |
| - | `.codex/agents/generated/rl-trading-agent-platform-v1/08c-original-hf-full-training-run.md` | - | Add explicit Stage `08A` parity-report context and exact source-profile constraints for HF training. | `compatible-change` prompt artifact only |
| - | `.codex/agents/generated/rl-trading-agent-platform-v1/08e-roehub-native-full-training-run.md` | - | Add explicit Stage `08A`/`08C` context so Roehub-native training must preserve the frozen methodology and adaptation diff. | `compatible-change` prompt artifact only |
| - | `.codex/agents/generated/rl-trading-agent-platform-v1/08f-roehub-native-backtest-evaluation.md` | - | Add explicit Stage `08A` context and acceptance filters for the Roehub-native research-save gate. | `compatible-change` prompt artifact only |
| - | `docs/architecture/README.md` | - | Docs index regeneration after Markdown updates. | `compatible-change` docs index only |

Outside expected paths: none.

Runtime artifacts created by Stage `08A`: none.

Delivery state: local-only docs/prompt update; no branch, commit, PR, deploy, runtime ML artifact, browser/auth proof, or target-host sync was performed by Stage `08A`.

## Methodology Matrix

| Upstream component | Upstream fact at `f7113090` | Roehub requirement | Current Roehub gap / status |
|---|---|---|---|
| `config.py` / `configs/alpha.py` | Feature order is `open`, `high`, `volume_weighted_average`, `low`, `close`, `volume`, `num_trades`; base sequence is `full_seq_len=150`, `pre_signal_len=90`, demo `agent_history_len=30`, `agent_session_len=10`; `alpha.py` sets `action_history_len=3`, CNN maps `[32,64,128]`, kernels `[7,5,3]`, strides `[2,1,1]`, dense heads `[128,64]`, dropout `0.1`, `episodes=55_000`, `batch_size=16`, `learning_rate=1e-4`, `train_start=10_000`, PER capacity `230_000`, validation every `1000`, backtest `max_parallel_sessions=2`, `position_fraction=0.5`, advantage thresholds `0.012695/0.009902/0.001141`, ensemble samples `5`, `ensemble_max_sigma=0.01`. | Stage `08B` must implement a config surface that can represent these values and record deviations. Stage `08C` must default to this profile for HF-original training. Stage `08E` may adapt only with an explicit diff. | Historical Stage `07B` used Roehub `roehub_d3qn_mlp_v1`, `planned_training_steps=100_000`, `batch_size=256`, no upstream CNN profile, and no validation-selected `best.pth` candidate contract. |
| `utils.py` | Loads NPZ by `_keys_map_`; preserves `(ticker, signal_datetime)` keys; selects/reorders channels; computes normalization stats from training sequences only; normalizes price channels as log returns and volume channels as log volume; creates signal groups by signal datetime. | Roehub must keep HF-original key lineage, train-only normalization hash, channel arrangement, and grouped-signal backtest semantics. Validation/test/backtest splits must never recompute stats from non-train data. | Stage `06` has feature/session artifacts but Stage `07B` flattened prebuilt transitions and did not freeze upstream train-only normalization stats for candidate loading/evaluation. |
| `trading_environment.py` | Observation is normalized history plus extras `[position, unrealized_pnl, time_elapsed, time_remaining]` plus one-hot action history. Actions are `0 hold`, `1 long`, `2 short`, `3 close`; no pyramiding; last step blocks new open and forces close; reward is realized `pnl_change / initial_balance - inaction_penalty`; backtest has `backtest_step` and optional SL/TP/trailing risk management. | Stage `08B` must implement environment rollout as the candidate path while preserving Roehub Stage `02C` money-boundary semantics and last-step masks. `08D`/`08F` must use backtest-step semantics, not raw action replay only. | Stage `02C` covers action/reward semantics; Stage `07B` candidate path generated scripted offline transitions instead of agent-environment rollouts. |
| `model.py` | `DuelingQNetwork` uses Conv2d blocks with dropout, concatenates CNN features with extras/action history, and computes `value + advantage - mean(advantage)`. | Stage `08B` must expose `roehub_d3qn_cnn_dueling_v1` with CNN encoder, dropout, separate value/advantage streams, target network compatibility, state-shape tests, and parameter/config hash. | Current trainer is an MLP dueling network over flat observations (`roehub_d3qn_mlp_v1`); it is retained only as smoke/debug. |
| `agent.py` / `replay_buffer.py` | Agent owns policy/target networks, epsilon-greedy exploration, PER SumTree sampling, beta schedule, `train_start`, Double-DQN target selection, Smooth L1 weighted loss, gradient clipping, target sync, Q-value cache, MC-dropout ensemble inference. | Stage `08B` must implement these lifecycle pieces and tests: epsilon action interaction, train-start warmup, target sync, gradient clipping, PER priority updates/importance weights, Q-value cache identity, MC-dropout uncertainty. | Current Roehub PER exists but uses prebuilt transitions; no epsilon-greedy environment interaction, no upstream SumTree parity fixture, no MC-dropout policy gate. |
| `train.py` | Loads train/val/test NPZ, computes train-only stats, creates train and validation environments, trains by episodes through `env.step`, stores experiences, calls `agent.learn`, validates every `val_freq`, saves `best.pth` by validation metric and `final.pth`, then runs test evaluation and plots distributions. | Stage `08C`/`08E` must produce durable episode/env-step progress, train-only stats hash, validation curves, `best` and `final` checkpoints, and a manifest that selects `best` by validation metric by default. | Stage `07B` trained from deterministic transition sets and saved one final checkpoint as the candidate; validation curve did not choose a default best checkpoint. |
| `test_agent.py` | Recomputes train-only normalization stats, loads `best.pth` by default unless `use_final_model`, evaluates test sessions through environment actions, records and plots action sequences. | Stage `08D` must evaluate `hf_original_candidate` `best` checkpoint by default, record raw action/session diagnostics, and keep `final` checkpoint diagnostic-only unless justified. | Historical Stage `08` loaded Stage `07B` checkpoint and used raw argmax policy; it did not test the full upstream environment lifecycle. |
| `backtest_engine.py` | Groups backtest signals by timestamp, caps `max_parallel_sessions`, sizes positions by `position_fraction`, uses Q-value cache, supports advantage-based or MC-dropout ensemble filters, rejects weak actions to hold, runs `backtest_step`, records trade metrics and balance curve. | Stage `08D` and `08F` must implement grouped backtest acceptance with rejection counts, skipped-signal counts, thresholds, cache stats, trade/accounting metrics, balance curve hash, and sanity baselines. Raw argmax-only evaluation is diagnostic only. | Stage `08` produced a scorecard, but not the upstream grouped filtered backtest lifecycle. |
| `optimize_cfg.py` | Tunes backtest thresholds/risk-management knobs with Optuna over `run_backtest`; saves `best_backtest_cfg.json`. | Current `08A`-`08F` chain must not run tuning as acceptance. It may record threshold/config tuning as later calibration/promotion work, not as a way to hide weak methodology. | Stage `10`/`10A` remain the right place for calibration/promotion thresholds after `08F`. |
| `baseline_cnn_classifier.py` | Optional supervised CNN sanity baseline using the same normalized price history and test metrics. | Treat as a sanity baseline only. It must not become the business benchmark or replace RL acceptance; use it to detect obviously broken data/model plumbing. | Historical Stage `08` had simple baselines; future `08D`/`08F` should record technical baselines without treating them as promotion evidence. |

## Rejected Historical Candidate Path

The Stage `07B`/`08` candidate path is not accepted for candidate quality because it used:

- MLP-D3QN instead of upstream CNN dueling D3QN with dropout;
- offline scripted transitions instead of environment-rollout training through agent actions;
- raw argmax-only evaluation instead of grouped filtered backtest with Q-value cache, thresholds, rejection counts, and optional MC-dropout uncertainty;
- one final candidate checkpoint instead of validation-selected `best.pth` plus `final.pth`;
- no frozen train-only normalization stats shared across training/evaluation manifests.

Therefore Stage `09` remains blocked until Stage `08F` is accepted. Stage `07B` artifacts remain historical rejection evidence only.

## Required Downstream Checklist

| Stage | Must prove before acceptance |
|---|---|
| `08B` | `roehub_d3qn_cnn_dueling_v1`; environment rollout candidate path; epsilon-greedy agent interaction; train-only normalization helper; PER priority/update fixtures; target sync; gradient clipping; Q-value cache; advantage/ensemble filtered policy; historical MLP/scripted-transition path marked smoke/debug only. |
| `08C` | Full HF-original training from Stage `04` files/hashes; upstream `alpha.py` default profile or explicit deviation; durable episode/env-step `progress.jsonl`; train-only normalization stats hash; validation-selected `best` and `final`; no Stage `06` data and no evaluation acceptance. |
| `08D` | HF test evaluation and grouped backtest using the `best` checkpoint by default; raw argmax diagnostics separated; action rejection counts, thresholds, cache stats, scorecards, baselines, balance curve, and methodology-parity verdict. `08E` remains blocked if `08D` is blocked. |
| `08E` | Full Roehub-native Stage `06` training with the same methodology as `08C`; exact adaptation diff for dataset size/splits/cost/resource policy; validation-selected `best` and `final`; no HF data and no evaluation acceptance. |
| `08F` | Roehub-native held-out evaluation and grouped backtest with the same filtered lifecycle as `08D`; simulator/accounting parity; sanity baselines; positive net PnL after costs and no scorecard/sanity/overfit blocker before saving a research candidate. |
| `09` | May start only after accepted `08F`; it may register candidate metadata but still must not grant promotion-grade or runtime activation. |
| `10` / `10A` | Remain blocked until accepted `08F` and their own prerequisites; tuning/promotion thresholds belong here, not in `08D`/`08F` acceptance. |

## Accepted Deviations

| Deviation | Accepted? | Reason / guardrail |
|---|---|---|
| Keep demo `agent_history_len=30`, `agent_session_len=10` for HF parity chain | yes for `08C`/`08D` unless a later prompt explicitly scales | `configs/alpha.py` is the active upstream profile and the README/Habr call it demo mode. Larger 90/60/full model is future work, not a hidden requirement for `08C`. |
| Map upstream `initial_balance=10_000`, fee/slippage into Roehub artifact config instead of changing global live money semantics | yes | Stage `02C` money-boundary semantics and live execution ownership are higher-priority Roehub contracts. Offline research artifacts must record their cost policy explicitly. |
| Use Roehub runtime artifact roots and manifests instead of upstream `output/<config>` layout | yes | Needed for Roehub lineage, redaction, hash and Mac Studio artifact policy. Do not commit checkpoints/raw arrays. |
| Use Roehub Stage `06` high-volatility sessions for `08E`/`08F` | yes after `08D` accepted | This is the platform-native branch; the adaptation diff must state differences from HF-original splits and source windows. |
| Treat `baseline_cnn_classifier.py` as sanity-only | yes | It is not a business benchmark and does not reopen Stage `09`. |

No accepted deviation allows MLP-D3QN, scripted offline transitions, or raw argmax-only evaluation to be used as candidate-quality evidence.

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No Python ports/protocols changed in Stage `08A`. |
| DTO schema | `none` | No wire schema changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `none` | No runtime config changed; future prompts will add or classify config artifacts. |
| Request hash / cache key / persistence identity | `none` | No implementation identity changed. |
| Benchmark / rollout gates | `compatible-change` | Stage chain acceptance gates are tightened: Stage `09` remains blocked until accepted `08F`. |
| Browser-visible behavior | `none` | Browser QA is N/A and no UI changed. |
| Performance hot path | `none` | Docs/prompt-only stage; no runtime code path changed. |
| Docs / prompt artifacts | `compatible-change` | Adds parity report and prompt/ledger handoff requirements. |

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08a-upstream-methodology-parity-audit.md` | passed; `f62fec8437c642a11a7872610d1bc24bca8bd7fee4f193dac6006a0c947673e4` |
| Stage `08` prerequisite ledger gate | passed; Stage `08` is `blocked`, `current_stage=08A`, and corrective Stage `08A` is allowed |
| Upstream source SHA | passed; observed `f71130903f8237351164f4b875494185465bf1ea` |
| Upstream required files read | passed; all required files plus `utils.py` read from the pinned SHA |
| Habr methodology cross-check | passed; article agrees with repo on D3QN/PER, high-volatility `(150, 7)` sessions, environment loop, PER, train/test/backtest lifecycle, and filtered backtest/tuning scope |
| Browser/auth | N/A; no browser/auth surface used |
| Training/evaluation/runtime ML | N/A; forbidden by Stage `08A` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed; `docs/architecture/README.md` is up-to-date |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because the available multi-agent tool contract forbids spawning subagents unless the user explicitly asks for one.

Review scope: Stage `08A` report, plan/ledger handoff, downstream prompt updates for `08C`/`08E`/`08F`, file manifest, proof-boundary/browser-auth wording, and Stage `08B` handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: Release after fixes.

Blockers fixed: replaced the pending docs-index gate with a passed check; added explicit local-only delivery state; updated the ledger validation row to record docs-index success.

Local follow-up check: completed; `uv run python -m tools.docs.generate_docs_index --check` passed before final handoff.

Residual risks: Stage `08A` validates the methodology contract only, not implementation parity; `08B`-`08F` still must prove the actual port, training, and evaluation. Upstream `HEAD` may advance after observed SHA `f71130903f8237351164f4b875494185465bf1ea`; future audits should either keep this pin or deliberately refresh it.

## Next-Stage Handoff

Stage `08B` may start after verifying this report remains `accepted` and the ledger `current_stage` is `08B`.

The `08B` executor must not repair Stage `07B` or reuse its candidate path. The next implementation must port the upstream-compatible core first, prove parity fixtures, and leave full HF/Roehub-native training to `08C` and `08E`.
