---
doc: rl-trading-agent-platform-v1-stage-08b-upstream-methodology-core-port
stage: "08B"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08B: Upstream Methodology Core Port

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08B` started after checking the ledger: Stage `08A` is `accepted`, `current_stage=08B`, and corrective Stage `08B` is allowed. Browser/auth QA is `N/A` for this offline ML core stage; the Roehub smoke Keycloak username and host-local password source were not used.

This stage ports the upstream-compatible core mechanics only. It does not run full HF or Roehub-native training, evaluate candidate quality, write registry rows, promote/activate a model, enable paper/testnet/live execution, or submit exchange orders.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/1a225941-0f63-4e4b-8906-afa949a603bb/pasted-text.txt` |
| Prompt sha256 | `67ae71b07420a33baa470f0042d3719a618703316e32f191b887f34bc44a2325` |
| Upstream repo | `https://github.com/YuriyKolesnikov/rl-trading-binance` |
| Upstream pinned SHA | `f71130903f8237351164f4b875494185465bf1ea` |
| Required upstream files checked | `config.py`, `configs/alpha.py`, `model.py`, `agent.py`, `replay_buffer.py`, `trading_environment.py`, `train.py`, `test_agent.py`, `backtest_engine.py`, `utils.py` |
| External code vendored | none |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | - | New upstream-compatible `08B` core: alpha-profile config, train-only normalization, state builder, environment rollout, SumTree PER, CNN dueling Torch agent, checkpoint policy, Q-value cache and filtered policy. | `compatible-change` additive Python domain surface |
| `scripts/rl_trading/stage08b_upstream_methodology_core_smoke.py` | - | - | Bounded operator smoke for the new core without full training or checkpoint artifacts. | `compatible-change` additive CLI |
| `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | - | Focused golden/unit coverage for normalization, state/action history, rollout masking, PER, CNN/target sync, checkpoint policy and filters. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08b_upstream_methodology_core_port.py` | - | - | Tiny CLI smoke fixture for local/target-host resource evidence. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08b-upstream-methodology-core-port.md` | - | - | This Stage `08B` report and evidence manifest. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export the new `08B` core identifiers and helpers. | `compatible-change` additive Python export |
| - | `apps/worker/rl_trading_trainer/main/main.py` | - | Add `stage08b` dispatch to the existing trainer worker entrypoint. | `compatible-change` additive worker subcommand |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark `08B` accepted and advance `current_stage` to `08C`. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after Markdown updates. | `compatible-change` docs index |

Runtime artifacts:

| Path | Host | Reason | sha256 |
|---|---|---|---|
| `/tmp/roehub_stage08b_core_smoke_local/stage08b_core_smoke_report.json` | local checkout | Local smoke fallback because local `/opt/roehub` was not writable. Contains sanitized metrics only. | `e25d09b01e78ff223f782ee46f1b23cd6fa5ec161e7be157fd0220fd245e5f5f` |
| `/opt/roehub/state/rl_trading/training_smokes/stage08b_upstream_methodology_core_port_v1/code_snapshot/upstream_methodology.py` | Mac Studio | Non-production exact code snapshot used for target-host smoke without mutating the already-dirty Mac Studio git checkout. Contains code only, no secrets or payload data. | `66e510f2298c015f8de709a4a899a1e023b53b4a16a8404e7b02ab842031b7fb` |
| `/opt/roehub/state/rl_trading/training_smokes/stage08b_upstream_methodology_core_port_v1/macstudio_smoke/stage08b_core_smoke_report.json` | Mac Studio | `target_host_non_production_sample_pre_main` core smoke report. Contains sanitized metrics only. | `29301e5b15f4d3c790e2ab26ca2d8a82b4242fc4c906c93b401f65247591d657` |

Outside expected paths: local `/tmp/roehub_stage08b_core_smoke_local/...` only, because `/opt/roehub` was not writable on the local checkout host. The target-host artifact used the required `/opt/roehub/state/rl_trading/` root.

Delivery state: `local-only` implementation plus `target_host_non_production_sample_pre_main` smoke. No branch, commit, PR, production deploy, `/opt/roehub/app` sync, browser/auth proof, registry write, or exchange side effect was performed.

## Implemented Core

| Area | Result |
|---|---|
| Architecture id | New public id `roehub_d3qn_cnn_dueling_v1`; required literal `upstream_methodology_parity` is in the config payload. |
| CNN dueling model | Dynamic Torch `ModuleDict` with Conv2d/ReLU/Dropout blocks, value and advantage streams, dueling merge and target-network state copy. Torch remains optional and imported only in Torch paths. |
| State builder | Builds upstream-shaped state from normalized `(29, 7)` history, extras `[position, unrealized, elapsed, remaining]`, and 3-action one-hot history. |
| Normalization | Computes stats from train sequences only; price channels use log returns, volume channels use `log(x + 1)`, and application does not recompute val/test stats. |
| Environment rollout | `UpstreamTradingEnvironment` uses agent-selected actions, action masks, no pyramiding, last-step open blocking/forced close, realized-PnL reward and flat-hold penalty through the accepted Stage `02C` reward function. |
| D3QN/PER agent | Adds epsilon-greedy interaction, deterministic seed handling, SumTree PER, `train_start`, Double-DQN target selection, Smooth L1 weighted loss, gradient clipping and target sync. |
| Checkpoint policy | Adds best/final selection interface; `best.pth` remains the default evaluation checkpoint and `final.pth` is diagnostic unless explicitly selected. |
| Backtest policy | Adds Q-value cache and filtered action policy with advantage thresholds and MC-dropout uncertainty rejection counts. |
| Historical MLP path | Left intact as Stage `07A/07B` smoke/debug/historical evidence only; it is not the new candidate-training default. |

## Parity Evidence

| Evidence | Result |
|---|---|
| Architecture literal | `roehub_d3qn_cnn_dueling_v1` exported from domain and present in smoke reports. |
| Epsilon rollout, not scripted transitions | Unit and smoke evidence show `scripted_transition_sequence_used=false`; local and Mac smoke each ran `30` environment transitions selected through epsilon-greedy agent interaction. |
| Train-only normalization | Unit fixture proves train-only stats hash differs from a leaked train+validation stats hash and validation application uses the frozen train stats. |
| Action/reward compatibility | Smoke report carries `ACTION_STATE_REWARD_CONTRACT_HASH_V1`; environment tests prove no-pyramiding and last-step forced close. |
| PER/target/gradient | Unit tests cover SumTree priority updates, `train_start`, gradient clipping and target sync. |
| Filtered backtest policy | Unit tests prove weak actions are rejected to hold, cache hits/misses are observable, and ensemble uncertainty can reject actions. |
| Mac Studio core/resource smoke | `status=accepted_smoke`, `selected_device=cpu`, `learn_update_count=29`, `target_sync_count=29`, `transition_count=30`, `rss_mb_after=272.796875`, report hash `1b3c42461acd98e86ff4bd5aef49ec8d989ef9cfee1bce34d30c38443e58d25c`. |

## Alpha Deviations

| Deviation | Reason / guardrail |
|---|---|
| Tiny smoke config used `initial_balance=100`, `batch_size=2`, `train_start=2`, `replay_capacity=128`, `target_update_freq=1`, fixed epsilon `1.0`, and `episodes=3`. | Stage `08B` is a non-production core smoke, not full training. The full `configs/alpha.py` defaults are represented in `UpstreamAlphaConfig`; full-size use belongs to `08C`/`08E`. |
| No checkpoint tensor is saved by the `08B` smoke. | The stage proves interfaces and rollout mechanics only. Checkpoint materialization and best/final artifacts belong to full training stages `08C` and `08E`. |
| Filter policy rejects to hold when action advantage is weak or MC-dropout uncertainty is high. | This is a conservative fail-to-hold policy for the port fixture. Future `08D`/`08F` can report exact evaluation threshold choices, but weak/uncertain actions are observable and do not silently pass. |
| The Mac Studio smoke executed from a code snapshot under `/opt/roehub/state/rl_trading/.../code_snapshot`. | The target checkout was already dirty before `08B`; the snapshot avoided mutating `/Users/daniildegtyarev/Projects/roehub.com` or `/opt/roehub/app` while still running the changed core on the target host. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol or service boundary changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `compatible-change` | Additive Python `UpstreamAlphaConfig` and additive `stage08b` worker/CLI subcommand; existing defaults remain unchanged. |
| Request hash / cache key / persistence identity | `compatible-change` | New in-memory Q-value cache identity for the new filtered policy only; no existing cache key changes. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call, auth, retry or external adapter behavior changed. |
| External side effects / unknown-state semantics | `none` | No exchange, DB, Redis, registry, paper/testnet/live or mainnet side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized Stage `08B` smoke reports and ledger evidence. |
| Alert / runbook semantics | `none` | No alerting or runbook behavior changed. |
| Benchmark / rollout gates | `compatible-change` | Stage gate is advanced from `08B` to `08C`; Stage `09` remains blocked until accepted `08F`. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline ML core path only; no live inference hot path or API runtime path changed. Resource evidence is smoke-level, not a performance claim. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `08A` is `accepted`, `current_stage=08B`, and Stage `08B` may run |
| Prompt hash | passed; `67ae71b07420a33baa470f0042d3719a618703316e32f191b887f34bc44a2325` |
| Upstream SHA | passed; `git ls-remote` observed `f71130903f8237351164f4b875494185465bf1ea` |
| Focused 08B tests | passed; `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py tests/perf_smoke/contexts/rl_trading/test_stage08b_upstream_methodology_core_port.py` -> `12 passed` |
| `uv run ruff check src/trading/contexts/rl_trading apps/worker/rl_trading_trainer scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/apps tests/perf_smoke/contexts/rl_trading` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps/worker/rl_trading_trainer scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/apps tests/perf_smoke/contexts/rl_trading` | passed, `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `409 passed, 3 warnings` |
| Local bounded core smoke | passed; `accepted_smoke`, report sha256 `e25d09b01e78ff223f782ee46f1b23cd6fa5ec161e7be157fd0220fd245e5f5f` |
| Mac Studio non-production core smoke | passed; `accepted_smoke`, report sha256 `29301e5b15f4d3c790e2ab26ca2d8a82b4242fc4c906c93b401f65247591d657` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because the available multi-agent tool contract requires an explicit user request before spawning subagents.

Review scope: Stage `08B` implementation/report, ledger handoff, file/runtime manifest, proof-boundary/browser-auth wording, contract impact, quality gates and `08C` handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: accepted after local follow-up fixes.

Blockers fixed: the report records prompt hash, file/runtime artifacts, target-host proof boundary, no candidate-quality claim and no production side effects; the ledger advances to `08C`; the docs index was regenerated and checked.

Local follow-up check: completed. Focused `08B` tests, required RL/app unit gate, ruff, pyright and docs index check passed after implementation.

Residual review risk: Stage `08B` remains local implementation plus non-production target-host smoke. Full HF training/evaluation and Roehub-native quality gates remain owned by `08C`-`08F`.

## Residual Risks

- `08B` proves core mechanics and tiny smoke behavior only; it makes no candidate-quality, profitability, registry, promotion, activation or runtime trading claim.
- Full HF-original training, validation-selected checkpoint materialization and resource progress evidence remain for `08C`.
- HF evaluation/backtest methodology parity remains for `08D`; Roehub-native full training/evaluation remain for `08E`/`08F`.
- Mac Studio smoke used a code snapshot because the target checkout was already dirty. It is target-host non-production evidence, not post-main production runtime proof.
- Funding, mark/index, leverage and production-grade futures realism remain governed by later evaluation/calibration stages; `08B` did not change those contracts.

## 08C Handoff

Stage `08C` may start from this accepted `08B` report and the ledger after verifying `current_stage=08C`.

`08C` must use `roehub_d3qn_cnn_dueling_v1` and the upstream-compatible core as the candidate-training path, not the historical MLP/scripted-transition Stage `07B` path. It must run full HF-original training only, persist train-only normalization stats, write durable episode/env-step progress, save both `best.pth` and `final.pth`, default evaluation to `best.pth`, and make no Roehub-native quality claim before `08D` accepts HF evaluation/backtest parity.
