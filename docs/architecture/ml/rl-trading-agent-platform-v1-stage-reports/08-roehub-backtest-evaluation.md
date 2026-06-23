---
doc: rl-trading-agent-platform-v1-stage-08-roehub-backtest-evaluation
stage: "08"
status: blocked
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08: Roehub Backtest Evaluation

Статус: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08` started after checking the ledger: Stage `07B` is `accepted`, `current_stage=08`, and the Stage `07B` report hands off candidate manifest `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/candidate_manifest.json` with file sha256 `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158`.

Stage `08` produced the offline evaluation harness, scorecards, sanity baselines, and simulator/accounting parity fixture. The stage is still `blocked` because the accepted Stage `07B` candidate is not research-save eligible: out-of-sample PnL after fees/slippage/research-funding policy is negative and does not clear the no-trade/hold sanity baseline.

## Scope

In scope:

- evaluate the accepted Stage `07B` research candidate on the accepted Stage `06` held-out dataset split;
- produce a scorecard with PnL after fees/slippage/funding policy, drawdown, trades, ticker stability, out-of-sample period, overfit indicators, and latency/resource notes;
- compare against `hold`, `no_trade`, deterministic `random`, and simple threshold sanity baselines;
- prove simulator/accounting parity with focused fixtures;
- write sanitized evaluation artifacts under `/opt/roehub/state/rl_trading/` and keep only report hashes/summaries in git.

Out of scope:

- model registry, candidate promotion, activation, calibration, paper/testnet/live/mainnet;
- exchange SDKs, exchange secrets, order intents, source events, or execution paths;
- public API, browser/UI, schema, migration, or service-runtime changes;
- user-owned custom model training or cloud/S3/model hosting.

## Короткое Объяснение

Stage `08` отвечает на один практический вопрос: можно ли сохранять исследовательскую модель Stage `07B` как кандидат для следующих registry/promotion stages. Ответ по текущим данным отрицательный. На held-out sessions модель теряет деньги после комиссий и slippage, не обгоняет базовый вариант `hold`/`no_trade`, а training/evaluation metrics показывают признаки переобучения. Поэтому результат Stage `08` намеренно блокирующий: следующий stage не должен регистрировать, продвигать или подключать эту модель к paper/testnet/live execution.

После review 2026-06-24 этот blocker трактуется шире, чем плохой seed/config. Stage `07B` candidate не считается полным переносом methodology из `YuriyKolesnikov/rl-trading-binance`: он использовал Roehub MLP-D3QN, offline scripted transitions и raw argmax evaluation вместо upstream CNN dueling model, environment-rollout training, epsilon-greedy/PER lifecycle, train-only normalization, validation-selected checkpoint and filtered backtest. Поэтому следующий допустимый путь не повторяет Stage `07B`; он начинается с corrective Stage `07C` methodology parity audit, затем `07D` core port, `07E` HF-original training, `08A` HF evaluation, `07F` Roehub-native training and `08B` Roehub-native evaluation.

## File Manifest

Planned concrete file list before implementation edits:

- `src/trading/contexts/rl_trading/domain/backtest_evaluation.py`
- `src/trading/contexts/rl_trading/domain/__init__.py`
- `scripts/rl_trading/stage08_roehub_backtest_evaluation.py`
- `tests/unit/contexts/rl_trading/domain/test_backtest_evaluation.py`
- `tests/perf_smoke/contexts/rl_trading/test_stage08_backtest_evaluation.py`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08-roehub-backtest-evaluation.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `docs/architecture/README.md` only if docs index regeneration is required

Final file manifest:

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/backtest_evaluation.py` | - | - | Additive offline Stage `08` simulator/evaluator, scorecard builder, sanity baseline policies, candidate D3QN policy loader, and parity fixture. | `compatible-change` offline domain helper |
| `scripts/rl_trading/stage08_roehub_backtest_evaluation.py` | - | - | Opt-in CLI for validating Stage `07B`/Stage `06` hashes, running candidate/baseline evaluation, and writing sanitized runtime artifacts under `/opt/roehub/state/rl_trading/`. | `compatible-change` operator helper |
| `tests/unit/contexts/rl_trading/domain/test_backtest_evaluation.py` | - | - | Deterministic tests for scorecard accounting, threshold baseline, parity fixture, and research-save status. | `compatible-change` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08_backtest_evaluation.py` | - | - | Optional `rl-ml` smoke proving Stage `08` can consume a tiny Stage `07B` checkpoint artifact end to end. | `compatible-change` optional test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08-roehub-backtest-evaluation.md` | - | - | Stage `08` report. | `compatible-change` docs/report only |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08` domain surface. | `compatible-change` additive exports |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record Stage `08` blocked status, evidence, blockers, touched files/contracts, and next-stage gate. | `compatible-change` docs/ledger only |
| - | `docs/architecture/README.md` | - | Docs index regeneration/check after adding this report. | `compatible-change` docs index only |

Outside expected paths:

- `tests/perf_smoke/contexts/rl_trading/test_stage08_backtest_evaluation.py` is outside the prompt's primary unit-test path, but is justified as a narrow optional `torch` runtime artifact smoke for the Stage `08` evaluator. It does not affect production code or default gates.

## Prompt Evidence

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08-roehub-backtest-evaluation.md` |
| Prompt sha256 | `d58781247270d3556a6ba436ed53241fe4e892c1069102e4367bfb1d006d6751` |
| Ledger state before implementation | Stage `07B` accepted; `current_stage=08`; Stage `08` pending |
| Required prerequisites | Stage `07B` accepted |
| Stage `07B` candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage07b_full_candidate_training_run_v1/stage07b_candidate_b43be9c1_61995c61_c5fbee2b/candidate_manifest.json` |
| Stage `07B` candidate manifest file sha256 | `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158` |
| Stage `06` sessionized manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Delivery state | `local-only`; Mac Studio evidence is `target_host_non_production_evaluation_pre_main` from a scoped source sync into `/Users/daniildegtyarev/Projects/roehub.com`; not delivered to `origin/main`; no `/opt/roehub/app` deploy or service reload |

## Methodology

| Field | Value |
|---|---|
| Depth | `standard_analysis`; executor prompt fixes the evaluation scope and does not require a separate user methodology approval gate. |
| Task type | ML research evaluation / backtest scorecard. |
| Method | Out-of-sample deterministic simulator evaluation against sanity baselines. |
| Simple explanation | Run the candidate and simple baselines on the same held-out sessions, then compare realized accounting metrics after costs. |
| Business explanation | This catches reward/simulator bugs and weak candidates before any registry or trading activation; it is not promotion evidence. |
| Unit of analysis | Session-level `binance:futures` held-out evaluation, aggregated by candidate/baseline and ticker. |
| Main metrics | Net PnL after costs, return percent, max drawdown, closed trades, win rate, ticker stability, out-of-sample period, overfit signs, latency/resource notes. |
| Data-quality check | Accepted Stage `06` manifest and Stage `07B` candidate manifest hashes matched before evaluation. |
| Interpretation risk | Research-only approximation; Stage `10A` owns promotion thresholds; positive PnL would not allow registry activation. |

## Logging And Redaction

| Surface | Status | Rule |
|---|---|---|
| Credentials / tokens / cookies | N/A | Stage `08` does not require credentials and does not read or write secrets. |
| Raw provider or exchange payloads | N/A | Stage `08` does not call exchange/provider APIs and does not record raw provider payloads. |
| Runtime logs and artifacts | covered | Runtime artifacts under `/opt/roehub/state/rl_trading/` contain sanitized scorecards, hashes, local file paths, policy names, and aggregate metrics only. |
| Report and ledger | covered | Git-tracked docs record sanitized summaries/hashes only; no checkpoint tensors, raw arrays, provider data dumps, or secret-bearing logs are included. |

## Implementation Summary

| Area | Result |
|---|---|
| Candidate policy | Loads the Stage `07B` `candidate_checkpoint` through the same D3QN architecture/forward helpers used by the trainer and selects argmax actions. |
| Simulator | Applies each model/baseline action through `apply_training_reward_step_v1`, preserving Stage `02C` action/reward/accounting semantics including last-step forced close. |
| Costs | PnL is after transaction fees and slippage. Funding is explicitly recorded as `research_zero_funding_no_point_in_time_arrays` because this offline evaluator does not consume point-in-time funding arrays. |
| Baselines | Runs `hold`, `no_trade`, deterministic `random`, and `simple_threshold` on the exact same selected sessions. |
| Scorecard | Records net PnL, return, max drawdown, trades, win rate, ticker stability, out-of-sample period, overfit warning codes, latency/resource notes, action counts and cost policy. |
| Safety | No registry write, promotion, activation, source event, order intent, exchange SDK, paper/testnet/live/mainnet, API, browser, schema, or service-runtime behavior is enabled. |

## Runtime Evidence

Mac Studio actual evaluation command:

```text
uv run --extra rl-ml python scripts/rl_trading/stage08_roehub_backtest_evaluation.py --torch-num-threads 1 --torch-num-interop-threads 1 --generated-at-utc 2026-06-24T12:00:00Z
```

Result:

| Field | Value |
|---|---|
| Evidence label | `target_host_non_production_evaluation_pre_main` |
| Host | `MacStudioDaniil` |
| Remote checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Candidate manifest sha256 | `709b4cc39d54ab1415e29c095aea6306d7ff9e0e25e0785e2605d42602a1a158` |
| Sessionized manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Evaluation manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08_roehub_backtest_evaluation_v1/stage08_eval_7765be0568ba3d29f207/stage08_evaluation_manifest.json` |
| Evaluation manifest sha256 | `7141c354c2408100c3f9313e38352fc040369c6c69a635d77c9a0112f0703f09` |
| Evaluation hash | `6a8a5d1258c10de7af8461f82038770b9050824f17a490c3ba243ce75fab2a11` |
| Scorecards path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08_roehub_backtest_evaluation_v1/stage08_eval_7765be0568ba3d29f207/scorecards.json` |
| Scorecards sha256 | `5a169f88234cd80fd6c666a8dea3c3579260b762fe8ff98443cff7ec41fe0009` |
| Selected sessions / symbols | `12,346` sessions; `300` symbols |
| Data-quality status | `pass_with_warnings`; warning is the explicit research-only zero-funding approximation |
| Parity fixture | passed; expected/observed total net PnL `9.78011` |
| Research candidate save allowed | `false` |
| Stage `09` allowed | `false` |

Scorecard summary:

| Policy | Kind | Net PnL after costs | Return % | Max drawdown % | Closed trades | Win rate | Ticker positive ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `stage07b_candidate` | candidate | `-3209.03013521` | `-0.25992468` | `0.26467203` | `17100` | `0.45339181` | `0.2` |
| `hold` | baseline | `0.0` | `0.0` | `0.0` | `0` | `0.0` | `0.0` |
| `no_trade` | baseline | `0.0` | `0.0` | `0.0` | `0` | `0.0` | `0.0` |
| `random` | baseline | `-4850.09984845` | `-0.39284787` | `0.39284787` | `23975` | `0.43537018` | `0.07` |
| `simple_threshold` | baseline | `-1990.12980951` | `-0.16119632` | `0.64558459` | `21462` | `0.4070916` | `0.21333333` |

Overfit / sanity warning codes:

- `candidate_non_positive_out_of_sample_pnl`
- `candidate_does_not_clear_best_sanity_baseline`
- `validation_td_mse_much_higher_than_train_loss`

## Quality Gates

| Gate | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08-roehub-backtest-evaluation.md` | passed; `d58781247270d3556a6ba436ed53241fe4e892c1069102e4367bfb1d006d6751` |
| `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_backtest_evaluation.py` | passed; `4 passed` |
| `uv run --extra rl-ml pytest -q tests/perf_smoke/contexts/rl_trading/test_stage08_backtest_evaluation.py` | passed; `1 passed` |
| Focused `uv run ruff check` on Stage `08` files | passed |
| Focused `uv run pyright` on Stage `08` files | passed; `0 errors` |
| Prompt gate `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| Prompt gate `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| Prompt gate `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `398 passed, 3 warnings` |
| Mac Studio optional-ML focused tests on scoped source sync | passed; `5 passed` |
| Mac Studio actual Stage `08` evaluation | completed; artifact status `blocked` because candidate failed research-save gate |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold Self-Review

Independent subagent mode was not used because the available subagent tool policy requires an explicit user request for delegation. The same checklist was applied locally as a cold self-review fallback.

```text
Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage 08 report, Stage 08 ledger rows/blocker/change log, additive evaluator/CLI/test contract, and Mac Studio non-production evaluation evidence.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release
Blockers fixed: ledger Stage 08 status/evidence/blocker rows and report docs-index gate were corrected before final handoff; no remaining artifact blocker.
Local follow-up check: completed
Residual risks: Stage 08 candidate remains blocked; Stage 09 is not allowed; work is local-only plus Mac Studio scoped source-sync evidence, not origin/main delivery or production proof; funding is a research-only zero-funding approximation.
```

## Contract Impact

Final classification:

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response payload, auth, or browser-visible behavior changed. |
| Port contract | `none` | No existing port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration, database table, registry table, or persisted service schema changed. |
| Config schema/defaults | `compatible-change` | Adds opt-in CLI flags and local runtime artifact schema only; no service defaults. |
| Request hash / cache key / persistence identity | `none` | No existing runtime identity, cache key, or request hash changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call contract changed. |
| External side effects / unknown state | `none` | Runtime writes are local files under `/opt/roehub/state/rl_trading/`; no exchange/account/order/provider side effect. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds Stage `08` report/ledger semantics and sanitized runtime evaluation artifact semantics. No credentials, tokens, cookies, raw provider payloads, exchange payloads, or secret-bearing logs are written to the report/ledger; runtime artifacts contain scorecards, hashes, and local file paths only. |
| Alert/runbook semantics | `none` | No monitoring config or operator alert behavior changed. |
| Browser-visible behavior | `none` | Prompt disables browser runtime verification and no UI changed. |
| Performance/resource evidence | `compatible-change` | Adds offline evaluation latency/resource notes; no verified hot path changed. |

## Mac Studio Proof Boundary

| Boundary label | Status | Evidence / rule |
|---|---|---|
| `target_host_readiness_pre_main` | collected | SSH reached `MacStudioDaniil`; remote git commands used `/Users/daniildegtyarev/Projects/roehub.com`; Stage `07B` candidate and Stage `06` manifest hashes matched. |
| `target_host_non_production_evaluation_pre_main` | collected | Scoped Stage `08` source/test files were synced into the Mac Studio checkout for non-production evaluation; focused optional-ML tests passed; actual evaluation wrote sanitized artifacts under `/opt/roehub/state/rl_trading/`. |
| `read_only_existing_runtime_smoke` | N/A | No existing `/opt/roehub/app` service or browser/runtime behavior was checked or changed. |
| `post_main_production_runtime_proof` | N/A | Not claimed. A future post-main production proof requires the changed Stage `08` code to be on `main`/`origin/main`, green GitHub Actions/CI, and Mac Studio deploy/sync verification before any production runtime evidence can be recorded; this chat did not deliver code to `origin/main` or deploy `/opt/roehub/app`. |

## Blockers And Residual Risks

| Item | Status | Next action |
|---|---|---|
| Candidate research PnL | Blocker | Candidate net PnL after costs is `-3209.03013521`; Stage `09` must not register it. |
| Sanity baseline comparison | Blocker | Candidate does not clear the best sanity baseline; `hold`/`no_trade` are `0.0` net PnL. |
| Training/evaluation divergence | Warning | Final validation TD MSE is much higher than train loss; repair must inspect reward scaling, feature normalization, action distribution, or training stability before another candidate handoff. |
| Funding realism | Warning | This Stage `08` evidence uses explicit research-only zero-funding approximation; production-grade futures promotion still needs a point-in-time funding/metadata policy in a later accepted stage. |
| Delivery | Residual | Work remains `local-only` plus Mac Studio scoped source-sync evidence; not delivered to `origin/main`, not deployed, and not production runtime proof. |

## Next-Stage Handoff

Stage `09` is not allowed. Do not register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit this candidate.

This report is now historical rejection evidence. The next active prompt is `.codex/agents/generated/rl-trading-agent-platform-v1/07c-upstream-methodology-parity-audit.md`.

Do not rerun historical Stage `08` or repair Stage `07B` as the active path unless the user explicitly asks to reproduce the rejection evidence. The corrective path is Stage `07C -> 07D -> 07E -> 08A -> 07F -> 08B`; Stage `09` remains blocked until Stage `08B` accepts a Roehub-native research candidate. Preserve the completed Stage `08` artifact paths/hashes above as the rejection evidence.
