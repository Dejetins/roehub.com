---
doc: rl-trading-agent-platform-v1-stage-08l-reward-warm-start-research
status: accepted
stage: 08L
updated_at: 2026-07-02
---

# Stage 08L: reward/warm-start research fallback

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08L` выполнен как bounded research после accepted `08I3`/`08I4`/`08J` и blocked `08K`. Он не заменял Stage `02C` reward/action contract, не писал registry, не активировал модель, не запускал paper/testnet/live/mainnet trading, не выполнял exchange/provider side effects, не использовал browser/auth и не читал секреты.

Доказательная граница: `target_host_readiness_pre_main`; runtime subtype: `target_host_non_production_research_pre_main`. Это Mac Studio non-production research artifact под `/opt/roehub/state/rl_trading/`, а не `post_main_production_runtime_proof` и не production-runtime claim для changed code.

Обязательный marker: `reward_research_not_contract_replacement`.

## Gate

| Check | Result |
|---|---|
| Ledger before work | `current_stage=08L`; `08I3` accepted; `08I4` accepted; `08J` accepted; `08K` blocked; `09` blocked |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08l-reward-warm-start-research.md` |
| Prompt sha256 | `9bf1d2dc9934fc31ab1cb18232662eacf9f20a2a64ed392d49824aa54842b407` |
| `08I2` matrix | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i2_exhaustive_methodology_discrepancy_audit_v1/stage08i2_methodology_discrepancy_matrix.json`, sha256 `abe3a0c8ba42d6b453e2166bf3a9089aba4bfc6e6e07656708829990bba81c30`; all `8` mandatory rows present |
| `08I4` recheck matrix | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i4_post_repair_methodology_recheck_v1/stage08i4_methodology_recheck_matrix.json`, sha256 `a03da05df6aef2a59d13c28c167561afbfce230df347f01f5a5a7f61d79dc0b3`; all `8` rows present; evaluator parity closed by `08I3` |
| `08J` article dataset | `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json`, sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| `08K` blocked summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/stage08k_dual_branch_cpu_run_summary.json`, sha256 `70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d` |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no orders, registry promotion, paper/testnet/live execution or mainnet submit |
| Overall verdict | `accepted` as research-path decision; `stage09_allowed=false`; next explicit prompt is `08M` |

## Методология Анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `research`: reward/warm-start/contextual-bandit sanity after a blocked candidate-quality stage |
| Тип задачи | Проверить, оправдан ли отдельный supervised warm-start / behavior-cloning / contextual-bandit candidate path |
| Выбранная методология | Bounded NumPy diagnostics over accepted Stage `08J` train/test/backtest splits; no Torch training and no Optuna in `08L` |
| Простое объяснение метода | Из past-only признаков обучается closed-form ridge classifier на oracle labels, затем его решения проверяются как fixed-horizon contextual-bandit proxy на untouched backtest split |
| Объяснение на языке бизнеса | Мы проверили, есть ли стабильный простой сигнал, который не требует заново менять reward-контракт и не повторяет unstable DQN failure из `08K` |
| Единица анализа | One `article_future_10m_5pct_contrast_v1` session from the accepted Stage `08J` dataset |
| Основные метрики | Balanced accuracy, fixed-horizon proxy PnL, trade count, monthly/ticker/volatility dominance, ticker positive group ratio, action balance |
| Базовые сравнения | hold/no-trade, deterministic random, simple recent-return threshold, Stage `08K` native final scorecard, oracle-label upper bound as non-candidate reference |
| Риски интерпретации | Fixed-horizon contextual-bandit proxy is not a full candidate scorecard; it can justify `08M`, not `09` |

## Bounded Experiment Matrix

| Hypothesis | Dataset branch | Profile | Max runtime | Metrics | Stop conditions | Artifact |
|---|---|---|---|---|---|---|
| Stage `02C` realized-PnL reward remains the baseline; dense/shaped reward is research-only. | `roehub_native_article_selector_30_10` | `30/10` | Bounded NumPy diagnostics only; no Torch training and no Optuna. | Current reward non-zero step proxy; dense mark-to-market proxy. | Missing input matrix; `08K` not blocked. | `stage08l_reward_warm_start_research_summary.json` |
| Past-window supervised oracle-label warm start should beat simple technical baselines before a new candidate stage is justified. | `roehub_native_article_selector_30_10` | `30/10` | Closed-form ridge classifier on loaded Stage `08J` arrays. | Balanced accuracy; prediction counts; fixed-horizon proxy PnL. | Classifier does not beat recent-return baseline; proxy PnL does not clear best baseline. | `stage08l_reward_warm_start_research_summary.json` |
| Contextual-bandit sanity proxy must be stable across month/ticker/volatility groups before it can justify a bounded candidate prompt. | `roehub_native_article_selector_30_10` | `30/10` | Single fixed-horizon proxy pass on untouched backtest split. | Monthly/ticker/volatility dominance; positive group ratios; action balance. | Single group dominates; ticker stability broken; action distribution pathologically one-sided. | `stage08l_reward_warm_start_research_summary.json` |

## Runtime Artifact

Mac Studio command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run python scripts/rl_trading/stage08l_reward_warm_start_research.py --generated-at-utc 2026-07-02T16:30:00Z"'
```

Result:

```json
{"candidate_path_justified": true, "run_dir": "/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553", "run_id": "stage08l_reward_warm_start_99a00ffa43c83b9ac553", "stage09_allowed": false, "status": "accepted", "summary_path": "/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json", "summary_sha256": "5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1"}
```

| Field | Value |
|---|---|
| Summary path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json` |
| Summary file sha256 | `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1` |
| Summary payload hash | `59bdb534baa97bd172266edb4405774ecc12e2005900386ce4d4bae479f28216` |
| Status | `accepted` |
| `candidate_path_justified` | `true` |
| `stage09_allowed` | `false` |

## Comparison Evidence

| Policy / reference | Final proxy PnL after costs | Closed trades | Return pct after costs | Candidate? |
|---|---:|---:|---:|---|
| `hold_no_trade` | `0.0` | `0` | `0.0` | baseline |
| `deterministic_random_contextual_bandit` | `-9100.9261598118` | `2758` | `-0.910092616` | baseline |
| `simple_recent_return_threshold_contextual_bandit` | `-53438.2414711871` | `4157` | `-5.3438241471` | baseline |
| `supervised_oracle_label_warm_start_contextual_bandit` | `23018.4187849668` | `4162` | `2.3018418785` | research proxy candidate path |
| `oracle_label_upper_bound_not_candidate` | `425716.5220601573` | `4162` | `42.571652206` | non-candidate upper bound |
| Stage `08K` native DQN final scorecard | `12502.65333026` | `316` | not used as proxy | blocked reference |

Supervised backtest sanity:

| Metric | Value |
|---|---:|
| Ridge balanced accuracy | `0.584192378` |
| Recent-return baseline balanced accuracy | `0.418019662` |
| Majority baseline balanced accuracy | `0.5` |
| Ridge prediction counts | `hold=0`, `long=1664`, `short=2498` |
| Backtest oracle labels | `hold=0`, `long=2068`, `short=2094` |

Stability for `supervised_oracle_label_warm_start_contextual_bandit`:

| Gate | Value | Result |
|---|---:|---|
| Monthly dominance | `0.4996531449664722` (`2025-05`, `3` groups) | passed vs `0.8` limit |
| Ticker dominance | `0.023532616397542595` (`ACTUSDT`, `323` groups) | passed vs `0.8` limit |
| Volatility-bucket dominance | `0.786834239482547` (`high`, `3` groups) | passed narrowly vs `0.8` limit |
| Monthly positive group ratio | `0.6666666667` | passed vs `0.25` minimum |
| Ticker positive group ratio | `0.653250774` | passed vs `0.25` minimum |
| Open-side dominance | `0.6001922152811149` (`open_long=1664`, `open_short=2498`) | passed vs `0.95` limit |

Reward proxy:

| Metric | Value |
|---|---:|
| Current reward non-zero trade-step ratio proxy | `0.2` |
| Dense mark-to-market non-zero step ratio proxy | `0.5537962518` |
| Mean flat-wait penalty before oracle entry | `0.0011753964` |
| Positive oracle trade count | `4162` |

Interpretation: `08L` does not accept a reward replacement. It shows that a past-only supervised warm-start/contextual-bandit path is strong enough to justify a bounded `08M` candidate-scorecard stage.

## Decision

| Field | Value |
|---|---|
| `08L` status | `accepted` |
| Candidate path justified | `true` |
| Decision reason | `next_corrective_warm_start_candidate_stage_required` |
| Next prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08m-supervised-warm-start-candidate-scorecard.md` |
| Next prompt sha256 | `01dcaebae004fd566dac3082ff450d435f0afefc5381523071535268b41e469b` |
| `stage09_allowed` | `false` |

Stage `09` remains blocked because `08L` is research-proxy evidence, not a full strict candidate scorecard. `08M` must convert this into a bounded final-holdout candidate scorecard and may open `09` only if the strict native gate passes.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API route, DTO or browser behavior changed. |
| Port contract | `none` | No application port/interface changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration or persisted application schema changed. |
| Config schema/defaults | `none` | No production runtime config/default changed. |
| Request hash / cache key / persistence identity | `none` | No production request/cache identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side-effect and unknown-state semantics | `none` | No exchange/provider submit or money-moving call occurred. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized `08L` research summary and report; no secrets or raw provider payloads. |
| Benchmark / rollout gates | `compatible-change` | Advances ledger from `08L` to explicit corrective `08M`; keeps `stage09_allowed=false`. |
| Browser-visible behavior | `none` | Browser/auth scope is `N/A`. |
| Performance hot path | `none` | Offline research script only; no API/live inference hot path changed. |

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
| `scripts/rl_trading/stage08l_reward_warm_start_research.py` | created | Opt-in bounded `08L` reward/warm-start/contextual-bandit research CLI. | `compatible-change` additive offline research CLI |
| `tests/unit/scripts/rl_trading/test_stage08l_reward_warm_start_research.py` | created | Focused coverage for fixed-horizon proxy, candidate-path decision and matrix validation. | `none` test-only |
| `.codex/agents/generated/rl-trading-agent-platform-v1/08m-supervised-warm-start-candidate-scorecard.md` | created | Explicit next corrective prompt required after accepted `08L` path decision. | `compatible-change` prompt-pack artifact |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md` | created | Stage `08L` accepted report and evidence handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08L` accepted, advance `current_stage` to `08M`, keep `09` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative with accepted `08L` and `08M` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding Stage `08L` report. | `compatible-change` docs index |

Runtime artifact manifest:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json` | created outside git; accepted research summary sha256 `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1` |

## Quality Gates

| Gate | Result |
|---|---|
| Focused local pytest | passed: `uv run pytest -q tests/unit/scripts/rl_trading/test_stage08l_reward_warm_start_research.py` -> `3 passed` |
| Focused local ruff | passed: `uv run ruff check scripts/rl_trading/stage08l_reward_warm_start_research.py tests/unit/scripts/rl_trading/test_stage08l_reward_warm_start_research.py` |
| Focused local pyright | passed: `uv run pyright scripts/rl_trading/stage08l_reward_warm_start_research.py tests/unit/scripts/rl_trading/test_stage08l_reward_warm_start_research.py` -> `0 errors` |
| Remote Mac Studio focused pytest | passed: `3 passed` |
| Remote Mac Studio focused ruff | passed |
| Mac Studio bounded research run | passed with `status=accepted`; summary sha256 `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1`; `stage09_allowed=false` |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` -> `117 passed` |
| Docs index | passed: `uv run python -m tools.docs.generate_docs_index --check` |
| Whitespace diff | passed: `git diff --check` |

## Residual Risks

- `08L` uses a fixed-horizon contextual-bandit proxy, not the final grouped evaluator scorecard. It justifies `08M`, not `09`.
- Volatility-bucket dominance for the warm-start proxy passed narrowly at `0.786834239482547` versus the `0.8` limit; `08M` must recheck this with final scorecard semantics.
- The dense reward proxy shows better signal coverage, but no reward replacement is accepted in `08L`.
- Production proof was not collected. This stage has no production runtime surface and no `/opt/roehub/app` mutation.

## Cold-Head Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `08L` report, `08M` generated prompt, plan/ledger handoff, docs index, and `08L` Python/test artifacts.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release
Blockers fixed: replaced stale pending gate rows with passed prompt-level evidence; normalized `08M` prompt proof-boundary to `target_host_readiness_pre_main` while keeping Mac Studio non-production detail as subtype.
Local follow-up check: completed
Residual risks: `08L` remains proxy research only; `08M` must rerun strict final-holdout scorecard before any Stage `09` opening.

## Handoff

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/08m-supervised-warm-start-candidate-scorecard.md`.

`08M` must build a bounded supervised warm-start/contextual-bandit candidate scorecard from the accepted `08L` evidence. It may set `stage09_allowed=true` only if the strict final-holdout gate passes on the accepted Stage `08J` article-selector surface. Until then, Stage `09` remains blocked.
