---
doc: rl-trading-agent-platform-v1-stage-08m-supervised-warm-start-candidate-scorecard
status: accepted
stage: 08M
updated_at: 2026-07-02
---

# Stage 08M: supervised warm-start candidate scorecard

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08M` преобразовал accepted `08L` research path `supervised_oracle_label_warm_start_contextual_bandit` в bounded final-holdout candidate scorecard на accepted Stage `08J` article-selector surface. Strict native final gate, тот же gate family, который заблокировал `08K`, прошел без blockers. Stage `09` теперь может стартовать, но `08M` не пишет registry, не активирует модель и не открывает paper/testnet/live/mainnet execution.

Validation evidence type: `target_host_non_production_candidate_scorecard_pre_main`. Prompt proof-boundary label: `target_host_readiness_pre_main`. In this report that label means only Mac Studio host-readiness plus non-production scorecard artifact execution under `/opt/roehub/state/rl_trading/`; it is not `read_only_existing_runtime_smoke`, not `post_main_production_runtime_proof`, and not a production-runtime claim for changed code. For `post_main_production_runtime_proof`, a later stage must first have target revision on `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and then the relevant smoke from that production runtime tree; `08M` did none of those steps and claims none of them.

Proof-boundary separation:

| Boundary label | `08M` status | Meaning |
|---|---|---|
| `target_host_readiness_pre_main` | collected | Host-readiness plus non-production candidate scorecard artifact creation under `/opt/roehub/state/rl_trading/`; no production runtime tree was used. |
| `read_only_existing_runtime_smoke` | not collected | Existing `/opt/roehub/app` production runtime was not smoked or used as evidence for this stage. |
| `post_main_production_runtime_proof` | not collected | No changed-code production runtime proof is claimed. This would require `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and production runtime smoke after that. |

Обязательные markers: `08M`, `reward_research_not_contract_replacement`, `supervised_oracle_label_warm_start_contextual_bandit`.

## Gate

| Check | Result |
|---|---|
| Ledger before work | `current_stage=08M`; `08I3`, `08I4`, `08J`, `08L` accepted; `08K` blocked; `09` blocked |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08m-supervised-warm-start-candidate-scorecard.md` |
| Prompt sha256 | `01dcaebae004fd566dac3082ff450d435f0afefc5381523071535268b41e469b` |
| `08L` summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json` |
| `08L` summary file sha256 | `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1` |
| `08L` summary payload hash | `59bdb534baa97bd172266edb4405774ecc12e2005900386ce4d4bae479f28216` |
| Article dataset | Stage `08J` manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no orders, no registry promotion, no paper/testnet/live execution or mainnet submit |
| Overall verdict | `accepted`; `stage09_allowed=true`; strict gate blockers `[]` |

## Методология Анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `research_candidate_scorecard` |
| Тип задачи | Candidate scorecard after accepted proxy research |
| Выбранная методология | Closed-form ridge classifier on Stage `08J` train split, evaluated on test and untouched backtest/final split; no DQN retraining and no `Optuna` in `08M` |
| Простое объяснение метода | Из past-only признаков обучается supervised warm-start classifier на oracle labels; затем его long/short decisions считаются как final-holdout contextual-bandit scorecard |
| Объяснение на языке бизнеса | Проверка отвечает, можно ли превратить сильный proxy signal из `08L` в отдельный research candidate, который проходит те же baseline/stability/action gates, не меняя reward contract |
| Единица анализа | One `article_future_10m_5pct_contrast_v1` session |
| Основные метрики | Final PnL after costs, best sanity baseline, closed trades, month/ticker/volatility dominance, positive group ratios, action balance |
| Проверка качества данных | Stage `08J` manifest hash matched; `08L` summary hash matched; train/test/backtest split artifacts loaded fully |
| Риски интерпретации | Candidate accepted only as research input for Stage `09`; registry state, activation, backup/restore, paper/testnet/live remain downstream gates |

## Bounded Candidate Matrix

| Implementation path | Dataset branch | Profile | Max runtime | Metrics | Stop conditions | Artifact |
|---|---|---|---|---|---|---|
| Closed-form ridge classifier warm-start over past-window features; no DQN retraining, no `Optuna`, no registry write | `roehub_native_article_selector_30_10` | `30/10` | Bounded NumPy fit plus one final holdout scorecard pass | balanced accuracy, final PnL, best baseline delta, closed trades, monthly/ticker/volatility dominance, action balance | `08L` summary hash mismatch; article dataset missing; strict native final gate fails | `stage08m_supervised_warm_start_candidate_scorecard_summary.json` |

## Runtime Artifacts

| Artifact | Value |
|---|---|
| Run dir | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55` |
| Summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json` |
| Summary file sha256 | `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7` |
| Summary payload hash | `4ac03f25ea78310568c4a59a12caa6b6215440056a5c995399b98b6c8205bdca` |
| Candidate manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` |
| Candidate manifest sha256 | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| Candidate id | `stage08m_a3823cbd01143878_fd7c614b` |
| Model state hash | `a3823cbd011438787e07817ad68a17b9a089a2be7798352639349615a3eae839` |
| Status | `accepted` |
| `stage09_allowed` | `true` |

Non-production artifact command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && PYTHONUNBUFFERED=1 uv run python scripts/rl_trading/stage08m_supervised_warm_start_candidate_scorecard.py --generated-at-utc 2026-07-02T18:00:00Z"'
```

Non-production artifact result:

```json
{"candidate_id": "stage08m_a3823cbd01143878_fd7c614b", "candidate_manifest_path": "/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json", "candidate_manifest_sha256": "9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c", "run_id": "stage08m_supervised_warm_start_fe2fe3c5257fd9992c55", "stage09_allowed": true, "status": "accepted", "strict_gate_blockers": [], "summary_sha256": "ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7"}
```

## Dataset And Candidate Lineage

| Split | Selected sessions | Split artifacts | Summary hash |
|---|---:|---:|---|
| `train` | `24179` | `218/218` | `5df09e09131a56aafa4ef73547d139fbd9a36210cf9fcc596d513d672cc8c5d7` |
| `test` | `4162` | `299/299` | `26d4acc9ea94f68dc879cd1f3beb9370da10c3d1016c8fe8d2405d7356b006ef` |
| `backtest` | `4162` | `323/323` | `da5633f9805d9d892bdc2657453672ba424512b7e160cff5fa9fa6173360866f` |

All three splits use `selector_id=article_future_10m_5pct_contrast_v1`, `dataset_version=hf_period_rebuild_current_trading`, `manifest_stage=08J`, and `allow_fixture_hashes=false`.

## Final Holdout Scorecards

| Policy / reference | Kind | Final PnL after costs | Closed trades | Return pct after costs |
|---|---|---:|---:|---:|
| `hold_no_trade` | baseline | `0.0` | `0` | `0.0` |
| `deterministic_random_contextual_bandit` | baseline | `-9100.9261598118` | `2758` | `-0.910092616` |
| `simple_recent_return_threshold_contextual_bandit` | baseline | `-53438.2414711871` | `4157` | `-5.3438241471` |
| `supervised_oracle_label_warm_start_contextual_bandit` | candidate | `23018.4187849668` | `4162` | `2.3018418785` |
| `oracle_label_upper_bound_not_candidate` | diagnostic | `425716.5220601573` | `4162` | `42.571652206` |
| Stage `08K` native DQN final scorecard | blocked reference | `12502.65333026` | `316` | `125.0265333` |

The Stage `08K` native reference had positive PnL and beat baseline but stayed blocked because its strict gate had blockers `single_group_dominates_final_result` and `ticker_stability_obviously_broken`.

## Strict Native Gate

| Gate | Value | Result |
|---|---:|---|
| Candidate final PnL after costs | `23018.4187849668` | passed |
| Best sanity baseline PnL after costs | `0.0` | passed |
| Closed trades | `4162` | passed; minimum `100` |
| Monthly dominance | `0.4996531449664722` (`2025-05`, `3` groups) | passed; limit `0.8` |
| Ticker dominance | `0.023532616397542595` (`ACTUSDT`, `323` groups) | passed; limit `0.8` |
| Volatility-bucket dominance | `0.786834239482547` (`high`, `3` groups) | passed; limit `0.8` |
| Monthly positive group ratio | `0.6666666666666666` | passed; minimum `0.25` |
| Ticker positive group ratio | `0.653250773993808` | passed; minimum `0.25` |
| Open-side dominance | `0.6001922152811149` (`open_long=1664`, `open_short=2498`) | passed; limit `0.95` |

Strict gate status: `accepted_for_research`; blockers: `[]`; `stage09_allowed=true`.

## Decision

| Field | Value |
|---|---|
| `08M` status | `accepted` |
| Candidate | `stage08m_a3823cbd01143878_fd7c614b` |
| Candidate policy | `supervised_oracle_label_warm_start_contextual_bandit` |
| Reward/action contract | Stage `02C` realized-PnL reward preserved; `reward_research_not_contract_replacement` |
| `stage09_allowed` | `true` |
| Next prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/09-model-registry-activation.md` |

## Business Impact

`08M` changes the research handoff from "proxy path justified" to "bounded research candidate accepted for registry-stage evaluation". This does not create a tradable product capability yet. It only allows Stage `09` to build registry state, artifact hash validation, lifecycle gates and activation invariants around the accepted candidate identity.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API/UI route or DTO changed. |
| Port contract | `none` | No application port/interface changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration or persisted application schema changed. |
| Config schema/defaults | `none` | No production runtime config/default changed. |
| Request hash / cache key / persistence identity | `none` | No production request/cache identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side-effect and unknown-state semantics | `none` | No exchange/provider submit or money-moving call occurred. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized `08M` candidate scorecard artifacts and docs/report. |
| Benchmark / rollout gates | `compatible-change` | Advances ledger from `08M` to `09`; `stage09_allowed=true` for this candidate only. |
| Browser-visible behavior | `none` | Browser/auth scope is `N/A`. |
| Performance hot path | `none` | Offline research scorecard script only; no API/live inference hot path changed. |

## Conditional Operational Coverage

| Surface | Coverage |
|---|---|
| Service calls | `N/A`; no Roehub API, worker, queue, Redis, ClickHouse write, external provider, exchange SDK or browser service call was added or changed. |
| Timeout / retry / idempotency | `N/A`; no retry loop or side-effecting operation was introduced. |
| Unknown external side-effect state | `N/A`; no exchange/provider submit or money-moving call occurred. |
| Secrets and redaction | No secrets, tokens, cookies, credentials, raw provider payloads, account identifiers, HMACs or API keys were read or written. |
| Alerts / monitoring / runbook | `N/A`; no production runtime, scheduler, alert route, notification provider, incident workflow or runbook action changed. |
| Browser/auth | `N/A`; browser-visible behavior and authenticated UI were out of scope. |
| Mac Studio path contract | ML artifacts are under `/opt/roehub/state/rl_trading/`; no git command or smoke was run under `/opt/roehub/app`. |

## File Manifest

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `scripts/rl_trading/stage08m_supervised_warm_start_candidate_scorecard.py` | created | Opt-in bounded `08M` supervised warm-start candidate-scorecard CLI. | `compatible-change` additive offline research CLI |
| `tests/unit/scripts/rl_trading/test_stage08m_supervised_warm_start_candidate_scorecard.py` | created | Focused coverage for strict gate reuse, `08L` summary validation and candidate-manifest safety markers. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md` | created | Stage `08M` accepted report and evidence handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08M` accepted, advance `current_stage` to `09`, and record `stage09_allowed=true`. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative with accepted `08M` and Stage `09` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding Stage `08M` report. | `compatible-change` docs index |

Runtime artifact manifest:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json` | created outside git; accepted summary sha256 `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` | created outside git; candidate manifest sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |

## Quality Gates

| Gate | Result |
|---|---|
| Focused local pytest | passed: `uv run pytest -q tests/unit/scripts/rl_trading/test_stage08m_supervised_warm_start_candidate_scorecard.py` -> `3 passed` |
| Focused local ruff | passed: `uv run ruff check scripts/rl_trading/stage08m_supervised_warm_start_candidate_scorecard.py tests/unit/scripts/rl_trading/test_stage08m_supervised_warm_start_candidate_scorecard.py` |
| Focused local pyright | passed: `uv run pyright scripts/rl_trading/stage08m_supervised_warm_start_candidate_scorecard.py tests/unit/scripts/rl_trading/test_stage08m_supervised_warm_start_candidate_scorecard.py` -> `0 errors` |
| Remote Mac Studio focused pytest | passed: `3 passed` |
| Remote Mac Studio focused ruff | passed |
| Mac Studio non-production artifact run | passed with `status=accepted`; summary sha256 `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7`; `stage09_allowed=true`; no `/opt/roehub/app` production runtime smoke |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading` -> `120 passed in 4.09s` |
| Docs index | passed after regeneration: `uv run python -m tools.docs.generate_docs_index --check` |
| Whitespace diff | passed: `git diff --check` |

## Residual Risks

- `08M` accepts a supervised/contextual-bandit research candidate, not a production activation. Stage `09` must still implement registry state-machine invariants, artifact hash validation and lifecycle controls.
- Volatility-bucket dominance passes narrowly at `0.786834239482547` versus limit `0.8`; Stage `09` and later calibration stages must keep this concentration risk visible.
- This is pre-main non-production Mac Studio artifact evidence, not `post_main_production_runtime_proof`; post-main proof still requires `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and then production runtime smoke.
- RL paper/testnet/live stages `15`/`16` remain blocked by the classic strategy producer dependency.

## Cold-Head Review

Mode: cold self-review fallback. Independent review tooling was unavailable in this environment.

Verdict: accepted. Review checked the owned diff, Stage `08M` report, ledger `current_stage=09`, plan handoff, contract-impact classifications, prompt-level gates, docs-index state and proof-boundary language. No blocking issue remains. Residual risks are the downstream Stage `09` registry/activation gates, narrow volatility-bucket margin and missing `post_main_production_runtime_proof`, all recorded above.

## Handoff

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/09-model-registry-activation.md`.

Stage `09` may consume only candidate id `stage08m_a3823cbd01143878_fd7c614b` and its manifest sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` as the accepted research candidate input. Stage `09` must not treat `08M` as registry write, activation, paper/testnet/live readiness or production runtime proof.
