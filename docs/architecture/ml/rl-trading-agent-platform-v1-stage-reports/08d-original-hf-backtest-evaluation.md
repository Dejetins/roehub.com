---
doc: rl-trading-agent-platform-v1-stage-08d-original-hf-backtest-evaluation
stage: "08D"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08D: Original HF Backtest Evaluation

Status: `accepted` for methodology execution, with quality warnings.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08D` started after checking the ledger: Stage `08C` is `accepted`,
`current_stage=08D`, and the accepted `08C` report provides the completed
`hf_original_candidate` manifest. Browser/auth QA is `N/A` for this offline
evaluation stage; the Roehub smoke Keycloak username and host-local password
source were not used.

The HF lifecycle ran end to end on Mac Studio as
`target_host_non_production_evaluation_pre_main`. After the 2026-06-24 user
review, Stage `08D` is reclassified as an execution/parity gate: the
methodology lifecycle is accepted because checkpoint loading, train-only
normalization, split use, grouped backtest mechanics, action filters, Q-cache,
parallel-session handling and scorecard artifacts completed coherently.

The weak untuned HF-demo score remains a warning, not a blocker: the filtered
grouped backtest is positive after costs, but it does not beat the simple
technical sanity baseline and only `3.24699%` of selected backtest sessions are
positive. Stage `08E` is allowed to start as Roehub-native experimental/research
training, without treating the HF-demo result as model-quality approval.

This is not `post_main_production_runtime_proof`: no production
`/opt/roehub/app` sync, service reload, browser/auth proof, registry write,
promotion, activation, exchange side effect, paper/testnet/live run, or mainnet
submit was performed.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/402763a1-fee2-428b-b201-22ce4d51aa32/pasted-text.txt` |
| Prompt sha256 | `0b83685425d0788848375e358cd2ca8b3a83bba789846eab5f47bdc65bd39e71` |
| Repo prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08d-original-hf-backtest-evaluation.md` |
| Repo prompt sha256 after 2026-06-24 gate amendment | `75480e8824393b0b7b9fc8938748aa13fde6d54492e6827a33cbe90a0dc6ef5c` |
| Previous stage gate | passed: Stage `08C` is `accepted`, `current_stage=08D`, and `hf_original_candidate` is present |
| Candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_candidate_manifest.json` |
| Candidate manifest sha256 | `189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4` |
| Candidate manifest logical hash | `c144111b5e74246589b55b1160aa869e0e6de9505f1311a12d8dadd452c50abc` |
| Evaluation checkpoint | `best.pth` by default; `final.pth` remains diagnostic only |
| Best checkpoint sha256 | `3538c77abb363f6ade74cc98113fc5a19be78b2f63c5449e675485ee8ce36e0c` |
| Train-only normalization stats hash | logical stats hash `d56be74b3f4f2779ea9dbe72302b5e918a806e23d3903810c77e43d615c2b254`; Stage `08C` file sha256 `c4e03bdb28447d789a8a097d44c73c77140348d841edfd9a4de7b752fd60f51e` |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/hf_original_evaluation.py` | - | - | Stage `08D` evaluator: loads `08C` train-only stats and `best.pth`, runs raw test diagnostics, grouped filtered backtest, baselines, scorecard/verdict manifest. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage08d_original_hf_backtest_evaluation.py` | - | - | Opt-in operator CLI for original HF `test`/`backtest` NPZ evaluation with strict hash checks by default. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_hf_original_evaluation.py` | - | - | Focused coverage for candidate loading, raw-vs-filtered separation and artifact writes. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08d_original_hf_evaluation.py` | - | - | Tiny CLI smoke over fixture HF NPZ files and sanitized manifest output. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08d-original-hf-backtest-evaluation.md` | - | - | Stage `08D` report, amended to accepted-for-methodology-execution with quality warnings. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08D` evaluator identifiers and helpers. | `compatible-change` additive Python export |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08D` `accepted`, advance `current_stage=08E`, and carry weak HF-demo score as warning-only. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index |

Outside expected paths: none in git.

Runtime artifacts (`proof_boundary=target_host_non_production_evaluation_pre_main`):

| Path | Host | Reason | sha256 / state |
|---|---|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/code_snapshot/` | Mac Studio | Non-production code snapshot used to run current local Stage `08D` code without mutating `/Users/daniildegtyarev/Projects/roehub.com` or `/opt/roehub/app`. | directory artifact |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/macstudio_smoke/stage08d_macstudio_smoke/stage08d_evaluation_manifest.json` | Mac Studio | Tiny target-host smoke over `16` test and `16` backtest sessions. | `3ce4c3908e5bec904bd18452103a2e148ce235c8d68752190d0078dacbac352c`; runtime status `blocked` under the pre-amendment quality-stop policy, now treated as warning-only for stage handoff |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/macstudio_smoke/stage08d_macstudio_smoke/scorecards.json` | Mac Studio | Tiny smoke scorecards. | `36b2eae8055cf194fb3ce064fdccdae86e3e8b7e849be671e100926a77f2ced1` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/macstudio_smoke/stage08d_macstudio_smoke/filtered_backtest_balance_curve.json` | Mac Studio | Tiny smoke filtered balance curve. | `53108ef96321c03ce9408452e672f98f5da4f127304ec4f4dd4dec0e8d866328` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/full/stage08d_hf_original_full/stage08d_evaluation_manifest.json` | Mac Studio | Full HF `test_data.npz` and `backtest_data.npz` evaluation manifest. | `61a55d1fd812f4f37f6444b23e0ea7c3ea64ff77147bd1a1c6900f9133de8fa7`; evaluation hash `69a6a967431b1347b6fa0354f3cdf962cda591893173902b0dec179970ae791f`; runtime status `blocked` under the pre-amendment quality-stop policy, now treated as warning-only for stage handoff |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/full/stage08d_hf_original_full/scorecards.json` | Mac Studio | Full test/backtest scorecards. | `16cc51b4d88be54a4093c43edf3b57c47dca667ce2761326020a91908d97d4b5` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08d_original_hf_backtest_evaluation_v1/full/stage08d_hf_original_full/filtered_backtest_balance_curve.json` | Mac Studio | Full filtered backtest balance curve. | `5cf7260ec741368d38f3e41084d83ff6887c275f532aed1bb7746c799b9ade38` |

Delivery state: `local-only` implementation plus
`target_host_non_production_evaluation_pre_main` managed evaluation. No branch,
commit, PR, deploy, production runtime sync, browser/auth proof, registry write,
promotion, activation or exchange side effect was performed by Stage `08D`.

## Методология Оценки

| Поле | Значение |
|---|---|
| Уровень глубины | `standard_analysis` |
| Тип задачи | ML/backtest evaluation gate for an offline research candidate |
| Единица анализа | HF original session keyed by `_keys_map_` ticker and signal datetime |
| Основные метрики | net PnL after costs, return %, win-rate, closed trades, reward sum, rejection counts, period/ticker stability |
| Baselines | `hold`, `no_trade`, `simple_recent_return_threshold` |
| Acceptance surface | `hf_original_candidate_filtered_backtest`; raw argmax is diagnostic only |
| Проверка утечки | Evaluation uses Stage `08C` train-only normalization stats; test/backtest splits do not recompute normalization |
| Статус вывода | `частично подтверждено`: lifecycle works and permits `08E`; untuned HF-demo score quality is warning-only and not model-quality approval |

## HF Evaluation Evidence

Raw argmax test diagnostics (`test_data.npz`, diagnostic only):

| Metric | Value |
|---|---:|
| Sessions | `3,400` |
| Decisions | `34,000` |
| Closed trades | `4,489` |
| Profitable trades | `2,374` |
| Win rate | `0.5288483` |
| Net PnL after costs | `102,922.86273306` |
| Return after costs | `0.3027143%` |
| Reward sum | `8.48132513` |

Filtered grouped backtest (`backtest_data.npz`, acceptance surface):

| Metric | Value |
|---|---:|
| Source sessions | `3,186` |
| Selected sessions after timestamp grouping / `max_parallel_sessions=2` | `2,741` |
| Skipped sessions due parallel cap | `445` |
| Signal-time groups | `2,554` |
| Decisions | `27,410` |
| Q-value cache entries / hits / misses | `27,410 / 0 / 27,410` |
| Closed trades | `215` |
| Profitable trades | `132` |
| Win rate | `0.61395349` |
| Net PnL after costs | `2,064.37744919` |
| Return after costs | `0.00753148%` |
| Reward sum | `-25.61812451` |
| Positive session ratio | `0.0324699` |

Action filter:

| Field | Value |
|---|---|
| Selection strategy | `advantage_based_filter` |
| Thresholds | long `0.012695`, short `0.009902`, close `0.001141`, ensemble sigma `0.01` |
| Rejection counts | `weak_advantage_threshold=5,227` |
| Raw argmax action counts | hold `19,141`, open_long `5,322`, open_short `2,864`, close `83` |
| Requested filtered action counts | hold `21,753`, open_long `5,315`, open_short `118`, close `224` |
| Effective action counts | hold `26,980`, open_long `163`, open_short `52`, close `215` |

Baselines on the same grouped backtest selection:

| Policy | Net PnL after costs | Return after costs | Closed trades | Win rate | Reward sum |
|---|---:|---:|---:|---:|---:|
| `hold` | `0.0` | `0.0%` | `0` | `0.0` | `-27.41` |
| `no_trade` | `0.0` | `0.0%` | `0` | `0.0` | `-27.41` |
| `simple_recent_return_threshold` | `4,508.37753925` | `0.01644793%` | `3,167` | `0.43132302` | `0.59367551` |

Period stability for filtered grouped backtest:

| Period | Sessions | Net PnL after costs | Return after costs | Closed trades | Win rate |
|---|---:|---:|---:|---:|---:|
| `2025-03` | `693` | `-2,520.79738369` | `-0.03637514%` | `52` | `0.55769231` |
| `2025-04` | `1,052` | `1,860.25563554` | `0.01768304%` | `63` | `0.57142857` |
| `2025-05` | `996` | `2,724.91919733` | `0.02735863%` | `100` | `0.67` |

Ticker stability examples:

| Group | Symbol | Sessions | Net PnL after costs | Return after costs | Closed trades | Win rate |
|---|---|---:|---:|---:|---:|---:|
| Worst | `TUTUSDT` | `31` | `-4,002.65713957` | `-1.29117972%` | `3` | `0.33333333` |
| Worst | `BMTUSDT` | `16` | `-624.80869846` | `-0.39050544%` | `1` | `0.0` |
| Worst | `ACTUSDT` | `29` | `-540.8474326` | `-0.18649911%` | `3` | `0.33333333` |
| Best | `OMUSDT` | `23` | `1,912.24309972` | `0.83141004%` | `9` | `0.77777778` |
| Best | `GPSUSDT` | `21` | `883.2045925` | `0.42057362%` | `8` | `0.875` |
| Best | `BANKUSDT` | `28` | `735.52609236` | `0.26268789%` | `7` | `0.85714286` |

## Methodology-Execution Verdict

Verdict: `accepted` with warnings.

| Check | Result |
|---|---|
| HF lifecycle end to end | passed |
| `best.pth` used by default | passed |
| Train-only normalization reused from `08C` | passed |
| Raw argmax kept diagnostic-only | passed |
| Grouped filtered backtest with timestamp grouping and `max_parallel_sessions` | passed |
| Action filter / Q-cache / parallel-session mechanics | passed |
| Scorecards and manifests complete | passed |
| Leakage or data inconsistency blocker | none observed |
| Candidate positive after costs | warning metric only: `2,064.37744919` |
| Candidate beats best sanity baseline | warning metric only: best baseline is `4,508.37753925` |
| Session-level stability | warning metric only: positive session ratio is `0.0324699` |

Blocker criteria, all clear:

- `best.pth` load failure;
- non-train-only normalization;
- incorrect test/backtest split use;
- grouped backtest lifecycle mismatch;
- broken action filter, Q-cache or parallel-session mechanics;
- incomplete scorecards or manifest;
- leakage or data inconsistency.

Warnings carried to Stage `08E` / `08F`:

- `candidate_does_not_clear_best_sanity_baseline`;
- `low_positive_session_ratio`;
- no Optuna/tuned backtest threshold search in Stage `08D`;
- demo `30/10` profile rather than stronger `90/60` or larger-profile training.

Because Stage `08D` is accepted for methodology execution, Stage `08E` may start.
This does not register, promote, activate, paper/testnet/live trade, or approve
the `hf_original_candidate` as a production-quality model.

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
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized Stage `08D` scorecards and evaluation manifests under `/opt/roehub/state/rl_trading/`. |
| Benchmark / rollout gates | `compatible-change` | Stage `08D` is accepted for methodology execution; `08E` is allowed as native experimental/research training. Weak untuned HF-demo metrics remain warning-only. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline evaluation only; no API or live inference hot path changed. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `08C` is `accepted`, `current_stage=08D`, and `hf_original_candidate` is present |
| Prompt hash | passed; `0b83685425d0788848375e358cd2ca8b3a83bba789846eab5f47bdc65bd39e71` |
| Focused ruff | passed; `uv run ruff check src/trading/contexts/rl_trading/domain/hf_original_evaluation.py scripts/rl_trading/stage08d_original_hf_backtest_evaluation.py tests/unit/contexts/rl_trading/domain/test_hf_original_evaluation.py tests/perf_smoke/contexts/rl_trading/test_stage08d_original_hf_evaluation.py` |
| Focused tests | passed; `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_hf_original_evaluation.py tests/perf_smoke/contexts/rl_trading/test_stage08d_original_hf_evaluation.py` -> `2 passed` |
| Focused pyright | passed; `0 errors` |
| Mac Studio tiny HF smoke | completed; runtime manifest emitted `blocked` under the pre-amendment quality-stop policy; manifest sha256 `3ce4c3908e5bec904bd18452103a2e148ce235c8d68752190d0078dacbac352c` |
| Mac Studio full HF evaluation/backtest | completed; runtime manifest emitted `blocked` under the pre-amendment quality-stop policy, now accepted as methodology-execution evidence with warnings; manifest sha256 `61a55d1fd812f4f37f6444b23e0ea7c3ea64ff77147bd1a1c6900f9133de8fa7` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `412 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because
the available multi-agent tool contract requires an explicit user request before
spawning subagents.

Review scope: Stage `08D` implementation/report, ledger update, file/runtime
manifest, proof-boundary/browser-auth wording, contract impact, quality gates,
warning register and `08E` handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: Release after fixes for an accepted-with-warnings Stage `08D` result.

Blockers fixed: replaced placeholder quality-gate rows with final passed
ruff/pyright/unit/docs-index evidence and, after the 2026-06-24 user review,
separated execution/parity blockers from quality warnings so the ledger can
advance to `08E`.

Local follow-up check: completed. The report and ledger explicitly record the
accepted execution/parity verdict, runtime artifact manifest, no
browser/auth/exchange side effects, warning-only score weakness, and `08E`
handoff.

Residual risks: Stage `08D` does not prove HF profitability, tuned thresholds,
author-checkpoint parity or production-grade model quality; those remain future
research/promotion concerns.

## Residual Risks

- The evaluator proves HF lifecycle mechanics, but the untuned demo-profile
  score does not prove model quality.
- The filtered backtest positive PnL is concentrated in very few sessions; the
  median selected session PnL is `0.0` and the positive session ratio is only
  `0.0324699`.
- The simple technical baseline outperforms the filtered candidate on the same
  grouped backtest selection; this is warning-only for `08E` and must be
  revisited before promotion-grade review.
- Stronger HF training (`90/60` or larger profile), multiple seeds and Optuna
  tuning remain future research hardening, not prerequisites for native training.
- This remains non-production target-host evidence from a code snapshot, not
  `post_main_production_runtime_proof`.

## 08E Handoff

Stage `08E` is allowed.

The next action is Stage `08E` Roehub-native full training on the accepted Stage
`06` dataset. The `08E` executor must carry forward the warnings above in the
adaptation diff and must not treat this HF-demo score as quality approval. No
registry, promotion, activation, paper/testnet/live execution or mainnet submit
is allowed from this `08D` result.
