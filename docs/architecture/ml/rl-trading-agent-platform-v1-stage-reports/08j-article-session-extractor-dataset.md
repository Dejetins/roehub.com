---
doc: rl-trading-agent-platform-v1-stage-08j-article-session-extractor-dataset
status: accepted
stage: 08J
updated_at: 2026-07-02
---

# Stage 08J: article session extractor dataset

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08J` выполнен как dataset/materialization stage для отдельной Roehub-native article-selector ветки. Исторические Stage `06` artifacts не перезаписывались. Новое обучение, `Optuna`, registry write, paper/testnet/live/mainnet trading, exchange/provider side effects, browser/auth smoke, secret reads и `/opt/roehub/app` production mutation не выполнялись.

Доказательная граница: `target_host_non_production_dataset_pre_main`. Это Mac Studio non-production dataset artifact/read/write под `/opt/roehub/state/rl_trading/`, а не `post_main_production_runtime_proof` и не production-runtime claim для changed code. Для `post_main_production_runtime_proof` отдельно требуются target revision on `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and then runtime smoke from the production runtime tree; `08J` этого не выполнял и не заявляет.

## Gate

| Check | Result |
|---|---|
| Ledger before work | `current_stage=08J`; `08I3` accepted; `08I4` accepted with `08j_allowed=true`, `08k_allowed=false`, `stage09_allowed=false` |
| `08I4` row ownership | `session_extractor_policy` and `dataset_geometry_and_distribution` assigned to `08J`; model-quality rows remain assigned to `08K` |
| Unresolved material evaluator/session/action/reward blocker | none before `08J`; closed by accepted `08I3` and accepted `08I4` recheck |
| `post_main_production_runtime_proof` | `not collected`; this stage did not require or claim production proof |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Training / `Optuna` / exchange effects | `N/A`; not run |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08j-article-session-extractor-dataset.md` |
| Prompt sha256 | `77dda8c649fcc29c654e18005b12807a1ca7d074d10cd1a5f4a72bad8c278023` |

## Методология Анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `integration`: selector implementation + full target-host materialization + distribution comparison |
| Тип задачи | Dataset extraction, leakage/lifecycle proof, selector distribution comparison |
| Выбранная методология | Keep Stage `06` as historical accepted selector, add article-style policy beside it, then compare HF-original, Stage `06` current selector and article selector on the same reporting surfaces |
| Единица анализа | One materialized session keyed by `symbol`, split and `signal_ts_open` |
| Основные метрики | Session count, split artifacts, long/short oracle label counts, oracle best return, pre-signal volatility/range, symbol/month coverage, past-only supervised sanity |
| Проверка качества данных | Split boundaries, lifecycle availability, first-kline/gap handling, cross-split leakage, embargo, deterministic hashes, rejected-window reasons |
| Риски интерпретации | Article selector closes dataset-surface rows only. It does not prove model quality, baseline beating, action stability, or `stage09_allowed=true`; those remain `08K` work. |

## Selector Contract

| Field | Value |
|---|---|
| Policy id | `article_future_10m_5pct_contrast_v1` |
| Event formula | `abs(close[event_end_idx - 1] / open[event_end_idx - 10] - 1) >= 0.05` |
| Event time semantics | `event_end_t = signal_ts_open` |
| Contrast rule | Previous `90m` context before the selected event window must not already contain a non-overlapping `10m` impulse with absolute move `>= 0.05` |
| Pre-window | `pre_window=[signal_ts_open-90m, signal_ts_open)` |
| Post-window | `post_window=[signal_ts_open, signal_ts_open+60m)` |
| Materialized session length | `150` one-minute rows |
| Later source/demo profile | Stage `08K` can train/evaluate with `agent_history_len=30`, `agent_session_len=10` on the materialized `90/60` session tensor |
| Stage `06` artifact behavior | No overwrite or mutation; article selector writes a separate dataset root |

Accepted deviation from the external article/repo: the upstream repo consumes pre-built HF sessions and does not ship a canonical raw-minute materializer for Roehub's current Binance Futures universe. Roehub therefore implements the article-style event policy against accepted Stage `05` raw feature slabs and Stage `04C` lifecycle/source-window evidence, while preserving Roehub split boundaries, lifecycle exclusions, gap handling, overlap policy and embargo.

## Runtime Artifacts

| Artifact | Value |
|---|---|
| Article dataset root | `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1` |
| Article manifest | `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json` |
| Article manifest sha256 | `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| Deterministic rebuild hash | `6ceff2407430587fe8c3ff8f618999efa5d2462a683aca620c7a6fe9c9c4da55` |
| Leakage report | `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_leakage_report.json` |
| Distribution comparison | `/opt/roehub/state/rl_trading/evaluation_runs/stage08j_article_session_extractor_dataset_v1/stage08j_selector_comparison_83aa89d9885092fbec72/stage08j_selector_distribution_comparison.json` |
| Distribution comparison sha256 | `3f3771367d9e1794c7a6bf83d29a445889bbe30e1c9d81d2c60aceced1a4963b` |
| Distribution comparison internal hash | `fea434758bed5fa59332e67b2d29e1acc1f3788fc95e8ee5dfff0ac3d14558f2` |
| Status | `accepted` |
| Proof boundary | `target_host_non_production_dataset_pre_main` |

Mac Studio command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run python scripts/rl_trading/stage08j_article_session_extractor_dataset.py --all-symbols --dataset-version hf_period_rebuild_current_trading"'
```

Mac Studio result:

```json
{"article_manifest_path": "/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json", "article_manifest_sha256": "fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a", "comparison_path": "/opt/roehub/state/rl_trading/evaluation_runs/stage08j_article_session_extractor_dataset_v1/stage08j_selector_comparison_83aa89d9885092fbec72/stage08j_selector_distribution_comparison.json", "comparison_sha256": "3f3771367d9e1794c7a6bf83d29a445889bbe30e1c9d81d2c60aceced1a4963b", "status": "accepted"}
```

## Materialization Summary

| Metric | Value |
|---|---:|
| Total sessions | `34252` |
| Split artifacts | `1079` |
| Symbols | `352` |
| Train sessions | `24179` |
| Validation sessions | `1749` |
| Test sessions | `4162` |
| Backtest sessions | `4162` |
| Train artifacts | `218` |
| Validation artifacts | `239` |
| Test artifacts | `299` |
| Backtest artifacts | `323` |

Leakage/lifecycle evidence:

| Check | Value |
|---|---:|
| Leakage status | `accepted` |
| Cross-split overlap violations | `0` |
| Embargo violations | `0` |
| Lookahead violations | `0` |
| Lifecycle violations | `0` |
| Within-split overlap pairs | `1901` |
| Rejected windows | `353` |
| Rejected `lifecycle_no_signal_overlap_for_split` | `304` |
| Rejected `no_article_future_impulse_candidates` | `49` |
| Gap status | `accepted=1079` |

Within-split overlap is allowed by the Stage `06`/`08J` policy. Cross-split leakage and embargo violations are blockers and were not observed.

## Selector Distribution Comparison

| Selector | Sessions | Symbols | Train labels | Validation labels | Test labels | Backtest labels |
|---|---:|---:|---|---|---|---|
| `hf_original` | `32049` | `478` | `hold=1,long=13514,short=10571` | `hold=1,long=779,short=597` | `hold=0,long=2086,short=1314` | `hold=0,long=1581,short=1605` |
| `stage06_current_selector` | `50707` | `358` | `hold=56,long=7133,short=6192` | `hold=48,long=4821,short=5380` | `hold=44,long=6004,short=6298` | `hold=56,long=7508,short=7167` |
| `article_future_10m_5pct_contrast_v1` | `34252` | `352` | `hold=0,long=13533,short=10646` | `hold=0,long=1026,short=723` | `hold=0,long=2431,short=1731` | `hold=0,long=2068,short=2094` |

| Selector / split | Mean oracle best return | Positive ratio | Ridge balanced accuracy | Recent baseline balanced accuracy |
|---|---:|---:|---:|---:|
| `hf_original` train | `0.0425022942` | `0.9999584821` | `0.4124646211` | `0.5862025192` |
| `hf_original` validation | `0.0327465371` | `0.9992737836` | `0.4541012006` | `0.2019726360` |
| `hf_original` test | `0.0415684845` | `1.0` | `0.6385915526` | `0.3552420938` |
| `hf_original` backtest | `0.0348674254` | `1.0` | `0.5728238959` | `0.4274078278` |
| `stage06_current_selector` train | `0.0308896468` | `0.9958149615` | `0.3731899526` | `0.5798877785` |
| `stage06_current_selector` validation | `0.0152397328` | `0.9953166163` | `0.3522408160` | `0.5924252554` |
| `stage06_current_selector` test | `0.0214586342` | `0.9964360927` | `0.3399569124` | `0.6607522581` |
| `stage06_current_selector` backtest | `0.0157745443` | `0.9961984930` | `0.3428400835` | `0.5866836232` |
| `article_future_10m_5pct_contrast_v1` train | `0.0386532669` | `1.0` | `0.6163616317` | `0.3823521532` |
| `article_future_10m_5pct_contrast_v1` validation | `0.0319942445` | `1.0` | `0.6913364555` | `0.3072339100` |
| `article_future_10m_5pct_contrast_v1` test | `0.0386376062` | `1.0` | `0.6241556384` | `0.3748765762` |
| `article_future_10m_5pct_contrast_v1` backtest | `0.0331027960` | `1.0` | `0.5841923780` | `0.4180196620` |

The full comparison JSON records `range_and_volatility` for every selector and split under `split_diagnostics.<split>.range_and_volatility`, including `pre_signal_realized_volatility` and `pre_signal_range_ratio` distribution payloads. The accepted command output relayed the article-selector train excerpt: pre-signal volatility mean `0.0051110057`, pre-signal range mean `0.1023159985`. The complete exact range/volatility tables are in the hash-pinned comparison artifact rather than duplicated in this report.

The article selector produces fewer sessions than the Stage `06` current selector, more sessions than HF-original, close test/train oracle-return geometry to HF-original, and materially stronger past-only ridge sanity than Stage `06` on every split. This closes the `session_extractor_policy` and `dataset_geometry_and_distribution` ownership rows for `08J`; it does not close `08K` model-quality rows.

## Decision

| Field | Value |
|---|---|
| `08J` status | `accepted` |
| `session_extractor_policy` | closed for `08J` by implemented `article_future_10m_5pct_contrast_v1` and target-host materialization |
| `dataset_geometry_and_distribution` | closed for `08J` by HF-original vs Stage `06` vs article-selector comparison |
| `08k_allowed` | `true` |
| `stage09_allowed` | `false` |
| Next prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08k-article-demo-profile-training-evaluation.md` |

Stage `08K` may start because `08J` accepted the article-selector dataset variant and updated the handoff. Stage `09` remains closed because there is still no accepted `08K` or later corrective research candidate with `stage09_allowed=true`.

## Business Impact

`08J` replaces a methodology guess with a durable dataset artifact: Roehub can now test the article/demo `30/10` path against a Roehub-native article-style selector rather than rerunning model training on the previously failed Stage `06` high-volatility selector. The expected business effect is reduced research waste and clearer evidence for the next candidate-quality gate.

This is not a production trading capability change. No model was trained, registered, promoted or activated; no user-facing API/UI changed; no paper/testnet/live execution path changed.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API/UI route or DTO changed. |
| Port contract | `none` | No application port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration/table/storage schema changed. |
| Config schema/defaults | `none` | No production config/default changed. |
| Request hash / cache key / persistence identity | `compatible-change` | Adds offline article-selector policy id and dataset artifact identity; no production request/cache identity changed. |
| Runtime artifacts | `compatible-change` | Adds sanitized non-production dataset/comparison artifacts under `/opt/roehub/state/rl_trading/`. |
| Benchmark / rollout gate | `compatible-change` | Ledger advances from `08J` to `08K`; `stage09_allowed=false` remains explicit. |
| Browser-visible behavior | `none` | Browser/auth scope is `N/A`. |
| Performance hot path | `none` | Offline materialization script only; no API/live inference hot path changed. |

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
| `src/trading/contexts/rl_trading/domain/sessionized_dataset.py` | modified | Add article selector policy, event/contrast scoring, article metadata payload fields and Stage `08J` manifest stage support. | `compatible-change` additive offline dataset policy |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modified | Export article selector constants/functions. | `compatible-change` additive Python export |
| `scripts/rl_trading/stage08j_article_session_extractor_dataset.py` | created | Opt-in Stage `08J` materialization and selector-comparison CLI. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py` | modified | Cover article event semantics and contrast blocking. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage08j_article_session_extractor_dataset.py` | created | Cover report payload for oracle labels, symbol/month counts, volatility/range. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md` | created | Stage `08J` accepted report. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08J` accepted, advance `current_stage` to `08K`, keep `09` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative with accepted `08J` and `08K` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding the stage report. | `compatible-change` docs index |

Runtime artifact manifest:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json` | created outside git; accepted article dataset manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_leakage_report.json` | created outside git; leakage/lifecycle/gap report with `status=accepted` |
| `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/{train,validation,test,backtest}/` | created outside git; `1079` split artifact files |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08j_article_session_extractor_dataset_v1/stage08j_selector_comparison_83aa89d9885092fbec72/stage08j_selector_distribution_comparison.json` | created outside git; selector comparison sha256 `3f3771367d9e1794c7a6bf83d29a445889bbe30e1c9d81d2c60aceced1a4963b` |

## Quality Gates

| Gate | Result |
|---|---|
| Focused selector/script tests | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py tests/unit/scripts/rl_trading/test_stage08j_article_session_extractor_dataset.py` -> `10 passed` |
| Focused ruff gate | passed: `uv run ruff check src/trading/contexts/rl_trading/domain/sessionized_dataset.py scripts/rl_trading/stage08j_article_session_extractor_dataset.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py tests/unit/scripts/rl_trading/test_stage08j_article_session_extractor_dataset.py` |
| Focused pyright gate | passed: `uv run pyright src/trading/contexts/rl_trading/domain/sessionized_dataset.py scripts/rl_trading/stage08j_article_session_extractor_dataset.py tests/unit/contexts/rl_trading/domain/test_sessionized_dataset.py tests/unit/scripts/rl_trading/test_stage08j_article_session_extractor_dataset.py` -> `0 errors` |
| Prompt ruff gate | passed: `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` |
| Prompt pyright gate | passed: `uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `0 errors` |
| Prompt pytest gate | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `103 passed` |
| Mac Studio bounded smoke | passed with `status=accepted`; bounded vectorized backtest smoke manifest sha256 `d60e830aa691f97f294513449fe7d78539646b1eb02260b6c076c3c3ba9b83a3` |
| Mac Studio full materialization | passed with `status=accepted`; article manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| Docs index | passed after regenerating `docs/architecture/README.md`; final result `OK` |

## Residual Risks

- `08J` closes dataset selection and distribution rows only. It does not prove that the model will beat baselines or avoid action/Q bias.
- `08K` must still run HF-original control plus Roehub-native article-selector training/evaluation, `Optuna`, untouched final holdout, strict baseline-beating and stability gates.
- `stage09_allowed=false` remains explicit. Registry, activation, paper/testnet/live and mainnet paths remain blocked.
- The proof boundary is `target_host_non_production_dataset_pre_main`. It is not `post_main_production_runtime_proof`; production proof would require `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and runtime smoke after that.

## Cold-Head Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `08J` implementation, selector formula, Mac Studio materialization evidence, distribution comparison, leakage/lifecycle report, ledger update, plan sync, docs index, proof-boundary wording, browser/auth redaction and downstream gates.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: optimized contrast scoring to avoid impractical full-scale runtime; added bounded vectorized smoke before full materialization; preserved Stage `06` artifact root; recorded exact selector formula, prompt hash, artifact paths/hashes, `08k_allowed=true`, and `stage09_allowed=false`.
Local follow-up check: completed; required report literals, ledger stage state, Mac Studio artifact hashes, code gates and docs index passed.
Residual risks: `08K` model-quality rows remain open; no production runtime proof, candidate-quality claim, model training or `Optuna` was collected in `08J`.

## Handoff

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/08k-article-demo-profile-training-evaluation.md`.

`08K` must close `past_only_signal_strength`, `reward_sparsity_and_semantics`, `action_q_policy_distribution`, `optuna_and_calibration_overfit`, and `sanity_baselines` using the accepted article-selector dataset and HF-original control before any `stage09_allowed=true` decision can be considered.
