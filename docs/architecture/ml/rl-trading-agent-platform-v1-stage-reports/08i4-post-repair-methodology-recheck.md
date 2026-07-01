---
doc: rl-trading-agent-platform-v1-stage-08i4-post-repair-methodology-recheck
status: accepted
stage: 08I4
updated_at: 2026-07-02
---

# Stage 08I4: post-repair methodology recheck

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08I4` выполнен как audit/report-only recheck stage после принятого `08I3`. Новое обучение, `Optuna`, registry write, paper/testnet/live/mainnet trading, exchange/provider side effects, browser/auth smoke, secret reads и `/opt/roehub/app` production mutation не выполнялись.

Доказательная граница: `target_host_non_production_forensic_pre_main`. Это Mac Studio non-production artifact/read/write под `/opt/roehub/state/rl_trading/`, а не `post_main_production_runtime_proof` и не production-runtime claim для changed code. Для `post_main_production_runtime_proof` отдельно требуются target revision on `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and then runtime smoke from the production runtime tree; `08I4` этого не выполнял и не заявляет.

## Gate

| Check | Result |
|---|---|
| Ledger before work | `current_stage=08I4`; `08I3` accepted; `08J`/`08K`/`09` pending with `Can run now=no` |
| `post_main_production_runtime_proof` | `not collected`; this stage did not require or claim production proof |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Training / `Optuna` / exchange effects | `N/A`; not run |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08i4-post-repair-methodology-recheck.md` |
| Prompt sha256 | `6631b758b40fda28ba29c945b98408fab46502a57e3aa124f86098f700ce7d03` |
| `08I2` matrix input | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i2_exhaustive_methodology_discrepancy_audit_v1/stage08i2_methodology_discrepancy_matrix.json` |
| `08I2` matrix sha256 | `abe3a0c8ba42d6b453e2166bf3a9089aba4bfc6e6e07656708829990bba81c30` |
| `08I3` repair report | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md` |
| `08I3` repair report sha256 | `fb1ed4add288927233ea7851dc35bb8eec720f344087c298552f27fc7ba1cbed` |
| `08I3` repaired trace manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/stage08i_trace_manifest.json` |
| `08I3` repaired trace manifest sha256 | `b22896c6202613032db2e4773ae4c9e55f2dd323a8db8902e5692529f77f0384` |
| `08I3` first material diff | `first_material_diff=null`; `first_material_diff.json` sha256 `43465cb4b9e732d24e06f8f0946d6f06d7c655fb521c15cf01f894d5f23aba80` |

The local workstation `/opt` view did not contain the old `08I2` matrix, but the required Mac Studio artifact path was present and hash-matched the ledger/report. The `08I4` artifact was therefore written on the same Mac Studio non-production artifact root.

## Методология анализа

| Поле | Значение |
|---|---|
| Уровень глубины | `стандартный анализ`: matrix recheck / ML methodology gate |
| Тип задачи | Row-by-row readiness classification after accepted evaluator/action/reward repair |
| Выбранная методология | Reconcile the `08I2` matrix rows against accepted `08I3` repair evidence and the plan-defined ownership split for `08J`/`08K` |
| Единица анализа | One original `08I2` methodology row |
| Основные метрики | Row disposition, remaining pre-`08J` blocker, downstream owner, `08j_allowed`, `stage09_allowed` |
| Проверка качества данных | `08I2` matrix row count `8`; Mac Studio input hashes matched; repaired `08I3` trace status `accepted` and `first_material_diff=null` |
| Риски интерпретации | `08J` may start, but this does not mean dataset/model-quality gaps are solved; rows assigned to `08J`/`08K` still block `09` |

## Runtime Artifact

| Field | Value |
|---|---|
| Artifact kind | `methodology_recheck_matrix` |
| Path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08i4_post_repair_methodology_recheck_v1/stage08i4_methodology_recheck_matrix.json` |
| sha256 | `a03da05df6aef2a59d13c28c167561afbfce230df347f01f5a5a7f61d79dc0b3` |
| Row count | `8` |
| Disposition counts | `assigned_to_08j=2`, `assigned_to_08k=5`, `closed_by_08i3=1` |
| `08j_allowed` | `true` |
| `08k_allowed` | `false` |
| `stage09_allowed` | `false` |

The runtime artifact contains the required row fields: `surface`, `08i2_status`, `08i2_severity`, `08i3_evidence`, `recheck_disposition`, `source_backed_reason`, `remaining_blocker`, `owner_next_stage`, `recheck_required`, `08j_allowed_for_row`, and `stage09_allowed_for_row`.

## Pre-08J Blocker Recheck

| Required pre-`08J` blocker | Disposition | Evidence |
|---|---|---|
| Full evaluator/backtest parity | `closed_by_08i3` | Mac Studio repaired trace has `status=accepted`, `first_material_diff=null`, manifest sha256 `b22896c6202613032db2e4773ae4c9e55f2dd323a8db8902e5692529f77f0384` |
| Rolling `open_sessions` scheduling | `closed_by_08i3` | `08I3` report records rolling active sessions until `signal_dt + agent_session_len` and trace parity |
| Shared balance and `position_fraction` sizing | `closed_by_08i3` | `08I3` report records `shared_balance * position_fraction`, sequential balance updates and trace parity |
| Action/Q mask/filter order semantics | `closed_by_08i3` | `08I3` report records raw unmasked Q-values into `FilteredBacktestPolicy`, explicit raw/masked/requested/effective action trace fields and source-compatible ensemble rejection |
| `training_reward` vs `backtest_reporting_reward` trace semantics | `closed_by_08i3` | `08I3` report records separated fields and source-compatible `backtest_reporting_reward=0.0`; legacy `reward` aliases backtest reporting |

No remaining pre-`08J` blocker was found.

## Matrix Summary

| Surface | `08I2` status / severity | Recheck disposition | Owner | `08J` row allowed | `09` row allowed | Source-backed reason |
|---|---|---|---|---|---|---|
| `session_extractor_policy` | `gap` / High | `assigned_to_08j` | `08J` | `true` | `false` | `08I3` closed evaluator prerequisites but did not materialize `article_future_10m_5pct_contrast_v1`; `08J` owns article selector, leakage/embargo/lifecycle proof and selector comparison. |
| `dataset_geometry_and_distribution` | `gap` / High | `assigned_to_08j` | `08J` | `true` | `false` | Geometry/distribution is dataset-surface work; `08J` must compare HF-original, Stage `06` current selector and article-selector distributions. |
| `past_only_signal_strength` | `gap` / High | `assigned_to_08k` | `08K` | `true` | `false` | Weak current-selector signal from `08H` must be rerun after `08J`; this is a model-quality diagnostic, not a blocker to starting `08J`. |
| `reward_sparsity_and_semantics` | `gap` / Medium | `assigned_to_08k` | `08K` | `true` | `false` | `08I3` closed trace semantics; remaining reward-sparsity analysis belongs to `08K` and must not silently redesign reward. |
| `action_q_policy_distribution` | `gap` / High | `assigned_to_08k` | `08K` | `true` | `false` | `08I3` closed mask/filter order; final raw/requested/effective action distribution and bias checks belong to `08K`. |
| `optuna_and_calibration_overfit` | `gap` / High | `assigned_to_08k` | `08K` | `true` | `false` | No tuning was run in `08I4`; calibration/final-holdout isolation and trade-sufficient `Optuna` selection must be rerun only after `08J` data exists. |
| `sanity_baselines` | `gap` / High | `assigned_to_08k` | `08K` | `true` | `false` | Baseline dominance remains a hard native research-candidate gate; `08K` must recompute baselines on the same article-selector/final-holdout surface. |
| `full_evaluator_backtest_parity` | `blocked_by_prior_gap` / High | `closed_by_08i3` | none | `true` | `false` | Prior `08I` scheduler/sizing/action/reward trace blocker is repaired; Mac Studio trace now reports `first_material_diff=null`. |

## Decision

| Field | Value |
|---|---|
| `08j_allowed` | `true` |
| `08k_allowed` | `false` |
| `stage09_allowed` | `false` |
| Next prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08j-article-session-extractor-dataset.md` |

Stage `08J` may start because all pre-`08J` evaluator/session/action/reward-reporting blockers are closed by accepted `08I3`, and every original `08I2` row now has a source-backed disposition.

Stage `08K` remains closed because `08J` is not accepted yet. Stage `09` remains closed because `08J` and `08K` accepted candidate gates are still missing and `08I4` is not allowed to open `09`.

## Business Impact

`08I4` removes the process blocker that prevented article-selector dataset work from starting. The practical effect is narrow: the team can now spend the next stage on the known dataset-methodology gap (`08J`) instead of rerunning training or tuning against a partially repaired evaluator. This reduces the risk of another expensive research loop where one repaired gap hides a different already-known methodology mismatch.

This is not a business or production trading capability change. There is still no accepted Roehub-native research candidate, no model registry opening, no paper/testnet/live execution, and no user-facing product behavior change.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API/UI route or DTO changed. |
| Port contract | `none` | No Python port/interface changed. |
| DTO schema | `none` | No persisted or public DTO changed. |
| Persisted schema | `none` | No DB migration or persisted application schema changed. |
| Config schema | `none` | No runtime config/default changed. |
| Request hash / cache key / persistence identity | `none` | No identity/hash semantics changed. |
| Benchmark / rollout gate | `compatible-change` | Ledger gate advances from `08I4` to `08J`; `stage09_allowed=false` remains explicit. |
| Browser-visible behavior | `none` | Browser/auth scope is `N/A`. |
| Runtime artifacts | `compatible-change` | Added sanitized non-production matrix artifact under `/opt/roehub/state/rl_trading/`. |

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
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md` | created | Stage `08I4` report and row-disposition summary. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08I4` accepted, advance `current_stage` to `08J`, keep `08K`/`09` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative with accepted `08I4` and `08J` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding the stage report. | `compatible-change` docs index |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i4_post_repair_methodology_recheck_v1/stage08i4_methodology_recheck_matrix.json` | created outside git | Durable sanitized runtime recheck matrix. | `compatible-change` non-production artifact |

## Quality Gates

| Gate | Result |
|---|---|
| `uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading` | `N/A`; no Python helper/test changes |
| `uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading` | `N/A`; no Python helper/test changes |
| `uv run pytest -q tests/unit/scripts/rl_trading` | `N/A`; no Python helper/test changes |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after regenerating `docs/architecture/README.md`; final result `OK` |

## Residual Risks

- Rows assigned to `08J`/`08K` are still open downstream work. They do not block starting `08J`, but they still block `08K`/`09` according to their ownership.
- `08I4` did not train, tune, register, promote, activate, paper/testnet/live trade or claim candidate quality.
- The proof boundary is `target_host_non_production_forensic_pre_main`. It is not `post_main_production_runtime_proof`; production proof would require `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and runtime smoke after that.

## Cold-Head Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `08I4` report, ledger update, plan sync, docs index, runtime matrix artifact schema/hash, proof-boundary wording, browser/auth redaction, downstream stage gates and file manifest.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: added explicit `post_main_production_runtime_proof` non-claim and requirements; added business impact and conditional operational `N/A` coverage; advanced ledger to `current_stage=08J`; recorded `08I4` change-log entry; regenerated and checked `docs/architecture/README.md`.
Local follow-up check: completed; required report literals, ledger stage state and Mac Studio matrix schema/booleans passed; docs index and diff whitespace gates passed.
Residual risks: downstream `08J`/`08K` rows remain open; `stage09_allowed=false`; no production runtime proof or candidate-quality claim was collected.

## Handoff

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/08j-article-session-extractor-dataset.md`.

`08J` must close `session_extractor_policy` and `dataset_geometry_and_distribution` by materializing `article_future_10m_5pct_contrast_v1`, rerunning leakage/embargo/lifecycle proof, and comparing HF-original, Stage `06` current selector and article-selector distributions.

`08K` must later close `past_only_signal_strength`, `reward_sparsity_and_semantics`, `action_q_policy_distribution`, `optuna_and_calibration_overfit`, and `sanity_baselines` before any `stage09_allowed=true` decision can be considered.
