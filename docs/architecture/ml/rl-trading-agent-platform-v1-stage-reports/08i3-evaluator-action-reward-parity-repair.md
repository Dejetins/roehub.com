---
doc: rl-trading-agent-platform-v1-stage-08i3-evaluator-action-reward-parity-repair
stage: "08I3"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-07-02"
---

# Stage 08I3: evaluator/action/reward parity repair

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08I3` выполнен как repair stage для pre-`08J` evaluator/action/reward-reporting blockers. Новое обучение, `Optuna`, registry write, paper/testnet/live/mainnet trading, exchange/provider side effects, browser/auth smoke, secret reads и `/opt/roehub/app` production mutation не выполнялись.

Доказательная граница: `target_host_non_production_forensic_pre_main`. Это Mac Studio non-production forensic trace по runtime artifacts под `/opt/roehub/state/rl_trading/`, а не production-runtime claim для changed code.

## Gate

| Gate | Result |
|---|---|
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08i3-evaluator-action-reward-parity-repair.md` |
| Prompt sha256 | `775dcf90bf99637045e25030d2012d6eaba09d438d511b959b8e52d3a77df6cf` |
| Ledger before work | `current_stage=08I3`; `08I` blocked; `08I2` blocked; `08J`/`08K`/`09` pending/no |
| Browser/auth | `N/A`; `smoke_e2e_keycloak` and `ROEHUB_SMOKE_E2E_PASSWORD` were not used |
| Upstream commit | `YuriyKolesnikov/rl-trading-binance@f71130903f8237351164f4b875494185465bf1ea` |
| Verdict | `accepted`; repaired trace has `first_material_diff=null` |
| Next stage | `08I4` may run; `08J`, `08K` and `09` remain closed |

## Upstream Source Hashes

Source files were read from the pinned upstream commit without vendoring or checkout.

| File | sha256 |
|---|---|
| `agent.py` | `49ef8faaba845eb31207704fae23a73a9f784af0a4b6aef9323fd8be769e2fab` |
| `backtest_engine.py` | `d05e426fdad3acb24df4c74fce17536d584e56a0b9e528160c5cb9762e179892` |
| `config.py` | `65bfc4b8fa0722defe75ecf38dbb0ce92c53d5edc2e96b8b5fe0d849fc6219d6` |
| `configs/alpha.py` | `c8f0348379ed4deaf7dc306bbab039203e22e4039321ab294caedd2f5f698f9e` |
| `optimize_cfg.py` | `f6b2c542958cdce4c1cec6096cdae619304f67740b79098e136bf8dbfbe646dd` |
| `trading_environment.py` | `c38154ee416f1fb3de59c2f7085092d0237216c7854e70ba89863d9676920c8c` |

## Repaired Semantics

| Surface | Repair |
|---|---|
| `rolling_open_sessions` | `hf_original_evaluation._grouped_backtest_indices()` now keeps rolling active sessions until `signal_dt + agent_session_len`, rather than capping only exact `signal_time` groups. It records selected/skipped session indices, first selected/skipped samples, max observed open sessions and `scheduling_rule=rolling_open_sessions`. |
| `shared_balance_position_fraction` | Filtered backtest and baselines now process selected sessions sequentially from `shared_balance`, size each selected session as `shared_balance * position_fraction`, and update `shared_balance` only when a trade closes. Scorecards record `position_fraction_application=shared_balance_position_fraction`, sizing samples, balance update order and final shared balance. |
| Q/action filter order | Candidate filtered backtest now sends raw unmasked Q-values into `FilteredBacktestPolicy`; invalid/no-op/last-step action coercion stays in environment/backtest semantics. Balance-curve trace rows record raw argmax, masked Q diagnostic hash, unmasked filter action, filter-selected action and effective action separately. |
| Ensemble filter source correction | `ensemble_q_filter` now follows upstream `backtest_engine.py`: high uncertainty rejects only together with a weak advantage threshold, not by itself. |
| Reward reporting | Stage `02C` training reward is unchanged. Backtest scorecards and trace rows now separate `training_reward` / `training_reward_sum` from source-compatible `backtest_reporting_reward=0.0` / `backtest_reporting_reward_sum=0.0`. The legacy `reward` trace field is retained as an alias of `backtest_reporting_reward` for compatibility. |
| Close/open price index and last-step close | Existing `session_close_price_v1()` semantics are retained; last-step open is blocked and open position is forced closed through environment semantics. Risk-management last-step forced close is now classified explicitly when risk management is enabled. |
| Q-cache/state normalization | Candidate backtest cache key now follows upstream `(symbol, signal_dt + step minutes)` shape; train-only normalization stats remain unchanged and are still loaded from the candidate manifest. |

## Parity Evidence

Mac Studio command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run --extra rl-ml python - --trace-session-count 20 --compare-session-count 50 --torch-num-threads 1 --torch-num-interop-threads 1"' < scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py
```

Result:

```json
{"first_material_diff": null, "manifest_path": "/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/stage08i_trace_manifest.json", "manifest_sha256": "b22896c6202613032db2e4773ae4c9e55f2dd323a8db8902e5692529f77f0384", "run_dir": "/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde", "run_id": "stage08i_forensic_3291fdbb8de3d4d01cde", "status": "accepted"}
```

Artifact hashes:

| Artifact | sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/first_material_diff.json` | `43465cb4b9e732d24e06f8f0946d6f06d7c655fb521c15cf01f894d5f23aba80` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/selection_comparison.json` | `ee54d233df4bbae2b7b7df82fbcaeefe25dce465c470fde9dc1e5e298b3ac502` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/source_derived_upstream_trace.jsonl` | `92f100688f02954394f264d52ed4ec8d32fb0b6f08ed3e396ec98231f27d6ee0` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/roehub_current_trace.jsonl` | `371b182caffe177158d16d70777db979e7bc7c953ddc3ca15191b5f3a0582eaf` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/stage08i_trace_manifest.json` | `b22896c6202613032db2e4773ae4c9e55f2dd323a8db8902e5692529f77f0384` |

Notes:

- The runtime file name `roehub_current_trace.jsonl` is retained for compatibility with the existing `08I` artifact schema, but rows now record `implementation=roehub_repaired`.
- Local forensic runtime was not accepted as evidence because the local machine did not have `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_candidate_manifest.json`. The accepted real-boundary evidence is the Mac Studio run above.

## Quality Gates

| Check | Result |
|---|---|
| Focused regression tests | passed: `uv run pytest -q tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py tests/unit/contexts/rl_trading/domain/test_hf_original_evaluation.py tests/unit/contexts/rl_trading/domain/test_roehub_native_evaluation.py` -> `8 passed` |
| Prompt ruff gate | passed: `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` |
| Prompt pyright gate | passed: `uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `0 errors` |
| Prompt pytest gate | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `100 passed` |
| Mac Studio forensic trace | passed with `status=accepted`; manifest sha256 `b22896c6202613032db2e4773ae4c9e55f2dd323a8db8902e5692529f77f0384` |
| Docs index | passed after regeneration: `uv run python -m tools.docs.generate_docs_index --check` |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration/table/storage schema changed. |
| Config schema/defaults | `none` | No production config/default changed. Existing evaluation config fields keep the same shape. |
| Request hash / cache key / persistence identity | `compatible-change` | Offline evaluator Q-cache identity now matches upstream `(symbol, signal_dt + step)` semantics. No production request/cache identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, paper/testnet/live/mainnet or provider side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Offline scorecards/traces now add explicit scheduling, shared-balance, Q/action and reward-reporting fields. |
| Benchmark / rollout gates | `compatible-change` | `08I3` may unlock only `08I4`; it does not open `08J`, `08K` or `09`. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline research evaluator only; no API/live inference hot path changed. |

## File Manifest

| Path | Change | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/hf_original_evaluation.py` | modified | Repair grouped filtered backtest scheduling, shared-balance sizing, raw-Q filter order, explicit reward semantics and trace fields. | `compatible-change` offline evaluator/scorecard semantics |
| `src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py` | modified | Apply repaired rolling/shared-balance scheduler to native random baseline so Stage `08F` scorecards stay internally coherent. | `compatible-change` offline evaluator baseline semantics |
| `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | modified | Align ensemble filter rejection with upstream source behavior. | `compatible-change` offline filtered policy semantics |
| `scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py` | modified | Compare source-derived trace against repaired Roehub trace and keep legacy exact-group fixture for regression evidence. | `compatible-change` opt-in forensic CLI |
| `tests/unit/contexts/rl_trading/domain/test_hf_original_evaluation.py` | modified | Add regression coverage for rolling scheduler, shared-balance sizing, unmasked-Q filter order and reward separation. | `none` test-only |
| `tests/unit/contexts/rl_trading/domain/test_roehub_native_evaluation.py` | indirectly covered, no file edit | Existing native evaluation test verifies repaired scheduler integration through shared evaluator. | `none` test-only |
| `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | modified | Update ensemble-policy regression to source-compatible weak-advantage plus high-uncertainty semantics. | `none` test-only |
| `tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py` | modified | Assert repaired scheduler matches source and legacy exact-group scheduler still shows old diff. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md` | created | Stage `08I3` accepted report and evidence manifest. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08I3` accepted, advance `current_stage` to `08I4`, and keep `08J`/`08K`/`09` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync the plan with accepted `08I3` evidence and the next active `08I4` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index regeneration after adding this report. | `compatible-change` docs index |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/*` | modified outside git | Sanitized Mac Studio forensic trace artifacts. | `compatible-change` non-production runtime evidence |

Outside expected `08I3` paths: none introduced by this stage. The current worktree also contains unrelated non-RL changes outside this report scope; they were not used as `08I3` acceptance evidence and must not be staged as part of this stage.

## Cold-Head Review

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `08I3` report, ledger update, plan sync, docs index, file/runtime artifact manifest, proof-boundary wording, browser/auth redaction, contract impact, quality gates and `08I4` handoff.
Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`
Verdict: Release
Blockers fixed: docs index pending state was replaced with passed check evidence; ledger `current_stage` advanced to `08I4`; file manifest records the report, ledger, plan and docs index updates; proof boundary is labeled `target_host_non_production_forensic_pre_main`.
Local follow-up check: completed; docs index check, `git diff --check`, and scoped grep sanity passed.
Residual risks: `08I4` must still recheck the full `08I2` matrix; `08J`, `08K` and `09` remain blocked; Mac Studio evidence is pre-main non-production forensic evidence, not production proof.

## Residual Risks

- `08I3` accepts evaluator/action/reward-reporting parity for the required pre-`08J` surface only. It does not accept the article session extractor, dataset geometry, native model quality, `Optuna`, registry, promotion, paper/testnet/live execution or Stage `09`.
- The Mac Studio trace used the current stage script through stdin and wrote sanitized artifacts under `/opt/roehub/state/rl_trading/`; it is valid `target_host_non_production_forensic_pre_main` evidence, not a production deploy proof.
- `08I4` must still recheck all `08I2` rows and decide whether `08J` may start. Until then, `08J`, `08K` and `09` remain blocked.

## Handoff

Next allowed prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/08i4-post-repair-methodology-recheck.md`.

`08I4` must use this accepted `08I3` report plus the blocked `08I2` matrix to classify every methodology row as closed, assigned to `08J`/`08K`, superseded, not applicable or still blocking. `08I4` may allow `08J` only if no material evaluator/session/action/reward-reporting blocker remains and it records `08j_allowed=true`. Stage `09` remains blocked.
