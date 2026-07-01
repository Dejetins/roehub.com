---
doc: rl-trading-agent-platform-v1-stage-08i-upstream-evaluator-session-parity-forensic
stage: "08I"
status: blocked
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-07-01"
---

# Stage 08I: forensic-проверка parity upstream evaluator/session

Статус: `blocked`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08I` начат только после проверки ledger-gate: Stage `08H` имеет статус `blocked`, `current_stage=08I`, а Stage `09` закрыт. Browser/auth QA для этой offline-forensic стадии `N/A`; Roehub smoke Keycloak username и host-local источник пароля не использовались.

Эта стадия не обучала, не подбирала гиперпараметры, не регистрировала, не продвигала, не активировала модель, не запускала paper/testnet/live/mainnet trading, не читала секреты, не меняла состояние exchange/provider и не трогала `/opt/roehub/app`.

## Бизнес-смысл

`08I` защищает Roehub от нового длинного обучения на данных/reward, пока сама поверхность оценки не совместима с upstream-логикой. На бизнес-языке результат простой: текущие model-quality числа из Roehub evaluator нельзя сравнивать с оригинальным article/repo backtest так, будто они отвечают на один и тот же вопрос.

Платформа не должна регистрировать или продвигать кандидата из этой цепочки, пока backtest не использует те же правила допуска сессий, тот же active-session cap и тот же shared-balance sizing, что upstream `backtest_engine.py`. Практический эффект стадии - задержка исследовательской цепочки, а не ухудшение production-поведения: customer-facing trading не менялся. Ценность результата в том, что он предотвращает ложное принятие или ложный отказ модели на неэквивалентном evaluator.

## Операционные поверхности

| Поверхность | Статус |
|---|---|
| Service calls | `N/A`: Roehub API, worker queue, browser route, Redis, ClickHouse, provider SDK и exchange calls не выполнялись. |
| Exchange/provider side effects | `N/A`: не было paper/testnet/live/mainnet submit и не было provider-state mutation. |
| Secrets/redaction | `N/A`: credentials не читались; artifacts содержат sanitized hashes, paths, scalar trace decisions и не содержат raw provider payloads. |
| Database/API/browser contracts | `N/A`: migrations, API routes, DTOs, auth, UI и browser-visible behavior не менялись. |
| Alerts/monitoring | `N/A` для production: это offline forensic artifact под `/opt/roehub/state/rl_trading/`; Monit/launchd/runtime alert rules не менялись. |
| Runbook action | Не запускать `08J`, `08K`, `09` или новое обучение из этой цепочки. Нужно repair/supersede `08I`, затем повторить тот же forensic trace. |

## Закрепление источников

| Источник | Evidence |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08i-upstream-evaluator-session-parity-forensic.md` |
| Prompt sha256 | `bc7a1746696f8d0fe1643af276fc97582109b422ec5f1571a8df7e39092dbf75` |
| Upstream repo | `https://github.com/YuriyKolesnikov/rl-trading-binance` |
| Upstream commit | `f71130903f8237351164f4b875494185465bf1ea` |
| Upstream `backtest_engine.py` sha256 | `d05e426fdad3acb24df4c74fce17536d584e56a0b9e528160c5cb9762e179892` |
| Upstream `trading_environment.py` sha256 | `c38154ee416f1fb3de59c2f7085092d0237216c7854e70ba89863d9676920c8c` |
| Upstream `configs/alpha.py` sha256 | `c8f0348379ed4deaf7dc306bbab039203e22e4039321ab294caedd2f5f698f9e` |
| Upstream source mode | `source_derived_from_pinned_backtest_engine_without_external_repo_checkout`; upstream code не vendored в Roehub. |
| Raw source links | `https://raw.githubusercontent.com/YuriyKolesnikov/rl-trading-binance/f71130903f8237351164f4b875494185465bf1ea/backtest_engine.py`; `https://raw.githubusercontent.com/YuriyKolesnikov/rl-trading-binance/f71130903f8237351164f4b875494185465bf1ea/trading_environment.py`; `https://raw.githubusercontent.com/YuriyKolesnikov/rl-trading-binance/f71130903f8237351164f4b875494185465bf1ea/configs/alpha.py` |

## Входные артефакты

| Artifact | Evidence |
|---|---|
| HF dataset | `/opt/roehub/state/rl_trading/hf_reproducibility/dataset/ResearchRL/open-rl-trading-binance-dataset` |
| HF `backtest_data.npz` sha256 | `dce732fda8fe1d33e92617d12f0defa3e202013617b91bb34df4d0b65aa023ee`; `3,186` sessions; hash matched expected Stage `04` manifest |
| `08C` candidate manifest | `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_candidate_manifest.json` |
| `08C` candidate manifest sha256 | `189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4` |
| `best.pth` sha256 | `3538c77abb363f6ade74cc98113fc5a19be78b2f63c5449e675485ee8ce36e0c` |
| Train-only normalization stats hash | `d56be74b3f4f2779ea9dbe72302b5e918a806e23d3903810c77e43d615c2b254` |
| Evaluation config hash | `e4116f3cef24f550c1ce6e63c4581b76f294829be822c2f8dc7ad0ba794bff77` |

## Runtime-доказательства

Proof boundary: `target_host_non_production_forensic_pre_main`.

Mac Studio trace run:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && uv run --extra rl-ml python - --trace-session-count 20 --compare-session-count 50 --torch-num-threads 1 --torch-num-interop-threads 1"' < scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py
```

Команда намеренно вернула exit code `2`, потому что verdict стадии `blocked`.

| Artifact | sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/stage08i_trace_manifest.json` | `6e33daa8bf4b857d9aef3db3bdf2ccf93fab20f90a7d400c3ff2ea1d764ad13d` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/first_material_diff.json` | `229df2c6cb72179b84d69b8e015ce02f5aeb3ad188bf0bea8b1bce2bc963ca21` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/selection_comparison.json` | `1c7126ec3602261bdecec49ebe7241e21bdd31b7fc175a7f8ad7a04a2c56721c` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/source_derived_upstream_trace.jsonl` | `6835e5be56c135e827dc64d101b68c64a0441dc13eaefe67853eea0dab4fe9c3`; `200` rows |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08i_upstream_evaluator_session_parity_forensic_v1/stage08i_forensic_3291fdbb8de3d4d01cde/roehub_current_trace.jsonl` | `c08dd707063ad697232c15693f9f91fd5e7721320c183b0c718f37b73e99e605`; `200` rows |

## Первый материальный diff

Первый материальный diff найден в порядке выбора сессий до любого нового обучения или model-quality claim:

| Field | Upstream `backtest_engine.py` source-derived behavior | Roehub current `08D` behavior |
|---|---|---|
| Diff type | `session_selection_order` | `session_selection_order` |
| Selected order | `23` | `23` |
| Session index | `93` | `76` |
| Signal time | `2025-03-02T15:37:00Z` | `2025-03-02T15:27:00Z` |
| Rule | `rolling_open_sessions_cap` | `exact_signal_time_group_cap` |

Почему это materially blocks parity: upstream `backtest_engine.py` держит `open_sessions` между signal groups и освобождает слоты только когда `end_time <= signal_dt`. Roehub current grouped backtest ограничивает только группы с одинаковым `signal_time` перед оценкой. Это меняет набор сессий, порядок, active-session cap semantics и итоговую сопоставимость PnL/action selection.

Trace также нашел первый non-material step-field diff:

| Field | Upstream source-derived trace | Roehub trace | Material? |
|---|---:|---:|---|
| `reward` at selected order `0`, `VANAUSDT`, step `0` | `0.0` | `-0.001` | no for final PnL/action selection |

Причина: upstream `TradingEnvironment.backtest_step()` возвращает `reward = 0.0` в backtest mode, а Roehub current trace показывает training reward field. Это разница reporting/trace semantics, но не первый материальный blocker для PnL/action comparability.

## Классификация parity gap

| Surface | Result | Evidence |
|---|---|---|
| Shared balance vs independent aggregation | gap | Upstream sizes each selected session as `balance * position_fraction`; Roehub current scorecard records `position_fraction_application=initial_balance_scaled_for_session_pnl` and initializes aggregate equity as `initial_balance * selected_session_count`. |
| Signal group ordering and active-session cap | material gap | First material diff at selected order `23`: upstream rolling `open_sessions` selects session `93`; Roehub exact-group cap selects session `76`. |
| Close/open price index semantics | no first material diff observed before scheduler blocker | First trace row state/q/price/action hashes matched; deeper claim blocked until scheduler parity is fixed. |
| Last-step action mask | not accepted | Controlled trace did not reach this as first material blocker; current Roehub masks before filter, while upstream filters unmasked q-values and lets `backtest_step()` coerce/no-op actions. Must be rechecked after scheduler parity. |
| Commission/slippage application | no first material diff observed before scheduler blocker | Source review shows both paths use entry/exit slippage and fees; final acceptance blocked by scheduler/sizing parity first. |
| Action filter thresholds | no first material diff observed before scheduler blocker | `configs/alpha.py` thresholds were used; deeper parity blocked by selected-session mismatch. |
| Risk-management timing | `N/A` for this config | `configs/alpha.py` has `use_risk_management=false`; keep this as a later Optuna/risk-management parity check when enabled. |

## Semantic gap в извлечении сессий

`08I` подтверждает, что evaluator/backtest parity еще не принята. Отдельно остается dataset/session-selection gap:

- Upstream article/repo HF sessions are event-selected externally into `fetcher_N` arrays keyed by `(ticker, signal_datetime)` and then consumed by `backtest_engine.py`.
- Historical Stage `06` Roehub-native sessions use `pre_signal_realized_volatility_plus_range_v1`.
- Plan correction уже определяет Stage `08J` как место для отдельного `article_future_10m_5pct_contrast_v1`: найти future/event window с движением цены минимум `5%` за `10m`, исключить события, где предыдущие `90m` уже содержали похожий импульс, зафиксировать `event_end_t` как `signal_ts_open`, затем строить `pre_window=[signal_ts_open-90m, signal_ts_open)` и `post_window=[signal_ts_open, signal_ts_open+60m)`.

Stage `08J` нельзя начинать из текущего результата `08I`, потому что evaluator parity заблокирована.

## Вердикт

`08I` is `blocked`.

Не переходить к `08J`, `08K`, `09`, `10` или `10A`. Следующая работа должна repair/supersede `08I` evaluator parity: привести Roehub backtest session scheduling и shared-balance sizing к source-compatible behavior, затем повторить тот же trace на 20-50 сессиях и focused gates. В этой стадии не было speculative reward/session redesign.

## Манифест файлов

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py` | - | - | Add opt-in source-derived upstream vs Roehub trace CLI for `08I` forensic evidence. | `compatible-change` additive research CLI |
| `tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py` | - | - | Cover scheduler diff and first trace diff detection. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md` | - | - | Stage `08I` blocked report. | `compatible-change` docs/report |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Record `08I` blocked, keep `08J`/`09` blocked, and add evidence handoff. | `compatible-change` docs/ledger |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Sync plan narrative with final `08I` blocked evidence. | `compatible-change` docs/plan |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report, if required by the docs tool. | `compatible-change` docs index |

Outside expected paths: none.

## Влияние на контракт

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No database migration/table/storage schema changed. |
| Config schema/defaults | `none` | No production config/default changed; the new CLI is opt-in. |
| Request hash / cache key / persistence identity | `none` | No production request/cache/persistence identity changed. Trace uses local artifact hashes only. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, paper/testnet/live/mainnet or provider side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized forensic trace artifacts and stage report under `/opt/roehub/state/rl_trading/` and docs. |
| Benchmark / rollout gates | `compatible-change` | Keeps `08I` and downstream Stage `09` blocked until evaluator parity is repaired. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline forensic CLI only; no API/live inference hot path changed. |

## Проверки качества

| Gate | Result |
|---|---|
| Ledger prerequisite gate | passed: `08H` is `blocked`, `current_stage=08I`, `09` blocked |
| Prompt hash | passed: `bc7a1746696f8d0fe1643af276fc97582109b422ec5f1571a8df7e39092dbf75` |
| Upstream source read | passed: pinned raw source read; `backtest_engine.py`, `trading_environment.py`, `configs/alpha.py` hashes recorded |
| Focused unit test | passed: `uv run pytest -q tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py` -> `2 passed` |
| Focused ruff | passed: `uv run ruff check scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py` |
| Focused pyright | passed: `uv run pyright scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py` -> `0 errors` |
| Mac Studio forensic trace | completed with blocked verdict; manifest sha256 `6e33daa8bf4b857d9aef3db3bdf2ccf93fab20f90a7d400c3ff2ea1d764ad13d` |
| Prompt ruff gate | passed: `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` |
| Prompt pyright gate | passed: `uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `0 errors` |
| Prompt pytest gate | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading` -> `98 passed` |
| Docs index | regenerated after new report and passed `uv run python -m tools.docs.generate_docs_index --check` |

## Cold-head review

Cold-head review: completed.

Mode: `cold self-review fallback`.

Independent subagent review не использовался, потому что доступный multi-agent tool contract запрещает spawn subagents без прямой просьбы пользователя на subagents/delegation/parallel agent work.

Review scope: Stage `08I` report, ledger update, plan sync, file/runtime artifact manifest, proof-boundary wording, redaction/browser-auth scope, contract impact, quality gates и `08J`/`09` blocker handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: `Release after final gates`.

Blockers fixed: добавлены явный `blocked` verdict, точный first material diff, runtime artifact hashes, запрет advancement в `08J`/`09`, ограничение source-derived upstream execution, browser/auth `N/A`, отсутствие production proof claim, file manifest и contract impact.

Local follow-up check: completed for report/ledger/plan consistency before final prompt gates.

Residual risks: source trace построен по pinned raw `backtest_engine.py`, а не через full external repo checkout/run. Этого достаточно, чтобы fail closed на найденном semantic mismatch, но accepted repair должен повторить parity после source-compatible scheduling/sizing и отдельно перепроверить action-mask/cache/risk-management детали за пределами первого material diff.

## Handoff следующей стадии

Следующая разрешенная работа: только `08I` repair/supersede pass.

Следующий executor должен:

- сохранить этот blocked report как evidence;
- сделать Roehub grouped backtest совместимым с upstream rolling `open_sessions` semantics и shared `balance * position_fraction` sizing либо явно supersede это требование reviewed compatibility decision;
- повторить `08I` trace на 20-50 тех же HF `backtest_data.npz` sessions с тем же `08C` `best.pth`;
- только после этого рассматривать `08J`.

Stage `08J` remains blocked. Stage `09` remains blocked.
