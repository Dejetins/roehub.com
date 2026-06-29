# Stage 08H — oracle/supervised/session/reward диагностика и `90/60` research repair

## Статус

| Поле | Значение |
|---|---|
| `stage` | `08H` |
| `status` | `blocked` |
| `started_at` | `2026-06-26` |
| `completed_at` | `2026-06-29` |
| `previous_blocker` | `08G` Roehub-native final holdout PnL `-145.16434371` после `Optuna` |
| `stage09_allowed` | `false` |
| `blocker` | `90/60` trade-sufficient Roehub-native `Optuna` candidates failed final Stage `06` holdout and lost to sanity baseline |

`08H` открыт после пользовательского review результатов `08G`. Цель stage — не “подобрать еще один порог”, а проверить более базовые вопросы: есть ли в сессиях теоретические точки входа/выхода, видны ли они из прошлого окна без lookahead, почему HF-original и Roehub-native ведут себя по-разному, не слишком ли разрежен reward, и меняет ли картину обязательный профиль `90/60`.

## Бизнес-смысл

`08H` защищает продукт от преждевременной регистрации модели, которая технически обучается, но не даёт полезного торгового поведения. Если Stage `09` открыть без этой проверки, Roehub рискует сохранить и дальше продвигать модель, которая:

- торгует слишком редко, как в `08G` с `19` и `3` сделками;
- выглядит положительной только из-за подбора порогов на маленьком числе сделок;
- проигрывает простым baseline-правилам;
- обучается на сессиях, где движение есть, но направление не предсказуемо из доступной истории;
- получает слишком редкий reward-сигнал и поэтому не учится устойчивой политике.

В этом stage нет live trading, exchange submit, пользовательских денег, новых secrets, миграций БД, API-контрактов или browser-visible UI. Операционный риск ограничен CPU/MPS/RAM/IO на `macstudio` и дисковыми ML artifacts под `/opt/roehub/state/rl_trading/`.

## Операционные поверхности

| Поверхность | Статус |
|---|---|
| Service calls | N/A: stage читает локальные HF/Stage `06` artifacts и пишет локальные ML summaries/checkpoints. |
| Exchange/provider side effects | N/A: нет live/testnet/paper submit и нет provider API calls. |
| Secrets/redaction | N/A: новые secrets не читаются; reports фиксируют только sanitized paths, counts, hashes и метрики. |
| Database/API/browser contracts | N/A: нет migrations, API routes, DTO, UI или browser-visible changes. |
| Alerts/monitoring | N/A для production; runtime monitoring этого stage — ручной polling PID/status files на `macstudio`. |
| Runbook action | `08H` завершён как blocked evidence; Stage `09` не запускать. Следующий research stage должен разбирать calibration-to-holdout overfit, reward/session selector/features или выполнять upstream byte-for-byte forensic parity. |

## Что проверяется

| Проверка | Простое объяснение | Статус |
|---|---|---|
| Oracle opportunity | Смотрим в будущее только для диагностики и спрашиваем: “была ли в сессии прибыльная сделка после costs?” | completed |
| Past-only supervised sanity | Учим простую модель только на прошлом окне и проверяем, можно ли предсказать oracle-направление лучше простых правил. | completed |
| Selector regimes | Сравниваем high-volatility, trend, liquidity и mean-reversion proxies. | completed |
| Reward sparsity proxy | Оцениваем, насколько редко текущий reward дает полезный PnL-сигнал внутри эпизода. | completed |
| `90/60` training/evaluation | Переобучили HF-original и Roehub-native с `agent_history_len=90`, `agent_session_len=60`, затем снова запустили `Optuna` и финальный holdout. | completed_blocked |

## Уже полученные артефакты

| Артефакт | Значение |
|---|---|
| Diagnostic CLI | `scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py` |
| Full diagnostics summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/stage08h_full_dataset_diagnostics_20260626/stage08h_dataset_diagnostics_summary.json` |
| Full diagnostics sha256 | `9a0fe21114dfc25cf3fb2c2f183f5a8cf8bc2faf398ad9295fc1d71ca8cae338` |
| Full diagnostics `summary_hash` | `461153e08f581459b5e37e391ecf3b15d7d3f7292e5b5a5fec206371fdea0e7c` |
| `90/60` smoke summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08g_dual_branch_cpu_optuna_training_evaluation_v1/dual_branch_runs/stage08h_90_60_real_smoke2_20260626/stage08g_dual_branch_cpu_run_summary.json` |
| `90/60` smoke sha256 | `d95ef5b47f18adc1671811e1f126393da323081516eecdb4d00f2c9358f1d194` |
| Superseded single-thread `90/60` run id | `stage08h_dual_branch_cpu_90_60_full_20260626T141800Z` |
| Superseded single-thread PIDs | `97712`, `97713`, `97714` stopped on `macstudio` after user requested max CPU threads |
| Superseded max-thread CPU `90/60` run id | `stage08h_dual_branch_cpu_90_60_full_max_threads_20260626T211226Z` |
| Deleted CPU runtime artifacts | Removed exact old CPU paths under `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/dual_branch_runs/` and `/opt/roehub/state/rl_trading/training_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/hf_original/`; `stage08h_mps_speed_canary_v1` was left intact. |
| `MPS` speed canary run id | `stage08h_hf_90_60_mps_speed_canary_20260626T214237Z` completed `1000/1000` episodes on `device=mps` in `619.546755542` sec, about `1.61` episodes/sec. |
| Final full `90/60` run id | `stage08h_dual_branch_mps_90_60_full_20260626T215849Z` |
| Final full `90/60` summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/dual_branch_runs/stage08h_dual_branch_mps_90_60_full_20260626T215849Z/stage08h_dual_branch_cpu_run_summary.json` |
| Final full `90/60` summary sha256 | `f4820678327b78137522418e1e4b7e105c702ccb6f3e3fc52b57176a6b3dc82b` |
| Corrected native holdout recheck summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/optuna/stage08h_roehub_native_9f6e307f_7cbdd825ddd9f8aacb88/manual_final_holdout_rechecks_20260629/manual_final_holdout_recheck_summary.json` |
| Corrected native holdout recheck summary sha256 | `01bb66fbf69a0ee7871b12600af0fb5a752ceaa559442f5425d8c6848b0a1f46` |

## Диагностический вывод

Oracle показывает, что движение в сессиях есть. На `90/60` backtest split:

| Branch | Sessions | Oracle positive ratio | Mean best net return | Long/short labels |
|---|---:|---:|---:|---|
| `hf_original` | `3186` | `1.0` | `0.0739193702` | `1545` / `1641` |
| `roehub_native` | `14731` | `0.9969452176` | `0.0426501432` | `7554` / `7132` |

Это значит: проблема не выглядит как “в наших сессиях вообще нет движения”. Движение есть, но оно слабее по среднему oracle-return на native backtest.

Past-only supervised sanity слабее на Roehub-native. На `90/60` backtest:

| Branch | Ridge accuracy | Balanced accuracy | Majority baseline | Recent-return baseline |
|---|---:|---:|---:|---:|
| `hf_original` | `0.578782172` | `0.5780207427` | `0.4849340866` | `0.4030131827` |
| `roehub_native` | `0.5536623447` | `0.3677879955` | `0.5127961442` | `0.4170117439` |

Простыми словами: даже простая past-only модель на native не дает сильного и устойчивого сигнала по классам. Это поддерживает гипотезу, что текущая связка “Stage `06` selector + текущие признаки + текущий reward + DQN profile” плохо извлекает предсказуемое направление.

Reward proxy показывает разреженность сигнала. Для `90/60` текущая realized-only схема дает полезный trade-step сигнал примерно `0.0333333333` эпизода, тогда как dense mark-to-market proxy покрывает примерно `0.55` эпизода. Это не доказывает, что reward надо менять, но объясняет, почему RL может плохо учиться: полезная награда приходит редко и поздно.

## Почему baseline разные

`baseline` — это контрольная стратегия на конкретном branch/split/profile. Его нельзя читать отдельно от датасета.

Например:

- `hf_original` baseline считается на original HF split.
- `roehub_native` baseline считается на Stage `06` split.
- `majority baseline` в supervised sanity — “всегда предсказывать самый частый класс”.
- `recent-return baseline` — простое правило по прошлому движению цены.
- backtest baseline из `08G` — торговая стратегия в grouped backtest с costs.

Поэтому разные baseline — это не ошибка сама по себе. Ошибка была бы сравнивать число baseline без branch, split, profile и правила расчета.

## Что исправлено в коде

| Изменение | Причина |
|---|---|
| `UpstreamAlphaConfig.as_payload()` теперь сохраняет `full_seq_len`, `pre_signal_len`, `agent_history_len`, `agent_session_len`. | Без этого `90/60` checkpoint записывался с правильной формой, но evaluator восстанавливал default `30/10`. |
| `stage08c_original_hf_full_training_run.py` и `stage08e_roehub_native_full_training_run.py` получили `--agent-history-len` и `--agent-session-len`. | Нужно реально запускать `90/60`, а не только описывать его в документах. |
| `stage08c_original_hf_full_training_run.py`, `stage08e_roehub_native_full_training_run.py`, `stage08g_cpu_optuna_calibration.py`, `stage08g_dual_branch_cpu_training_evaluation.py` и `UpstreamAlphaConfig` теперь используют `os.cpu_count()` как default для `torch_num_threads`. | Новый стандартный режим должен использовать максимально доступное число CPU threads на target host, если пользователь явно не передал другой `--torch-num-threads`. |
| `stage08g_cpu_optuna_calibration.py` получил профильные параметры и `--stage-label`. | `Optuna` должен оценивать checkpoint с той же формой входа, с которой модель обучалась. |
| `stage08g_dual_branch_cpu_training_evaluation.py` получил профильные параметры и `--stage-label 08H`. | `08H` summary должен быть помечен как `08H`, а не как новый `08G`. |
| `stage08g_dual_branch_cpu_training_evaluation.py` получил `--device-policy`. | Полный `08H` должен уметь запускать HF-original и Roehub-native training на `mps_preferred_cpu_fallback`, а не быть навсегда зафиксированным на `cpu_only_deterministic`. |
| `stage08g_cpu_optuna_calibration.py` больше не оптимизирует `-closed_trades` и выбирает лучший trial из всех завершённых попыток по явному правилу. | Старый многоцелевой Pareto-front selection мог выбрать zero-trade trial вместо trade-sufficient плюсового calibration trial. |
| `stage08g_cpu_optuna_calibration.py` получил `--min-calibration-closed-trades`, default `100`, и summary field `selection_rule`. | Выбор лучшей конфигурации теперь сначала требует достаточное число закрытых сделок, затем сортирует по `return_pct_after_costs`, `win_rate`, просадке и baseline delta. |
| Добавлен `scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py`. | Нужна отдельная диагностика данных/цели до финального качества модели. |

## Финальный `90/60` результат после исправленного выбора trial

Полный `MPS` run `stage08h_dual_branch_mps_90_60_full_20260626T215849Z` завершился на `macstudio`, но не открыл Stage `09`.

HF-original ветка в `90/60` профиле завершила обучение и `100/100` `Optuna` trials, но финальный holdout был отрицательным:

| Метрика | Значение |
|---|---:|
| `closed_trades` | `33675` |
| `net_pnl_after_costs_quote` | `-201598.796937` |
| `return_pct_after_costs` | `-0.7354936` |
| `win_rate` | `0.4569265` |

Roehub-native ветка тоже завершила обучение и `100/100` `Optuna` trials. Первичный summary ошибочно выбрал `best_trial_number=1`: на калибровке у него было `0` закрытых сделок и `0.0` return, потому что старый selection брал первый элемент из `study.best_trials` многоцелевой `Optuna`. Это Pareto-front, а не “лучший по доходности” список; при цели `-closed_trades` zero-trade trial становился удобным Pareto-кандидатом.

После исправления правила выбора вручную пересчитаны финальные Stage `06` holdout для trade-sufficient плюсовых calibration trials `82`, `63` и `11`:

| Trial | Calibration return | Calibration trades | Final trades | Final PnL after costs | Final return | Final win rate | Best sanity baseline | Candidate beats baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `82` | `0.1704538` | `3316` | `35015` | `-229005.38413725` | `-0.44389491` | `0.35444809` | `481012.90631972` | `false` |
| `63` | `0.16679456` | `3365` | `35015` | `-228818.95542758` | `-0.44353354` | `0.35441953` | `481012.90631972` | `false` |
| `11` | `0.16632319` | `3366` | `35014` | `-228765.96850913` | `-0.44343084` | `0.35442966` | `481012.90631972` | `false` |

Вывод: selection bug был реальным, но он не скрывал принятую рабочую native-модель. Даже если выбрать лучший trade-sufficient calibration trial `82`, финальный untouched holdout становится сильно отрицательным и проигрывает baseline. Это означает calibration-to-holdout overfit или слабую устойчивость текущей связки `Stage 06 sessions + 90/60 profile + current reward + current features + DQN + Optuna thresholds`.

Финальный verdict `08H`: `blocked`, `stage09_allowed=false`.

## Завершённый полный run

Первый полный run `stage08h_dual_branch_cpu_90_60_full_20260626T141800Z` был остановлен после пользовательского запроса использовать максимально доступное число CPU cores. Причина остановки: этот run был запущен с `--torch-num-threads 1`, фактически грузил около одного CPU core, а пользователь признал такой режим неприемлемо долгим для полного `08H`.

Второй run `stage08h_dual_branch_cpu_90_60_full_max_threads_20260626T211226Z` был остановлен после `MPS` speed canary, потому что пробный HF-original `90/60` запуск `stage08h_hf_90_60_mps_speed_canary_20260626T214237Z` завершил `1000/1000` episodes на `device=mps` за `619.546755542` sec, примерно `1.61` episodes/sec, что примерно в `3x` быстрее наблюдавшегося CPU-хвоста `0.46-0.52` episodes/sec. По явной просьбе пользователя старые CPU runtime artifacts были удалены; `MPS` canary оставлен как evidence.

Финальный полный run был запущен на `macstudio` с `--device-policy mps_preferred_cpu_fallback`, `--torch-num-threads 12` и `--torch-num-interop-threads 1`:

| Поле | Значение |
|---|---|
| `run_id` | `stage08h_dual_branch_mps_90_60_full_20260626T215849Z` |
| `status` | `completed` |
| `stage09_allowed` | `false` |
| `hf_original_training_device` | `mps` |
| `roehub_native_training_device` | `mps` |
| `device_policy` | `mps_preferred_cpu_fallback` |
| `torch_num_threads` | `12` |
| `torch_num_interop_threads` | `1` |
| `hf_training_run_id` | `stage08c_hf_original_0df6fc8eac61d2e0bc02` |
| `native_training_run_id` | `stage08e_roehub_native_61995c61_e1f366f4b49aba71fc37` |
| `summary_sha256` | `f4820678327b78137522418e1e4b7e105c702ccb6f3e3fc52b57176a6b3dc82b` |

Так как `stage09_allowed=false`, Stage `09` не может стартовать.

Proof boundary: это `target_host_non_production_training_pre_main` / offline ML runtime evidence на `macstudio`. Это не `post_main_production_runtime_proof`, не browser-visible proof и не проверка `/opt/roehub/app`. Для `post_main_production_runtime_proof` потребовались бы target revision on `main`, зеленые GitHub Actions/CI, deploy или verified sync из `main` checkout в `/opt/roehub/app`, а затем production smoke/API/browser verification по измененному production runtime.

Environment/comparability note: `MPS` canary выполнен на том же `macstudio`, в том же `08H` HF-original `90/60` training stage, на том же HF dataset и с теми же основными hyperparameters. Отличался только лимит `episodes=1000` и отдельный `run_id/output-root`; canary является directional speed evidence, а не финальным acceptance proof полного `55000`-episode run. Он подтвердил, что training CLI реально работает на `device=mps`, а не уходит в CPU fallback.

Performance observation: `MPS` canary `stage08h_hf_90_60_mps_speed_canary_20260626T214237Z` завершил `60000/60000` env steps за `619.546755542` sec, около `96.77` steps/sec. Полный `MPS` run стартовал с той же device policy, первый observed status подтвердил `device=mps`, и затем обе training ветки завершились на `MPS`.

## Следующие задачи

`08H` завершён как blocked research evidence. Следующий проход не должен открывать Stage `09`. Полезные следующие задачи:

1. Провести upstream forensic parity: запустить pinned original repo `f71130903f8237351164f4b875494185465bf1ea` с минимальными техническими изменениями путей/зависимостей и сравнить наш evaluator с оригинальным `backtest_engine.py` по одной и той же конфигурации.
2. Разобрать calibration-to-holdout overfit: почему native trials `82`/`63`/`11` дают `+0.166%`-`+0.170%` на calibration, но около `-0.443%` на final holdout.
3. Проверить action/filter distribution для rechecked trials: почему final holdout даёт около `35015` сделок и `win_rate` около `0.354`, хотя calibration имел около `3300` сделок и `win_rate` около `0.56`.
4. Пересмотреть reward: текущий realized-only reward остаётся разреженным; dense mark-to-market proxy должен быть исследован как отдельный stage, а не молча смешан с `08H`.
5. Сравнить selectors: high-volatility Stage `06` против trend/liquidity/mean-reversion session populations на одинаковом `90/60` профиле.
6. Добавить trade-sufficient selection rule как обязательный safety guard для будущих `Optuna` stages; zero-trade trial больше не должен становиться финальным кандидатом.
7. Не трогать `/opt/roehub/app` и не заявлять `post_main_production_runtime_proof`: этот stage пишет offline ML artifacts под `/opt/roehub/state/rl_trading/`, а не меняет production service.

## File manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py` | - | - | Stage `08H` oracle/supervised/selector/reward diagnostics CLI. | `compatible-change` additive opt-in research CLI |
| `tests/unit/scripts/rl_trading/test_stage08h_oracle_supervised_dataset_diagnostics.py` | - | - | Focused tests for oracle and supervised sanity logic. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08h-oracle-supervised-selector-reward-90-60-research.md` | - | - | Stage `08H` final blocked report after corrected `Optuna` selection rechecks. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | Persist explicit training profile fields and validate profile bounds. | `compatible-change` artifact metadata fix; new config hashes for new runs |
| - | `scripts/rl_trading/stage08c_original_hf_full_training_run.py` | - | Add profile CLI parameters for HF training. | `compatible-change` additive opt-in CLI options |
| - | `scripts/rl_trading/stage08e_roehub_native_full_training_run.py` | - | Add profile CLI parameters for native training. | `compatible-change` additive opt-in CLI options |
| - | `scripts/rl_trading/stage08g_cpu_optuna_calibration.py` | - | Add profile-aware evaluation, stage labeling, explicit trade-sufficient best-trial selection, and remove the `-closed_trades` objective. | `compatible-change` opt-in research CLI behavior correction |
| - | `scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py` | - | Add profile-aware orchestration, `08H` stage label and explicit `--device-policy` for full `MPS` training. | `compatible-change` additive opt-in CLI options |
| `tests/unit/scripts/rl_trading/test_stage08g_cpu_optuna_calibration.py` | - | - | Regression tests for removing the fewer-trades objective, selecting trade-sufficient high-return trials, and blocking no-trade candidate selection. | `none` test-only |
| - | `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | Lock payload profile fields. | `none` test-only |
| - | `tests/unit/scripts/rl_trading/test_stage08g_dual_branch_cpu_training_evaluation.py` | - | Cover `stage-label=08H` and `mps_preferred_cpu_fallback` dry-runs. | `none` test-only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Mark Stage `08H` blocked and keep Stage `09` dependent on a future accepted corrective research candidate. | `compatible-change` docs/plan |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark `08H` blocked after corrected holdout rechecks and keep `09` blocked. | `compatible-change` docs/ledger |
