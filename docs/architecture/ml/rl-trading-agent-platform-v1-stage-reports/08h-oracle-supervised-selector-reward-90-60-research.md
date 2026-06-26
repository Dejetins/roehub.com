# Stage 08H — oracle/supervised/session/reward диагностика и `90/60` research repair

## Статус

| Поле | Значение |
|---|---|
| `stage` | `08H` |
| `status` | `in_progress` |
| `started_at` | `2026-06-26` |
| `previous_blocker` | `08G` Roehub-native final holdout PnL `-145.16434371` после `Optuna` |
| `stage09_allowed` | `false` |

`08H` открыт после пользовательского review результатов `08G`. Цель stage — не “подобрать еще один порог”, а проверить более базовые вопросы: есть ли в сессиях теоретические точки входа/выхода, видны ли они из прошлого окна без lookahead, почему HF-original и Roehub-native ведут себя по-разному, не слишком ли разрежен reward, и меняет ли картину обязательный профиль `90/60`.

## Бизнес-смысл

`08H` защищает продукт от преждевременной регистрации модели, которая технически обучается, но не даёт полезного торгового поведения. Если Stage `09` открыть без этой проверки, Roehub рискует сохранить и дальше продвигать модель, которая:

- торгует слишком редко, как в `08G` с `19` и `3` сделками;
- выглядит положительной только из-за подбора порогов на маленьком числе сделок;
- проигрывает простым baseline-правилам;
- обучается на сессиях, где движение есть, но направление не предсказуемо из доступной истории;
- получает слишком редкий reward-сигнал и поэтому не учится устойчивой политике.

В этом stage нет live trading, exchange submit, пользовательских денег, новых secrets, миграций БД, API-контрактов или browser-visible UI. Операционный риск ограничен CPU/RAM/IO на `macstudio` и дисковыми ML artifacts под `/opt/roehub/state/rl_trading/`.

## Операционные поверхности

| Поверхность | Статус |
|---|---|
| Service calls | N/A: stage читает локальные HF/Stage `06` artifacts и пишет локальные ML summaries/checkpoints. |
| Exchange/provider side effects | N/A: нет live/testnet/paper submit и нет provider API calls. |
| Secrets/redaction | N/A: новые secrets не читаются; reports фиксируют только sanitized paths, counts, hashes и метрики. |
| Database/API/browser contracts | N/A: нет migrations, API routes, DTO, UI или browser-visible changes. |
| Alerts/monitoring | N/A для production; runtime monitoring этого stage — ручной polling PID/status files на `macstudio`. |
| Runbook action | Проверять PID `9704`/`9706`/`9707`, `latest_status.json`, `progress.jsonl` и итоговый `stage08h_dual_branch_cpu_run_summary.json`; Stage `09` не запускать до accepted verdict. |

## Что проверяется

| Проверка | Простое объяснение | Статус |
|---|---|---|
| Oracle opportunity | Смотрим в будущее только для диагностики и спрашиваем: “была ли в сессии прибыльная сделка после costs?” | completed |
| Past-only supervised sanity | Учим простую модель только на прошлом окне и проверяем, можно ли предсказать oracle-направление лучше простых правил. | completed |
| Selector regimes | Сравниваем high-volatility, trend, liquidity и mean-reversion proxies. | completed |
| Reward sparsity proxy | Оцениваем, насколько редко текущий reward дает полезный PnL-сигнал внутри эпизода. | completed |
| `90/60` training/evaluation | Переобучаем HF-original и Roehub-native с `agent_history_len=90`, `agent_session_len=60`, затем снова запускаем `Optuna` и финальный holdout. | running |

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
| Current full `90/60` run id | `stage08h_dual_branch_cpu_90_60_full_max_threads_20260626T211226Z` |
| Current full `90/60` PIDs | orchestrator `9704`, Python orchestrator `9706`, HF training `9707` on `macstudio` |
| Latest observed HF training status | `401/55000` episodes, `24060/3300000` env steps, `progress_pct=0.7290909091`, `device=cpu`, `status=running`, `torch_num_threads=12` |

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
| Добавлен `scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py`. | Нужна отдельная диагностика данных/цели до финального качества модели. |

## Текущий полный run

Первый полный run `stage08h_dual_branch_cpu_90_60_full_20260626T141800Z` был остановлен после пользовательского запроса использовать максимально доступное число CPU cores. Причина остановки: этот run был запущен с `--torch-num-threads 1`, фактически грузил около одного CPU core, а пользователь признал такой режим неприемлемо долгим для полного `08H`.

Новый полный run запущен на `macstudio` с `--torch-num-threads 12` и `--torch-num-interop-threads 1`:

| Поле | Значение |
|---|---|
| `run_id` | `stage08h_dual_branch_cpu_90_60_full_max_threads_20260626T211226Z` |
| `status` | `running` |
| `completed_episodes` | `401/55000` |
| `completed_env_steps` | `24060/3300000` |
| `progress_pct` | `0.7290909091` |
| `device` | `cpu` |
| `torch_num_threads` | `12` |
| `torch_num_interop_threads` | `1` |
| `latest_status_timestamp` | `2026-06-26T21:20:10Z` |
| `eta_sec` | `61928.6657257932` for the current HF-original training branch only |
| `main_pid` | `9704` |
| `python_orchestrator_pid` | `9706` |
| `child_training_pid` | `9707` |
| `summary_exists` | `false` for `stage08h_dual_branch_cpu_run_summary.json` |
| `roehub_native_branch_started` | `false` |

Пока этот run не завершится, `08H` не может быть `accepted`, а Stage `09` не может стартовать.

Proof boundary: это `target_host_non_production_training_pre_main` / offline ML runtime evidence на `macstudio`. Это не `post_main_production_runtime_proof`, не browser-visible proof и не проверка `/opt/roehub/app`. Для `post_main_production_runtime_proof` потребовались бы target revision on `main`, зеленые GitHub Actions/CI, deploy или verified sync из `main` checkout в `/opt/roehub/app`, а затем production smoke/API/browser verification по измененному production runtime.

Environment/comparability note: наблюдение выполнено на том же `macstudio` (`12` CPU cores: `8` performance + `4` efficiency), в том же `08H` HF-original `90/60` training stage, на том же HF dataset и с теми же основными hyperparameters; отличался `torch_num_threads` и restart state. Это не финальный benchmark acceptance: старый single-thread run был уже прогрет и находился на episode `2801 -> 2901`, новый max-thread run был свежим и наблюдался на episode `201 -> 301`, поэтому сравнение является directional runtime evidence, а не строгим proof ускорения.

Performance observation: max-thread режим был включен и наблюдался `5` минут. Новый training PID `9707` использовал около `185%` CPU, то есть больше одного core, но не насыщал все `12` cores. После начала learning updates хвост `201 -> 301` episodes занял примерно `194.145129625` seconds, около `0.515078` episode/sec. Последний наблюдавшийся single-thread хвост старого run `2801 -> 2901` episodes занимал примерно `163.728928584` seconds, около `0.610766` episode/sec. Вывод: max-thread default включен по пользовательскому решению, но 5-минутная проверка пока не доказывает ускорение; вероятен overhead на маленьком batch/PyTorch updates.

## Пауза до завершения обучения

Текущий этап зафиксирован как `08H in_progress`. Это не accepted verdict и не blocked verdict по качеству модели. Это handoff-состояние: диагностика данных завершена, полный `90/60` dual-branch CPU run запущен и живой, но финальные HF/Roehub-native результаты еще не существуют.

Следующий агент или следующий проход должен начинать не с перезапуска обучения, а с проверки уже идущего запуска:

1. Проверить, живы ли PID `9704`, `9706`, `9707`, либо завершился ли orchestrator.
2. Прочитать свежий HF-original `latest_status.json` и `progress.jsonl` под `/opt/roehub/state/rl_trading/training_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/hf_original/stage08c_hf_original_7bad309170f3ddf2749b/`.
3. Проверить, появился ли итоговый файл `/opt/roehub/state/rl_trading/evaluation_runs/stage08h_oracle_supervised_selector_reward_90_60_v1/dual_branch_runs/stage08h_dual_branch_cpu_90_60_full_max_threads_20260626T211226Z/stage08h_dual_branch_cpu_run_summary.json`.
4. Если HF-original training завершился, проверить candidate manifest, `best.pth`, `final.pth`, final status, количество эпизодов `55000/55000` и шагов `3300000/3300000`.
5. Проверить HF-original `Optuna`: должно быть `100/100` trials, профиль должен остаться `agent_history_len=90`, `agent_session_len=60`.
6. Проверить HF-original final holdout: PnL после costs, количество закрытых сделок, action distribution, skipped/filtered signals, сравнение с sanity baselines и `stage09_allowed` на уровне branch.
7. Дождаться и проверить Roehub-native training с тем же профилем `90/60`, без отката к `30/10`.
8. Проверить Roehub-native `Optuna`: `100/100` trials, тот же профиль, корректные overrides и сохраненный summary.
9. Проверить Roehub-native final holdout: PnL после costs, количество закрытых сделок, positive session ratio, monthly/volatility stability, action/filter distribution и сравнение с best sanity baseline.
10. Отдельно проверить, что итог не повторяет дефект `08G` с очень малым числом сделок (`19` и `3`) без объяснения. Малое число сделок допустимо только если оно явно объяснено calibration/result artifacts и не маскирует переобучение.
11. Сравнить финальные HF-original и Roehub-native результаты с уже готовой диагностикой `08H`: oracle opportunity, past-only supervised sanity, selector regimes и reward sparsity proxy.
12. Если Roehub-native final holdout не проходит quality gate, оставить Stage `09` закрытым и записать конкретный blocker по runtime manifests.
13. Если Roehub-native проходит gate, все равно проверить, что нет переобучения на calibration split, что baseline-сравнение честное, а `stage09_allowed=true` подтверждено итоговым summary.
14. Обновить этот `08H` report, общий ledger и общий план только по фактическим runtime artifacts.
15. Запустить документационные проверки, минимум `uv run python -m tools.docs.generate_docs_index --check` и `git diff --check`; для code changes дополнительно вернуть focused `ruff`, `pyright`, `pytest`.
16. Синхронизировать локальный `main`, `origin/main` и checkout на `macstudio`, если документы или код менялись.
17. Не трогать `/opt/roehub/app` и не заявлять `post_main_production_runtime_proof`: этот stage пишет offline ML artifacts под `/opt/roehub/state/rl_trading/`, а не меняет production service.

## File manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `scripts/rl_trading/stage08h_oracle_supervised_dataset_diagnostics.py` | - | - | Stage `08H` oracle/supervised/selector/reward diagnostics CLI. | `compatible-change` additive opt-in research CLI |
| `tests/unit/scripts/rl_trading/test_stage08h_oracle_supervised_dataset_diagnostics.py` | - | - | Focused tests for oracle and supervised sanity logic. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08h-oracle-supervised-selector-reward-90-60-research.md` | - | - | Stage `08H` in-progress report. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | Persist explicit training profile fields and validate profile bounds. | `compatible-change` artifact metadata fix; new config hashes for new runs |
| - | `scripts/rl_trading/stage08c_original_hf_full_training_run.py` | - | Add profile CLI parameters for HF training. | `compatible-change` additive opt-in CLI options |
| - | `scripts/rl_trading/stage08e_roehub_native_full_training_run.py` | - | Add profile CLI parameters for native training. | `compatible-change` additive opt-in CLI options |
| - | `scripts/rl_trading/stage08g_cpu_optuna_calibration.py` | - | Add profile-aware evaluation and stage labeling. | `compatible-change` additive opt-in CLI options |
| - | `scripts/rl_trading/stage08g_dual_branch_cpu_training_evaluation.py` | - | Add profile-aware orchestration and `08H` stage label. | `compatible-change` additive opt-in CLI options |
| - | `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | Lock payload profile fields. | `none` test-only |
| - | `tests/unit/scripts/rl_trading/test_stage08g_dual_branch_cpu_training_evaluation.py` | - | Cover `stage-label=08H` dry-run. | `none` test-only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | Insert Stage `08H` as current corrective research gate. | `compatible-change` docs/plan |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark `08H` current/in-progress and keep `09` blocked. | `compatible-change` docs/ledger |
