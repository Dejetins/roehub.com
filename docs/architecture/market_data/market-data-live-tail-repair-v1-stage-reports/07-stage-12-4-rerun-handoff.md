# Stage 07: Stage 12.4 Rerun Handoff

Статус: `accepted`.

Дата анализа: `2026-06-30`.

## Перед Стартом

Требования к пользователю до старта: `nothing`. Используются уже существующий SSH-доступ к `macstudio`, runtime-артефакты в `/opt/roehub/state/...` и репозиторные docs/ledgers. Секреты, cookies, DSN, exchange keys или provider payloads не требуются и не должны попадать в отчеты.

Проверка предыдущего stage: `market-data-live-tail-repair-v1-stage-ledger.md` маркирует Stage `06` как `accepted`; `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` маркирует Strategy Producer Stage `12.4` как `blocked`, а `12.5` как закрытый до `12.4 accepted`.

Граница доказательств: `post_main_production_runtime_proof`.

| Обязательное условие | Требование |
|---|---|
| `main` | Наблюдаемый revision должен быть доставлен в `origin/main`. |
| Green GitHub Actions / CI | Релевантные GitHub Actions / CI checks должны быть зелеными до runtime claim. |
| Deploy/sync | Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com` и runtime tree `/opt/roehub/app` должны быть синхронизированы с этим revision до сбора proof. |
| Runtime proof | Только после этих условий runtime artifacts из `/opt/roehub/state/...` могут использоваться как changed-code production proof. |

Текущий отчет не заявляет принятие нового runtime-кода; он классифицирует уже собранный rerun evidence.

## Candidate Artifact Повторного Прогона

| Поле | Значение |
|---|---|
| artifact root | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T012705Z-stage07-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| `latest_status.json` | есть; `status=passed`, `reason=completed_6h`, `elapsed_seconds=21600`, `snapshot_count=7` |
| `snapshots.jsonl` | есть; `7` snapshots: `start`, `hour_1`, `hour_2`, `hour_3`, `hour_4`, `hour_5`, `final` |
| runtime run id | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| selected profile | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| owner | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| instrument | `binance:spot:BTCUSDT` |

## Что Прошло

Candidate artifact дает сильное evidence по signal path:

| Поверхность | Доказательство |
|---|---|
| 6h elapsed | `elapsed_seconds=21600`, `reason=completed_6h`. |
| Active run | Каждый snapshot видел выбранный run в `running`, `last_error_present=false`, `active_selected_runs=1`. |
| Signal continuity | Final cumulative window содержит `359` processed candles, `359` unique `StrategySignal`, `359` unique `ExecutionSourceEvent`, `unlinked_signal_rows=0`. |
| Hourly continuity | Каждое окно от `hour_1` до `final` содержит `60` processed candles, `60` unique signals, `60` unique source events. |
| Dedupe | В каждом snapshot `duplicate_signal_ids=0`, `duplicate_run_bars=0`, `duplicate_source_idempotency_groups=0`. |
| DB latency | Final cumulative p99 `candle.bar_ts_close -> StrategySignal.created_at=3.123s`; final cumulative p99 `StrategySignal.created_at -> ExecutionSourceEvent.received_at=0.06462636000000001s`. |
| Redis | Candle pending `0`, candle lag `0`, execution pending `0`; retry stream остался `1`, DLQ остался `2`. |
| Unknown/mainnet | `unknown_order_count=0`; `mainnet_order_count=2` остался равен baseline count. |

## Бизнес-Смысл

6h rerun показывает, что пользовательский strategy loop может продолжать выпускать сигналы и строки `ExecutionSourceEvent` для выбранной Testnet-стратегии после Market Data live-tail repair-cycle. Это ключевой бизнес-результат перед closure: стратегии не должны молча останавливаться после короткого пропуска в хвосте свечей.

Оставшийся blocker связан с полнотой evidence, а не с поведением стратегии: без same-window process resource history и final browser/API proof операторы пока не могут считать прогон полноценным production readiness gate.

## Service-Call Coverage

| Поверхность | Покрытие |
|---|---|
| New code/service calls | `N/A`; этот Stage `07` report не реализует новый runtime code и не вводит новые service calls. |
| Runtime reads | Существующие Mac Studio runtime artifacts, health endpoints, Redis, Postgres, Prometheus и Monit используются как read-only proof surfaces. |
| External providers | `N/A`; Stage `07` не должен отправлять orders или вызывать Binance/Bybit только ради acceptance soak. |
| Secrets / credentials | `N/A`; для этого отчета секреты не нужны. Если следующий executor собирает authenticated browser/API proof, он обязан использовать host-local smoke password source и редактировать credentials. |
| Alerts / monitoring / runbook | Monitoring surface проверена read-only через `strategy-producer` `/metrics`, `market-data-ws-worker` `/metrics`, Prometheus range queries и repair audit aggregate. Новых alert rules или runbook edits нет; operator runbook остается `docs/runbooks/market-data-live-tail-repair.md`. |
| Retry / unknown-state behavior | `N/A` для этой docs-only classification; если следующий executor повторяет collector, retry/DLQ/unknown deltas должны оставаться в thresholds prompt `12.4`. |

## Evidence Closure Recheck

Повторная read-only проверка `2026-06-30` не нашла надежный historical source, который мог бы восстановить обязательные process rows для того же 6h window `2026-06-30T01:27:05Z..07:27:05Z`.

| Поверхность | Проверенный факт | Решение |
|---|---|---|
| Candidate files | На `macstudio` в artifact directory есть только `collector.out`, `collector.py`, `baseline.json`, `snapshots.jsonl`, `latest_status.json`, `collector.pid`, `launcher.pid`; `collector.out` пустой. | Дополнительного process/browser proof artifact рядом с candidate нет. |
| Collector defect | `collector.py` собирал процессы через broad `ps -axo ...`, а helper `run_cmd` обрезал stdout до `4000` символов до парсинга. | `processes=[]` правдоподобно объясняется truncation/parsing bug; сам collector не валидировал non-empty process rows, поэтому `status=passed` нельзя считать acceptance. |
| Same-window Prometheus process history | Для окна `01:27:05Z..07:27:05Z` не найдено `process_cpu_seconds_total` / `process_resident_memory_bytes` для `job="strategy-producer"` и `job="exchange-execution"`; доступны только частичные exporter/service series вроде Redis memory/CPU, Prometheus CPU и Postgres exporter CPU/up. | Это не эквивалент обязательного process CPU/RSS evidence для `strategy_live_runner`, `exchange_execution`, Redis, Postgres и Prometheus. |
| Current process check | Текущий exact `pgrep -f` / `ps -p` показывает живые процессы `strategy_live_runner`, `exchange_execution`, Redis, Postgres и Prometheus. | Это доказывает, что сервисы есть сейчас, но не восстанавливает process history candidate window. |
| Repair metrics endpoint | `strategy-producer` `/metrics` экспонирует HELP/TYPE для `market_data_live_tail_gap_total`, `market_data_live_tail_repair_total`, `market_data_live_tail_repair_latency_seconds`, `strategy_live_runner_checkpoint_stall_total`, `strategy_live_runner_deferred_ack_total`; hot-cache metrics и `market_data_clickhouse_repair_circuit_state 0.0` доступны. `market-data-ws-worker` экспонирует `market_data_hot_cache_*`. | Repair observability surface доступна; отсутствие repair/stall/deferred-ACK samples в candidate window трактуется как `0` событий, а не как отсутствующий endpoint. |
| Repair audit query path | `public.market_data_candle_repair_events` доступна через runtime DB; aggregate `241` rows, status counts `miss=240`, `succeeded=1`; `window_count=0` для `2026-06-30T01:27:05Z..07:27:05Z`. | Audit surface доступна; во время candidate soak repair events не зафиксированы. |

## Пробелы В Evidence

Candidate artifact пока нельзя использовать для открытия `12.5`, потому что в artifact/report set есть не все обязательные acceptance surfaces Stage `12.4`.

| Gap | Наблюдаемый факт | Влияние | Что нужно закрыть |
|---|---|---|---|
| Process CPU/RSS snapshots | В каждом JSON snapshot записано `processes=[]`; same-window Prometheus не содержит process CPU/RSS для `strategy-producer` и `exchange-execution`; рядом нет другого resource artifact. | Stage `12.4` требует CPU/RAM/process RSS evidence на каждом snapshot. Пустые process rows не принимаются как resource evidence. | Rerun `12.4` с исправленным process collector, который использует точные `pgrep -f` / `ps -p` queries и валидирует non-empty rows. |
| Browser/API proof | Artifact directory содержит только `collector.out`, `collector.py`, `baseline.json`, `snapshots.jsonl`, `latest_status.json`, `collector.pid`, `launcher.pid`. Browser/API proof file отсутствует. | User-visible состояние `/strategies` не доказано для final accepted state. | Собрать final `/strategies` browser/API proof или явно перенести browser на `12.5` только при полном API evidence и non-safety reason. |
| Repair metrics/audit availability | Candidate snapshots не фиксировали repair metrics/audit, но повторная read-only проверка доказала endpoint/audit availability; в candidate window repair events `0`. | Не блокирует acceptance само по себе после этого report update, но должно попасть в следующий accepted rerun report как штатная surface. | В новом 6h rerun включить эти checks в collector/report, чтобы acceptance не зависела от post-fact ручной проверки. |
| Stage reports/ledgers | Repo reports все еще описывали старый blocked `20260626T234757Z-6h` attempt; Stage `07` report до этой correction отсутствовал. | Downstream executors могли бы стартовать `12.5` из chat memory вместо durable docs. | Обновить Strategy Producer `12.4` report, оба ledgers и docs index до любого stage advancement. |

## Промежуточное Решение По Старому Candidate

Промежуточное решение по старому candidate artifact: `blocked`.

Это не отменяет 6h rerun. Candidate artifact нужно сохранить как валидное signal-path evidence, но `latest_status.json.status=passed` недостаточно для acceptance Strategy Producer Stage `12.4`, потому что same-window resource/process evidence отсутствует и final browser/API proof не приложен. Repair observability surface после повторной проверки доступна, но должна быть встроена в следующий accepted rerun report.

На этом промежуточном шаге Stage `12.5` оставался закрытым.

Publish/delivery status для этого промежуточного шага: `N/A` для acceptance, потому что старый candidate artifact не закрывал Stage `07`; этот docs-only handoff не заявлял accepted-stage delivery, CI/deploy или runtime sync.

## Следующее Действие После Старого Candidate

Запускать обновленный prompt:

```text
.codex/agents/generated/market-data-live-tail-repair-v1/07-stage-12-4-rerun-handoff.md
```

Executor должен rerun Stage `12.4` с исправленным process collector. Старый candidate можно использовать как historical signal-path evidence, но не как acceptance. Перед новым таймером нужно сохранить repair metric/audit checks в collector/report и после финала собрать `/strategies` browser/API proof.

## Новый Fixed-Collector Rerun

Стартовая запись: `2026-06-30T16:20:58Z`.

User required before start: `nothing`. Preflight на `macstudio` подтвердил, что Strategy Producer Stage `12.3 accepted`, `12.4 blocked`, `12.5` закрыт до `12.4 accepted`; Market Data Stage `06 accepted`, а Stage `07` должен идти по ветке fixed collector rerun.

Новый artifact root:

```text
/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757
```

Collector launch: `2026-06-30T16:30:17Z`; planned final snapshot: `2026-06-30T22:30:17Z`.

Collector requirements for this rerun:

| Surface | Requirement |
|---|---|
| process evidence | exact `pgrep -f` / `ps -p` rows for `strategy_live_runner`, `exchange_execution`, Redis, Postgres and Prometheus; empty required tags block the run |
| signal path | per-window and cumulative DB p50/p95/p99/max plus duplicate counters |
| repair observability | `strategy-producer` repair/stall/deferred-ACK metric families, `market-data-ws-worker` hot-cache metrics, and `public.market_data_candle_repair_events` aggregate/window counts |
| final proof | final `/strategies` browser/API proof before any `12.4 accepted` decision |

## Итог Fixed-Collector Rerun

Итоговое решение по Stage `07`: `accepted`.

Повторный 6-часовой прогон с исправленным collector закрыл именно те пробелы, которые блокировали старый candidate: process rows больше не пустые, collector валидировал required process tags fail-closed, repair metric/audit checks стали частью snapshot/report evidence, а финальное `/strategies` browser/API proof собрано и скопировано в artifact root.

| Поле | Значение |
|---|---|
| accepted artifact root | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| `latest_status.json` | `status=passed`, `phase=completed_6h`, `elapsed_seconds=21600`, `snapshot_count=7` |
| window | `2026-06-30T16:30:17.687519Z` .. `2026-06-30T22:30:17.968239Z` |
| selected run | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` |
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| final browser/API proof | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/browser_api_proof.json` |
| final browser screenshot | `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/strategies-final-selected-run.png` |

Accepted evidence:

| Поверхность | Доказательство |
|---|---|
| 6h elapsed | `21600s`, `7` snapshots: `start`, `hour_1`, `hour_2`, `hour_3`, `hour_4`, `hour_5`, `final`. |
| Active run | Каждый acceptance snapshot видел выбранный run `running`, без `last_error_present`. |
| Signal continuity | Final cumulative window содержит `360` processed candles, `360` unique `StrategySignal`, `360` unique `ExecutionSourceEvent`, `unlinked_signal_rows=0`. |
| Hourly continuity | Каждое часовое окно от `hour_1` до `final` содержит `60` processed candles, `60` unique signals, `60` unique source events. |
| Dedupe | Final duplicate counters `0/0/0` для `signal_id`, `(strategy_run_id, bar_ts_open)` и source-event idempotency groups. |
| DB latency | Cumulative p99 `candle.bar_ts_close -> StrategySignal.created_at=3.14769s`; cumulative p99 `StrategySignal.created_at -> ExecutionSourceEvent.received_at=0.064028s`. |
| Process evidence | Final exact process rows: `strategy_live_runner=1`, `exchange_execution=1`, Redis `1`, Postgres `10`, Prometheus `1`; required tags были non-empty во всех snapshots. |
| Redis / safety counters | Candle pending/lag `0/0`; execution pending `0`; retry/DLQ без роста; unknown orders delta `0`; mainnet orders delta `0`. |
| Repair observability | `strategy-producer` live-tail repair/stall/deferred-ACK metric families доступны; `market-data-ws-worker` hot-cache metrics доступны; audit aggregate `miss=240`, `succeeded=1`; fixed-rerun window repair events `0`. |
| Browser/API proof | `/strategies` selected strategy `live`, producer `running`, selected run `running`, dashboard API `200`, console errors `0`, request failures `0`, observed latency gap `0s`. |

Business decision: исходная проблема live-tail repair остается закрытой Stage `06`; Stage `07` теперь подтвердил, что после repair выбранный Strategy Producer `12.4` проходит sustained 6h active strategy runtime gate с process/resource evidence и user-visible `/strategies` proof. Поэтому Strategy Producer Stage `12.4` может быть marked `accepted`, а Stage `12.5` можно открыть.

Service-call coverage для accepted rerun: новый repo runtime code не добавлялся, Binance/Bybit orders ради acceptance не отправлялись, mainnet delta `0`, unknown delta `0`, credentials не записывались. Browser/API proof использовал host-local smoke credential source без вывода секрета.

Known non-blocking residuals: финальный dashboard API все еще содержит отдельные unavailable/not-migrated панели вроде `strategy_paper_accounting`, chart/stat/fills/events и stale exchange account projection. Это не blocker Stage `07` / `12.4`, потому что acceptance surface здесь - 6h signal/source continuity, process/resource snapshots, Redis/DB safety, repair observability и `/strategies` active runtime proof. Полная задержка до testnet order ACK остается отдельной последующей задачей после `12.4` / `12.5`.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/07-stage-12-4-rerun-handoff.md` | none | none | Stage `07` candidate classification, fixed-collector rerun evidence, browser/API proof summary, and accepted handoff to Strategy Producer `12.5`. | `none`; documentation/evidence only. |
| none | `.codex/agents/generated/market-data-live-tail-repair-v1/07-stage-12-4-rerun-handoff.md` | none | Prompt теперь валидирует existing artifact directories до rerun 6h и требует process/browser/repair evidence. | `none`; prompt artifact only. |
| none | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-4-sustained-6h-soak.md` | none | Усилить acceptance `12.4` для repair-metrics и non-empty process evidence. | `none`; prompt artifact only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` | none | Синхронизировать plan с Stage `12.4` evidence closure rule. | `none`; architecture docs sync. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-4-sustained-6h-soak.md` | none | Добавить fixed-collector rerun evidence и отметить Strategy Producer Stage `12.4` как `accepted`. | `none`; stage report update. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Открыть Strategy Producer Stage `12.5` после accepted `12.4`. | `none`; ledger/handoff. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Зафиксировать repair Stage `07` как `accepted` после fixed-collector rerun. | `none`; ledger/handoff. |
| none | `docs/architecture/README.md` if generated index changes | none | Docs index после добавления этого report. | `none`; generated docs index only. |
