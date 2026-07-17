# Production Dashboard Metrics Reference (RU)

Статус: рекомендованный production dashboard для Grafana (native runtime на `Mac Studio`).

Назначение документа:
- дать готовую структуру dashboard по контурам `worker/scheduler/backtest artifacts/ClickHouse/Redis/PostgreSQL/host`;
- использовать английские названия панелей (Grafana UI на английском);
- для каждой панели дать полный PromQL-запрос, тип визуализации, описание и интерпретацию.

Source of truth scrape-конфига:
- `infra/macos/prometheus/prometheus.prod.yml`

Связанные документы:
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`

## Как читать таблицы

- `Panel title (EN)` — рекомендуемое название панели в Grafana.
- `Metric name(s)` — базовые метрики, которые используются в запросе.
- `PromQL query (full)` — полный copy-paste запрос для Grafana Query editor.
- `Expected behavior` — как выглядит нормальный режим.
- `Deviation` — что считать отклонением и когда идти в логи.

## Dashboard structure (recommended rows)

1. `Overview`
2. `Worker`
3. `Scheduler`
4. `Backtest Artifacts`
5. `ClickHouse`
6. `Redis`
7. `PostgreSQL`
8. `Host`

## Row: Overview

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Jobs Up` | `up` | `sum by (job) (up{job="market-data-ws-worker"} or up{job="market-data-scheduler"} or up{job="backtest-artifact-publisher"} or up{job="clickhouse-exporter"} or up{job="postgres-exporter"} or up{job="redis-exporter"} or up{job="node-exporter"})` | `Bar gauge` | Состояние ключевых jobs | Для каждого job значение `1` | Любой `0` = сервис down или scrape failed |
| `Probe Success` | `probe_success` | `min by (job, instance) (probe_success{job="blackbox-http"} or probe_success{job="blackbox-tcp"})` | `Table` | Статус blackbox probes (HTTP/TCP) | Все строки `1` | `0` = endpoint недоступен или не проходит проверку |
| `Worker E2E Latency p95` | `ws_closed_to_insert_done_seconds_bucket` | `histogram_quantile(0.95, sum(rate(ws_closed_to_insert_done_seconds_bucket{job="market-data-ws-worker"}[5m])) by (le))` | `Stat` | p95 latency от закрытой свечи до записи в raw | Стабильная, обычно < 1s | Рост p95 = деградация ingestion path |
| `Scheduler Errors (15m)` | `scheduler_job_errors_total` | `sum(increase(scheduler_job_errors_total{job="market-data-scheduler"}[15m]))` | `Stat` | Ошибки scheduler за 15 минут | Обычно `0` | >0 = смотреть breakdown по `job` и логи scheduler |
| `Artifact Publish Success Age` | `backtest_artifact_publish_last_success_unixtime` | `clamp_min(time() - max(backtest_artifact_publish_last_success_unixtime{job="backtest-artifact-publisher"}), 0)` | `Stat` | Возраст последнего успешного publish артефактов | После запуска сервиса держится ниже окна freshness и сбрасывается после daily run | Рост выше ожидаемого окна = сервис не публикует новые success |
| `ClickHouse Inserted Rows/s` | `clickhouse_system_event_total{event="InsertedRows"}` | `sum(rate(clickhouse_system_event_total{job="clickhouse-exporter",event="InsertedRows"}[5m]))` | `Time series` | Скорость записи строк в ClickHouse | Положительная при живом ingestion | Падение к `0` при активном рынке = проблема pipeline/DB |
| `Host CPU Busy %` | `node_cpu_seconds_total` | `100 * (1 - avg(rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m])))` | `Gauge` | Общая загрузка CPU хоста | Умеренная, без длинных пиков | Длительно высокая загрузка = риск роста latency |

## Row: Worker

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `WS Connections` | `ws_connected` | `ws_connected{job="market-data-ws-worker"}` | `Stat` | Текущее число активных WS соединений | >0 | Длительный `0` = отвал upstream WS или reconnect loop |
| `WS Messages/s` | `ws_messages_total` | `sum(rate(ws_messages_total{job="market-data-ws-worker"}[1m]))` | `Time series` | Скорость входящих WS сообщений | Положительная при активном рынке | Резкое падение = входной поток деградировал |
| `WS Reconnects (15m)` | `ws_reconnects_total` | `increase(ws_reconnects_total{job="market-data-ws-worker"}[15m])` | `Bar gauge` | Переподключения WS за окно | Низкий фон | Всплески = сеть/endpoint нестабилен |
| `WS Errors (15m)` | `ws_errors_total` | `increase(ws_errors_total{job="market-data-ws-worker"}[15m])` | `Stat` | Ошибки обработки WS за окно | `0` или очень низко | >0 = разбор логов worker |
| `Insert Rows/s` | `insert_rows_total` | `sum(rate(insert_rows_total{job="market-data-ws-worker"}[5m]))` | `Time series` | Скорость вставки в raw | Стабильный рост при живом WS потоке | Падение к `0` = stop в ingestion |
| `Insert Errors (15m)` | `insert_errors_total` | `increase(insert_errors_total{job="market-data-ws-worker"}[15m])` | `Stat` | Ошибки insert за окно | `0` | >0 = проблема записи в ClickHouse/данных |
| `Insert Duration p95` | `insert_duration_seconds_bucket` | `histogram_quantile(0.95, sum(rate(insert_duration_seconds_bucket{job="market-data-ws-worker"}[5m])) by (le))` | `Time series` | p95 длительности insert batch | Стабильный профиль | Рост p95 = write bottleneck |
| `Closed->Insert Done p95` | `ws_closed_to_insert_done_seconds_bucket` | `histogram_quantile(0.95, sum(rate(ws_closed_to_insert_done_seconds_bucket{job="market-data-ws-worker"}[5m])) by (le))` | `Time series` | End-to-end p95 latency ingestion | Стабильная и предсказуемая | Рост p95 = деградация pipeline |
| `Redis Publish Errors (15m)` | `redis_publish_errors_total` | `increase(redis_publish_errors_total{job="market-data-ws-worker"}[15m])` | `Stat` | Ошибки публикации в Redis Streams | `0` | >0 = Redis connectivity/latency issue |
| `REST Fill Active` | `rest_fill_active` | `rest_fill_active{job="market-data-ws-worker"}` | `Gauge` | Кол-во активных REST fill задач | Колеблется около рабочей нормы | Длительно высокий уровень + ошибки = backlog |

## Row: Scheduler

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Job Runs by Job (15m)` | `scheduler_job_runs_total` | `sum by (job) (increase(scheduler_job_runs_total{job="market-data-scheduler"}[15m]))` | `Bar chart` | Частота запусков scheduler jobs | Соответствует расписанию | Просадка запусков = scheduler loop issue |
| `Job Errors by Job (15m)` | `scheduler_job_errors_total` | `sum by (job) (increase(scheduler_job_errors_total{job="market-data-scheduler"}[15m]))` | `Bar chart` | Ошибки jobs по label `job` | Обычно `0` | Рост конкретной job = точечная деградация |
| `Job Duration p95 by Job` | `scheduler_job_duration_seconds_bucket` | `histogram_quantile(0.95, sum by (job, le) (rate(scheduler_job_duration_seconds_bucket{job="market-data-scheduler"}[5m])))` | `Time series` | p95 длительности jobs | Стабильно, без хвостов | Скачки p95 = внешние задержки/нагрузка |
| `Tasks Planned by Reason (15m)` | `scheduler_tasks_planned_total` | `sum by (reason) (increase(scheduler_tasks_planned_total{job="market-data-scheduler"}[15m]))` | `Bar chart` | Планируемые задачи по `reason` | Профиль зависит от фазы работы | Всплеск без enqueue/progress = backlog |
| `Tasks Enqueued by Reason (15m)` | `scheduler_tasks_enqueued_total` | `sum by (reason) (increase(scheduler_tasks_enqueued_total{job="market-data-scheduler"}[15m]))` | `Bar chart` | Реально enqueued задачи | Близко к planned, но может быть ниже | Длительный ноль при planned>0 = issue в очереди |
| `Catchup Instruments by Status (15m)` | `scheduler_rest_catchup_instruments_total` | `sum by (status) (increase(scheduler_rest_catchup_instruments_total{job="market-data-scheduler"}[15m]))` | `Bar chart` | Статус обработки инструментов periodic catchup | Рост `status="ok"` | Рост `failed` = смотреть REST/DB/логи |
| `Gap Ranges Filled (15m)` | `scheduler_rest_catchup_gap_ranges_filled_total` | `increase(scheduler_rest_catchup_gap_ranges_filled_total{job="market-data-scheduler"}[15m])` | `Stat` | Закрытые gap-диапазоны | Растет при gap recovery | Длительный `0` при известных gaps = catchup stuck |
| `Gap Rows Written (15m)` | `scheduler_rest_catchup_gap_rows_written_total` | `increase(scheduler_rest_catchup_gap_rows_written_total{job="market-data-scheduler"}[15m])` | `Stat` | Записанные строки по gaps | Положительный рост при догрузке | Ноль при активном catchup = нет реального прогресса |

## Row: Backtest Artifacts

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Publisher Up` | `up` | `max(up{job="backtest-artifact-publisher"})` | `Stat` | Доступность `/metrics` long-running publisher service | `1` | `0` = сервис down или Prometheus не может scrape'ить `:9203` |
| `Publish Runs by Status (7d)` | `backtest_artifact_publish_runs_total` | `sum by (status) (increase(backtest_artifact_publish_runs_total{job="backtest-artifact-publisher"}[7d]))` | `Bar chart` | Итоговые scheduler runs по status | Основной рост у `status="succeeded"`; blocked/error остаются низкими | Рост `validation_failed`, `inactive_slot_pinned`, `lock_held` или `unexpected_error` = нужна разборка логов и lock state |
| `Publish Duration p95` | `backtest_artifact_publish_duration_seconds_bucket` | `histogram_quantile(0.95, sum by (le) (rate(backtest_artifact_publish_duration_seconds_bucket{job="backtest-artifact-publisher"}[30d])))` | `Stat` | p95 длительности полного daily publish-цикла | Стабильный профиль в пределах окна nightly обработки | Рост p95 = расширение universe, деградация ClickHouse/Postgres/FS или hidden retries |
| `Symbols by Status (7d)` | `backtest_artifact_publish_symbols_total` | `sum by (status) (increase(backtest_artifact_publish_symbols_total{job="backtest-artifact-publisher"}[7d]))` | `Bar chart` | Сколько symbol roots прошли через scheduler по итоговому status | Основная масса у `status="succeeded"`; `failed` и blocked-статусы остаются исключением | Рост degraded статусов = drift в universe, pinning или ошибки publish-пайплайна |
| `Blocked Runs by Reason (7d)` | `backtest_artifact_publish_blocked_total` | `sum by (reason) (increase(backtest_artifact_publish_blocked_total{job="backtest-artifact-publisher"}[7d]))` | `Bar chart` | Разбиение блокировок по конечным причинам | Обычно `0`; допустим единичный `lock_held` при ручном overlap | Рост `inactive_slot_pinned` = зависшие background runs; `validation_failed` = broken slot; `unexpected_error` = разбирать stacktrace |
| `Last Success Age` | `backtest_artifact_publish_last_success_unixtime` | `clamp_min(time() - max(backtest_artifact_publish_last_success_unixtime{job="backtest-artifact-publisher"}), 0)` | `Gauge` | Возраст последнего успешного publish в секундах | После `03:05 Europe/Moscow` метрика резко обновляется и остаётся ниже 30 часов | Длительное превышение freshness window = ночной publish не прошёл или сервис не жил в нужное окно |
| `Tail Rebuild Bars by Stage (7d)` | `backtest_artifact_tail_rebuild_bars_total` | `sum by (stage) (increase(backtest_artifact_tail_rebuild_bars_total{job="backtest-artifact-publisher"}[7d]))` | `Bar chart` | Сколько баров scheduler переписал в bounded tail по stage | Значения растут пропорционально universe и tail budgets | Резкий скачок = массовый full rebuild fallback или неожиданное расширение lookback budgets |

## Row: ClickHouse

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Exporter Scrape Success` | `clickhouse_exporter_scrape_success` | `clickhouse_exporter_scrape_success{job="clickhouse-exporter"}` | `Stat` | Успех последнего scrape exporter | `1` | `0` = exporter или ClickHouse недоступен |
| `Exporter Scrape Duration` | `clickhouse_exporter_scrape_duration_seconds` | `clickhouse_exporter_scrape_duration_seconds{job="clickhouse-exporter"}` | `Time series` | Длительность scrape exporter | Стабильная | Рост = замедление запросов exporter |
| `ClickHouse Uptime` | `clickhouse_uptime_seconds` | `clickhouse_uptime_seconds{job="clickhouse-exporter"}` | `Stat` | Uptime процесса ClickHouse | Растет монотонно | Сброс = рестарт ClickHouse |
| `Queries/s` | `clickhouse_system_event_total{event="Query"}` | `sum(rate(clickhouse_system_event_total{job="clickhouse-exporter",event="Query"}[5m]))` | `Time series` | Общая скорость запросов в CH | Соответствует профилю нагрузки | Резкое падение/рост = изменение нагрузки или инцидент |
| `Select Queries/s` | `clickhouse_system_event_total{event="SelectQuery"}` | `sum(rate(clickhouse_system_event_total{job="clickhouse-exporter",event="SelectQuery"}[5m]))` | `Time series` | Скорость SELECT запросов | Стабильный профиль | Аномальный рост = read pressure |
| `Inserted Rows/s` | `clickhouse_system_event_total{event="InsertedRows"}` | `sum(rate(clickhouse_system_event_total{job="clickhouse-exporter",event="InsertedRows"}[5m]))` | `Time series` | Скорость вставки строк | >0 при активном ingestion | Длительный `0` = ingestion break |
| `HTTP Connections` | `clickhouse_system_metric_value{metric="HTTPConnection"}` | `clickhouse_system_metric_value{job="clickhouse-exporter",metric="HTTPConnection"}` | `Time series` | Текущие HTTP подключения к CH | Умеренные колебания | Рост без стабилизации = connection pressure |
| `Native TCP Connections` | `clickhouse_system_metric_value{metric="TCPConnection"}` | `clickhouse_system_metric_value{job="clickhouse-exporter",metric="TCPConnection"}` | `Time series` | Текущие native TCP подключения | Умеренные колебания | Резкий рост/плато = client pressure |

## Row: Redis

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Redis Up` | `redis_up` | `redis_up{job="redis-exporter"}` | `Stat` | Доступность Redis | `1` | `0` = Redis недоступен |
| `Exporter Scrape Error` | `redis_exporter_last_scrape_error` | `redis_exporter_last_scrape_error{job="redis-exporter"}` | `Stat` | Ошибка последнего scrape | `0` | `1` = exporter не может получить метрики |
| `Commands/s` | `redis_commands_processed_total` | `rate(redis_commands_processed_total{job="redis-exporter"}[5m])` | `Time series` | Скорость обработки команд Redis | Положительная при рабочей нагрузке | Падение к `0` при активном воркере = канал/Redis проблема |
| `Connected Clients` | `redis_connected_clients` | `redis_connected_clients{job="redis-exporter"}` | `Time series` | Количество подключенных клиентов | Стабильный рабочий уровень | Резкий рост = утечки/переподключения |
| `Blocked Clients` | `redis_blocked_clients` | `redis_blocked_clients{job="redis-exporter"}` | `Stat` | Количество blocked clients | Близко к `0` | Длительный рост = блокировки/задержки |
| `Memory Used` | `redis_memory_used_bytes` | `redis_memory_used_bytes{job="redis-exporter"}` | `Time series` | Используемая память Redis | Рост в рамках capacity | Резкий рост = риск eviction/latency |
| `Memory Fragmentation Ratio` | `redis_mem_fragmentation_ratio` | `redis_mem_fragmentation_ratio{job="redis-exporter"}` | `Gauge` | Фрагментация памяти | Умеренный уровень | Стабильно высокий уровень = tuning/defrag |
| `Evicted Keys (15m)` | `redis_evicted_keys_total` | `increase(redis_evicted_keys_total{job="redis-exporter"}[15m])` | `Stat` | Вытесненные ключи за окно | Обычно `0` | >0 = давление по памяти |
| `Expired Keys/s` | `redis_expired_keys_total` | `rate(redis_expired_keys_total{job="redis-exporter"}[5m])` | `Time series` | Скорость TTL expire | Нормально зависит от модели данных | Неожиданный рост = аномальный churn |
| `Command Failures by CMD (15m)` | `redis_commands_failed_calls_total` | `topk(10, sum by (cmd) (increase(redis_commands_failed_calls_total{job="redis-exporter"}[15m])))` | `Table` | Ошибки команд по `cmd` | Обычно пусто/нули | Значимые значения = проблемные команды/лимиты |

## Row: PostgreSQL

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `Postgres Up` | `pg_up` | `pg_up{job="postgres-exporter"}` | `Stat` | Доступность PostgreSQL для exporter | `1` | `0` = БД недоступна |
| `Exporter Last Scrape Error` | `pg_exporter_last_scrape_error` | `pg_exporter_last_scrape_error{job="postgres-exporter"}` | `Stat` | Ошибка последнего scrape | `0` | `1` = exporter не может читать метрики |
| `Exporter Scrape Duration` | `pg_exporter_last_scrape_duration_seconds` | `pg_exporter_last_scrape_duration_seconds{job="postgres-exporter"}` | `Time series` | Длительность scrape exporter | Стабильно | Рост = тяжелые collector queries |
| `Collector Success` | `pg_scrape_collector_success` | `min by (collector) (pg_scrape_collector_success{job="postgres-exporter"})` | `Table` | Успех collector-ов exporter | Все `1` | `0` у collector = частичная деградация |
| `Connections by DB` | `pg_stat_database_numbackends` | `pg_stat_database_numbackends{job="postgres-exporter"}` | `Time series` | Текущее число backend connections по БД | В рабочем диапазоне | Рост к лимиту = риск отказов подключения |
| `Commits/s by DB` | `pg_stat_database_xact_commit` | `sum by (datname) (rate(pg_stat_database_xact_commit{job="postgres-exporter"}[5m]))` | `Time series` | Скорость commit транзакций | Коррелирует с бизнес-нагрузкой | Падение при активных сервисах = деградация доступа |
| `Rollbacks/s by DB` | `pg_stat_database_xact_rollback` | `sum by (datname) (rate(pg_stat_database_xact_rollback{job="postgres-exporter"}[5m]))` | `Time series` | Скорость rollback транзакций | Обычно низкая | Рост = ошибки бизнес-операций/конфликты |
| `Database Size by DB` | `pg_database_size_bytes` | `pg_database_size_bytes{job="postgres-exporter"}` | `Bar chart` | Размер баз данных | Плавный рост | Резкий рост = проверить retention/indexes |
| `Locks by Mode` | `pg_locks_count` | `sum by (mode) (pg_locks_count{job="postgres-exporter",datname="roehub"})` | `Bar chart` | Locks по режимам в целевой БД | Низкий стабильный фон | Рост contention = latency/timeouts риск |
| `Replication Lag` | `pg_replication_lag_seconds` | `pg_replication_lag_seconds{job="postgres-exporter"}` | `Stat` | Лаг репликации | Для single-node обычно `0` | Рост (если есть replica) = задержка репликации |

## Row: Host

| Panel title (EN) | Metric name(s) | PromQL query (full) | Recommended visualization | Description (RU) | Expected behavior | Deviation |
|---|---|---|---|---|---|---|
| `CPU Busy %` | `node_cpu_seconds_total` | `100 * (1 - avg(rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m])))` | `Gauge` | Общая загрузка CPU хоста | Без долгих пиков | Длительно высокий уровень = saturation |
| `Load 1m` | `node_load1` | `node_load1{job="node-exporter"}` | `Time series` | Краткосрочная системная нагрузка | Контролируемый профиль | Устойчиво высокий уровень = перегрузка |
| `Load 5m` | `node_load5` | `node_load5{job="node-exporter"}` | `Time series` | Средняя нагрузка за 5 минут | Контролируемый профиль | Длительный рост = накопление очередей |
| `Load 15m` | `node_load15` | `node_load15{job="node-exporter"}` | `Time series` | Долгосрочная нагрузка | Плавный профиль | Длительно высокий уровень = хроническая перегрузка |
| `Memory Free (bytes)` | `node_memory_free_bytes` | `node_memory_free_bytes{job="node-exporter"}` | `Time series` | Свободная память хоста | Достаточный запас | Длительное снижение = memory pressure |
| `Memory Free %` | `node_memory_free_bytes`, `node_memory_total_bytes` | `100 * node_memory_free_bytes{job="node-exporter"} / node_memory_total_bytes{job="node-exporter"}` | `Gauge` | Доля свободной памяти | Рабочий запас | Низкое значение длительно = риск OOM/latency |
| `Disk Free % (Data)` | `node_filesystem_avail_bytes`, `node_filesystem_size_bytes` | `100 * node_filesystem_avail_bytes{job="node-exporter",mountpoint="/System/Volumes/Data",fstype="apfs"} / node_filesystem_size_bytes{job="node-exporter",mountpoint="/System/Volumes/Data",fstype="apfs"}` | `Gauge` | Свободное место data volume | Есть запас по диску | Низкий процент = риск отказов записи/logging |
| `Disk Read MB/s` | `node_disk_read_bytes_total` | `sum(rate(node_disk_read_bytes_total{job="node-exporter",device=~"disk[0-9]+"}[5m])) / 1024 / 1024` | `Time series` | Скорость чтения диска | В рамках нормального профиля | Неожиданные пики = IO contention |
| `Disk Write MB/s` | `node_disk_written_bytes_total` | `sum(rate(node_disk_written_bytes_total{job="node-exporter",device=~"disk[0-9]+"}[5m])) / 1024 / 1024` | `Time series` | Скорость записи диска | В рамках нормального профиля | Пики + рост latency = storage bottleneck |
| `Network RX MB/s` | `node_network_receive_bytes_total` | `sum(rate(node_network_receive_bytes_total{job="node-exporter",device!="lo0"}[5m])) / 1024 / 1024` | `Time series` | Входящий сетевой трафик | Соответствует рабочей нагрузке | Аномальные пики/просадки = network issue |
| `Network TX MB/s` | `node_network_transmit_bytes_total` | `sum(rate(node_network_transmit_bytes_total{job="node-exporter",device!="lo0"}[5m])) / 1024 / 1024` | `Time series` | Исходящий сетевой трафик | Соответствует рабочей нагрузке | Аномальные пики/просадки = network issue |

## Grafana display recommendations

- Использовать английские названия панелей (как в колонке `Panel title (EN)`).
- Для `Stat/Gauge` настроить thresholds (green/yellow/red) и units (`s`, `ops/s`, `bytes`, `%`).
- Для latency histogram панелей показывать p95; p99 можно добавить отдельной серией.
- Для error panel использовать окна `15m` (как в таблицах) для одинаковой интерпретации.
- Для label-зависимых панелей (`by job/reason/status`) серия может появиться только после первого события.
- Для `Last Success Age` использовать unit `s` и threshold ниже 30 часов, чтобы отклонение совпадало с alert freshness window.

## Minimum done state for dashboard

Dashboard считается operationally useful, когда одновременно выполняется всё ниже:
- health панели (`Jobs Up`, `Probe Success`, `Up/Scrape Error`) показывают норму;
- worker/scheduler latency панели без системной деградации;
- backtest artifact publisher обновляет `Last Success Age` после nightly окна и не накапливает blocked/error series;
- throughput панели (`Insert Rows/s`, `Commands/s`, `Commits/s`) отражают ожидаемый профиль нагрузки;
- host ресурсы (`CPU`, `memory`, `disk`, `network`) не показывают saturation.
