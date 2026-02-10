# Market Data Metrics Reference (RU)

Подробный справочник метрик для:
- `market-data-ws-worker` (`http://<host>:9201/metrics`)
- `market-data-scheduler` (`http://<host>:9202/metrics`)

В production Prometheus опрашивает их по DNS внутри compose-сети:
- `http://market-data-ws-worker:9201/metrics`
- `http://market-data-scheduler:9202/metrics`

Документ отвечает на вопросы:
- что считает каждая метрика;
- когда метрика должна расти;
- какие аномалии это обычно означает;
- какие label значения ожидаемы.

## Общие правила интерпретации

- Метрики процесса живут в памяти процесса. После рестарта counters/histograms стартуют с нуля.
- `Counter` только растет в рамках одного процесса.
- `Gauge` может расти и уменьшаться.
- `Histogram` публикует серии:
  - `_bucket{le="..."}`
  - `_sum`
  - `_count`

## Worker: `market-data-ws-worker`

| Метрика | Тип | Labels | Что означает | Нормальное поведение |
|---|---|---|---|---|
| `ws_connected` | Gauge | - | Текущее число активных WS-соединений | Положительное число, меняется при reconnect |
| `ws_reconnects_total` | Counter | - | Кол-во переподключений WS | Редкий рост; бурный рост = проблемы сети/endpoint |
| `ws_messages_total` | Counter | - | Кол-во полученных WS-сообщений | Постоянный рост при нормальном рынке |
| `ws_errors_total` | Counter | - | Кол-во ошибок WS-обработки | Должен быть близко к 0; рост требует проверки логов |
| `ignored_non_closed_total` | Counter | - | Сообщения, отброшенные как non-closed kline | Рост ожидаем, это штатная фильтрация |
| `insert_rows_total` | Counter | - | Кол-во строк, записанных в raw | Постоянный рост при ingestion |
| `insert_batches_total` | Counter | - | Кол-во батчей вставки в raw | Рост вместе с `insert_rows_total` |
| `insert_duration_seconds` | Histogram | - | Длительность одного raw insert batch | p95/p99 должны оставаться стабильными |
| `insert_errors_total` | Counter | - | Ошибки вставки в raw | Норма: 0 или редкие единичные всплески |
| `ws_closed_to_insert_start_seconds` | Histogram | - | Латентность closed-candle -> начало insert | Основная pre-insert часть SLO |
| `ws_closed_to_insert_done_seconds` | Histogram | - | Латентность closed-candle -> успешный insert done | Основной SLO для EPIC 3 (p95 <= 1s локально) |
| `ws_out_of_order_total` | Counter | - | WS-свечи пришли с минутой меньше `last_seen` (out-of-order) | Допустим редкий рост |
| `ws_duplicates_total` | Counter | - | WS-дубли по минутам | Возможен редкий рост |
| `rest_fill_tasks_total` | Counter | - | Принятые в очередь REST fill задачи | Растет при gap/reconnect/bootstrap/tail |
| `rest_fill_active` | Gauge | - | Текущее число выполняющихся REST fill задач | Колеблется от 0 до `rest_concurrency_instruments` |
| `rest_fill_errors_total` | Counter | - | Ошибки выполнения REST fill задач | Норма: близко к 0 |
| `rest_fill_duration_seconds` | Histogram | - | Длительность REST fill задачи | Зависит от размера диапазона и лимитов API |
| `redis_publish_total` | Counter | - | Успешные публикации WS candle в Redis Streams | Должен стабильно расти при живом WS потоке |
| `redis_publish_errors_total` | Counter | - | Ошибки публикации в Redis Streams | Рост указывает на недоступность/ошибки Redis |
| `redis_publish_duplicates_total` | Counter | - | Дубликаты/нарушение монотонности Stream ID (XADD) | Допустим редкий рост, обработка best-effort |
| `redis_publish_duration_seconds` | Histogram | - | Длительность вызова publish (XADD) | Используется для контроля латентности live feed |

## Scheduler: `market-data-scheduler`

| Метрика | Тип | Labels | Что означает | Нормальное поведение |
|---|---|---|---|---|
| `scheduler_job_runs_total` | Counter | `job` | Кол-во запусков job | Растет по расписанию |
| `scheduler_job_errors_total` | Counter | `job` | Ошибки job | В идеале 0 |
| `scheduler_job_duration_seconds` | Histogram | `job` | Длительность job | Стабильная, без резких скачков |
| `scheduler_tasks_planned_total` | Counter | `reason` | Сколько fill-задач запланировано planner'ом | На старте обычно заметный рост |
| `scheduler_tasks_enqueued_total` | Counter | `reason` | Сколько из запланированных реально enqueued в queue | Обычно <= planned (из-за дедупликации) |
| `scheduler_startup_scan_instruments_total` | Counter | - | Сколько инструментов обработано startup scan | Растет на каждый startup scan |
| `scheduler_rest_catchup_instruments_total` | Counter | `status` | Сколько инструментов обработал periodic rest catchup | Рост по `status="ok"` в штатном режиме |
| `scheduler_rest_catchup_tail_minutes_total` | Counter | - | Сколько tail-минут суммарно запрошено periodic catchup | Растет на каждом S3-запуске |
| `scheduler_rest_catchup_tail_rows_written_total` | Counter | - | Сколько tail-строк реально записано periodic catchup | Обычно растет при отставании хвоста |
| `scheduler_rest_catchup_gap_days_scanned_total` | Counter | - | Сколько UTC-дней просканировано на gaps | Растет стабильно при каждом full scan |
| `scheduler_rest_catchup_gap_days_with_gaps_total` | Counter | - | Сколько дней найдено с gaps | Должен уменьшать темп роста после стабилизации |
| `scheduler_rest_catchup_gap_ranges_filled_total` | Counter | - | Сколько gap-диапазонов отправлено в fill | Рост в фазе восстановления истории |
| `scheduler_rest_catchup_gap_rows_written_total` | Counter | - | Сколько gap-строк записано | Ключевой индикатор, что дыры реально закрываются |

### `job` labels

- `sync_whitelist`
- `enrich`
- `startup_scan`
- `rest_insurance_catchup`

### `reason` labels (scheduler tasks)

- `scheduler_bootstrap` — canonical пустой, диапазон `[earliest, now_floor)`.
- `historical_backfill` — canonical начинается позже earliest, диапазон `[earliest, canonical_min)`.
- `scheduler_tail` — страховочный хвост `[max(canonical_max+1m, now_floor-lookback), now_floor)`.

### `status` labels (`scheduler_rest_catchup_instruments_total`)

- `ok` — инструмент успешно обработан `RestCatchUp1mUseCase`.
- `failed` — processing упал с исключением (смотреть логи).
- `skipped_no_seed` — canonical пустой; инструмент оставлен planner-ветке bootstrap.

## Быстрые проверки с `curl`

```bash
curl -fsS http://localhost:9201/metrics | rg "ws_|insert_|rest_fill_"
curl -fsS http://localhost:9202/metrics | rg "scheduler_(job_|tasks_|startup_scan_)"
curl -fsS http://localhost:9202/metrics | rg "scheduler_rest_catchup_"
curl -fsS http://localhost:9201/metrics | rg "redis_publish_"
```

Исторические задачи scheduler:

```bash
curl -fsS http://localhost:9202/metrics | rg "scheduler_tasks_(planned|enqueued)_total.*historical_backfill"
curl -fsS http://localhost:9202/metrics | rg "scheduler_rest_catchup_gap_(days|ranges|rows)"
```

Ошибка job:

```bash
curl -fsS http://localhost:9202/metrics | rg "scheduler_job_errors_total"
```

## PromQL для SLO и диагностики

Worker p95 closed->insert done (5m окно):

```promql
histogram_quantile(
  0.95,
  sum(rate(ws_closed_to_insert_done_seconds_bucket[5m])) by (le)
)
```

Worker p95 insert duration:

```promql
histogram_quantile(
  0.95,
  sum(rate(insert_duration_seconds_bucket[5m])) by (le)
)
```

Рост ошибок REST fill:

```promql
increase(rest_fill_errors_total[15m])
```

Рост ошибок Redis публикации:

```promql
increase(redis_publish_errors_total[15m])
```

Ошибки startup scan:

```promql
increase(scheduler_job_errors_total{job="startup_scan"}[1h])
```

## Типовые операционные сценарии

- Симптом: `first_ts` новых инструментов "свежий", истории нет.
- Смотрите:
  - `scheduler_tasks_planned_total{reason="historical_backfill"}`
  - `scheduler_tasks_enqueued_total{reason="historical_backfill"}`
  - `scheduler_job_errors_total{job="startup_scan"}` и логи scheduler.

- Симптом: много planned, мало enqueued.
- Обычно это дедуп задач в очереди (ожидаемо). Критично только если при этом прогресс в canonical не двигается.

- Симптом: высокий `rest_fill_active`, растет `rest_fill_duration_seconds`, истории догружается медленно.
- Проверьте лимиты API/ошибки REST и `rest_concurrency_instruments`.
