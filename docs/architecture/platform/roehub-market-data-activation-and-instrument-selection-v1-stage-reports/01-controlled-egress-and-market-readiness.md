# Этап 01 — контролируемый исходящий доступ и готовность рыночных данных

## Результат

Этап принят. Внутренняя сеть `roehub` сохранена с `internal: true`. Вторая сеть `market-data-egress` создаётся только в профилях `trading` и `ml` и подключается исключительно к `market-data-scheduler` и `market-data-ws`.

`market-data-ws` больше не завершает работу с пустым списком: он через ограниченный интервал перечитывает effective enabled set, создаёт subscriptions при появлении инструмента и безопасно заменяет их при изменении списка. Временный CSV применяется исключительно для изолированного Stage 01 proof и содержит один инструмент `binance:futures:BTCUSDT`; он не является целевой пользовательской политикой Stage 02.

## Реальное доказательство

Изолированный проект `roehub-stage01` запускался последовательно через Docker Desktop с `COMPOSE_PARALLEL_LIMIT=1`, свежими томами и без обращения к производственной среде. `market-data-ws` и `market-data-scheduler` получили обе сети; `api`, `clickhouse` и publisher остались только в `roehub`.

Проверка [`01-market-readiness-proof.json`](evidence/01-market-readiness-proof.json) зафиксировала:

- `ws_connected = 1`;
- `ws_messages_total = 169`;
- `insert_rows_total = 2`;
- свежесть последней canonical 1m-свечи — `13.33` секунды при SLA `180` секунд;
- `ws_errors_total`, `insert_errors_total`, `rest_fill_errors_total` и `scheduler_job_errors_total` равны `0`;
- `scheduler_job_runs_total = 3`: выполнены sync, enrich и funding metadata job.

Проверяющий инструмент `tools/release/verify_market_data_readiness.py` опрашивает метрики обоих сервисов и ClickHouse; простой HTTP liveness не засчитывается как готовность. Он принимает только соединение, сообщения, вставку, свежую свечу, успешные scheduler sync/enrichment jobs и нулевые релевантные ошибки.

## Ресурсы и ограничения

Во время успешной проверки потребление составило: scheduler `76.07 MiB / 512 MiB`, WS `76.15 MiB / 768 MiB`, ClickHouse `748.3 MiB / 2 GiB`, publisher в idle `283.1 MiB / 1 GiB`. Операции выполнялись последовательно; свободное место после proof — `155 GiB`.

Расширенный набор инструментов и расписание publisher не включались. Текущий scheduler funding metadata refresh всё ещё видит справочные записи, но не создаёт selections или historical backfill вне `BTCUSDT` Stage 01 proof. Полная замена файлового policy на организационные selections — следующий этап.

## Проверки

- `uv run pytest -q tests/unit/tools/test_verify_market_data_readiness.py tests/unit/tools/test_generate_runtime_topology.py tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py` — `9 passed`;
- `uv run ruff check ...` для изменённых runtime, verifier и tests — успешно;
- `uv run python -m tools.release.generate_runtime_topology --check` — успешно;
- `docker compose ... config --quiet` — успешно;
- `uv run python -m tools.release.verify_market_data_readiness ...` — успешно;
- `git diff --check` — успешно.

Следующий разрешённый этап: `02`.
