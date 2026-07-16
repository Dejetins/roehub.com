# Этап 02 — пользовательский выбор инструмента и onboarding

## Результат

Этап принят. Файловый `whitelist.csv` больше не участвует в runtime policy
профилей `base`, `trading` и `ml`. Каталог поддерживаемых рынков глобален и
metadata-only; пользовательский выбор хранится отдельно на уровне организации.

В PostgreSQL добавлены `market_data_instrument_selections`,
`market_data_instrument_selection_audit_events`,
`market_data_instrument_history_bounds` и
`market_data_catalog_refresh_state`. Применение фазы
`market-data-selections-0022` подтверждено durable marker.

`OrganizationEffectiveSelection` объединяет выбор организации и pin активной
стратегии. Удаление выбора остаётся идемпотентным и разрешённым при pin: сервис
не останавливает стратегию и не удаляет данные. `GlobalEffectiveCollectorSet`
читает только workers и объединяет effective set всех организаций; API не
раскрывает соседние selections или pins.

## Реальное доказательство

Локальный Docker `runtime smoke` выполнен в изолированном проекте
`roehub-stage02`, без производственных данных и без торговых операций.
Проверка [`02-market-data-selection-runtime-proof.json`](evidence/02-market-data-selection-runtime-proof.json)
зафиксировала `ready=true`:

- `ws_connected = 1`, `ws_messages_total = 754`, `insert_rows_total = 9`;
- возраст последней свечи — `20.09 s` при лимите `180 s`;
- `ws_errors_total`, `insert_errors_total`, `rest_fill_errors_total` и
  `scheduler_job_errors_total` равны `0`;
- выполнено `27` scheduler jobs.

В реальном браузере через `/settings#market_data` подтверждены поиск каталога,
`PUT` и `DELETE` selection через UI и ответы `200` для
`/api/market-data/markets`, `/api/market-data/catalog` и
`/api/market-data/selections/*`. После проверки в базе оставлен единственный
bounded selection `binance:futures:BTCUSDT` (`market_id=2`), а не полный
каталог из `5 805` доступных инструментов.

Для `BTCUSDT` интерфейс показывает фактическое покрытие `0.0%`; для символов
без подтверждённой нижней границы истории отображается `Unknown`, а не ложные
`100%`. Размер artefact остаётся `Unknown` до ручной публикации этапа `03`.

Мобильный real browser smoke на `390×844` подтвердил отсутствие горизонтального
переполнения таблицы, доступность кнопки `Remove` и семантику живого статуса
`role="status"`. Три ошибки консоли относятся к заранее не настроенным
account integrations (`limits`, scoped notifications и exchange connections);
новых ошибок маршрутов `market-data` после исправления нет.

## Исправления во время проверки

- UI использовал `/market-data/*` вместо proxy-маршрутов `/api/market-data/*`;
  исправлены три endpoint-атрибута и добавлена SSR-проверка.
- Отмена динамической WebSocket subscription могла оставить дочерний
  `socket.recv()` без join. Binance и Bybit теперь всегда отменяют и ожидают
  receive/stop tasks; регрессия покрыта тестами.
- На узком экране таблица имела ширину `412px`. Для viewport до `760px`
  применена фиксированная сетка пяти колонок и компактная кнопка действия.
- Исторические документы о CSV помечены как superseded; reference API описывает
  organization selection, coverage и фактический artifact inventory.

## Проверки

- `uv run pytest -q ...` — `118 passed`, `3` существующих предупреждения
  `httpx` о будущем изменении cookies API;
- `uv run ruff check ...` — успешно;
- `uv run pyright ...` — `0 errors, 0 warnings`;
- `node --check apps/web/dist/js/pages/market-data-settings.js` — успешно;
- `uv run python -m tools.release.generate_runtime_topology --check` — успешно;
- `uv run python -m tools.docs.generate_docs_index --check` — успешно;
- `docker compose ... config --quiet`, PostgreSQL migration marker, Docker
  readiness verifier, desktop/mobile real browser smoke и `git diff --check`
  — успешно.

Следующий разрешённый этап: `03`. Он ограничен одной ручной публикацией
`binance:futures:BTCUSDT`, фактическим измерением памяти/диска и явным решением
о расширении набора только после успешного proof.
