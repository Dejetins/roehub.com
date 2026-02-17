# Strategy Telegram notifier v1: best-effort adapter + notification policy

Документ фиксирует контракт STR-EPIC-05: как Strategy live-runner отправляет Telegram-уведомления по ключевым событиям без влияния на устойчивость pipeline.

## Цель

Обеспечить пользователю Telegram-уведомления по ключевым событиям стратегии (`signal`, `trade_open`, `trade_close`, `failed`) с best-effort семантикой: ошибки канала уведомлений не ломают обработку run и не роняют worker.

## Контекст

- В Strategy уже есть live-runner и realtime output в Redis Streams: `src/trading/contexts/strategy/application/services/live_runner.py`, `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`, `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`.
- В Identity уже есть модель Telegram binding: `identity_telegram_channels(user_id, chat_id, is_confirmed, confirmed_at)` в `migrations/postgres/0001_identity_v1.sql`.
- Для STR-EPIC-05 v1 нужно:
  - best-effort отправка;
  - `log-only` адаптер в dev/test;
  - реальная отправка в prod только при подтвержденном chat binding.
- Для anti-spam на этапе v1 достаточно debounce одинаковых ошибок.

## Scope

- Port `TelegramNotifier` в strategy context.
- ACL/port в identity для резолва подтвержденного `chat_id` по `user_id`.
- Notification policy:
  - какие `event_type` идут в Telegram (`signal`, `trade_open`, `trade_close`, `failed`);
  - debounce одинаковых ошибок `failed`.
- Два режима адаптера:
  - `log_only` (dev/test),
  - `telegram` (prod).
- Интеграция в live-runner как best-effort side effect после успешной доменной обработки события.
- Runtime config в `strategy_live_runner.yaml` для telegram notifier.
- Метрики, логи и unit-тесты на policy/adapters/wiring.

## Non-goals

- Сложные шаблоны сообщений и локализация.
- Guarantee delivery, exactly-once, producer retries с подтверждением доставки.
- Distributed debounce и межпроцессная синхронизация anti-spam.
- Новый Telegram onboarding/handshake flow для подтверждения chat binding.
- Расширение бизнес-логики генерации `signal`/`trade_*` событий.

## Ключевые решения

### 1) Точка интеграции: Strategy live-runner

Telegram notify вызывается из live-runner в момент формирования Strategy event, как дополнительный best-effort side effect.

Последствия:
- Минимальная задержка и простой operational-контур без отдельного процесса.
- Notifier не меняет доменную state machine run и не влияет на ack/read pipeline.

### 2) Фиксированный список notify `event_type` v1

В Telegram идут только события:
- `signal`
- `trade_open`
- `trade_close`
- `failed`

Другие типы событий в v1 не маршрутизируются в Telegram.

Последствия:
- Контракт предсказуем и легко тестируется.
- Расширение `event_type` требует явного обновления документа/контракта.

### 3) Chat binding через identity ACL, только `is_confirmed=true`

Strategy не читает identity-таблицы напрямую и не хранит собственный `chat_id`; вместо этого использует identity ACL/port и берет только подтвержденный binding.

Рекомендуемый read-contract:
- `find_confirmed_chat_id(user_id) -> chat_id | None`
- детерминистический выбор при многих строках: `ORDER BY confirmed_at DESC NULLS LAST, chat_id ASC LIMIT 1`.

Последствия:
- Сохраняется граница bounded contexts.
- Без подтвержденного binding отправки нет (skip + warning + метрика).

### 4) Адаптеры v1: `log_only` и `telegram`

- `log_only` адаптер (dev/test): пишет структурированное сообщение в лог и считает метрики.
- `telegram` адаптер (prod): отправляет сообщение в Telegram Bot API (`sendMessage`).

Последствия:
- Выполняется DoD: рабочий safe mode для dev/test.
- Prod получает реальную доставку при наличии подтвержденного binding.

### 5) Best-effort runtime + fail-fast конфигурация для prod

Runtime-ошибки notify-канала (ACL/HTTP/Telegram API timeout/serialization) не пробрасываются из notifier path и не прерывают iteration live-runner.

При этом:
- если `mode=telegram` и нет обязательной конфигурации (`TELEGRAM_BOT_TOKEN`, valid API base), worker не стартует (fail-fast).

Последствия:
- Устойчивость pipeline сохраняется.
- Конфигурационные ошибки ловятся на старте, а не в production traffic.

### 6) Anti-spam policy: debounce одинаковых `failed` ошибок

Минимальная политика v1:
- debounce только для `failed`;
- окно по умолчанию `600s` (10 минут);
- key debounce: `(user_id, strategy_id, normalized_error_text)`.

`normalized_error_text` определяется детерминированно (`trim + collapse spaces`).

Последствия:
- Повторы одной и той же ошибки не фладят чат.
- Политика process-local (in-memory), что принято для v1 single-instance worker.

### 7) Общий rate-limit v1 не вводится

Вне debounce для `failed` дополнительный глобальный rate-limit не применяется.

Последствия:
- Минимальная сложность v1.
- При росте `signal`/`trade_*` event volume возможен переход к rate-limit в v2.

### 8) Формат сообщения: короткий plain text EN

V1 не использует Markdown/HTML и локализацию; формат строки стабильный и однозначный.

Примеры:
- `FAILED | strategy_id=<...> | run_id=<...> | error=<...>`
- `SIGNAL | strategy_id=<...> | run_id=<...> | instrument=<...> | timeframe=<...> | signal=<...>`
- `TRADE OPEN | strategy_id=<...> | run_id=<...> | instrument=<...> | timeframe=<...> | side=<...> | price=<...>`
- `TRADE CLOSE | strategy_id=<...> | run_id=<...> | instrument=<...> | timeframe=<...> | side=<...> | price=<...>`

Последствия:
- Легкая поддержка и предсказуемый output.
- Миграция на templating/l10n возможна отдельным контрактом v2.

### 9) Runtime config: секция `strategy_live_runner.telegram`

Добавляется минимальный конфигурационный блок:

- `enabled: bool`
- `mode: "log_only" | "telegram"`
- `bot_token_env: "TELEGRAM_BOT_TOKEN"`
- `api_base_url: "https://api.telegram.org"`
- `send_timeout_s: float`
- `debounce_failed_seconds: int`

Политика по средам:
- dev/test: `mode=log_only`
- prod: `mode=telegram`

Последствия:
- Единая точка управления notifier-поведением.
- Контракт согласован с текущим подходом `strategy_live_runner` runtime config.

## Контракты и инварианты

- `TelegramNotifier` в strategy context работает по best-effort contract и не ломает pipeline.
- В Telegram маршрутизируются только `signal|trade_open|trade_close|failed`.
- `chat_id` используется только при подтвержденном identity binding (`is_confirmed=true`).
- При отсутствии/неподтвержденном binding происходит skip, warning log и increment метрик.
- Debounce применяется только для `failed` по ключу `(user_id, strategy_id, normalized_error_text)` в окне `600s` по умолчанию.
- В dev/test должен быть доступен `log_only` адаптер.
- В prod при `mode=telegram` и невалидной обязательной конфигурации включается fail-fast startup.
- Секреты (`TELEGRAM_BOT_TOKEN`) не логируются.
- Notify side effect не меняет состояние run и не влияет на правило `ack after processing`.

## Связанные файлы

- `docs/architecture/roadmap/milestone-3-epics-v1.md` - источник scope и DoD STR-EPIC-05.
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md` - контракт execution pipeline worker.
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md` - смежный контракт realtime events/metrics.
- `migrations/postgres/0001_identity_v1.sql` - schema identity telegram channels (`identity_telegram_channels`).
- `src/trading/contexts/strategy/application/services/live_runner.py` - точка интеграции notifier side effect.
- `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py` - расширение runtime config секцией `telegram`.
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` - wiring mode (`log_only|telegram`), fail-fast и метрики.
- `configs/dev/strategy_live_runner.yaml` - `telegram.mode=log_only`.
- `configs/test/strategy_live_runner.yaml` - `telegram.mode=log_only`.
- `configs/prod/strategy_live_runner.yaml` - `telegram.mode=telegram`.

Ожидаемые новые файлы в реализации:
- `src/trading/contexts/strategy/application/ports/telegram_notifier.py` - port и DTO notifier contract.
- `src/trading/contexts/strategy/application/services/telegram_notification_policy.py` - policy notify + debounce.
- `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/log_only_telegram_notifier.py` - log-only адаптер.
- `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/telegram_bot_api_notifier.py` - Telegram Bot API адаптер.
- `src/trading/contexts/strategy/adapters/outbound/acl/identity/confirmed_telegram_chat_binding_resolver.py` - ACL resolver confirmed chat binding.
- `tests/unit/contexts/strategy/**` - unit-тесты для policy/adapters/wiring.

## Как проверить

```bash
# запускать из корня repo
uv run ruff check .
uv run pyright
uv run pytest -q

# docs index update/check
uv run python -m tools.docs.generate_docs_index
uv run python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: process-local debounce сбросится при restart worker.
- Риск: без global rate-limit возможен высокий объем notify при burst `signal/trade_*` событий.
- Риск: transient Telegram API outages приводят к пропуску уведомлений (принимается best-effort моделью v1).
- Открытые вопросы: нет, все решения для v1 зафиксированы.
