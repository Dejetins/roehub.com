# Strategy live worker runbook

Runbook для `apps/worker/strategy_live_runner`: как поднять worker, проверить метрики, проверить Redis Streams realtime output и (опционально) Telegram notify.

## 1) Предусловия

Нужны сервисы:

- Postgres (strategy storage + identity tables)
- ClickHouse (canonical candles 1m)
- Redis (market data streams + strategy realtime output streams)
- Market data WS worker (чтобы в Redis появлялись live свечи)
- Roehub API (чтобы создать стратегию и запустить run)

Документация и контракты:

- `docs/architecture/strategy/strategy-runtime-config-v1.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md`
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md`

## 2) Рекомендуемая модель запуска (единый compose стек)

Базовый стек уже описан в `infra/docker/docker-compose.yml` (postgres/clickhouse/redis/market-data/prometheus).

Для smoke STR-EPIC-06 рекомендуется добавить 2 сервиса в тот же compose (пример):

```yaml
  api:
    build:
      context: ${MARKET_DATA_BUILD_CONTEXT:-../..}
      dockerfile: ${MARKET_DATA_DOCKERFILE:-infra/docker/Dockerfile.market_data}
    restart: unless-stopped
    depends_on:
      - postgres
      - clickhouse
    environment:
      ROEHUB_ENV: ${ROEHUB_ENV:-prod}
      STRATEGY_PG_DSN: ${STRATEGY_PG_DSN}
      IDENTITY_PG_DSN: ${IDENTITY_PG_DSN}
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:-}
      # + identity secrets (см. identity wiring), если в prod fail-fast включен
    command: ["python", "-m", "apps.api.main.main", "--host", "0.0.0.0", "--port", "8000"]
    ports:
      - "127.0.0.1:8000:8000"

  strategy-live-worker:
    build:
      context: ${MARKET_DATA_BUILD_CONTEXT:-../..}
      dockerfile: ${MARKET_DATA_DOCKERFILE:-infra/docker/Dockerfile.market_data}
    restart: unless-stopped
    depends_on:
      - postgres
      - clickhouse
      - redis
    environment:
      ROEHUB_ENV: ${ROEHUB_ENV:-prod}
      STRATEGY_PG_DSN: ${STRATEGY_PG_DSN}
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:-}
      ROEHUB_REDIS_PASSWORD: ${ROEHUB_REDIS_PASSWORD:-}
      CH_HOST: clickhouse
      CH_PORT: 8123
      CH_DATABASE: ${CH_DATABASE:-market_data}
      CH_USER: ${CLICKHOUSE_USER:-default}
      CH_PASSWORD: ${CLICKHOUSE_PASSWORD:-}
      CH_SECURE: "0"
      CH_VERIFY: "1"
    command:
      - python
      - -m
      - apps.worker.strategy_live_runner.main.main
      - --config
      - /app/configs/prod/strategy_live_runner.yaml
      - --metrics-port
      - "9203"
    expose:
      - "9203"
```

Примечания:

- Используется тот же образ, что и для market-data (`infra/docker/Dockerfile.market_data`) — в нем есть `src/`, `apps/`, `configs/`.
- В конфигурации STR-EPIC-06 source-of-truth — `configs/prod/strategy.yaml`, а `configs/prod/strategy_live_runner.yaml` остается как shim/alias.

## 3) Миграции/DDL

### Postgres

Identity таблицы в repo представлены как legacy baseline SQL (могут быть уже применены в prod):

```bash
psql "$IDENTITY_PG_DSN" -f migrations/postgres/0001_identity_v1.sql
psql "$IDENTITY_PG_DSN" -f migrations/postgres/0002_identity_2fa_totp_v1.sql
psql "$IDENTITY_PG_DSN" -f migrations/postgres/0004_identity_exchange_keys_v2.sql
```

Strategy схемы применяются через alembic runner:

```bash
export POSTGRES_DSN="$STRATEGY_PG_DSN"
uv run python -m apps.migrations.main
```

### ClickHouse

Market data DDL (если еще не применен):

```bash
clickhouse-client --host 127.0.0.1 --port 9000 \
  --user "${CLICKHOUSE_USER:-default}" --password "${CLICKHOUSE_PASSWORD:-}" \
  --multiquery < migrations/clickhouse/market_data_ddl.sql
```

## 4) Запуск и логи

### Запуск через compose

```bash
docker compose -f infra/docker/docker-compose.yml up -d --build
docker compose -f infra/docker/docker-compose.yml ps
```

Логи:

```bash
docker compose -f infra/docker/docker-compose.yml logs -f --tail=200 strategy-live-worker
docker compose -f infra/docker/docker-compose.yml logs -f --tail=200 api
```

### Запуск вне docker (только если Redis доступен с хоста)

Если запускаете worker на хосте, Redis должен быть доступен по `localhost:6379` (или через port mapping).
В этом случае используйте `configs/dev/strategy.yaml` (и shim `configs/dev/strategy_live_runner.yaml`, если entrypoint еще смотрит туда).

## 5) Проверка метрик (/metrics)

Strategy live worker поднимает Prometheus endpoint.

Проверка с хоста:

```bash
curl -fsS http://localhost:9203/metrics | head
```

Полезные метрики:

- `strategy_live_runner_iterations_total`
- `strategy_live_runner_iteration_errors_total`
- `strategy_live_runner_messages_read_total`
- `strategy_live_runner_messages_acked_total`
- `strategy_realtime_output_publish_total`
- `strategy_realtime_output_publish_errors_total`
- `strategy_telegram_notify_total`
- `strategy_telegram_notify_errors_total`
- `strategy_telegram_notify_skipped_total`

Проверка scrape из контейнера Prometheus (DNS внутри сети):

```bash
docker exec -it prometheus wget -T 2 -qO- http://strategy-live-worker:9203/metrics | head
```

## 6) Проверка Redis Streams

### Live market data candles

Market-data WS worker публикует свечи в streams:

- `md.candles.1m.<instrument_key>`

Пример:

```bash
docker exec -it redis redis-cli XLEN md.candles.1m.binance:spot:BTCUSDT
docker exec -it redis redis-cli XREVRANGE md.candles.1m.binance:spot:BTCUSDT + - COUNT 3
```

### Strategy realtime output

Strategy live worker публикует per-user streams:

- metrics: `strategy.metrics.v1.user.<user_id>`
- events: `strategy.events.v1.user.<user_id>`

Чтобы узнать `user_id`, используйте identity endpoint `/auth/current-user` (см. smoke ниже).

Пример проверки:

```bash
docker exec -it redis redis-cli XLEN strategy.events.v1.user.00000000-0000-0000-0000-000000001111
docker exec -it redis redis-cli XREVRANGE strategy.events.v1.user.00000000-0000-0000-0000-000000001111 + - COUNT 5
```

## 7) Smoke сценарий (DoD)

Цель: "поднять стек → создать стратегию → запустить run → увидеть metrics/event в Redis → получить telegram notify (или log-only)".

### Шаг 1: поднять стек

Поднимите compose (postgres/clickhouse/redis/market-data/api/strategy-live-worker) и убедитесь что все сервисы в `running`.

### Шаг 2: Telegram login (вариант A — Login Widget)

В v1 identity flow зафиксирован как Telegram Login Widget (Variant A).

Для smoke удобно сгенерировать валидный payload локально (нужен `TELEGRAM_BOT_TOKEN`):

```bash
export TELEGRAM_BOT_TOKEN="..."

python - <<'PY'
import hashlib
import hmac
import json
import os
import time

bot_token = os.environ["TELEGRAM_BOT_TOKEN"]

payload = {
  "id": "411001",
  "auth_date": str(int(time.time())),
  "first_name": "Roe",
  "username": "roehub_test",
}

data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
payload["hash"] = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
print(json.dumps(payload))
PY
```

Логин + сохранение cookie:

```bash
PAYLOAD_JSON="$(python - <<'PY'
import hashlib, hmac, json, os, time
bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
payload = {"id":"411001","auth_date":str(int(time.time())),"first_name":"Roe","username":"roehub_test"}
dcs = "\n".join(f"{k}={v}" for k, v in sorted(payload.items()))
secret = hashlib.sha256(bot_token.encode()).digest()
payload["hash"] = hmac.new(secret, dcs.encode(), hashlib.sha256).hexdigest()
print(json.dumps(payload))
PY
)"

curl -fsS -c cookies.txt \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD_JSON" \
  http://localhost:8000/auth/telegram/login
```

Узнать `user_id`:

```bash
curl -fsS -b cookies.txt http://localhost:8000/auth/current-user
```

### Шаг 3: создать стратегию

Пример payload (из контрактных тестов Strategy API):

```bash
curl -fsS -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{
    "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
    "instrument_key": "binance:spot:BTCUSDT",
    "market_type": "spot",
    "timeframe": "1m",
    "indicators": [{"name": "MA", "params": {"fast": 20, "slow": 50}}],
    "signal_template": "MA(20,50)"
  }' \
  http://localhost:8000/strategies
```

Сохраните `strategy_id` из ответа.

### Шаг 4: запустить run

```bash
curl -fsS -b cookies.txt -X POST http://localhost:8000/strategies/<strategy_id>/run
```

### Шаг 5: убедиться что live worker работает

- Метрики `/metrics` доступны и `strategy_live_runner_iterations_total` растет.
- В Redis есть `md.candles.1m.binance:spot:BTCUSDT` и live worker читает сообщения.
- В Redis появляются записи в `strategy.events.v1.user.<user_id>` и/или `strategy.metrics.v1.user.<user_id>`.

### Шаг 6: Telegram notify (или log-only)

- Dev/test: ожидается `log_only` режим — уведомления видны в логах `strategy-live-worker`.
- Prod: ожидается реальная отправка в Telegram, но только если есть **подтвержденный** chat binding в identity.

Если flow подтверждения chat_id еще не автоматизирован, для smoke можно вручную подтвердить binding в Postgres:

```sql
-- выполнить через psql по IDENTITY_PG_DSN
INSERT INTO identity_telegram_channels (user_id, chat_id, is_confirmed, confirmed_at)
VALUES ('<user_id>', <chat_id>, TRUE, now())
;
```

После этого следующий `failed/signal/trade_*` event в стратегии должен приводить к notify.

## 8) Troubleshooting

### Симптом: worker не видит Redis / DNS `redis` не резолвится

Причина:
- worker запущен на хосте, но Redis не опубликован наружу или конфиг указывает `host=redis`.

Решения:
- запускать worker внутри compose-сети (как отдельный сервис),
- или опубликовать Redis порт на хост и использовать `host=localhost` в конфиге.

### Симптом: нет событий/метрик Strategy в Redis

Проверки:
- `strategy.realtime_output.redis_streams.enabled=true` и live worker реально использует Redis publisher.
- `instrument_key` в стратегии совпадает со stream-именем market-data (`md.candles.1m.<instrument_key>`).
- market-data WS worker реально пишет live свечи (см. `docs/runbooks/market-data-redis-streams.md`).

### Симптом: Telegram notify всегда "skipped"

Проверки:
- `strategy.telegram.enabled=true`.
- `identity_telegram_channels` содержит строку с `user_id`, `chat_id` и `is_confirmed=true`.
- В режиме `telegram` задан `TELEGRAM_BOT_TOKEN`.
