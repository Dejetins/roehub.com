# Runbook — Market Data Redis Streams

## Назначение

Операционные команды для live feed stream'ов, которые публикует `market-data-ws-worker`.

Шаблон stream:

- `md.candles.1m.<instrument_key>`

Пример stream:

- `md.candles.1m.binance:spot:BTCUSDT`

## Предусловия (native runtime)

Redis должен быть запущен как host service:

```bash
brew services list | grep redis
redis-cli -h 127.0.0.1 -p 6379 PING
```

## Базовая проверка stream'ов

Проверка длины stream:

```bash
redis-cli -h 127.0.0.1 -p 6379 XLEN md.candles.1m.binance:spot:BTCUSDT
```

Информация о stream:

```bash
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM md.candles.1m.binance:spot:BTCUSDT
```

Чтение диапазона (первые записи):

```bash
redis-cli -h 127.0.0.1 -p 6379 XRANGE md.candles.1m.binance:spot:BTCUSDT - + COUNT 5
```

Чтение последних записей:

```bash
redis-cli -h 127.0.0.1 -p 6379 XREVRANGE md.candles.1m.binance:spot:BTCUSDT + - COUNT 5
```

Разовое чтение без consumer groups:

```bash
redis-cli -h 127.0.0.1 -p 6379 XREAD COUNT 10 STREAMS md.candles.1m.binance:spot:BTCUSDT 0-0
```

## Consumer groups

Создание группы (выполняется один раз):

```bash
redis-cli -h 127.0.0.1 -p 6379 XGROUP CREATE md.candles.1m.binance:spot:BTCUSDT strategy.demo '$' MKSTREAM
```

Проверка групп:

```bash
redis-cli -h 127.0.0.1 -p 6379 XINFO GROUPS md.candles.1m.binance:spot:BTCUSDT
```

Чтение через группу:

```bash
redis-cli -h 127.0.0.1 -p 6379 XREADGROUP GROUP strategy.demo consumer-1 COUNT 10 BLOCK 5000 STREAMS md.candles.1m.binance:spot:BTCUSDT '>'
```

Сводка pending-сообщений:

```bash
redis-cli -h 127.0.0.1 -p 6379 XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

Подтверждение обработанного сообщения:

```bash
redis-cli -h 127.0.0.1 -p 6379 XACK md.candles.1m.binance:spot:BTCUSDT strategy.demo 1739181240000-0
```

Удаление группы:

```bash
redis-cli -h 127.0.0.1 -p 6379 XGROUP DESTROY md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

## Retention и trim

Publisher использует приблизительное ограничение длины stream (`MAXLEN ~ <N>`).

Ручной trim по длине:

```bash
redis-cli -h 127.0.0.1 -p 6379 XTRIM md.candles.1m.binance:spot:BTCUSDT MAXLEN '~' 10080
```

Ручной trim по минимальному ID:

```bash
redis-cli -h 127.0.0.1 -p 6379 XTRIM md.candles.1m.binance:spot:BTCUSDT MINID 1738576440000-0
```

## Диагностика

Публикация в Redis работает в режиме best-effort.
Если Redis недоступен, worker продолжает запись raw свечей в ClickHouse.

Проверка Redis-метрик worker:

```bash
curl -fsS http://127.0.0.1:9201/metrics | rg 'redis_publish_(total|errors_total|duplicates_total|duration_seconds)'
```

Проверка логов worker по ошибкам publish:

```bash
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/market-data-ws-worker.err.log | rg 'redis publish failed|live candle publish'
```
