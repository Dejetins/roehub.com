# Runbook — Market Data Redis Streams

## Назначение
Операционные команды для live feed stream’ов, которые публикует `market-data-ws-worker`.

Шаблон stream:
- `md.candles.1m.<instrument_key>`

Пример stream:
- `md.candles.1m.binance:spot:BTCUSDT`

## Предусловия
Redis должен быть запущен в docker-compose как сервис `redis`.

Проверка контейнера:

```bash
docker ps --filter name=redis
```

Все команды ниже используют `redis-cli` внутри контейнера (установка `redis-cli` на хост не требуется).

## Базовая проверка stream’ов
Проверка доступности Redis:

```bash
docker exec -it redis redis-cli PING
```

Проверка длины stream:

```bash
docker exec -it redis redis-cli XLEN md.candles.1m.binance:spot:BTCUSDT
```

Информация о stream:

```bash
docker exec -it redis redis-cli XINFO STREAM md.candles.1m.binance:spot:BTCUSDT
```

Чтение диапазона (первые записи):

```bash
docker exec -it redis redis-cli XRANGE md.candles.1m.binance:spot:BTCUSDT - + COUNT 5
```

Чтение последних записей:

```bash
docker exec -it redis redis-cli XREVRANGE md.candles.1m.binance:spot:BTCUSDT + - COUNT 5
```

Разовое чтение без consumer groups:

```bash
docker exec -it redis redis-cli XREAD COUNT 10 STREAMS md.candles.1m.binance:spot:BTCUSDT 0-0
```

## Consumer Groups
Создание группы (выполняется один раз):

```bash
docker exec -it redis redis-cli XGROUP CREATE md.candles.1m.binance:spot:BTCUSDT strategy.demo '$' MKSTREAM
```

Проверка групп:

```bash
docker exec -it redis redis-cli XINFO GROUPS md.candles.1m.binance:spot:BTCUSDT
```

Чтение через группу:

```bash
docker exec -it redis redis-cli XREADGROUP GROUP strategy.demo consumer-1 COUNT 10 BLOCK 5000 STREAMS md.candles.1m.binance:spot:BTCUSDT '>'
```

Сводка pending-сообщений:

```bash
docker exec -it redis redis-cli XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

Подтверждение обработанного сообщения:

```bash
docker exec -it redis redis-cli XACK md.candles.1m.binance:spot:BTCUSDT strategy.demo 1739181240000-0
```

Удаление группы (операционное обслуживание):

```bash
docker exec -it redis redis-cli XGROUP DESTROY md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

## Retention и trim
Publisher использует приблизительное ограничение длины stream (`MAXLEN ~ <N>`).

Ручной trim по длине (approximate):

```bash
docker exec -it redis redis-cli XTRIM md.candles.1m.binance:spot:BTCUSDT MAXLEN '~' 10080
```

Ручной trim по минимальному ID:

```bash
docker exec -it redis redis-cli XTRIM md.candles.1m.binance:spot:BTCUSDT MINID 1738576440000-0
```

## Диагностика
Публикация в Redis работает в режиме best-effort.
Если Redis недоступен, worker должен продолжать запись raw свечей в ClickHouse.

Проверка Redis-метрик worker:

```bash
curl -fsS http://localhost:9201/metrics | rg 'redis_publish_(total|errors_total|duplicates_total|duration_seconds)'
```

Проверка логов worker по ошибкам publish:

```bash
docker logs --tail 200 market-data-ws-worker | rg 'redis publish failed|live candle publish'
```

## Примечание для Windows PowerShell
Если используется Docker Desktop на Windows, команды те же:

```powershell
docker exec -it redis redis-cli XLEN md.candles.1m.binance:spot:BTCUSDT
```
