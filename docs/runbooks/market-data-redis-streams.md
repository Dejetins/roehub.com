# Runbook — Market Data Redis Streams

## Scope
Operational commands for live feed streams published by `market-data-ws-worker`.

Stream pattern:
- `md.candles.1m.<instrument_key>`

Example stream:
- `md.candles.1m.binance:spot:BTCUSDT`

## Preconditions
Redis runs in docker compose as service `redis`.

Check container:

```bash
docker ps --filter name=redis
```

All commands below use `redis-cli` inside container (host install is not required).

## Basic Inspection
Check server ping:

```bash
docker exec -it redis redis-cli PING
```

List stream length:

```bash
docker exec -it redis redis-cli XLEN md.candles.1m.binance:spot:BTCUSDT
```

Show stream info:

```bash
docker exec -it redis redis-cli XINFO STREAM md.candles.1m.binance:spot:BTCUSDT
```

Read range (first entries):

```bash
docker exec -it redis redis-cli XRANGE md.candles.1m.binance:spot:BTCUSDT - + COUNT 5
```

Read latest entries:

```bash
docker exec -it redis redis-cli XREVRANGE md.candles.1m.binance:spot:BTCUSDT + - COUNT 5
```

Ad-hoc read without groups:

```bash
docker exec -it redis redis-cli XREAD COUNT 10 STREAMS md.candles.1m.binance:spot:BTCUSDT 0-0
```

## Consumer Groups
Create group (only once):

```bash
docker exec -it redis redis-cli XGROUP CREATE md.candles.1m.binance:spot:BTCUSDT strategy.demo '$' MKSTREAM
```

Inspect groups:

```bash
docker exec -it redis redis-cli XINFO GROUPS md.candles.1m.binance:spot:BTCUSDT
```

Read with group:

```bash
docker exec -it redis redis-cli XREADGROUP GROUP strategy.demo consumer-1 COUNT 10 BLOCK 5000 STREAMS md.candles.1m.binance:spot:BTCUSDT '>'
```

Pending entries summary:

```bash
docker exec -it redis redis-cli XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

Acknowledge processed message:

```bash
docker exec -it redis redis-cli XACK md.candles.1m.binance:spot:BTCUSDT strategy.demo 1739181240000-0
```

Destroy group (maintenance):

```bash
docker exec -it redis redis-cli XGROUP DESTROY md.candles.1m.binance:spot:BTCUSDT strategy.demo
```

## Retention and Trimming
Publisher uses approximate maxlen (`MAXLEN ~ <N>`).
Manual trim (approximate):

```bash
docker exec -it redis redis-cli XTRIM md.candles.1m.binance:spot:BTCUSDT MAXLEN '~' 10080
```

Manual trim by min ID:

```bash
docker exec -it redis redis-cli XTRIM md.candles.1m.binance:spot:BTCUSDT MINID 1738576440000-0
```

## Troubleshooting
Redis publish is best-effort. If Redis is unavailable, worker should continue writing raw candles to ClickHouse.

Check worker metrics for Redis path:

```bash
curl -fsS http://localhost:9201/metrics | rg 'redis_publish_(total|errors_total|duplicates_total|duration_seconds)'
```

Check worker logs for publish errors:

```bash
docker logs --tail 200 market-data-ws-worker | rg 'redis publish failed|live candle publish'
```

## Windows PowerShell Notes
If running on Windows with Docker Desktop, commands are the same:

```powershell
docker exec -it redis redis-cli XLEN md.candles.1m.binance:spot:BTCUSDT
```
