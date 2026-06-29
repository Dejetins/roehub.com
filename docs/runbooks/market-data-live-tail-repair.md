# Market Data Live Tail Repair Runbook

Статус: operator runbook для Stage `05` плана `market-data-live-tail-repair-v1`.

Документ описывает безопасные проверки live-tail repair без вывода секретов, DSN, токенов, Redis password, `Authorization` headers и raw provider payloads.

## Runtime Surfaces

| Поверхность | Где смотреть | Зачем |
|---|---|---|
| `strategy-producer` metrics | `http://127.0.0.1:9207/metrics` | Gap, repair source/status/latency, ClickHouse circuit, checkpoint stall, deferred ACK, `StrategySignal` growth. |
| `market-data-ws-worker` metrics | `http://127.0.0.1:9201/metrics` | Redis hot-cache write/read hit/miss/error со стороны WS ingestion. |
| Redis hot cache | `md:hot:1m:<instrument_key>:z`, `md:hot:1m:<instrument_key>:h` | Short-tail range store для закрытых 1m свечей. |
| Postgres repair audit | `market_data_candle_repair_events` | Redacted audit of repair attempts and remaining missing minutes. |
| ClickHouse | HTTP ping / native client | Historical source health for bounded repair chain. |

## Safe Checks

Проверка Stage `05` метрик в `strategy-producer`:

```bash
curl -fsS http://127.0.0.1:9207/metrics | rg '^(market_data_live_tail_|market_data_hot_cache_|market_data_clickhouse_repair_circuit_state|strategy_live_runner_(checkpoint_stall|deferred_ack)_total)'
```

Проверка hot-cache метрик в `market-data-ws-worker`:

```bash
curl -fsS http://127.0.0.1:9201/metrics | rg '^(market_data_hot_cache_|redis_hot_cache_)'
```

Проверка наличия hot-cache ключей без печати значений:

```bash
redis-cli --scan --pattern 'md:hot:1m:*' | head
redis-cli ZCARD 'md:hot:1m:<instrument_key>:z'
redis-cli HLEN 'md:hot:1m:<instrument_key>:h'
```

Проверка repair audit агрегатом без raw payload:

```sql
SELECT status, COUNT(*)
FROM market_data_candle_repair_events
WHERE created_at >= now() - interval '30 minutes'
GROUP BY status
ORDER BY status;
```

Проверка ClickHouse health:

```bash
curl -fsS http://127.0.0.1:8123/ping
clickhouse client --query "SELECT 1"
```

Проверка readiness `strategy-producer`:

```bash
curl -fsS http://127.0.0.1:9207/health/ready
```

Проверка логов без секретов:

```bash
tail -n 200 ~/Library/Logs/roehub/strategy-live-runner.err.log | rg 'repair|checkpoint|deferred|ClickHouse|Redis|REST'
```

## Stage 05 Alert Actions

| Alert | Что значит | Безопасные действия |
|---|---|---|
| `MarketDataLiveTailUnrepairedGapBeyondPolicy` | `StrategyLiveRunner` отложил ACK из-за unrepaired closed 1m gap дольше policy window. | Проверить `strategy_live_runner_deferred_ack_total`, `strategy_live_runner_checkpoint_stall_total`, Redis pending messages, агрегат `market_data_candle_repair_events`, hot-cache key cardinality и readiness `strategy-producer`. |
| `MarketDataClickHouseRepairCircuitOpenTooLong` | Repair provider открыл ClickHouse circuit и временно не полагается на ClickHouse как repair source. | Проверить `market_data_clickhouse_repair_circuit_state`, `curl -fsS http://127.0.0.1:8123/ping`, `clickhouse client --query "SELECT 1"` и последние redacted audit statuses. |
| `MarketDataRestTailRepairErrors` | REST fallback достигнут, но попытки завершились `failed` или `rate_limited`. | Проверить `market_data_live_tail_repair_total{source="rest"}`, provider health/rate-limit признаки в редактированных логах, не печатать API credentials и raw provider payloads. |
| `MarketDataHotCacheShortTailMiss` | Redis hot cache не содержит нужный short-tail диапазон. | Проверить `market_data_hot_cache_miss_total`, `market_data_hot_cache_write_total`, `market_data_hot_cache_error_total`, `md:hot:1m:<instrument_key>:z`, `md:hot:1m:<instrument_key>:h` и работу `market-data-ws-worker`. |
| `StrategyProducerNoSignalGrowth` | Есть active instruments, но `strategy_signal_total{outcome="signal"}` не растет. | Проверить checkpoint progress, deferred ACK counters, repair audit, readiness, producer health и live-runner logs перед изменением allowlist или запуском rerun. |

## Что Нельзя Выводить

Не печатать в чат, stage report, логи расследования и screenshots:

- переменные окружения целиком;
- DSN и connection strings;
- токены, cookies, passphrases, API keys;
- `Authorization` headers;
- Redis password;
- raw provider payloads;
- приватные account/order payloads.

Если нужна глубокая диагностика с секретами, выполнять ее локально в защищенном shell и в отчете оставлять только редактированные агрегаты: счетчики, статусы, timestamps, cardinality и hash/commit identifiers.
