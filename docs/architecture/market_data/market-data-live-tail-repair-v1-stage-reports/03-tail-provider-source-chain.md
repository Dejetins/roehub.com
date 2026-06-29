# Stage 03: Tail Provider Source Chain

Статус: `accepted`.

Дата: `2026-06-30`.

## Pre-Start

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=03`; Stage `02` принят, доставлен в `origin/main` commit `1f7fc67e9888ad29794821d45a3a8a92eeaf79e0`, CI run `28407486401` завершился `success`, downstream `Deploy Backend` `28407578128`, `Deploy Web` `28407585789` / `28407578249` и `Publish App Image` `28407578134` завершились `success`.

Stage `04` до Stage `03 accepted` был закрыт.

## Что Реализовано

| Область | Итог |
|---|---|
| Provider chain | Добавлен `MarketDataClosedCandleTailProvider`, owned by Market Data application service. |
| Source order | Реализован порядок Redis hot cache -> ClickHouse canonical reader with failure circuit -> REST tail -> audit miss/failure. |
| Redis hot cache | Provider читает `ClosedCandleTailRow` из hot cache и short-circuits на continuous hit. REST-restored candles пишутся в hot cache до возврата success. |
| ClickHouse fallback | ClickHouse exception не завершает provider: provider фиксирует `clickhouse:failed`, открывает in-process circuit и продолжает к REST tail. |
| REST boundary | REST fallback использует `CandleIngestSource.stream_1m(...)` внутри Market Data, а не Strategy/direct exchange REST. |
| Closed-candle guard | Range, включающий current open minute, возвращает `continuous=false`, audit miss и не вызывает Redis/ClickHouse/REST source side effects. |
| Tail window guard | REST fallback не вызывается для ranges older than `ClosedCandleTailRepairPolicy.rest_tail_limit_minutes`. |
| Audit | Каждый provider call пишет `MarketDataCandleRepairAuditEvent`; Postgres repository boundary покрыт fake gateway proof. |
| Docs | Основной plan уточняет business impact, Stage `03` knobs и `post_main_production_runtime_proof` boundary. |

## Provider Source-Order Proof

Focused test `test_provider_falls_back_from_clickhouse_failure_to_rest_and_writes_hot_cache` выполнил direct provider call:

| Step | Evidence |
|---|---|
| Redis hot cache | initial miss -> `redis_hot_cache:miss`. |
| ClickHouse | forced `RuntimeError("clickhouse unavailable")` -> `clickhouse:failed`, provider continued. |
| REST tail | fake `CandleIngestSource` returned closed candle `2026-06-30T12:00:00.000Z` -> `rest:succeeded`. |
| Result | `continuous=True`, returned one sorted row with source `rest`. |
| Hot cache write | `hot_cache.write_calls == 1` before provider returned success. |
| Audit | one audit record with status `succeeded` and restored ts_open `2026-06-30T12:00:00.000Z`. |

Second direct provider call for the same range proved Redis hit:

| Check | Evidence |
|---|---|
| Redis hot cache | returned continuous row with source `redis_hot_cache`. |
| ClickHouse | not called again; `canonical_reader.read_calls` stayed `1`. |
| REST | not called again; REST call count stayed `1`. |
| Audit | second audit record written for the Redis-hit call. |

Focused test `test_provider_returns_sorted_rows_when_sources_restore_out_of_order` proved that a mixed source result is sorted by `ts_open`: Redis already had `12:02`, REST restored `12:00` and `12:01`, and the returned range was `12:00`, `12:01`, `12:02`.

## Audit DB Boundary Proof

Focused test `test_provider_records_audit_through_postgres_repository_boundary` wired provider to `PostgresCandleRepairAuditRepository` with a deterministic fake `MarketDataPostgresGateway`.

Evidence:

| Check | Result |
|---|---|
| Insert boundary | Query contained `INSERT INTO market_data_candle_repair_events`. |
| List boundary | `list_for_correlation("stage03-postgres-audit-proof")` returned one persisted event. |
| Sources attempted JSON | `redis_hot_cache:miss`, `clickhouse:failed/clickhouse_exception`, `rest:succeeded`. |
| Redaction | No raw provider payload, secret, DSN, token, cookie, Authorization header, or Redis auth value was recorded. |
| ClickHouse isolation | Joined fake gateway queries did not contain `canonical_candles` or `ClickHouse`. |

## Miss / Closed-Candle Proof

| Scenario | Evidence |
|---|---|
| Missing REST tail | Provider returned `continuous=false`, `missing_ts_opens=(2026-06-30T12:00:00.000Z)`, audit status `miss`, error_code `missing_closed_tail`. |
| Current open range | Provider returned `continuous=false`, error_code `non_closed_range`, and did not call Redis, ClickHouse, or REST. |
| REST tail limit exceeded | Provider skipped REST when the missing range was older than `rest_tail_limit_minutes`; audit status `miss`. |

## Service Calls / Ops Coverage

| Surface | Stage `03` decision |
|---|---|
| Strategy -> provider | Provider method shape matches `ClosedCandleTailProvider`, but Strategy runner wiring is explicitly deferred to Stage `04`. |
| Provider -> Redis hot cache | Reads first; writes REST-restored closed candles before success. |
| Provider -> ClickHouse | Uses injected `CanonicalCandleReader`; exceptions are redacted to `clickhouse_exception` and open an in-process circuit. |
| Provider -> REST | Uses injected Market Data `CandleIngestSource`; direct Strategy REST calls remain forbidden. |
| Provider -> Postgres audit | Writes one redacted `MarketDataCandleRepairAuditEvent` per provider call. |
| Runtime deployment | Stage `03` does not claim changed-code production behavior; Stage `06` owns `post_main_production_runtime_proof`. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/market_data/application/services/closed_candle_tail_provider.py` | none | none | Concrete Market Data provider source chain. | `compatible-change`; new application service, no existing caller changed. |
| `tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py` | none | none | Provider integration tests for source order, REST fallback, hot-cache write/hit, audit, miss guards. | `none`; tests only. |
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/03-tail-provider-source-chain.md` | none | none | Stage report and validation evidence. | `none`; docs/evidence only. |
| none | `src/trading/contexts/market_data/application/services/__init__.py` | none | Export provider service and hot-cache protocol. | `compatible-change`; additive import surface. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1.md` | none | Add business impact, Stage `03` knobs, and explicit Stage `06` proof boundary. | `none`; docs sync. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Mark Stage `03 accepted` and open Stage `04`. | `none`; staged workflow state only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after adding Stage `03` report. | `none`; generated docs index only. |

Files outside prompt expected paths: none. Strategy runner wiring and `configs/prod/strategy.yaml` were intentionally not changed because Stage `03` non-goal says not to modify Strategy runner processing/ACK behavior.

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Focused provider tests | passed | `uv run pytest -q tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py` -> `6 passed in 0.13s`. |
| Focused ruff | passed | `uv run ruff check src/trading/contexts/market_data/application/services/closed_candle_tail_provider.py src/trading/contexts/market_data/application/services/__init__.py tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py` -> `All checks passed!`. |
| Focused pyright | passed | `uv run pyright src/trading/contexts/market_data/application/services/closed_candle_tail_provider.py tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt `ruff` gate | passed | `uv run ruff check src/trading/contexts/market_data apps/worker/strategy_live_runner tests` -> `All checks passed!`. |
| Prompt `pyright` gate | passed | `uv run pyright src/trading/contexts/market_data apps/worker/strategy_live_runner tests` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt pytest gate adaptation | passed | `tests/integration` is absent; `uv run pytest -q tests/unit/contexts/market_data` -> `143 passed in 1.80s`. |
| Docs index | passed | `uv run python -m tools.docs.generate_docs_index --check` -> `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Publish ruff gate | passed | `uv run ruff check .` -> `All checks passed!`. |
| Publish pyright gate | passed | `uv run pyright` -> `0 errors, 0 warnings, 0 informations`. |
| Publish pytest gate | passed | `uv run pytest -q -ra` -> `1465 passed, 3 warnings in 65.47s`. |
| Publish whitespace gate | passed | `git diff --check` -> clean. |

Prompt gate `uv run pytest -q tests/unit/contexts/market_data tests/integration` is adapted because `tests/integration` does not exist in this repository snapshot. The equivalent Stage `03` evidence is `tests/unit/contexts/market_data` plus direct provider integration tests above.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API route or external payload changed. |
| Port contract | `none` | `ClosedCandleTailProvider` protocol shape is unchanged; provider implements the existing method structurally. |
| DTO schema | `none` | Uses existing repair DTOs/audit DTOs from Stage `01`; no schema change. |
| Persisted schema | `none` | Uses existing `market_data_candle_repair_events`; no migration. |
| Config schema | `none` | Uses `ClosedCandleTailRepairPolicy` constructor fields; no runtime YAML parser change. |
| Cache key / persistence identity | `none` | Reuses Stage `02` hot-cache keys; no new cache key pattern. |
| Service-call semantics | `compatible-change` | New provider chain changes behavior only for future callers; no existing runtime caller is wired in this stage. |
| External side-effect idempotency / unknown state | `compatible-change` | Audit write is fail-closed through repository exception propagation; hot-cache write failure prevents REST success from being returned. |
| Logs / metrics / redaction | `compatible-change` | Provider records bounded source/status/error codes only; no raw payloads or secrets. |
| Alert / runbook semantics | `none` | Stage `05` owns alerts/runbook. |
| Browser-visible behavior | `none` | No UI/browser behavior changed. |
| Performance benchmark / latency claim | `N/A` | Stage `03` makes no speed/latency claim and collected no comparable benchmark; provider proof is functional/integration evidence. |

## Delivery Status

Accepted delivery evidence for this stage is reviewed scoped staging, direct-main commit/push, local publish gates and GitHub Actions/CI after push. The exact commit hash is fixed in the final executor report because the hash cannot be written into the commit that creates it.

## Next Stage Handoff

Stage `04` can start after Stage `03` scoped direct-main delivery and CI are green. The next executor can rely on:

- provider source order is Redis hot cache -> ClickHouse -> REST -> audit miss/failure;
- ClickHouse failure does not stop fallback to REST;
- REST-restored closed candles are written to hot cache before provider success;
- missing/non-closed/too-old REST tail returns `continuous=false` and writes audit miss;
- Strategy runner processing and ACK policy are still unchanged and must be handled in Stage `04`.
