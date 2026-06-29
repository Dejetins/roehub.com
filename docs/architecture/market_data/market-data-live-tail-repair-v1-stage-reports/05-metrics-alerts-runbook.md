# Stage 05: Metrics Alerts Runbook

Статус: `accepted`.

Дата: `2026-06-30`.

## Pre-Start

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=05`; Stage `04` принят, доставлен в `origin/main` commit `9f0dd463db216783c439509e68c94a6aaaaed384`, CI run `28409427540` завершился `success`, downstream `Deploy Backend` `28409612604`, `Publish App Image` `28409612671`, `Deploy Web` `28409612603` и `Deploy Web` `28409618746` завершились `success`.

Stage `06` до Stage `05 accepted` был закрыт.

## Что Реализовано

| Область | Итог |
|---|---|
| Provider metrics hooks | `MarketDataClosedCandleTailProvider` emits bounded source/status attempt counters, latency observations, and ClickHouse circuit state through optional hooks. |
| Strategy runner ACK metrics hooks | `StrategyLiveRunner` emits gap, checkpoint-stall, and deferred-ACK callbacks when a repair gap blocks checkpoint progress. |
| Runtime wiring | `strategy-producer` wires `ClosedCandleTailProviderHooks`, `StrategyLiveRunnerRepairHooks`, and Redis hot-cache hooks into `StrategyLiveRunnerMetrics`. |
| WS hot-cache metrics | `market-data-ws-worker` keeps existing `redis_hot_cache_*` metrics and also emits canonical `market_data_hot_cache_*` counters. |
| Prometheus rules | Added `infra/macos/prometheus/rules/market-data-live-tail-repair.rules.yml` and wired it into `prometheus.prod.yml` plus `bootstrap_native_prod.sh`. |
| Runbook | Added Russian operator runbook `docs/runbooks/market-data-live-tail-repair.md` with safe checks for metrics, Redis, Postgres repair audit, ClickHouse, REST tail, checkpoint, and logs. |
| Docs sync | Updated Market Data plan and monitoring metric catalogs with Stage `05` metrics, alerts, and runbook reference. |

## Metrics Added / Wired

| Metric | Labels | Emitted by | Meaning |
|---|---|---|---|
| `market_data_live_tail_gap_total` | `source_stage` | `strategy-producer` | Closed 1m gap detected by Strategy live-tail path. |
| `market_data_live_tail_repair_total` | `source`, `status` | `strategy-producer` | Repair attempts by bounded source/status. |
| `market_data_live_tail_repair_latency_seconds` | `source`, `status` | `strategy-producer` | Provider repair latency observation. |
| `market_data_hot_cache_hit_total` | none | `strategy-producer`, `market-data-ws-worker` | Redis hot-cache read returned candles. |
| `market_data_hot_cache_miss_total` | none | `strategy-producer`, `market-data-ws-worker` | Redis hot-cache read returned no candles. |
| `market_data_hot_cache_write_total` | none | `strategy-producer`, `market-data-ws-worker` | Redis hot-cache write succeeded. |
| `market_data_hot_cache_error_total` | none | `strategy-producer`, `market-data-ws-worker` | Redis hot-cache read/write failed. |
| `market_data_clickhouse_repair_circuit_state` | none | `strategy-producer` | `1` when ClickHouse repair circuit is open, `0` when closed. |
| `strategy_live_runner_checkpoint_stall_total` | `reason` | `strategy-producer` | Checkpoint could not advance because repair was incomplete. |
| `strategy_live_runner_deferred_ack_total` | `reason` | `strategy-producer` | Redis message ACK was deferred until repair/retry succeeds. |

Bounded label allowlists are code-level: repair sources `redis_hot_cache|clickhouse|rest`, statuses `attempted|succeeded|miss|failed|circuit_open|rate_limited`, gap stage `strategy_runner`, ACK reason `repair_incomplete`. Unknown values collapse to `unknown`; no `user_id`, `run_id`, `order_id`, `symbol`, `instrument_key`, raw provider error, or secret-bearing label was added.

## Alert Rules

| Alert | Severity | Owner | Evidence |
|---|---|---|---|
| `MarketDataLiveTailUnrepairedGapBeyondPolicy` | `critical` | `market-data` | `strategy_live_runner_deferred_ack_total{reason="repair_incomplete"}` remains non-zero beyond policy window. |
| `MarketDataClickHouseRepairCircuitOpenTooLong` | `warning` | `market-data` | `market_data_clickhouse_repair_circuit_state == 1` for `2m`. |
| `MarketDataRestTailRepairErrors` | `warning` | `market-data` | REST source repair attempts report `failed` or `rate_limited`. |
| `MarketDataHotCacheShortTailMiss` | `warning` | `market-data` | `market_data_hot_cache_miss_total` grows for repair/WS jobs. |
| `StrategyProducerNoSignalGrowth` | `critical` | `strategy-producer` | Active instruments exist but `strategy_signal_total{outcome="signal"}` does not grow for `30m`. |

Alert validation uses the repository's current monitoring asset test path: YAML parse through `_load_yaml(...)`, exact alert-name assertions, bounded labels, runbook anchors, and required action annotations. Local `promtool` is not installed in this environment, so no `promtool check rules` evidence is claimed.

## Synthetic Metrics Proof

`test_live_tail_metrics_are_bounded_and_scrapable` creates a direct `CollectorRegistry`, calls the Stage `05` repair/provider/hot-cache hooks, scrapes `generate_latest(registry)`, and proves the metric names are present:

- `market_data_live_tail_gap_total`
- `market_data_live_tail_repair_total`
- `market_data_live_tail_repair_latency_seconds`
- `market_data_hot_cache_hit_total`
- `market_data_hot_cache_miss_total`
- `market_data_hot_cache_write_total`
- `market_data_hot_cache_error_total`
- `market_data_clickhouse_repair_circuit_state`
- `strategy_live_runner_checkpoint_stall_total`
- `strategy_live_runner_deferred_ack_total`

The same test passes a UUID-like unbounded stage value and asserts the raw value is absent from scrape output while `source_stage="unknown"` exists.

`test_provider_emits_bounded_repair_metrics_hooks` forces Redis miss, ClickHouse failure, and REST success through `MarketDataClosedCandleTailProvider`; it proves emitted attempts `redis_hot_cache/miss`, `clickhouse/failed`, `rest/succeeded`, non-negative latency observations, and ClickHouse circuit state `1`.

## Runbook

Operator runbook: `docs/runbooks/market-data-live-tail-repair.md`.

It contains safe commands for:

- `strategy-producer` metrics endpoint `http://127.0.0.1:9207/metrics`;
- `market-data-ws-worker` metrics endpoint `http://127.0.0.1:9201/metrics`;
- Redis key cardinality checks for `md:hot:1m:<instrument_key>:z` and `md:hot:1m:<instrument_key>:h`;
- Postgres aggregate query on `market_data_candle_repair_events` without raw payload;
- ClickHouse health checks;
- `strategy-producer` readiness;
- redacted log tailing.

The runbook explicitly forbids printing env dumps, DSN, tokens, cookies, passphrases, API keys, `Authorization` headers, Redis password, raw provider payloads, and private account/order payloads.

## Service Calls / Ops Coverage

| Surface | Stage `05` decision |
|---|---|
| Metrics endpoint | Direct registry scrape proof covers the metric surface before production deployment; runtime endpoint proof is deferred to Stage `06` `post_main_production_runtime_proof`. |
| Redis | No new Redis behavior; Stage `05` only adds counters around accepted hot-cache hooks. |
| Postgres audit | No schema change; runbook uses aggregate status counts from `market_data_candle_repair_events`. |
| ClickHouse | No repair behavior change; only circuit state metric and alert/runbook coverage. |
| REST tail | No provider behavior change; only source/status metrics and redacted alert guidance. |
| Secrets/redaction | Runbook and alert actions prohibit secret/raw payload output. |
| Production runtime proof | Stage `05` does not claim changed-code production behavior; Stage `06` owns `post_main_production_runtime_proof`. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/05-metrics-alerts-runbook.md` | none | none | Stage report and validation evidence. | `none`; docs/evidence only. |
| `docs/runbooks/market-data-live-tail-repair.md` | none | none | Russian operator runbook for Stage `05` alerts and safe checks. | `compatible-change`; additive ops docs. |
| `infra/macos/prometheus/rules/market-data-live-tail-repair.rules.yml` | none | none | Prometheus alert rules for live-tail repair. | `compatible-change`; additive monitoring config. |
| none | `src/trading/contexts/market_data/application/services/closed_candle_tail_provider.py` | none | Add optional metrics hooks for repair attempts, latency, and ClickHouse circuit state. | `compatible-change`; constructor arg is optional. |
| none | `src/trading/contexts/market_data/application/services/__init__.py` | none | Export `ClosedCandleTailProviderHooks`. | `compatible-change`; additive export. |
| none | `src/trading/contexts/strategy/application/services/live_runner.py` | none | Add optional repair/ACK hooks and emit gap/stall/deferred-ACK events. | `compatible-change`; constructor arg is optional. |
| none | `src/trading/contexts/strategy/application/services/__init__.py` | none | Export `StrategyLiveRunnerRepairHooks`. | `compatible-change`; additive export. |
| none | `src/trading/contexts/strategy/application/__init__.py` | none | Re-export `StrategyLiveRunnerRepairHooks`. | `compatible-change`; additive export. |
| none | `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` | none | Register and wire live-tail repair/cache/circuit/checkpoint metrics. | `compatible-change`; additive metrics only. |
| none | `apps/worker/market_data_ws/wiring/modules/market_data_ws.py` | none | Add canonical `market_data_hot_cache_*` counters alongside existing `redis_hot_cache_*` counters. | `compatible-change`; additive metrics only. |
| none | `infra/macos/prometheus/prometheus.prod.yml` | none | Load Stage `05` alert rules in production Prometheus config. | `compatible-change`; additive rule file. |
| none | `scripts/macos/bootstrap_native_prod.sh` | none | Install Stage `05` alert rules into `/opt/roehub/config/prometheus.rules`. | `compatible-change`; additive install step. |
| none | `tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py` | none | Provider hook emission proof. | `none`; tests only. |
| none | `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | none | Gap/stall/deferred-ACK hook proof around failed repair retry. | `none`; tests only. |
| none | `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py` | none | Synthetic registry scrape proof and bounded-label proof. | `none`; tests only. |
| none | `tests/unit/infra/test_monitoring_assets.py` | none | Prometheus config/rules/bootstrap validation. | `none`; tests only. |
| none | `docs/runbooks/market-data-metrics.md` | none | Add Stage `05` metrics to monitoring catalog. | `none`; docs sync. |
| none | `docs/runbooks/market-data-metrics-reference-ru.md` | none | Add Stage `05` metrics to detailed Russian reference. | `none`; docs sync. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1.md` | none | Sync implemented metrics/alerts/runbook and remove high-cardinality metric labels from plan text. | `none`; docs sync. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Mark Stage `05 accepted` and open Stage `06`. | `none`; staged workflow state only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after adding Stage `05` report/runbook. | `none`; generated docs index only. |

Files outside prompt expected paths: none. The pre-existing foreign change in `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` remains out of scope.

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Focused Stage `05` tests | passed | `uv run pytest -q tests/unit/contexts/market_data/application/services/test_closed_candle_tail_provider.py tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py tests/unit/infra/test_monitoring_assets.py` -> `40 passed in 0.50s`. |
| Focused ruff | passed | `uv run ruff check ...` on modified Python files/tests -> `All checks passed!`. |
| Focused pyright | passed | `uv run pyright ...` on modified Python files/tests -> `0 errors, 0 warnings, 0 informations`. |
| Prompt `ruff` gate | passed | `uv run ruff check apps src tests` -> `All checks passed!`. |
| Prompt `pyright` gate | passed | `uv run pyright apps src tests` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt pytest gate adaptation | passed | `tests/integration` is absent; `uv run pytest -q tests/unit` -> `1456 passed, 3 warnings in 50.37s`. |
| Alert rule validation | passed | `test_macos_prometheus_live_tail_repair_rules_are_repo_managed` parses YAML and asserts alert names, bounded labels, runbook anchors, and actions. |
| Synthetic direct registry scrape | passed | `test_live_tail_metrics_are_bounded_and_scrapable` asserts required metric names in `generate_latest(registry)` output and proves unbounded stage value collapses to `unknown`. |
| Docs index | passed | `uv run python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md`; `uv run python -m tools.docs.generate_docs_index --check` -> `OK`. |
| Publish ruff gate | passed | `uv run ruff check .` -> `All checks passed!`. |
| Publish pyright gate | passed | `uv run pyright` -> `0 errors, 0 warnings, 0 informations`. |
| Publish pytest gate | passed | `uv run pytest -q -ra` -> `1470 passed, 3 warnings in 60.12s`. |
| Publish whitespace gate | passed | `git diff --check` -> clean. |

Prompt gate `uv run pytest -q tests/unit tests/integration` is adapted because `tests/integration` does not exist in this repository snapshot. The equivalent Stage `05` evidence is full `tests/unit`, focused metric/rules tests, and direct registry scrape proof above.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No HTTP/API payload changed. |
| Port contract | `compatible-change` | Optional metrics hook dataclasses are additive; existing constructor call sites still work. |
| DTO schema | `none` | No DTO shape changes. |
| Persisted schema | `none` | No migration or table change. |
| Config / ops | `compatible-change` | Adds Prometheus rule file and bootstrap install line. |
| Redis behavior | `none` | No Redis key/write/read behavior changed. |
| ClickHouse behavior | `none` | No query or circuit policy changed. |
| Strategy ACK behavior | `none` | Stage `04` behavior unchanged; Stage `05` observes existing ACK deferral. |
| Logs / redaction | `compatible-change` | Runbook adds explicit redaction rules; no new secret-bearing labels/logs. |
| Browser-visible behavior | `none` | No UI/browser behavior changed. |
| Performance benchmark / latency claim | `N/A` | Stage `05` records latency metrics but makes no performance improvement claim. |

## Delivery Status

Accepted delivery evidence for this stage is reviewed scoped staging, direct-main commit/push, local publish gates and GitHub Actions/CI after push. The exact commit hash is fixed in the final executor report because the hash cannot be written into the commit that creates it.

## Next Stage Handoff

Stage `06` can start after Stage `05` scoped direct-main delivery and CI are green. The next executor can rely on:

- `strategy-producer` exposes live-tail repair/cache/circuit/checkpoint-stall/deferred-ACK metrics through wired hooks;
- `market-data-ws-worker` exposes canonical `market_data_hot_cache_*` counters in addition to existing `redis_hot_cache_*`;
- Prometheus production config loads `market-data-live-tail-repair.rules.yml`;
- `bootstrap_native_prod.sh` installs that rules file into `/opt/roehub/config/prometheus.rules`;
- operator runbook is `docs/runbooks/market-data-live-tail-repair.md`;
- production changed-code runtime proof is still not claimed and belongs to Stage `06`.
