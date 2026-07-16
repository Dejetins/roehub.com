---
prompt_name: roehub-market-data-activation-01
scope: "Give only market-data scheduler and WS controlled outbound access, remove subscription startup race, and prove one-symbol readiness."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md]
skill_routing: [root-cause-debugging, contract-impact-analysis, backend-quality-gates, backend-performance-evidence, staged-plan-runner]
file_manifest:
  expected_primary_touches: [configs/installation/runtime-service-manifest.json, schemas/config/runtime-service-manifest.schema.json, src/trading/platform/config/runtime_topology.py, apps/worker/market_data_ws/, apps/scheduler/market_data_scheduler/, apps/api/routes/market_data_reference.py, tests/unit/]
  possible_secondary_touches: [configs/installation/generated/, infra/monitoring/, docs/runbooks/, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/]
validation_strategy: {depth: runtime, acceptance_surfaces: [compose topology, real public BTCUSDT REST/WS, ClickHouse/Redis freshness, error metrics, memory observation]}
proof_boundary: {label: isolated_local_runtime, exclusions: [production, orders, arbitrary public egress]}
---

# Этап 01 — controlled egress и readiness

Сохрани `roehub` internal. Рендери вторую сеть только для `market-data-scheduler` и `market-data-ws`; точный scope должен быть schema-validated и test-covered. Не называй Docker bridge FQDN allowlist: adapters permit only supported exchange endpoints and bounded retry/timeouts.

WS обязан периодически сверять effective enabled set, безопасно добавлять/отменять subscriptions и ждать непустого списка, а не оставаться idle после одного чтения. Добавь readiness, который доказывает connection, messages, inserts, свечу не старше SLA и отсутствие роста relevant errors; ливнес `/metrics` не достаточен. Initial runtime workload только `binance:futures:BTCUSDT`, sequential, limits recorded. Update ledger then report; do not enable broader schedule.
