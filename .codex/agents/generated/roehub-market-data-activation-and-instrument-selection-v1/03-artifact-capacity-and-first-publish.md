---
prompt_name: roehub-market-data-activation-03
scope: "Bound publisher memory to container capacity, manually publish BTCUSDT once, then encode an explicit expansion policy."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md]
skill_routing: [root-cause-debugging, backend-performance-evidence, backend-quality-gates, contract-impact-analysis, staged-plan-runner]
file_manifest:
  expected_primary_touches: [configs/prod/backtest_artifacts.yaml, configs/installation/runtime-service-manifest.json, apps/scheduler/backtest_artifact_publisher/, apps/cli/commands/backtest_artifact_publish.py, tests/unit/]
  possible_secondary_touches: [configs/installation/generated/, docs/runbooks/, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/]
validation_strategy: {depth: performance, acceptance_surfaces: [one manual isolated publish, RSS/container limit comparison, artifact pointer/inventory, no full-catalog work]}
proof_boundary: {label: isolated_local_runtime, exclusions: [scheduled expansion, production artifact store]}
---

# Этап 03 — память publisher и первый artifact

Make internal worker count and memory budget internally consistent with actual container limit; leave explicit margin. First run only one manual `BTCUSDT` publish after readiness. Measure peak container memory and artifact bytes with a comparable bounded workload. The publisher must read only `GlobalEffectiveCollectorSet`, never all legacy enabled reference rows. Its periodic schedule remains disabled until an explicit capacity decision after users expand selections; do not hide a full-catalog publish behind schedule.
