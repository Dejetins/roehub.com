---
prompt_name: roehub-market-data-activation-02
scope: "Replace file whitelist policy with organization-owned instrument selections, effective strategy pins, catalog/coverage/artifact API and browser UX."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md]
skill_routing: [architecture-design, contract-impact-analysis, backend-quality-gates, browser-qa-evidence, staged-plan-runner]
file_manifest:
  expected_primary_touches: [apps/migrations/, src/trading/contexts/market_data/, apps/api/routes/market_data_reference.py, apps/api/dto/market_data_reference.py, apps/web/, tests/unit/, tests/integration/]
  possible_secondary_touches: [configs/installation/, configs/installation/generated/, docs/runbooks/, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/]
validation_strategy: {depth: browser, acceptance_surfaces: [PostgreSQL migration, API isolation and idempotency, real browser onboarding/settings, desktop/mobile/accessibility smoke]}
proof_boundary: {label: isolated_local_runtime, exclusions: [production user data, strategy execution changes]}
---

# Этап 02 — выбор инструмента пользователем

Implement `InstrumentSelection` as organization-owned intent. `OrganizationEffectiveSelection` is that organisation's selections union active strategy pins; `GlobalEffectiveCollectorSet` is the union across organisations consumed only by workers. Updating selection must succeed even when a strategy is active; report only the current organisation's pin and never block change or delete data. Replace whitelist as runtime policy, retaining only versioned one-symbol bootstrap default when no user selection exists.

Refresh the global catalog through bounded metadata-only per-exchange jobs that persist refresh time and redacted error state without selecting/backfilling instruments. Expose supported catalog with `fresh|stale|failed`, selected/effective states, coverage (`unknown` when expected range is unknown) and actual current artifact byte size. All writes are RBAC/audit scoped and idempotent; API never leaks another organisation's selection or pin. Avoid automatic selection of the full catalog. Browser must provide initial chooser and editable settings with error/loading/empty states.
