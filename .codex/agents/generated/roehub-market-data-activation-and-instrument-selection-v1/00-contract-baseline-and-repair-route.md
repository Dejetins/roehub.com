---
prompt_name: roehub-market-data-activation-00
scope: "Freeze actual market-data, instrument-selection and OpenBao contracts; make the repair route executable."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md]
skill_routing: [architecture-design, contract-impact-analysis, staged-plan-runner]
file_manifest:
  expected_primary_touches: [docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/00-contract-baseline-and-repair-route.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md]
validation_strategy: {depth: integration, acceptance_surfaces: [contract consumer search, config schema, baseline runtime audit]}
proof_boundary: {label: N/A, exclusions: [production, secrets, provider writes]}
---

# Этап 00 — baseline и repair route

Проверь источники фактов и явно классифицируй API, DTO, storage, config, service-call, side-effect, observability и browser impact. Зафиксируй тот факт, что Docker network сам по себе не обеспечивает FQDN allowlist. Не менять рабочее поведение, не создавать секреты. Создай отчёт, обнови ledger после проверок; разреши `01` и `04` только если все три execution artifacts связаны и baseline доказан.
