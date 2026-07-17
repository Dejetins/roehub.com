---
prompt_name: roehub-market-data-activation-05
scope: "Rebuild the affected candidate and run a fresh bounded Docker Desktop lifecycle with browser proof, then hand off original Stage 22-24 recertification."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
skill_routing: [backend-quality-gates, backend-performance-evidence, browser-qa-evidence, staged-plan-runner]
file_manifest:
  expected_primary_touches: [tools/release/, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/]
  possible_secondary_touches: [docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [new OCI/config provenance, Docker Desktop clean lifecycle, browser onboarding/settings, one-symbol data/artifact proof, cleanup/memory]}
proof_boundary: {label: local_macos_docker_desktop_arm64, exclusions: [native linux amd64, production, durable OpenBao custody if keys unavailable]}
---

# Этап 05 — новый кандидат и локальный lifecycle

Rebuild only after prior change evidence. Execute operations sequentially with bounded parallelism and clean build containers/caches that are not required for retained candidate. Fresh local launch must prove normal ingress (no manual `bridge` workaround), UI onboarding/settings and one-symbol market data/artifact state. The candidate change invalidates original Stage `22`/`23` evidence; write a truthful handoff to re-run them. `linux/amd64` and durable OpenBao owner custody must remain explicitly blocked if unavailable.
