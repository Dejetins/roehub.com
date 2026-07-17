---
prompt_name: roehub-market-data-activation-04
scope: "Implement owner-operated secure OpenBao initialization and least-privilege credential delivery without generating or exposing custody material."
language: {implementation: en, agent_report: ru}
context_sources:
  always_read: [.codex/AGENTS.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md, docs/runbooks/openbao-secrets-and-recovery.md]
skill_routing: [architecture-design, contract-impact-analysis, backend-quality-gates, staged-plan-runner]
file_manifest:
  expected_primary_touches: [apps/cli/, apps/control_agent/, infra/openbao/, configs/openbao/, src/trading/platform/secrets/, tests/unit/]
  possible_secondary_touches: [configs/installation/, configs/installation/generated/, docs/runbooks/, docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/]
validation_strategy: {depth: runtime, acceptance_surfaces: [disposable three-recipient 2-of-3 PGP-public-key flow, sealed failure, policy/credential permission/revocation tests]}
proof_boundary: {label: disposable_runtime_and_owner_handoff, exclusions: [durable user PGP private keys, unseal shares, admin tokens, production]}
---

# Этап 04 — OpenBao

Build a separate idempotent owner-init command/flow; never repurpose a verifier or legacy 1-of-1 native script. Require exactly three public PGP recipients and a `2-of-3` threshold, per-service policies/AppRoles and response-wrapped narrowly mounted token files. Preserve the private internal OpenBao trust boundary and fail degraded when sealed/uninitialized.

Do not create or retain private PGP keys, unseal shares, recovery identity, SecretIDs or root token. A durable real initialization is an owner action after the stage: the owner supplies three public recipient fingerprints/key material through an approved local source and confirms external custody. This stage is accepted after its disposable proof and documented handoff, not by pretending durable custody occurred.
