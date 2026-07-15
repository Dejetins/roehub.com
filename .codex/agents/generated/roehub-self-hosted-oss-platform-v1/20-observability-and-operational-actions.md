---
prompt_name: 20-observability-and-operational-actions
repo: roehub.com
scope: "Containerize independent observability, expose Roehub domain health states and connect alerts to machine-readable runbooks and typed actions."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "20", prerequisites: ["02", "17", "18", "19"], previous_stage_gate: "Stages 02, 17, 18 and 19 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: runtime/alert/runbook rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: independent observability contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: topology/control/UI prerequisites}
  task_entrypoints:
    - {path: apps/monitoring/, why: existing exporters}
    - {path: infra/monitoring/, why: existing Prometheus/Grafana assets}
    - {path: infra/macos/prometheus/, why: current alert rules}
    - {path: docs/runbooks/, why: operational guidance}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before metrics/alerts/actions, reason: operational consumers and response semantics}
  - {skill: backend-quality-gates, timing: verification, reason: exporter/API gates}
  - {skill: browser-qa-evidence, timing: admin health verification, reason: real domain-state UI evidence}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [apps/monitoring/, apps/api/, apps/web/, src/trading/contexts/operations/, infra/monitoring/, infra/docker/, schemas/ops/, docs/runbooks/, tests/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated dashboards/alerts and keep metric compatibility where documented
file_manifest:
  expected_primary_touches: [apps/monitoring/, infra/monitoring/, infra/docker/, schemas/ops/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/20-observability-and-operational-actions.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/api/, apps/web/, src/trading/contexts/operations/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: runtime, acceptance_surfaces: [runtime smoke, failure injection, independent metrics/log survival, ready/degraded/stopped/unknown mapping, alert-runbook-action links, browser drill-down, redaction]}
proof_boundary: {label: N/A, exclusions: [production incident mutation, Monit shutdown on current host]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Keep metrics, alerts and logs available when Roehub Web/API fail while giving users a coherent domain-level operational view and approved actions.

# Requirements

- Containerize/pin Prometheus, Alertmanager, Grafana, blackbox/exporters and the selected log store; generated provisioning only.
- Standardize live/ready/metrics and service labels without leaking organization, account or secret data.
- Map technical evidence to `ready/degraded/stopped/unknown`, affected capability, runbook and allowlisted action.
- Generate scrape/rule/dashboard fragments from release/plugin operational manifests where appropriate.
- Roehub admin UI provides common operations; deep diagnostics may link/SSO to Grafana but must not depend on anonymous embedding.
- Observability storage persists independently of Web/API containers.

# Browser authentication and evidence safety

- Use a disposable local admin identity for the isolated target runtime. If browser comparison uses the current Keycloak-backed runtime, the only default username is `smoke_e2e_keycloak` and its password comes only from host-local `ROEHUB_SMOKE_E2E_PASSWORD` in `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Never request credentials in chat. Redact cookies, tokens, authorization headers, organization/account labels, secret-bearing logs and provider payloads from screenshots, traces, reports and the ledger.

# Validation

Run a real runtime smoke with controlled failure injection for Web, API, worker, PostgreSQL, ClickHouse, Redis, OpenBao and a plugin. Prove metrics/logs survive, alerts fire/resolve, runbook/action links are valid, admin state is correct after recovery, and browser drill-down has no console/network/security failures.

# Stop rules

Block on monitoring that disappears with Web/API, alert without runbook, unsafe action, secret/high-cardinality labels, missing degraded semantics or static dashboards that cannot follow generated topology. Update ledger after evidence.
