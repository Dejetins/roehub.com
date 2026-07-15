---
prompt_name: 00-freeze-current-state-and-contracts
repo: roehub.com
scope: "Audit every current component and freeze the evidence-backed greenfield self-hosted OSS contract map before implementation."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution:
  mode: goal_driven
  plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md
  prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/
  stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
stage: {id: "00", prerequisites: [], previous_stage_gate: "Plan artifacts and their cold-head review are present."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: accepted target architecture}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: execution gate}
    - {path: docs/architecture/project-map/AGENT_GUIDE.md, why: bounded repository navigation}
    - {path: docs/architecture/project-map/project-map.json, why: current component inventory}
  task_entrypoints:
    - {path: README.md, why: current native-first product contract}
    - {path: infra/docker/docker-compose.yml, why: current incomplete container topology}
    - {path: docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md, why: superseded target to reconcile}
skill_routing:
  - {skill: staged-plan-runner, timing: before any stage action, reason: enforce ledger and goal continuation rules}
  - {skill: architecture-review, timing: during audit, reason: evidence-backed current-state and drift review}
  - {skill: contract-impact-analysis, timing: before freezing contracts, reason: classify all changed surfaces}
change_ownership:
  parallel_main_expected: true
  owned_change_scope:
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/00-freeze-current-state-and-contracts.md
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md
    - docs/architecture/project-map/
    - docs/architecture/README.md
  foreign_changes_policy: preserve and exclude all unrelated files and hunks
file_manifest:
  expected_primary_touches:
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/00-freeze-current-state-and-contracts.md
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md
  possible_secondary_touches:
    - docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md
    - docs/architecture/project-map/project-map.json
    - docs/architecture/README.md
validation_strategy:
  depth: integration
  acceptance_surfaces: [all-current-component coverage, service-call map, state ownership, contract-impact matrix, docs continuity]
proof_boundary: {label: N/A, exclusions: [runtime behavior, implementation readiness, production changes, migration of current production data]}
authority: {implementation_write: docs_only, git_publish: false, production_mutation: false}
---

# Objective

Freeze the current code/runtime evidence baseline for all components in `project-map.json` and verify that the greenfield plan covers every app, context, core package and worker. Produce the Stage `00` report and correct only evidence-backed plan/map gaps. Current production data is not an input to the target state.

# Non-goals

- Do not implement target architecture.
- Do not change runtime, databases, providers, secrets or production.
- Do not inspect, repair, backfill, copy or reconcile current production data; the target starts from empty stores.
- Do not infer facts from historical plans when current code disagrees.

# Required workflow

1. Confirm the ledger links and `current_stage: 00`; otherwise record `blocked` and stop.
2. Inventory first, then read bounded slices for composition roots, ports, persistence, config, runtime and tests.
3. For every component record current owner, entrypoint, dependencies, state, deployment, trust boundary, target stage and evidence path.
4. Freeze service calls including auth, secrets, timeout, retry, idempotency, unknown-state behavior, metrics, alerts and runbook gaps.
5. Search subscription/`paid_level`, Keycloak-only, Monit/native, Docker, telemetry, plugin and artifact contracts as implementation inputs, not as data-migration obligations.
6. Mark facts, inferences, accepted decisions and unknowns separately.
7. Treat missing referential/ownership constraints in the current model as fresh-schema requirements. Do not make anomalies in current production rows a gate for a greenfield target.

# Acceptance and evidence

- All current components are mapped exactly once or have an explicit multi-stage reason.
- No missing money-moving, secret, greenfield bootstrap/schema, browser, performance or operations boundary remains hidden.
- Contract impact is classified with source anchors.
- Run `uv run python -m tools.docs.generate_project_map`, its `--check` form, docs-index generation/check, and `git diff --check` when applicable.
- Obtain one read-only architecture review for the Stage `00` report only if the active repository contract requires a stage artifact review; do not create a reviewer chain.

# Ledger, report and stop rules

Update the ledger after validation and before the Russian final report. Record created/modified/deleted/outside paths and foreign exclusions. Mark `blocked` for any uncovered component, contradictory accepted decision, unknown critical trust boundary or missing source evidence. Current production-row inconsistencies are retained only as non-gating design evidence because current data is explicitly outside the greenfield target. Allow Stage `01` and `02` only after `00=accepted`.
