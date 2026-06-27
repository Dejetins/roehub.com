---
prompt_name: "Notifications v1 Stage 11 - Final Docs And Main Closure"
repo: "roehub.com"
branch: "main"
scope: "Close prompt pack, docs, ledger, runbooks and final delivery/readiness state"
language:
  implementation: "docs/ops"
  agent_report: "ru"
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
stage_execution_ledger:
  path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
  plan_doc: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
  current_stage: "11"
  required_update: true
validation_strategy:
  depth: "ci_deploy"
  acceptance_surfaces: ["docs", "ledger", "runbooks", "CI/deploy if code changed"]
proof_boundary:
  label: "post_main_production_runtime_proof"
  changed_code_production_claim_allowed: true
user_presence_required: "required only for product sign-off beyond smoke/test recipients"
context_sources:
  always_read:
    - path: ".codex/AGENTS.md"
      why: "repo contract"
    - path: ".codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md"
      why: "main/access contract"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1.md"
      why: "plan closure"
    - path: "docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md"
      why: "stage closure"
skill_routing:
  - skill: "pre-ship-gate"
    timing: "before final report"
    reason: "release-readiness and docs drift review"
  - skill: "publish-ci-deploy"
    timing: "if delivery/deploy is requested or incomplete"
    reason: "main CI/deploy/runtime proof"
  - skill: "architecture-review"
    timing: "before declaring closure"
    reason: "plan/docs/ledger completeness review"
expected_primary_touches:
  - "docs/architecture/notifications/"
  - "docs/runbooks/"
  - ".codex/agents/generated/web-execution-telegram-notifications-v1/"
  - ".codex/PLANS.md"
possible_secondary_touches:
  - "docs/architecture/README.md"
quality_gates:
  - "uv run python -m tools.docs.generate_docs_index --check"
  - "Focused gates for any code/config touched during closure"
  - "pre-ship-gate or publish-ci-deploy evidence if delivery/deploy is in scope"
---

# Task

Implement Stage `11`: final closure for Notifications v1. Reconcile docs, prompt pack, stage ledger, runbooks, CI/deploy/readiness state and residual risks.

User required before start: `required only for product sign-off beyond smoke/test recipients`.

## Requirements

- Verify Stage `10` accepted.
- Check every stage report exists or has an explicit blocked/superseded record.
- Check synthetic matrix has evidence or explicit blockers for every type.
- Check user-presence/access matrix is still accurate.
- Check runbooks cover unknown delivery, replay, route disable/rebind, missed reports, admin alerts and canary rollback.
- Do not turn smoke/test recipient proof into full product rollout approval.
- Update `.codex/PLANS.md` with current checkpoint and next work only if materially changed.

## Acceptance Criteria

- Ledger has no ambiguous stage state.
- Docs index check passes.
- Final report clearly separates accepted, blocked, deferred and requires-user-approval items.
- If code changed, CI/deploy/runtime proof is recorded through the correct proof boundary.

## Final Report

Respond in Russian with: final status by stage, delivery/deploy evidence, user sign-off status, residual risks, file manifest and next recommended action.
