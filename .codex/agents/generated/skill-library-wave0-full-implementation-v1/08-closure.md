---
prompt_name: 08-closure
repo: roehub.com
branch: main
scope: "Close all baseline recommendations and reconcile the current inventory, docs, compatibility and recovery evidence."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: completion criteria}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: final gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact scope}
    - {path: .codex/skill-system/catalog-v1.json, why: final state}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S001-S085, S086-S096]
hard_requirements: {update_stage_ledger: true, preserve_foreign_changes: true, no_pending_rows: true, no_unverified_claims: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 08"
  - "pre-ship-gate: report-only readiness"
  - "contract-impact-analysis: final matrix"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1.md
    - docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/08-closure.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
    - docs/architecture/README.md
  possible_secondary_touches:
    - ".codex/skill-system/evidence/stage-08-final-reconciliation.json"
proof_boundary: {product_runtime: "N/A", reason: "documentation and skill-system closure"}
---

# Stage 08 — Closure

Reproduce Stage 07 critical evidence from durable artifacts. Require `85/85` baseline terminal, `96/96` current inventory classified, no pending/blocked required row, valid recovery evidence and explicit proof boundaries. Update the historical backlog with implementation references without erasing findings.

Run docs continuity/index and diff checks. Update ledger to completed before generating the final report from ledger/evidence. Report actual direct/resource/deprecated/no-change facts, contract impact, exposure behavior, fresh-process boundary and upstream drift risk. Do not commit, push, deploy or edit managed cache.
