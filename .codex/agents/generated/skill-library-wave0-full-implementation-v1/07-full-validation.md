---
prompt_name: 07-full-validation
repo: roehub.com
branch: main
scope: "Validate baseline 85, current inventory 96, effective contracts, installation and fresh-process routing."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: reload gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact validation scope}
    - {path: .codex/skill-system/catalog-v1.json, why: expected effective state}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S001-S085, S086-S096]
hard_requirements: {update_stage_ledger: true, validation_first: true, preserve_foreign_changes: true, live_external_mutation_forbidden: true, fresh_process_proof: true}
skill_routing:
  - "staged-plan-runner: enforce reload gate"
  - "backend-quality-gates: test triage"
  - "contract-impact-analysis: final compatibility"
  - "architecture-review: evidence discipline; no second independent cold-head"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - .codex/skill-system/catalog-v1.json
    - .codex/skill-system/evidence/stage-07-validation.json
    - .codex/skill-system/evidence/stage-07-fresh-process.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/07-full-validation.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "narrow fixes only to exact failing paths in ownership-v1, with rollback and report entry"
proof_boundary: {product_runtime: "N/A", reason: "Codex skill discovery, not Roehub runtime"}
---

# Stage 07 — Full validation

Rebuild recursive current inventory and require baseline `85/85`, current `96/96`, supplemental `11/11`, zero unexplained drift. Verify ownership and rollback manifests, catalog hash parity, resolver aliases, activation matrix, result-contract providers, every direct/effective quick_validate, family fixtures, representative emitted `skill-result/v1` envelopes and docs gates.

Run a separate sanitized read-only proof using `codex exec --ephemeral -s read-only -C /Users/daniildegtyarev/Projects/roehub.com`. It must report plugin/catalog/resolver routing for representative public, internal, dormant, alias and capability-absence cases without paid/external/production mutation or secrets. Compare session exposure before/after and prove dormant skills were not activated.

Only narrow evidence-driven fixes are allowed. Update catalog/evidence, ledger, then the report.
