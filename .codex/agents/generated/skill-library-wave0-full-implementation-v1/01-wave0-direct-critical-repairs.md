---
prompt_name: 01-wave0-direct-critical-repairs
repo: roehub.com
branch: main
scope: "Repair direct P0 skills S067,S075,S078,S081,S085."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact ownership and hashes}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md, why: required rows}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S067, S075, S078, S081, S085]
hard_requirements: {update_stage_ledger: true, source_hash_precondition: true, rollback_before_mutation: true, preserve_foreign_changes: true, secrets_in_artifacts: false}
skill_routing:
  - "staged-plan-runner: enforce Stage 01"
  - "skill-creator: edit and split skills"
  - "contract-impact-analysis: classify behavior changes"
  - "backend-quality-gates: verify fixtures"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/.codex/skills/.system/skill-installer/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/last30days/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/pre-ship-gate/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/SKILL.md
    - .codex/skill-system/catalog-v1.json
    - .codex/skill-system/rollback/manifest-v1.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/01-wave0-direct-critical-repairs.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "exact one-hop references and agents/openai.yaml paths listed for S067,S075,S078,S081,S085 in ownership-v1"
    - ".codex/skill-system/evidence/stage-01-*.json"
proof_boundary: {product_runtime: "N/A", reason: "skill contracts only"}
---

# Stage 01 — Wave 0 direct critical repairs

Read ownership-v1 and fail closed on any path/hash mismatch. Snapshot every existing target before mutation.

Implement the audit proposals exactly: S067 provenance/system-skill/network gates; S075 valid concise portable router with execution/synthesis/security one-hop references and no cookie/policy override; S078 valid quoted metadata and strict report-only shared-main evidence; S081 conditional delivery prerequisites, deploy relevance and no-runtime terminal; S085 compact stack-aware router with opt-in persist/install and browser acceptance.

Run structural validation and the required deterministic fixtures. Then update catalog/evidence, ledger, and finally the stage report. No managed cache, plugin, AGENTS, marketplace, Git publication or Roehub runtime change.
