---
prompt_name: 03-wave1-direct-repairs
repo: roehub.com
branch: main
scope: "Repair direct P1 skills S063,S064,S068,S069,S072,S076,S077,S079,S080,S082,S083."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact paths}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md, why: rows}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S063, S064, S068, S069, S072, S076, S077, S079, S080, S082, S083]
hard_requirements: {update_stage_ledger: true, source_hash_precondition: true, rollback_before_mutation: true, preserve_foreign_changes: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 03"
  - "skill-creator: edit and split skills"
  - "contract-impact-analysis: routing and output contracts"
  - "backend-quality-gates: deterministic gates"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/.codex/skills/.system/imagegen/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/.system/openai-docs/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/architecture-design/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/architecture-review/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/browser-qa-evidence/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/numba-jit-performance/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/playwright/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/production-risk-review/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/prompt-manager/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/root-cause-debugging/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/staged-plan-runner/SKILL.md
    - .codex/skill-system/catalog-v1.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/03-wave1-direct-repairs.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "exact one-hop references and agents/openai.yaml paths listed for target IDs in ownership-v1"
    - ".codex/skill-system/evidence/stage-03-*.json"
proof_boundary: {product_runtime: "N/A", reason: "skill contracts only"}
---

# Stage 03 — Wave 1 direct repairs

Fail closed on ownership/hash drift and snapshot targets first. Implement every target proposal, including portable architecture/prompt cores with Roehub one-hop profiles, single permitted reviewer contract, browser readiness rather than ship verdict, deterministic Numba evidence, secret-safe Playwright, exact review base, diagnose-only versus fix-authorized debugging and strict stage schema. S065/S066 are already bootstrap-complete and must not be edited here.

Run structural/family/fixture validation; update catalog/evidence, ledger, then report.
