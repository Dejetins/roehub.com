---
prompt_name: 05-wave2-wave3-completion
repo: roehub.com
branch: main
scope: "Complete every P2/P3 row with semantic acceptance or a scoped direct/resource improvement."
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
target_ids: [S001, S012, S024, S025, S026, S031, S032, S033, S034, S035, S036, S038, S039, S040, S042, S044, S046, S070, S071, S073, S074, S084]
hard_requirements: {update_stage_ledger: true, source_hash_precondition: true, rollback_before_mutation: true, managed_cache_write_forbidden: true, dormant_activation_forbidden: true, preserve_foreign_changes: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 05"
  - "skill-creator: direct/resource contracts"
  - "contract-impact-analysis: acceptance changes"
  - "backend-quality-gates: semantic fixtures"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/.codex/skills/backend-performance-evidence/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/backend-quality-gates/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/contract-impact-analysis/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/data-analytics-methodology/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/topological-data-analysis/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/.codex-plugin/plugin.json
    - .codex/skill-system/catalog-v1.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/05-wave2-wave3-completion.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "exact overlay resource paths and direct one-hop references for target IDs in ownership-v1"
    - ".codex/skill-system/evidence/stage-05-*.json"
proof_boundary: {product_runtime: "N/A", reason: "skill semantic acceptance only"}
---

# Stage 05 — Wave 2 and Wave 3 completion

Close every listed row. Use accepted_no_change only when the current body already satisfies the proposal and record anchored evidence. For artifact templates, implement deterministic semantic gates rather than verbose prose. Snapshot direct files first; keep all plugin corrections as resolver resources and preserve dormant activation.

Run structural, semantic and result-envelope fixtures. Update catalog/evidence, ledger, then report. No cache edit or plugin installation.
