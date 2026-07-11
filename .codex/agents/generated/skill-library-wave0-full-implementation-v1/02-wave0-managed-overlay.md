---
prompt_name: 02-wave0-managed-overlay
repo: roehub.com
branch: main
scope: "Package P0 managed improvements without editing cache or activating dormant skills."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact ownership}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md, why: required rows}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S008, S013, S014, S019, S023, S030, S047, S048, S052, S053, S059, S060, S061]
hard_requirements: {update_stage_ledger: true, source_hash_precondition: true, rollback_before_mutation: true, preserve_foreign_changes: true, managed_cache_write_forbidden: true, dormant_activation_forbidden: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 02"
  - "plugin-creator: create and validate source plugin only"
  - "skill-creator: author corrected resource contracts"
  - "contract-impact-analysis: exposure and behavior changes"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/.codex-plugin/plugin.json
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S008/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S013/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S014/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S019/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S023/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S030/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S047/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S048/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S052/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S053/SKILL.md
    - .codex/skill-system/catalog-v1.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/02-wave0-managed-overlay.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "exact one-hop resource references listed in ownership-v1"
    - ".codex/skill-system/evidence/stage-02-*.json"
proof_boundary: {product_runtime: "N/A", reason: "source plugin packaging only; installation is Stage 06"}
---

# Stage 02 — Wave 0 managed overlay

Use only exact overlay resource paths from ownership-v1. Do not create top-level plugin skills in this stage: corrected copies are resolver resources, so dormant HF/templates and internal helpers cannot become session-exposed. Canonicalize S008 to S023 and S059/S060/S061 to S077 as specified; retain row-level evidence. Implement paid-job budget/target gates, legal high-stakes boundary, Product Design selection/licensing/privacy/consent gates, and Playwright redaction behavior.

Validate the source plugin and every resource; do not install yet. Update catalog/evidence, ledger, then report. Never edit managed cache.
