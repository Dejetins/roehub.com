---
prompt_name: 04-wave1-managed-overlay
repo: roehub.com
branch: main
scope: "Implement all remaining managed P1 rows as non-cache overlay resources or deprecations."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact IDs and paths}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md, why: rows}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [S002, S003, S004, S005, S006, S007, S009, S010, S011, S015, S016, S017, S018, S020, S021, S022, S027, S028, S029, S037, S041, S043, S045, S049, S050, S051, S054, S055, S056, S057, S058, S062]
hard_requirements: {update_stage_ledger: true, source_hash_precondition: true, managed_cache_write_forbidden: true, dormant_activation_forbidden: true, preserve_foreign_changes: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 04"
  - "plugin-creator: validate bounded plugin resources"
  - "skill-creator: author contracts"
  - "contract-impact-analysis: aliases and exposure"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/.codex-plugin/plugin.json
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S002/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S003/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S004/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S009/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S010/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S011/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S015/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S016/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S017/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S018/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S020/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S021/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S022/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S027/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S028/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S029/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S037/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S041/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S043/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S045/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S049/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S050/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S051/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S054/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S055/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S056/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S057/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S058/SKILL.md
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/S062/SKILL.md
    - .codex/skill-system/catalog-v1.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/04-wave1-managed-overlay.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "exact resource references listed for target IDs in ownership-v1"
    - ".codex/skill-system/evidence/stage-04-*.json"
proof_boundary: {product_runtime: "N/A", reason: "managed replacements are source resources, not cache edits"}
---

# Stage 04 — Wave 1 managed overlay

Implement all listed P1 proposals. S005/S006/S007 are deprecated to S020/S021/S022 and receive no resource body. Public logical skills, internal Product Design helpers and dormant HF/template resources must retain their Stage 00 activation decisions. Corrected resources never become top-level plugin skills.

Validate aliases, names, relations, progressive disclosure, artifact semantics and family fixtures. Update catalog/evidence, ledger, then report. Never edit cache or install the plugin in this stage.
