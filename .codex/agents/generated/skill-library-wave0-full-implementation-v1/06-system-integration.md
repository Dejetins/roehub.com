---
prompt_name: 06-system-integration
repo: roehub.com
branch: main
scope: "Install the validated resolver resource plugin and connect global/repo routing to the canonical global catalog."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: plan}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: gate}
    - {path: .codex/skill-system/ownership-v1.json, why: exact integration paths}
    - {path: .codex/skill-system/catalog-v1.json, why: catalog snapshot}
ownership_manifest: .codex/skill-system/ownership-v1.json
target_ids: [effective_catalog]
hard_requirements: {update_stage_ledger: true, rollback_before_mutation: true, preserve_foreign_changes: true, managed_cache_write_forbidden: true, fresh_process_required_next: true}
skill_routing:
  - "staged-plan-runner: enforce Stage 06"
  - "plugin-creator: cachebuster, validation and reinstall"
  - "contract-impact-analysis: public discovery and policy"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - /Users/daniildegtyarev/.codex/AGENTS.md
    - /Users/daniildegtyarev/Projects/roehub.com/.codex/AGENTS.md
    - /Users/daniildegtyarev/.codex/skill-system/catalog-v1.json
    - /Users/daniildegtyarev/Projects/roehub.com/.codex/skill-system/catalog-v1.json
    - /Users/daniildegtyarev/plugins/codex-skill-system-overrides/.codex-plugin/plugin.json
    - /Users/daniildegtyarev/.agents/plugins/marketplace.json
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/06-system-integration.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - "Codex CLI-managed installation state reported by codex plugin add/list; never edited directly"
    - ".codex/skill-system/evidence/stage-06-plugin-reload-receipt.json"
proof_boundary: {product_runtime: "N/A", reason: "Codex local discovery boundary only"}
---

# Stage 06 — System integration

Validate catalog parity and plugin source first. Snapshot AGENTS, marketplace and plugin manifest through rollback-v1. Add a concise global policy that requires resolve-skill for aliases/conflicts and reads the returned effective resource; keep repo precedence and root pointer semantics. Do not claim the catalog filters the loader.

Use plugin-creator cachebuster and reinstall through `codex plugin add codex-skill-system-overrides@personal`. Record marketplace, plugin version, source path, installed state, loaded skill list and exposure delta. The plugin contains resources, not extra discovered skills.

Set the ledger gate to `fresh_process_required`; Stage 07 is allowed only with a separate sanitized fresh-process proof. Update ledger, then report.
