---
prompt_name: 00-contract-and-catalog-foundation
repo: roehub.com
branch: main
scope: "Build the dual inventory, contracts, resolver, exact ownership, rollback, fixtures, and bootstrap repairs for S065/S066."
model_preferences: {primary_agent_model: gpt-5.5, reasoning_effort: xhigh}
language: {implementation: python_docs_skills, agent_report: ru}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: repository contract}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1.md, why: target architecture}
    - {path: docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md, why: current gate}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md, why: immutable baseline}
    - {path: docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md, why: row requirements}
ownership_manifest: .codex/skill-system/ownership-v1.json
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  source_hash_precondition: true
  preserve_foreign_changes: true
  secrets_in_artifacts: false
  recursive_hidden_inventory: true
  rollback_before_mutation: true
skill_routing:
  - "staged-plan-runner: enforce the ledger gate"
  - "skill-creator: bootstrap S066 and authoring contract"
  - "plugin-creator: bootstrap S065 and overlay packaging contract"
  - "contract-impact-analysis: classify contracts"
  - "backend-quality-gates: run focused Python gates"
branch_policy: {default_branch: main, separate_branch_allowed: false, worktree_allowed: false, stash_allowed: false}
file_manifest:
  expected_primary_touches:
    - tools/codex_quality_benchmark/skill_audit.py
    - tools/codex_quality_benchmark/skill_contract.py
    - tools/codex_quality_benchmark/skill_catalog.py
    - tools/codex_quality_benchmark/skill_contract_fixtures.py
    - tools/codex_quality_benchmark/schemas/skill-spec-v1.schema.json
    - tools/codex_quality_benchmark/schemas/skill-result-v1.schema.json
    - tools/codex_quality_benchmark/schemas/skill-contract-case-result-v1.schema.json
    - tests/unit/tools/test_codex_skill_contract.py
    - tests/unit/tools/test_codex_skill_catalog.py
    - tests/unit/tools/test_codex_skill_contract_fixtures.py
    - .codex/skill-system/catalog-v1.json
    - .codex/skill-system/policy-v1.json
    - .codex/skill-system/ownership-v1.json
    - .codex/skill-system/fixtures/skill-contract-cases-v1.json
    - .codex/skill-system/rollback/manifest-v1.json
    - /Users/daniildegtyarev/.codex/skill-system/catalog-v1.json
    - /Users/daniildegtyarev/.codex/skills/.system/plugin-creator/SKILL.md
    - /Users/daniildegtyarev/.codex/skills/.system/skill-creator/SKILL.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/00-contract-and-catalog-foundation.md
    - docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md
  possible_secondary_touches:
    - ".codex/skill-system/rollback/blobs/<sha256>.md, only from ownership-v1"
    - ".codex/skill-system/evidence/stage-00-*.json"
    - "/Users/daniildegtyarev/.codex/skills/.system/plugin-creator/references/skill-system-contract-v1.md"
    - "/Users/daniildegtyarev/.codex/skills/.system/skill-creator/references/skill-system-contract-v1.md"
proof_boundary: {product_runtime: "N/A", reason: "local skill system only"}
---

# Stage 00 — Contract and catalog foundation

Confirm the ledger still blocks mutations pending local cold-head follow-up. After it is released, perform only this stage.

## Must

- Preserve immutable audit baseline `S001–S085`; build a recursive hidden-path current inventory of `96` canonical paths and assign/classify `S086–S096`.
- Fix discovery so nested Playwright and other hidden dependency skills are found.
- Generate exact per-ID ownership including source/effective path, stage, operation, before hash, discovery state, exposure and activation policy.
- Before direct edits, create content-addressed rollback blobs and verify the manifest. Scan durable snapshots for forbidden secret patterns.
- Implement schemas/parsers for `skill-spec/v1`, `skill-result/v1` and contract fixture results.
- Implement the global-catalog/repo-snapshot resolver and hash-parity check. Do not claim it hides loader entries.
- Create deterministic fixtures covering budget, target, destination, visibility, authority, unknown provider state, secret evidence, read-only intent, dirty main, capability absence and alias resolution.
- Repair `S065` and `S066` first, after snapshots, so later stages use corrected authoring contracts.
- Do not edit any other source skill, plugin cache, marketplace or AGENTS file.
- Validate, update catalog/evidence, update ledger, then generate the durable report from accepted evidence.

## Acceptance

Baseline catalog `85/85`; current inventory `96/96`; supplemental `11/11`; ownership complete; rollback verifier passes; schemas and fixtures parse; S065/S066 validate; global/repo catalog hashes match; no dormant activation.

## Gates

`uv run ruff check tools/codex_quality_benchmark tests/unit/tools`

`uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py tests/unit/tools/test_codex_skill_contract.py tests/unit/tools/test_codex_skill_catalog.py tests/unit/tools/test_codex_skill_contract_fixtures.py`

`uv run python -m tools.docs.generate_docs_index --check`

`git diff --check`

Report in Russian with stage/status, exact IDs, file manifest, gates, real-boundary evidence, contract impact, ledger handoff and residual risks.
