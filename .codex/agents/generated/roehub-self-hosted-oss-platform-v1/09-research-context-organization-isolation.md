---
prompt_name: 09-research-context-organization-isolation
repo: roehub.com
scope: "Add organization ownership and isolation to market data access, indicators, backtests, artifacts and optimization without changing deterministic compute semantics."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "09", prerequisites: ["05", "08"], previous_stage_gate: "Stages 05 and 08 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: tenancy/performance rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: research tenancy and state ownership}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: organization and secret prerequisites}
  task_entrypoints:
    - {path: src/trading/contexts/market_data/, why: market data ownership and queries}
    - {path: src/trading/contexts/indicators/, why: deterministic definitions/compute}
    - {path: src/trading/contexts/backtest/, why: jobs and results}
    - {path: src/trading/contexts/backtest_artifacts/, why: artifact ownership}
    - {path: src/trading/contexts/optimize/, why: optimization ownership}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before ownership changes, reason: DTO/persistence/request identity semantics}
  - {skill: backend-quality-gates, timing: verification, reason: focused context gates}
  - {skill: backend-performance-evidence, timing: if a verified hot path changes, reason: preserve deterministic compute performance}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/market_data/, src/trading/contexts/indicators/, src/trading/contexts/backtest/, src/trading/contexts/backtest_artifacts/, src/trading/contexts/optimize/, apps/api/, apps/scheduler/, apps/worker/backtest_job_runner/, migrations/, tests/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated benchmark and feature work
file_manifest:
  expected_primary_touches: [src/trading/contexts/market_data/, src/trading/contexts/indicators/, src/trading/contexts/backtest/, src/trading/contexts/backtest_artifacts/, src/trading/contexts/optimize/, migrations/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/09-research-context-organization-isolation.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/api/, apps/scheduler/, apps/worker/backtest_job_runner/, docs/architecture/README.md]
validation_strategy: {depth: performance, acceptance_surfaces: [two-organization API/database isolation, ClickHouse query scoping, deterministic result parity, benchmark comparison, fresh ownership invariants]}
proof_boundary: {label: N/A, exclusions: [plugin runtime, full container profile, production data mutation]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Scope all research resources and operations to organizations while preserving shared immutable market data where explicitly designed and preserving exact compute outputs.

# Requirements

- Classify each table/resource as installation-shared, organization-owned or user-owned within an organization.
- Add server-derived organization scope to jobs, configs, results, reports and artifact metadata.
- Do not duplicate canonical market candles per organization; enforce authorization at query/use-case boundaries.
- Create all mutable research resources with explicit organization ownership; no legacy owner backfill is in scope.
- Use an organization-scoped, versioned request/dedupe namespace from the first v1 write. Content hashes remain organization-neutral only when they describe identical bytes or deterministic compute content.
- Do not regress verified fast paths without comparable benchmark evidence.

# Validation and acceptance

Run focused ruff/pyright/pytest, disposable PostgreSQL and ClickHouse integration with two organizations, negative cross-org API/repository cases, deterministic golden parity, and comparable benchmark evidence for touched hot paths. Record shared-data rules and any performance unknowns.

# Stop rules

Block on cross-org leakage, implicit client-trusted scope, orphan/cross-owner fresh records, changed deterministic results, dependency on current production data or unmeasured regression on a verified hot path. Update ledger after evidence.
