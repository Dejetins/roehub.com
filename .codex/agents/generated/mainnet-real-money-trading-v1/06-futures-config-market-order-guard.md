---
prompt_name: mainnet-real-money-trading-v1-06-futures-config-market-order-guard
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Apply/read back safe mainnet futures config and prove market-order guard before canary orders."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "futures and market-order policy"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: apps/exchange_execution
      why: "native exchange adapters and guards"
      inspect_symbols: ["futures", "leverage", "market"]
    - path: src/trading/contexts/live_execution
      why: "order model, risk, account config guard"
      inspect_symbols: ["market", "futures", "config_guard"]
    - path: docs/runbooks/exchange-execution.md
      why: "operator account config and rollback instructions"
      inspect_symbols: ["Rollback", "futures", "market"]
  conditional_bundles:
    official_docs:
      read_when: "provider endpoint semantics are implemented or questioned"
      paths:
        - "web: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info"
        - "web: https://bybit-exchange.github.io/docs/v5/order/create-order"
  consult_if_needed:
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      read_when: "connection/account config readiness is unclear"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_05_must_be_accepted: true, explicit_user_futures_config_required: true, no_strategy_order: true}
task_toggles: {allow_mainnet_futures_config_side_effect: true, allow_market_order_guard: true, allow_canary_order_submit: false}
skill_routing:
  - skill: contract-impact-analysis
    use_when: "futures config command or order request contract changes"
    timing: during investigation
    reason: "compatibility and rollback classification"
  - skill: publish-ci-deploy
    use_when: "accepted code/config changes need delivery"
    timing: before ship
    reason: "post-main runtime proof"
target_envs: ["macstudio", "Binance mainnet", "Bybit mainnet"]
required_literals: ["isolated 1x", "market orders only", "No strategy order submit"]
non_goals: ["Do not run strategy canary.", "Do not place open/close canary order yet unless explicitly scoped to config API dry-run."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 06 futures config/order guard/report/ledger only"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "user_required", "futures_config", "market_guard", "files", "next_stage"]}
quality_gates:
  - cmd: "uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes"
  - cmd: "uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["mainnet futures config read-back", "API/DB audit", "no strategy order", "metrics"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/06-futures-config-market-order-guard.md
proof_boundary:
  required_when: "futures config or changed runtime is verified"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["API keys", "secrets", "tokens", "raw provider payloads"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "06", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/exchange_execution/**", "src/trading/contexts/live_execution/**", "apps/api/**"]
    config_infra_migrations: ["alembic/versions/**", "configs/prod/**"]
    docs_runbooks: ["docs/runbooks/exchange-execution.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/06-futures-config-market-order-guard.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["apps/exchange_execution", "src/trading/contexts/live_execution", "docs/runbooks/exchange-execution.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/06-futures-config-market-order-guard.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["apps/api", "alembic/versions", "configs/prod", "docs/architecture/README.md"]
safety_notes: ["Futures config is a real exchange side effect; require user approval and no open orders/positions preflight."]
---

# Task

Implement/prove mainnet futures config command and market-order guard before real canary orders.

## Context / Current State

User allows platform to change futures leverage/margin mode as explicit user-selected parameter. Default is isolated `1x`. This is not allowed as hidden behavior inside order submit.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `05` is `accepted`. If not, update Stage `06` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: futures config mutation is explicit and audited; hidden auto-config inside order submit is forbidden.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `05 accepted`; if not, write Stage `06 blocked`, update the ledger, and stop.
- Record `User required before start: user approves default isolated 1x or explicit futures config`.
- Preflight no open orders/positions for the symbol before changing futures config.
- Apply/read back Binance and Bybit futures config or block with exact reason.
- Prove order model allows only `market` for mainnet v1.
- Prove no strategy order submit yet.

## Requirements (Should)

- Persist config audit evidence.
- Keep rollback/operator notes in runbook.

## Requirements (Nice-to-have)

- Show UI status for futures config readiness if existing UI supports it.

# Context acquisition protocol

Read plan/ledger and exchange-execution/live_execution entrypoints. Use official provider docs only when implementing endpoint semantics.

Reading budget: target `<= 12 files`; expand only for provider config failures or missing audit persistence.

# Reading manifest

Use `official_docs` as web/primary-source check when endpoint semantics matter.

# Work plan (agent should follow)

1. Verify prior stages and user approval.
2. Preflight no open orders/positions.
3. Implement/apply config command if missing.
4. Read back isolated `1x` or user-selected safe config.
5. Prove market-order-only guard and no strategy order.
6. Deliver and collect post-main proof.
7. Update report/ledger.

# Acceptance criteria (Definition of Done)

- Binance futures and Bybit futures config accepted or stage blocked with provider/user-action reason.
- Market-order guard proven.
- Stage `07` opens only after config/readiness is accepted.

# Implementation constraints

- Real exchange config side effects must be explicit and audited.
- No hidden auto-config in submit path.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No strategy-driven trading.
- No long soak.

# Quality gates (must run and pass)

Focused gates, docs index, real read-back/runtime proof, and secret-safe report.

# Final output: report format (strict)

Russian report: user approval, config commands/read-back, market guard, no-submit proof, file manifest, delivery, next stage.
