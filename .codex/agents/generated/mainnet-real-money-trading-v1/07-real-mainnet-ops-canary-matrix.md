---
prompt_name: mainnet-real-money-trading-v1-07-real-mainnet-ops-canary-matrix
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Run real bounded mainnet ops canary orders and immediate auto-close across Binance/Bybit spot/futures."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "canary matrix policy"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage gate"
  task_entrypoints:
    - path: apps/exchange_execution
      why: "mainnet order submit/close boundary"
      inspect_symbols: ["submit", "status", "fills", "reconciliation"]
    - path: src/trading/contexts/live_execution
      why: "source event, intent, risk, ledger"
      inspect_symbols: ["ops_test", "risk", "order"]
    - path: docs/runbooks/exchange-execution.md
      why: "canary and rollback protocol"
      inspect_symbols: ["Safe canary", "Rollback", "Unknown"]
  conditional_bundles:
    ui_browser:
      read_when: "public /strategies proof is required"
      paths: ["apps/web", "apps/api/wiring/modules/ui_strategies_dashboard.py"]
  consult_if_needed:
    - path: docs/runbooks/prod-dashboard-metrics-reference-ru.md
      read_when: "latency/slippage metric interpretation is unclear"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_06_must_be_accepted: true, explicit_canary_window_required: true, auto_close_required: true}
task_toggles: {allow_real_mainnet_orders: true, max_order_notional_usdt: 15, auto_close_immediately: true}
skill_routing:
  - skill: backend-performance-evidence
    use_when: "measuring signal/intent/dispatch/submit/fill latency"
    timing: during verification
    reason: "latency evidence must be comparable and segment-bounded"
  - skill: root-cause-debugging
    use_when: "auto-close, fill, or reconciliation fails"
    timing: if blocker
    reason: "unknown state must be reconciled before retry"
target_envs: ["macstudio", "Binance mainnet", "Bybit mainnet"]
required_literals: ["15 USDT", "auto-close", "No blind retry after unknown state"]
non_goals: ["Do not enable strategy producer live mode.", "Do not run repeated or soak trading."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 07 canary harness/report/ledger only"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "canary_matrix", "latency_slippage", "residual_state", "files", "next_stage"]}
quality_gates:
  - cmd: "uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes if code changed"
  - cmd: "uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests"
    expect: "passes if code changed"
  - cmd: "uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes if code changed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["real mainnet orders", "auto-close", "DB ledger", "Redis", "Prometheus", "user alert", "browser/API"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/07-real-mainnet-ops-canary-matrix.md
proof_boundary:
  required_when: "real canary uses deployed code"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then real runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["API keys", "secrets", "signed payloads", "cookies", "raw provider payloads"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "07", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/exchange_execution/**", "src/trading/contexts/live_execution/**", "scripts/**"]
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/07-real-mainnet-ops-canary-matrix.md", "docs/runbooks/exchange-execution.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md", "/opt/roehub/state/live_execution/mainnet-real-money-trading-v1/"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/07-real-mainnet-ops-canary-matrix.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["apps/exchange_execution", "src/trading/contexts/live_execution", "scripts", "docs/runbooks/exchange-execution.md", "docs/architecture/README.md"]
safety_notes: ["This stage moves real money. Stop immediately on unknown state, auto-close failure, alert failure, cap breach, or unexpected residual exposure."]
---

# Task

Run the real mainnet ops canary matrix using bounded `ops_test`/internal source events, not strategy producer.

## Context / Current State

This is the first stage allowed to submit real mainnet orders. User approval for the canary window is mandatory.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `06` is `accepted`. If not, update Stage `07` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: this is the first real order stage; user canary-window approval, caps, auto-close, reconciliation, alerts, and residual-state proof are mandatory.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `06 accepted`; if not, write Stage `07 blocked`, update the ledger, and stop.
- Record `User required before start: explicit bounded canary window approval`.
- If approval is absent, mark blocked.
- Execute required matrix with market orders `<=15 USDT`: Binance spot long, Binance futures long/short, Bybit spot long, Bybit futures long/short.
- Immediately auto-close every opened position.
- Measure latency and slippage from durable timestamps/fills.
- Prove user/operator alerts.
- Stop on first unknown/auto-close failure and reconcile before any retry.

## Requirements (Should)

- Run canaries sequentially, not concurrently.
- Keep global cap `60 USDT` until allocation manifest says otherwise.

## Requirements (Nice-to-have)

- Capture browser `/strategies` outcome proof for each canary row.

# Context acquisition protocol

Read plan/ledger/runbook first. Do not inspect UI unless browser proof requires missing outcome fields.

Reading budget: target `<= 12 files`; expand only for canary harness defects or reconciliation blockers.

# Reading manifest

Use `ui_browser` only if browser/API proof needs implementation or route context.

# Work plan (agent should follow)

1. Verify gates and user approval.
2. Snapshot DB/Redis/metrics/open orders/positions before canary.
3. Run canary rows sequentially.
4. After each row, verify fill, close, reconciliation, alert, latency/slippage, no residual exposure.
5. Stop and mark blocked on any unknown/failure.
6. Final snapshot DB/Redis/metrics/browser.
7. Update report/ledger.

# Acceptance criteria (Definition of Done)

- All required canary rows accepted.
- Residual open orders/positions are zero or expected closed state.
- No unexplained retry/DLQ/unknown growth.
- User alert proof exists.
- Stage `08` opens only after all rows pass.

# Implementation constraints

- No concurrent canaries.
- No order > `15 USDT`.
- No raw provider payloads.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No strategy producer live mode.
- No long soak.

# Quality gates (must run and pass)

Run relevant local gates if code changed, then real runtime canary proof. Tests are not acceptance.

# Final output: report format (strict)

Russian report with canary matrix, latency/slippage table, residual state, alert proof, file manifest, blockers, next stage.
