---
prompt_name: mainnet-real-money-trading-v1-09-strategy-driven-mainnet-canaries
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Run scoped automatic strategy-driven real mainnet canaries and measure signal-to-fill path."
language: {implementation: python, agent_report: ru}
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "strategy canary policy"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth"
  task_entrypoints:
    - path: apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
      why: "strategy signal production"
      inspect_symbols: ["StrategySignal", "source event", "checkpoint"]
    - path: apps/exchange_execution
      why: "real order submit and close"
      inspect_symbols: ["submit", "fills", "reconciliation"]
    - path: apps/api/wiring/modules/ui_strategies_dashboard.py
      why: "browser/API proof for outcomes"
      inspect_symbols: ["Execution outcomes", "latency"]
  conditional_bundles:
    backtest_strategy:
      read_when: "canary strategy selection or launch path is unclear"
      paths: ["docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md", "src/trading/contexts/strategy"]
    metrics:
      read_when: "latency metrics are missing or stale"
      paths: ["docs/runbooks/prod-dashboard-metrics-reference-ru.md", "infra/macos/prometheus/rules"]
  consult_if_needed:
    - path: docs/runbooks/exchange-execution.md
      read_when: "close/reconciliation/unknown handling is unclear"
style_references: [".codex/agents/stage_execution_ledger_template.md"]
hard_requirements: {stage_08_must_be_accepted: true, explicit_strategy_window_required: true, auto_close_required: true}
task_toggles: {allow_real_strategy_mainnet_orders: true, max_order_notional_usdt: 15, minimal_observation: true}
skill_routing:
  - skill: backend-performance-evidence
    use_when: "measuring candle/signal/source/intent/submit/fill latency"
    timing: during verification
    reason: "segment evidence must be comparable and durable"
  - skill: browser-qa-evidence
    use_when: "/strategies outcome proof is required"
    timing: during verification
    reason: "prove user-visible execution path"
target_envs: ["macstudio", "roehub.com", "Binance mainnet", "Bybit mainnet"]
required_literals: ["automatic under agent supervision", "15 USDT", "auto-close"]
non_goals: ["Do not run long soak.", "Do not enable unscoped live strategies."]
branch_policy: {default_branch: main, separate_branch_allowed: false, single_allowed_branch: null, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false, approval_required_for_branch_or_worktree: true}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 09 strategy canary/report/ledger only"]
  foreign_changes_policy: "preserve unrelated"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format: {language: ru, sections: ["status", "strategy_canaries", "latency", "alerts", "residual_state", "next_stage"]}
quality_gates:
  - cmd: "uv run ruff check apps/worker/strategy_live_runner apps/exchange_execution src/trading/contexts tests"
    expect: "passes if code changed"
  - cmd: "uv run pyright apps/worker/strategy_live_runner apps/exchange_execution src/trading/contexts tests"
    expect: "passes if code changed"
  - cmd: "uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/live_execution tests/unit/apps"
    expect: "passes if code changed"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["live candles", "strategy signals", "real mainnet orders", "fills/reconciliation", "alerts", "browser/API", "Prometheus"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/09-strategy-driven-mainnet-canaries.md
proof_boundary:
  required_when: "strategy canary uses deployed code"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires origin/main, green CI, deploy/sync, then runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["API keys", "secrets", "signed payloads", "cookies", "raw provider payloads"]
remote_command_quoting: {applies_when: "SSH uses SQL/JSON", required_pattern: "quoted heredoc/stdin", forbidden_pattern: "nested inline SQL/JSON", temporary_files_allowed_only_when_task_requires_durable_artifact: true}
stage_execution_ledger: {path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, current_stage: "09", required_update: true, template: .codex/agents/stage_execution_ledger_template.md}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md, prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/, stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["apps/worker/strategy_live_runner/**", "apps/exchange_execution/**", "src/trading/contexts/**", "apps/api/**", "apps/web/**"]
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/09-strategy-driven-mainnet-canaries.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md", "/opt/roehub/state/live_execution/mainnet-real-money-trading-v1/"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/09-strategy-driven-mainnet-canaries.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["apps/worker/strategy_live_runner", "apps/exchange_execution", "src/trading/contexts", "apps/api", "apps/web", "docs/architecture/README.md"]
safety_notes: ["This stage moves real money automatically. Stop on first unknown, cap breach, alert failure, or residual exposure."]
---

# Task

Run scoped automatic strategy-driven mainnet canaries and measure the full signal-to-fill path.

## Context / Current State

User does not want manual confirmation per order. The stage must use scoped allowlists, caps, kill switches, alerts, and agent supervision.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `08` is `accepted`. If not, update Stage `09` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: strategy-driven orders are automatic under agent supervision, but only inside the scoped strategy/exchange/market allowlist and canary window recorded in the ledger.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before implementation/execution, read the ledger and confirm Stage `08 accepted`; if not, mark Stage `09 blocked`.
- Record `User required before start: explicit scoped automatic strategy canary window approval`.
- If approval is missing, stop blocked.
- Select or create a canary strategy/run that can produce a real signal on live candles with minimal observation.
- Execute required market surfaces with market orders `<=15 USDT`.
- Auto-close positions immediately after proof.
- Measure `candle close -> signal -> source event -> intent -> risk -> Redis -> exchange ack -> fill -> reconciliation -> alert`.
- Prove no duplicates and no residual exposure.

## Requirements (Should)

- Prefer sequential market-surface execution.
- Keep global cap strict.

## Requirements (Nice-to-have)

- Produce a compact latency table with p50/p95/p99 where enough samples exist.

# Context acquisition protocol

Read plan/ledger and producer/execution entrypoints first. Expand only for strategy selection, UI evidence, or latency blocker.

Reading budget: target `<= 14 files`; expand only for signal timing, adapter, or reconciliation issues.

# Reading manifest

Use `backtest_strategy` if selecting launchable strategy is unclear. Use `metrics` if latency metrics are missing.

# Work plan (agent should follow)

1. Verify gates and user approval.
2. Snapshot runtime state.
3. Configure scoped live strategy canary.
4. Wait minimal bounded time for signal per market surface.
5. Execute order, auto-close, reconcile, alert, measure.
6. Stop on unknown/failure.
7. Collect browser/API/DB/Redis/Prometheus proof.
8. Cleanup scopes and update report/ledger.

# Acceptance criteria (Definition of Done)

- Required market surfaces show strategy signal -> real order -> close -> matched reconciliation -> user alert.
- No duplicate signal/source/order side effects.
- No residual open orders/positions.
- Stage `10` opens only after clean closure evidence.

# Implementation constraints

- No broad allowlist.
- No long soak.
- No provider payload leakage.

# Files to indicate (expected touched areas)

Final report must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No ML-agent mainnet trading.
- No multi-symbol expansion.

# Quality gates (must run and pass)

Run focused gates if code changed plus real runtime/browser/exchange evidence. Tests are never acceptance.

# Final output: report format (strict)

Russian report with canary matrix, latency breakdown, alerts, residual state, file manifest, blockers, next stage.
