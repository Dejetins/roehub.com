---
prompt_name: mainnet-real-money-trading-v1-02-mainnet-exchange-connections-readiness
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Validate Binance/Bybit mainnet exchange connections read-only before any order submit."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety rules"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "mainnet plan"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage gate"
  task_entrypoints:
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "exchange connection lifecycle and permissions"
      inspect_symbols: ["mainnet", "IP restriction", "trade"]
    - path: docs/runbooks/exchange-secret-management.md
      why: "custody and validation runbook"
      inspect_symbols: ["exchange-control", "mainnet", "validation"]
    - path: src/trading/contexts/live_execution/application
      why: "account projection/readiness use cases"
      inspect_symbols: ["ExchangeAccountProjectionService"]
  conditional_bundles:
    ui_settings:
      read_when: "browser-visible settings readiness changes are needed"
      paths:
        - apps/api/dto/ui_account.py
        - apps/web/templates/pages/settings.html
        - apps/web/dist/js/pages/settings.js
    exchange_adapters:
      read_when: "mainnet read-only validation adapter changes are needed"
      paths:
        - apps/exchange_execution
        - src/trading/contexts/exchange_control
  consult_if_needed:
    - path: docs/runbooks/exchange-execution.md
      read_when: "exchange-execution readiness interaction is unclear"
style_references:
  - .codex/agents/stage_execution_ledger_template.md
hard_requirements:
  stage_01_must_be_accepted: true
  read_only_only: true
  no_order_submit: true
task_toggles:
  allow_mainnet_readonly_calls: true
  allow_mainnet_orders: false
skill_routing:
  - skill: root-cause-debugging
    use_when: "a mainnet connection fails readiness unexpectedly"
    timing: if blocker
    reason: "separate credential/IP/balance/provider failures"
  - skill: browser-qa-evidence
    use_when: "settings UI readiness changes are browser-visible"
    timing: during verification
    reason: "prove UI without secret leakage"
  - skill: publish-ci-deploy
    use_when: "accepted code/docs changes need delivery"
    timing: before ship
    reason: "main + CI/deploy + post-main proof"
target_envs: ["local", "macstudio", "roehub.com"]
required_literals:
  - "No order submit"
  - "withdrawals disabled"
  - "IP allowlist required"
non_goals:
  - "Do not create or request raw API keys in chat."
  - "Do not place orders."
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  single_allowed_branch: null
  stage_specific_branches_forbidden: true
  worktree_allowed: false
  stash_allowed: false
  approval_required_for_branch_or_worktree: true
change_ownership:
  parallel_main_expected: true
  owned_change_scope: ["Stage 02 readiness code/docs/UI/report/ledger only"]
  foreign_changes_policy: "preserve unrelated changes"
  mixed_file_policy: "stage only owned hunks"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format:
  language: ru
  sections: ["status", "readiness_matrix", "evidence", "files", "next_stage"]
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests"
    expect: "passes if code touched"
  - cmd: "uv run pyright src/trading/contexts/exchange_control src/trading/contexts/live_execution apps tests"
    expect: "passes if code touched"
  - cmd: "uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/identity tests/unit/apps"
    expect: "passes if relevant"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["mainnet read-only provider calls", "API", "DB", "browser if UI changed", "Prometheus if metrics changed"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/02-mainnet-exchange-connections-readiness.md
proof_boundary:
  required_when: "changed code is verified on Mac Studio"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Requires target revision on origin/main, green CI, deploy/sync, then runtime proof."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["API keys", "secrets", "tokens", "cookies", "raw provider payloads"]
remote_command_quoting:
  applies_when: "SSH commands contain SQL or JSON"
  required_pattern: "quoted heredoc/stdin"
  forbidden_pattern: "nested inline SQL/JSON"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true
stage_execution_ledger:
  path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  current_stage: "02"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
prompt_pack_execution:
  mode: goal_driven
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  prompt_pack_dir: .codex/agents/generated/mainnet-real-money-trading-v1/
  stage_ledger: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  goal_mode_optional: true
  goal_artifact_required: false
file_manifest:
  required_for_stage_prompts: true
  expected_groups:
    code: ["src/trading/contexts/exchange_control/**", "src/trading/contexts/live_execution/**", "apps/api/**", "apps/web/**"]
    docs_runbooks: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/02-mainnet-exchange-connections-readiness.md", "docs/runbooks/exchange-secret-management.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches:
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/02-mainnet-exchange-connections-readiness.md
  - docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
possible_secondary_touches:
  - src/trading/contexts/exchange_control
  - src/trading/contexts/live_execution
  - apps/api
  - apps/web
  - docs/runbooks/exchange-secret-management.md
  - docs/architecture/README.md
safety_notes:
  - "Read-only mainnet calls may verify permissions/balances; order submit is forbidden."
---

# Task

Validate Binance/Bybit mainnet exchange connections in read-only mode for `spot` and `futures`.

Done means:

- Stage `01 accepted`;
- user has connected/funded required mainnet credentials via UI/settings, without sharing secrets in chat;
- Binance/Bybit spot/futures readiness is proven by real read-only calls;
- trade permission, no-withdrawal expectation, IP restriction, and usable balance buckets are recorded safely;
- no order submit occurred.

## Context / Current State

The existing exchange connection flow supports `mainnet`, but real-money readiness must be proven without exposing raw credentials or sending orders.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `01` is `accepted`. If not, update Stage `02` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: do not infer exchange readiness from UI presence alone; use actual read-only provider/API/DB/browser evidence.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Before any implementation or runtime action, read the stage ledger and confirm Stage `01 accepted`; if not, write Stage `02 blocked`, update the ledger, and stop.
- Record `User required before start: Binance/Bybit mainnet keys connected, IP allowlist enabled, balances funded`.
- Fail closed if credentials, IP allowlist, trade permission, or balances are absent.
- Prove both exchanges and both market types read-only.
- Keep withdrawal permission disabled as a required safety expectation.
- Use browser proof only if UI is changed or readiness is user-visible.

## Requirements (Should)

- Reuse existing exchange-control custody and validation paths.
- Do not duplicate secret-handling logic in API/UI.

## Requirements (Nice-to-have)

- Record stable readiness reason codes for each market surface.

# Context acquisition protocol

Read plan/ledger first, then exchange connection docs/runbook. Do not inspect adapter code unless readiness gaps require code changes.

Reading budget: target `<= 10 files` and `<= 55k tokens`. Expand only for failing readiness, missing DTO, UI drift, or provider-specific blocker.

Stop when required connections, read-only validation path, touched files, and acceptance commands are clear.

# Reading manifest

Use `ui_settings` if readiness UI changes. Use `exchange_adapters` only for implementation gaps.

# Work plan (agent should follow)

1. Verify previous stage accepted.
2. Gather read-only readiness inventory for Binance/Bybit spot/futures.
3. Implement only missing readiness/custody/read-model pieces.
4. Run local gates if code changed.
5. Deliver to main if implementation changed; collect post-main production runtime proof.
6. Run real read-only provider/API/DB/browser checks.
7. Update report and ledger.

# Acceptance criteria (Definition of Done)

- Four required surfaces have explicit status: Binance spot, Binance futures, Bybit spot, Bybit futures.
- No submit/cancel/market-order endpoint was called.
- Stage report includes sanitized service calls and blockers.
- Ledger opens Stage `03` only if all required readiness is accepted.

# Implementation constraints

- No raw secrets in logs/reports.
- No fallback fake validation.
- No broad staging.

# Files to indicate (expected touched areas)

List exact code/docs/UI files touched and justify any outside paths.

Final file manifest must include `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

# Non-goals

- No risk policy implementation.
- No futures config mutation.
- No canary orders.

# Quality gates (must run and pass)

Run focused `ruff`, `pyright`, relevant `pytest`, docs index, and browser/runtime checks matching touched surfaces.

# Final output: report format (strict)

Report in Russian: status, user prerequisites, readiness matrix, provider/API/DB/browser/metrics evidence, file manifest, next stage gate.
