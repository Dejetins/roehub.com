---
prompt_name: mainnet-real-money-trading-v1-17-strategy-live-mode-contract-no-order
repo: /Users/daniildegtyarev/Projects/roehub.com
branch: main
scope: "Strategy producer live-mode contract and no-order enablement"
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo workflow, proof-boundary, branch policy, scoped staging and redaction rules"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
      why: "source architecture plan for this prompt pack"
    - path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
      why: "stage source of truth and previous-stage gate"
  task_entrypoints:
    - path: docs/runbooks/exchange-execution.md
      why: "exchange-execution runtime and safety contract"
    - path: docs/runbooks/strategy-live-worker.md
      why: "strategy producer runtime contract"
    - path: docs/runbooks/prod-dashboard-metrics-reference-ru.md
      why: "metrics reference when metrics are touched"
  consult_if_needed:
    - path: docs/architecture/README.md
      read_when: "docs index or linked docs are ambiguous"
style_references:
  - .codex/agents/stage_execution_ledger_template.md
hard_requirements:
  no_unscoped_mainnet_orders: true
  stage_ledger_update_required: true
  secrets_redaction_required: true
  tests_only_acceptance_allowed: false
task_toggles:
  allow_code_changes: true
  allow_runtime_checks: true
  allow_mainnet_orders: false
skill_routing:
  - skill: contract-impact-analysis
    use_when: "stage work crosses the contract-impact-analysis boundary"
    timing: during investigation or verification
    reason: "required by this stage surface"
  - skill: browser-qa-evidence
    use_when: "stage work crosses the browser-qa-evidence boundary"
    timing: during investigation or verification
    reason: "required by this stage surface"
target_envs: ["local", "macstudio", "roehub.com"]
required_literals:
  - "User required before start: Stage 16 accepted"
  - "previous stage"
  - "file manifest"
non_goals:
  - "Do not broaden mainnet access outside this stage scope."
  - "Do not print secrets, tokens, raw API keys, signed payloads, cookies or sensitive provider payloads."
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
  owned_change_scope: ["Stage 17 scoped files/hunks only", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
  foreign_changes_policy: "preserve and exclude unrelated changes from other chats"
  mixed_file_policy: "stage only owned hunks; block mixed file if safe hunk separation is impossible"
  forbidden_git_commands: ["git add .", "git add -A", "git add --all", "git commit -a", "git commit -am", "git reset ."]
  required_pre_commit_check: "git diff --cached --name-status"
  required_commit_push_marker: "ROEHUB_SCOPED_STAGING_REVIEWED=1"
final_report_format:
  language: ru
  sections: ["status", "user_required", "evidence", "files", "blockers", "next_stage"]
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if Markdown docs changed"
  - cmd: "git diff --check"
    expect: "passes"
  - cmd: "uv run ruff check apps src tests"
    expect: "passes if Python code changed; use narrower targets first when possible"
  - cmd: "uv run pyright apps src tests"
    expect: "passes if typed Python code changed; use narrower targets first when possible"
  - cmd: "uv run pytest -q tests/unit"
    expect: "focused subset passes if code changed; broaden only when risk requires"
validation_strategy:
  depth: target_runtime
  e2e_required: true
  acceptance_surfaces: ["API/DB/Redis/Monit/Prometheus/browser no-order proof", "stage report", "stage ledger"]
  tests_only_allowed_reason: null
  evidence_target: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md
proof_boundary:
  required_when: "Mac Studio or production runtime proof is collected"
  label: post_main_production_runtime_proof
  changed_code_production_claim_allowed: true
  blocked_or_deferred_reason: "Changed-code production proof requires origin/main, green CI, deploy/sync, then runtime verification."
runtime_env_sources:
  roehub_env_file_order: ["$ROEHUB_ENV_FILE", "/Users/daniildegtyarev/.config/roehub/roehub.env", "/etc/roehub/roehub.env"]
  report_only_key_presence: true
  forbidden_in_reports: ["API keys", "secrets", "tokens", "cookies", "signed payloads", "raw provider payloads", "Telegram token", "chat id"]
remote_command_quoting:
  applies_when: "SSH commands contain SQL, JSON, multiline payloads, apostrophes, backticks, or dollar signs"
  required_pattern: "quoted heredoc or stdin"
  forbidden_pattern: "nested inline SQL/JSON or shell payloads"
  temporary_files_allowed_only_when_task_requires_durable_artifact: true
browser_auth_contract:
  username: smoke_e2e_keycloak
  password_source: "macstudio env /Users/daniildegtyarev/.config/roehub/roehub.env key ROEHUB_SMOKE_E2E_PASSWORD"
  redaction: "never print raw password, session cookies, screenshots with secrets, traces with credentials, or provider payloads"
stage_execution_ledger:
  path: docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md
  plan_doc: docs/architecture/live_execution/mainnet-real-money-trading-v1.md
  current_stage: "17"
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
    code: ["apps/**", "src/trading/contexts/**", "tests/**"]
    config_infra_migrations: ["configs/prod/**", "infra/macos/**", "alembic/versions/**"]
    docs_runbooks: ["docs/runbooks/**", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md", "docs/architecture/README.md"]
    ledger_and_evidence: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md", "/opt/roehub/state/live_execution/mainnet-real-money-trading-v1/"]
  final_report_required_fields: ["created", "modified", "deleted", "outside_expected_paths", "outside_expected_paths_justification", "foreign_changes_excluded", "mixed_files"]
expected_primary_touches: ["docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md", "docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md"]
possible_secondary_touches: ["apps", "src/trading/contexts", "tests", "configs/prod", "infra/macos", "alembic/versions", "docs/runbooks", "docs/architecture/README.md"]
safety_notes:
  - "live mode must not fan out or create unexpected orders"
  - "No blind retry after unknown provider state."
---

# Task

Strategy producer live-mode contract and no-order enablement.

## Context / Current State

This prompt belongs to `Mainnet Real-Money Trading v1`. The executor must use only `plan_doc`, `prompt_pack_dir`, and `stage_ledger` as the staged execution source of truth. Do not infer readiness from chat history.

Execution anchors: `plan_doc=docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `prompt_pack_dir=.codex/agents/generated/mainnet-real-money-trading-v1/`, `stage_ledger=docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, `execution_mode=goal_driven`.

## Stage Gate And Execution Anchors

- `plan_doc`: `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/mainnet-real-money-trading-v1/`
- `stage_ledger`: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- previous-stage ledger gate / previous stage: before any implementation or runtime action, read `stage_ledger` and confirm Stage `16` is `accepted`. If not, update Stage `17` as `blocked`, write the blocker in `stage_ledger`, and stop.
- Stage-gate instruction: do not continue to the next stage unless this stage has real-boundary evidence and the ledger explicitly allows the next stage.
- File manifest: final report must list `created`, `modified`, `deleted`, `outside_expected_paths`, `outside_expected_paths_justification`, `foreign_changes_excluded`, and `mixed_files`.

## Requirements (Must)

- Record `User required before start: Stage 16 accepted` in the stage report and ledger.
- Define how live sizing is computed from strategy profile, allocation, entry sizing and risk mode.
- Bind strategy profile to owned mainnet exchange connection and canary scope.
- Pass one-shot canary token into risk context without exposing secrets.
- Prevent fan-out to multiple signals/orders; stop strategy after first accepted canary per scope.
- Handle duplicate signal, restart and idempotency without creating duplicate orders.
- Prove live mode remains no-order until Stage 18 approved window.
- Update `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md` with evidence, blockers, file manifest, contract impact and next-stage handoff.
- Update `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md` after validation and before final report.

## Requirements (Should)

- Keep changes narrowly scoped to this stage.
- Prefer existing Roehub ports/adapters/runbook patterns over new abstractions unless the stage explicitly requires a new contract.
- Use provider docs only as current contract references; do not copy raw provider payloads into reports.

## Requirements (Nice-to-have)

- Add a compact table summarizing before/after state and residual risks.

# Context acquisition protocol

Read in this order: `.codex/AGENTS.md`, `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`, `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`, then only the task entrypoints required by this stage. Do not eagerly read the whole repository.

Reading budget: target `<= 10` files before implementation. Expand only for blockers, failing gates, contract ambiguity, or runtime evidence gaps.

# Work plan (agent should follow)

1. Verify the previous stage status in `stage_ledger`.
2. Confirm whether the user-required prerequisite is satisfied; if missing, mark blocked and stop.
3. Classify affected contracts: API/DTO, persistence, config/env, browser-visible, ops/runtime, metrics, docs.
4. Implement or verify only the stage-scoped behavior.
5. Run local gates first, then collect the real-boundary evidence required by `validation_strategy`.
6. Update the stage report and ledger with accepted/blocked status, evidence, residual risk and next-stage allowance.
7. If publishing is required by the stage outcome, use scoped staging only and follow `publish-ci-deploy`; do not create branches or worktrees.

# Acceptance criteria (Definition of Done)

- Stage `17` is not accepted unless: Live-mode contract defines sizing, profile binding, canary token propagation, fan-out guard, stop-after-first-canary and restart/dedup semantics; no strategy-driven real order yet.
- Tests-only acceptance is forbidden.
- Secrets and raw provider payloads are absent from logs, reports, screenshots, traces and ledgers.
- `stage_ledger` records status, evidence, blockers, touched contracts, file manifest and next-stage handoff.
- No broad mainnet access is enabled outside this stage scope.

# Implementation constraints

- Work on `main`; do not create branches, worktrees, stashes, temporary checkouts or auxiliary folders unless the user explicitly asks.
- Preserve unrelated dirty files and foreign hunks.
- Use quoted heredoc/stdin for SSH commands with SQL/JSON/multiline payloads.
- For browser-visible work, use `smoke_e2e_keycloak` and the host-local `ROEHUB_SMOKE_E2E_PASSWORD` source; never print the password.

# Files to indicate (expected touched areas)

- Primary: `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md`, `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md`.
- Secondary only if the stage requires it: `apps/**`, `src/trading/contexts/**`, `tests/**`, `configs/prod/**`, `infra/macos/**`, `alembic/versions/**`, `docs/runbooks/**`, `docs/architecture/README.md`.

# Non-goals

- Do not execute later stage responsibilities.
- Do not bypass user-required approval or Telegram gate.
- Do not treat testnet evidence as mainnet acceptance.

# Quality gates (must run and pass)

Run the commands listed in front matter when applicable. If a command is not applicable, explain why in the report. For runtime/provider stages, include actual API/DB/Redis/Prometheus/Monit/browser/exchange evidence as applicable.

# Final output: report format (strict)

Russian report with:
- status: `accepted` or `blocked`;
- user prerequisite result;
- evidence with commands/artifacts;
- file manifest;
- contract impact;
- residual risks;
- exact next prompt if next stage is allowed.
