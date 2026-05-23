---
prompt_name: identity_exchange_connections_v1_07_production_readiness_exchange_key_storage
repo: roehub.com
branch: main
scope: "Stage 7: final production-readiness gate for exchange key storage and validation, without trading execution."

language:
  implementation: python_ops_browser_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and release gates"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 7 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md
      why: "accepted Stage 6 evidence"
  task_entrypoints:
    - path: docs/architecture/identity/exchange-connections-stage-reports
      why: "stage evidence chain"
      inspect_symbols:
        - 00 baseline
        - 01 security
        - 02 process
        - 03 transit
        - 04 backfill
        - 05 validation
        - 06 UI
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "exchange-control scrape evidence"
      inspect_symbols:
        - exchange-control job
    - path: docs/runbooks/mac-studio-monitoring-plan.md
      why: "ops monitoring continuity"
      inspect_symbols:
        - exchange-control
    - path: docs/runbooks/exchange-secret-management.md
      why: "secret custody runbook"
      inspect_symbols:
        - Transit
        - emergency disable
  conditional_bundles:
    backend_gates:
      read_when: "focused backend readiness gates fail"
      paths:
        - tests/unit/apps/api/test_identity_exchange_keys_routes.py
        - tests/unit/apps/api/test_ui_account_routes.py
        - tests/unit/contexts/exchange_control
        - tests/unit/apps/migrations
    browser_gates:
      read_when: "browser-visible settings behavior must be rechecked"
      paths:
        - apps/web/templates/fragments/account/exchange_keys.html
        - apps/web/dist/js/pages/settings.js
        - tests/unit/apps/web/test_app_routes.py
    ops_gates:
      read_when: "exchange-control runtime or Monit/Prometheus evidence is missing"
      paths:
        - infra/macos/launchd/com.roehub.exchange-control.plist
        - infra/scripts/monit/roehub-exchange-control.monitrc
        - infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml
  consult_if_needed:
    - path: docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
      read_when: "legacy storage policy compatibility is unclear"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md"
    - "docs/runbooks/mac-studio-monitoring-plan.md"
    - "docs/runbooks/exchange-secret-management.md"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md"
  canonical_shape: "stage report with Markdown evidence tables: stage, required evidence, observed evidence, verdict, residual risk"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

hard_requirements:
  previous_stage_must_be_accepted: true
  all_stage_reports_required: true
  no_trading_execution_required: true
  runtime_ops_evidence_required: true
  mac_studio_runtime_evidence_required: true
  secret_leak_grep_required: true
  docs_index_required: true

task_toggles:
  implementation_changes_allowed_only_for_readiness_fix: true
  run_full_focused_gates: true
  run_browser_qa_if_ui_changed_or_evidence_missing: true
  publish_after_success: false

skill_routing:
  - skill: production-risk-review
    use_when: "assessing final readiness across secrets, schema, ops, UI, validation"
    timing: "before final report"
    reason: "final production-risk gate"
  - skill: backend-quality-gates
    use_when: "running focused pytest, ruff, pyright, docs-index"
    timing: "during verification"
    reason: "backend readiness evidence"
  - skill: browser-qa-evidence
    use_when: "UI evidence is missing, stale, or changed"
    timing: "during verification"
    reason: "settings is browser-visible and secret-sensitive"
  - skill: contract-impact-analysis
    use_when: "readiness fixes touch API, DTO, schema, config, metrics, or browser defaults"
    timing: "if blocker"
    reason: "avoid silent contract drift"

target_envs:
  - local-dev
  - mac-studio
  - browser

required_literals:
  - "exchange-control"
  - "exchange_control_active"
  - "exchange_connection_validation_total"
  - "up{job=\"exchange-control\"}"
  - "roehub_exchange_control"
  - "future execution work заблокирован"
  - "signal-to-execution"

non_goals:
  - "Do not implement signal-to-execution."
  - "Do not implement exchange-execution."
  - "Do not place live orders."
  - "Do not merge/publish/deploy unless explicitly asked by the user after this prompt."

final_report_format:
  language: ru
  sections:
    - "Вердикт"
    - "Evidence matrix"
    - "Security и secrets"
    - "Ops и runtime"
    - "Residual risks"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control tests/unit/apps/migrations"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "uv run pyright apps/api src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "! rg -n \"/order|createOrder|submit_order|place_order|exchange-execution\" src/trading/contexts/exchange_control apps/api apps/web"
    expect: "no execution/order placement surface is included in this scope"

expected_primary_touches:
  - "docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md"

possible_secondary_touches:
  - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
  - "docs/runbooks/mac-studio-monitoring-plan.md"
  - "docs/runbooks/exchange-secret-management.md"
  - "docs/architecture/README.md"

safety_notes:
  - "This is a readiness gate, not a request to ship or deploy."
  - "Do not claim production-ready if any mandatory stage evidence is missing."
---

# Task

Run the final production-readiness gate for Exchange Control v1 key storage and validation.

Done means:

- all stage reports 00-06 are present and accepted;
- API/UI/storage/validation/metrics/audit evidence is coherent;
- focused gates pass;
- runtime ops evidence exists for `exchange-control`;
- secret leakage checks are clean;
- Stage 7 report gives a clear ready/not-ready verdict;
- trading execution remains explicitly out of scope.

## Context / Current State

This prompt is not a broad implementation prompt. It is the final gate after stages 0-6. If any previous stage report is missing, stale, or says blocked, stop and report not-ready.

Do not publish, deploy, or open a PR unless the user explicitly requests delivery after this readiness gate.

## Requirements (Must)

- Verify stage evidence chain from 00 through 06.
- Run the focused backend/API/UI/migration gates.
- Verify docs index.
- Verify `exchange-control` health/metrics/Prometheus/Monit evidence on the target runtime. If Mac Studio/runtime access is unavailable, the verdict must be `not-ready`, not accepted.
- Verify security acceptance: CSRF/recent-auth, no secret fields, secret grep.
- Confirm no `exchange-execution`, order placement, order ledger, or signal-to-execution implementation is included.
- Create `docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md`.

## Requirements (Should)

- Present readiness as a compact evidence matrix, not a long narrative.
- Separate implemented facts, runtime evidence, skipped checks, and residual risks.

## Requirements (Nice-to-have)

- Include exact next prompt needed for future signal-to-execution design, but do not implement it.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 6 report and earlier stage reports only as needed to confirm the chain
3. architecture document Stage 7
4. task entrypoints
5. conditional bundles only for missing/failing evidence

Pre-implementation reading target:

- `<= 10 files`
- `<= ~50k tokens`

Stop reading once evidence chain, gate commands, ops targets, and readiness report requirements are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not turn this readiness pass into unrelated cleanup.

# Work plan (agent should follow)

Skill routing for this task:

- `production-risk-review`: use before final report to assess readiness.
- `backend-quality-gates`: use during verification.
- `browser-qa-evidence`: use if UI evidence is missing/stale/changed.
- `contract-impact-analysis`: use only if a readiness fix changes a contract.

1. Confirm all previous stage reports are present and accepted.
2. Run focused gates.
3. Collect runtime ops acceptance if environment is available.
4. Run secret grep and no-order grep.
5. Create Stage 7 readiness report with ready/not-ready verdict.

# Acceptance criteria (Definition of Done)

- Stage 7 report exists and includes an evidence matrix.
- Backend/API/UI/migration gates pass or failures are classified.
- Runtime health/metrics/Prometheus/Monit evidence is present from the target runtime; if unavailable, report `not-ready` with exact blocker.
- Security acceptance calls and secret grep are recorded.
- Report states that future execution work is blocked until separate signal-to-execution design.

# Implementation constraints

## Determinism & ordering

- Keep report order stable: stage chain, tests, runtime, security, risks.
- Do not make opportunistic refactors.

## API / contracts

- Do not change contracts unless fixing a readiness blocker; classify any such fix.

## Documentation

- Create Stage 7 report.
- Update architecture/runbooks only if readiness evidence proves drift.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for the final evidence matrix: stage, required evidence, observed evidence, command/artifact, verdict, blocker.
- Run docs-index check after Markdown changes.

## Tests

- Run focused gates exactly unless a narrower blocker must be isolated first.

# Files to indicate (expected touched areas)

Primary touches:

- `docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md`

Possible secondary touches:

- `docs/architecture/identity/identity-exchange-connections-live-trading-v1.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/runbooks/exchange-secret-management.md`
- `docs/architecture/README.md`

# Non-goals

- Publishing.
- Deployment.
- Signal-to-execution design.
- Exchange execution.
- Order placement.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control tests/unit/apps/migrations`
- `uv run ruff check apps/api apps/web src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control`
- `uv run pyright apps/api src/trading/contexts/identity src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control`
- `python -m tools.docs.generate_docs_index --check`
- `! rg -n "/order|createOrder|submit_order|place_order|exchange-execution" src/trading/contexts/exchange_control apps/api apps/web`
- `curl -fsS http://127.0.0.1:9205/health/ready`
- `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_control_active|exchange_connection_validation_total'`
- `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job=\"exchange-control\"}'`
- `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg 'roehub_exchange_control'`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Вердикт**
2. **Evidence matrix**
3. **Security и secrets**
4. **Ops и runtime**
5. **Residual risks**
