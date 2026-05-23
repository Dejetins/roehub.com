---
prompt_name: identity_exchange_connections_v1_02_exchange_control_process_service_identity
repo: roehub.com
branch: main
scope: "Stage 2: introduce mandatory `exchange-control` supervised process, service identity, health, metrics, Prometheus, and Monit boundary before real exchange validation."

language:
  implementation: python_ops_prometheus_monit_docs_tests
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and ops rules"
    - path: docs/architecture/identity/identity-exchange-connections-live-trading-v1.md
      why: "Stage 2 source of truth"
    - path: docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md
      why: "shared iteration ledger and next-stage handoff facts"
    - path: docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md
      why: "accepted Stage 1 evidence"
  task_entrypoints:
    - path: infra/macos/prometheus/prometheus.prod.yml
      why: "Prometheus target pattern"
      inspect_symbols:
        - scrape_configs
        - roehub jobs
    - path: docs/runbooks/mac-studio-monitoring-plan.md
      why: "current monitoring runbook"
      inspect_symbols:
        - service list
        - Prometheus
        - Monit
    - path: apps/api
      why: "existing service/runtime conventions"
      inspect_symbols:
        - app factory
        - metrics wiring
    - path: tests/unit/apps/api
      why: "available service/API test conventions"
      inspect_symbols:
        - health tests
        - metrics tests
  conditional_bundles:
    monit_launchd_patterns:
      read_when: "adding launchd or Monit configuration"
      paths:
        - infra/macos/launchd
        - infra/scripts/monit
        - scripts/macos/reload_launchd_services.sh
    alert_rules:
      read_when: "adding exchange-control Prometheus alert rules"
      paths:
        - infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml
    exchange_control_module:
      read_when: "creating new Python module/package"
      paths:
        - src/trading/contexts
        - pyproject.toml
  consult_if_needed:
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "Mac Studio service conventions are unclear"

style_references: []

documentation_continuity:
  old_current_docs:
    - "docs/architecture/identity/identity-exchange-connections-live-trading-v1.md"
    - "docs/runbooks/mac-studio-monitoring-plan.md"
    - "infra/macos/prometheus/prometheus.prod.yml"
  new_doc_artifact: "docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md"
  canonical_shape: "stage report with Markdown evidence tables: endpoint, command, expected result, observed result, blocker"
  docs_gate: "python -m tools.docs.generate_docs_index --check"

iteration_ledger:
  path: "docs/architecture/identity/exchange-connections-stage-reports/identity-exchange-connections-live-trading-v1-iteration-ledger.md"
  update_required: true
  required_sections:
    - "Stage status"
    - "Facts for next stages"
    - "Contracts and migrations"
    - "Publish / deploy handoff"

hard_requirements:
  iteration_ledger_update_required: true
  github_yeet_after_validation_required: true
  previous_stage_must_be_accepted: true
  exchange_control_process_mandatory_before_validation: true
  service_identity_required: true
  service_identity_name_required: "exchange-control"
  metrics_port_9205_required: true
  controlled_restart_evidence_required: true
  no_real_exchange_calls: true
  no_secret_decrypt_required_in_this_stage: true

task_toggles:
  implement_runtime_boundary: true
  implement_metrics: true
  implement_ops_configs: true
  update_runbook: true
  github_yeet_after_validation: true

skill_routing:
  - skill: architecture-design
    use_when: "service boundary or dependency direction is unclear"
    timing: "before implementation only if needed"
    reason: "new supervised process boundary"
  - skill: contract-impact-analysis
    use_when: "adding runtime config, metrics, health, or ops contracts"
    timing: "before implementation and final report"
    reason: "ops/runtime contracts become stable stage gates"
  - skill: backend-quality-gates
    use_when: "running service, metrics, lint, type, docs gates"
    timing: "during verification"
    reason: "backend and ops verification"

  - skill: github:yeet
    use_when: "stage implementation, validation, stage report, and iteration ledger update are complete"
    timing: "before final report"
    reason: "user requires each validated iteration to be pushed/deployed through GitHub draft PR handoff"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "exchange-control"
  - "127.0.0.1:9205"
  - "/health/ready"
  - "/metrics"
  - "exchange_control_active"
  - "up{job=\"exchange-control\"}"
  - "roehub_exchange_control"

non_goals:
  - "Do not implement Transit/OpenBao ACL in this stage."
  - "Do not decrypt credentials."
  - "Do not call Binance or Bybit."
  - "Do not implement exchange_connections schema."
  - "Do not place, cancel, or reconcile orders."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Runtime boundary"
    - "Metrics и ops"
    - "Проверки"
    - "Stage 3 readiness"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes; create focused tests if needed"
  - cmd: "uv run ruff check apps/api src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes for touched paths"
  - cmd: "uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown changes"
  - cmd: "curl -fsS http://127.0.0.1:9205/health/ready"
    expect: "ready response when exchange-control is started; otherwise Stage 2 is blocked, not accepted"
  - cmd: "curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_control_active|exchange_connection_'"
    expect: "metrics are exposed and secret-safe"
  - cmd: "curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job=\"exchange-control\"}'"
    expect: "Prometheus can query the exchange-control target on the target runtime"
  - cmd: "/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg 'roehub_exchange_control'"
    expect: "Monit sees exchange-control on the target runtime"

  - cmd: "gh --version && gh auth status"
    expect: "GitHub CLI is installed/authenticated before github:yeet; otherwise publish handoff is blocked"

expected_primary_touches:
  - "src/trading/contexts/exchange_control/**"
  - "apps/exchange_control/** or equivalent runtime entrypoint"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "infra/macos/launchd/com.roehub.exchange-control.plist"
  - "infra/scripts/monit/roehub-exchange-control.monitrc"
  - "docs/runbooks/mac-studio-monitoring-plan.md"
  - "docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md"

possible_secondary_touches:
  - "infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml"
  - "tests/unit/contexts/exchange_control"
  - "docs/architecture/README.md"

safety_notes:
  - "Real exchange validation must remain disabled until Stage 5."
  - "Metrics labels must not include user_id, connection_id, credential ids, API keys, or raw exception text."
---

# Task

Implement the mandatory `exchange-control` process and service identity boundary.

Done means:

- `exchange-control` has a supervised runtime entrypoint;
- readiness and metrics endpoints exist on `127.0.0.1:9205`;
- Prometheus and Monit configuration are present;
- service identity is explicitly named `exchange-control` and documented for Stage 3 Transit ACL work;
- no real exchange validation is called.

## Context / Current State

Stage 2 exists to prevent external exchange validation from being added inside an opaque API path. The service must be observable before Stage 5 can call Binance/Bybit.

If Stage 1 evidence is missing or blocked, stop.

## Requirements (Must)

- Update the iteration ledger with stage status, evidence paths, changed contracts, migrations/config/env, blockers, and facts required by following stages.
- After validation and ledger update, run `github:yeet`: inspect mixed worktree, stage only intended changes, commit, push branch, and open a draft PR. Record branch, commit, PR URL, and deploy/runtime status in the ledger and final report.
- Add the smallest production-shaped `exchange-control` runtime boundary.
- Expose `GET /health/ready` and `/metrics`.
- Export `exchange_control_active`.
- Add Prometheus scrape target for `127.0.0.1:9205`.
- Add launchd and Monit configuration using repository conventions.
- Update Mac Studio monitoring runbook.
- Prove controlled restart through Monit/launchd or mark Stage 2 blocked; a static config-only change is not enough for acceptance.
- Create Stage 2 evidence report.
- Keep validation adapters disabled and absent from runtime calls.

## Requirements (Should)

- Use existing metrics/server patterns if available.
- Keep process startup config explicit and fail-fast on invalid production settings.

## Requirements (Nice-to-have)

- Include a no-op internal readiness check that can later include Transit/backend dependencies.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Stage 1 report
3. architecture document Stage 2
4. task entrypoints
5. conditional ops bundles only for touched ops files

Pre-implementation reading target:

- `<= 8 files`
- `<= ~45k tokens`

Stop reading once runtime entrypoint, metrics pattern, Prometheus target, Monit/launchd pattern, and report path are bounded.

# Reading manifest

Use front-matter `context_sources` as the canonical reading map. Do not eagerly preload every ops file.

# Work plan (agent should follow)

Skill routing for this task:

- `architecture-design`: use only if process boundary/dependency direction is unclear.
- `contract-impact-analysis`: use for runtime config, health, metrics, and ops contract changes.
- `backend-quality-gates`: use for tests/lint/type/docs verification.

1. Confirm Stage 1 accepted.
2. Add/choose the `exchange-control` entrypoint and service identity model.
3. Implement health/metrics without external exchange calls.
4. Add Prometheus/Monit/launchd/runbook updates.
5. Run quality gates and create Stage 2 report.

After the stage-specific implementation and validation steps:

- Update the iteration ledger with stage status, evidence, blockers, and next-stage facts.
- Run `github:yeet` for targeted staging, commit, push, and draft PR. Do not stage unrelated user changes.

# Acceptance criteria (Definition of Done)

- Iteration ledger is updated with facts required by the next stage.
- `github:yeet` publish/deploy handoff is completed after validation, or the stage is marked blocked with the exact reason.
- `curl -fsS http://127.0.0.1:9205/health/ready` returns ready in local smoke when service is started.
- `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_control_active|exchange_connection_'` finds expected metrics.
- Prometheus config includes `job="exchange-control"`.
- Monit config identifies `roehub_exchange_control`.
- Controlled restart through Monit/launchd succeeds and service returns ready afterward, or Stage 2 is reported blocked.
- Stage report proves no real exchange endpoints are called.

# Implementation constraints

## Determinism & ordering

- Keep service startup deterministic and explicit.
- Do not hide service identity in ad hoc environment names.

## API / contracts

- `/health/ready` and `/metrics` become operational contracts.
- Metrics label cardinality must be bounded and secret-safe.

## Documentation

- Update the iteration ledger before running `github:yeet`; this is the canonical cross-stage handoff document.
- Create the Stage 2 report.
- Update monitoring runbook and docs index if Markdown changes.
- Review old/current docs listed in `documentation_continuity.old_current_docs`; if they describe stale behavior as current, update them in the same change, otherwise state that no stale text was found.
- Use Markdown tables for service identity, health, metrics, Prometheus, Monit, restart, and no-external-call evidence.

## Tests

- Add focused tests around health/metrics/startup config where practical.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/exchange_control/**`
- `apps/exchange_control/** or equivalent runtime entrypoint`
- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/macos/launchd/com.roehub.exchange-control.plist`
- `infra/scripts/monit/roehub-exchange-control.monitrc`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md`

Possible secondary touches:

- `infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml`
- `tests/unit/contexts/exchange_control`
- `docs/architecture/README.md`

# Non-goals

- Transit ACL.
- Credential storage migration.
- Binance/Bybit calls.
- UI work.
- Order execution.

# Quality gates (must run and pass)

- `gh --version && gh auth status`
- `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api`
- `uv run ruff check apps/api src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`
- `uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`
- `python -m tools.docs.generate_docs_index --check`
- `curl -fsS http://127.0.0.1:9205/health/ready`
- `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_control_active|exchange_connection_'`
- `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="exchange-control"}'`
- `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg 'roehub_exchange_control'`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

Your final message MUST include `github:yeet` branch, commit, draft PR URL, and deploy/runtime status.

1. **Что реализовано**
2. **Runtime boundary**
3. **Metrics и ops**
4. **Проверки**
5. **Stage 3 readiness**
