---
prompt_name: web_ui_backend_v1_12_capacity_load_validation
repo: roehub.com
branch: current
scope: "Этап 12: capacity/load validation текущего backend host для Web UI/API."

language:
  implementation: python_docs_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "performance evidence, gates, delivery rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Этап 12 source of truth"
    - path: docs/runbooks/web-ui-gateway-same-origin.md
      why: "local/prod same-origin topology"
  task_entrypoints:
    - path: tools/load
      why: "target location for lightweight capacity harness if missing"
    - path: apps/web/main/app.py
      why: "web host and API proxy route surface"
    - path: apps/api/main/app.py
      why: "API metrics and route surface"
    - path: apps/api/monitoring.py
      why: "existing Prometheus HTTP metrics"
  conditional_bundles:
    deploy_host:
      read_when: "when running or documenting current host/VPS/Mac Studio capacity"
      paths:
        - infra/caddy/Caddyfile.vps
        - .github/workflows/deploy-web.yml
        - docs/runbooks/mac-studio-native-backend-operations.md
    scenario_endpoints:
      read_when: "when implementing scenarios"
      paths:
        - apps/api/routes/backtests.py
        - apps/api/routes/strategies.py
        - apps/api/routes/identity.py
  consult_if_needed:
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      read_when: "if browser-visible first-paint constraints are ambiguous"

hard_requirements:
  capacity_report_required: true
  no_external_load_dependencies_without_justification: true
  no_secrets_in_report: true
  green_yellow_red_classification_required: true
  host_class_commit_config_required: true
  p50_p95_p99_error_rss_cpu_required: true
  publish_after_success: true

task_toggles:
  implement_lightweight_harness_if_missing: true
  implement_runbook_optional: true
  run_read_mostly_scenarios: true
  run_controlled_create_burst_after_8_5: true
  publish_after_success: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing/running/reporting load/capacity tests and classifying green/yellow/red"
    timing: "before and during measurement"
    reason: "capacity claims require comparable measurements and evidence"
  - skill: backend-quality-gates
    use_when: "running harness tests/lint/type checks"
    timing: "during verification"
    reason: "new tool/runbook must be gated"
  - skill: contract-impact-analysis
    use_when: "capacity profile settings, rollout gates, config defaults, or public docs become delivery contracts"
    timing: "before final report"
    reason: "capacity thresholds can become rollout/config contracts"
  - skill: publish-ci-deploy
    use_when: "harness/report/docs/tests pass, no secrets are recorded, and capacity findings are committed as scoped artifacts"
    timing: "after verification"
    reason: "full Roehub delivery chain after complete success"

target_envs:
  - local-dev
  - current-host
  - github-actions
  - macstudio

required_literals:
  - "tools/load/web_capacity_smoke.py"
  - "green"
  - "yellow"
  - "red"
  - "p50"
  - "p95"
  - "p99"
  - "RSS"
  - "CPU"
  - "no secrets"

non_goals:
  - "Do not add external load-test dependency unless separately justified."
  - "Do not run destructive flows outside isolated test account."
  - "Do not claim production capacity without recording host/config/dataset."
  - "Do not publish cookies/tokens in reports."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды, capacity evidence, результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "uv run python tools/load/web_capacity_smoke.py --help"
    expect: "passes if harness is added"
  - cmd: "uv run ruff check tools/load tests"
    expect: "passes for touched paths"
  - cmd: "uv run pyright"
    expect: "passes if typed Python touched"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs/runbooks changed"

expected_primary_touches:
  - "tools/load/web_capacity_smoke.py"
  - "docs/runbooks/web-ui-capacity-smoke.md"
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"

possible_secondary_touches:
  - "tests/unit/tools/test_web_capacity_smoke.py"
  - "docs/architecture/README.md"

safety_notes:
  - "No secrets, cookies, tokens, env dumps in capacity report."
  - "Read-mostly by default; destructive create bursts only controlled and isolated."
  - "If scenario is yellow/red, record mitigation before public rollout."
---

# Task

Implement Stage 12 capacity/load validation for the current backend host.

Done means:

- lightweight capacity harness exists or an existing one is used;
- scenarios cover shell/assets, dashboard, settings reads, monitoring snapshot/SSE, backtests history/results/trades, and controlled preflight/create burst after Stage 8.5;
- report records host class, commit, config, dataset/cache state, concurrency/duration;
- report records p50/p95/p99, error rate, payload size, RSS, CPU, DB/Redis latency signs, active SSE connections where applicable;
- each area is classified `green`, `yellow`, or `red`;
- yellow/red scenarios include mitigation before public rollout.

## Context / Current State

- Plan targets current backend host capacity, including 1 vCPU / 2 GB class constraints.
- Existing API has Prometheus HTTP metrics.
- This is measurement/evidence work, not feature implementation.

## Requirements (Must)

- Use `backend-performance-evidence`.
- Do not add Node/load-test runtime.
- Prefer Python + already available `httpx` if adding harness.
- Record exact commands.
- Never record secrets/cookies/tokens.
- Use `publish-ci-deploy` only after harness/report/docs/tests pass.

## Requirements (Should)

- Keep scenarios read-mostly by default.
- Make auth bootstrap configurable without hardcoding credentials.
- Output machine-readable JSON plus readable summary if practical.

## Requirements (Nice-to-have)

- Add a runbook for local/prod capacity smoke.

# Context acquisition protocol

Read `.codex/AGENTS.md`, plan Stage 12, same-origin runbook, then entrypoints. Expand into endpoint routes only for scenario implementation.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use front matter `context_sources`.

# Work plan (agent should follow)

1. Design minimal harness and report schema.
2. Implement or reuse harness.
3. Add safe scenario set and auth/session handling without secret logging.
4. Run local smoke.
5. Run current-host capacity assessment if environment is available.
6. Classify results and record mitigations.
7. Run gates.
8. Use `publish-ci-deploy` only after complete success.

# Acceptance criteria (Definition of Done)

- Capacity report contains commands, host class, commit and config.
- No first-paint endpoint transfers unbounded data.
- Polling/SSE loops do not overlap requests under latency.
- Backtest create path does not execute full compute in API request path.
- p95/RSS/error trends are recorded.
- Known limits are in rollout notes and handoff.

# Implementation constraints

## API / contracts

- Public API contract: `none` if measurement only.
- Config schema: `compatible-change` if capacity profile settings are added.
- Performance risk: measured evidence.
- Rollout gates: `compatible-change` if capacity report becomes pre-ship gate.

# Files to indicate (expected touched areas)

Use front matter touched paths.

# Non-goals

- Feature implementation.
- Unbounded stress tests against production users.
- Secret-bearing reports.

# Quality gates (must run and pass)

```bash
uv run python tools/load/web_capacity_smoke.py --help
uv run ruff check tools/load tests
uv run pyright
python -m tools.docs.generate_docs_index --check
```

Example capacity smoke:

```bash
uv run python tools/load/web_capacity_smoke.py \
  --base-url http://127.0.0.1:8010 \
  --api-base-url http://127.0.0.1:8000 \
  --profile local-smoke \
  --duration-s 60 \
  --concurrency 10 \
  --scenario dashboard,monitoring,backtests_history
```

# Final output: report format (strict)

Report in Russian: `Intent`, `Scope`, `Measurement setup`, `Results`, `Classification`, `Contract impact`, `Tests`, `Risks`, `Handoff`, `Publish/deploy`.
