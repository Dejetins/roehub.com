---
prompt_name: backtest_ai_configurator_assistant_v1_07_ops_monit_metrics
repo: roehub.com
branch: main
scope: "Implement assistant worker service lifecycle, Monit/autostart, readiness, and Prometheus metrics."

language:
  implementation: python_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "Ops/Monit/metrics requirements"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 06 human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "Iteration 06 gate"
  task_entrypoints:
    - path: apps/worker/backtest_ai_configurator/
      why: "worker process"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "runtime/lifecycle config"
    - path: infra/
      why: "service/monit/prometheus patterns"
    - path: docs/runbooks/
      why: "operations docs"
  conditional_bundles:
    existing_monitoring:
      read_when: "adding Prometheus/Grafana/Monit target docs"
      paths:
        - docs/architecture/README.md
        - docs/runbooks/
        - infra/monit/
        - infra/prometheus/
    lmstudio_lifecycle:
      read_when: "checking model-server readiness behavior"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/macstudio_blocker.md

hard_requirements:
  previous_iteration_accepted_required: true
  monit_management_required: true
  autostart_after_reboot_required: true
  readiness_requires_loaded_model_generation: true
  prometheus_metrics_required: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_ops: true
  implement_benchmark: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding health/ready/metrics endpoints or config keys"
    timing: "before final report"
    reason: "operational API/config contract"
  - skill: backend-quality-gates
    use_when: "running worker/health tests"
    timing: "during verification"
    reason: "service correctness"
  - skill: publish-ci-deploy
    use_when: "local gates and Mac Studio two-cycle service checks pass, marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit/service on Mac Studio"

target_envs: [local-dev, mac-studio]

required_literals:
  - "/health/live"
  - "/health/ready"
  - "/metrics"
  - "monit"
  - "loaded model + lightweight generation"
  - "lm_studio"

non_goals:
  - "Do not treat `/v1/models` as readiness."
  - "Do not expose LM Studio publicly."
  - "Do not run S1/S5/S10 benchmark here."

final_report_format:
  language: ru
  sections: ["Что изменено", "Ops contract", "Metrics", "Monit/Mac Studio", "Проверки", "Delivery"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/worker tests/unit/contexts/backtest/application/ai_configurator"
    expect: "focused worker/health tests pass"
  - cmd: "uv run ruff check apps/worker src/trading/contexts/backtest infra tests/unit/apps/worker"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - apps/worker/backtest_ai_configurator/
  - configs/prod/backtest_ai_configurator.yaml
  - infra/monit/
  - infra/prometheus/
  - docs/runbooks/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/ops_runbook.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_07_ops.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_07_ops.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

safety_notes:
  - "High-load user message should be friendly and include estimated wait, not raw code."
  - "No public model-server exposure."
---

# Task

Implement Iteration 07: production operation for the assistant worker with Monit/autostart, readiness, and metrics.

## Requirements (Must)

- Stop if Iteration 06 is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Worker exposes `/health/live`, `/health/ready`, `/metrics`.
- Readiness requires loaded LM Studio model plus lightweight generation smoke; `/v1/models` alone is insufficient.
- Monit manages start/stop/restart and autostart after reboot.
- Mac Studio acceptance requires two stop/start/restart cycles, no restart loop, readiness pass, and metrics scrape.
- Add metrics for requests, statuses, queue wait, LLM latency, validation/repair, load_action, security blocks, and high-load responses.
- Create ops runbook/evidence and update progress.
- After accepted evidence, use `publish-ci-deploy`; sync/verify accepted commit/service on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Local tests pass.
- Mac Studio service lifecycle evidence passes two cycles.
- `/metrics` scrape has expected assistant metrics.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian with Monit commands/results summary, health/ready/metrics output summary, tests, and delivery status.
