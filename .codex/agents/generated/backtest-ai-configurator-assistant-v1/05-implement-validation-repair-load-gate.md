---
prompt_name: backtest_ai_configurator_assistant_v1_05_validation_repair_load_gate
repo: roehub.com
branch: main
scope: "Implement validator, one-attempt repair, preflight/artifact gates, and backend-only load_action."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "Iteration 05 source"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 04 human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "Iteration 04 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/
      why: "pipeline/validator area"
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "/backtests business validation"
    - path: apps/api/routes/backtest_ai_config.py
      why: "API response/load action mapping"
    - path: tests/unit/contexts/backtest/application/ai_configurator/
      why: "focused tests"
  conditional_bundles:
    web_contract:
      read_when: "load_action response shape affects browser"
      paths:
        - apps/web/dist/js/pages/backtests.js
        - apps/web/templates/pages/backtests.html
    indicator_axis:
      read_when: "explicit/no-window indicators fail preflight"
      paths:
        - configs/prod/indicators.yaml
        - tests/unit/contexts/backtest/application/services/v2/

hard_requirements:
  previous_iteration_accepted_required: true
  repair_attempts_one: true
  same_lmstudio_runtime_repair: true
  backend_gated_load_action: true
  no_auto_run_backtest: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_pipeline: true
  implement_ui: false

skill_routing:
  - skill: root-cause-debugging
    use_when: "preflight rejects published/default indicators"
    timing: "during implementation"
    reason: "fix root cause, not local workaround"
  - skill: contract-impact-analysis
    use_when: "changing config DTO, preflight, or load_action semantics"
    timing: "before final report"
    reason: "browser/API contract"
  - skill: backend-quality-gates
    use_when: "running validator/pipeline tests"
    timing: "during verification"
    reason: "backend correctness"
  - skill: publish-ci-deploy
    use_when: "validator/repair/preflight gates pass on local and Mac Studio, marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit on Mac Studio"

target_envs: [local-dev, mac-studio]

required_literals:
  - "repair_attempts: 1"
  - "load_action.enabled"
  - "ready"
  - "needs_clarification"
  - "unsupported_request"
  - "auto_run_backtest_attempt"

non_goals:
  - "Do not add a second model-to-model conversation."
  - "Do not let frontend infer load_action from assistant text."
  - "Do not run or enqueue backtests."

final_report_format:
  language: ru
  sections: ["Что изменено", "Validation/repair", "Load gate", "Проверки", "Mac Studio", "Delivery"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/contexts/backtest/application/services/v2"
    expect: "focused tests pass"
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/ai_configurator apps/api tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/
  - apps/api/routes/backtest_ai_config.py
  - tests/unit/contexts/backtest/application/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/validation_repair_contract.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_05_validation_repair.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_05_validation_repair.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

safety_notes:
  - "Only backend `ready` state can expose Load configuration."
  - "Malicious prompt classes must never produce loadable configs."
---

# Task

Implement Iteration 05: pipeline validation, one repair attempt via same LM Studio runtime with repair prompt, and backend-only `load_action`.

## Requirements (Must)

- Stop if Iteration 04 is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Parse model JSON safely; reject unsafe text/output injection.
- Validate schema, business rules, context snapshot availability, indicator axis rules, and preflight.
- Repair loop is exactly one attempt; backend calls same adapter with repair prompt and validator errors.
- `load_action.enabled=true` only when final config is validated and status is `ready`.
- Unsupported symbols/indicators return human-readable clarification, not fabricated config.
- Auto-run requests must be blocked; chat never calls core backtest jobs.
- Run the visible/default indicator preflight audit: every visible indicator with default UI/model values must be preflight-valid (`40/40`) or the lower count is accepted only when excluded/hidden indicators and reasons are documented.
- Create validation/repair docs/evidence and update progress.
- After accepted evidence, use `publish-ci-deploy`; sync/verify accepted commit on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Tests cover valid, unsupported, schema failure, repair success, repair failure, prompt injection, and auto-run attempt.
- Published/default indicator configs do not fail preflight due to UI-style synthetic ranges; indicator default audit is `40/40 valid` or documented exclusions match the model-facing/UI-hidden set.
- Mac Studio smoke completes at least one supported prompt to `ready`.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian with validation paths, repair evidence, tests, Mac Studio smoke, and delivery status.
