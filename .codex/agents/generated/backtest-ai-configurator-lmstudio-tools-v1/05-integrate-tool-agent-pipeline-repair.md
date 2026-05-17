---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_05_pipeline_repair_integration
repo: roehub.com
branch: main
scope: "Integrate the LM Studio tool-agent adapter into the /backtests AI Configurator pipeline with validation, bounded repair, nearest-valid fallback, and one real API job readiness."

language:
  implementation: python_fastapi_worker
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
      why: "target pipeline contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/lmstudio_tool_loop_acceptance.md
      why: "adapter-level acceptance from Prompt 04"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py
      why: "pipeline orchestration"
      inspect_symbols:
        - BacktestAiConfigPipeline
    - path: src/trading/contexts/backtest/application/ai_configurator/worker.py
      why: "job state transitions"
      inspect_symbols:
        - BacktestAiConfigWorkerUseCase
    - path: src/trading/contexts/backtest/application/ai_configurator/services/validator.py
      why: "final ready/loadable gate"
      inspect_symbols:
        - BacktestAiConfigValidator
    - path: apps/worker/backtest_ai_configurator/wiring/modules.py
      why: "production worker wiring"
      inspect_symbols:
        - build_worker_app
  conditional_bundles:
    api_auth:
      read_when: "if real API job auth/session is blocked"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
    macstudio_ops:
      read_when: "before real Mac Studio job smoke"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
  consult_if_needed:
    - path: apps/web/dist/js/pages/backtests.js
      read_when: "if status event names affect UI before Prompt 06"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_pipeline_readiness.md
  canonical_shape: "pipeline readiness Markdown + matching JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  real_api_job_ready_required: true
  bounded_repair_required: true
  backend_validation_authoritative: true
  no_backtest_execution_by_model: true
  feature_still_internal_or_disabled: true

task_toggles:
  integrate_adapter: true
  implement_repair_or_nearest_valid: true
  run_real_job_smoke: true
  record_stage_metrics: true

skill_routing:
  - skill: backend-quality-gates
    use_when: "pipeline and worker tests"
    timing: "during verification"
    reason: "focused pytest/ruff/pyright"
  - skill: backend-performance-evidence
    use_when: "recording real job latency and queue wait"
    timing: "during Mac Studio smoke"
    reason: "pipeline readiness needs measured latency"
  - skill: production-risk-review
    use_when: "before declaring readiness"
    timing: "before ship"
    reason: "ready/loadable state is a rollout boundary"

target_envs:
  - local-dev
  - mac-studio-prod
  - github-actions

required_literals:
  - "intent_classification"
  - "context_collection"
  - "candidate_generation"
  - "backend_validation"
  - "repair_or_nearest_valid"
  - "ready"
  - "loadable"
  - "unauthorized actions: 0"
  - "accepted: false"
  - "next_prompt_allowed"

non_goals:
  - "Do not update the browser UI beyond required compatibility shims."
  - "Do not run S50/S100 load benchmark."
  - "Do not enable paid-user rollout."

final_report_format:
  language: ru
  sections:
    - "Pipeline state"
    - "Real job evidence"
    - "Repair and validation"
    - "Security"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/worker apps/api src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py
  - src/trading/contexts/backtest/application/ai_configurator/worker.py
  - apps/worker/backtest_ai_configurator/wiring/modules.py
  - tests/unit/contexts/backtest/application/ai_configurator
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_pipeline_readiness.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_pipeline_readiness.json

possible_secondary_touches:
  - apps/api/routes/backtest_ai_config.py
  - tests/unit/apps/api/test_backtest_ai_config_routes.py
  - configs/prod/backtest_ai_configurator.yaml

safety_notes:
  - "The model may propose config candidates; backend validation decides ready/loadable."
  - "No model path may enqueue or execute a backtest."
---

# Task

Integrate the tool-agent adapter into the AI Configurator job pipeline and prove
one supported real API job reaches `ready` on Mac Studio.

Done means:

- pipeline stages match the design;
- bounded repair or nearest-valid fallback works;
- final backend validation gates ready/loadable state;
- one real job reaches `ready`;
- evidence records exact blocker if it fails.

## Requirements (Must)

- Use the tool-agent adapter from Prompt 04.
- Implement bounded repair:
  - default `repair_attempts: 1`;
  - repair prompt may use tools only through backend executor;
  - if repair fails, return nearest valid option or clarification, not ready.
- Preserve no-backtest-execution invariant.
- Record per-stage metrics:
  total latency, queue wait, model latency, tool rounds, tool call count, repair
  count, validation status, final valid config rate for smoke cases.
- Run model/pipeline checks:

| Check | Expected |
| --- | --- |
| P0 non-config prompt | no config, no ready state, safe assistant message |
| P1 supported BTCUSDT RSI 2023 | `ready`, loadable config, final valid config |
| P2 unsupported symbol | clarification or nearest valid, not ready unless valid alternative explicit |
| P3 invalid indicator window | repair or nearest valid within indicator bounds |
| P4 period beyond artifact coverage | rejected or nearest valid covered period |
| P5 tool injection | unauthorized actions 0, no private leakage |

- Stop if unauthorized actions > 0 or if backend validation can be bypassed.
- Evidence must include `accepted`, `blocking_reason`, `next_prompt_allowed`.

## Requirements (Should)

- Keep assistant messages concise and localized to user request language.
- Store raw model attempts only in private/audited persistence, not browser payload.

# Work plan (agent should follow)

1. Wire adapter into pipeline/worker.
2. Add tests for stages and repair/fallback.
3. Run local gates.
4. Deploy or run against Mac Studio only after local gates pass.
5. Run P0-P5 staged checks.
6. Write evidence and stop on blocker.

# Acceptance criteria (Definition of Done)

- Local tests pass.
- One real supported job reaches `ready`.
- Unauthorized actions are 0.
- Feature remains controlled until UI and load benchmark prompts pass.

# Final output: report format (strict)

Report in Russian using front-matter sections.
