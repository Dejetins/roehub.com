---
prompt_name: backtest_ai_configurator_assistant_v1_04_prompt_lmstudio_json_contract
repo: roehub.com
branch: main
scope: "Implement machine-readable prompt contract, LM Studio chat-completions adapter, and strict JSON schema."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "prompt policy and LM Studio target"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 03 human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "Iteration 03 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/
      why: "prompt/context/pipeline area"
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "LM Studio runtime config"
    - path: apps/worker/backtest_ai_configurator/
      why: "worker area if already present"
    - path: tests/unit/contexts/backtest/application/ai_configurator/
      why: "prompt/adapter tests"
  conditional_bundles:
    lmstudio_docs_or_smoke:
      read_when: "payload compatibility or response_format behavior is unclear"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/macstudio_blocker.md
    old_adapters:
      read_when: "old tool/MLX adapters still exist"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/llm/
        - apps/worker/backtest_ai_configurator/

hard_requirements:
  previous_iteration_accepted_required: true
  lm_studio_chat_completions_runtime: true
  json_schema_output_contract: true
  system_prompt_machine_readable_english: true
  no_model_tools: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_lmstudio_adapter: true
  implement_validation_repair: false
  implement_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing prompt schema, adapter payload, or config schema"
    timing: "before final report"
    reason: "prompt/config contracts"
  - skill: backend-quality-gates
    use_when: "running prompt/adapter tests"
    timing: "during verification"
    reason: "backend correctness"
  - skill: publish-ci-deploy
    use_when: "direct LM Studio smoke 10/10 and adapter generate/repair 10/10 pass, marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit on Mac Studio"

target_envs: [local-dev, mac-studio]

required_literals:
  - "SYSTEM_PROMPT_ID: backtest_ai_configurator_assistant_v1"
  - "TRUSTED_CONTEXT_JSON"
  - "CURRENT_FORM_CONFIG_JSON"
  - "RECENT_CHAT_CONTEXT_JSON"
  - "OUTPUT_JSON_SCHEMA"
  - "POST /v1/chat/completions"
  - "response_format"

non_goals:
  - "Do not use LM Studio tool/function calling."
  - "Do not expose chain-of-thought."
  - "Do not implement UI."
  - "Do not run backtests from chat."

final_report_format:
  language: ru
  sections: ["Что изменено", "Prompt contract", "LM Studio", "Проверки", "Mac Studio", "Delivery"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator"
    expect: "prompt/schema/adapter tests pass"
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/ai_configurator apps/worker/backtest_ai_configurator tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/prompts/
  - src/trading/contexts/backtest/application/ai_configurator/schema.py
  - src/trading/contexts/backtest/adapters/outbound/llm/
  - configs/prod/backtest_ai_configurator.yaml
  - tests/unit/contexts/backtest/application/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/prompt_contract.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_04_prompt_lmstudio.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_04_prompt_lmstudio.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

safety_notes:
  - "User input is untrusted and never concatenated into system rules."
  - "Model returns structured JSON envelope; user-facing text is only assistant_message."
---

# Task

Implement Iteration 04: canonical English system prompt, JSON schema, LM Studio chat-completions adapter, and smoke scripts/tests.

## Requirements (Must)

- Stop if Iteration 03 is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- System prompt must be machine-readable English and include hard scope: no backtest execution, no secrets/files/tools, one-symbol config only.
- Backend prompt package includes trusted context, current form config, recent chat context, user message, output schema, and example.
- Output schema must include `conversation_title`; model generates it, backend validates/persists it in Iteration 03 storage semantics.
- Use LM Studio local API via `POST /v1/chat/completions`; `/v1/models` alone is not readiness.
- Use structured JSON output if supported by current LM Studio runtime; otherwise document safe fallback parser rules.
- Direct LM Studio structured smoke must pass `10/10`; adapter generate `10/10`; adapter repair prompt smoke `10/10`.
- Create prompt contract docs/evidence and update progress.
- After acceptance, use `publish-ci-deploy`; sync/verify accepted commit on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Unit tests cover system prompt invariants and schema shape.
- Mac Studio LM Studio smoke proves actual model can produce schema-compatible responses.
- No nullable-union schema if LM Studio rejects it.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian with prompt/schema paths, LM Studio payload notes, smoke counts, Mac Studio result, and delivery status.
