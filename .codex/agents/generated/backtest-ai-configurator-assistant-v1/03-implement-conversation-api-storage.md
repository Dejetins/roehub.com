---
prompt_name: backtest_ai_configurator_assistant_v1_03_conversation_api_storage
repo: roehub.com
branch: main
scope: "Implement one-chat conversation API and storage for /backtests AI assistant, without old mode/job endpoints."

language:
  implementation: python_fastapi_sqlalchemy
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "Iteration 03 source"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "Iteration 02B human-readable gate"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "02B acceptance gate"
  task_entrypoints:
    - path: apps/api/routes/backtest_ai_config.py
      why: "API boundary for assistant"
    - path: apps/api/routes/
      why: "route patterns/auth dependencies"
    - path: src/trading/contexts/backtest/application/ai_configurator/
      why: "assistant application layer"
    - path: alembic/versions/
      why: "migration pattern if persistence is added"
  conditional_bundles:
    persistence:
      read_when: "adding conversation/message/run tables"
      paths:
        - src/trading/contexts/backtest/adapters/outbound/persistence/
        - apps/api/dependencies.py
    tests:
      read_when: "adding route/storage tests"
      paths:
        - tests/unit/apps/api/
        - tests/unit/contexts/backtest/application/ai_configurator/

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md

hard_requirements:
  previous_iteration_accepted_required: true
  single_chat_backend_intent_classification: true
  old_job_endpoints_removed: true
  user_isolation_required: true
  retention_30_days: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_api: true
  implement_storage: true
  implement_llm_call: false
  implement_ui: false

skill_routing:
  - skill: architecture-design
    use_when: "defining conversation/message/run tables and use-case boundaries"
    timing: "before implementation"
    reason: "storage and application boundary"
  - skill: contract-impact-analysis
    use_when: "changing API, persistence schema, retention, or old endpoint removal"
    timing: "before final report"
    reason: "public same-origin contract and DB migration"
  - skill: backend-quality-gates
    use_when: "running route/storage/migration tests"
    timing: "during verification"
    reason: "backend correctness"
  - skill: publish-ci-deploy
    use_when: "all gates pass, Mac Studio API smoke passes, marker accepted=true"
    timing: "before final report"
    reason: "publish accepted changes and verify accepted commit on Mac Studio"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "retention_days=30"
  - "max_conversations_per_user=50"
  - "max_messages_per_conversation=100"
  - "conversation_id"
  - "message_id"
  - "conversation_title"
  - "New backtest chat"
  - "load_action"

non_goals:
  - "Do not call LM Studio yet."
  - "Do not implement browser UI in this iteration."
  - "Do not keep old `/backtests/ai-config/jobs*` compatibility endpoints."

final_report_format:
  language: ru
  sections: ["Что изменено", "API contract", "Storage", "Контрактное влияние", "Проверки", "Mac Studio", "Delivery"]

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api tests/unit/contexts/backtest/application/ai_configurator"
    expect: "focused assistant route/storage tests pass"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - apps/api/routes/backtest_ai_config.py
  - apps/api/dto/
  - src/trading/contexts/backtest/application/ai_configurator/
  - src/trading/contexts/backtest/adapters/outbound/persistence/
  - alembic/versions/
  - tests/unit/apps/api/
  - tests/unit/contexts/backtest/application/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/conversation_api_contract.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_03_conversation_api.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_03_conversation_api.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

possible_secondary_touches:
  - configs/prod/backtest_ai_configurator.yaml
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md

safety_notes:
  - "History is stored in Roehub DB, not LM Studio state."
  - "The assistant must never invoke core backtest job creation."
---

# Task

Implement Iteration 03: one-chat conversation API and storage for `/backtests` AI assistant.

## Requirements (Must)

- Stop if Iteration 02B is not accepted.
- Also stop if the previous iteration accepted commit is not recorded as pushed to `origin/main` and verified on Mac Studio in its evidence/progress marker.
- Add conversation/message/run persistence with owner isolation.
- Backend classifies intent later; no browser-visible mode field.
- Conversation title is generated by the model as `conversation_title`; backend only validates length/safety, persists the first valid title, and may use deterministic fallback `New backtest chat` if the model title is missing or unsafe.
- Keep startup message language as platform-selected locale, but assistant replies later follow user prompt language.
- Store chat history for MVP with `retention_days=30`, `max_conversations_per_user=50`, `max_messages_per_conversation=100`.
- Old AI job endpoints must remain removed.
- API must support list/create conversation, send message, get messages/status, and load action placeholder gated by backend state.
- Create `conversation_api_contract.md`, iteration evidence, and progress updates.
- After accepted local/Mac Studio evidence, use `publish-ci-deploy` and verify accepted commit on Mac Studio.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

# Acceptance criteria (Definition of Done)

- Route/storage tests pass and prove owner isolation.
- Tests prove model-generated `conversation_title` persistence plus unsafe/missing-title fallback.
- Old endpoint `rg` current refs are zero.
- Mac Studio API smoke confirms routes/migrations work or documents a real blocker.
- Evidence JSON has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Final output: report format (strict)

Report in Russian with API paths, migration id, tests, Mac Studio result, and delivery status.
