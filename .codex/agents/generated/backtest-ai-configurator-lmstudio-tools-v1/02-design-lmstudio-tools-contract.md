---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_02_design_tools_contract
repo: roehub.com
branch: main
scope: "Design the LM Studio OpenAI-compatible tools contract, tool registry boundaries, pipeline stages, expected model behavior, and acceptance matrix for the /backtests AI Configurator."

language:
  implementation: architecture_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo architecture and safety contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current reset state and retained foundation"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/cleanup_readiness.md
      why: "cleanup result from Prompt 01"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/ports/agent_gateway.py
      why: "pending gateway boundary to refine"
      inspect_symbols:
        - BacktestConfigAgentGateway
        - BacktestConfigAgentRequest
        - BacktestConfigAgentResponse
    - path: src/trading/contexts/backtest/application/ai_configurator/services/catalog.py
      why: "backend-owned source inventory"
      inspect_symbols:
        - BacktestAiCatalogResolver
        - BacktestAiAllowedCatalog
    - path: src/trading/contexts/backtest/application/ai_configurator/services/validator.py
      why: "authoritative final config gate"
      inspect_symbols:
        - BacktestAiConfigValidator
    - path: src/trading/contexts/backtest/application/ai_configurator/services/security.py
      why: "input/output security gate boundary"
      inspect_symbols:
        - BacktestAiInputGate
        - BacktestAiOutputGate
  conditional_bundles:
    lmstudio_docs:
      read_when: "before finalizing request/response shapes"
      paths:
        - https://lmstudio.ai/docs/developer/openai-compat/tools
        - https://lmstudio.ai/docs/developer/openai-compat/chat-completions
    ui_contract:
      read_when: "if the design affects browser-visible state names or load behavior"
      paths:
        - apps/web/templates/pages/backtests.html
        - apps/web/dist/js/pages/backtests.js
        - tests/unit/apps/web/test_backtests_page.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
      read_when: "if already created by another run"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  new_doc_artifact: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  canonical_shape: "architecture decision + phased rollout plan in Russian"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  design_only: true
  lmstudio_tools_required: true
  backend_owned_tools_required: true
  explicit_model_test_matrix_required: true
  no_runtime_implementation: true

task_toggles:
  define_tool_registry: true
  define_tool_loop: true
  define_repair_policy: true
  define_stage_benchmarks: true
  define_rollout_gates: true

skill_routing:
  - skill: architecture-design
    use_when: "designing the target tool-agent architecture and staged rollout"
    timing: "before writing docs"
    reason: "define boundaries, contracts, stages, and tradeoffs"
  - skill: contract-impact-analysis
    use_when: "classifying public API, port, DTO, config, and UI contract impact"
    timing: "during design"
    reason: "make compatibility and rollout effects explicit"

target_envs:
  - local-dev
  - mac-studio-prod
  - github-actions

required_literals:
  - "tools"
  - "tool_calls"
  - "choices[0].message.tool_calls"
  - "finish_reason=tool_calls"
  - "backend-owned tool executor"
  - "tool_agent_pending"
  - "intent_classification"
  - "context_collection"
  - "candidate_generation"
  - "backend_validation"
  - "repair_or_nearest_valid"
  - "Load configuration"
  - "Загрузить конфигурацию"
  - "accepted: false"
  - "next_prompt_allowed"

non_goals:
  - "Do not write production code in this prompt."
  - "Do not re-enable the feature."
  - "Do not accept a model or concurrency setting."

final_report_format:
  language: ru
  sections:
    - "Архитектурное решение"
    - "Tool contract"
    - "Stage expectations"
    - "Benchmarks and gates"
    - "Следующий prompt"

quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md

possible_secondary_touches:
  - docs/architecture/backtest/README.md
  - docs/architecture/README.md

safety_notes:
  - "Tool use means the model requests tool calls; backend code executes allowlisted tools."
  - "The model must never get arbitrary filesystem access."
---

# Task

Design the new LM Studio tools-based `/backtests` AI Configurator contract.
This is a design prompt, not implementation.

Done means:

- target stages and data flow are defined;
- backend-owned tool registry/executor contract is defined;
- model request/response expectations are defined per stage;
- benchmark/security acceptance matrix is defined before implementation;
- docs are updated and indexed.

## Context / Current State

- completed:
  - old single-shot contract retired;
  - retained foundation remains authoritative;
  - LM Studio supports OpenAI-compatible `tools` and `tool_calls`.
- open_items:
  - tool registry/executor not implemented;
  - LM Studio tool loop adapter not implemented;
  - UI contract still needs update after backend pipeline exists.
- contract_changes:
  - model receives tool definitions and may request calls;
  - backend executes only allowlisted tools and returns bounded tool results;
  - final backend validation decides `ready`, not the model.
- risks:
  - confusing tool requests with direct filesystem access;
  - letting model choose arbitrary paths/resources;
  - accepting malformed tool calls or fallback text as valid config.
- next_focus:
  - implementation-ready architecture doc for Prompts 03-07.

## Requirements (Must)

- Use LM Studio OpenAI-compatible tools as the target integration shape.
- Define stages:
  - `intent_classification`
  - `context_collection`
  - `candidate_generation`
  - `backend_validation`
  - `repair_or_nearest_valid`
  - `final_response`
- Define allowed backend tools by purpose, not raw filesystem paths:
  - get config template/defaults;
  - list supported symbols/timeframes/sources/risk/sizing/ranking;
  - get indicator spec and window bounds;
  - get artifact coverage for requested symbol/timeframe/period;
  - validate candidate config;
  - propose nearest valid alternative.
- Define forbidden actions:
  arbitrary file read, API job creation, backtest execution, delete/mutate jobs,
  secret/env access, raw prompt/security manifest exposure, HTML/script output.
- Define expected model behavior for each stage, including when no tool call is
  expected and when `choices[0].message.tool_calls` with
  `finish_reason=tool_calls` is expected.
- Define a model-test and benchmark matrix for later prompts. Include expected
  outcomes, pass/fail thresholds and stop rules for every stage that contacts
  the model.
- Keep feature disabled until Prompt 07 acceptance passes.

## Requirements (Should)

- Prefer a compact stable set of tools over many narrow one-off tools.
- Keep tool outputs hashable/auditable and redacted.
- Define trace fields for tool call id, tool name, sanitized args hash,
  result hash, duration, denial reason and model round.

## Requirements (Nice-to-have)

- Include a mermaid sequence diagram in the architecture doc if readable.

# Context acquisition protocol

Read in the order from front matter. Use web only for official LM Studio docs
if the local context is insufficient or stale.

Reading budget: max 8 repo files plus official LM Studio tools docs.

# Work plan (agent should follow)

1. Reconcile current reset state.
2. Read official LM Studio tools behavior.
3. Design tool registry and executor boundaries.
4. Design model loop and fallback behavior.
5. Define stage test matrix and benchmark thresholds.
6. Update docs and run docs gates.

# Acceptance criteria (Definition of Done)

- Architecture doc exists and is indexed.
- It explicitly says `accepted: false` until implementation and Mac Studio
  acceptance pass.
- Every model-contact stage has expected model behavior and measurable gates.
- Contract impact is classified.

# Implementation constraints

- No production code changes except docs/index.
- No feature enablement.

# Quality gates (must run and pass)

Use the front-matter `quality_gates`.

# Final output: report format (strict)

Report in Russian using the front-matter sections.
