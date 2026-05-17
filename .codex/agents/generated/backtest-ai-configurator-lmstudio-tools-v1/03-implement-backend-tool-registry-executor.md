---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_03_backend_tool_registry_executor
repo: roehub.com
branch: main
scope: "Implement the backend-owned /backtests AI Configurator tool registry and deterministic executor without contacting LM Studio."

language:
  implementation: python_backend
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
      why: "target tool contract from Prompt 02"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/services/catalog.py
      why: "source inventory for allowed values"
      inspect_symbols:
        - BacktestAiCatalogResolver
    - path: src/trading/contexts/backtest/application/ai_configurator/services/validator.py
      why: "final validation gate exposed via tools"
      inspect_symbols:
        - BacktestAiConfigValidator
    - path: src/trading/contexts/backtest/application/ai_configurator/services/security.py
      why: "security gates and policy inputs"
      inspect_symbols:
        - BacktestAiInputGate
        - BacktestAiOutputGate
    - path: tests/unit/contexts/backtest/application/ai_configurator
      why: "existing AI configurator unit test style"
      inspect_symbols:
        - "*"
  conditional_bundles:
    config_defaults:
      read_when: "when implementing template/defaults tools"
      paths:
        - configs/prod/backtest_ai_configurator.yaml
        - configs/prod/indicators.yaml
    artifact_coverage:
      read_when: "when implementing artifact coverage tools"
      paths:
        - src/trading/contexts/backtest/application/services/v2/preflight.py
        - src/trading/contexts/backtest_artifacts/application/services/v2
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "if retained foundation boundary is ambiguous"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/backend_tool_executor.md
  canonical_shape: "implementation evidence Markdown + matching JSON gate markers"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  no_lmstudio_contact: true
  backend_owned_executor_required: true
  deterministic_unit_tests_required: true
  deny_unknown_tools_required: true
  audit_hashes_required: true

task_toggles:
  implement_tool_specs: true
  implement_executor: true
  implement_denials: true
  implement_audit_records: true

skill_routing:
  - skill: backend-quality-gates
    use_when: "implementing and validating Python backend changes"
    timing: "during verification"
    reason: "run focused tests, ruff, pyright"
  - skill: contract-impact-analysis
    use_when: "defining tool request/result DTOs and persistence/audit impact"
    timing: "before implementation"
    reason: "tool DTOs become adapter and audit contracts"
  - skill: production-risk-review
    use_when: "before final report"
    timing: "before ship"
    reason: "review trust boundary and denial behavior"

target_envs:
  - local-dev
  - github-actions

required_literals:
  - "backend-owned tool executor"
  - "get_config_template"
  - "list_allowed_universe"
  - "get_indicator_spec"
  - "get_artifact_coverage"
  - "validate_candidate_config"
  - "propose_nearest_valid"
  - "tool_call_id"
  - "unauthorized_tool"
  - "accepted: false"

non_goals:
  - "Do not call LM Studio."
  - "Do not implement the LM Studio adapter."
  - "Do not enable the feature."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Tool registry"
    - "Security boundary"
    - "Проверки"
    - "Следующий prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest/application/ai_configurator"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/tools
  - src/trading/contexts/backtest/application/ai_configurator/services
  - tests/unit/contexts/backtest/application/ai_configurator
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/backend_tool_executor.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/backend_tool_executor.json

possible_secondary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/__init__.py
  - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md

safety_notes:
  - "Tool executor must deny unknown tools and invalid arguments before touching data sources."
  - "Tool outputs must be bounded and must not include secrets, private paths, DSN, raw manifests or raw prompts."
---

# Task

Implement the backend-owned tool registry and deterministic executor for the
future LM Studio tools adapter. This prompt must not contact LM Studio.

Done means:

- tool specs and request/result DTOs exist;
- allowlisted backend tools execute deterministically;
- unknown/disallowed tool calls are denied and audited;
- unit tests prove allowed and denied paths;
- docs/evidence are updated.

## Context / Current State

- completed:
  - target tools architecture is documented;
  - old single-shot runtime is retired;
  - retained catalog/validator/security foundations exist.
- open_items:
  - no backend tool registry/executor exists;
  - no adapter consumes LM Studio `tool_calls` yet.
- contract_changes:
  - tool executor becomes the trust boundary between model and backend sources.
- risks:
  - leaking raw source data;
  - executing model-supplied arbitrary actions;
  - broad tool outputs increasing latency and model confusion.
- next_focus:
  - create deterministic backend tools for Prompt 04 adapter.

## Requirements (Must)

- Implement a typed tool registry with OpenAI-compatible function schemas.
- Implement executor methods for the tools defined in Prompt 02.
- Enforce:
  - allowlisted tool names only;
  - schema/argument validation before execution;
  - no arbitrary filesystem paths;
  - output size limits;
  - redaction of private paths, DSN, secrets, raw manifests and raw prompts;
  - structured denial result for unknown/unauthorized tools.
- Add audit metadata: `tool_call_id`, tool name, sanitized args hash, result
  hash, duration, status, denial reason.
- Preserve final validator authority; tool validation helper may call validator
  but must not mark jobs ready by itself.
- Write deterministic tests for every tool and for denial cases.

## Requirements (Should)

- Keep tool result shapes compact and model-friendly.
- Prefer existing catalog/validator APIs over duplicate parsing.
- Keep names stable and snake_case.

## Requirements (Nice-to-have)

- Add small golden snapshots for tool schemas if the repo pattern supports it.

# Context acquisition protocol

Read front-matter sources in order. Do not inspect UI or Mac Studio ops docs.

# Work plan (agent should follow)

1. Read design doc.
2. Define DTOs and registry.
3. Implement executor with allowlist and denials.
4. Wire exports only as needed.
5. Add unit tests.
6. Add evidence docs.
7. Run gates.

# Acceptance criteria (Definition of Done)

- All declared tools have schemas and tests.
- Unknown tool and invalid args return denial, not exceptions that escape the loop.
- No LM Studio network call is made.
- Evidence says `accepted: false` because adapter/pipeline acceptance is pending,
  but `next_prompt_allowed: true` if local gates pass.

# Quality gates (must run and pass)

Use front-matter gates.

# Final output: report format (strict)

Report in Russian using front-matter sections.
