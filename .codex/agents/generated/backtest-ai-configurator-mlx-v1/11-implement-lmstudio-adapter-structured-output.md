---
prompt_name: backtest_ai_configurator_lmstudio_v1_11_adapter_structured_output
repo: roehub.com
branch: main
scope: "Iteration 11: replace the stale MLX HTTP adapter contract with an LM Studio structured-output adapter and prove adapter-level generation on Mac Studio."

language:
  implementation: python_backend
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_serving_gate.md
      why: "required previous Mac Studio serving acceptance"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "current architecture to update"
  task_entrypoints:
    - path: src/trading/contexts/backtest/adapters/outbound/llm/mlx_openai_compatible.py
      why: "stale adapter to replace or remove"
      inspect_symbols:
        - MLXOpenAICompatibleAdapter
        - MLXOpenAICompatibleAdapterError
    - path: src/trading/contexts/backtest/application/ai_configurator/services/validator.py
      why: "JSON schema and parse contract"
      inspect_symbols:
        - backtest_ai_model_output_schema
        - _parse_json_object
    - path: apps/worker/backtest_ai_configurator/wiring/modules.py
      why: "worker wiring chooses adapter"
      inspect_symbols:
        - build_backtest_ai_configurator_worker_app
        - MLXOpenAICompatibleAdapter
    - path: src/trading/contexts/backtest/adapters/outbound/__init__.py
      why: "stale outbound export/import zone that can keep the old adapter reachable"
      inspect_symbols:
        - MLXOpenAICompatibleAdapter
        - LMStudioOpenAICompatibleAdapter
    - path: src/trading/contexts/backtest/adapters/outbound/llm/__init__.py
      why: "stale llm package export/import zone that can keep the old adapter reachable"
      inspect_symbols:
        - MLXOpenAICompatibleAdapter
        - LMStudioOpenAICompatibleAdapter
    - path: tests/unit/contexts/backtest/application/ai_configurator/test_mlx_openai_compatible_adapter.py
      why: "stale tests to migrate"
      inspect_symbols:
        - "*"
  conditional_bundles:
    lmstudio_docs:
      read_when: "before shaping request payload"
      paths:
        - "https://lmstudio.ai/docs/developer/openai-compat/chat-completions"
        - "https://lmstudio.ai/docs/developer/openai-compat/structured-output"
        - "https://lmstudio.ai/docs/developer/rest/list"
    runtime_config:
      read_when: "when changing config schema"
      paths:
        - configs/prod/backtest_ai_configurator.yaml
        - configs/dev/backtest_ai_configurator.yaml
        - configs/test/backtest_ai_configurator.yaml
        - tests/unit/contexts/backtest/application/ai_configurator/test_backtest_ai_configurator_runtime_config.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/s1_s5_s10_mlx_benchmark_2026-05-12.md
      read_when: "only to preserve failed HTTP 400 lessons"

style_references:
  - src/trading/contexts/backtest/adapters/outbound/llm
  - tests/unit/contexts/backtest/application/ai_configurator

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_adapter_acceptance.md
  canonical_shape: "benchmark evidence markdown plus JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_10_accepted: true
  lmstudio_structured_output_required: true
  no_mlx_lm_server_runtime: true
  remove_stale_mlx_adapter_contract: true
  capture_http_400_body_safely: true
  macstudio_adapter_smoke_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  implement_lmstudio_adapter: true
  migrate_runtime_config: true
  add_response_format_json_schema: true
  run_real_adapter_smoke: true
  run_load_benchmark: false
  change_browser_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "renaming adapter, config runtime keys or output contract"
    timing: "before implementation"
    reason: "config and port compatibility"
  - skill: root-cause-debugging
    use_when: "LM Studio payload returns HTTP 400 or invalid JSON"
    timing: "if blocker"
    reason: "capture response body and isolate payload mismatch"
  - skill: backend-quality-gates
    use_when: "running adapter, config, worker and pipeline tests"
    timing: "during verification"
    reason: "backend gates"
  - skill: publish-ci-deploy
    use_when: "after local and Mac Studio adapter gates pass"
    timing: "final delivery step"
    reason: "ship code and sync Mac Studio"

target_envs:
  - local-dev
  - unit-tests
  - mac-studio-prod
  - github-actions

required_literals:
  - "LMStudioOpenAICompatibleAdapter"
  - "response_format"
  - "json_schema"
  - "POST /v1/chat/completions"
  - "messages[].content"
  - "choices[0].message.content"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "gemma-4-e2b-it-4bit"
  - "runtime: lm_studio"
  - "/api/v1/models"
  - "/v1/chat/completions"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not run S1/S5/S10/S50/S100 benchmark."
  - "Do not expose raw LM Studio errors to users."
  - "Do not embed LM Studio UI into Roehub."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Runtime/config contract"
    - "Mac Studio adapter evidence"
    - "Проверки"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/worker/test_backtest_ai_configurator_worker.py"
    expect: "passes"
  - cmd: "uv run ruff check src/trading/contexts/backtest apps/worker tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/adapters/outbound/llm/
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_ai_configurator_runtime_config.py
  - apps/worker/backtest_ai_configurator/wiring/modules.py
  - configs/prod/backtest_ai_configurator.yaml
  - tests/unit/contexts/backtest/application/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/

possible_secondary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  - apps/api/wiring/modules/backtest_ai_config.py
  - tests/unit/apps/worker/test_backtest_ai_configurator_worker.py

safety_notes:
  - "Structured output reduces custom parsing; do not add broad markdown scraping unless explicitly justified."
  - "Sanitize HTTP 400 body before logs/evidence."
  - "Old MLX adapter names may remain only in historical evidence or old prompts."
---

# Task

Implement the LM Studio adapter and structured JSON output contract for `/backtests` AI Configurator.

This prompt starts only after Iteration 10 accepted direct LM Studio serving on Mac Studio. If that evidence is missing or blocked, stop.

Done means:

- stale `MLXOpenAICompatibleAdapter` production contract is replaced by an LM Studio-specific adapter or compatibility layer with clear naming;
- request payload uses LM Studio/OpenAI-compatible `response_format: {"type":"json_schema", "json_schema": ...}`;
- system policy and user prompt are sent as proper chat messages, not one opaque user-only blob;
- HTTP 400 and malformed response bodies are captured in sanitized internal diagnostics;
- adapter-level Mac Studio smoke proves 10/10 generate calls and 10/10 repair calls return valid schema JSON;
- old adapter tests are migrated and no production code imports the stale MLX adapter name.

## Context / Current State

Context ledger:

- completed:
  - LM Studio direct serving gate should be accepted in Iteration 10.
  - Current pipeline has schema validation and one repair attempt.
- open_items:
  - current adapter sends no `response_format`;
  - current adapter hides HTTP 400 response body;
  - worker wiring still imports `MLXOpenAICompatibleAdapter`.
- contract_changes:
  - runtime becomes `lm_studio` behind the same `BacktestConfigLLMGateway` port.
- risks:
  - breaking audit storage or prompt safety while changing payload shape;
  - relying on model behavior instead of structured output;
  - leaving dead MLX runtime code in production path.
- next_focus:
  - prove adapter contract before any full pipeline or benchmark test.

## Requirements (Must)

- Stop if Iteration 10 serving evidence is missing or not accepted.
- Use LM Studio structured output for both generate and repair.
- Send `response_format` using a LM Studio-compatible version of the actual
  `backtest_ai_model_output_schema()`: every JSON Schema `type` value sent to
  LM Studio must be a string (`"string"`, `"boolean"`, `"integer"`,
  `"object"`). Do not send nullable unions such as
  `"type": ["string", "null"]`; encode absence as an empty string or an
  explicit status/boolean field at the adapter boundary.
- Use the documented LM Studio chat shape: HTTP `POST` JSON body to
  `/v1/chat/completions`, natural-language prompt text in `messages[].content`,
  `response_format.type=json_schema`, and parse
  `choices[0].message.content` as the model's JSON string.
- Keep loopback-only base URL validation.
- Add a runtime/config key such as `runtime: lm_studio`; do not keep `mlx_lm_server` as current prod runtime.
- Rename or replace stale adapter classes and tests so production code no longer says `MLXOpenAICompatibleAdapter`.
- Explicitly inspect and update/remove these stale zones so old imports cannot survive behind package exports:
  - `apps/worker/backtest_ai_configurator/wiring/modules.py`;
  - `src/trading/contexts/backtest/adapters/outbound/__init__.py`;
  - `src/trading/contexts/backtest/adapters/outbound/llm/__init__.py`;
  - `src/trading/contexts/backtest/adapters/outbound/llm/mlx_openai_compatible.py`;
  - `tests/unit/contexts/backtest/application/ai_configurator/test_mlx_openai_compatible_adapter.py`.
- Capture response status and sanitized response body for HTTP errors in internal logs/evidence, without exposing them to users.
- Keep the application port `BacktestConfigLLMGateway` stable unless a contract review requires a narrow change.
- Run real Mac Studio adapter smoke: 10 generate + 10 repair calls through the adapter or a small script using Roehub config.
- Update docs and evidence.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Run `publish-ci-deploy` after gates pass.

## Requirements (Should)

- Keep fallback JSON extraction minimal and secondary; structured output is the primary contract.
- Add deterministic MockTransport tests that assert `response_format` shape.
- Add tests proving HTTP 400 body is sanitized and available for diagnostics.
- Preserve raw model response audit table behavior.

## Requirements (Nice-to-have)

- Add a small reusable adapter smoke script under `scripts/backtest_ai/`.
- Add `seed` only if LM Studio supports it reliably for this model and tests prove compatibility.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 10 serving gate evidence
3. task entrypoints
4. LM Studio docs bundle only for payload semantics
5. runtime config bundle only if config schema changes
6. consult-if-needed failed benchmark only for HTTP 400 lessons

Do not eagerly read all historical prompt files.

Reading budget: max 8 repo files plus 2 LM Studio docs pages.

# Reading manifest

- `always_read`: repo contract, accepted serving gate, current architecture doc.
- `task_entrypoints`: stale adapter, validator schema, worker wiring, stale adapter tests.
- `conditional_bundles`: LM Studio docs and config tests only when needed.
- `consult_if_needed`: failed benchmark evidence only for known failure details.

Stop reading once adapter write set, config changes and Mac Studio smoke command are clear.

# Work plan (agent should follow)

1. Verify Iteration 10 accepted evidence.
2. Classify contract impact for adapter/config rename.
3. Implement LM Studio adapter with structured output and sanitized diagnostics.
4. Wire worker/config to the new runtime.
5. Migrate tests away from stale MLX adapter name.
6. Run local unit/type/lint gates.
7. Run Mac Studio adapter smoke against real LM Studio runtime.
8. Write adapter acceptance evidence.
9. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Production code imports `LMStudioOpenAICompatibleAdapter` or a clearly runtime-neutral name, not `MLXOpenAICompatibleAdapter`.
- Public package exports and worker wiring no longer expose or import `MLXOpenAICompatibleAdapter`; any remaining old name is historical evidence only.
- Request payload includes `response_format.type=json_schema`.
- Request payload does not include JSON Schema nullable union
  `type: ["string", "null"]`; tests assert all emitted schema `type` values are
  strings.
- Adapter parses the HTTP response JSON, then parses `choices[0].message.content`
  as JSON and validates that object.
- Mock tests assert payload shape and error diagnostics.
- Mac Studio adapter smoke has 10/10 generate and 10/10 repair valid JSON responses.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; downstream prompts may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- No S1 benchmark is run.
- Docs and evidence are updated.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Documentation

- Update old/current architecture doc and create `lmstudio_adapter_acceptance.md`.
- Keep old failed benchmark evidence as historical evidence, not current target architecture.
- Run docs index check.

## Security

- Do not log full prompts or raw responses outside existing restricted audit tables.
- Sanitize model paths, tokens, cookies, DSNs and private URLs in evidence.

## Compatibility

- Existing `/backtests/ai-config/*` API responses must stay compatible.
- Existing `/backtests/jobs` request hash semantics must stay unchanged.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/adapters/outbound/llm/`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_ai_configurator_runtime_config.py`
- `apps/worker/backtest_ai_configurator/wiring/modules.py`
- `configs/prod/backtest_ai_configurator.yaml`
- `tests/unit/contexts/backtest/application/ai_configurator/`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_adapter_acceptance.md`

Possible secondary touches:

- `scripts/backtest_ai/`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`
- `tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`

# Non-goals

- No launchd/Monit LM Studio lifecycle yet.
- No browser UI changes.
- No benchmark scenarios.
- No remote or non-MLX fallback.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/worker/test_backtest_ai_configurator_worker.py`
- `uv run ruff check src/trading/contexts/backtest apps/worker tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio adapter smoke: 10/10 generate and 10/10 repair valid structured JSON.

If a gate cannot run, classify it and stop before benchmark work.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: adapter/config/test/docs summary.
- `Runtime/config contract`: runtime key, endpoint, response_format, readiness impact.
- `Mac Studio adapter evidence`: exact smoke command, counts, failures if any.
- `Проверки`: commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke.
