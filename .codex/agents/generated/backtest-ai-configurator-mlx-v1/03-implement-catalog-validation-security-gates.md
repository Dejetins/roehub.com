---
prompt_name: backtest_ai_configurator_mlx_v1_03_catalog_validation_security_gates
repo: roehub.com
branch: main
status: superseded
do_not_execute: true
scope: "Iteration 03: implement catalog resolver, current-form mapping, JSON schema, business validation, deterministic security input/output gates, and correction/clarification behavior without MLX."
superseded_reason: "Retired as an executable prompt on 2026-05-17 during LM Studio tools cleanup. The implemented validator/catalog/security foundation is retained, but this old prompt pack must not be rerun."
replacement_direction: "Use the forthcoming tool-based LM Studio prompt pack while preserving the existing validator, catalog resolver, artifact coverage, indicator bounds and security gates."

language:
  implementation: python_fastapi
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo safety and contract rules"
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      why: "catalog, validation and security contract"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "business validation source of truth"
      inspect_symbols:
        - BacktestRuntimeDefaultsService
        - BacktestPreflightService
    - path: configs/prod/indicators.yaml
      why: "supported indicator/param source"
      inspect_symbols:
        - defaults
    - path: apps/api/wiring/modules/ui_backtests.py
      why: "workstation catalog and config draft source"
      inspect_symbols:
        - _build_indicator_catalog
        - _build_config_draft
    - path: src/trading/contexts/backtest/application/ai_configurator
      why: "existing API/storage foundation"
      inspect_symbols:
        - "*"
  conditional_bundles:
    artifact_symbols:
      read_when: "when resolving supported symbols from artifact or market reference"
      paths:
        - apps/api/wiring/modules/ui_backtests.py
        - src/trading/contexts/backtest/adapters/outbound
    route_integration:
      read_when: "when connecting validation pipeline to API shell"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/dto/runtime_preflight.py
      read_when: "if DTO or validation issue shape is ambiguous"
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only to check for a newer executor handoff; ignore if stale/unrelated"

style_references:
  - path: src/trading/contexts/backtest/application/services/v2/preflight.py
    purpose: "strict validation and issue reporting style"
  - path: apps/api/wiring/modules/ui_backtests.py
    purpose: "current /backtests form mapping"

hard_requirements:
  depends_on_iteration_02: true
  no_mlx_runtime: true
  deterministic_security_gates: true
  business_validation_required: true
  unsupported_values_never_loadable: true
  single_symbol_mvp: true
  publish_ci_deploy_required: true
  main_branch_deployment_required: true
  macstudio_sync_required: true

task_toggles:
  implement_catalog_resolver: true
  implement_json_schema_validation: true
  implement_business_validation: true
  implement_security_gates: true
  implement_repair_loop: false
  implement_mlx: false
  implement_ui: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "mapping validated_config to current /backtests form and statuses"
    timing: "before implementation"
    reason: "DTO/browser-visible and validation contract"
  - skill: backend-quality-gates
    use_when: "running resolver, validator and API tests"
    timing: "during verification"
    reason: "backend validation quality gates"
  - skill: production-risk-review
    use_when: "before final report if security or unsupported-value behavior changed"
    timing: "before ship"
    reason: "trust boundary and user-content risk review"
  - skill: publish-ci-deploy
    use_when: "after implementation and local gates pass, deliver this iteration to main, sync Mac Studio, and run post-deploy verification"
    timing: "final delivery step"
    reason: "required end-to-end Roehub GitHub CI, main deployment, Mac Studio sync and smoke"

target_envs:
  - local-dev
  - unit-tests
  - github-actions
  - mac-studio-prod

required_literals:
  - "BacktestPreflightService"
  - "configs/prod/indicators.yaml"
  - "blocked_by_policy"
  - "input_too_large"
  - "security_review"
  - "needs_clarification"
  - "tp_sl_grid"
  - "15m"

non_goals:
  - "Do not call a model or implement MLX adapter."
  - "Do not add browser UI behavior."
  - "Do not support multi-symbol loadable configs in MVP."
  - "Do not change current backtest runtime defaults."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Validation/security behavior"
    - "Контрактное влияние"
    - "Проверки"
    - "Доставка и Mac Studio"
    - "Следующая итерация"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtests_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/ai_configurator/services/catalog.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/validator.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/security.py"
  - "src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py"
  - "tests/unit/contexts/backtest/application/ai_configurator/"

possible_secondary_touches:
  - "apps/api/routes/backtest_ai_config.py"
  - "apps/api/wiring/modules/backtest_ai_config.py"
  - "tests/unit/apps/api/test_backtest_ai_config_routes.py"

safety_notes:
  - "System prompt is not a security boundary; deterministic gates must enforce safety."
  - "Assistant-controlled text must be plain text and pass output gate."
  - "Unsupported indicator/timeframe/symbol must never produce load_action.enabled=true."
---

# Task

Implement Iteration 03 of the `/backtests` AI Configurator: catalog resolver, current form mapping, JSON schema validation, business validation through `BacktestPreflightService`, deterministic pre-LLM input gate, deterministic output gate, and correction/clarification behavior. Keep the worker/model path fake or deterministic; do not call MLX yet.

Done means:

- resolver builds a compact allowed catalog from runtime defaults, `configs/prod/indicators.yaml`, and supported symbol sources;
- validated configs match the current `/backtests` form mapping and single-symbol MVP semantics;
- unsupported values cannot reach a loadable `ready` state;
- input gate can return `blocked_by_policy`, `input_too_large`, or `security_review` with friendly messages;
- output gate rejects HTML/script/Markdown links/private infra leakage/unsupported values;
- business validation uses `BacktestPreflightService`;
- focused tests cover safe prompts, unsupported values, prompt injection patterns, output injection, single-symbol handling, and `tp_sl_grid` coverage.

## Context / Current State

Context ledger:

- completed:
  - Iteration 01 storage/quota foundation should exist;
  - Iteration 02 API shell, SSE and fake worker should exist.
- open_items:
  - real prompt profiles, repair loop and MLX adapter are not implemented yet;
  - UI is not enabled yet.
- contract_changes:
  - `validated_config` must match current form/job payload shape;
  - MVP remains single-symbol.
- risks:
  - model or fake pipeline could return unsupported configs;
  - assistant text could become an XSS vector later;
  - current visible `strategy` field is not in job payload and must not be mutated by AI.
- next_focus:
  - deterministic validator and safety foundation for prompt/model iterations.

## Requirements (Must)

- Verify Iterations 01-02 artifacts exist; if not, stop and report blocker.
- Build catalog from existing runtime defaults and indicator YAML; do not hard-code indicator lists from the doc.
- Enforce current form mapping:
  - `coordinates.exchange`
  - `coordinates.market_type`
  - `coordinates.symbol`
  - `timeframe`
  - `time_range`
  - `indicators[]`
  - `risk`
  - `execution`
  - `ranking`
  - `top_n`
- Preserve single-symbol loadable config semantics.
- Do not return `symbols[]` or mutate `strategy` field.
- Validate with JSON Schema before business validation.
- Validate with `BacktestPreflightService` before `load_action.enabled=true`.
- Implement deterministic input gate for off-topic, oversized, encoded/jailbreak and obvious secret prompts.
- Implement deterministic output gate for plain text, no HTML/script/Markdown links/private path/Tailscale/model URL/secret leakage.
- Ensure failed validation has friendly user-facing message and no load button.
- Tests must include security and unsupported-value cases.

## Requirements (Should)

- Keep security pattern checks curated and testable; do not claim complete protection.
- Make validator output structured enough for later repair prompt.
- Keep catalog subset compact and stable for prompt budgeting.
- Reuse `BacktestValidationIssue` or compatible issue shape where practical.

## Requirements (Nice-to-have)

- Add small fixtures for prompt-injection and safe prompt eval sets.
- Include correction warnings when a safe default is chosen.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report, if available
4. task entrypoints
5. only conditional bundles required by symbols/API integration ambiguity
6. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once current form mapping, runtime default sources and validation touch points are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repo rules and architecture contract.
- `task_entrypoints`: preflight, indicator catalog, workstation defaults and AI foundation.
- `conditional_bundles`: artifact symbols and route integration only when needed.
- `consult_if_needed`: DTO shape only for ambiguity.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation; owns current form mapping and status semantics.
- `backend-quality-gates`: use during verification; owns tests/lint/type checks.
- `production-risk-review`: use before final report if trust boundary behavior changed materially.

1. Verify prior iteration artifacts and current `/backtests` form/request shape.
2. Implement catalog resolver and current-form config mapping.
3. Implement JSON schema validator, business validator and final load-action gate.
4. Implement deterministic input/output security gates.
5. Integrate validator/gates into fake pipeline/API states.
6. Add focused unit/API tests for safe, unsupported, malicious and boundary prompts.
7. Run quality gates and report residual risks.

# Acceptance criteria (Definition of Done)

- Safe supported prompt can produce `status=ready` with `load_action.enabled=true`.
- Unsupported timeframe/indicator/symbol produces correction warning or `needs_clarification`; never enabled load button if invalid.
- Multi-symbol prompt produces one loadable config plus suggestions or `needs_clarification`, not `symbols[]`.
- Off-topic/jailbreak/secret/oversized prompt is blocked or security-reviewed before model call.
- Output injection/private leakage samples fail output gate.
- `BacktestPreflightService` is the final business validator.

- `publish-ci-deploy` terminal state is `deployed`, or `green-pr`/`blocked` is reported with exact blocker evidence.

# Implementation constraints

## Determinism & ordering

- Keep resolver output stable for the same runtime defaults/catalog inputs.
- Do not rely on model judgment for safety decisions.

## API / contracts

- Existing `/backtests/jobs` and `/backtests/preflight` semantics must remain unchanged.
- AI status additions are only on AI routes.

## Security

- No exact security signatures in user-facing messages.
- Do not store secrets in test fixtures.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/ai_configurator/services/catalog.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/validator.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/security.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/pipeline.py`
- `tests/unit/contexts/backtest/application/ai_configurator/`

Possible secondary touches:

- `apps/api/routes/backtest_ai_config.py`
- `apps/api/wiring/modules/backtest_ai_config.py`
- `tests/unit/apps/api/test_backtest_ai_config_routes.py`

# Non-goals

- No MLX runtime.
- No repair loop.
- No browser UI enablement.
- No launchd/Monit.
- No benchmarking.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py`
- `uv run pytest -q tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtests_routes.py`
- `uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest`
- `uv run pyright`
- `git diff --check`

If a gate cannot run, classify it as introduced, required-path pre-existing, unrelated pre-existing, environmental, or flaky.

Required delivery step: after the quality gates above pass, invoke `publish-ci-deploy` as the final step. The expected terminal state for this prompt is `deployed`: intended files committed and pushed, GitHub Actions green, revision shipped to `main`, `/opt/roehub/app` on `macstudio` pulled to that revision, the relevant production services reloaded through the repository runbook, and `bash scripts/macos/smoke_prod.sh` passed. If the skill reaches `green-pr` because a human merge/approval is required, or `blocked` because of missing auth, unrelated dirty scope, external CI, Mac Studio access, or production verification failure, report that exact state and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: resolver, schema, business validation, gates.
- `Validation/security behavior`: safe/unsupported/malicious behavior summary.
- `Контрактное влияние`: current form mapping, single-symbol, statuses.
- `Проверки`: exact commands and results.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state, main/PR SHA, CI result, Mac Studio pull/reload/smoke evidence, or exact blocker.
- `Следующая итерация`: prompt profiles, repair loop and LLM gateway.
