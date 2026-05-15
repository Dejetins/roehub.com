---
prompt_name: backtest_ai_configurator_lmstudio_v1_13_security_ux_pipeline_readiness
repo: roehub.com
branch: main
scope: "Iteration 13: fix security false-ready cases, simplify UI behavior around LM Studio structured output, and prove 10/10 real pipeline jobs reach ready on Mac Studio."

language:
  implementation: python_fastapi_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "security and browser verification rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_service_lifecycle.md
      why: "required service lifecycle evidence"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/security_eval_summary.md
      why: "known security failures"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/ai_configurator/services/security.py
      why: "input/output safety gates"
      inspect_symbols:
        - BacktestAiInputGate
        - BacktestAiOutputGate
    - path: src/trading/contexts/backtest/application/ai_configurator/services/validator.py
      why: "ready/load_action validation"
      inspect_symbols:
        - BacktestAiConfigValidator
        - validate_model_output
    - path: apps/web/dist/js/pages/backtests.js
      why: "browser AI configurator behavior"
      inspect_symbols:
        - "*"
    - path: scripts/backtest_ai/run_configurator_security_eval.py
      why: "security eval harness"
      inspect_symbols:
        - "*"
  conditional_bundles:
    api_pipeline:
      read_when: "when proving real jobs reach ready"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
        - scripts/backtest_ai/run_configurator_load_test.py
    browser_tests:
      read_when: "when UI behavior changes"
      paths:
        - tests/unit/apps/web/test_backtests_ai_configurator.py
        - apps/web/dist/templates/pages/backtests.html
    lmstudio_docs:
      read_when: "when deciding whether LM Studio UI can reduce Roehub UI code"
      paths:
        - "https://lmstudio.ai/docs/developer/core/server"
        - "https://lmstudio.ai/docs/developer/openai-compat/structured-output"
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "if security or UI contract is ambiguous"

style_references:
  - tests/unit/apps/web/test_backtests_ai_configurator.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/security_eval_summary.md

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
    - docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/security_eval_summary.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md
  canonical_shape: "security evidence markdown plus JSON results"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_12_accepted: true
  zero_unauthorized_load_actions_required: true
  real_macstudio_pipeline_required: true
  browser_safe_text_required: true
  lmstudio_ui_not_user_facing: true
  safe_prompt_false_positive_metric_required: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  fix_security_gates: true
  review_ui_simplification: true
  run_real_pipeline_smoke: true
  run_browser_qa_if_ui_changes: true
  run_load_benchmark: false

skill_routing:
  - skill: production-risk-review
    use_when: "before final report for security and trust-boundary changes"
    timing: "before ship"
    reason: "prompt injection and unsafe load-action risk"
  - skill: browser-qa-evidence
    use_when: "if /backtests browser behavior changes"
    timing: "during verification"
    reason: "browser-visible acceptance"
  - skill: backend-quality-gates
    use_when: "running API, security, pipeline and web unit tests"
    timing: "during verification"
    reason: "test gates"
  - skill: publish-ci-deploy
    use_when: "after local, browser and Mac Studio pipeline checks pass"
    timing: "final delivery step"
    reason: "ship fixes and verify production host"

target_envs:
  - local-dev
  - browser
  - mac-studio-prod
  - github-actions

required_literals:
  - "secrets_env_vars"
  - "output_script_injection"
  - "auto_run_backtest_attempt"
  - "unauthorized actions: 0"
  - "safe prompts blocked: 0/10"
  - "10/10 ready"
  - "Load configuration"
  - "no chain-of-thought"
  - "POST /v1/chat/completions"
  - "choices[0].message.content"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not edit old prompt files 01-09."
  - "Do not expose LM Studio Developer UI to Roehub users."
  - "Do not auto-run backtest jobs from AI output."
  - "Do not run S50/S100 benchmark yet."

final_report_format:
  language: ru
  sections:
    - "Что реализовано"
    - "Security eval"
    - "Pipeline readiness"
    - "UI/UX decision"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest scripts tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - src/trading/contexts/backtest/application/ai_configurator/services/security.py
  - src/trading/contexts/backtest/application/ai_configurator/services/validator.py
  - scripts/backtest_ai/run_configurator_security_eval.py
  - tests/unit/contexts/backtest/application/ai_configurator/
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/

possible_secondary_touches:
  - apps/web/dist/js/pages/backtests.js
  - apps/web/dist/templates/pages/backtests.html
  - tests/unit/apps/web/test_backtests_ai_configurator.py
  - apps/api/routes/backtest_ai_config.py

safety_notes:
  - "LM Studio UI is an operator/developer tool, not a Roehub user-facing component."
  - "Use LM Studio structured output to reduce custom client-side complexity."
  - "Ready state must never be produced for secret exfiltration, script injection or auto-run attempts."
---

# Task

Fix the security and UX readiness blockers before benchmark rerun.

This prompt starts only after Iteration 12 proves LM Studio lifecycle and loaded-model readiness on Mac Studio. It must fix the known unauthorized load-action cases and prove the real `/backtests/ai-config` pipeline can produce valid `ready` jobs before any S1 benchmark.

Done means:

- security eval has 0 unauthorized load actions;
- `secrets_env_vars`, `output_script_injection`, and `auto_run_backtest_attempt` do not produce loadable ready configs;
- direct real API pipeline smoke on Mac Studio produces 10/10 supported jobs in `ready`;
- unsupported/off-topic prompts return friendly blocked/clarification states;
- UI remains simple: status stages, final assistant text, and `Load configuration` only for safe ready states;
- LM Studio UI is evaluated as ops/dev-only and not embedded into Roehub user UI.

## Context / Current State

Context ledger:

- completed:
  - LM Studio service lifecycle should be accepted.
  - Structured output adapter should exist.
- open_items:
  - previous security eval had 3 unauthorized load actions.
  - full pipeline ready rate was 0% in failed benchmark.
- contract_changes:
  - security gate behavior may become stricter.
  - UI may be simplified to rely on backend structured result.
- risks:
  - overblocking safe prompts;
  - exposing HTML/script through assistant text;
  - writing extra UI code that LM Studio structured output makes unnecessary.
- next_focus:
  - prove one-user real pipeline readiness and security before load.

## Requirements (Must)

- Stop if Iteration 12 service lifecycle evidence is missing or blocked.
- Fix known security eval failures to 0 unauthorized load actions.
- Keep assistant text rendered as text, never HTML.
- Ensure AI output cannot trigger auto-run of backtest jobs.
- Ensure `Load configuration` appears only for backend `ready` with validated config.
- Prove 10/10 real Mac Studio API jobs reach `ready` for supported prompts.
- Prove safe-prompt false positives are controlled: at least 10 supported safe `/backtests` prompts must have `safe prompts blocked: 0/10`. If any safe prompt is blocked, either reclassify it with a concrete unsafe reason or keep the stage blocked.
- Prove repair path works at least 5/5 on real Mac Studio or report blocker.
- Document UI decision: LM Studio Developer UI/logs are ops-only; Roehub UI should not embed them.
- Prefer using LM Studio structured output and backend statuses instead of adding custom frontend parsing. For LM Studio requests, use `POST /v1/chat/completions`, prompt text in `messages[].content`, `response_format.type=json_schema`, string-only JSON Schema `type` values, and parse `choices[0].message.content` as JSON.
- Run browser QA if UI files change.
- Markdown and JSON evidence must include explicit machine-readable gate fields: `accepted: true/false`, `blocking_reason: null|string`, and `next_prompt_allowed: true/false`.
- Run `publish-ci-deploy` after gates pass.

## Requirements (Should)

- Keep false-positive block rate visible for safe prompts.
- Add regression tests for each known security failure case.
- Add a small real-host pipeline smoke command or documented harness mode.
- Keep user-facing messages friendly and localized enough for RU/EN.

## Requirements (Nice-to-have)

- Add compact security evidence JSON alongside markdown.
- Include a small UX note comparing "use LM Studio features" vs "write Roehub code".

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 12 service lifecycle evidence
3. known security eval summary
4. task entrypoints
5. API/browser bundles only if touched
6. LM Studio docs only for UI/structured-output decision
7. consult-if-needed architecture doc

Do not load all old benchmark artifacts.

Reading budget: max 10 repo files plus 2 docs pages.

# Reading manifest

- `always_read`: repo contract, service lifecycle evidence, known security failures.
- `task_entrypoints`: security, validator, UI JS, security harness.
- `conditional_bundles`: API, browser tests, LM Studio docs only when needed.
- `consult_if_needed`: architecture doc if contract is ambiguous.

Stop reading once failure cases, code paths and verification commands are clear.

# Work plan (agent should follow)

1. Verify Iteration 12 accepted evidence.
2. Reproduce or inspect the 3 known security failure classes.
3. Fix input/output/validator gates narrowly.
4. Review UI behavior and simplify only where needed.
5. Run unit and browser tests.
6. Run Mac Studio real pipeline smoke: supported 10/10 ready, repair 5/5, security eval 0 unauthorized.
7. Write evidence.
8. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Security eval reports unauthorized actions: 0.
- Safe supported prompt false-positive blocks: 0/10, or every blocked prompt is documented as unsafe and the stage remains blocked unless the safe set still has 10/10 pass.
- Private/system leakage: 0.
- Rendered HTML/script: 0.
- Supported prompt pipeline readiness: 10/10 `ready` on Mac Studio.
- Repair smoke: 5/5 valid final states or exact blocker.
- Browser QA evidence exists if UI changed.
- LM Studio UI decision is documented.
- Evidence contains top-level gate markers: `accepted`, `blocking_reason`, and `next_prompt_allowed`; downstream prompts may proceed only when `accepted=true` and `next_prompt_allowed=true`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## UI/UX

- Do not show chain-of-thought.
- Do not add JSON editor UI unless necessary.
- Do not embed LM Studio UI or expose LM Studio server URLs to users.
- Use backend statuses and structured output to keep UI code smaller.

## Security

- Treat all model output as untrusted until schema, output gate and business validation pass.
- Do not render assistant/model text as HTML.
- Do not allow AI output to call APIs, run backtests, or create jobs.

## Documentation

- Update old/current docs and create `security_pipeline_readiness.md`.
- Run docs index check.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/ai_configurator/services/security.py`
- `src/trading/contexts/backtest/application/ai_configurator/services/validator.py`
- `scripts/backtest_ai/run_configurator_security_eval.py`
- `tests/unit/contexts/backtest/application/ai_configurator/`
- `docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md`

Possible secondary touches:

- `apps/web/dist/js/pages/backtests.js`
- `apps/web/dist/templates/pages/backtests.html`
- `tests/unit/apps/web/test_backtests_ai_configurator.py`
- `apps/api/routes/backtest_ai_config.py`

# Non-goals

- No S1/S5/S10/S50/S100 benchmark.
- No paid rollout.
- No LM Studio UI embedding.
- No old prompt edits.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py`
- `uv run ruff check apps/api apps/web src/trading/contexts/backtest scripts tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio real API pipeline smoke: supported 10/10 ready, repair 5/5, security eval unauthorized actions 0.
- Browser QA on `/backtests` if UI changed.

If Mac Studio real pipeline smoke fails, stop and report blocker. Do not run benchmark.

# Final output: report format (strict)

Report in Russian with:

- `Что реализовано`: security/UI/pipeline changes.
- `Security eval`: cases, unauthorized actions, leakage, rendered HTML/script.
- `Pipeline readiness`: Mac Studio supported/repair counts and commands.
- `UI/UX decision`: what LM Studio helps with and what Roehub still owns.
- `Доставка и Mac Studio`: publish-ci-deploy terminal state and host evidence.
