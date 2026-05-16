---
prompt_name: backtest_ai_configurator_lmstudio_v1_19_auto_intent_ui_security_regression
repo: roehub.com
branch: main
scope: "Iteration 19: verify the auto-intent single-chat UI/backend contract with security, browser and Mac Studio pipeline evidence."

language:
  implementation: qa_security_ops
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "verification, browser and delivery contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/single_chat_ui_acceptance.md
      why: "required UI acceptance"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/backend_auto_intent_acceptance.md
      why: "required backend acceptance"
  task_entrypoints:
    - path: scripts/backtest_ai/run_configurator_security_eval.py
      why: "security eval harness"
      inspect_symbols:
        - "*"
    - path: scripts/backtest_ai/run_configurator_load_test.py
      why: "pipeline/load smoke harness"
      inspect_symbols:
        - "*"
    - path: apps/web/templates/pages/backtests.html
      why: "final browser UI contract"
      inspect_symbols:
        - data-ai-prompt
        - data-ai-mode
    - path: apps/web/dist/js/pages/backtests.js
      why: "final browser payload and no-auto-run behavior"
      inspect_symbols:
        - currentAiPayload
        - submitAiPrompt
        - applyAiConfiguration
  conditional_bundles:
    lmstudio_runtime:
      read_when: "before real Mac Studio pipeline smoke"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/lmstudio_service_lifecycle.md
        - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/security_pipeline_readiness.md
    browser_tests:
      read_when: "when browser-visible behavior fails or changed"
      paths:
        - tests/unit/apps/web/test_backtests_ai_configurator.py
        - tests/unit/apps/api/test_ui_backtests_routes.py
    backend_tests:
      read_when: "when intent/security/pipeline behavior fails"
      paths:
        - tests/unit/contexts/backtest/application/ai_configurator
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
      read_when: "if target contract is ambiguous"
    - path: docs/runbooks/mac-studio-native-backend-operations.md
      read_when: "before Mac Studio production verification"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-13_lmstudio_serving_recovery/
  - docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
    - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/auto_intent_security_regression.md
  canonical_shape: "security/browser/pipeline regression evidence markdown plus JSON marker"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  depends_on_iteration_18_accepted: true
  real_browser_verification_required: true
  security_eval_required: true
  old_mode_ui_zero_reference_required: true
  model_reply_language_regression_required: true
  no_ai_backtest_execution_capability_required: true
  trusted_capabilities_regression_required: true
  external_policy_hooks_regression_required: true
  macstudio_pipeline_required_if_lmstudio_ready: true
  publish_ci_deploy_required: true
  macstudio_sync_required: true

task_toggles:
  run_security_eval: true
  run_browser_qa: true
  run_macstudio_pipeline_smoke: true
  run_full_s50_s100_benchmark: false
  update_docs: true

skill_routing:
  - skill: production-risk-review
    use_when: "reviewing final trust boundary and prompt-injection exposure"
    timing: "before verdict"
    reason: "security and production risk"
  - skill: browser-qa-evidence
    use_when: "verifying live /backtests UI"
    timing: "during verification"
    reason: "browser-visible contract"
  - skill: backend-quality-gates
    use_when: "running tests/lint/type/doc gates"
    timing: "during verification"
    reason: "local gates"
  - skill: backend-performance-evidence
    use_when: "recording Mac Studio pipeline latency and queue evidence"
    timing: "during verification"
    reason: "small pipeline regression evidence"
  - skill: publish-ci-deploy
    use_when: "after evidence/docs/config changes and gates pass"
    timing: "final delivery step"
    reason: "ship evidence and verify production host"

target_envs:
  - local-dev
  - browser
  - mac-studio-prod
  - github-actions

required_literals:
  - "unauthorized actions: 0"
  - "safe prompts blocked: 0/10"
  - "old mode UI current-active references: 0"
  - "model replies in request language"
  - "startup message uses platform locale"
  - "AI cannot run backtests"
  - "TRUSTED_CAPABILITIES"
  - "ROEHUB_BACKTEST_AI_SYSTEM_PROMPT_PATH"
  - "ROEHUB_BACKTEST_AI_SECURITY_GATES_PATH"
  - "model does not read repository source code"
  - "indicator window bounds"
  - "artifact publisher coverage"
  - "Load configuration"
  - "Загрузить конфигурацию"
  - "POST /v1/chat/completions"
  - "choices[0].message.content"
  - "JSON Schema type values must be strings"
  - "do not use type: [\"string\", \"null\"]"
  - "accepted: true/false"
  - "blocking_reason"
  - "next_prompt_allowed"

non_goals:
  - "Do not run S50/S100 unless explicitly requested after this regression."
  - "Do not reintroduce mode selector."
  - "Do not roll out to paid users automatically."
  - "Do not edit old prompt files 01-18."

final_report_format:
  language: ru
  sections:
    - "Regression verdict"
    - "Browser QA"
    - "Security eval"
    - "Mac Studio pipeline"
    - "Доставка и Mac Studio"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py"
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
  - docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/auto_intent_security_regression.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/auto_intent_security_regression.json

possible_secondary_touches:
  - scripts/backtest_ai/run_configurator_security_eval.py
  - scripts/backtest_ai/run_configurator_load_test.py
  - tests/unit/scripts/
  - docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md
  - docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md

safety_notes:
  - "Do not treat local fake-worker smoke as Mac Studio acceptance."
  - "If LM Studio runtime is not accepted/ready, record blocker instead of faking pipeline acceptance."
  - "The model must never be able to start or enqueue a backtest."
---

# Task

Run the final regression for the auto-intent single-chat `/backtests` AI Configurator contract.

This prompt starts only after Iteration 18 UI acceptance exists and passed. It verifies that the old mode-selector logic is gone, the new UI works in a real browser, the backend intent/security behavior is safe, and Mac Studio pipeline evidence exists when LM Studio runtime is ready.

Done means:

- old mode selector has zero current-active UI references;
- RU/EN startup message behavior is verified;
- model response language policy is tested;
- AI cannot run/enqueue/start backtests;
- security eval passes with unauthorized actions 0;
- safe prompt false-positive metric is visible;
- Mac Studio real pipeline smoke is green if runtime is ready, otherwise blocker is explicit;
- evidence is written and delivered.

## Context / Current State

Context ledger:

- completed:
  - Iteration 17 should implement backend auto-intent.
  - Iteration 18 should remove mode selector UI.
- open_items:
  - Need final verification across UI/backend/security.
  - Need determine whether Mac Studio LM Studio runtime is accepted and ready.
- contract_changes:
  - final user-facing AI contract is one chat, auto-intent, no mode selection.
- risks:
  - stale hidden mode UI remains;
  - backend still accepts old mode as authority;
  - safe prompts are overblocked;
  - browser UI suggests AI can run backtests;
  - previous benchmark evidence becomes stale after UX/backend contract changes.
- next_focus:
  - regression verdict and evidence.

## Requirements (Must)

- Stop if Iteration 18 evidence is missing or blocked.
- Verify no old mode UI current-active references:
  - `data-ai-mode`;
  - `backtests-ai-modes`;
  - `backtests-ai-mode`;
  - `selectAiMode`;
  - `state.ai.mode`;
  - `AI_DEFAULT_MODE`;
  - visible `CREATE`, `EDIT`, `EXPLAIN`, `REPAIR`, `SAFER` AI mode controls.
- Run browser QA on `/backtests`:
  - no mode selector;
  - startup message appears by default;
  - startup message language follows platform locale;
  - chat prompt can be submitted;
  - `Load configuration` only appears for validated ready config;
  - existing run button remains the only backtest job creation control;
  - console/network clean.
- Run security eval:
  - unauthorized actions: 0;
  - private/system leakage: 0;
  - rendered HTML/script: 0;
  - safe prompts blocked: 0/10 or every exception is documented and stage blocked unless a clean safe set still passes.
- Test language behavior:
  - RU prompt gets RU assistant response policy;
  - EN prompt gets EN assistant response policy;
  - platform locale affects only startup trusted copy, not model response language.
- Verify no AI code path calls `/api/backtests/jobs` or creates a backtest job.
- Verify the final backend trust boundary:
  - model receives `TRUSTED_CAPABILITIES`, not repository source code, raw
    manifests, private paths, DB state, runtime URLs, or platform internals;
  - active capabilities include only backend-executable indicators;
  - indicator windows outside `configs/prod/indicators.yaml` bounds or explicit
    values do not produce loadable `ready`;
  - periods beyond artifact publisher coverage do not produce loadable `ready`;
  - external system prompt/security gate hooks are preserved and covered by
    regression tests or documented host configuration evidence.
- Verify every model answer is still schema/security gated. There must be no
  free-form general backtesting chat path that bypasses output gate and
  validator; explain/discussion-style answers must remain non-loadable unless a
  validated config is present.
- If LM Studio serving/lifecycle evidence from Iterations 10-15 is accepted, run Mac Studio real pipeline smoke with at least:
  - 5 create/edit supported prompts;
  - 2 explain prompts with no load action;
  - 2 suggest safer prompts;
  - 1 repair invalid config prompt.
- If LM Studio runtime is not ready, do not fake acceptance; record blocker and keep `accepted=false`.
- If this prompt performs any direct LM Studio diagnostic, use the accepted
  request contract: `POST /v1/chat/completions`, prompt text in
  `messages[].content`, `response_format.type=json_schema`, string-only JSON
  Schema `type` values, parse `choices[0].message.content` as JSON, and never
  use `type: ["string", "null"]`.
- Write markdown and JSON evidence with `accepted`, `blocking_reason`, `next_prompt_allowed`.
- Run `publish-ci-deploy` after gates pass.

## Requirements (Should)

- Mark old Iteration 15 benchmark as pre-auto-intent if it was already run before this UX/backend change.
- Keep evidence concise but include exact commands and host.
- Include browser screenshot paths if available.

## Requirements (Nice-to-have)

- Add a small harness mode for language regression if missing.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. Iteration 18 and 17 evidence
3. task entrypoints
4. LM Studio runtime bundle only before real Mac Studio smoke
5. browser/backend test bundles only for failures
6. runbook only before production host checks

Do not preload all benchmark folders.

Reading budget: max 12 repo files plus evidence files.

# Reading manifest

- `always_read`: repo contract, UI acceptance, backend acceptance.
- `task_entrypoints`: security/load harnesses, template, JS.
- `conditional_bundles`: LM Studio runtime, browser tests, backend tests only as needed.
- `consult_if_needed`: target architecture and runbook only for ambiguity/host checks.

Stop reading once verification commands, evidence target and blocker conditions are clear.

# Work plan (agent should follow)

1. Verify Iteration 18 and 17 accepted evidence.
2. Run stale mode UI reference check.
3. Run local tests/lint/type/docs gates.
4. Run browser QA on `/backtests`.
5. Run security eval and language regression.
6. Check LM Studio/Mac Studio readiness; run real pipeline smoke only when ready.
7. Write markdown and JSON evidence.
8. Use `publish-ci-deploy`.

# Acceptance criteria (Definition of Done)

- Old mode UI current-active references: 0, or every retained historical/test fixture reference is classified and not browser-active.
- Browser QA proves no mode selector and localized startup message.
- Security metrics:
  - unauthorized actions: 0;
  - private/system leakage: 0;
  - rendered HTML/script: 0;
  - safe prompts blocked: 0/10.
- Language regression proves model response policy follows request language.
- No AI path can run/enqueue/start backtests.
- Mac Studio real pipeline smoke passes if runtime is accepted/ready; otherwise evidence records a blocker and `next_prompt_allowed=false`.
- Evidence contains `accepted`, `blocking_reason`, `next_prompt_allowed`.
- `publish-ci-deploy` reaches `deployed`, or exact `green-pr`/`blocked` state is recorded.

# Implementation constraints

## Verification

- Do not accept source inspection alone for browser-visible behavior.
- Do not accept local fake-worker smoke as Mac Studio pipeline evidence.
- Stop at first failed production safety gate.

## Documentation

- Create `auto_intent_security_regression.md` and matching JSON.
- Update target architecture docs if regression changes the contract.
- Run docs index check.

# Files to indicate (expected touched areas)

Expected primary touches:

- `docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/auto_intent_security_regression.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-15_ai_configurator_auto_intent_ux/auto_intent_security_regression.json`

Possible secondary touches:

- `scripts/backtest_ai/run_configurator_security_eval.py`
- `scripts/backtest_ai/run_configurator_load_test.py`
- `tests/unit/scripts/`
- `docs/architecture/backtest/backtest-ai-configurator-auto-intent-chat-v1.md`
- `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md`

# Non-goals

- No S50/S100 benchmark.
- No paid rollout.
- No UI redesign beyond regression fixes.
- No old prompt edits.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py`
- `uv run ruff check apps/api apps/web src/trading/contexts/backtest scripts tests/unit`
- `uv run pyright`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Browser QA on `/backtests`.
- Security eval with unauthorized actions 0.
- Mac Studio pipeline smoke if LM Studio runtime is accepted/ready.

Required delivery step: after gates pass, invoke `publish-ci-deploy` as the final step. The expected terminal state is `deployed`; if `green-pr` or `blocked`, report exact blocker and do not claim deployment.

# Final output: report format (strict)

Report in Russian with:

- `Regression verdict`: accepted/blocked/internal-only and why.
- `Browser QA`: target, locale checks, no mode selector, no auto-run.
- `Security eval`: unauthorized actions, leakage, rendered HTML/script, safe false positives.
- `Mac Studio pipeline`: real-host checks or exact blocker.
- `Доставка и Mac Studio`: publish-ci-deploy state, CI, Mac Studio sync/smoke evidence or blocker.
