---
prompt_name: backtest_ai_configurator_assistant_v1_01_reset_current_branch
repo: roehub.com
branch: main
scope: "Reset the current broken /backtests AI configurator path before rebuilding assistant v1."

language:
  implementation: python_fastapi_jinja_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
      why: "assistant v1 source of truth"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
      why: "iteration state"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json
      why: "machine-readable iteration state"
  task_entrypoints:
    - path: configs/prod/backtest_ai_configurator.yaml
      why: "old AI configurator runtime/config shape"
    - path: apps/api/routes/backtest_ai_config.py
      why: "old one-shot AI job API boundary"
    - path: apps/web/templates/pages/backtests.html
      why: "browser-visible AI block"
    - path: apps/web/dist/js/pages/backtests.js
      why: "browser AI client behavior"
  conditional_bundles:
    locales_and_tests:
      read_when: "old mode labels, old routes, or old tests are found"
      paths:
        - apps/web/locales/en.json
        - apps/web/locales/ru.json
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
        - tests/unit/apps/web/test_backtests_ai_configurator.py
    old_prompt_packs:
      read_when: "current code/docs refer to old tool-agent or old job endpoints"
      paths:
        - .codex/agents/generated/backtest-ai-configurator-mlx-v1/
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "you need to classify historical references without deleting evidence"

style_references:
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md

hard_requirements:
  remove_old_modes_from_current_path: true
  remove_old_ai_job_endpoints_from_current_path: true
  preserve_core_backtest_jobs_api: true
  update_progress_artifacts: true
  direct_main_publish_after_acceptance: true

task_toggles:
  implement_backend_api: true
  implement_web_ui_cleanup: true
  implement_lmstudio_runtime: false
  run_macstudio_smoke: true

skill_routing:
  - skill: root-cause-debugging
    use_when: "a removal breaks /backtests loading or existing backtest job behavior"
    timing: "during implementation"
    reason: "localize regression instead of masking it"
  - skill: contract-impact-analysis
    use_when: "removing old routes, config keys, mode labels, or browser payload fields"
    timing: "before final report"
    reason: "this is a deliberate breaking change for the old AI path"
  - skill: backend-quality-gates
    use_when: "running API/backend tests"
    timing: "during verification"
    reason: "backend correctness"
  - skill: browser-qa-evidence
    use_when: "checking browser-visible /backtests after UI cleanup"
    timing: "during verification"
    reason: "the AI block is browser-visible"
  - skill: publish-ci-deploy
    use_when: "all tests pass, Mac Studio smoke passes, evidence marker has accepted=true, and delivery is allowed"
    timing: "before final report"
    reason: "direct push accepted commit to origin/main, wait main CI/deploy, then sync/verify Mac Studio; no PR or feature branch"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "backtests.ai.mode.create"
  - "backtests.ai.mode.edit_current"
  - "backtests.ai.mode.explain_current"
  - "backtests.ai.mode.repair_invalid"
  - "backtests.ai.mode.suggest_safer"
  - "POST /backtests/ai-config/jobs"
  - "GET /backtests/ai-config/jobs/{job_id}"

non_goals:
  - "Do not implement the new assistant conversation API in this iteration."
  - "Do not implement LM Studio adapter in this iteration."
  - "Do not delete historical evidence documents; classify them as historical if needed."
  - "Do not alter the core /backtests/jobs manual backtest API."

final_report_format:
  language: ru
  sections:
    - "Что изменено"
    - "Что удалено из current path"
    - "Контрактное влияние"
    - "Проверки"
    - "Mac Studio"
    - "Delivery"
    - "Следующий этап"

quality_gates:
  - cmd: "rg -n \"lm_studio_tools|tool_agent|backtests\\.ai\\.mode|edit_current|repair_invalid|suggest_safer|/backtests/ai-config/jobs\" src apps configs infra scripts tests docs/architecture .codex/agents/generated"
    expect: "all matches classified as removed, historical, or intentionally retained; no current production refs"
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py"
    expect: "passes, or missing/renamed tests replaced with focused equivalent"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - configs/prod/backtest_ai_configurator.yaml
  - configs/dev/backtest_ai_configurator.yaml
  - configs/test/backtest_ai_configurator.yaml
  - apps/api/routes/backtest_ai_config.py
  - apps/web/templates/pages/backtests.html
  - apps/web/dist/js/pages/backtests.js
  - apps/web/locales/en.json
  - apps/web/locales/ru.json
  - tests/unit/apps/api/test_backtest_ai_config_routes.py
  - tests/unit/apps/web/test_backtests_ai_configurator.py
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_01_reset.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/iteration_01_reset.json
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_ai_configurator_assistant_v1/implementation_progress.json

possible_secondary_touches:
  - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
  - docs/architecture/backtest/README.md
  - docs/architecture/README.md
  - .codex/agents/generated/backtest-ai-configurator-mlx-v1/

safety_notes:
  - "Leave /backtests usable for manual configuration and manual backtest runs."
  - "Do not silently stage unrelated user changes."
---

# Task

Implement Iteration 01 from `backtest-ai-configurator-assistant-v1.md`: remove the current broken AI configurator branch from active code/docs/config so the assistant v1 rebuild starts from a clean boundary.

Done means:

- old AI mode buttons and old mode payload are gone from current browser-visible UI;
- old one-shot AI job endpoints are removed from active API/docs/tests;
- historical evidence remains available but is not described as current target;
- implementation progress and iteration evidence are updated;
- accepted changes are published with `publish-ci-deploy` and verified on Mac Studio only after all gates pass.

## Context / Current State

The previous LM Studio/tool-agent attempts are not the target. Assistant v1 uses a single chat, backend intent classification, LM Studio chat completions, backend validation/repair, and no model-triggered backtest execution.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Remove old mode labels and payload fields from active UI, locales, tests, and API contracts.
- Remove or rewrite old `/backtests/ai-config/jobs*` current routes; do not leave a compatibility bridge.
- Preserve core `/backtests/jobs` manual backtest behavior.
- Create `iteration_01_reset.md` and `iteration_01_reset.json`.
- Update `implementation_progress.md` and `implementation_progress.json`.
- Evidence JSON must contain `accepted`, `blocking_reason`, `next_iteration_allowed`, `commit`, `host`, `pushed_to_main`, `origin_main_commit`, `macstudio_verified`, `macstudio_commit`, and touched paths.
- If any old current reference remains, set `accepted=false` and stop.
- After local checks, browser smoke, Mac Studio smoke, and accepted evidence pass, use `publish-ci-deploy` to publish. Then sync/verify the accepted commit on Mac Studio and record the deployed/checked commit.
- Delivery contract: use `publish-ci-deploy` in explicit direct-main mode only. Do not create a feature branch, draft PR, or PR branch. Stage only scoped files, commit on `main`, and push to `origin/main` only after all gates pass and evidence has `accepted=true`; wait for relevant main CI/deploy; then pull/sync the exact commit on Mac Studio and run the iteration-specific smoke. If direct main push, CI, or Mac Studio verification cannot be completed, set `accepted=false`, `next_iteration_allowed=false`, and report the blocker.

## Requirements (Should)

- Prefer narrow deletion/rewrite over broad refactor.
- Keep tombstone wording clear for historical prompt packs/docs.

## Requirements (Nice-to-have)

- Add a short note in evidence describing why old AI path was retired.

# Context acquisition protocol

Read only in this order: `.codex/AGENTS.md`, assistant v1 source doc, progress artifacts, task entrypoints, then conditional bundles only if matches/gates require them. Pre-implementation target: <= 8 files unless a gate fails.

# Acceptance criteria (Definition of Done)

- `rg` stale-reference classification proves zero current production references to old modes/job endpoints/tool-agent.
- Relevant focused tests pass or are replaced with focused equivalent.
- `/backtests` still loads and manual backtest controls are not broken.
- Mac Studio smoke confirms the accepted branch/commit does not expose the old current AI branch.
- `iteration_01_reset.json` has `accepted=true`, `next_iteration_allowed=true`, `pushed_to_main=true`, and `macstudio_verified=true` after delivery.

# Implementation constraints

Use `apply_patch` for manual edits. Do not use destructive git commands. Do not stage unrelated changes.

# Final output: report format (strict)

Respond in Russian with the sections listed in front matter. Include exact evidence file paths, commands run, Mac Studio result, main commit/deploy if `publish-ci-deploy` was used, and blockers if not accepted.
