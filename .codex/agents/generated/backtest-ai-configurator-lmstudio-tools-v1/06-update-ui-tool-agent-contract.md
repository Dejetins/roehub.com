---
prompt_name: backtest_ai_configurator_lmstudio_tools_v1_06_update_ui_contract
repo: roehub.com
branch: main
scope: "Update the /backtests UI contract for tool-agent stages, safe assistant rendering, load-configuration behavior, and browser-visible verification."

language:
  implementation: fastapi_jinja_plain_js_css
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "browser-visible verification and repo contract"
    - path: docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
      why: "target UI/pipeline contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_pipeline_readiness.md
      why: "backend stage evidence from Prompt 05"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "AI panel markup"
      inspect_symbols:
        - data-backtests-root
    - path: apps/web/dist/js/pages/backtests.js
      why: "AI job polling/SSE and load behavior"
      inspect_symbols:
        - "*"
    - path: apps/web/dist/css/pages/backtests.css
      why: "AI panel and status styling"
      inspect_symbols:
        - "*"
    - path: tests/unit/apps/web
      why: "web route/render tests"
      inspect_symbols:
        - "*"
  conditional_bundles:
    browser_qa:
      read_when: "before runtime browser verification"
      paths:
        - docs/runbooks/mac-studio-native-backend-operations.md
    api_contract:
      read_when: "if API response/status names need clarification"
      paths:
        - apps/api/routes/backtest_ai_config.py
        - tests/unit/apps/api/test_backtest_ai_config_routes.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md
      read_when: "if old UI behavior conflicts with current tool-agent design"

documentation_continuity:
  old_current_docs:
    - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md
  new_doc_artifact: docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_ui_acceptance.md
  canonical_shape: "UI acceptance Markdown + matching JSON evidence"
  docs_gate: "uv run python -m tools.docs.generate_docs_index --check"

hard_requirements:
  safe_text_rendering_required: true
  no_raw_model_json_in_browser: true
  load_configuration_fills_existing_form: true
  browser_runtime_evidence_required: true
  no_spa_migration: true

task_toggles:
  update_stage_display: true
  update_load_button_contract: true
  run_browser_qa: true
  keep_feature_controlled: true

skill_routing:
  - skill: browser-qa-evidence
    use_when: "after browser-visible changes"
    timing: "during verification"
    reason: "verify actual /backtests behavior, console and network"
  - skill: backend-quality-gates
    use_when: "running web/API unit checks"
    timing: "during verification"
    reason: "focused pytest/ruff/pyright"
  - skill: contract-impact-analysis
    use_when: "changing browser-visible status/event names or API payload assumptions"
    timing: "before implementation"
    reason: "UI contract is externally visible"

target_envs:
  - local-dev
  - browser-runtime
  - mac-studio-prod

required_literals:
  - "intent_classification"
  - "context_collection"
  - "candidate_generation"
  - "backend_validation"
  - "repair_or_nearest_valid"
  - "Load configuration"
  - "Загрузить конфигурацию"
  - "textContent"
  - "no innerHTML"
  - "accepted: false"

non_goals:
  - "Do not convert /backtests to React/Vite/SPA."
  - "Do not stream raw model JSON to the browser."
  - "Do not add a second job creation path from Load configuration."

final_report_format:
  language: ru
  sections:
    - "UI contract"
    - "Browser QA"
    - "Security rendering"
    - "Проверки"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/web tests/unit/apps/api/test_backtest_ai_config_routes.py"
    expect: "passes"
  - cmd: "uv run ruff check apps/web apps/api tests/unit/apps"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - apps/web/templates/pages/backtests.html
  - apps/web/dist/js/pages/backtests.js
  - apps/web/dist/css/pages/backtests.css
  - tests/unit/apps/web
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_ui_acceptance.md
  - docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/tool_agent_ui_acceptance.json

possible_secondary_touches:
  - apps/api/routes/backtest_ai_config.py
  - docs/architecture/backtest/backtest-ai-configurator-tool-agent-v1.md

safety_notes:
  - "Assistant-controlled text must render as inert text."
  - "The load button fills the current form locally; it must not auto-run a backtest."
---

# Task

Update `/backtests` UI for the tool-agent pipeline.

Done means:

- UI shows expected tool-agent stages;
- assistant output is rendered safely;
- `Load configuration` / `Загрузить конфигурацию` fills the existing form only;
- browser runtime QA passes;
- feature remains controlled until final benchmark/security acceptance.

## Requirements (Must)

- Render assistant-controlled text with `textContent`, never `innerHTML`.
- Do not expose raw model JSON, tool calls, raw tool results, raw prompts, or
  private audit records in browser payload.
- Show clear stage/status behavior for:
  `intent_classification`, `context_collection`, `candidate_generation`,
  `backend_validation`, `repair_or_nearest_valid`, final ready/blocked.
- Load action:
  - available only for backend-validated loadable config;
  - fills existing `/backtests` form controls;
  - does not create or enqueue a backtest job;
  - records user feedback if existing API supports it.
- Browser QA must cover:
  - non-config prompt;
  - supported config ready path;
  - unsupported/clarification path;
  - malicious HTML/script assistant text remains inert;
  - load button fills form without navigation/job creation.

## Requirements (Should)

- Keep the UI quiet and operational, not a marketing surface.
- Preserve RU/EN behavior: startup message uses platform locale, assistant
  answer language follows user request.

# Work plan (agent should follow)

1. Read backend pipeline evidence.
2. Update JS/template/CSS contract narrowly.
3. Add unit tests.
4. Run local dev server or deployed target as appropriate.
5. Use browser QA and record evidence.
6. Write docs/evidence.

# Acceptance criteria (Definition of Done)

- Browser console has no relevant errors.
- Network calls do not expose raw model/tool internals.
- Load configuration works only for validated config.
- Evidence has `accepted`, `blocking_reason`, `next_prompt_allowed`.

# Final output: report format (strict)

Report in Russian using front-matter sections.
