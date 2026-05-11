---
prompt_name: web_ui_backend_v1_stage09_00_inventory_current_contract
repo: roehub.com
branch: main
scope: "Stage 09 readiness inventory: сверить текущий Web UI `/backtests`, backend result endpoints и документацию перед реализацией result panels."

language:
  implementation: python_fastapi_jinja_plain_js
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, prompt execution and reporting rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "Stage 09 source of truth"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "current backtest artifact/result contracts"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      why: "runner/materialization dependency for cache-miss result data"
    - path: .codex/agents/generated/web-ui-backend-v1/09-implement-backtests-results.md
      why: "parent Stage 09 prompt"
  task_entrypoints:
    - path: apps/web/templates/pages/backtests.html
      why: "current UI data endpoints and panel expectations"
    - path: apps/web/dist/js/pages/backtests.js
      why: "current browser calls and result-state model"
    - path: apps/api/routes/backtests.py
      why: "public backtest result routes"
    - path: apps/api/dto/backtests.py
      why: "public result DTOs"
    - path: apps/api/routes/ui_backtests.py
      why: "workstation aggregate route"
    - path: apps/api/dto/ui_backtests.py
      why: "workstation DTO shape"
    - path: src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
      why: "result read methods and lazy trades dependency"
    - path: src/trading/contexts/backtest/application/services/v2/result_series.py
      why: "series/stat/trades read model builders"
    - path: tests/unit/apps/api/test_backtests_routes.py
      why: "current endpoint coverage"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "current UI route/asset expectations"

hard_requirements:
  read_current_code_first: true
  no_runtime_behavior_change_by_default: true
  verify_no_endpoint_drift: true
  identify_backend_gaps_for_stage09: true
  do_not_reimplement_existing_routes: true

task_toggles:
  implementation_changes_allowed: false
  docs_patch_allowed_if_contract_drift_found: true
  publish_after_success: false

package_contract:
  depends_on:
    - "Stage 08/08.5 prompt outcomes or current code inspected directly"
    - "current /backtests UI treated as source of truth"
  owns:
    - "optional narrow docs/prompt corrections if a concrete drift is found"
  forbidden:
    - "runtime code changes"
    - "database migrations"
    - "template/CSS/JS implementation"
  integration_points:
    - "Stage 09 backend materialization prompt"
    - "Stage 09 UI prompt"
  handoff:
    - "endpoint inventory table"
    - "UI expectation table"
    - "gap list with blocker/non-blocker classification"

skill_routing:
  - skill: prompt-manager
    use_when: "auditing prompt scope and handoff readiness"
    timing: "before final report"
    reason: "this file is a prompt-pack readiness task"
  - skill: architecture-review
    use_when: "checking docs/code/prompt drift without implementation"
    timing: "during inventory"
    reason: "review-first stage"
  - skill: contract-impact-analysis
    use_when: "classifying endpoint/DTO/schema/browser-visible gaps"
    timing: "during final classification"
    reason: "Stage 09 changes cross API, DTO, cache and browser contracts"

target_envs:
  - local-dev
  - github-actions

required_literals:
  - "/api/backtests/jobs/{job_id}/summary"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}"
  - "POST /api/backtests/jobs/{job_id}/variants/{variant_key}/trades"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats"
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats"
  - "GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page="
  - "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv"
  - "/api/ui/backtests/workstation"
  - "backtest_lazy_trades_materializations"
  - "renderBacktestSeries"

non_goals:
  - "Do not implement charts or result panels in this prompt."
  - "Do not add new endpoints in this prompt."
  - "Do not change current UI behavior."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_app_routes.py"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web"
    expect: "passes or exact pre-existing failure classification"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
---

# Task

Сделать readiness inventory для реализации Stage 09 `backtests results`.

Нужно не писать runtime-код, а точно сверить, что уже реализовано, что ожидает текущий `/backtests` UI, какие result/statistics endpoints уже есть, и какие gaps нужно закрыть следующими промтами pack-а.

## Context / Current State

Текущий Web UI `/backtests` является первичным источником правды для browser behavior. Stage 09 не должен начинать с generic result UI и не должен дублировать уже существующие result endpoints.

Известный риск: часть существующих result/statistics методов может синхронно пересчитывать lazy trades при cache miss. Для production это должно быть вынесено в materialization/worker path или явно возвращать typed pending/degraded state.

## Requirements (Must)

- Составить таблицу текущих browser-visible endpoints, которые реально вызывает `/backtests`.
- Составить таблицу backend endpoints, которые уже есть в `apps/api/routes/backtests.py`.
- Составить таблицу use-case методов в `BacktestJobsUseCase`, которые обслуживают summary, variant, equity, drawdown, monthly stats, symbol stats, paginated trades, CSV.
- Отметить для каждого метода источник данных: persisted summary, top variant row, lazy/materialized trades, artifact fallback, unavailable/degraded.
- Отметить, где есть риск sync recompute в API process.
- Проверить тестовое покрытие и назвать missing tests, которые должны быть добавлены следующими промтами.
- Если документация или parent prompt прямо противоречат текущей реализации, внести узкую docs/prompt правку. Если противоречий нет, не менять файлы.

## Requirements (Should)

- Разделить findings на `blocker`, `must-fix before UI expansion`, `can-fix with UI`, `non-blocker`.
- Сохранять distinction между browser-visible `/api/...` path и backend router path без двойного `/api`.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, Stage 09 section in the plan, parent Stage 09 prompt, then the listed task entrypoints. Stop reading once route/method/DTO/test inventory is complete.

Reading budget: default `<= 12 files`, `<= ~60k tokens`. Do not inspect unrelated strategy/settings/dashboard code unless a direct import or failing test requires it.

# Work Plan

1. Inspect current `/backtests` template and JS data endpoints.
2. Inspect backend route declarations and DTO names.
3. Inspect `BacktestJobsUseCase` result methods and data dependencies.
4. Inspect focused tests.
5. Produce inventory and gap matrix.
6. Run focused gates.
7. Patch docs/prompt only if there is a concrete contradiction with current code.

# Acceptance Criteria

- Inventory lists every current `/backtests` UI call and every existing result/statistics route.
- Inventory explicitly says whether current UI calls charts/stat/trades endpoints or only summary/variant/CSV.
- All gaps needed before prompt 09 implementation are named with exact files/functions.
- No runtime behavior is changed.
- Focused gates run or failures are classified.

# Final Output: Report Format

Report in Russian with these exact sections:

- `Intent`: цель inventory.
- `Current contract`: tables or concise lists for UI calls, backend endpoints, use-case methods.
- `Gaps`: blockers and required follow-up prompts.
- `Contract impact`: classify API, DTO, persisted schema, config, cache identity, browser-visible behavior.
- `Tests`: exact commands, cwd, result.
- `Docs`: docs/prompt changes or explicit none.
- `Risks`: sync recompute, missing materialization, runner dependency, UI blockers.
- `Handoff`: what prompt 01/02/03 must implement next.
- `Publish/deploy`: skipped unless a narrow docs/prompt patch was intentionally shipped by a separate explicit delivery request.
