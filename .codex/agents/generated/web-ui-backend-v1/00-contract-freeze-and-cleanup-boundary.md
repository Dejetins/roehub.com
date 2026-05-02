---
prompt_name: web_ui_backend_v1_00_contract_freeze_and_cleanup_boundary
repo: roehub.com
branch: current
scope: "Этап 0: зафиксировать контракты, route/endpoint map и границы удаления старого Web UI перед реализацией."

language:
  implementation: docs_inventory_python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "нормативный engineering contract, DDD, browser verification, contract impact, delivery rules"
    - path: docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md
      why: "source of truth для этапов, контрактов, API notation и параллелизации"
    - path: docs/architecture/apps/web/web-ui-design-manifest-v1.md
      why: "визуальный source of truth и правила тем"
  task_entrypoints:
    - path: apps/web/main/app.py
      why: "текущие web routes, protected gate, local /api proxy"
      inspect_symbols:
        - create_app
        - _register_routes
        - _render_protected_page
    - path: apps/web/templates
      why: "текущие templates, которые нужно классифицировать как replace/move/delete"
    - path: apps/web/dist
      why: "текущие CSS/JS assets, которые нужно классифицировать как replace/move/delete"
    - path: tests/unit/apps/web/test_app_routes.py
      why: "текущие route smoke patterns"
  conditional_bundles:
    api_surface:
      read_when: "если endpoint map неясен из плана или web routes"
      paths:
        - apps/api/main/app.py
        - apps/api/routes/backtests.py
        - apps/api/routes/strategies.py
        - apps/api/routes/identity.py
    edge_contract:
      read_when: "если меняется или уточняется same-origin /api contract"
      paths:
        - docs/runbooks/web-ui-gateway-same-origin.md
        - infra/caddy/Caddyfile.vps
  consult_if_needed:
    - path: docs/web-ui+backend-plan-deep-research.md
      read_when: "только если текущий план противоречит observed code"

hard_requirements:
  do_not_implement_page_ui: true
  classify_current_web_files: true
  freeze_route_map: true
  freeze_endpoint_map: true
  preserve_api_prefix_notation: true
  update_docs_index_if_docs_change: true

task_toggles:
  write_docs_inventory: true
  edit_runtime_code: false
  delete_files: false
  publish_after_success: true

skill_routing:
  - skill: architecture-review
    use_when: "перед правками и при сравнении плана с текущими routes/assets"
    timing: "before implementation"
    reason: "этап является review-first фиксацией фактов и границ"
  - skill: contract-impact-analysis
    use_when: "если route map, endpoint map, browser-visible defaults или API prefix contract уточняются"
    timing: "before final report"
    reason: "контрактные поверхности должны быть классифицированы до передачи агентам"
  - skill: backend-quality-gates
    use_when: "при запуске docs index или web route tests"
    timing: "during verification"
    reason: "нужно отделить introduced/pre-existing failures"
  - skill: publish-ci-deploy
    use_when: "только если docs inventory завершен на 100%, все проверки прошли, diff содержит только scope этапа"
    timing: "after verification"
    reason: "полная Roehub delivery chain: publish, CI, deploy, post-deploy verification"

target_envs:
  - local-dev
  - github-actions

required_literals:
  - "replace"
  - "move"
  - "delete"
  - "/api/... browser-visible"
  - "terminal-orange"
  - "site.css"
  - "strategy_ui.js"
  - "backtest_ui.js"

non_goals:
  - "Не реализовывать новый UI, backend endpoints или миграции."
  - "Не удалять старые файлы физически на этом этапе."
  - "Не менять production edge routing."

final_report_format:
  - "Summary: что сделано"
  - "Files changed: пути и назначение"
  - "Contracts: classification and API/schema/UI impact"
  - "Verification: команды и результаты"
  - "Publish/deploy: terminal state publish-ci-deploy или причина пропуска"
  - "Risks / follow-up: остаточные риски"

quality_gates:
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/apps/web/test_app_routes.py"
    expect: "passes if route tests are touched or route inventory claims are changed"

expected_primary_touches:
  - "docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md"
  - "docs/architecture/README.md"

possible_secondary_touches:
  - ".codex/PLANS.md"
  - "tests/unit/apps/web/test_app_routes.py"

safety_notes:
  - "Этот этап должен дать stable handoff для следующих агентов, а не начать реализацию."
  - "Не stage unrelated local changes."
  - "Если найдено противоречие между `.codex/AGENTS.md` и планом, остановись и зафиксируй conflict."
---

# Task

Зафиксировать границы реализации Web UI v1 до старта coding-этапов.

Done means:

- текущие `apps/web` routes, templates и assets классифицированы как `replace`, `move` или `delete`;
- route map и endpoint map из плана сверены с текущим кодом;
- `/api/...` нотация подтверждена как browser-visible contract, без добавления `/api` prefix в backend routers;
- список handoff-инвариантов готов для этапов 1-2;
- docs index проходит.

## Context / Current State

- Новый план и design manifest уже существуют в `docs/architecture/apps/web/`.
- Текущий UI будет утилизирован и не сохраняется как legacy skin.
- `apps/web` сейчас содержит SSR routes, static `/assets`, встроенный local `/api/*` proxy и protected gate через `/api/auth/current-user`.
- Production edge снимает `/api` prefix перед backend API.

## Requirements (Must)

- Работай в review-first режиме: сначала факты, затем узкие docs-правки.
- Добавь или уточни в плане только то, что нужно для безопасной передачи этапов агентам.
- Не меняй runtime behavior.
- Не удаляй старые templates/assets физически.
- Классифицируй contract impact: public API, browser-visible behavior, persisted schema, config.
- Если все проверки прошли и этап завершен на 100%, запусти delivery chain через `publish-ci-deploy`. Если нет, не публикуй и отчитай blocker.

## Requirements (Should)

- Сохрани документ на русском языке, технические identifiers оставь как есть.
- Держи diff минимальным.
- Зафиксируй open questions только если они реально блокируют следующий этап.

## Requirements (Nice-to-have)

- Добавить короткую таблицу `current file -> target decision`, если ее еще нет или она недостаточно явная.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
3. `docs/architecture/apps/web/web-ui-design-manifest-v1.md`
4. task entrypoints
5. conditional bundles only when the stated condition applies

Pre-implementation reading target: `<= 8 files`, `<= ~35k tokens`.

Stop reading once changed contracts are identified, touched files are bounded, and no route/API ambiguity remains.

Reading budget: keep pre-implementation reading to the smallest sufficient set; default target `<= 8 files`, `<= ~45k tokens` unless this prompt states a tighter number.
Stop reading when touched files, contract surfaces, and acceptance gates are bounded enough to implement safely.
Do not eager-load all `context_sources`, `conditional_bundles`, or `consult_if_needed` files at startup.

# Reading manifest

Use the front matter `context_sources` as the canonical reading map. Do not turn it into a repo-wide crawl.

# Work plan (agent should follow)

1. Inspect current web routes/templates/assets.
2. Compare them with the target route/file structure in the plan.
3. Record replace/move/delete decisions and handoff boundaries.
4. Update only the plan or directly related docs if gaps are found.
5. Run docs index check and relevant route tests if touched.
6. If and only if DoD is fully satisfied and all checks pass, use `publish-ci-deploy` for publish/CI/deploy/post-deploy verification.

# Acceptance criteria (Definition of Done)

- Every current web route has a target decision.
- Every current top-level template/asset has a replace/move/delete decision.
- Stage 1 and Stage 2 can start without public API or persistence ambiguity.
- Docs index check passes.
- Final report includes `Intent`, `Scope`, `Design`, `Contract impact`, `Tests`, `Docs`, `Performance`, `Runtime evidence`, `Risks`, `Handoff`.

# Implementation constraints

## API / contracts

- Do not alter public API behavior.
- Do not change same-origin production routing.

## Browser-visible behavior

- No browser claims are allowed unless runtime evidence exists; this stage is primarily docs/inventory.

# Files to indicate (expected touched areas)

- `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`
- `docs/architecture/README.md` if docs index changes

# Non-goals

- New CSS/JS implementation.
- New backend routes.
- File deletion.
- UI screenshots.

# Quality gates (must run and pass)

```bash
python -m tools.docs.generate_docs_index --check
```

Run focused route tests only if route tests or route behavior are touched:

```bash
uv run pytest -q tests/unit/apps/web/test_app_routes.py
```

# Final output: report format (strict)

Report in Russian:

- `Что сделано`
- `Contract impact`
- `Проверки`
- `Handoff`
- `Publish/deploy`: state whether `publish-ci-deploy` was run; if not, explain which DoD condition was missing.
