# Iteration 06 Single-chat UI

Дата: 2026-05-18.

Статус: accepted.

## Предварительный gate

Iteration 05 проверен перед началом:

- `implementation_progress.json`: `05-validation-repair.accepted=true`;
- `next_iteration_allowed=true`;
- `iteration_05_validation_repair.json`: `pushed_to_main=true`;
- `macstudio_verified=true`;
- recorded accepted commit: `30b5153754ee4d8de224a80dd947a009642d225f`;
- commit is ancestor of current `origin/main`.

## Что изменено

- `/backtests` AI panel now uses one conversation UI with no Create/Edit/Explain/Repair/Safer mode row.
- Added `New chat` and `History` controls inside the existing AI panel.
- The browser sends one chat input plus current `/backtests` form config to conversation API.
- Startup assistant message is created by backend with platform locale.
- Stage chips are limited to `queued`, `preparing_context`, `generating`, `validating`, `repairing`, `ready/error`.
- `Apply configuration` is rendered only from backend `load_action.enabled=true`, `state=ready`, and non-null config.
- Apply rechecks backend load-action, fills the current form, and does not call preflight or create a backtest job.
- Indicator axis rendering now keeps catalog semantics: range uses numeric fields, explicit uses discrete select, no-window uses a chip and no synthetic from/to/step inputs.

## Browser QA

Local browser target:

```text
http://127.0.0.1:8765/backtests
```

Harness:

```text
real apps.web SSR /backtests template with httpx MockTransport API responses; no production data writes
```

Evidence:

- `output/playwright/backtest-ai-iteration-06/browser_qa.json`;
- `output/playwright/backtest-ai-iteration-06/desktop-initial.png`;
- `output/playwright/backtest-ai-iteration-06/desktop-ready-apply.png`;
- `output/playwright/backtest-ai-iteration-06/desktop-indicator-axes.png`;
- `output/playwright/backtest-ai-iteration-06/mobile-ru-initial.png`.

Observed:

- no `[data-ai-mode]`, `.backtests-ai-mode`, or `.backtests-ai-modes`;
- `New chat` and `History` visible on desktop;
- Russian platform locale shows `Новый чат`, `История`, and Russian startup copy;
- Russian user prompt returns Russian assistant response in mocked backend smoke;
- `Apply configuration` appears only after backend `ready`;
- apply loads config into form and does not create a job;
- explicit `structure.percent_rank` axis renders as a select;
- no-window `volume.obv` axis renders as `No window` chip with no synthetic window inputs;
- console warnings/errors: `0`;
- unexpected failed network responses: `0`.

## Проверки

Completed locally before delivery:

```text
node --check apps/web/dist/js/pages/backtests.js
```

Result: passed.

```text
python -m json.tool apps/web/locales/en.json
python -m json.tool apps/web/locales/ru.json
```

Result: passed.

```text
uv run pytest -q tests/unit/apps/web/test_backtests_ai_configurator.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/api/test_backtest_ai_config_routes.py
```

Result: `16 passed`.

Required local gates completed before direct-main publish:

```text
uv run pytest -q tests/unit/apps/web tests/unit/apps/api
```

Result: `197 passed, 3 warnings`.

```text
uv run ruff check apps/web apps/api tests/unit/apps/web tests/unit/apps/api
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed after regenerating `docs/architecture/README.md`.

## Contract Impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | none | Existing conversation endpoints are used; no payload schema change. |
| Browser-visible behavior | compatible-change | Retired mode buttons remain absent; single-chat UI replaces disabled placeholder shell. |
| DTO schema | none | No API DTO fields changed. |
| Persisted schema | none | No migrations or storage shape changes. |
| Config schema | none | Runtime config files unchanged. |
| Request hash/cache identity | none | Backtest job request hashes are unaffected; Apply only fills form. |

## Delivery

Direct-main delivery completed.

- implementation commit: `4ce800699de3d7b1fbf49485b1498e0410c1fda6`;
- pushed to `origin/main`: true;
- `CI`: success, run `26037019297`;
- `Publish App Image`: success, run `26037019303`;
- `Deploy Backend`: success, run `26037019339`;
- `Deploy Web`: success, run `26037122636`.

Mac Studio:

- host: `MacStudioDaniil`;
- repo path: `/Users/daniildegtyarev/projects/roehub.com`;
- repo commit: `4ce800699de3d7b1fbf49485b1498e0410c1fda6`;
- runtime path: `/opt/roehub/app`;
- runtime `/backtests` template, JS, and locale file hashes match repo files;
- `scripts/macos/smoke_prod.sh`: passed;
- direct API smoke on `127.0.0.1:8000`: auth `200`, conversation list `200`, create `201`, messages `200`, status `200`, load-action `200`;
- created smoke conversation locale: `ru`;
- initial load action enabled: `false`;
- note: Mac Studio exposes Roehub API on `127.0.0.1:8000`; `127.0.0.1:3000` is Grafana, not Roehub Web SSR.

Acceptance marker:

- accepted: true;
- pushed to `origin/main`: true;
- Mac Studio verified: true;
- next iteration allowed: true.
