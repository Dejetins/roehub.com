# Iteration 01 Reset старой AI ветки

Дата: 2026-05-17.

Статус: accepted.

## Цель

Убрать из current code/docs/config старую `/backtests` AI configurator ветку:
mode buttons, one-shot `/backtests/ai-config/jobs*`, `lm_studio_tools` и
tool-agent как текущий runtime target. Исторические документы и prompt packs
остаются только как evidence/tombstone.

## Что изменено

- Browser-visible `/backtests` больше не публикует `data-ai-config-*` endpoints
  и не рендерит old mode buttons.
- JS страницы `/backtests` больше не отправляет old one-shot job payload, не
  открывает SSE для AI jobs и не записывает feedback в old endpoint.
- API route `apps/api/routes/backtest_ai_config.py` оставлен как пустой retired
  router без `/backtests/ai-config/jobs*` endpoints.
- UI workstation state возвращает disabled reset state: `enabled=false`,
  `state=reset`, `modes=[]`, `endpoints={}`.
- Current env configs используют disabled placeholder
  `runtime: assistant_v1_pending` вместо retired runtime.
- Worker/runtime wording переименован с tool-agent pending на assistant v1
  runtime pending.
- Unit tests переписаны так, чтобы old UI/API контракт не закреплялся как
  active behavior.

## Что удалено из current path

| Поверхность | Результат |
| --- | --- |
| Browser template | removed old `data-ai-config-*` endpoint attrs and mode row |
| Browser JS | removed old one-shot submit, status polling/SSE, mode payload and feedback client |
| Locales | removed `backtests.ai.mode*` labels |
| API route | removed old POST/GET/SSE/feedback routes from active router |
| UI read model | `modes=[]`, `endpoints={}`, `state=reset` |
| Config | `lm_studio_tools` no longer appears in current env configs |
| Worker health | `assistant_v1_runtime_pending` replaces old readiness blocker |

## Stale-reference classification

Command:

```text
rg -n "lm_studio_tools|tool_agent|backtests\\.ai\\.mode|edit_current|repair_invalid|suggest_safer|/backtests/ai-config/jobs" src apps configs infra scripts tests docs/architecture .codex/agents/generated
```

Current production refs in `src`, `apps`, `configs`, `infra`, and active tests:
none after reset.

Remaining matches are classified as:

| Location | Classification | Reason |
| --- | --- | --- |
| `docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md` | intentionally retained current requirements | Source doc names old literals as things to remove and records breaking-change scope. |
| `docs/architecture/backtest/backtest-ai-configurator-mlx-v1.md` | historical/tombstone | Superseded reset document; updated to stop describing old runtime as current. |
| `docs/architecture/backtest/benchmark_iterations/2026-05-12_*` and `2026-05-13_*` | historical evidence | Old benchmark/blocker artifacts preserved for traceability only. |
| `docs/architecture/backtest/benchmark_iterations/2026-05-17_lmstudio_tools_contract/*` | historical evidence | Prior cleanup evidence for retired path. |
| `.codex/agents/generated/backtest-ai-configurator-mlx-v1/*` | historical prompt pack | Old prompt pack preserved, not executable current guidance. |
| `.codex/agents/generated/backtest-ai-configurator-assistant-v1/*` | prompt-pack requirements | Current prompt artifacts preserve literals as removal gates. |
| `scripts/backtest_ai/configurator_benchmark_common.py` | legacy benchmark helper | Retained only for historical benchmark evidence; not a current acceptance path for assistant v1. |

Focused active-path command:

```text
rg -n "lm_studio_tools|tool_agent|backtests\\.ai\\.mode|edit_current|repair_invalid|suggest_safer|/backtests/ai-config/jobs" src apps configs infra tests
```

Result: no matches.

## Browser smoke

Local SSR harness opened `/backtests` at `http://127.0.0.1:18110/backtests`.

Observed state:

- old mode labels in DOM: false;
- old AI endpoint references in DOM: false;
- `data-ai-config-*` dataset keys: empty;
- manual backtest run button exists and is enabled;
- AI prompt and submit controls are disabled reset shell;
- console errors/warnings: 0;
- workstation request returned 200.

Screenshots:

- `output/playwright/backtests-reset-desktop.png`;
- `output/playwright/backtests-reset-mobile.png`.

## Проверки

Completed local checks:

```text
uv run pytest -q tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/contexts/backtest/application/ai_configurator/test_backtest_ai_configurator_runtime_config.py tests/unit/contexts/backtest/application/ai_configurator/test_lmstudio_runtime_lifecycle.py tests/unit/contexts/backtest/application/ai_configurator/test_backtest_ai_configurator.py tests/unit/contexts/backtest/application/ai_configurator/test_backtest_ai_config_pipeline.py tests/unit/apps/worker/test_backtest_ai_configurator_worker.py
```

Result: `52 passed`.

```text
uv run pytest -q tests/unit/apps/api/test_backtest_ai_config_routes.py tests/unit/apps/web/test_backtests_ai_configurator.py
```

Result: `7 passed`.

```text
uv run ruff check apps/api apps/web src/trading/contexts/backtest tests/unit/apps/api tests/unit/apps/web
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

```text
uv run ruff check .
```

Result: passed.

```text
uv run pytest -q -ra
```

Result: `966 passed, 3 warnings`.

```text
git diff --check
```

Result: passed.

## Contract impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | breaking-change | Old `/backtests/ai-config/jobs*` endpoints are removed from active router. No compatibility bridge. |
| Browser-visible behavior | breaking-change | Old mode buttons and old AI job client are gone; AI block is disabled reset shell. |
| DTO schema | breaking-change for retired AI API | Old `mode` request is no longer browser/API-active; dormant internals now use `assistant_v1`. |
| Config schema | breaking-change | Current configs require `runtime: assistant_v1_pending`; `lm_studio_tools` is rejected. |
| Port contract | compatible-change | Existing internal gateway protocol remains, but pending runtime wording changed. |
| Persisted schema | none | No migrations or table shape changes in this iteration. |
| Request hash/cache identity | none | Core `/backtests/jobs` manual request identity unchanged. |
| Backtest jobs API | none | Manual `/backtests/jobs` path is not changed. |

## Mac Studio

Accepted commit: `a6d49673fc83de71923a1a0982b80ad1ccadcd34`.

Mac Studio host: `MacStudioDaniil`.

Runtime layout: `/opt/roehub/app` is a deployed source copy without `.git`, so
the accepted commit was verified by deployment workflow SHA plus file hashes for
the changed runtime surfaces.

Checks:

- local and Mac Studio SHA256 match for:
  - `apps/web/templates/pages/backtests.html`;
  - `apps/web/dist/js/pages/backtests.js`;
  - `apps/api/routes/backtest_ai_config.py`;
  - `configs/prod/backtest_ai_configurator.yaml`.
- Active deployed path grep:

```text
grep -RInE "lm_studio_tools|tool_agent|backtests\\.ai\\.mode|edit_current|repair_invalid|suggest_safer|/backtests/ai-config/jobs" src apps configs infra tests
```

Result: no matches.

- Retired endpoints:

```text
POST /backtests/ai-config/jobs -> 404
GET /backtests/ai-config/jobs/test -> 404
```

- Production smoke:

```text
ssh macstudio 'cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh'
```

Result: passed.

## Delivery

Direct-main delivery:

- commit: `a6d49673fc83de71923a1a0982b80ad1ccadcd34`;
- `origin/main`: `a6d49673fc83de71923a1a0982b80ad1ccadcd34`;
- CI: `26001986711`, success;
- Deploy Backend: `26001986709`, success;
- Publish App Image: `26001986707`, success;
- Deploy Web: `26002013188`, success.

## Acceptance marker

Accepted: true.

Blocking reason: none.

Next iteration allowed: true.
