# Roehub Web UI v2.1 — браузерная и инженерная проверка

Документ фиксирует воспроизводимые локальные доказательства переноса Roehub Workbench в production-контур `apps/web`, границы проверки и обязательный production gate.

## Статус

- Дата локальной проверки: `2026-07-10`.
- Статус: `local browser verified; production delivery pending`.
- Runtime: FastAPI SSR, Jinja2, CSS, vanilla JavaScript и HTMX.
- Проверяемая версия: рабочее дерево до scoped commit в `main`.
- Production считается обновлённым только после green CI, workflow `Deploy Web` и повторного smoke на `https://roehub.com`.

## Реализованная поверхность

| Область | Основные владельцы |
|---|---|
| Единая оболочка и маршруты | `apps/web/main/app.py`, `apps/web/templates/base.html` |
| Адаптивность, модульный ритм, темы | `apps/web/dist/css/workbench.css`, `apps/web/dist/css/themes.css` |
| Command palette, dialogs, focus restore | `apps/web/dist/js/core/workbench-shell.js` |
| Auth Gateway | `apps/web/templates/pages/login.html` |
| Overview | `apps/web/templates/pages/dashboard.html`, `apps/web/dist/js/pages/dashboard.js` |
| Strategies и Models | `apps/web/templates/pages/strategies.html`, `apps/web/dist/js/pages/strategies.js` |
| Backtests | `apps/web/templates/pages/backtests.html`, `apps/web/dist/js/pages/backtests.js` |
| Live | `apps/web/templates/pages/monitoring.html`, существующий dashboard summary contract |
| Connections и Settings | `apps/web/templates/pages/connections.html`, account fragments, `apps/web/dist/js/pages/settings.js` |
| Безопасные mutation outcomes | `apps/web/dist/js/core/api.js` |
| Прототип и visual source | `prototypes/roehub-v2` |

## Локальная браузерная методика

Проверка выполнена в реальном встроенном Chromium. SSR использовал локальный тестовый principal `browser-smoke/free`; API upstream намеренно не запускался. Это позволяет проверить реальные `loading -> error -> retry` состояния и отсутствие ложных данных, но не доказывает production-ready payload. Ready-state и реальная авторизация проверяются после deploy отдельно.

Маршруты:

- `/`;
- `/dashboard`;
- `/strategies`;
- `/backtests`;
- `/monitoring`;
- `/models`;
- `/connections`;
- `/settings`.

Размеры:

- `1440x1024`;
- `1024x768`;
- `820x800`;
- `390x844`.

Темы: `graphite` и `paper`. Проверена безопасная browser-local смена темы без изменения server enum. На representative маршрутах проверены английская оболочка, доступные названия, `Escape`, начальный фокус и восстановление фокуса.

## Результаты браузерной проверки

| Проверка | Результат |
|---|---|
| Primary navigation | ровно одна sidebar на каждом authenticated route; на mobile она скрыта и заменена bottom navigation |
| Горизонтальное переполнение | отсутствует на `1440`, `1024`, `820` и `390` |
| Заголовочная структура | ровно один `h1` внутри `main` на каждом рабочем маршруте |
| Нижняя status strip | отсутствует |
| Desktop active targets | нет активных целей меньше `36x36px` |
| Mobile active targets | нет активных целей меньше `44x44px` |
| Command palette | input получает фокус; `Escape` закрывает dialog; фокус возвращается opener |
| False affordances | действия без production handler отключены с причиной либо скрыты |
| Error states | Overview, Strategies, Backtests, Monitoring и Models показывают локальный error/retry state без фиктивных данных |
| Connections/Settings failure | mutation controls остаются безопасными; status не сообщает ложный `ready` |
| Console | неожиданные `error` и `warn` отсутствуют |

## Инженерные проверки

- `node --check` для изменённых production JavaScript assets — выполнено;
- `uv run pytest -q tests/unit/apps/web/...` — `47 passed`, четыре известные `httpx` deprecation warnings;
- `uv run ruff check apps/web/main tests/unit/apps/web` — выполнено;
- `uv run pyright apps/web/main` — `0 errors, 0 warnings`;
- `npm run build` в `prototypes/roehub-v2` — выполнено; остаётся неблокирующее Vite warning о prototype chunk `795.54 kB`;
- `git diff --check` — выполнено на локальном дереве.

## Независимая проверка архитектурного контракта

- Режим: независимый агент.
- Первичный вердикт: `Block`.
- Найденные блокеры: неверная web deploy boundary, несуществующая новая RBAC-модель, необоснованная идемпотентность exchange mutations, drift оболочки и маршрутов, темы вне server enum, неполная accessibility/performance matrix.
- Исправлено: целевой deploy закреплён за GHCR/VPS workflow; Mac Studio оставлен backend upstream; backend остаётся единственным источником полномочий; неизвестный mutation result требует authoritative refetch без слепого повтора; темы сохраняются только в browser-local preference с legacy mapping; маршруты, владельцы и проверочная матрица зафиксированы явно.
- Локальная повторная проверка: противоречия устранены; production readiness остаётся закрыт до delivery proof.

## Остаточные риски и следующий gate

- Локальный smoke не доказывает реальные data-ready состояния, задержки API и production payload volume.
- `200%` zoom, screen-reader walkthrough и Web Vitals требуют отдельного production evidence; полное соответствие WCAG не заявляется.
- Прототипный React bundle не переносится в production и не влияет на размер SSR assets.
- Нельзя использовать слово `deployed`, пока scoped commit не опубликован, CI и `Deploy Web` не завершены успешно, а production browser smoke не пройден.

Contract impact: публичные API/DTO/persistence — `none`; новые route entry `/models` и `/connections` — `compatible-change`; `/monitoring` из placeholder в read-only workspace — `compatible-change`; shell, navigation и guest `/` — намеренный `breaking-change` browser-visible поведения.
