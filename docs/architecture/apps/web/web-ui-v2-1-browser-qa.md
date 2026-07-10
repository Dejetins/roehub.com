# Roehub Web UI v2.1 — браузерная и инженерная проверка

Документ фиксирует воспроизводимые локальные доказательства переноса Roehub Workbench в production-контур `apps/web`, границы проверки и обязательный production gate.

## Статус

- Дата локальной проверки: `2026-07-10`.
- Статус: `deployed; production browser verified`.
- Runtime: FastAPI SSR, Jinja2, CSS, vanilla JavaScript и HTMX.
- Runtime revision: `c07befd119062b067d6df9e35287a839fe69d1e4`.
- Production URL: `https://roehub.com`.

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

## Production delivery и повторная проверка

- CI: run `29092837934`, `success`.
- Publish App Image: run `29092905501`, `success`.
- Deploy Web: run `29092963148`, `success`.
- Развёрнутый образ: `ghcr.io/dejetins/roehub-app:c07befd119062b067d6df9e35287a839fe69d1e4`.
- VPS container: `roehub-web-web-1`, image SHA-tag совпадает с runtime revision.
- Public edge smoke workflow подтвердил `/ -> 200`, `/api/auth/current-user -> 401` для гостя и redirect `www -> 301`.
- Authenticated smoke выполнен учётной записью `smoke_e2e_keycloak`; значение учётных данных получено из host-local env и не выводилось, не сохранялось и не попадало в снимки.
- `/dashboard`, `/strategies`, `/backtests`, `/monitoring`, `/models`, `/connections`, `/settings` загрузили production data-ready состояния без видимых alert/error.
- На каждом маршруте: одна sidebar, один `h1`, нет глобальной status strip, нет горизонтального overflow и активных целей меньше `36px` при фактическом desktop viewport `1280x1024`.
- Strategies проверен при `1024x768`, `820x800` и `390x844`; на `390` sidebar скрыт, mobile navigation активна, document width равен `390px`, цели не меньше `44px`.
- Production data обнаружили и позволили исправить два дополнительных дефекта: compact delete target `30px -> 36px` и mobile action group, расширявшую документ до `481px`.
- Русская оболочка, темы `graphite`/`paper`, command dialog, `Escape` и restore focus подтверждены в production.
- Отдельный authenticated обход приложения в новом tab не зафиксировал `console error/warn`; единственный ранний `AbortError: Transition was skipped` принадлежал навигации Keycloak и не воспроизводился внутри Roehub app.

## Остаточные риски

- Production smoke подтверждает текущий smoke-account и наблюдавшиеся payloads, но не покрывает другие уровни доступа, предельные объёмы данных и искусственно увеличенные задержки API.
- Двукратный CSS layout-scale smoke Auth Gateway прошёл без глобального overflow; нативный browser zoom, screen-reader walkthrough и Web Vitals требуют отдельного специализированного evidence. Полное соответствие WCAG не заявляется.
- Прототипный React bundle не переносится в production и не влияет на размер SSR assets.
- Prototype Vite chunk warning остаётся изолированным от SSR production bundle и не блокирует текущий Web UI.

Contract impact: публичные API/DTO/persistence — `none`; новые route entry `/models` и `/connections` — `compatible-change`; `/monitoring` из placeholder в read-only workspace — `compatible-change`; shell, navigation и guest `/` — намеренный `breaking-change` browser-visible поведения.
