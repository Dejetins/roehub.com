# Roehub Web UI v2.1 — промышленный контракт переноса

**Статус:** историческое доказательство совместимости развёрнутого контура до
будущей смены маршрутов. Базовые требования к следующему Web UI зафиксированы в
[`roehub-product-transformation-requirements-v1.md`](../../platform/roehub-product-transformation-requirements-v1.md),
но не выбирают исполнительную задачу автоматически.
Same-origin, backend authority, защита секретов, unknown-result reconciliation,
доступность и безопасный rollback остаются обязательными входными контрактами.

## Статус и цель

- Статус: `deployed; production browser verified`.
- Цель: перенести принятую концепцию Roehub Workbench в действующий `apps/web`, сохранив серверную архитектуру, API/DTO, авторизацию, deep links и безопасные операционные границы.
- Пользователи: аутентифицированный пользователь Roehub и пользователь с ограниченной backend-capability. Новая ролевая модель в v2.1 не вводится: текущий principal передаёт только identity и paid tier, а разрешение каждой операции остаётся ответственностью backend.
- Цена ошибки: скрытое изменение торгового состояния, повтор опасной операции, раскрытие секрета, неверный operational status или потеря доступа к существующему сценарию.

## Не входит

- переход production Web UI на React или SPA;
- изменение доменной семантики стратегий, бэктестов, exchange connections или ML runtime;
- добавление новых money-moving mutation contracts без отдельного доменного решения;
- хранение секретов или provider payloads в браузере, HTML, логах или доказательствах.

## Подтверждённое текущее состояние

- `apps/web` использует FastAPI SSR, Jinja2, CSS и vanilla JavaScript/HTMX.
- `/strategies`, `/backtests` и `/settings` уже используют same-origin `/api/*` и реальные UI read/write contracts.
- Exchange connection lifecycle уже включает create, validate, rotate, disable, archive, account configuration и audit events.
- `/monitoring` реализован как read-only поверхность поверх существующего dashboard summary contract.
- Прототип `prototypes/roehub-v2` является визуальным эталоном, но не технологическим шаблоном production.
- Production shell уже перенесён в `apps/web`: единственная sidebar navigation, command palette, mobile bottom navigation, шесть browser-local тем и route-owned состояния.

## Решения оболочки

### Единственная навигация

- На desktop существует одна primary sidebar navigation.
- Отдельный global rail запрещён.
- Command search принадлежит top context bar и не дублируется в sidebar.
- Notifications имеют один вход в utility section sidebar.
- User menu находится только в sidebar footer.
- На mobile используются bottom navigation из пяти пунктов и отдельный More sheet.

### Контекст и детали

- Persistent inspector появляется только когда текущая страница имеет соответствующий выбранный объект.
- Inspector не переносит strategy context на Backtests, Connections, Settings или другую несвязанную поверхность.
- Для страниц без выбранного объекта используется компактный page context либо inspector скрыт.
- Document tabs из прототипа не переносятся в production: они дублируют primary navigation и не имеют устойчивого server-side ownership contract.

### Глобальный статус

- Постоянная нижняя status strip из прототипа не переносится в production.
- Environment и общий connection state принадлежат top context bar; freshness и refresh controls принадлежат конкретному data module.
- Activity drawer не является обязательной постоянной поверхностью и не занимает viewport по умолчанию.
- На mobile freshness показывается локально внутри модулей.

## Канонические маршруты v2.1

| Возможность | Маршрут | Реализация |
|---|---|---|
| Auth Gateway | `/` и `/login` | guest видит login-only surface; authenticated user перенаправляется на `/dashboard` |
| Overview | `/dashboard` | существующий dashboard read model |
| Strategies | `/strategies`, `/strategies/{strategy_id}` | существующий strategies dashboard и lifecycle contracts |
| Backtests | `/backtests`, `/backtests/{job_id}` | существующий workstation, queue, result/materialization contracts |
| Live | `/monitoring` | новая read-only композиция на существующих dashboard/strategy/account read contracts |
| Models | `/models` | канонический вход в существующий Strategies RL/ML mode |
| Connections | `/connections` | канонический вход в существующий Account exchange connection lifecycle |
| Settings | `/settings` | profile, limits, integrations, notifications, sessions, security и audit |

Compatibility routes сохраняются. Deep links используют реальные entity identifiers; hash используется только для локального tab state и не заменяет URL identity.

### Владельцы новых маршрутов

| Route | Server owner | Template / asset owner | Compatibility |
|---|---|---|---|
| `/models` | `apps/web/main/app.py`; existing `/api/ui/strategies/*` | `pages/strategies.html`, `strategies.css`, `strategies.js`, initial mode `rl_ml` | `/strategies?mode=rl_ml` продолжает работать |
| `/connections` | `apps/web/main/app.py`; existing `/api/ui/account/*` | `pages/connections.html`, account fragments, `settings.css`, `settings.js` | `/settings#api` продолжает работать |
| `/monitoring` | `apps/web/main/app.py`; existing `/api/ui/dashboard/summary` | `pages/monitoring.html`, `dashboard.css`, `dashboard.js` | прежний placeholder заменяется read-only workspace |

Каждый route имеет собственный active navigation key и canonical link. Новые страницы не создают отдельные доменные API и не меняют identity существующих сущностей.

## Визуальный и адаптивный контракт

- Dark baseline: `graphite`; light baseline: `paper`.
- Дополнительные темы: `abyss`, `slate`, `frost`, `sand`.
- Body copy: `14-16px`; data labels не менее `11px`.
- Desktop control height: `36-40px`; mobile control/touch target: не менее `44px`.
- Spacing scale: `4, 8, 12, 16, 24, 32, 48`.
- Module radius: `6-8px`; overlays: `8-12px`.
- Таблицы имеют controlled local scroll, sticky headings и keyboard-accessible row actions.
- `200%` zoom, reduced motion, `en`/`ru`, `1440`, `1024`, `820` и `390` входят в acceptance.

Тема сохраняется локально в browser preference и не расширяет server `ThemeValue` в этой итерации. Legacy browser values отображаются безопасно: `terminal-orange -> abyss`, `matrix-green -> slate`, `high-contrast -> paper`; неизвестное значение становится `graphite`. Account preferences API остаётся совместимым и не получает новые enum values в рамках visual migration.

## Контракт состояний

Каждый data module поддерживает применимые состояния:

`loading -> ready -> empty -> degraded -> stale -> error -> retrying -> restricted -> unauthorized -> rate_limited -> materializing -> unknown mutation result`.

- Галерея состояний не считается доказательством, пока состояние не встроено в реальный модуль.
- Unknown mutation result никогда не предлагает слепой повтор: UI читает authoritative state или показывает reconciliation path.
- Loading не стирает последнюю валидную snapshot без явной причины.
- Ошибка одного модуля не должна скрывать здоровые соседние модули.

## Действия, роли и доверие

Каждое видимое действие получает один из статусов:

- `implemented` — связано с production contract;
- `navigation-only` — выполняет явный переход;
- `disabled-with-reason` — недоступно и объясняет причину;
- `deferred` — не показывается как активная кнопка.

Для mutation action обязательны:

- backend authorization как единственный источник истины; UI не выводит доступность из визуальной роли и не расширяет полномочия;
- capability/permission gate только там, где соответствующая capability уже присутствует в production DTO;
- pending/disabled state;
- success/error feedback;
- confirmation для destructive/high-risk операций;
- recent-auth там, где её требует текущий security contract;
- отсутствие автоматического повтора небезопасной mutation; при timeout/transport/`5xx` UI помечает результат как неизвестный и перечитывает authoritative state;
- `X-Request-Id` для трассировки, без заявления об идемпотентности там, где endpoint не принимает стабильный idempotency key;
- audit event без secret-bearing payload.

## Интеграционные границы

| Caller | Callee | Изменение v2.1 | Failure contract |
|---|---|---|---|
| Browser | `apps/web` SSR/assets | новая shell и templates | recoverable HTML error; v2.1 добавляет `private, no-store` для protected HTML |
| Browser | same-origin `/api/ui/dashboard/*` | composition для Overview/Monitoring | bounded read retry; module-local degraded/stale state |
| Browser | same-origin `/api/ui/strategies/*` | новый visual composition, semantics unchanged | mutation state читается authoritative endpoint до повторения |
| Browser | same-origin `/api/ui/backtests/*` | новый visual composition, semantics unchanged | create identity и materialization state сохраняются |
| Browser | same-origin `/api/ui/account/*` | Connections становится отдельным route entry | secrets write-only; rotate/disable/archive подтверждаются и reconciled |
| Browser | Keycloak/OIDC | login-only public entry | Roehub не принимает local password; timeout проверяет session |

`Cache-Control: private, no-store` для protected SSR responses является целевым изменением v2.1, а не описанием прежнего состояния. Оно проверяется unit-тестом и через production response headers; public login не содержит персональных данных.

## Производительность

- SSR first paint сохраняется.
- Новые shell assets не добавляют внешние CDN или runtime dependency.
- Page JavaScript остаётся route-scoped.
- Target: `LCP < 2.5s`, `INP < 200ms`, `CLS < 0.1` на production profile при измеряемом сценарии.
- Таблицы свыше 50 строк используют pagination/virtualization contract; графики получают bounded series.
- Browser evidence фиксирует viewport, route, auth state, console и failed network requests.

### Воспроизводимая матрица проверки

- маршруты: `/`, `/dashboard`, `/strategies`, `/backtests`, `/monitoring`, `/models`, `/connections`, `/settings`;
- состояния: guest, authenticated ready, loading, empty, degraded/stale, error, restricted и unknown mutation result там, где применимо;
- экраны: `1440x1024`, `1024x768`, `820x800`, `390x844`, плюс `200%` browser zoom без глобального horizontal overflow;
- темы и язык: `graphite`, `paper`, `en`, `ru` на representative Auth, Strategies, Monitoring и Connections;
- клавиатура: skip link, sidebar, command dialog, tabs, menus, exchange dialog, Escape, trap и restore focus;
- производительность: production Chrome, cold navigation, normal network profile, representative production payload; по три измерения на Auth, Overview и Backtests, медиана для `LCP`/`INP`/`CLS`;
- доступность: целевой уровень WCAG 2.2 AA; автоматический smoke дополняется ручной клавиатурой, `200%` zoom и проверкой доступных имён. Полная сертификация экранным диктором не заявляется без отдельного прогона.

## Этапы и откат

1. Foundation: tokens, themes, single navigation shell, command palette, responsive frame.
2. Core pages: Auth, Overview, Strategies, Backtests, contextual inspector.
3. Operational pages: Monitoring, Models, Connections, Settings.
4. Hardening: states, backend capabilities, i18n, accessibility, performance, browser evidence.
5. Historical delivery: scoped commit to `main`, publication evidence and
   browser proof on `https://roehub.com`.

Каждый этап обратим через narrow file revert. Описанный здесь runtime rollout
является historical evidence: его workflow и host-specific upstream выведены из
эксплуатации. До появления отдельного утвержденного runtime ticket публикация
не дает production claim и не выполняет rollback workflow.

Этот раздел сохраняется как историческое описание прежней поставки и не
разрешает её возобновление. Любая будущая работа начинается из отдельно
выбранного текущего ticket по действующему delivery contract.

## Contract impact

| Измерение | Классификация |
|---|---|
| Public API | `none` для visual migration; additive read composition при необходимости — `compatible-change` |
| DTO/persistence identity | `none` |
| Config/theme preferences | `compatible-change` |
| Browser-visible shell/navigation | `breaking-change` — намеренная замена |
| `/` guest behavior | `breaking-change` — marketing landing заменяется login-only entry |
| `/models`, `/connections` | `compatible-change` — новые route entry поверх существующих contracts |
| `/monitoring` | `compatible-change` — placeholder заменяется read-only workspace |

## Доказательства завершения

- focused unit/security/assets tests;
- docs index check;
- browser flow для Auth, Overview, Strategies, Backtests, Monitoring, Models, Connections и Settings;
- desktop/tablet/mobile, dark/light, keyboard/focus, reduced motion и `200%` zoom;
- отсутствие console errors и unexpected failed network requests;
- scoped Git manifest и green CI;
- historical browser/API proof на `https://roehub.com`; он не является текущим
  runtime acceptance.

Исторический факт относится к runtime revision
`c07befd119062b067d6df9e35287a839fe69d1e4`: CI, публикация SHA-tagged GHCR
image, public edge smoke и authenticated production browser smoke завершились
успешно.
