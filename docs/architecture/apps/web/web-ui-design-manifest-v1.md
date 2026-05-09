# Дизайн-манифест Roehub Web UI v1

Документ фиксирует визуальный источник правды для новой версии Roehub Web UI: терминальный продуктовый интерфейс поверх `apps/web`, без сохранения текущей светлой реализации как наследуемого режима.

## Статус

- источник правды для визуальной модели Web UI v1, синхронизированный с текущей реализацией `apps/web`;
- если этот документ расходится с уже реализованным Stage 1-5 shell/dashboard/settings baseline, первичным источником считается текущая реализация, а манифест должен быть уточнен без изменения runtime behavior;
- план реализации: `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md`;
- исследовательский ввод: `docs/web-ui+backend-plan-deep-research.md`;
- обновление 2026-05-03: палитра, header и shell-инварианты сверены с `personal_settings.png`, `stategy_backtest.png` и текущим Stage 1 screenshot; текущая orange-grid реализация не является целевым дизайном;
- обновление 2026-05-04: для функциональных страниц введен `reference fidelity contract`; после пересборки Stage 1-5 он применяется как target для еще не выполненных или явно переоткрытых этапов, но не отменяет текущий принятый shell/dashboard/settings baseline;
- обновление 2026-05-05: зафиксированы auth UX contract, фирменные dropdown/popover controls вместо системных select-меню, а также manual refresh/autorefresh contract для live-data страниц;
- обновление 2026-05-09: зафиксирована текущая UI-модель Stage 1-5: `>_ Roehub` shell, login modal, отдельная `/register`, compact theme/language dropdowns, глобальная нижняя status/refresh строка, dashboard/settings без отдельной full-width page command bar;
- актуальная canonical map Web UI v1 содержит ровно 5 визуальных страниц: `/`, `/dashboard`, `/settings`, `/strategies`, `/backtests`;
- референсы дизайна:
  - `/Users/daniildegtyarev/Projects/roehub_web_ui/general_page.png`;
  - `/Users/daniildegtyarev/Projects/roehub_web_ui/personal_settings.png`;
  - `/Users/daniildegtyarev/Projects/roehub_web_ui/personal_dashboard.png`;
  - `/Users/daniildegtyarev/Projects/roehub_web_ui/stategy_backtest.png`;
  - `/Users/daniildegtyarev/Projects/roehub_web_ui/strategy_statistic.png`.

## Цель

Дать агентам единый контракт по стилю, сетке, компонентам, цветам, темам, типографике и визуальным критериям приемки. Страницы должны реализовываться параллельно, но выглядеть как части одного продукта.

## Контекст

Текущий `apps/web` уже является FastAPI SSR/Jinja2-фасадом поверх backend JSON API и содержит принятый Stage 1-5 baseline: темный terminal shell, Jinja2 pages/fragments, CSS-токены, branded dropdown/listbox/combobox controls, login modal, i18n helper, theme helper, dashboard/settings page modules и same-origin `/api/*` proxy. Оставшиеся legacy templates или routes можно использовать только как источник маршрутов и API-точек во время миграции, но не как визуальную основу для новых функциональных страниц.

Референсы задают другой продуктовый язык: темная терминальная рабочая панель, плотные таблицы, тонкие янтарно-оранжевые контуры, моноширинные числа, зеленые/красные значения доходности и минимальная декоративность. В референсах нет полноэкранной декоративной orange-grid подложки: сетка допустима внутри графиков, таблиц и data panels, но не как фон всего приложения.

## Жесткий reference fidelity contract

Для функциональных страниц Roehub Web UI v1 PNG-референс является не moodboard и не "inspiration", а визуальным контрактом. Часть страницы **после глобальной шапки сайта** должна быть `reference-shaped`: повторять структуру, плотность, панельную сетку, порядок информационных зон, таблицы, графики, командную строку страницы и нижнюю статусную строку назначенного референса. Допускается заменить бренд `QUANT CLI` на `Roehub`, перевести пользовательский текст на `en`/`ru`, адаптировать ширины под реальные данные и свернуть layout на mobile, но нельзя заменять референс generic cards, marketing blocks или dashboard overview без соответствующей панельной структуры.

Текущий checkpoint Stage 1-5 уточняет этот контракт: уже реализованные `/dashboard` и `/settings` считаются валидным baseline даже там, где они используют глобальную нижнюю status/refresh строку и panel toolbars вместо отдельной full-width command bar сразу под header. Для еще не выполненных `/strategies` и `/backtests` агент должен следовать назначенному PNG жестче и явно перечислять panel inventory до верстки. Если какой-либо этап требуется переделать под буквальную PNG-точность, это считается новым scope и должно быть отражено в соответствующем prompt pack до реализации.

Каноническая карта страниц:

| Route | Канонический PNG | Смысл страницы | Обязательный подход |
|---|---|---|---|
| `/` | `general_page.png` | публичный лендинг | отдельный landing contract; текущим обновлением не пересматривается. |
| `/dashboard` | `personal_dashboard.png` | dashboard по всем стратегиям/fleet workstation | полная панельная рабочая поверхность all-strategies monitoring, а не обзорные карточки. |
| `/settings` | `personal_settings.png` | личный кабинет/account settings | профиль, биржевые подключения, лимиты, интеграции, уведомления, безопасность, сессии, аудит. |
| `/strategies` | `strategy_statistic.png` | dashboard/statistics по конкретной выбранной стратегии | selected-strategy analytics workstation: strategy control, visual tabs, statistics tabs, trades history. |
| `/backtests` | `stategy_backtest.png` | backtest workstation/configurator | конфигурация, AI/config zone, instruments, indicators, optimization progress/results в одной плотной рабочей поверхности. |
| `/monitoring` | нет отдельного PNG в v1 map | compatibility/ops route only | не является primary strategy dashboard; если route сохраняется, он не должен забирать reference у `/strategies`. |
| `/backtests/{job_id}` | нет отдельного PNG в v1 map | optional deep link/API state | не является шестой функциональной страницей v1; если route сохраняется, он должен открывать `/backtests` с выбранной job/result state или служить API-backed detail state без отдельного reference layout. |

Визуальные страницы v1 ограничены пятью PNG выше. `/monitoring`, `/strategies/new`, `/strategies/{strategy_id}`, `/backtests/new` и `/backtests/{job_id}` могут существовать только как compatibility redirects/aliases или state внутри canonical page.

Reference fidelity acceptance:

- каждый implementation-agent обязан открыть назначенный PNG до верстки и перечислить panel inventory в implementation notes или final report;
- desktop Playwright screenshot должен визуально совпадать с референсом по крупной сетке: те же зоны, аналогичный порядок, аналогичная плотность и command/status bars;
- generic `rh-dashboard-card` / "overview cards" вместо панельной workstation-компоновки считается introduced failure;
- если backend пока не дает данные для панели, UI сохраняет форму панели из референса и показывает typed `unavailable/degraded/empty` state, а не удаляет панель;
- финансовые значения, графики и таблицы должны использовать реальные bounded DTOs или явный mock/test fixture только в local visual QA; production route не должен выдумывать PnL/ROI;
- mobile может перестраивать панели во вкладки/stack, но порядок и приоритет данных остаются совместимыми с desktop reference;
- отсутствующий canonical PNG для функциональной страницы является blocker для implementation stage, а не поводом строить страницу по памяти.

## Охват

- публичный лендинг;
- каркас приложения: шапка, вкладки, действия входа/регистрации, управление пользователем и аккаунтом;
- мультиязычный UI: основной язык `English`, дополнительный язык `Русский`, переключатель языка;
- обзорная панель;
- настройки и аккаунт;
- dashboard/statistics выбранной стратегии внутри `/strategies`;
- backtest workstation/configurator;
- переиспользуемые панели, таблицы, бейджи, кнопки, формы, вкладки, прогресс, графики и статусные строки;
- переключение тем и палитр.

## Что не входит

- React, Next.js, SPA-маршрутизатор или runtime Node-сервер;
- миграция на Tailwind/Bootstrap;
- сохранение текущего светлого UI как наследуемого режима;
- декоративные градиенты, сферы или абстрактные фоны;
- вычисление backtest-задач или крупная агрегация данных на клиенте;
- произвольные SVG-иконки внутри страниц без общего механизма;
- изменение цветов доходности через пользовательские темы.
- локализация URL-маршрутов, API paths, DTO fields, market symbols, strategy ids, `variant_key`, `job_id` и других технических identifiers.

## Ключевые решения

### 1) Бренд: Roehub, терминальный мотив

Канонический продуктовый бренд в репозитории остается `Roehub`. Надпись `QUANT CLI` из части макетов является стилистическим референсом по плотности, терминальному ритму и header-композиции, а не переименованием продукта. В UI допустим терминальный мотив `>_`, но брендовые подписи, заголовки документов, метаданные страниц и продуктовый текст используют `Roehub`.

Инварианты:

- публичный текстовый логотип: `ROEHUB` или `Roehub` в зависимости от страницы и языка;
- квадратный монограммный знак `RH` не используется в v1 shell; это не часть референсов и должен быть удален из шапки;
- в защищенном app shell брендовая зона компактная: текстовый бренд слева, без отдельной крупной иконки и без декоративного logo-card;
- терминальный маркер `>_` допустим рядом с брендом или в command bar, но он не заменяет бренд и не должен превращаться в повторяющийся corner badge;
- правый край header занят auth/account controls; не добавлять в правый или левый угол отдельные `RH`, `v1` или аналогичные декоративные corner labels;
- идентификаторы маршрутов и API остаются на английском;
- технические подписи вроде `BTCUSDT`, `PnL`, `TP`, `SL`, `API`, `variant_key` сохраняются в исходном написании.

### 2) Мультиязычность: English first, Russian second

Весь сайт и защищенное приложение строятся как мультиязычные. Основной язык продукта - английский (`en`), дополнительный язык - русский (`ru`). Документация проекта может оставаться на русском, но пользовательский UI-copy не должен быть зашит только на русском.

Инварианты:

- поддерживаемые locale на v1: `en`, `ru`;
- default locale: `en`;
- `ru` является полным вторым языком для пользовательских labels, headings, button text, empty/error states и help text;
- `<html lang>` и корневой `data-locale` соответствуют выбранному языку;
- все маршруты, API paths, DTO fields, enum values, market symbols, strategy ids, config keys и technical identifiers остаются английскими и не переводятся;
- финансовые обозначения `PnL`, `ROI`, `Return`, `Drawdown`, `Win Rate`, `TP`, `SL`, `variant_key`, `job_id` допускается оставлять в исходном техническом виде;
- default copy для новых страниц пишется на английском, русская версия добавляется в тот же этап;
- hardcoded user-visible strings в templates/JS допустимы только для технических identifiers или временного dev-only состояния, явно помеченного в handoff.

Правила выбора языка:

1. authenticated account preference, когда settings/backend preference API уже доступен;
2. locale cookie, установленная language switcher-ом для SSR;
3. browser-local preference в `localStorage` для client-side hydration/switching;
4. `Accept-Language`, если нет пользовательского выбора;
5. fallback `en`.

Language switcher:

- доступен в shell рядом с account/auth controls, компактнее primary navigation;
- на desktop это короткий control `EN / RU` или меню `Language`;
- на mobile он остается доступным из account/menu drawer;
- переключение языка не меняет route и не требует локализованных path aliases;
- переключение обновляет cookie/localStorage и перерисовывает SSR/fragment surface через reload или controlled refresh;
- после реализации settings/account preference выбор синхронизируется в backend preference.

### 2.1) Auth UX: login как modal, registration как отдельная страница

Авторизация в Roehub Web UI v1 не должна открывать отдельную полноэкранную login page как основной пользовательский путь. Основной login UX - фирменное модальное окно поверх текущей страницы или landing/shell, оформленное теми же токенами, рамками и typography, что и остальные controls.

Правила:

- кнопка `Login` / `Войти` открывает branded auth modal/dialog;
- modal содержит короткий account/auth summary, sanitized `next` target и primary action, запускающий существующий Keycloak/OIDC flow через `/api/auth/login`;
- modal не реализует локальную username/password форму Roehub;
- прямой `GET /login?next=...` допускается только как compatibility/deep-link entrypoint: он рендерит shell/landing с login modal уже открытым или выполняет контролируемый переход к такому состоянию;
- `next` всегда остается safe local path; внешние URL отбрасываются;
- login modal закрывается по `Esc`, backdrop и явной кнопке close, возвращая focus на исходный control;
- при `401` во время polling/SSE UI останавливает live loops и показывает login modal/banner, а не уводит пользователя на отдельную страницу без контекста;
- registration остается отдельной страницей `GET /register`, потому что onboarding/registration требует больше контекста, legal/security copy и Keycloak-backed registration/get-started flow;
- registration page использует тот же global shell, i18n, theme tokens и branded controls, но не является modal внутри dashboard/workstation.

### 3) Визуальная модель: темная терминальная панель

Интерфейс должен ощущаться как плотный операционный терминал для системной торговли, а не как маркетинговая панель с декоративной карточной компоновкой.

Базовые признаки:

- почти черный холст;
- тонкие оранжевые контуры;
- компактные панели с жесткой геометрией;
- плотные таблицы и сетки метрик;
- малоконтрастные разделители;
- акцентный цвет используется экономно для фокуса и основных действий;
- графики в первую очередь показывают данные и не превращаются в иллюстрации.

Запрещено:

- полноэкранная декоративная orange-grid / graph-paper подложка страницы;
- крупные скругленные карточки;
- вложенные карточки;
- большие hero-карточки;
- пастельные поверхности;
- однообразные синие, slate- или фиолетовые палитры;
- фоновые сферы, боке и абстрактные градиенты;
- стоковые изображения на страницах приложения.

Разрешенная сетка:

- chart-grid внутри графика equity/PnL/latency;
- табличные разделители и panel-grid для данных;
- очень тонкие внутренние линии в dense data panels.

Запрещенная сетка:

- `background-image: linear-gradient(...)` на `body`, `.app-shell`, `.page` или hero-секции;
- крупная orange-grid, видимая за всеми блоками страницы;
- декоративная сетка, которая конкурирует с данными и рамками панелей.

### 4) Палитра по умолчанию и переключение тем

Янтарно-оранжевая терминальная палитра из референсов фиксируется как **палитра по умолчанию**. Она темнее, холоднее и менее зеленая, чем текущая Stage 1 реализация на screenshot: фон должен быть почти черным с легким сине-зеленым оттенком, а не темно-зеленой сеткой. Это не единственная допустимая палитра: продукт должен поддерживать переключение тем/цветовых палитр.

Обязательные правила:

- все цвета UI задаются через CSS-переменные в `apps/web/dist/css/tokens.css`;
- выбранная тема применяется через атрибут на корневом элементе, например `data-theme="terminal-orange"`;
- тема по умолчанию: `terminal-orange`;
- системная тема может использоваться только как первичный запасной вариант до пользовательского выбора;
- пользовательский выбор темы должен сразу сохраняться локально в браузере, а после реализации настроек аккаунта - синхронизироваться через backend-предпочтение;
- переключение темы не должно требовать перезагрузки страницы;
- переключатель темы должен быть доступен из каркаса приложения, но не как ряд крупных текстовых theme-buttons в primary header; целевое место - компактное меню аккаунта/настроек или отдельный компактный control;
- полное управление темой доступно на странице настроек;
- новые страницы не имеют права фиксировать цвета напрямую вне слоя токенов.

Обязательные слои темы:

- `base`: фон, поверхности, текст, линии, сетки;
- `accent`: основные действия, фокус, активная вкладка, контуры;
- `state`: статусы процессов, предупреждения, отключение, активность, очередь;
- `financial`: доходность, процентные изменения, PnL, ROI, return, drawdown.

Критичный инвариант: **тема может менять цвета `base`, `accent` и `state`, но не должна менять базовую семантику финансовых цветов**. Все значения, связанные с доходностью и процентными изменениями, всегда используют одно и то же семантическое соответствие:

- положительная доходность / положительный PnL / положительный `return_pct`: зеленый;
- отрицательная доходность / отрицательный PnL / drawdown/loss: красный;
- нулевое или отсутствующее изменение: приглушенный нейтральный цвет;
- предупреждение или процессный статус не должны окрашивать финансовые значения, если сами значения доходности положительные или отрицательные.

Это значит, что переключатель палитр может сделать интерфейс, например, с cyan- или graphite-акцентом, но `+3.84%`, `-1.12%`, `Total PnL`, `ROI`, `Return`, `Drawdown`, `Win Rate` и аналогичные финансовые изменения сохраняют фиксированную семантическую окраску.

### 5) Источник CSS-токенов

Токены реализуются в `apps/web/dist/css/tokens.css`. CSS страниц должен использовать токены, а не повторять буквальные значения цветов.

Палитра по умолчанию:

```css
:root,
:root[data-theme="terminal-orange"] {
  color-scheme: dark;

  --rh-bg: #020607;
  --rh-bg-elevated: #050a0c;
  --rh-surface: #071012;
  --rh-surface-2: #0a1416;
  --rh-surface-3: #10191b;

  --rh-line: rgba(255, 106, 0, 0.62);
  --rh-line-strong: #ff6a00;
  --rh-line-muted: rgba(255, 106, 0, 0.28);
  --rh-divider: rgba(255, 106, 0, 0.18);
  --rh-chart-grid-line: rgba(111, 84, 53, 0.18);

  --rh-text: #d8d2c9;
  --rh-text-strong: #fff1df;
  --rh-muted: #918a82;
  --rh-muted-2: #5f5a53;

  --rh-accent: #ff7a00;
  --rh-accent-2: #ff9d00;
  --rh-warning: #ff9d00;
  --rh-info: #00a6c8;
  --rh-focus: #ffb000;
  --rh-shadow-panel: none;
  --rh-shadow-glow: 0 0 18px rgba(255, 106, 0, 0.08);

  --rh-financial-positive: #57d33b;
  --rh-financial-negative: #ff3b30;
  --rh-financial-neutral: #9d9890;

  --rh-state-success: #57d33b;
  --rh-state-danger: #ff3b30;
  --rh-state-warning: #ff9f0a;
  --rh-state-info: #24b8ff;
  --rh-state-muted: #6d6961;

  --rh-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --rh-font-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;

  --rh-space-1: 4px;
  --rh-space-2: 8px;
  --rh-space-3: 12px;
  --rh-space-4: 16px;
  --rh-space-5: 24px;
  --rh-space-6: 32px;

  --rh-radius-1: 2px;
  --rh-radius-2: 4px;
  --rh-radius-3: 6px;

  --rh-header-height: 44px;
  --rh-commandbar-height: 36px;
  --rh-footer-height: 30px;
  --rh-page-max: 1920px;
}
```

Важно: `--rh-chart-grid-line` используется только внутри графиков, chart panels и отдельных data-areas. Для общего фона страницы сетка запрещена. Если временно сохраняется устаревший alias `--rh-grid-line`, он не должен применяться к `body`, `.app-shell`, `.page-shell`, hero или общему container background.

Минимальный набор тем для текущего Web UI v1:

- `terminal-orange` - тема по умолчанию, совпадает с референсами;
- `graphite` - нейтральная темная тема с приглушенным акцентом.

Дополнительные палитры добавляются только через новые блоки токенов. Они не должны менять `--rh-financial-positive`, `--rh-financial-negative`, `--rh-financial-neutral`, если это не отдельное решение доступности с явным ревью.

### 6) Типографика и плотность

Типографика приложения по умолчанию:

- моноширинный шрифт для навигации, подписей, таблиц, метрик, кнопок и кодоподобных значений;
- sans-шрифт только для длинных маркетинговых или текстовых абзацев, где он улучшает читаемость;
- не масштабировать размер шрифта от ширины viewport;
- межбуквенный интервал равен `0`, кроме малых uppercase-подписей навигации и секций, где максимум `0.04em`;
- отрицательный межбуквенный интервал запрещен.

Шкала:

- основной текст приложения: 13-14px;
- ячейки таблиц: 12-13px;
- заголовки панелей: 13-15px, uppercase или терминальный стиль;
- значения метрик: 16-24px в зависимости от размера панели;
- только `H1` лендинга может использовать 32-44px.

Текст не должен вылезать за кнопки, вкладки, заголовки таблиц или статусные бейджи. Длинные значения в таблицах обрезаются с `title`/tooltip, а не расширяют компоновку.

Требования к мультиязычному тексту:

- английские и русские строки проверяются на одинаковых responsive breakpoints;
- UI containers проектируются под более длинную из двух локалей;
- если русская строка не помещается в compact control, использовать короткий перевод или меню, а не уменьшать шрифт через viewport units;
- language switcher не должен ломать header width и не должен вытеснять primary nav.

### 7) Система компоновки

Каркас приложения:

- фиксированная верхняя шапка высотой около `--rh-header-height`: текстовый бренд слева, primary navigation по центру/слева, auth/account controls справа;
- активный nav item оформляется тонкой оранжевой рамкой/подложкой как в `personal_settings.png` и `stategy_backtest.png`;
- header не содержит крупный theme switcher из нескольких текстовых кнопок; theme control допускается только как компактный элемент account/settings;
- header содержит компактный language switcher (`EN/RU` или меню), расположенный рядом с account/auth controls и не конкурирующий с primary nav;
- header не содержит квадратный `RH`-логотип, отдельный `GUEST` chip как декоративный badge, corner `v1` или footer-brand дубликаты;
- текущий shell использует бренд `>_ Roehub`, compact theme dropdown, compact locale dropdown, auth/account controls и общий bottom status/refresh bar;
- отдельная full-width command bar под шапкой не является обязательной для уже реализованных `/dashboard` и `/settings`: page actions могут жить в panel headers, toolbar controls и bottom status bar;
- для будущих reference-shaped страниц `/strategies` и `/backtests` command bar добавляется только если она нужна для совпадения с назначенным PNG или workflow;
- фиксированная или липкая нижняя статусная строка на плотных страницах приложения является частью текущей модели shell;
- максимальная ширина контента `--rh-page-max`; на лендинге допустимы полноширинные секции.

Настольная компоновка:

- плотная CSS-сетка; базовый промежуток 8px или 12px;
- панели выравниваются по сетке и имеют единый ритм границ;
- страницы приложения приоритизируют сканирование данных, а не пустое пространство;
- графики и большие таблицы получают стабильную минимальную высоту и `overflow: auto`.

Мобильная компоновка:

- не дублировать настольные сайдбары;
- основные двухколоночные экраны сворачиваются во вкладки или вертикальные секции;
- липкая нижняя статусная строка может стать компактной полосой статуса;
- каждая интерактивная цель на touch-экранах имеет высоту минимум 36px;
- таблицы либо получают горизонтальную прокрутку с явным affordance, либо переходят в строки-карточки с тем же приоритетом данных.

### 8) Правила компонентов

#### Панели

Панели используются для оформленных рабочих областей и повторяющихся карточек. Не вкладывать карточки в карточки.

Панель:

- 1px `--rh-line` border;
- прозрачная или малоконтрастная темная поверхность;
- прямые углы или радиус 2-4px;
- заголовочная строка с терминальным маркером `>_` там, где это уместно;
- без тяжелой тени.

#### Кнопки

Типы кнопок:

- primary: заливка акцентным цветом для одного доминирующего действия на экране;
- secondary: прозрачная кнопка с акцентной границей;
- destructive: прозрачная кнопка с красной границей/текстом; красная заливка только для критического подтверждения;
- icon-only: для знакомых действий, если есть иконка и доступное имя.

Правило иконок:

- если базовый этап добавляет self-hosted Lucide sprite/module, использовать Lucide-иконки для знакомых действий;
- не рисовать одноразовые SVG-иконки внутри кода страницы;
- если доставка иконок еще не готова, использовать короткие текстовые кнопки и не имитировать иконки декоративными символами, кроме терминального маркера.

#### Таблицы

Таблицы являются основным интерфейсным элементом:

- фиксированные заголовки там, где это полезно;
- плотные строки высотой 32-40px;
- числовые колонки выровнены вправо;
- статусные точки плюс текст;
- без zebra-полос, если контраста достаточно;
- hover строки меняет границу/фон только тонко;
- состояние пагинации и фильтров не должно менять размер таблицы.

#### Бейджи и статусы

Статусные бейджи кодируют состояние процесса, а не финансовый смысл:

- `ready`, `active`, `connected`, `succeeded`: `--rh-state-success`;
- `running`, `queued`, `warming_up`: `--rh-state-warning`;
- `failed`, `disconnected`, destructive: `--rh-state-danger`;
- `cancelled`, `stopped`, disabled: `--rh-state-muted`;
- unknown или stale: предупреждение плюс явный timestamp/lag там, где возможно.

Финансовые значения кодируют доходность независимо от темы:

- положительные PnL/ROI/return/change: `--rh-financial-positive`;
- отрицательные PnL/ROI/return/change/drawdown: `--rh-financial-negative`;
- нулевые/отсутствующие значения: `--rh-financial-neutral`.

#### Формы

Формы используют компактные терминальные элементы управления:

- подписи сверху или слева в зависимости от плотности;
- поля ввода высотой 34-40px;
- select/dropdown/listbox/menu-элементы реализуются через фирменные Roehub controls, а не через видимые системные popup-меню браузера/OS;
- видимый native `<select>` с серым системным dropdown, как в стандартном browser UI, считается визуальным failure для функциональных страниц;
- native `<select>` допустим только как hidden/progressive-enhancement fallback, если видимый control остается branded combobox/listbox/menu с теми же токенами;
- branded dropdown допускает GitHub-like popover-поведение: темная поверхность, тонкая рамка, icon/label rows, разделители, keyboard navigation, visible focus и portal/overlay-layer внутри shell;
- все выпадающие списки должны оставаться в одной стилистике с terminal panels: без системных серых меню, без light popup, без platform-specific arrow-only surface;
- сегментированные контролы для взаимоисключающих режимов;
- чекбоксы/переключатели для бинарных настроек;
- степперы или числовые поля для диапазонов backtest-задач;
- ошибки валидации остаются рядом с полем и дополнительно суммируются сверху для длинных форм.

Branded dropdown acceptance:

- ARIA pattern: `menu`, `listbox` или `combobox` по реальному поведению;
- клавиатура: `Tab`, `Esc`, стрелки, `Enter`/`Space`, typeahead для длинных списков;
- popover не обрезается ближайшей panel overflow-зоной и не выходит за viewport;
- мобильная версия использует тот же branded surface или controlled drawer, а не OS-native picker как основной видимый UX;
- Playwright evidence для backtests/settings/strategies обязан включать хотя бы один открытый custom dropdown/menu, если stage добавляет такие controls.

#### Переключатель темы

Переключатель темы является полноценным элементом управления:

- доступен из каркаса приложения через компактный account/settings control, но не занимает primary header-nav row;
- полные настройки доступны на странице настроек;
- тип контрола: сегментированный переключатель или меню с превью-свотчами;
- на страницах приложения предпочтительно меню/compact segmented control с swatches, а не крупные текстовые кнопки в шапке;
- превью-свотчи используют цвета `base`/`accent`, а не финансовые цвета;
- применение темы сразу обновляет `data-theme`;
- предпочтение сначала сохраняется в `localStorage`, затем в backend-настройку, когда API настроек аккаунта доступен.

#### Переключатель языка

Переключатель языка является обязательным shell control:

- поддерживает только `EN` и `RU` на v1;
- сохраняет выбор в locale cookie для SSR и в `localStorage` для client-side helpers;
- после появления backend-настроек синхронизирует `locale` в account preferences;
- не переводит URL и не меняет browser-visible `/api/*` contract;
- имеет доступное имя `Language` / `Язык`;
- в Playwright evidence проверяется хотя бы одна смена языка и корректный `<html lang>`.

#### Refresh и autorefresh

Live-data страницы (`/dashboard`, `/strategies`, `/backtests`) имеют явный ручной refresh и управляемый autorefresh. Refresh не является декоративной кнопкой: это контракт между UI, backend read-models и внешними exchange/source limits.

Правила UI:

- каждая live-data workstation имеет заметную, но компактную кнопку `Refresh` / icon control в command bar или panel toolbar;
- рядом с refresh доступен branded dropdown/segmented control для autorefresh: `Off`, `10s`, `15s`, `30s`, `1m`, `5m`, плюс custom seconds/minutes там, где stage это реализует;
- custom interval валидируется на client и server; значение ниже минимального безопасного интервала отклоняется или округляется по documented policy;
- один browser tab не должен запускать overlapping refresh requests: новый refresh disabled/queued, пока текущий не завершен или не abort-нут;
- hidden tab снижает частоту или ставит autorefresh на pause по shared `poller.js`;
- UI всегда показывает `updated_at`, `source_freshness`, `stale/degraded` state или equivalent status, чтобы пользователь понимал возраст данных.

Правила данных:

- browser никогда не обращается напрямую к биржам;
- manual refresh и autorefresh вызывают backend `/api/ui/*`/stream contracts, которые читают bounded read-model/cache и только при необходимости инициируют backend-controlled upstream refresh;
- если backend не может обратиться к бирже из-за rate limit, source degradation или отсутствия ключа, UI получает typed `refresh_status`/`degraded` state с `retry_after_seconds`, а не бесконечный spinner;
- финансовые значения обновляются только из trusted backend DTO/read-models, не вычисляются из случайных строк DOM.

#### Графики

Графики должны раскрывать данные, а не служить декором:

- темная область построения;
- тонкая сетка только внутри plot area;
- оранжево-желтая основная серия в теме по умолчанию; акцент темы в других палитрах там, где это уместно;
- маркеры покупки/положительного финансового события используют `--rh-financial-positive`;
- маркеры продажи/убытка/drawdown используют `--rh-financial-negative`;
- подписи и значения осей в приглушенном моноширинном стиле;
- серверное прореживание рядов для длинных диапазонов;
- стабильные размеры canvas/SVG с responsive-ограничениями.

### 9) Дизайн-заметки по страницам

#### Лендинг

Route: `/`. Канонический референс: `general_page.png`.

Текущий baseline лендинга - компактный terminal/CLI preview в центре страницы с CTA `Login` и `/register`, без декоративной full-page grid и без generic SaaS cards. `general_page.png` остается route reference, но landing не переоткрывается в текущем Stage 1-5 checkpoint. Если Stage 3 будет выполняться заново, prompt должен явно решить, сохраняется ли текущий CLI preview или возвращается требование продуктовой карты/диаграммы.

#### Dashboard всех стратегий

Route: `/dashboard`. Канонический референс: `personal_dashboard.png`.

Текущий Stage 04 baseline - плотная рабочая поверхность по всем стратегиям, а не generic overview cards. Реализованные зоны: selected-strategy summary panel, health/risk panel, PnL/equity chart с range controls `1H/4H/1D/1W/1M`, metric grid, таблица открытых позиций, таблица последних исполнений, alerts/events, symbol allocation, правый список стратегий с tabs/search/filter/sort controls и нижняя status/refresh/autorefresh строка. Отдельная full-width command bar и action buttons stop/restart/settings в верхней summary panel не являются обязательными для текущего принятого baseline; если их нужно добавить для буквального совпадения с `personal_dashboard.png`, Stage 04 должен быть переоткрыт отдельной задачей.

Dashboard показывает текущее состояние портфеля стратегий online, но читает его только через backend read-models/cache: strategy storage/run state, realtime Redis streams/readers, planned portfolio/position/execution/equity snapshots, market-data reference и exchange account read-models. Если источник еще не реализован, соответствующая panel остается в форме референса и показывает typed `unavailable/degraded/stale` state.

#### Настройки

Route: `/settings`. Канонический референс: `personal_settings.png`.

Страница является плотной операционной поверхностью аккаунта. Текущий Stage 05 baseline - двухколоночная panel grid: профиль, подключенные API-ключи бирж, лимиты, integrations/webhooks, уведомления, безопасность, недавние сессии и журнал событий. Верхние actions живут внутри соответствующих panel headers/forms, а глобальная нижняя status/refresh строка наследуется из shell. Отдельная full-width settings command bar сейчас не является обязательной. Секреты никогда не показываются, а замаскированные API-ключи должны выглядеть как операционные записи, а не как password-поля.

Settings является местом persistent preferences для theme, locale, density и autorefresh defaults. Все меню выбора темы, языка, биржи, режима уведомлений и интервалов refresh используют branded dropdown/listbox controls.

#### Dashboard конкретной стратегии

Route: `/strategies`. Канонический референс: `strategy_statistic.png`.

Страница является selected-strategy analytics workstation, а не обычной библиотекой карточек. Текущий Stage 6 baseline структурирует `strategy_statistic.png` в пять рабочих зон: слева сверху Statistics Workspace, слева ниже более широкий Visual Workspace с вкладками Trades / Candles, Equity, Drawdown, справа сверху Strategy Control с выбранной стратегией и lifecycle actions без create UI, справа ниже Saved Strategies как отдельная searchable зона выбора той же ширины, снизу Trades History на всю ширину. Выбор строки в Saved Strategies сразу обновляет статистику и графики; отдельная кнопка `Load statistics` запрещена. Statistics Workspace держит вкладки Overall, Long / Short, Hourly, Risk & Execution, Monthly с единым табличным стилем, где Long / Short является визуальным reference pattern; Monthly stats рендерится строками, а не широкой month-as-column таблицей. Equity и Drawdown не являются отдельными mini-panels, Symbol Results и отдельные Best/Worst/Profitable month tiles исключены из baseline до появления реального multi-symbol attribution/use case. Live/status controls и manual refresh/autorefresh сохраняются в footer/status, но не заменяют автозагрузку выбранной стратегии.

Источники данных: strategy storage/run state, Redis realtime output/readers, planned strategy position/execution/equity read-models, market-data candles/reference и exchange account snapshots. Панели обязаны показывать freshness/lag для live-состояний.

#### Backtest workstation / конфигуратор

Route: `/backtests`. Канонический референс: `stategy_backtest.png`.

Сохранить левую колонку конфигурации, центральную AI/config-зону с прогрессом/результатами, панели выбора инструмента/индикаторов, optimization overview/progress, events и таблицу вариантов/results. AI-зона может быть gated до Stage 10, но форма панели и ручной workflow должны совпадать с референсом. Реализация может перестраивать блоки ради адаптивности, но должна сохранить плотную рабочую поверхность.

Все выпадающие списки конфигуратора (`market`, `symbol`, `timeframe`, `risk mode`, `ranking metric`, `direction`, `preset`, `job/result filter`) используют branded combobox/listbox controls. `stategy_backtest.png` не допускает системный gray native dropdown как основной визуальный слой.

#### Backtest results state

Route: `/backtests/{job_id}` не является отдельной шестой страницей в Web UI v1. Если deep link сохраняется, он должен открыть `/backtests` с выбранной job/result state или отдать совместимый detail state внутри backtest workstation. Отдельный page layout под `strategy_statistic.png` не планируется, потому что этот PNG закреплен за `/strategies`.

### 10) Организация ассетов, CSS и i18n

Целевая структура:

```text
apps/web/dist/
  css/
    tokens.css
    themes.css
    base.css
    layout.css
    components.css
    pages/
      landing.css
      dashboard.css
      settings.css
      strategies.css
      backtests.css
  js/
    core/
    components/
    charts/
    pages/
  vendor/
apps/web/locales/
  en.json
  ru.json
```

Правила:

- `site.css`, `strategy_ui.js` и `backtest_ui.js` являются целями замены, а не долгосрочными файлами.
- Новый CSS должен быть разделен на tokens/themes/base/layout/components/pages.
- Во финальном v1-каркасе не должно быть внешних CDN-скриптов.
- Версионированные ассеты должны поддерживаться до production-выката.
- Пользовательские строки хранятся в locale catalogs или через общий i18n-helper, а не дублируются hardcoded в каждом шаблоне/JS page module.
- Locale catalogs должны иметь одинаковый набор ключей для `en` и `ru`; отсутствие ключа является gate failure для соответствующего этапа.
- JS modules используют общий `locale.js`/i18n helper для dynamic strings, empty/error states и client-side validation messages.

### 11) Доступность и runtime UX

Минимальные требования:

- видимый фокус на каждом интерактивном элементе;
- кнопки и icon-only-кнопки имеют доступные имена;
- семантические заголовки и landmarks;
- таблицы используют настоящий `<table>` там, где представлены табличные данные;
- состояния ошибки объявляются через `role="alert"`, когда это уместно;
- live-регионы ограничены и не спамят screen reader;
- клавиатурная навигация работает через вкладки шапки, фильтры, модальные окна, переключатель темы и основные формы;
- клавиатурная навигация работает через language switcher;
- обе поддерживаемые темы проходят те же layout-проверки, что и тема по умолчанию.

### 12) Приемка Browser QA

Каждый этап реализации страницы должен включать Playwright CLI evidence после запуска локального приложения. Доступный путь - глобальный Playwright CLI wrapper:

```bash
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
"$PWCLI" open http://127.0.0.1:8010/
"$PWCLI" snapshot
"$PWCLI" screenshot --filename output/playwright/<page>-desktop.png
```

Обязательные визуальные проверки для каждой страницы:

- desktop-скриншот на широком viewport, обычно около `1440x1000`;
- мобильный скриншот около `390x844`;
- для функциональных страниц: screenshot должен быть сравнен с назначенным PNG из карты выше; final report перечисляет совпадающие панели и любые intentional deviations;
- общий фон страницы не содержит orange-grid / graph-paper background;
- header соответствует референсам: текстовый бренд без `RH` monogram, compact nav, active item border, auth/account справа, без крупного theme switcher в primary row;
- language switcher видим/доступен в shell, меняет `en`/`ru`, обновляет `<html lang>` и не ломает header layout;
- desktop/mobile smoke выполняется минимум на default `en`; для страниц с новым copy дополнительно проверяется `ru` snapshot или явный locale-switch evidence;
- отсутствие горизонтального overflow документа на мобильном;
- отсутствие console errors;
- отсутствие failed same-origin network requests за пределами ожидаемых auth-redirects;
- видимое и стабильное состояние header/nav/auth;
- основной workflow страницы можно пройти без overlapping requests;
- текст не перекрывается, не обрезается некорректно и не меняет размеры controls;
- переключатель темы меняет палитру `base`/`accent`, но не меняет финансовые цвета.

Для страниц с chart/canvas добавить проверку nonblank canvas/SVG и evidence в отчет этапа.

## Контракты и инварианты

- Визуальный источник правды - этот документ и перечисленные PNG-референсы.
- Для функциональных страниц действует `reference fidelity contract`: часть после глобальной шапки должна повторять назначенный PNG по структуре и panel inventory.
- Новый UI заменяет старый; светлый наследуемый skin не сохраняется.
- Палитра по умолчанию - `terminal-orange`.
- Полноэкранная orange-grid подложка запрещена; сетка допустима только внутри графиков/data panels.
- Квадратный `RH` monogram и декоративные corner badges не являются частью v1 shell.
- Header строится по текущему Stage 1-5 shell: компактный `>_ Roehub` бренд, nav, active border, compact theme/language dropdowns, account/auth справа; command bar под header является page-specific, а не обязательным глобальным элементом.
- Login строится как branded modal/dialog; `/login` является compatibility modal state, а registration остается отдельной `/register` page.
- Выпадающие списки, меню и combobox/listbox controls выполняются фирменным Roehub UI; visible native system dropdown/select запрещен для protected functional pages.
- Live-data страницы имеют manual refresh и управляемый autorefresh с no-overlap, hidden-tab pause, freshness/degraded state и backend rate-limit awareness.
- UI мультиязычный: default `en`, вторичный `ru`, language switcher обязателен, routes/API/technical identifiers не локализуются.
- Переключение тем является обязательной продуктовой функцией.
- Same-origin browser contract остается `/api/*`.
- Web остается stateless SSR/fragment-фасадом; backend владеет JSON/SSE и доменной валидацией.
- Не раскрывать секреты в HTML, JS state, логах или скриншотах.
- Не загружать полные сделки или сырые многолетние массивы графиков на первом рендере страницы.
- Финансовые цвета для доходности и процентных изменений являются семантическими инвариантами и не меняются темами.
- Locale preference является browser-visible/config/account-preference контрактом; fallback всегда `en`.
- Baseline rollback target для пересборки Web UI pack: `bae8bd88229ceec4736deee5d61ad178e1ab9060`. Откат выполняется только явным revert/reset workflow по отдельному запросу; дизайн-манифест сам не разрешает destructive git operation.

## Связанные файлы

- `apps/web/main/app.py` - composition root web-маршрутов.
- `apps/web/templates/base.html` - текущий shell baseline: `>_ Roehub`, nav, preferences, login modal hook, bottom status/refresh bar.
- `apps/web/templates/pages/landing.html` - текущий `/` baseline.
- `apps/web/templates/pages/dashboard.html` и `apps/web/dist/js/pages/dashboard.js` - текущий `/dashboard` baseline.
- `apps/web/templates/pages/settings.html`, `apps/web/templates/fragments/account/**` и `apps/web/dist/js/pages/settings.js` - текущий `/settings` baseline.
- `apps/web/templates/pages/placeholder.html` - временный state для еще не реализованных canonical pages.
- `apps/web/dist/css/tokens.css`, `themes.css`, `base.css`, `layout.css`, `components.css`, `shell.css`, `pages/*.css` - текущий CSS source.
- legacy `strategy_ui.js`, `backtest_ui.js` и старый `site.css`, если появляются в ветке, считаются целями удаления/замены, а не долгосрочными файлами.
- `docs/web-ui+backend-plan-deep-research.md` - исследовательский ввод для архитектуры и карты endpoint-ов.
- `docs/architecture/apps/web/web-ui-backend-implementation-plan-v1.md` - план параллельной реализации.

## Как проверить

Для изменений дизайн-документа:

```bash
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Для каждого этапа реализации:

```bash
uv run pytest -q tests/unit/apps/web
uv run pytest -q tests/unit/apps/api
uv run pyright
uv run ruff check .
```

Добавлять Playwright CLI evidence, как описано в разделе `Приемка Browser QA`.

## Риски и открытые вопросы

- Риск: в макетах встречается `QUANT CLI`, тогда как продукт и репозиторий используют `Roehub`. Решение в этом манифесте: оставить `Roehub`; макеты использовать только для стиля и компоновки.
- Риск: переключение темы может случайно перекрасить финансовые значения. Митигация: отделить `financial`-токены от `base`/`accent`/`state`-токенов и проверять это через Playwright.
- Риск: агенты могут добавить русские hardcoded строки в page templates. Митигация: общий i18n-helper, parity check ключей `en`/`ru`, browser evidence для language switch.
- Риск: требования к иконкам могут привести к дублированию ad hoc SVG. Митигация: либо добавить явный self-hosted путь доставки Lucide на базовом этапе, либо использовать текстовые controls до появления такого пути.
- Риск: `strategy_statistic.png` может восприниматься как backtest-result reference по старому плану. Митигация: canonical map v1 закрепляет его за `/strategies`; `/backtests/{job_id}` не считается отдельной функциональной страницей.
