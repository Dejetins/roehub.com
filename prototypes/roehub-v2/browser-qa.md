# Browser QA Evidence — Roehub v2 prototype

`skill-result/v1`

- `browser_qa_readiness`: `ready`
- target: `http://localhost:4173`
- build: локальный Vite prototype, проверенный после успешного `npm run build`
- auth boundary: production protected UI предварительно исследован разрешенным smoke account; секрет не выводился и не переносился в prototype; prototype login использует только локальные mock interactions
- data safety: все mutations ограничены in-memory mock state локального prototype

## Проверенное покрытие

### Viewports

- desktop: `1440x1024`;
- пользовательский настольный viewport: `1189x611`;
- чистый настольный повторный проход: `1280x720`;
- tablet: `1024x768`;
- узкий tablet: `820x800`;
- mobile: `390x844`.

### Routes / workspaces

- Overview;
- Strategies;
- Backtests;
- Live operations;
- Model registry;
- Connections;
- Settings;
- Program map;
- Interface states;
- Login.

На desktop, tablet и mobile все перечисленные workspaces открылись с корректным главным заголовком. Для каждого route подтверждено равенство ширины клиентской и прокручиваемой области страницы; на mobile получено `390 === 390`, глобального горизонтального переполнения нет. Горизонтальная прокрутка сохранена только внутри предназначенных для неё таблиц и дорожки карты взаимодействий.

### Основные сценарии

- Login: CTA `Continue with Keycloak` не содержит локальных credential fields и переводит prototype в Overview; production contract остается redirect к Keycloak.
- Strategies: Library → Analytics → Runtime control → RL / ML; Stop меняется на Start и обновляет state.
- Backtests: Configure → смена Exchange → выбор инструмента → Preflight success → Run optimization → Queue → completed row → Results.
- Live: Pause → Resume state.
- Connections: Test connection → success toast.
- Command search: поиск Program map → navigation.
- Notification center: Backtest notification → Backtests и закрытие overlay.
- Theme system: Graphite → Abyss → Graphite; отдельно проверена Paper.
- Contrast smoke: проверенные foreground/surface пары во всех шести темах прошли порог `4.5:1`, минимум `4.74:1`.
- Shell: единственная primary navigation, её collapse/expand, command search, notifications, theme, user menu, close/reopen inspector, document tabs, mobile More sheet.
- UI states: error Retry → Workspace restored; empty, stale, restricted и disabled surfaces видимы.
- Deep link: `/map` после reload показывает активную вкладку `Program map`.

## Console / network

- ошибки и предупреждения консоли: `0` после чистого итогового прохода;
- неуспешные браузерные ресурсы по `PerformanceResourceTiming`: `0`;
- prototype не выполняет внешние API requests: данные локальные и детерминированные;
- отдельная сетевая трассировка не создавалась, потому что встроенная поверхность браузера не предоставляет журнал сетевых запросов, а prototype не имеет backend integration.

## Исправления по browser evidence

1. Login приведен к trust boundary: удалены локальные username/password fields, оставлен redirect CTA к Keycloak.
2. Mobile inspector теперь закрывается при ширине `≤820px`.
3. Активная document tab становится первой и сохраняется при прямом deep link.
4. Notification center закрывается после перехода.
5. Desktop shell, chart и tables уплотнены до пропорций визуального эталона.
6. Межблочные промежутки увеличены с `8px` до `12px`, внутренние поля модулей расширены, а оболочка отдаёт больше ширины рабочей области.
7. Пустые ячейки устранены в Strategies, Models и Settings без добавления новых сущностей или маршрутов.
8. Устранены вложенная кнопка карточки стратегии и повторяющиеся ключи строк Live positions.
9. Дублирующий global rail удалён; context navigation стала единственной основной панелью и приняла command search, notifications, theme и user menu.
10. Strategies переведена на текучую сетку `3 → 2 → 1`; при `1440x1024` и `1189x611` используется `3 + 2`, при `1024x768` и `820x800` — `2 + 2 + 1`, при `390x844` — одна колонка.
11. Свёрнутые кнопки основной панели получили явные доступные имена; клавиатурный фокус имеет видимый outline `2px`.

## Evidence artifacts

- `evidence/overview-desktop-final.jpg`;
- `evidence/overview-desktop-spacing-v2.jpg`;
- `evidence/overview-spacing-comparison.jpg`;
- `evidence/settings-spacing-comparison.jpg`;
- `evidence/models-spacing-comparison.jpg`;
- `evidence/overview-tablet.jpg`;
- `evidence/overview-mobile.jpg`;
- `evidence/overview-mobile-spacing-v2.jpg`;
- `evidence/strategies-mobile-spacing-v2.jpg`;
- `evidence/models-mobile-spacing-v2.jpg`;
- `evidence/settings-mobile-spacing-v2.jpg`;
- `evidence/backtests-mobile.jpg`;
- `evidence/settings-paper-mobile.jpg`;
- `evidence/login-mobile.jpg`;
- `evidence/program-map-mobile-final.jpg`.
- `evidence/strategies-user-reference.png`;
- `evidence/strategies-before-single-nav-1189x611.png`;
- `evidence/strategies-after-single-nav-1189x611.png`;
- `evidence/strategies-single-nav-comparison-1189x611.jpg`;
- `evidence/navigation-single-nav-focus-comparison-1189x611.jpg`;
- `evidence/strategies-single-nav-desktop-1440x1024.jpg`;
- `evidence/strategies-single-nav-mobile-390x844.png`.

## Proof boundary

- observed: local prototype build, listed workspaces, interactions, responsive layout, two theme families, console output;
- not observed: production API correctness, database, live exchange behavior, deployment, SSR integration, performance under production payloads;
- code changed during текущей итерации: `src/App.jsx`, `src/styles.css`, `src/pages.jsx`, `src/components.jsx`; точные сценарии повторно выполнены после исправлений;
- screenshots contain only deterministic mock trading data; smoke account credentials and account name are absent from the prototype evidence.

## Residual risks

- Vite reports a non-blocking bundle-size warning because Recharts and the icon library are bundled in one prototype chunk; production integration should code-split or reuse the existing SSR asset strategy.
- Production migration still requires `en`/`ru`, 200% zoom, actual API latency, Keycloak return flow and persisted preference verification.
