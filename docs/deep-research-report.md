# Техническое исследование Roehub web UI и план реализации новых интерфейсов

## Executive summary

По результатам анализа кода в репозитории на entity["company","GitHub","code hosting"] и приложенных прототипов оптимальная архитектура для Roehub на текущем этапе — **сохранить thin web UI на FastAPI SSR + Jinja2 + custom CSS + page-scoped JS islands**, а **HTMX использовать целенаправленно** для HTML-fragments, форм, модалок, фильтров и простых действий. Переход на React / Next / полноценный SPA сейчас не даёт архитектурного выигрыша, зато добавляет новый runtime- и build-слой, Node/JS toolchain, React-state complexity и риск расползания frontend-логики в проекте, который уже эффективно решает login gate, SSR delivery и same-origin API через Python web-процесс. fileciteturn63file0L1-L1 fileciteturn64file0L1-L1 fileciteturn50file0L1-L1 fileciteturn44file0L1-L1 citeturn6search1turn7search1

Главная архитектурная поправка к исходной гипотезе: **в репозитории фактический edge уже не Nginx, а Caddy**, и production-схема уже разведена как *public edge/VPS + private backend on Mac Studio*. Поэтому сейчас не стоит планировать миграцию reverse proxy ради самой миграции; нужно сохранить роли слоёв, а не менять зрелый edge-компонент без необходимости. Если у команды есть корпоративный стандарт на Nginx, его можно применить позже без смены остальной архитектуры, но **операционной пользы именно сейчас это почти не даст**. fileciteturn55file0L1-L1 fileciteturn45file0L1-L1 fileciteturn44file0L1-L1

Самый важный технический риск находится не во frontend-слое, а в backend backtest flow: **текущие backtest jobs в коде исполняются `sync_inline` внутри API-процесса**, при этом UI уже построен так, будто работает с асинхронными jobs и polling. Для будущего backtest configurator / optimization / results это несоответствие нужно устранить в первую очередь: API должен стать enqueue-only, а выполнение должно уйти в отдельный worker loop на backend-стороне, лучше всего сначала через уже существующий Postgres-backed job repository и lease-claim модель, без немедленного введения тяжёлого брокера. Это критично и для отзывчивости UI, и для стабильности backend, и особенно для режима слабого VPS, где web-слой должен оставаться почти чисто I/O-bound. fileciteturn30file0L1-L1 fileciteturn32file0L1-L1 fileciteturn62file0L1-L1

Второй большой риск — **передача слишком больших payload’ов в браузер**. В текущей реализации lazy trades endpoint отдаёт детальный payload с `trades` и `chart_overlay`, а UI уже сейчас делает пагинацию на клиенте по `25` строк, храня полный массив сделок в памяти страницы. Для экрана statistics/results из прототипа это не масштабируется: нужен серверный pagination для trades/logs/events, downsampled chart series и snapshot/DTO endpoints с заранее ограниченными объёмами данных. fileciteturn23file0L1-L1 fileciteturn33file0L1-L1 fileciteturn19file0L1-L1 fileciteturn58file0L1-L1

Итоговый вывод: **web UI Roehub должен остаться stateless SSR/BFF-слоем**. Его задача — рендерить HTML, собирать действия пользователя, запускать лёгкую интерактивность, открывать SSE/polling и показывать status/progress. Всё тяжёлое — вычисления, агрегации, бэктесты, оптимизация, AI-вспомогательные вычисления, обработка рыночных данных, генерация статистики и хранение больших результатов — должно жить в backend API, workers и storage-сервисах, а не в web-процессе. fileciteturn50file0L1-L1 fileciteturn63file0L1-L1 fileciteturn56file0L1-L1

## Анализ текущего состояния

### Что реально найдено в backend

Backend собирается как один FastAPI app, куда включаются операции `/health` и `/metrics`, identity-module, strategy API, market-data reference API, backtests API и indicators API. Это означает, что база для новых экранов уже есть: проект не стартует с нуля, а имеет рабочую domain/API основу для auth, стратегий, market reference, indicators и backtests. fileciteturn20file0L1-L1 fileciteturn21file0L1-L1 fileciteturn29file0L1-L1

Identity-слой уже достаточно зрелый: в коде есть Keycloak OIDC login/callback/logout/current-user flow, opaque session cookie, state validation, token exchange и introspection, а также fallback извлечения `sub` из access token payload. В production identity storage ожидает Postgres-backed sessions; exchange API keys в wiring модуля шифруются через AES-GCM envelope cipher с KEK из env. Это хорошая база для account/settings screen: UI не должен и не может работать с secret material в браузере, он должен отправлять write-only значения и получать только masked / metadata-friendly DTO обратно. fileciteturn25file0L1-L1 fileciteturn26file0L1-L1 fileciteturn27file0L1-L1

Strategy API уже реализует immutable CRUD и run control: `POST /strategies`, `POST /strategies/clone`, `GET /strategies`, `GET /strategies/{id}`, `POST /strategies/{id}/run`, `POST /strategies/{id}/stop`, `DELETE /strategies/{id}`. Это соответствует направлению прототипов: базовые lifecycle actions и сущности стратегии уже есть, но для monitoring screen не хватает агрегированных runtime DTO, live snapshot API и history/statistics API поверх run-time данных. fileciteturn22file0L1-L1 fileciteturn34file0L1-L1

Backtests backend тоже уже не пустой. Есть runtime defaults, preflight, creation/list/get/cancel job endpoints, top variants, variant details и lazy trades detail. При этом в wiring видно, что jobs use case строится поверх Postgres job repository и orchestration service, но при `create()` выполняет `claim_for_inline_execution()` и запускает executor прямо в API call path, а orchestration service прямо помечен как `Sync-inline v1 job executor`. Это и есть главный backend debt относительно целевого UI. fileciteturn23file0L1-L1 fileciteturn30file0L1-L1 fileciteturn32file0L1-L1 fileciteturn62file0L1-L1

Отдельно важно, что market reference и indicators registry уже существуют и хорошо подходят под backtest configurator: есть `/market-data/markets`, `/market-data/instruments`, `GET /indicators`, а также estimate/compute surface у indicators. Это позволяет не выдумывать кастомный frontend-only schema layer: инструменты, рынки и indicator metadata должны приходить из backend как source of truth. fileciteturn28file0L1-L1 fileciteturn52file0L1-L1

### Что реально найдено в apps/web

Текущий web — отдельное FastAPI SSR-приложение. Оно монтирует `/assets`, проксирует same-origin browser requests из `/api/{path}` в upstream API, а protected pages перед рендером делают server-side check через `/api/auth/current-user`; при `401` web уводит пользователя на `/login`. Это ровно тот thin-BFF паттерн, который имеет смысл сохранить. fileciteturn63file0L1-L1 fileciteturn36file0L1-L1

В apps/web уже есть SSR pages для landing/login/logout, strategies list, strategy builder, strategy details и backtests. Все они рендерятся через Jinja templates, а сложная интерактивность подключается page-specific ES modules из `/assets/strategy_ui.js` и `/assets/backtest_ui.js`. То есть текущий проект **уже работает как SSR + islands**, а не как чистый HTMX UI и не как SPA. fileciteturn39file0L1-L1 fileciteturn40file0L1-L1 fileciteturn41file0L1-L1 fileciteturn42file0L1-L1 fileciteturn43file0L1-L1 fileciteturn18file0L1-L1

Важно, что HTMX пока **подключён, но почти не материализован как архитектурная техника**. Base template загружает htmx с `unpkg`, но стратегия и backtest screens в reviewed templates решают сложное поведение через JS modules, а не через `hx-*` flow. Значит, потенциал HTMX есть, но его нужно вводить выборочно и дисциплинированно, а не переписывать под него весь UI. fileciteturn64file0L1-L1 fileciteturn42file0L1-L1 fileciteturn43file0L1-L1 fileciteturn18file0L1-L1 citeturn4search2

В текущем JS уже есть полезные паттерны, которые надо не выбрасывать, а обобщить: page bootstrap, data-attributes как contract между SSR shell и JS, polling timer, `AbortController`, `sessionStorage` для selected job, canvas-rendering overlay для trades и client-side local state. Но всё это пока сосредоточено внутри page files, без общего API client, общего poller manager и без единых ошибок/баннеров/redirect rules. fileciteturn19file0L1-L1

### Сильные стороны и технический долг

Сильные стороны текущей реализации: тонкий SSR web без лишнего frontend runtime; same-origin auth и proxy contract; отделение public edge/VPS от private backend; уже существующие domain APIs; canvas-first подход для терминальных/fintech экранов; наличие operational routes; и явное стремление держать web как HTML facade, а не как второй backend. fileciteturn50file0L1-L1 fileciteturn49file0L1-L1 fileciteturn55file0L1-L1

Ключевой технический долг: backtests sync-inline, отсутствие server-side pagination на тяжёлых result surfaces, отсутствие единого JS foundation, отсутствие строгой CSP-ready структуры из-за внешнего htmx script и inline scripts на login/logout, отсутствие asset versioning, и отсутствие готовых account/settings/monitoring SSR pages в web-слое. Это не повод менять стек; это повод довести **именно этот стек** до production discipline. fileciteturn32file0L1-L1 fileciteturn33file0L1-L1 fileciteturn64file0L1-L1 fileciteturn40file0L1-L1 fileciteturn41file0L1-L1

### Что дают приложенные прототипы

Прототипы подтверждают, что будущий продукт делится на четыре очень разные группы экранов. Landing/marketing — почти статический SSR. Account/settings — много таблиц, toggles, подключений, лимитов, сессий и event logs, то есть хороший HTMX territory. Monitoring — постоянные обновления, KPI, live list, PnL/equity chart, positions/trades/alerts/health blocks. Backtest configurator и results — самые сложные экраны: parameter grids, AI-config assistant, progress, top variants, trades overlay, drawdown/equity curves, monthly/hourly/symbol stats и большие табличные поверхности. Это очень сильный аргумент не за SPA вообще, а за **разные технологии на разных экранах**.

## Целевая архитектура и выбор frontend-подхода

Рекомендуемая целевая схема:

```text
Browser
  ↓
Caddy edge now / Nginx later if standardized
  ↓
FastAPI Web UI
  - SSR pages
  - Jinja2 templates
  - static assets
  - auth gate
  - HTML fragments for HTMX
  - same-origin /api/* pass-through
  ↓
Backend API
  - auth / account / exchange keys
  - strategies / monitoring DTOs
  - backtests / jobs / results DTOs
  - AI assistant
  ↓
Workers
  - backtest execution
  - optimization
  - async statistics materialization
  ↓
Postgres / ClickHouse / object/filesystem cache / optional Redis
```

Эта схема прямо продолжает уже существующую production topology: public edge держится на VPS, web-процесс отдает HTML и принимает browser traffic, backend живёт отдельно и приватно, а same-origin contract уже зафиксирован. Менять надо не роли слоёв, а качество их наполнения. fileciteturn55file0L1-L1 fileciteturn45file0L1-L1 fileciteturn63file0L1-L1

**Роли слоёв должны быть такими.** FastAPI Web UI: SSR, route guard, static assets, HTML fragments, ноль тяжёлых задач. Jinja2: layout, nav, first paint, empty/error/skeleton states, forms, tables, SEO/static sections. HTMX: partial table reloads, filters, tabs, modals, lightweight actions, save/update fragments. Plain JS islands: charts, polling/SSE, large dynamic forms, instrument pickers, AI chat, local validation, client ergonomics. Backend API: source of truth для всех DTO и side effects. Workers: long-running backtests/optimization/stat materialization. Database/storage: persistence и query backend. Reverse proxy: TLS, compression, cache policy, security headers. fileciteturn50file0L1-L1 citeturn4search2

### Почему не React / Next / полный SPA сейчас

**React SPA** для этого проекта сейчас даст больше минусов, чем плюсов. Прототипы действительно содержат сложные интерактивные экраны, но текущий repo уже решает login gate, same-origin auth, SSR shell и page-scoped JS without hydration/runtime duplication. У вас нет evidence в коде, что проблему сейчас создаёт именно отсутствие React; наоборот, проблема пока в отсутствии общих JS primitives и в backend sync-inline jobs. fileciteturn63file0L1-L1 fileciteturn19file0L1-L1 fileciteturn32file0L1-L1

**Next.js** технически зрелый, но он вводит отдельный Node-based application model, `next build` / `next start`, Node version floor и типичный современный frontend stack c bundler/lint/TS/Tailwind defaults уже на этапе scaffolding. Для Roehub это не решает главную проблему — jobs/results/DTO design — и увеличивает организационную стоимость platform layer. Для продукта с thin web edge и без явной отдельной frontend-команды это не рациональный обмен. citeturn6search1turn6search0

**Tailwind / Bootstrap migration** тоже не даёт главной пользы. Прототипы уже имеют выраженный terminal/CLI dark visual language. Гораздо дешевле и правильнее ввести design tokens и UI-kit на поверх текущего custom CSS, чем повторно изобретать визуальную систему через utility framework или framework CSS. Tailwind сам по себе тоже вносит отдельный scanning/build/config слой. fileciteturn64file0L1-L1 citeturn7search0

**TypeScript/Vite** стоит рассматривать не как “новый frontend”, а как optional tooling stage. Vite — это build tool с dev server и production bundling. Он полезен, но не обязателен, если цель сейчас — дёшево и быстро привести существующий стек в порядок. Поэтому правильное решение: **не вводить TS/Vite сейчас как обязательный базовый слой**, но сделать архитектуру JS такой, чтобы migration to TS-in-CI later была почти механической. citeturn7search1turn7search5

### Разделение экранов по технологиям

**Landing / marketing.** Рекомендованный стек: Jinja2 + custom CSS, максимум небольшой JS для hamburger/menu/analytics. Acceptance: page полностью читаема без JS, TTFB минимальный, SEO metadata и canonical links на месте. Для этих экранов React/SPA не нужен.

**Account / settings.** Рекомендованный стек: Jinja shell + HTMX fragments + немного JS для modal/confirm/masked inputs. Под HTMX должны уйти: exchange connections table, webhooks/integrations block, notifications matrix, sessions list, limits panel, event log, destructive actions с confirm. В JS должны остаться only enhancements: confirm modal, copy-to-clipboard, field masking, maybe debounce search. Acceptance: большая часть действий не требует full reload и не требует custom JSON rendering в клиенте.

**Strategy monitoring.** Рекомендованный стек: Jinja initial shell + JS polling/SSE + charts + HTMX buttons/fragments. Под HTMX: start/stop/restart, settings modal, switching filters/tabs, small table fragments. Под JS: live PnL snapshot, chart, side list search/sort state, positions/executions refresh, hidden-tab pause, sparkline rendering. Acceptance: один selected strategy обновляется без full reload, при потере связи UI деградирует в stale-state/banner, а не ломается.

**Backtest configuration.** Рекомендованный стек: Jinja shell + крупный page JS module + HTMX presets/fragments + SSE/polling for progress. Здесь чистый HTMX будет неудобен: instrument multi-select, indicators grid, local combination counter, AI chat, complex validation и interdependent form rules удобнее и дешевле как JS island. Acceptance: пользователь собирает config без JSON textarea, видит preflight, может сохранить draft/preset и запустить async job.

**Backtest results / strategy statistics.** Рекомендованный стек: Jinja shell + lazy-loaded chart modules + server-side pagination + partial fragments. Свечной chart/trades overlay и equity/drawdown — JS. KPI tables, filters, tabs, result sections — SSR + fragments. Acceptance: results page открывается быстро даже при больших backtests, а trades/history never arrive as “all rows into browser”.

## API, jobs и realtime

### Где прямое proxy API, а где UI-specific DTO

Прямое проксирование текущего backend API нужно оставить там, где backend contract уже хорошо отражает доменную сущность и не требует view-specific aggregation: `auth/*`, `strategies CRUD/run/stop`, `market-data/markets`, `market-data/instruments`, `indicators`, `backtests/preflight`. Эти маршруты уже достаточно чистые и не требуют дублировать business rules в web-слое. fileciteturn22file0L1-L1 fileciteturn23file0L1-L1 fileciteturn28file0L1-L1 fileciteturn52file0L1-L1

UI-specific DTO endpoints нужны там, где экрану нужен **aggregated snapshot, trimmed payload или view-specific shape**. Их лучше делать **в backend API**, а не в `apps/web`, чтобы web оставался тонким. Рекомендуемая группа: `/api/ui/account/*`, `/api/ui/monitor/*`, `/api/ui/backtests/*`, `/api/ui/limits`, `/api/ui/dashboard/*`. Web должен их только проксировать по same-origin contract. Это усиливает, а не нарушает исходную гипотезу thin BFF. fileciteturn50file0L1-L1 fileciteturn63file0L1-L1

### Рекомендуемый backend/API план

**Account / settings surface.**  
`GET /api/ui/account/overview` — профиль, тариф, timezone, locale, subscription, status badges.  
`GET /api/ui/account/exchange-connections?cursor=&page_size=` — таблица подключений бирж без секретов; response включает permission summary, environment, last sync, latency, action availability.  
`POST /api/account/exchange-connections` / `PATCH` / `DELETE` — write-only create/update/remove integration; request body содержит label, exchange, key metadata, secret payload; response secrets never echo назад.  
`GET /api/ui/account/limits` — plan, usage counters, coloured thresholds.  
`GET /api/ui/account/notifications` + `PATCH /api/account/notifications/{channel}` — matrix toggles.  
`GET /api/ui/account/sessions` и `GET /api/ui/account/audit-events` — server-side paginated session/audit lists.  
Ошибки: `401`, `403`, `409`, `422`. Performance: page size 20–50, cursor-based paging. Security: write-only secret fields, destructive actions only with CSRF + confirm.

**Monitoring surface.**  
`GET /api/ui/monitor/strategies?state=&exchange=&sort=&cursor=` — right-side strategy list screen DTO: runtime state, last activity, pnl snapshot, symbol summary, tiny sparkline/ref.  
`GET /api/ui/monitor/strategies/{id}/snapshot` — single payload for header, KPIs, open positions count, exposure, uptime, risk/health status.  
`GET /api/ui/monitor/strategies/{id}/chart?range=1h|6h|1d&points=` — equity/PnL/downsampled markers.  
`GET /api/ui/monitor/strategies/{id}/positions?page=&page_size=`.  
`GET /api/ui/monitor/strategies/{id}/executions?page=&page_size=`.  
`GET /api/ui/monitor/strategies/{id}/alerts?page=&page_size=`.  
`GET /api/ui/monitor/strategies/{id}/symbol-allocation` — grouped instrument exposure/PnL.  
`POST /api/strategies/{id}/run` и `POST /api/strategies/{id}/stop` остаются direct domain actions.  
Performance: selected strategy snapshot должен быть одним запросом, а не шестью; charts — downsampled. Security: live actions через confirm + audit log.

**Backtest / optimization surface.**  
`GET /api/backtests/runtime-defaults` и `POST /api/backtests/preflight` остаются базой configurator flow.  
`POST /api/backtests/jobs` должен измениться: вместо sync-inline completion возвращать быстрый `202 Accepted` с `job_id`, `status`, `created_at`, `links`. Сейчас в коде jobs выполняются inline, и именно это надо сломать в пользу async flow. fileciteturn23file0L1-L1 fileciteturn32file0L1-L1  
`GET /api/backtests/jobs?state=&cursor=&limit=` — оставить, но использовать как list/read model.  
`GET /api/backtests/jobs/{job_id}` — status snapshot.  
`POST /api/backtests/jobs/{job_id}/cancel` — оставить как action.  
`GET /api/backtests/jobs/{job_id}/top?cursor=&limit=` — заменить full-list expectation на paging.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/summary` — variant summary/kpis/params.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/equity?points=`.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown?points=`.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/candles?from=&to=&interval=` — downsampled OHLC for candle/trades view.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=&page_size=` — обязательно server-side paginate вместо текущего “верни весь trades массив”.  
`GET /api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats`, `symbol-stats`, `hourly-stats`, `best-worst-days`.  
Ошибки: `401`, `403`, `404`, `409`, `422`, `429`, `503`. Performance: trade pages 50/100 max; charts 500–3000 points; cached summaries.

**AI config assistant.**  
`POST /api/ai/backtest-config/chat` — принять user prompt и current draft.  
`GET /api/ai/backtest-config/conversations/{id}/events` — SSE stream chunks/tokens/status updates.  
`POST /api/ai/backtest-config/validate` — вернуть `validated|invalid` + normalized config/errors.  
`POST /api/ai/backtest-config/drafts` — save draft.  
AI **не должен** вызывать `POST /api/backtests/jobs` напрямую. Он предлагает `proposed_config`, UI валидирует через API, пользователь нажимает Apply/Save/Run отдельно. Это один из самых правильных guardrail’ов для Roehub.

### Job-based flow для backtest и optimization

Текущий код уже имеет `job_id`, job states, repository, progress model и terminal summary, но поверх sync-inline execution. Целевой flow должен быть таким: browser отправляет validated request → API создаёт job в `created/queued` → worker claim’ит job → пишет coarse-grained progress/stage → публикует terminal result + result refs → UI открывает results. Сами стадии можно фиксировать так: `created`, `queued`, `running`, `completed`, `failed`, `cancelled`. В response/status model должны быть `job_id`, `status`, `progress`, `stage`, `started_at`, `updated_at`, `estimated_remaining_sec`, `result_id`, `error`. fileciteturn33file0L1-L1 fileciteturn32file0L1-L1

**Рекомендация по реализации очереди:** не начинать с Celery/RQ/отдельного message bus, если это не требуется другими backend задачами. Для Roehub дешевле и безопаснее сделать **DB-backed worker**: API пишет `queued`, worker loop в backend-процессе или отдельном service на Mac Studio периодически claim’ит jobs через Postgres repository/lease. Это минимальное forward-only изменение, потому что соответствующие repository patterns уже существуют в коде. Redis можно подключить позже только если появится реальная потребность в higher-throughput orchestration. fileciteturn30file0L1-L1 fileciteturn32file0L1-L1 fileciteturn56file0L1-L1

**Retries/cancel/error handling.** Validation/domain errors не ретраятся. Infra/transient errors могут получить 1 автоматический retry на worker-стороне. Cancel должен быть cooperative: между крупными execution stages worker проверяет cancel flag и переводит job в `cancelled`. Если worker потерял lease, job не должен silently finish outside repository. Все destructive/live job actions обязаны попадать в audit trail.

### SSE и controlled polling

Для Roehub **SSE лучше WebSocket почти во всех нужных местах**, потому что у вас в основном односторонний server-to-browser поток: job progress, AI response stream, alerts/system events. MDN прямо описывает SSE как one-way server push, а WebSocket — как bidirectional API; при этом MDN отдельно отмечает, что у `WebSocket` API нет backpressure handling. Для Roehub это означает: если вам не нужен постоянный bidirectional messaging protocol, SSE проще, дешевле и безопаснее для архитектуры. citeturn3search4turn3search6turn3search2

Рекомендованная стратегия:  
SSE — `backtests/jobs/{id}/events`, AI assistant streaming, critical alerts/event feed.  
Controlled polling — strategy list, selected strategy snapshot, account limits, non-critical counters, fallback если SSE недоступен.  
Интервалы: selected monitoring snapshot 2–3s; strategy list 10s; jobs list 5s пока есть active jobs; limits/sessions 30–60s или manual refresh; hidden tab — pause или degrade до 30–60s. Backoff: 1x → 2x → 4x до потолка 30s. Каждый pollable resource должен иметь `AbortController`, защиту от overlapping requests и единый poll manager. В текущем `backtest_ui.js` уже есть и `POLL_INTERVAL_MS = 1500`, и `AbortController` — это хороший прототип foundation, но теперь его надо вынести в core. fileciteturn19file0L1-L1

## Frontend и UI system

### Архитектура JS

Рекомендуемая структура фронтенда на уровне static assets:

```text
static/js/
  core/
    api.js
    poller.js
    sse.js
    dom.js
    events.js
    formatters.js
    notifications.js
    validators.js
    auth.js
  components/
    table.js
    tabs.js
    modal.js
    dropdown.js
    badges.js
    progress.js
    confirm.js
  charts/
    sparkline.js
    equity_chart.js
    drawdown_chart.js
    candle_trades_chart.js
  pages/
    dashboard.js
    account_settings.js
    strategy_monitoring.js
    backtest_run.js
    backtest_result.js
    ai_backtest_assistant.js
```

`api.js` должен стать единым JS API client для всех страниц. Его ответственность: `credentials: 'include'`, единое чтение JSON/text, обработка `401` с redirect на login, `403`, `404`, `409`, `422`, `429`, `5xx`, timeout, `AbortError`, parse errors,统一 error banner/toast, и передача CSRF header для state-changing requests. Сейчас логика разбросана по page modules; в целевой архитектуре страницы не должны самостоятельно изобретать networking policy. fileciteturn19file0L1-L1

`poller.js` и `sse.js` должны быть отдельным foundation layer. Требования: start/stop on page mount/unmount, pause on `visibilitychange`, exponential backoff, jitter, stale-response guard, no overlap, request cancellation, hooks `onData/onError/onStateChange`, metrics/debug logging. Для monitoring и backtests это must-have, иначе код быстро снова превратится в набор ad-hoc timers.

### Jinja templates, macros и UI-kit

Предлагаемая структура templates:

```text
apps/web/templates/
  base.html
  pages/
    landing.html
    dashboard.html
    settings.html
    strategies.html
    strategy_monitoring.html
    backtests_run.html
    backtests_result.html
  components/
    panel.html
    metric_card.html
    data_table.html
    status_bar.html
    empty_state.html
    error_state.html
    confirm_modal.html
  fragments/
    account_connections_table.html
    account_limits.html
    account_sessions.html
    monitor_strategy_list.html
    monitor_positions_table.html
    monitor_executions_table.html
    event_log_rows.html
  macros/
    ui.html
```

Jinja macros нужны обязательно, потому что в прототипах повторяются одни и те же визуальные primitives: panel, panel header, badge, KPI card, data table, status row, inline progress, button/danger button, empty/error blocks. Их нужно вынести раньше, чем начнётся массовая разработка новых экранов; иначе вы закрепите рассыпанный HTML как новый technical debt. Это особенно важно, потому что текущие prototypes строятся не вокруг “уникальных страниц”, а вокруг повторяющегося terminal dashboard language.

UI-kit без React я рекомендую разделить так. **Jinja macros:** `Panel`, `PanelHeader`, `MetricCard`, `StatusBadge`, `RiskBadge`, `ActionButton`, `DangerButton`, `InlineProgress`, `EmptyState`, `ErrorState`, `ConfirmModal shell`, `StatusRow`, `Pagination controls`. **CSS classes:** grid/layout, table chrome, button variants, badge variants, form controls, tabs, progress bars, terminal accents. **JS helpers:** modal open/close, confirm, toast, tabs state, sortable headers, copy/mask/reveal. **HTMX fragments:** таблицы подключений, лимиты, event log, sessions, отдельные status blocks.

### CSS-архитектура и design tokens

Custom CSS стоит **оставить**, но перестроить из “одного growing файла” в source-структуру с tokens/layout/components/pages, которую при желании можно собирать обратно в один `site.css` на CI. Сейчас base template отдает один `/assets/site.css`, и это нормально как delivery-артефакт; менять нужно не runtime contract, а способ сопровождения стилевого слоя. fileciteturn64file0L1-L1

Практический вариант:

```text
static/css/
  tokens.css
  layout.css
  components.css
  pages/
    landing.css
    settings.css
    monitoring.css
    backtests.css
  site.css   # итоговый bundle/concat
```

Базовые токены рекомендую зафиксировать прежде, чем делать UI-kit:  
цвета — `--bg-0`, `--bg-1`, `--panel`, `--line`, `--text`, `--text-muted`, `--accent`, `--success`, `--warning`, `--danger`;  
spacing — `--space-1..6`;  
radius — `--radius-1..3`;  
typography — `--font-mono`, `--font-ui`, `--fs-12..24`;  
панели/таблицы — `--panel-border`, `--panel-shadow`, `--table-row-hover`, `--table-cell-padding`;  
responsive — `--content-max`, `--sidebar-width`, `--grid-gap`.

Визуально прототипы явно просят сохранить **terminal/CLI/dark** язык. Значит, не нужно вытеснять монопространственную типографику utility-framework’ом; наоборот, надо сделать тему формальной и воспроизводимой.

### Стратегия графиков

Для графиков Roehub не нужен “один молоток на всё”. Нужна трёхуровневая стратегия.

**Уровень 1: sparkline / tiny inline charts.** Делать на custom canvas/SVG. Это дешевле любой зависимости и идеально для правой strategy list, mini-PnL previews и badges.

**Уровень 2: equity / drawdown / simple performance series.** Здесь есть два хороших варианта: лёгкий custom canvas для very tailored charts или uPlot для dense time-series. uPlot позиционируется как маленький и быстрый chart library для time series, lines, OHLC и bars; на его официальной странице есть явный фокус на performance/memory efficiency. Это хороший кандидат именно для drawdown/equity/history surfaces, где нужно много точек и быстрый pan/zoom. citeturn5search0

**Уровень 3: candlestick + trades overlay.** Здесь лучшая рекомендация — библиотека от entity["company","TradingView","financial charts"] Lightweight Charts, причём грузить её только на нужных страницах. Official docs подчёркивают, что library finance-oriented, high-performance и компактная; на сайте указан размер около `35 KB`, а docs отдельно показывают standalone build варианты. Для Roehub это наилучшее соответствие экрану statistics с candle/trade overlay. citeturn3search0turn3search5

**Что не брать по умолчанию.** Chart.js — годится для generic charts, но он general-purpose, а docs прямо обсуждают bundlers/tree-shaking и примеры со сборкой; это не лучший базовый выбор, если у вас цель — терминальный fintech UI без лишнего веса. D3 слишком низкоуровневый для текущего объёма продукта и даст команде лишнюю реализационную стоимость без системного выигрыша. citeturn4search0turn4search1turn4search3

## Производительность, безопасность и deployment

### Data loading и performance rules

Базовое правило: **не отдавать в браузер большие JSON “на всякий случай”**. Для Roehub надо жёстко разграничить payload classes. Snapshot endpoints — компактные и часто обновляемые. Detail endpoints — ленивые и paginated. Chart endpoints — downsampled. CSV/export — отдельный файл/stream, не инлайн в page bootstrap.

Практические лимиты я рекомендую зафиксировать так. Client-side rendering допустим до примерно **100–200 строк** для нерастущих таблиц и только если данные уже нужны странице целиком. Всё, что может стать “сотни/тысячи строк” — trades, logs, optimization results, strategy history, exchange events, audit events — должно идти через **server-side pagination** с `page_size=50` default и `100` max. Для charts: 500–1500 точек на неспециализированный line chart; 1000–3000 на dense equity/drawdown; tick-by-tick полные серии в браузер не отправлять. Текущий lazy trades screen уже показывает, насколько быстро UI может начать зависеть от client-side paging, если backend не сдерживает объём ответа. fileciteturn19file0L1-L1 fileciteturn58file0L1-L1

Чтобы сэкономить и CPU, и network, monitoring screen должен работать не через пачку “one KPI = one endpoint”, а через **snapshot endpoints**. Для selected strategy это должен быть один read-model с KPI/status/summary counts и отдельные lazy запросы на positions/executions/alerts/chart. Для backtests result page — сначала summary + chart refs, затем lazy tabs/sections.

### Auth и security

Same-origin `/api/*` proxy надо **сохранять**. Это одна из сильнейших сторон текущей архитектуры: browser не знает private backend URL, auth остаётся cookie-based, а web/API играют в одном origin contract. fileciteturn49file0L1-L1 fileciteturn63file0L1-L1

Но security posture нужно усилить. Сейчас cookie flow уже использует HttpOnly, `Secure` в prod и `SameSite=lax`, что хорошо, но `SameSite` — это defense in depth, а не полноценная замена CSRF protection. Для state-changing actions Roehub должен ввести CSRF token strategy: synchronizer token или double-submit cookie плюс обязательный header `X-CSRF-Token` для `POST/PATCH/DELETE`. Особенно это важно для destructive account actions и live trading actions. fileciteturn25file0L1-L1 fileciteturn27file0L1-L1 citeturn8search2turn8search3turn8search4

CSP сейчас в reviewed web-слое строго не встанет, потому что `base.html` тянет htmx с `unpkg`, а `login.html` и `logout.html` содержат inline scripts. Поэтому порядок действий такой: self-host htmx в `/assets/vendor/`, убрать inline scripts в отдельные JS files, затем включить строгий CSP. Практический baseline policy для Roehub после этой чистки:  
`default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'; form-action 'self'; upgrade-insecure-requests`. CSP и HSTS должны выставляться на edge. В reviewed Caddy config явных security headers я не увидел, поэтому это отдельная обязательная hardening-задача. fileciteturn64file0L1-L1 fileciteturn40file0L1-L1 fileciteturn41file0L1-L1 fileciteturn45file0L1-L1 citeturn8search0turn8search1turn8search5

Для live trading / dangerous actions я рекомендую дополнительную политику: typed confirm для high-risk operations, audit trail с user/session/ip/old-state/new-state, и явное отображение exchange permission scope в UI. Exchange API secrets никогда не должны уходить обратно в browser после сохранения; UI должен получать только masked key id, exchange, label, permission summary, last validated timestamp и status.

### Edge, static assets и runtime на слабом VPS

В production web delivery сегодня идёт через Caddy: он уже делает TLS termination, `zstd gzip`, route-based reverse proxy для `/api/*` и отдает web process на `127.0.0.1:8010`. Docker compose web deployment запускает один Python command `python -m apps.web.main.main --port 8010`, то есть сейчас web фактически работает как один uvicorn process. Для VPS `1 vCPU / 2 GB RAM` это корректная отправная точка: я **не рекомендую** на этом хосте плодить несколько web workers. fileciteturn45file0L1-L1 fileciteturn46file0L1-L1 fileciteturn48file0L1-L1

Рекомендованный runtime: один web worker/process на VPS; heavy backend jobs — только на backend side, не на VPS. Protected HTML: `Cache-Control: no-store, private`. Static assets: long-lived immutable cache, `ETag`, gzip/zstd, asset versioning. Так как у проекта сейчас нет Node pipeline, самый дешёвый путь к versioning — не hashed filenames любой ценой, а **`asset_version = git_sha/image_tag`** в шаблоне и URL вида `/assets/site.css?v=<sha>`. Если позже появится CI-build step — можно перейти к manifest/hashes, но это не prerequisite.

## Roadmap, acceptance и финальная формула

### Поэтапный roadmap

**Этап audit.** Цель — зафиксировать source of truth по текущему web/API/prototypes. Делается inventory экранов, маршрутов, DTO gaps, payload risks, security gaps и ADR draft. Acceptance: есть signed-off audit doc, список целевых экранов, gap matrix “что уже есть / чего нет”. На этом этапе нельзя менять стек или переписывать UI.

**Этап архитектурных правил.** Цель — зафиксировать контракт команды: SSR + Jinja2 + HTMX + JS islands, web stateless, no heavy compute in web, same-origin `/api/*`, UI DTOs в backend, worker-only backtests. Acceptance: ADR/architecture rules в repo, checklist для code review. Нельзя начинать новые страницы без этого.

**Этап структуры templates/static/js/static/css.** Цель — разложить текущий web по предсказуемой структуре `pages/fragments/macros/core/components/charts`. Acceptance: существующие `/strategies` и `/backtests` продолжают работать после перемещения/реорганизации без UX drift. Нельзя одновременно переписывать весь UI kit.

**Этап Jinja macros и UI-kit.** Цель — вынести `Panel`, `MetricCard`, `StatusBadge`, `DataTable`, `EmptyState`, `ErrorState`, `InlineProgress`, `ConfirmModal`. Acceptance: минимум strategies/backtests переведены на общие macros и common CSS classes. Нельзя добавлять новые одноразовые HTML patterns без macro justification.

**Этап unified JS API client.** Цель — общий `api.js` и error handling policy. Acceptance: 401 redirect, 403/422 rendering, timeouts, parse errors, aborts и notifications работают единообразно минимум на двух существующих страницах. Нельзя держать произвольный `fetch` scattered по страницам.

**Этап polling/SSE foundation.** Цель — `poller.js` и `sse.js` как shared modules. Acceptance: backtests page и хотя бы один новый demo fragment используют foundation вместо page-local таймеров. Нельзя в новых страницах создавать raw `setInterval` без manager.

**Этап account/settings.** Цель — реализовать settings прототип поверх HTMX fragments и нескольких JS helpers. Трогаются страницы/templates/fragments и backend `/ui/account/*`. Acceptance: профиль, integrations, notifications, sessions, limits и event log работают без full reload, секреты masked/write-only, destructive actions audit-ятся. Нельзя тащить сюда charting framework или React modal system.

**Этап strategy list и monitoring.** Цель — собрать monitoring screen. Backend добавляет `/ui/monitor/*`, frontend — charts, list state, fragments, polling/SSE. Acceptance: list filters/sort/search работают, selected strategy snapshot обновляется, run/stop/restart actions безопасны, stale state и reconnect handled. Нельзя начинать с WebSocket-first решения.

**Этап backtest configurator.** Цель — вынести large form в модульный JS configurator с preflight, presets/drafts, instrument picker и indicator grid. Acceptance: конфигурация собирается без JSON textarea, preflight валидирует итоговый payload, presets сохраняются/загружаются. Нельзя запускать async optimization из AI напрямую.

**Этап job-based async flow.** Цель — убрать sync-inline create path и перевести backtests на queued worker execution. Acceptance: `POST /api/backtests/jobs` возвращает быстро с `job_id`, progress обновляется по SSE/polling, cancel работает, API request thread не держится до completion. Нельзя уносить worker execution на VPS web host.

**Этап backtest results/statistics.** Цель — построить result screen из прототипа на summary endpoints, chart endpoints и paginated trades. Acceptance: страница statistics открывается быстро, trades/history не отдаются целиком, charts получают downsampled data. Нельзя оставлять full trades payload как текущий client-side pagination model.

**Этап AI-config assistant.** Цель — SSE streaming, assistant states `draft/validated/invalid/applied/saved/submitted`, явный user confirmation. Acceptance: AI предлагает конфиг, UI валидирует, пользователь применяет/сохраняет/запускает сам. Нельзя давать AI право на direct run.

**Этап performance/security hardening.** Цель — self-host scripts, CSP/HSTS/headers, CSRF, asset versioning, caching rules, response size budgets. Acceptance: security headers проверяются на edge, protected HTML no-store, static immutable, payload budgets соблюдаются. Нельзя переносить этот этап “на потом”, потому что login/logout/base template уже влияют на CSP design.

**Этап QA, documentation, rollout.** Цель — Playwright smoke, manual ops runbooks, rollback checklist, monitoring. Acceptance: smoke на landing/settings/monitoring/backtests/results проходит, deploy/rollback описаны, metrics/alerts настроены. Нельзя завершать проект без acceptance artefacts.

### Общие acceptance criteria

**Performance.** SSR shell открывается быстро; UI не грузит large JSON; charts получают только downsampled series; strategies monitoring живёт на snapshot API; trades/logs/events paginated server-side; web process на VPS не исполняет CPU-heavy jobs.

**Security.** Same-origin cookie auth сохранён; CSRF есть на state-changing routes; secrets never echo back; CSP/HSTS/security headers на edge; live actions с confirm + audit; session handling и logout flow детерминированы. fileciteturn27file0L1-L1 citeturn8search0turn8search2turn8search5

**UX.** Page shells читаемы без JS там, где это уместно; monitoring/backtest screens корректно показывают loading/empty/error/stale states; background refresh не бомбит backend в hidden tab; errors user-visible и actionable.

**Backend contracts.** Все UI DTOs версионируемы и документированы; result surfaces не завязаны на внутренние Python objects; job states конечны и однозначны; no secret leakage.

**Frontend maintainability.** Нет uncontrolled `fetch`; есть общий API client; нет page-local ad-hoc pollers; есть macros и UI primitives; no framework rewrite without explicit ADR.

**Deployment/monitoring.** Edge policy формализована; asset versioning есть; rollback возможно без ручной охоты за файлами; `/health` и `/metrics` остаются стабильны. fileciteturn29file0L1-L1 fileciteturn44file0L1-L1

### Что не делать сейчас

Сейчас не нужно внедрять React, Next.js, полный SPA, frontend monorepo, тяжёлый UI framework, массовую Tailwind/Bootstrap migration, D3 как стандарт для всех графиков, WebSocket “для всего”, client-side tables на тысячи строк, рендер огромных JSON прямо в HTML, pandas-расчёты в web UI и отдельный Node.js frontend server. Для текущего состояния Roehub это в основном увеличит поверхность сложности, а не снизит риск. fileciteturn63file0L1-L1 fileciteturn55file0L1-L1 citeturn6search1turn7search1turn7search0

## Финальная рекомендация и ограничения

Финальная техническая формула проекта должна звучать так:

**Roehub web — это thin SSR/BFF facade: Jinja2 рендерит первичный HTML, HTMX обновляет HTML-fragments и выполняет простые действия, page-scoped JS islands берут на себя charts/realtime/complex forms, а все вычисления, агрегации, jobs, AI-оркестрация и хранение больших результатов находятся в backend API и workers.**

Это решение лучше всего соответствует одновременно четырём фактам: уже существующей кодовой базе, текущему production topology, слабому VPS для edge/web UI и характеру будущих экранов из прототипов. Я **согласен с исходной гипотезой в целом**, но усиливаю её двумя обязательными корректировками:  
во-первых, не обсуждать rewrite фронтенда, пока не исправлен backend backtest execution model;  
во-вторых, не пытаться строить results/monitoring на full payload delivery — сразу проектировать snapshot DTOs, server-side pagination и downsampled chart data.

**Open questions / limitations.** Я не инспектировал буквально каждый большой CSS/JS файл построчно целиком, потому что часть connector output была очень объёмной; выводы по ним опираются на просмотр ключевых entry files, templates и observed behavior. Кроме того, точные route names для account/exchange-keys surface я рекомендую как target contract, а не утверждаю как уже существующую полную реализацию, потому что в reviewed material я подтвердил wiring exchange-keys, но не делал исчерпывающую карту всех связанных transport DTO. Наконец, прототипы не фиксируют окончательный product-policy для high-risk live actions и mobile-breakpoint coverage, поэтому эти две области нужно закрыть отдельным кратким product/security decision до hardening-фазы.