# Navigator work pages — acceptance evidence

Status: blocked goal; local implementation and verification complete within the recorded scope. External-block audit: the same missing external prerequisites were confirmed for three consecutive goal turns, with no remaining independent implementation work. Private testnet and notification-provider integration remain unverified. Candidate localhost:20120 only; no publication.

## Baseline and ownership

Existing dirty/untracked files were preserved. apps/navigator-web and tools/qa/navigator_preview.py were already untracked; edits extend their current contents. apps/platform-web is not edited. The preview uses the existing PostgreSQL/ClickHouse volumes and API on 20111. No reseeding.

Reference browser: playwright-cli session roehub-workpages, Chromium, 1476×959 at DPR 1. Fresh captures and computed styles: [reference measurements](roehub-workpages-2026-10-03/reference-measurements.json). The initial Backtests screenshot caught an incomplete read; it was recaptured after metrics and canvas became visible.

## Requirement matrix

| requirement_id | Requirement | Status | Implementation | Check / evidence | Remaining gap |
|---|---|---|---|---|---|
| SETTINGS-01 | Согласованная навигация по категориям настроек. | verified | settings-page.tsx + ui_account | 16 final page captures; direct category navigation in check-workpages.js | None within the stated proof boundary |
| SETTINGS-02 | Формы на основе реальных возможностей и действующих контрактов. | implemented | settings-page.tsx + ui_account | Real profile/preferences/security; unavailable Telegram capability explicitly shown | Configured Telegram provider/confirmed recipient; live notification settings unverified |
| SETTINGS-03 | Чтение сохраненных значений и сохранение разрешенных изменений. | verified | settings-page.tsx + ui_account | Browser profile timezone save/reload/restore; preferences locale save/reload/restore; settings-profile.png; settings-preferences-narrow-ru.png | None within this criterion |
| SETTINGS-04 | Валидация у поля, понятные ошибки, состояния dirty/saving/saved. | verified | settings-page.tsx + ui_account | check-settings.js: invalid email aria-invalid, dirty/saved, cancellation; failures.log: 503 retain/403 hide | None within the stated proof boundary |
| SETTINGS-05 | Защита несохраненных изменений при уходе со страницы. | verified | settings-page.tsx + ui_account | check-settings.js: rejected departure retains draft; Cancel restores saved form | None within the stated proof boundary |
| SETTINGS-06 | Настройки языка и отображения работают через общие механизмы. | verified | settings-page.tsx + ui_account | display-localization.log: real theme/density save, reload, restore; locale save/reload/restore | None within the stated proof boundary |
| SETTINGS-07 | Сессии и безопасность сохраняют серверную модель авторизации. | verified | settings-page.tsx + ui_account | Account API 35 passed; independent session-hash and denial review; settings-security.png; API tests | None within this criterion |
| SETTINGS-08 | Данные рынка не спрятаны в общих настройках. | verified | settings-page.tsx + ui_account | Separate navigation links and final data/settings captures; legacy settings redirects | None within the stated proof boundary |
| CONNECTIONS-01 | Список биржевых подключений и интеграций с поиском и фильтрами. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-02 | Отдельные состояния подключения, проверки и готовности к использованию. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-03 | Детали выбранного подключения, окружение и поддерживаемые возможности. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-04 | Создание и редактирование по существующим разрешенным контрактам. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-05 | Проверка соединения с понятным результатом и временем проверки. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-06 | Привязки и зависимости: где используется подключение. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-07 | Разделение публичного доступа к данным и приватного торгового доступа. | verified | connections-page.tsx + ui_account | connections-1476.png: real public source available while private service reports typed 503 | None within the stated proof boundary |
| CONNECTIONS-08 | Разрешенные отключение и удаление с проверкой зависимостей. | implemented | connections-page.tsx + ui_account | connections-client.log: intercepted lifecycle; account API 36 tests and scoped independent review | Private service/testnet account absent; real lifecycle remains unverified |
| CONNECTIONS-09 | Stored secrets никогда не возвращаются и не отображаются. | verified | connections-page.tsx + ui_account | Independent security review; connection client tests and connections-client.log; secret fields cleared, schema drops key fragments, no placeholder in URL/storage/DOM after command | None within the stated proof boundary |
| DATA_SOURCES-01 | Провайдер, сегмент рынка, возможности и состояние доступности. | verified | connections-page.tsx + market_data_workspace | connections-1476.png and API markets/catalog: configured Binance Spot, capabilities, real snapshot and successful-read time | None within the stated proof boundary |
| DATA_SOURCES-02 | Различимые недоступность, отсутствие доступа и отсутствие данных. | verified | connections-page.tsx + market_data_workspace | Real private outage 503 separated from public source; API denial/not-found tests and read snapshots | None within the stated proof boundary |
| DATA_SOURCES-03 | Переход от источника к его каталогу и заданиям загрузки. | verified | connections-page.tsx + market_data_workspace | Source links to actual catalog and ingestion; controls.log and final direct URL captures | None within the stated proof boundary |
| DATA_SOURCES-04 | Проверка и настройка только реально поддерживаемых провайдеров. | verified | connections-page.tsx + market_data_workspace | Real catalog refresh uses configured market and existing provider adapter; no unsupported provider settings | None within the stated proof boundary |
| INSTRUMENTS-01 | Каталог инструментов с поиском, фильтрами и выбором. | verified | data-page.tsx + market_data_workspace | controls.log: bounded BTCUSDT search at 1476/390; races-recovery.log: selection/back; catalog unit tests | None within the stated proof boundary |
| INSTRUMENTS-02 | Биржа/источник, рынок, символ и доступные характеристики. | verified | data-page.tsx + market_data_workspace | data-1476.png: actual Binance Spot BTCUSDT metadata and constraints | None within the stated proof boundary |
| INSTRUMENTS-03 | Выбранные или закрепленные инструменты согласно доменному контракту. | verified | data-page.tsx + market_data_workspace | Browser select BTCUSDT, reload, restore prior unselected state; Real organization selection API | None within this criterion |
| INSTRUMENTS-04 | Обновление каталога с идентичностью snapshot и временем свежести. | verified | data-page.tsx + market_data_workspace | Real catalog job + immutable snapshot; Checkpoint below | None within this criterion |
| INSTRUMENTS-05 | Детали инструмента и доступность свечных данных. | verified | data-page.tsx + market_data_workspace | data-1476.png, real canonical 5/5 coverage with independent coverage read | None within the stated proof boundary |
| INSTRUMENTS-06 | Покрытие по таймфрейму и периоду, обнаруженные пробелы. | verified | data-page.tsx + market_data_workspace | Canonical before 0/5, after 5/5, gaps disappear; data-before-ingestion.png; data-after-ingestion.png | None within this criterion |
| INSTRUMENTS-07 | Переход к загрузке данных выбранного инструмента. | verified | data-page.tsx + market_data_workspace | races-recovery.log: Download candles transfers BTCUSDT and the five-minute range into the form | None within the stated proof boundary |
| INSTRUMENTS-08 | Стабильный выбор и URL, включая прямой вход и back/forward. | verified | data-page.tsx + market_data_workspace | races-recovery.log: delayed ETH read, BTC identity retained, back/reload stable; unit late-response rejection | None within the stated proof boundary |
| INGESTION-01 | Выбор источника, инструментов, таймфреймов и временного диапазона. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | ingestion-form-390-ru.png; actual bounded BTC request and client validation | None within the stated proof boundary |
| INGESTION-02 | Предварительная валидация возможностей и допустимого объема. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | API/runner bounds tests; races-recovery.log duplicate-symbol validation; unsupported timeframe rejected | None within the stated proof boundary |
| INGESTION-03 | Запуск задания через существующий backend/runtime. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | Real bounded public REST fetch via existing scheduler; Job f956150f-52dc-4f14-8bde-0e4304bdae5a | None within this criterion |
| INGESTION-04 | Очередь, текущие задания и история. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | ingestion-1476.png; persisted org history with bounded cursor pages; real queued/cancelled/succeeded requests | None within the stated proof boundary |
| INGESTION-05 | Реальный прогресс, объем, ошибки, попытки и время обновления. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | Actual completed request shows 5 units/5 read/5 written; attempts and timestamps persisted; runner tests do not invent rows | None within the stated proof boundary |
| INGESTION-06 | Отмена и повтор только там, где они разрешены сервером. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | cancel.log: real queued request cancelled, cancel/retry disabled after reload; real PostgreSQL tests verify eligibility and attempt fence | None within the stated proof boundary |
| INGESTION-07 | После завершения фактическое покрытие обновляется. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | Coverage updated from canonical storage after job completed; data-after-ingestion.png | None within this criterion |
| INGESTION-08 | Состояние задания восстанавливается после перезагрузки страницы. | verified | ingestion-page.tsx + existing MarketDataSchedulerApp work requests | Browser reload retained same terminal server job; ingestion-completed.png | None within this criterion |
| MONITORING-01 | Группы: ядро платформы, данные, вычисления, торговля, безопасность, расширения. | verified | monitoring-page.tsx + existing OperationalHealthService | monitoring final captures; six fixed groups, only four configured services; empty groups explicitly labelled | None within the stated proof boundary |
| MONITORING-02 | Только реально обнаруженные или настроенные сервисы. | verified | monitoring-page.tsx + existing OperationalHealthService | Actual API/Web health and PostgreSQL/ClickHouse configured TCP probes; GET /api/ui/monitoring 200: four configured services | None within this criterion |
| MONITORING-03 | Состояние, свежесть наблюдения и источник сигнала. | verified | monitoring-page.tsx + existing OperationalHealthService | Actual probe detail, age/source; failures.log old timestamp becomes stale | None within the stated proof boundary |
| MONITORING-04 | Детали сервиса, доступные метрики, backlog и связанные задания. | verified | monitoring-page.tsx + existing OperationalHealthService | russian-final.log: current-org backlog 0/0/0; actual event detail and related queue link; scoped API/PG tests | None within the stated proof boundary |
| MONITORING-05 | Зависимости и влияние сбоя, если это поддерживается контрактом. | verified | monitoring-page.tsx + existing OperationalHealthService | Observer contract has no dependency graph; no inferred dependencies or failure impact are presented | None within the stated proof boundary |
| MONITORING-06 | Различимые healthy, degraded, unavailable, unknown и stale. | verified | monitoring-page.tsx + existing OperationalHealthService | Real HTTP healthy/TCP unknown; real restart history, intercepted stale/503/403 in failures.log; projection tests | None within the stated proof boundary |
| MONITORING-07 | Ограниченная история событий/ошибок без утечки секретов. | verified | monitoring-page.tsx + existing OperationalHealthService | Real bounded event table; observer state-change/redaction tests; independent review | None within the stated proof boundary |
| MONITORING-08 | Ссылки на действующие runbooks, если они существуют. | verified | monitoring-page.tsx + existing OperationalHealthService | controls.log: existing selected runbook returns HTTP200 | None within the stated proof boundary |
| MONITORING-09 | Базы данных представлены сервисами, а не новыми разделами навигации. | verified | monitoring-page.tsx + existing OperationalHealthService | PostgreSQL/ClickHouse remain monitoring objects under the existing navigation | None within the stated proof boundary |
| AUTOMATED-01 | Выполни typecheck, релевантные тесты и production build выбранного клиента. | verified | Candidate frontend, existing API/runtime and test evidence | 235/235 Navigator tests; tsc --noEmit and production Vite build pass | None within the stated proof boundary |
| AUTOMATED-02 | Команды выбирай из актуального package.json и доступного runtime. | verified | Candidate frontend, existing API/runtime and test evidence | Commands selected from Navigator package scripts and current CI; no dependency/runtime installation | None within the stated proof boundary |
| AUTOMATED-03 | Для backend используй применимые проектные quality gates. | verified | Candidate frontend, existing API/runtime and test evidence | Whole-repository Ruff and Pyright plus focused backend/real PostgreSQL gates recorded below | None within the stated proof boundary |
| AUTOMATED-04 | Проверь маршрутизацию, валидацию, ошибки, права и гонки выбора. | verified | Candidate frontend, existing API/runtime and test evidence | API origin/auth/scope/validation tests; frontend denial, recovery and rapid-selection tests; real browser error/race proofs | None within the stated proof boundary |
| AUTOMATED-05 | Проверь ограниченность загрузки и отсутствие повторных команд. | verified | Candidate frontend, existing API/runtime and test evidence | PostgreSQL concurrent admission/claim/idempotency; bounded 50/30/20-row pages and cache caps; client double-submit test, browser recovery no replay | None within the stated proof boundary |
| AUTOMATED-06 | Не добавляй бессодержательные тесты, повторяющие реализацию. | verified | Candidate frontend, existing API/runtime and test evidence | Focused tests exercise outcomes: isolation, cancellation ownership, retained identity, leak boundaries, actual shared consumers | None within the stated proof boundary |
| BROWSER-01 | Проверь каждую страницу в реальном браузере на localhost:20120. | verified | Candidate frontend, existing API/runtime and test evidence | check-workpages.js: all eight final pages at localhost:20120 | None within the stated proof boundary |
| BROWSER-02 | Проверь desktop и узкую компоновку. | verified | Candidate frontend, existing API/runtime and test evidence | 16 final captures: 1476x959 and 390x959, document scrollWidth equals viewport width | None within the stated proof boundary |
| BROWSER-03 | Сравни однотипные элементы с тремя эталонными страницами. | verified | Candidate frontend, existing API/runtime and test evidence | fidelity.json and fidelity-differences.json: measured shared components; functional differences documented below | None within the stated proof boundary |
| BROWSER-04 | Проверь поиск, фильтры, reset, меню, вкладки, формы и раскрытие. | verified | Candidate frontend, existing API/runtime and test evidence | controls.log, regression.log, connections-client.log and settings checks: filters/reset, tabs, forms, menus, expansions | None within the stated proof boundary |
| BROWSER-05 | Проверь клавиатуру, Escape, focus и длинные локализованные подписи. | verified | Candidate frontend, existing API/runtime and test evidence | controls.log keyboard/Escape/focus; display-localization.log and russian-final.log: long Russian labels, reduced motion, bounded popup | None within the stated proof boundary |
| BROWSER-06 | Проверь initial loading, refresh, empty, error, stale и access denied. | verified | Candidate frontend, existing API/runtime and test evidence | failures.log and races-recovery.log: delay/initial loading, retained refresh, empty, 503, stale, 403 and 404; intercepted cases are client proof only | None within the stated proof boundary |
| BROWSER-07 | Проверь rapid selection, reload и back/forward. | verified | Candidate frontend, existing API/runtime and test evidence | races-recovery.log and recovery-reload.log; actual persisted selection/job reload; reference and monitoring back/forward | None within the stated proof boundary |
| BROWSER-08 | Проверь отсутствие новых console errors и утечек чувствительных данных. | verified | Candidate frontend, existing API/runtime and test evidence | Final captures have zero pageerror; expected missing-provider HTTP503 only; no credential screenshots or raw provider payloads | None within the stated proof boundary |
| END_TO_END-01 | Изменить безопасную настройку, перезагрузить и увидеть сохраненное значение. | verified | Existing production reader/API | Real profile/preferences mutation + reload + restoration; Settings browser checkpoint | None within this criterion |
| END_TO_END-02 | Проверить lifecycle подключения на изолированном тестовом окружении. | blocked | Existing exchange-control lifecycle; new client | Private service unavailable in current preview; authenticated HTTP503 rechecked on three consecutive goal turns | Existing authorized testnet account/service requested; no credentials should be sent in chat |
| END_TO_END-03 | Обновить каталог и выбрать инструмент. | verified | Existing production reader/API | Real catalog refresh + selection persisted/reverted; Checkpoint below | None within this criterion |
| END_TO_END-04 | Запустить ограниченную тестовую загрузку свечей. | verified | Existing production reader/API | Actual Binance BTCUSDT 5-minute request; Checkpoint below | None within this criterion |
| END_TO_END-05 | Увидеть серверный исход задания и фактическое покрытие. | verified | Existing production reader/API | Job succeeded and canonical coverage 100%; Checkpoint below | None within this criterion |
| END_TO_END-06 | Подтвердить чтение загруженных данных существующим потребителем. | verified | Existing production reader/API | Existing organization-scoped canonical candle consumer: five rows; read-candle-consumer.py | None within this criterion |
| END_TO_END-07 | Подтвердить реальные состояния мониторинга и честную обработку недоступности. | verified | Candidate frontend, existing API/runtime and test evidence | Actual API/Web probes healthy; PostgreSQL/ClickHouse TCP-only unknown; restart history unavailable-to-healthy; stale/error handling explicitly intercepted | None within the stated proof boundary |
| REGRESSION-01 | Backtests, Strategies и Overview продолжают открываться. | verified | Candidate frontend, existing API/runtime and test evidence | regression.log and six final reference screenshots | None within the stated proof boundary |
| REGRESSION-02 | Выбор объектов, вкладки, фильтры, графики и таблицы не регрессировали. | verified | Candidate frontend, existing API/runtime and test evidence | regression.log: all three references selection/chart/table/filter workflows | None within the stated proof boundary |
| REGRESSION-03 | Drawdown остается последней вкладкой. | verified | Candidate frontend, existing API/runtime and test evidence | regression.log asserts Drawdown is last in all three chart selectors | None within the stated proof boundary |
| REGRESSION-04 | Общие размеры и геометрия не изменились непреднамеренно. | verified | Candidate frontend, existing API/runtime and test evidence | fidelity.json; common CSS/type/control/table values match; reference styles not redesigned | None within the stated proof boundary |
| REGRESSION-05 | Существующие данные и работающие сервисы сохранены. | verified | Candidate frontend, existing API/runtime and test evidence | Existing user volumes preserved; no reset/reseed; only bounded proof jobs, disposable schemas and reversible account preference changes | None within the stated proof boundary |

## Observed gaps and implementation notes

- Existing preview API omitted account/reference modules; production composition modules are now wired without replacing auth or databases.
- Existing profile fallback fabricated name/email/contact; removed. Default unset theme aligned to Graphite; saved preferences are retained.
- Existing sessions response exposed bearer session UUIDs. Response now uses a SHA-256 domain-separated view identifier. No session mutation/revocation protocol exists on this route.
- Existing PostgreSQL empty session pages fabricated a current local session; removed.
- New settings client uses existing QueryClient, useReadSnapshot, ReadStatus, NavigatorTable, locale route and server command validation.
- New routes are enabled only by tools/qa/navigator_preview.py; accepted platform-web remains unchanged.

## Validation — final local boundary

- `pnpm --dir apps/navigator-web exec vitest run`: 235 passed, 0 failed.
- `pnpm --dir apps/navigator-web exec tsc --noEmit`: passed.
- `pnpm --dir apps/navigator-web exec vite build`: passed. Existing bundle warning:
  1,426.74 kB JS / 449.43 kB gzip; no performance claim or bundle-size fix.
- `.venv/bin/ruff check .`: passed. `.venv/bin/pyright`: 1,515 files, 0 errors, 0 warnings. Focused API/runner/strategy/observer/storage suites: 67 passed, then account
  rerun 36 passed after adding the notification unavailable/scope test. Scheduler
  compatibility suites: 16 passed. PostgreSQL: 3 passed in isolated schemas.
- Existing frontend failures were investigated: the library accepted cached data
  as a successful snapshot during a read error; it now retains/labels it correctly.
  Other expectations were aligned with existing limit=10 routing, current freshness
  wording, asynchronous URL effects and the actual chart series instead of a removed
  caption. No backend result data was changed to satisfy tests.
- Whole Python typing also found nullable XML/cache test values in two existing
  Backtests tests. Only explicit non-None assertions/local variables were added to
  those preexisting foreign test changes; the foreign implementation and assertions
  were preserved. Their focused tests: 4 passed.
- Independent security review: no open findings. Final notification error-handler
  follow-up confirms `503` unavailable, `403` scope mismatch and no raw resolver
  error text. This does not establish real private-provider access.

## Market Data integration decision — 2026-10-03

Status: implemented within the goal's delegated authority; real runtime proof
is recorded below. Scope is a bounded user command path, not a second ingestion engine.
Observed source: `MarketDataSchedulerApp` already composes `RestFillRange1mUseCase`
and public exchange adapters; `isolated-job-runtime-v1.md` forbids network access
inside compute jobs. The existing API catalog lacks stable snapshots and cursors.

The API validates authenticated organization-scoped requests and persists a
Market Data inbox. The existing scheduler claims those requests and invokes the
existing REST-fill/raw-writer/canonical pipeline. API and Web gain no exchange
client or egress. The alternative of executing the fetch in HTTP handlers loses
reload/recovery/cancellation; using the generic isolated runtime crosses its
network boundary. Existing candle persistence remains the sole candle truth.

An immutable PostgreSQL metadata projection published after successful reference
writes gives the catalog a stable snapshot ID, timestamp and bounded cursor pages.
It contains only normalized public metadata; organization selection/pins remain
in their existing repositories. ClickHouse ReplacingMergeTree cannot provide
historical snapshot stability from an `updated_at` cutoff alone.

Initial limits (superseded for downloads by the full-history amendment below):
1m source candles, at most 8 symbols and 10,080 instrument-minutes per command, one unfinished request per organization, one
60-minute work window at a time, five explicit attempts maximum. PostgreSQL
uniqueness enforces idempotency and active admission. Unknown submission is
reconciled by key; no automatic command replay. Worker tokens fence state updates;
a lost worker becomes an explicit failure. Cancellation is acknowledged at a
window boundary. Progress is completed work windows, while candle coverage is an
independent canonical read. Provider payloads and exception text are never saved.

Compatibility: new endpoints/projections/schema and opt-in runtime configuration
are `compatible-change`; existing catalog and candle consumers remain supported.
Migration 0023 is additive and must precede enabling
`ROEHUB_MARKET_DATA_WORK_REQUESTS_ENABLED=1`. Rollback disables consumption and
API commands while preserving requests, snapshots and existing candle rows.
`--requests-only` is a bounded local verification mode of the existing scheduler;
it omits automatic startup scans and periodic backfills. No trading, real-key
rotation, database reset or deployment is included.

Required proof: real PostgreSQL concurrent admission/claim/idempotency, cross-org
and origin denial, cancellation/retry/reload, real public catalog and small candle
fetch, canonical coverage and existing Backtests reader, desktop/narrow browser,
independent security review. Successful checks are recorded below; unverified criteria remain open in the matrix.

## Verified integration checkpoint — 2026-10-03

- Real public Binance Spot catalog request `b85aa8f9-1a63-4b09-875a-ca285dc2bf4f`
  completed through the existing scheduler. Snapshot
  `c105b3fd-0c20-435f-bd76-59de1310b826` contains 1,372 instruments.
- Browser submitted BTCUSDT/1m, `[2024-01-01T00:00Z, 2024-01-01T00:05Z)`.
  Request `f956150f-52dc-4f14-8bde-0e4304bdae5a` succeeded: 5 read, 5 written,
  5 completed instrument-minutes. Reload restored the same terminal request.
- Canonical coverage changed from 0/5 to 5/5 (100%), with no remaining gaps.
  Screenshots: `data-before-ingestion.png`, `ingestion-completed.png`,
  `data-after-ingestion.png` in `roehub-workpages-2026-10-03/`.
- `read-candle-consumer.py` read all five minutes using existing
  `OrganizationScopedCanonicalCandleReader` + `ClickHouseCanonicalCandleReader FINAL`.
  Timestamps are strictly one minute apart and OHLCV values finite. This proves the
  existing Backtests offline/precompute consumer, not artifact publication or a
  completed strategy/backtest execution. No historical data was reset.
- Real PostgreSQL tests: 3 passed in a disposable isolated schema, including
  concurrent idempotency/claim, cancellation, retry, membership revocation, a slow
  consumer excluded from takeover, and loss of the lock session.
- Scoped independent review: no open actionable findings after fixes. Reviewer ran
  six suites (63 passed), reran account API (35 passed), reproduced the snapshot
  expiry race now returning 409 and exchange-control outage now returning 503.
  Legacy SSR compatibility retained: absent limit preserves existing response;
  explicit bounded pagination returns 20+5 for the 25-row review fixture.
- Snapshot projections retain only the latest three per market. Slow live workers
  retain a PostgreSQL session advisory lock. Every subsequent write checks that
  exact session. An already-sent ClickHouse insert can still finish after connection
  loss; existing canonical minute deduplication remains the read boundary.

Local UI, error/race/responsive and reference regression proof is recorded below.
Private connection lifecycle, real Telegram provider/recipient and hardware passkey
verification remain unverified. No completed-goal, publication or production claim.

## Final browser and geometry evidence

Browser: disposable `playwright-cli` session `roehub-workpages`, authenticated local
Navigator, Chromium DPR 1. Desktop 1476×959 and narrow 390×959. Final English captures
are `profile`, `preferences`, `notifications`, `security`, `connections`, `data`,
`ingestion`, `monitoring` with `-1476.png` / `-390.png` suffixes in
`roehub-workpages-2026-10-03/`. Russian narrow captures cover filters, the ingestion
form and the actual current-organization backlog. The three reference regression
captures use `regression-{backtests,strategies,dashboard}-{1476,390}.png`.

`fidelity.json` compares the same table/tab/filter elements at both viewports and
normal/hover/keyboard-focus/open states. Shared rail, library/header axes, table
chrome, borders, typography, icons and matching control styles agree. Desktop rail
56px, library 224px, library header 48px, object heading 40px, icon controls 28px,
icons 16px, table toolbar 50px, table headings 32px and ordinary rows 44px. The narrow
rail is 48px. `form-controls.json` verifies that native form inputs/selects match
the measured Backtests form exactly: 34px height, 12.6px type, 10px radius, shared
padding/borders/colors; form command buttons are 33.59375px. Compact period controls
remain 28px. These rules extend the existing CSS selectors rather than duplicate
them per page. Geometry differences greater than 1 CSS px are functional:

- Backtests reserves 130px on the right for its job-state overlay; generic headings
  reserve the shared 12px inset. Heading height/type/left axis match.
- The narrow four-category settings navigation is 142px high; a long object library
  uses the 210px cap. Neither gains empty filler rows.
- Four long Backtests table tabs wrap to 89px on the narrow screen; the one/two-tab
  work-page toolbars remain 50px with identical control CSS.
- Wrapped Backtests row contents can reach 46px on narrow screens; short work-page
  cells remain 44px. Shared padding, fonts and borders match.
- Short form captions use one line; Backtests reserves two lines for its denser
  configuration labels. Caption font, line-height, inset and colors are identical.
- Form/metadata/table compositions fill their actual content; no empty charts or
  copy of the Backtests inspector is introduced.

`regression.log` verifies existing object selection/back, chart modes, Drawdown
last, table tabs, filters/reset, download menu, expansion/Escape/focus and no whole-
page horizontal overflow. Overview remains explicitly labelled synthetic demo data.
`controls.log` verifies shared filter keyboard/focus, table expansion, runbook HTTP200
and real bounded catalog search. `failures.log`, `races-recovery.log`,
`recovery-reload.log` and `connections-client.log` are explicitly **intercepted client
proof**, not provider verification. The recovery test aborted its only POST before
server dispatch; reload stayed locked and explicit lookup reconciled without replay.
The connection script created/validated/replaced/disconnected/archived only its
in-memory intercepted row; no private command reached a provider.

`display-localization.log` records real theme/density save/reload/restoration and
Russian keyboard/popover/reduced-motion checks. `russian-final.log` records actual
backlog counts 0 / 0 / 0 and no horizontal overflow. Final `workpage-measurements.json`
records page errors and expected HTTP failures. Deliberate intercepted 403/404/409/503
and aborted-request errors are separated from normal-route evidence. Temporary old
asset 404s/connection refusal occurred while a necessary preview restart replaced
the build; captures were repeated after readiness. No uncaught application exception
is accepted as a passing result.

## Additional actual runtime proof and remaining external work

- Real queued catalog request `9d7c33b9-feac-4b67-8287-e3184bfa4dfb` was explicitly
  cancelled; the persisted result is `cancelled`, with both cancel and retry false
  after reload (`cancel.log`). A preceding script expected HTTP201 instead of the
  endpoint's HTTP200 and stopped after submitting
  `7ad0607a-a4a1-4893-88e8-25ec807996f8`; inspection confirmed it succeeded. It was not
  automatically replayed. Both jobs only refreshed public metadata; no extra candle
  history was loaded.
- Missing notification provider now returns `notification_provider_unavailable`
  with HTTP503 instead of the previous fallback500. Foreign provider scope returns
  HTTP403; resolver exception text is replaced with a stable reason.
- Compatibility: added routes, cursor projections, additive schema and disabled-by-
  default feature are `compatible-change`. Existing candle and catalog consumers
  remain supported. Auth/origin/role/CSRF/recent-auth enforcement is unchanged (`none`);
  the session view identifier is intentionally non-bearer and has no revocation
  consumer on this read route. Monitoring snapshots stay compatible; history is
  additive. Candidate routing changes are scoped to its preview and legacy redirects.
- Required external proof: an existing authorized Binance/Bybit testnet connection
  service/account, and an existing scoped Telegram provider with confirmed recipient.
  Both were requested during the task; neither is configured in this preview.
  Enter credentials only through the protected form, never chat/evidence. A real
  passkey device step-up was not exercised; protocol/CSRF/no-replay tests passed.
  The existing shared preview API expects WebAuthn origin `http://localhost:20110`;
  direct assertions from `20120` therefore cannot complete with this configuration.
  No origin allowlist was widened. Recent-auth UI also offers the existing explicit
  logout/login path with an encoded local return URL; a fresh authenticated session
  satisfies the existing server recency check. This is a preview capability limit,
  not a successful hardware-passkey claim.
- No trade, production service control, publication, Git commit/push or deployment
  was performed. The goal is blocked after the required consecutive-turn audit;
  it is not complete. Resume when the external integration prerequisites are available.

## Final repository gates

- `.venv/bin/ruff check .`: passed.
- `.venv/bin/pyright`: 1,515 files, zero errors/warnings.
- Navigator complete suite: 235 passed; 11 directly affected UI tests rerun after
  form styling, then both connection tests after the existing-login fallback; typecheck and final production build passed.
- `.venv/bin/python -m tools.docs.generate_docs_index --check`: passed.
- `.venv/bin/python -m tools.docs.generate_project_map --check`: passed (5 artifacts).
- `.venv/bin/python -m tools.docs.generate_runbooks --check`: passed (9 docs + index).
- `git diff --check`: passed. No release, production or external provider result
  is inferred from these local checks. Generated maps/index include the current
  working tree, including preserved preexisting candidate files.

## External availability recheck — 2026-10-03 00:26 UTC

Authenticated again with the existing disposable preview account against the
running API on `localhost:20111` (login HTTP200). Read-only capability checks still
return HTTP503: `/ui/account/exchange-connections` reports
`exchange_control_unavailable`; `/ui/account/notifications/scoped` reports
`notification_provider_unavailable`. Credentials, cookies and raw bodies were not
printed or saved. No provider command was sent. Navigator `/login` on `20120`
remains HTTP200. Existing wiring and the open matrix criteria were rechecked;
there is no newly available independent implementation work or external authority.
This continuation confirms the existing impasse, not new implementation progress.
The previously requested authorized testnet service/account and scoped Telegram
provider/confirmed recipient remain the minimum external inputs needed.

## Blocking audit — 2026-10-03 00:27 UTC

The previous continuation was no progress: it confirmed the existing external
impasse without changing the next action. This third consecutive goal turn repeated
the authenticated checks against the current runtime and observed the same HTTP503
codes for both capabilities. The check's own temporary session was logged out
(HTTP204); Navigator `/login` remained HTTP200. No provider mutation or new credential
configuration was attempted. No new authority or external input has arrived, and no
independent required implementation work remains within the recorded proof boundary.
The Goals blocking threshold is met. Real private lifecycle and Telegram integration
criteria remain open; local tests and intercepted client workflows do not close them.

## User correction — active Settings categories

Removed the inset accent line from `.work-categories a[aria-current=page]` in
`apps/navigator-web/src/workpages.css`; the selected background remains. This
user-requested visual change supersedes the earlier category-selection appearance.
The rebuilt preview was checked in the user's in-app browser on Profile and
Preferences: computed `box-shadow: none`, no border or outline in the selected
state. Focus styling and navigation semantics are unchanged. Focused self-review
found no additional issue; API/persistence compatibility impact is `none`.
`pnpm --dir apps/navigator-web build` passed with the existing bundle-size warning.
The initial workspace-filter command matched no project and did not perform a
build; the direct package command above performed the verified build.
[Current screenshot](roehub-workpages-2026-10-03/settings-active-without-border.jpg).
This correction does not resolve or resume the blocked external integrations.

## User correction — Settings controls and compact Interface form

- Removed routine Refresh controls from Profile, Interface, Notifications and
  both Security tabs. Read errors keep an explicit retry; unknown save outcomes
  keep a read-only Check saved state action. The normal forms have no refresh row.
- Security opts out of the shared table expansion control. Loading feedback uses
  the existing table header, eliminating the separate routine toolbar without
  moving loaded rows during background reads. Tab identity is scoped separately
  so Sessions data cannot appear under Activity while its read is pending.
- Preferences is labelled Interface / Интерфейс, retaining `/settings/preferences`
  and existing stored values. Its panel is capped at 448px; desktop selects are
  205px wide and 34px high. The narrow layout stacks the fields.
- Focused self-review: no outstanding finding. UI behavior is an authorized
  compatible change; API/auth/persistence impact is `none`. The shared expansion
  option defaults to enabled for existing callers. No backend or provider work.
- `pnpm --dir apps/navigator-web test src/settings-page.test.tsx src/motion.test.tsx`:
  11 passed, including unknown-save reconciliation, denial, dirty navigation and
  delayed Security tab reads with keyboard focus. Typecheck, production build and
  `git diff --check` passed; the existing bundle-size warning remains.
- Real in-app browser: all four categories inspected, no Refresh; Security has no
  expansion button, ArrowRight/Home switch tabs and retain keyboard focus. Custom
  interval 5 fails validation, focuses its field and sends no valid save; Cancel
  restores the saved 15s draft. No account preference was saved during this check.
- Desktop 1414x959 and narrow 390x959 checked; document width equals viewport width.
  Russian Interface labels fit at 390px. Browser locale and viewport override were
  restored, leaving `/settings/preferences` open. Final-document console log query
  returned no errors; this is not a full-session network trace. Notifications still
  honestly reports its pre-existing missing-provider condition.
- Screenshots: [Interface desktop](roehub-workpages-2026-10-03/settings-interface-compact-1414.jpg),
  [Interface narrow Russian](roehub-workpages-2026-10-03/settings-interface-compact-390-ru.jpg),
  [Security desktop](roehub-workpages-2026-10-03/settings-security-simplified-1414.jpg).
  These supersede the corresponding earlier Settings appearance, not other page proof.
  The external integration goal remains blocked.

## User correction — Strategies library action axes

Restored `margin-inline-start: auto`, 6px gap and no wrapping for the common
Navigator library header `.actions` group in `navigator.css`. This overrides the
legacy report-workspace left-margin reset; Strategies and Backtests now use the
same right inset as Data. No data, command or chart behavior was changed.
At 1414x959, Strategies previously placed Refresh/Filters at x=155.4609375/187.4609375;
the final coordinates on all three pages are x=217/251, y=59, size 28x28. At 390x959
the stable final coordinates are x=307/341, y=55 on all three pages, with document
width 390. One initial Strategies measurement during navigation was transitional;
it was replaced after opening the user's selected strategy and observing its
settled layout. [Computed measurements](roehub-workpages-2026-10-03/strategies-actions-alignment.json).

The narrow filter popup remains inside the viewport (x=68 to 370); Escape closes
it and returns focus to Filters. Browser viewport override was reset and the
original strategy URL/Price & executions view retained. Build and `git diff --check`
passed; focused CSS self-review found no remaining issue. API/persistence impact:
`none`. Existing bundle-size warning remains; no new tests for this CSS-only fix.
[Final screenshot](roehub-workpages-2026-10-03/strategies-actions-aligned-1414.jpg).

## User request — Data ingestion layout and collection proposal

Read-only product inspection; no application implementation for this request.
The current create form at 1414x959 places From and To diagonally because the
five fields flow through a two-column grid. The Downloads library combines
catalog refresh requests and candle ingestion requests, with text-only states.
The completed-job view was also inspected at 1280x720.
[Create form](roehub-workpages-2026-10-03/ingestion-audit-01-create.jpg),
[Completed job](roehub-workpages-2026-10-03/ingestion-audit-02-completed.jpg).

Source inspection confirms that `ingestion-page.tsx` submits a bounded
`candle_ingestion` request. `work_requests.py` finishes after the requested range;
this command does not enable subsequent streaming. A separate existing
`market_data_ws` worker reads the effective collector set, formed from user
instrument selections and active strategy pins. Worker availability and live
collection were not proved by this inspection.

Proposed direction: select instruments in the left panel, show actual coverage
and freshness in the main workspace, place the two range inputs together, expose
ongoing collection separately, and move candle-request history below. Catalog
refresh belongs to the catalog context. Use distinct status icons plus labels;
an enabled collection preference must not be presented as proof of a healthy
stream, and strategy-required collection must remain visible when unselecting.

One generated visual concept uses illustrative data and is not implementation
or runtime evidence. Artifact:
`/Users/daniildegtyarev/.codex/generated_images/01a0fed5-002f-7850-af30-cb02b705bce6/exec-7e627eda-210b-403c-a08d-20137c626ef0.png`.
Focused self-review only; no automated tests needed for this proposal. Temporary
audit tab closed and viewport override reset. API/persistence impact is `none`
because no application behavior changed. The original external-integration goal
remains blocked.

## Accepted Data exchange workspace — 2026-10-03 implementation

User selected the third image concept, then explicitly requested implementation
using native Roehub colors, fonts and working areas. This supersedes the earlier
instrument-library proposal only for Data; Connections was not redesigned.
[Accepted image](roehub-workpages-2026-10-03/data-approved-exchange-concept.png),
[desktop result](roehub-workpages-2026-10-03/data-exchanges-final-1487.jpg),
[narrow result](roehub-workpages-2026-10-03/data-exchanges-final-390.jpg).

Owned implementation: `apps/navigator-web/src/data-page.tsx`,
`data-page.test.tsx`, `data-workspace-api.ts`, `data-workspace.css`, and the Data
root class in `app.tsx`; new `apps/api/routes/market_data_catalog.py` and its wiring;
new coverage port/ClickHouse reader; additive scoped repository reads in
`catalog_snapshot_repository.py`, `work_request_repository.py` and
`instrument_selection_repository.py`; new API tests. Current functional and
information-architecture documents describe the accepted layout. Foreign changes,
existing migrations and the platform-web baseline were preserved.

The library selects Binance/Bybit. The catalog combines their Spot/Futures markets,
filters and sorts before server pagination, displays actual period coverage and
separate per-instrument job progress, and supports at most eight symbols from one
market per bounded request. There is one stream switch in the inspector; active
strategy pins disable it and expose scoped strategy links. Stored-candle freshness
is separate from enabled intent. Reads retain identity-labelled snapshots and lock
commands during selection changes; denial clears protected data. One submission
owner manages the durable recovery key without automatic command replay.

Compatibility: browser behavior and additive API projections `compatible-change`;
existing command authorization, selection pin semantics, storage schema and
credential boundaries `none`. No new migration. Bounded server reads aggregate
canonical minute counts rather than returning candle payloads to the catalog.
Independent security/scope review found no organization leak in the new query
joins/projection. Four frontend findings were fixed (cross-market selections,
denial hiding, period synchronization and failure visibility), with focused
self-review and regression checks after repair. No publication or trading.

Validation:
- `pnpm --dir apps/navigator-web test`: 248 passed across 26 files, including 14
  Data tests for retained reads, late responses, denial, one locked stream toggle,
  unknown command reconciliation, filters, range validation, accepted request
  bounds, native datetime input, range navigation, batch scope and recovery reload.
- Navigator typecheck/build passed. Existing bundle-size warning remains.
- Focused API/workspace/wiring/selection-repository pytest: 12 passed. Ruff on the
  touched backend files and new API test passed; focused Pyright returned 0 errors.
- Real in-app browser: both exchange catalogs, search/market segments, descending
  coverage sort, page 2 (51–100 of 1,421 Bybit rows), instrument selection and Back.
  Streaming was enabled for Bybit Spot BTCUSDT, observed checked after reload, then
  restored off. Active-strategy lock has unit/SQL-policy review proof; no new live
  strategy was started for that case. Continuous worker endurance is unverified.
- Native datetime inputs initially changed displayed values without updating the
  React draft. The first request used the prior one-day range, not the intended
  five minutes. Fixed using input events plus accepted-server-range commitment;
  one shared command controller also removes duplicated recovery ownership and
  the stale requestkey race. No stored candles were removed during repair.
- Bybit Spot request `da1f85d5-5511-40c9-b9a4-feb40ad74e5b` succeeded for
  2026-10-02 18:35 through 2026-10-03 18:35 UTC: 1,440/1,440 coverage. The corrected
  five-minute request `da9bd7d4-6a72-4d0d-92d4-5bae4b319ac7` succeeded for
  2026-10-03 18:30–18:35 UTC with 5/5 canonical candles. That range was already
  covered by the first request; this second check proves correct request bounds,
  completion and preserved coverage, not new missing-candle acquisition.
- Historical Bybit request `afcecc30-9725-44e7-865d-40f1365c34ba` for
  2024-01-01 00:00–00:05 UTC failed with 0/5 and the redacted source/storage error.
  The UI displayed failure in catalog and job detail; the underlying cause remains
  undetermined. No automatic retry or false successful-provider claim.
- Existing local reference data had only market 1. The existing insert-only-missing
  `SeedRefMarketUseCase` added canonical markets 2/3/4, preserving prior values.
  Real public catalog refresh jobs succeeded: Binance Futures
  `31baf69d-b4bb-49af-bdfc-bb1771e9cacf`, Bybit Spot
  `c1070115-71d2-4144-95b2-57b41ec0dbb0`, Bybit Futures
  `4f4a2e4e-983b-4054-9a84-fdc7bb39721a`. These are metadata refreshes, not candle
  backfills or newly enabled streaming selections. Existing local databases retained.
- Desktop 1487×1058: 38px rows, one 38×24px switch, document width 1487. Narrow
  390×844: document width 390, 560px table inside a 324px scroll surface; inspector
  form fully flows below the table. Final-document console errors/warnings empty.
  Viewport override reset after checking. No claim of production delivery.

The current UI adaptation is implemented locally. The original Goal remains
blocked for its separately documented private exchange/notification integrations.


## 2026-10-03 — full-history defaults and download icons

User explicitly required the entire available instrument history, with server limits
changed as necessary. This supersedes the initial 10,080-minute download cap;
coverage reads retain their seven-day cap. No multi-year download was run for QA.

| Requirement | Status | Implementation and proof |
| --- | --- | --- |
| Row Download is an icon button | verified | Native library action, Lucide Download, accessible instrument/market name; measured 28×28px, rows remain 38px |
| Default period is the actual available history | verified | Scheduler-confirmed first 1m candle, current closed-minute end; independent of URL/table period; real Bybit Spot/Futures values below |
| Remove seven-day download restriction | verified | WorkRequest validation + migration 0024; authenticated API regression accepts years; real PostgreSQL accepts full range |
| Resume bounded work without losing progress | verified | 60-minute windows, ten-window turns, fenced persisted progress; unit cross-symbol resume and real PostgreSQL resume/cancel-at-yield |
| Preserve manual edits and identity/access boundaries | verified | Client tests cover metadata refresh, manual input during queued discovery, late instrument selection, and second-symbol 403 following first-symbol 503 |
| Show actual results independently of work progress | verified | Real three-minute historical request, 3 written, canonical coverage 3/3 |

### Implementation and compatibility

Frontend: `data-page.tsx`, `data-workspace-api.ts`, `market-data-api.ts`,
`data-workspace.css`, `ingestion-page.tsx`. Backend: catalog/workspace routes,
API/scheduler wiring, work-request runner/repository, new history-bounds runner,
repository and bounded metadata HTTP wrapper. Strict interactive discovery never
falls back to instrument listing metadata. Legacy scheduler source behavior keeps
its default; the new caller explicitly requires a confirmed candle.

Migration `0024_market_data_full_history_v1.sql` and its manifest/bootstrap phase
are registered. Existing 0023 bytes are preserved. Download ranges are positive,
closed, minute aligned, since 2017, 1–8 symbols, int32 total units. One unfinished
job per organization and existing idempotency/origin/authorization remain. After
ten windows a job queues itself with its checkpoint, without incrementing attempt.
Cancellation racing with yield is terminal and fenced. Active retries take priority
over newer terminal jobs for overlapping-period table progress.

`GET /market-data/workspace/history-bounds` authorizes the user/organization before
reading/warming a public metadata inbox. The API does not call exchanges. Admission
is capped at 64 queued/running probes, tested with 80 concurrent requests. Queue-full
is explicit 503/Retry-After. Metadata probes allow one read-only retry within a
25-second budget; no candle command is automatically replayed. Manual period input
remains available during missing/pending metadata. First-candle values are cached;
unknown dates are never substituted with an arbitrary recent period.

Compatibility: API/DTO extension, expanded download acceptance, persistence,
frontend behavior and scheduler wiring are `compatible-change`. Existing public
candle consumers, organization scope and credentials are `none`. Deploying this
extension requires migration 0024 and the updated API/worker together. Before a
rollback to the old range-constrained version, finish/cancel long jobs and assess
its obsolete constraints; this task performs no rollback or publication.

### Checks and review

- `pnpm --dir apps/navigator-web typecheck`: passed.
- `pnpm --dir apps/navigator-web test --run`: 252 passed in 26 files.
- `pnpm --dir apps/navigator-web build`: passed; existing >500 kB bundle warning.
- Focused `.venv/bin/pytest -q` targets: work_requests, history_bounds,
  rest_instrument_history_start_source, history_probe_http, API workspace/catalog,
  migration storage lifecycle, scheduler funding wiring: 47 tests passed in the
  final combined run.
- `.venv/bin/pytest -q tests/real_infra/market_data/test_work_requests_postgres.py`:
  6 passed using the existing local PostgreSQL in unique disposable schemas.
  Includes concurrent admission, full-history persistence/resume/cancel and retry
  projection; no user tables were cleared.
- Touched backend Ruff and Pyright: passed. Test doubles received explicit
  optional-value assertions after typecheck findings. One erroneous Pyright call
  included a TSX file; corrected Python-only rerun passed. TSX is checked by tsc.
- `git diff --check`: passed.
- Focused self-review plus one independent reviewer. Four P2 issues were fixed:
  listing-date fallback, long-range coverage link, false queue admission/manual
  input lock, and old active retry hidden by newer terminal work. Follow-up found
  a batch-denial check using only the first error; fixed to check every result and
  covered by a 503-then-403 regression. No outstanding review finding in this scope.

### Actual local runtime evidence

Existing preview/API/request consumer were rebuilt/restarted without fixture reset.
Migration 0024 applied successfully to the legacy preview schema. The attempt to
record a standard storage marker found that this preview fixture has no
`roehub_storage_migrations` table; no fabricated baseline markers were created.
The migration is registered/tested through the repository lifecycle, and local
post-apply inspection confirmed the new table and full-range constraint.

In the real browser on localhost:20120, Bybit ZRXUSDT Spot defaulted to
`2021-10-19T13:40`, and Bybit ZRXUSDT Futures eventually confirmed
`2022-03-28T10:51` (UTC). The upper boundary followed the current closed minute,
independent of the original table period `2026-10-03T18:30–18:35`.

The Futures discovery initially failed. Bounded diagnostics confirmed intermittent
`ReadTimeout` and exhaustion of the 25-second probe budget; identical reads also
returned HTTP 200. The UI exposed unavailable/retry/manual input. A subsequent
scheduler probe confirmed the actual first minute. This proves the failure path
and eventual success in this session, not provider uptime or latency guarantees.
[Bybit Kline API](https://bybit-exchange.github.io/docs/v5/market/kline) documents
start/end boundaries, minute intervals and reverse chronological rows used by the
confirmed-bound search; provider payloads were not saved to evidence.

A deliberately limited download of the first three Spot minutes
`2021-10-19T13:40–13:43` completed as job
`40b5e38b-1bcc-4cdb-978d-0824265c8288`: state `succeeded`, completed/total `3/3`,
rows_written `3`; the canonical coverage API and browser independently showed
`3 / 3`, `100%`. No multi-year ingestion, stream preference change or trading action
was performed. The original table period was restored and the download form again
shows the full default history.

Final browser measurements: viewport/document 1414px, row 38px, icon button 28px;
398px narrow view stacks the inspector with usable adjacent dates. The final
current-document console had no errors/warnings. This is not a full network trace.
Screenshots in `roehub-workpages-2026-10-03/`:
`data-full-history-final-1414.jpg`, `data-full-history-398.jpg`,
`data-first-history-three-candles.jpg`. Temporary viewport override was reset.

Proof boundary: all four exchange-market branches have deterministic source tests;
real history discovery in this amendment was verified on Bybit Spot and Futures,
and actual ingestion was intentionally limited to three Spot candles. Full-history
scheduling/resume is verified with isolated persistence and deterministic windows,
not by downloading years. Broader private exchange/Telegram Goal blockers remain
unchanged. This is local implementation evidence, not publication or production readiness.


### 2026-10-03 — Data filter icon alignment

Corrected only the Data catalog toolbar in `data-page.tsx` and
`data-workspace.css`. The search SVG previously started at the input's top edge:
its center was 7px above the 28px field center. The three selects used native
arrows positioned against their right edge. Labels now establish a block positioning
context; search and three decorative Lucide chevrons are 14px, centered vertically,
and inset 8px. Native select semantics, field heights and existing visual tokens remain.

Real browser at localhost:20120/data: all four vertical center offsets are 0px and
edge insets are 8px at both 1414×959 and 398×844. Narrow document width is 398px.
Search BTC plus Futures returned matching futures rows; reset restored defaults.
Typecheck, 18 existing Data tests and production build passed (existing bundle-size
warning). No new tests were added for this cosmetic change. Focused self-review;
no API/persistence compatibility impact. Local preview rebuilt and restarted.
Screenshots: `data-filter-icons-1414.jpg`, `data-filter-icons-detail.jpg` and
`data-filter-icons-398.jpg` in the workpages evidence directory. Temporary viewport
was reset. This entry verifies this toolbar change, not unrelated page functionality.


### 2026-10-03 — Downloads merged into Data Navigator

User-approved replacement: the standalone Downloads screen and duplicate creation
form are removed from the Navigator candidate. `data-downloads.tsx` provides an
optional compact journal below the still-mounted catalogue, with state/type
filters, source labels, status icons, bounded cursor history and a close control.
The instrument inspector now owns job progress, periods, cancellation, retry,
errors and expandable attempt details. Stored coverage and job progress are
labelled separately. Catalog refreshes and batch requests retain details too.
Connections and Monitoring links target Data; `/data/ingestion` only redirects
old bookmarks into the journal, preserving job/filter/recovery identities. A
legacy download range exceeding the coverage-read limit is not reused as a
catalogue coverage range; the requested job period remains visible in its details.

Touched client boundaries: `data-page.tsx`, `data-downloads.tsx`,
`data-workspace.css`, `app.tsx`, Connections/Monitoring links, query recovery budget,
Data and cache tests; removed `ingestion-page.tsx`. New journal reads occur only
while expanded. Commands lock during retained/error/pending states, use existing
server attempt fencing, and never retry automatically. Unknown action recovery is
subject/job scoped, survives inspector reopening for five minutes outside the read
budget, and is cleared with existing session cache cleanup. An older selected job
cannot enable a second download while the catalogue reports a current active job.
A local missing-job 404 preserves the instrument; denied reads/actions clear the
protected views through the existing access-loss path.

Validation: `pnpm --dir apps/navigator-web typecheck`, all 262 tests in 26 files,
production build, and `git diff --check` passed. Build retains the existing bundle
size warning. The 27 Data tests cover legacy redirect/filter preservation, lazy
journal reads and pagination, retained selection locks, cancellation confirmation,
retry attempt identity/double clicks, unknown-result reconciliation after reopening,
access denial, current-vs-historical job eligibility and catalog refresh selection.
Focused self-review found no outstanding issue in the changed scope. No independent
review threshold was crossed; no backend, authentication or persistence contract
was changed.

Real browser proof on localhost:20120: journal open/filter/select, failed-job reason
and retry availability, completed details/attempts, old-link redirect, refresh,
Enter/Escape and focus return, 1414×959 and 398×844 layouts. At both widths the
document matches the viewport; tables scroll within their panels. A new minimal
Bybit Spot ZRXUSDT request `dbe7e0d0-5f3b-4e12-9d1e-7a47b1381601` covered only
2021-10-19 13:42–13:43 UTC, succeeded with 1 candle read and 0 written because it
already existed. It finished before browser cancellation could be exercised.
Cancel/retry/unknown outcomes are verified by deterministic client tests, not by a
new live cancelled/retried job in this amendment. Existing failed jobs were not
mutated. Final current-document console query returned no errors or warnings;
no full network trace or multi-year ingestion test was performed.

Compatibility: API/DTO, backend side effects, auth and persistence `none`;
existing job bookmarks and cross-page links `compatible-change` via redirect;
standalone creation-form workflow `breaking-change`, intentionally retired by the
user and replaced by the existing instrument/batch controls. This assessment is
for the local Navigator candidate, not a default platform-client cutover or a
production delivery claim. Existing unrelated Goal blockers remain unchanged.

Screenshots under `roehub-workpages-2026-10-03/`: `data-journal-1414.jpg` and
`data-journal-398.jpg`. Preview rebuilt/restarted; temporary viewport reset.

### 2026-10-04 — Continuous download progress and ETA

The user authorized the four diagnosed corrections: one shared job-progress
source, quiet same-identity reads, a continuous public download state across worker
yields, and an estimate immediately after the progress bar. Implementation is
limited to the selected Navigator candidate. The ingestion worker, provider calls,
API/DTO, persistence and running user request were not changed by this amendment.

`download-progress.ts` shares one 2-second timer and QueryClient job key across the
catalogue cell, journal row and inspector. Only mounted jobs retain a subscription;
job payloads keep the existing QueryClient budget. Catalogue/inspector periodic
reads pause while the catalogue contains active work; journal history no longer
polls for progress. Commands, explicit refreshes and a terminal job transition
refresh the relevant derived snapshots. History-bound reads retain one-minute
freshness independently of background job reads. A first resolved terminal job
also reconciles a catalogue entry that still reported it as active.

The public state is Downloading for running jobs and for queued jobs with a
started_at or completed work. A genuinely unstarted job remains Queued; cancellation,
failed and terminal states remain explicit. Routine reads do not show Updating or
disable cancellation. Retained identities, failed reads, lost access, commands in
flight and uncertain command outcomes retain their existing safety locks. Inspector
status slots, table columns, percentage and ETA space have stable geometry.
Percentages are consistent across all three surfaces and do not round unfinished
work up to 100%.

ETA uses a rolling 120-second window of observed completed units and elapsed wall
clock, including queue/worker pauses. It warms up for at least 20 seconds, smooths
rate changes and recalculates at most every 10 seconds. No advancement for 60 seconds
shows No progress; read errors show Unavailable; neither invents a countdown. A new
attempt or reset of completed units clears the rate history. ETA is an estimate of
remaining work, not an ingestion-speed improvement or a completion guarantee.

Validation: typecheck and production build passed; all 268 tests in 27 files passed.
After the final completed-bar styling adjustment the 33 focused Data/ETA tests and
typecheck/build passed again. git diff --check passed (new untracked candidate files
are not included by Git's tracked diff check). The existing bundle-size warning
remains. New checks exercise twelve successive polls with row/journal/inspector
mounted, exactly one job request per tick, no repeat catalogue/coverage/bounds/history
reads despite a configured 15-second refresh preference, stable DOM/focus/scroll,
continuous state through running/queued yields, ETA agreement, pending-read command
eligibility, visible errors, explicit read recovery, and a single terminal snapshot
refresh. Rate tests cover warmup, pauses, smoothing, rolling-window adaptation,
stalls, read loss, retry reset and terminal percentage semantics.

Real browser proof: external Brave, existing Binance Spot ZROUSDT full-history
request 85e21d9a-ca42-42b8-91df-8fd1dd88d0cf. Its 1,202,849-minute job advanced from
62,280 to 69,180 and later 80,160 completed units while showing Downloading, matching
percentages/ETA in all three surfaces, and an enabled Cancel download control. No
cancel/retry/new ingestion command was issued. Observed estimates changed from
about 9 hours to 8 hours and later 6 h 45 min as the measured throughput changed.
Native screenshots confirm the desktop layout; the three saved 21:00:24–25 UTC
frames have the same 69,180 count and must not be described as three distinct
progress ticks. Stable nodes/focus/scroll and per-endpoint read counts are proved by
client tests, not a browser DOM/network trace. Mobile emulation was attempted but
was interrupted by native browser-control changes; the saved DevTools screenshot
is desktop, not mobile proof. A new mobile pass and console/network capture remain
unverified for this amendment. Device emulation was confirmed off and DevTools
closed afterward. Only the web preview on 20120 was restarted; ingestion continued.

Evidence: roehub-workpages-2026-10-03/data-progress-brave-0.png (plus -1/-2),
data-progress-brave-desktop.jpg, data-progress-brave-inspector.png, and
data-progress-brave-devtools-desktop.png. The inspector image is a crop of the
native desktop capture, not a separate rendering. Preview assets are
main-DyvD9IVB.css and main-DII_0tV8.js.

Compatibility: browser-visible progress behavior compatible-change; API/DTO,
authentication, persistence, backend side effects and configuration none. Focused
self-review found no remaining issue in these four changes. No independent-review
threshold was crossed. This does not establish a platform-client cutover, production
delivery, or completion of unrelated Goal blockers.


## Download pause/resume and recovery — 2026-10-04

Implemented in the selected Navigator candidate and existing Market Data worker:
checkpointed pause/resume, terminal cancel preserving candles, durable transient
source/storage backoff, lost-worker recovery, explicit wait states, reset ETA epoch,
and independent automatic browser GET reconnection. New controls follow existing
compact button/icon styles. Unchanged manual Retry semantics restart its work
counter; stored candles remain and the fill deduplicates their minute keys.

Verification:
- Full Navigator suite: 272 passed / 27 files. After the final storage-wait copy and
  test type corrections, affected suites: 37 passed / 2 files; typecheck/build pass.
  Build assets: main-BOJ1_pTp.css and main-vKoAnSB4.js; existing bundle-size warning.
- Python: `pytest -q tests/unit/contexts/market_data tests/unit/apps/api/test_market_data_workspace.py
  tests/unit/apps/api/test_market_data_catalog.py tests/unit/apps/migrations/test_storage_lifecycle.py`
  passed 208 tests. Final runner/HTTP/fill subset passed 19 (including one additional
  invalid-JSON regression). Focused Ruff passes; focused Pyright passed with 0 errors / 0 warnings.
- `tests/real_infra/market_data/test_work_requests_postgres.py`: 9 passed with an
  explicitly configured local PostgreSQL DSN, isolated disposable schemas. Covered
  pause during failing I/O, resume cooldown, concurrent controls, stale worker and
  command fencing, cancelled/paused priority, retry schedule and unchanged counters.
- One independent review found two P2s (truncated HTTP response classification,
  stale GET overwriting a command response). Both fixed and regression-tested;
  reviewer confirmed closure. Follow-up review of storage classification found no
  new blocker, noting ambiguous psycopg errors without SQLSTATE.

Browser proof used only Codex IAB, no external browser. Production built frontend,
production work-request router and repository served an isolated PostgreSQL job;
source was an explicitly synthetic bounded source with no candle writes. Verified:
pause at 2,460/20,160, reload keeps Paused and Resume; temporary source failure keeps
12.2% in retry_wait with due time; automatic recovery reached 4,140 and ETA; injected
browser GET503 disables commands and shows No connection at 4,800 while worker keeps
running; reconnect alone showed 6,480 and enabled controls, with no command replay;
Cancel became terminal at 9,120 with its progress preserved. IAB console-error query
returned no entries (not a claim that injected failed network reads did not occur).
Snapshots: data-pause-iab.png, data-retry-wait-iab.png,
data-browser-offline-iab.png, data-cancel-preserved-iab.png in
`.codex/delivery/evidence/roehub-workpages-2026-10-03/`.
This validates real UI/API/PostgreSQL controls, not external-provider fault recovery
or ClickHouse writes. Existing REST fill deduplication is covered by its unit tests.
Desktop 1280x720 checked; mobile/responsive proof remains outside this narrow pass.

Local runtime updated in order: cooperative requests-only shutdown, migration 0025,
new worker/API, new web bundle. Initial user's running job
85e21d9a-ca42-42b8-91df-8fd1dd88d0cf advanced from 172,920 to 173,400 at shutdown and
continued to 180,120/1,202,849 after restart. It then became failed with
source_or_storage_unavailable at 2026-10-03T21:46:43Z. Its state/progress was not reset,
paused, cancelled or retried by this task. A read-only next-window probe through
real fill with NoWrite returned 60 rows and found them already canonical, but this
does not reconstruct the failure: writer, lease-check and checkpoint were not
replayed. Cause remains undetermined; future errors now record phase+class only.
Do not present the user's current download as running or successfully recovered.
An initial API restart omitted its opt-in work-request flag; this was corrected,
the local restart helper now preserves it, and both command routes were verified
present afterward. This temporary read 404 did not stop the separate worker.

Compatibility: additive API fields/endpoints and current Navigator consumer are
compatible-change; migration preserves old rows. Expanded persisted states and the
new Store.defer port require coordinated worker/API upgrade (old-worker coexistence
and rollback are breaking-change). Browser default automatic GET recovery is an
accepted compatible-change with existing denial/unknown-command boundaries.
Org/origin/auth checks, candle identity, minute deduplication and trading behavior:
none. No platform-client cutover, publication, production delivery or completion of
unrelated Goal blockers is claimed. PostgreSQL connection setup errors lacking
SQLSTATE may be retried even when caused by misconfiguration; explicit auth/SQL
error codes and invalid instruments remain terminal. Bybit temporary error mapping
uses its official [error codes](https://bybit-exchange.github.io/docs/v5/error) and
[rate-limit headers](https://bybit-exchange.github.io/docs/v5/rate-limit).

## Failed-download diagnosis and manual retry observation — 2026-10-04

Discussion/diagnosis only; no application code or resource limits changed. This
follow-up resolves the previously undetermined cause above. ClickHouse
`system.query_log` records rejected INSERTs with exception 241
`MEMORY_LIMIT_EXCEEDED` at 2026-10-03T21:46:43Z and 2026-10-03T22:13:07Z,
matching respectively Binance job 85e21d9a-ca42-42b8-91df-8fd1dd88d0cf at
180,120/1,202,849 (14.97%) and Bybit job
3d3b986e-7781-459d-a6db-f54302feda79 at 5,220/1,203,041 (0.43%). The Bybit worker
log identifies `phase=fill_window error_type=DatabaseError` at the same instant.
No raw queries, credentials or provider payloads are included here.

The server reported a 3.44 GiB memory budget; both failing INSERT query memory
counters were about 6.65 MiB. Other SELECTs also hit the server memory limit.
This establishes storage memory-pressure rejection, not which concurrent process
or query consumed the remaining memory. The current storage retry classifier
handles ClickHouse OperationalError but not this DatabaseError, so both jobs
became terminal failed with the generic source_or_storage_unavailable code.

In Codex IAB, manually retried Bybit once. Its attempt became 2, progress reset
to zero and then advanced. At 2026-10-04T06:25:37Z it was running with 4,500
completed/read and zero newly written candles, consistent with rechecking the
already stored interval. Existing candles were preserved. This directly confirms
that current manual Retry restarts the counter rather than resuming its checkpoint.
At the final browser observation, Bybit passed the previous stop: 6,480 completed,
1,200 newly written, 0.53%, ETA approximately 6 h 45 min. Screenshot:
data-bybit-retry-observed-iab.png. This proves short-term progress after retry,
not full-history completion or that storage memory pressure cannot recur.
The expanded Details and attempts disclosure shows counters and the previous
generic failure, but no chronological diagnostic events or storage reason.

Retried Binance once while Bybit was active. The UI displayed "The state has
changed. Refresh before continuing." Binance remained failed, attempt 1, at
180,120 units. Source confirms the one-unfinished-job-per-organization constraint
raises HTTP409 with "Finish or cancel the current request first"; the frontend
loses that distinction. Screenshot: data-retry-conflict-iab.png in the existing
evidence directory. Bybit was left running; no pause/cancel, queue-policy change,
automatic retry fix or new log viewer was applied in this discussion turn.
