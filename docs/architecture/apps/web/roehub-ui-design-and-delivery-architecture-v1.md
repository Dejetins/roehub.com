# Roehub — архитектура проектирования и поставки интерфейса v1

Документ сохраняет принятую целевую Web-архитектуру Roehub, визуальную базу и
границы совместимости будущих изменений интерфейса.

## Статус и полномочия

- Статус: `accepted architecture`, обновлена 2026-09-04 после прекращения
  прежнего поэтапного процесса проектирования.
- Документ сохраняет принятые границы продукта, источники фактов и условия
  передачи дизайна в реализацию.
- Документ не задаёт визуальный стиль, дизайн-систему, обязательный
  дизайн-инструмент, план реализации или разрешение на изменение кода,
  публикацию и развёртывание.
- 2026-09-08 пользователь выбрал [общий план реализации UI](roehub-ui-implementation-plan-v1.md)
  и ограниченный промпт-пак Backtests. Прежние этапы, журналы и управляющие
  артефакты остаются историей; новый пак не является обязательным процессом
  для каждой последующей UI-задачи.

## Цель

Получить целостный, проверяемый интерфейс Roehub, не смешивая:

- локальную самостоятельно развёртываемую платформу;
- публичный сайт `roehub.com`;
- общие foundations и будущие product-owned `@roehub/*` packages;
- факты текущего SSR-интерфейса и целевой клиентский продукт;
- визуальный дизайн и доказательства работающего браузерного поведения.

## Сохраняемые продуктовые границы

1. Локальная платформа и `roehub.com` имеют разные trust, identity, release и
   deployment boundaries.
2. Публичный сайт не предоставляет доступ к локальной установке и не становится
   обязательным control plane для self-hosted продукта.
3. Общие UI foundations могут совместно использоваться только через явно
   выделенные Roehub-owned packages и versioned contracts. Приложения не
   импортируют page composition друг друга.
4. Дизайн не определяет роли, capabilities, API, persistence, secrets policy,
   mutation semantics или runtime truth. Эти границы принадлежат серверным
   контрактам и текущей реализации.
5. Реализация переносится законченными пользовательскими journeys, а не
   несвязанными наборами страниц или компонентов.
6. Responsive Web обязателен в принятом диапазоне локальной платформы. Отдельная
   mobile information architecture остаётся вне scope, пока пользователь явно
   её не разрешит. Публичный сайт сохраняет собственную responsive boundary.

## Принятая целевая Web-архитектура

Очистка visual requirements не отменяет принятую структуру будущего Web
workspace:

```text
apps/
  web/                 существующий SSR-сервер и same-origin gateway
  platform-web/        будущий клиент локальной платформы
  site/                отдельная сборка публичного roehub.com

packages/
  tokens/
  ui/
  charts/
  localization/
  web-contracts/
```

- `apps/platform-web` и `apps/site` не импортируют page composition или
  product-specific feature code друг друга.
- Совместно используемый код выносится только в явно ограниченные и
  версионируемые `@roehub/*` packages.
- `apps/web` сохраняет текущие same-origin, session и server-authority
  обязанности, пока отдельная реализационная задача не докажет совместимый
  перенос конкретного journey.
- Публичный сайт остаётся отдельной сборкой, release и deployment boundary;
  его responsive scope не расширяет mobile scope локальной платформы.
- Эта структура является принятой архитектурой, но не утверждает, что будущие
  каталоги уже реализованы, и сама по себе не разрешает их создание.

## Локальное подключение platform client (S1, 2026-09-08)

`apps/platform-web` содержит pnpm/React/TypeScript/Vite foundation. Единственный
выделенный общий пакет на S1 — `packages/web-contracts`; остальные целевые
пакеты создаются по мере появления потребителей.

`WEB_BACKTESTS_CLIENT_ENABLED` в `apps/web` принимает `true` или `false` и по
умолчанию выключен. При `true` только `/backtests`, `/backtests/new` и
`/backtests/{job_id}` после существующей проверки identity получают клиентский
HTML. `401` сохраняет локальный path/query в login `next`; недоступность identity
возвращает `502` без client bootstrap. Защищённый HTML и redirect имеют
`Cache-Control: private, no-store`. Другие страницы сохраняют SSR.

Команда `pnpm --filter @roehub/platform-web build` создаёт
`apps/platform-web/dist/.vite/manifest.json` и hashed assets. Web проверяет
manifest при старте, монтирует только `dist/assets` по
`/platform-assets/assets` и не предоставляет публичный SPA fallback. После
пересборки требуется перезапуск Web; включённый флаг при неполной сборке
останавливает старт с явной ошибкой. Возврат к `false` и перезапуск восстанавливают
SSR без миграции данных. Browser requests сохраняют same-origin `/api`.

S1 не реализует пользовательские действия Backtests. Локальная fixture,
проверенные команды, browser evidence и ограничения текущего workstation API
описаны в [S1 report](../../../../.codex/delivery/evidence/roehub-backtests-client-v1/S1-client-foundation.md).
Target-role cutover и deployment не выполнены.

## Библиотека platform client (S2, 2026-09-08)

`apps/platform-web/src/library.tsx` реализует оболочку v23 и библиотеку с
независимым чтением выбранного UUID. `library-api.ts` проверяет и оставляет
только используемые поля прямых jobs/status responses; сокращённый read DTO
содержит `request.risk_mode`, а не исходный `request.risk.mode`. Значения
`refresh_status=poll|terminal` задают режим обновления, а не свежесть данных;
клиент помечает снимок старше 60 секунд либо неудачное обновление как устаревший.

URL сохраняет `state`, `risk_mode`, `limit` (1–250), opaque `cursor` и `variant`
для detail. Неизвестные/некорректные query-параметры удаляются. API фильтрует
риск внутри серверной страницы, поэтому пустая страница может иметь следующий
cursor; клиент сохраняет такую навигацию и не объявляет всю историю пустой.
Workstation-проекция заданий сейчас недоступна из-за существующего backend
wiring; поиск/инструмент/даты явно недоступны, клиент не фильтрует весь набор
локально и не исправляет backend authorization wiring.

Снимки обновляются вручную с учётом серверных cooldowns. Ошибка обновления
сохраняет данные с пометкой устаревания; 403/404 скрывают данные соответствующего
чтения. Identity перепроверяется через 30 секунд и при возвращении в окно;
401, смена subject и недоступность identity останавливают защищённые чтения и
очищают private query cache. Поздние ответы отменённых чтений не восстанавливают
данные. In-memory cache не является recovery storage; S3 добавит отдельную
ограниченную recovery-семантику по тикету.

Обзор, стратегии и настройки ведут в существующий SSR. Пункт данных явно
недоступен: текущий Web не имеет `/data`. Конфигурация нового бэктеста,
команды выполнения и результаты пока недоступны. Подробные границы и
воспроизводимые проверки — в [S2 report](../../../../.codex/delivery/evidence/roehub-backtests-client-v1/S2-shell-and-library.md).
Флаг, default SSR, backend API, авторизация и persistence не изменены.

## Конфигурация и отправка platform client (S3, 2026-09-08)

`apps/platform-web/src/builder.tsx` подключает `/backtests/new` к реальным
runtime-defaults, workstation catalog, artifact-date-bounds, preflight и create.
`builder-api.ts` ограничивает DTO явными allowlists, `builder-model.ts` проверяет
поля по текущему каталогу и сохраняет серверную политику defaults. Каталог
символов запрашивается с выбранными `instrument_exchange` и
`instrument_market_type`; старые ответы и границы другого инструмента не дают
права отправки. Неизвестные или неподдерживаемые значения остаются видимыми для
исправления, без скрытого переключения координат. Ограничение workstation jobs
из S2 не мешает независимому каталогу и не считается исправленным.

Редактируемые поля покрывают тикет: упорядоченные индикаторы, их materialized
window/source grid, UTC `[start,end)`, TP/SL с явным включением сторон, направление,
капитал, ставки, sizing, ranking и top N. Комиссия и проскальзывание показываются
в процентах и один раз переводятся в доли; equity и TP/SL сохраняют процентные
единицы API. Funding/profit-lock/close-on-end и minimum-trade policy получены из
серверных defaults; preflight показывает нормализованные значения, стоимость,
покрытие и предупреждения. Любое result-affecting изменение, даже с возвратом
прежнего значения, требует нового preflight. Изменение подписи задания само по
себе не создаёт новое вычисление. Body `errors` блокируют отправку и при HTTP 200.

Перед create клиент заново проверяет subject, фиксирует точное allowlisted body
и один `Idempotency-Key`. Повторный клик не отправляет второй запрос. Успешный
ответ с валидным UUID открывает detail и удаляет resolved recovery payload.
`recovery.ts` хранит только запись одной попытки в памяти и per-tab sessionStorage
(до 256000 символов сериализованной записи); обычный черновик не сохраняется.
Недоступность storage явно показывается перед отправкой. Ошибка записи при
первоначально доступном storage останавливает отправку до нового явного действия.
Несохранённая форма использует подтверждение перехода/перезагрузки; data router
обеспечивает такую же защиту для back/forward. SSR-ссылки сохраняют свои адреса.

Original-organization-bound replay текущий API не гарантирует, current-user
организацию не возвращает. Поэтому replay отключён даже для того же subject:
unknown outcome сохраняет ключ и body и предлагает только чтение истории.
Пустая история в другой организации не доказывает отказ create. Известный result
UUID перечитывается непосредственно; 403/404 не разрешают новый автоматический
запрос. Проверенная локальная factory использует TTL 86400 секунд, но браузеру
это значение не предоставляется как replay-гарантия. Logout/смена subject очищают
recovery. 401 из любого защищённого запроса, включая imperative preflight/create,
немедленно снимает private UI и останавливает запросы. Identity outage не считается
logout. 429 учитывает и HTTP header, и allowlisted `error.details.retry_after_seconds`;
команды автоматически не повторяются. Лимиты admission показаны вместе с запросом
или текущим использованием, без generic response cache.

Воспроизведение: `ROEHUB_PROOF_STAGE=S3 pnpm --filter @roehub/platform-web test:e2e`.
Runner выполняет S1/S2 и S3 в двух отдельных fixture lifetimes, сохраняя cold-empty
assertions и прежние evidence. При явном имени файла выполняется только этот
набор. Реальное proof, controlled faults, совместимость и ограничения описаны
в [S3 report](../../../../.codex/delivery/evidence/roehub-backtests-client-v1/S3-configure-and-submit.md).
Backend/auth wiring, SSR default и rollback-флаг не изменены; lifecycle/cancel
и результаты остаются последующими заданиями. Полное journey и target-role cutover
этим этапом не подтверждены.

## Выполнение и отмена platform client (S4, 2026-09-08)

`execution.tsx` подключает выбранный UUID к ограниченному автоматическому чтению
существующего `GET /backtests/jobs/{job_id}`. Один таймер учитывает серверные
`next_allowed_refresh_at`, `retry_after_seconds` и cooldown ошибки GET; terminal,
403/404 и остановленная сессия не запускают защищённый polling. Переход на другой
job отменяет obsolete read. Терминальное и более новое состояние не откатывается
поздним ответом чтения или cancel. Библиотека сохраняет самостоятельный snapshot.

UI показывает только серверные processed/total units и percent с временем
измерения. Старое измерение и неудачное чтение помечаются явно; timer-based
progress и выдуманной ETA нет. Терминальное состояние имеет приоритет над
процентом. Started/finished timestamps и allowlisted count вариантов сохраняются;
ссылка ведёт в контекст результата с честной границей ещё не реализованного S5.

Native cancel dialog поддерживает Escape, удержание и восстановление фокуса.
Подтверждение заново проверяет subject и отправляет один POST. Успешный HTTP и
`cancel_requested_at` не означают `cancelled`. Pending/unknown outcome блокирует
повторную команду; последующие чтения устанавливают фактический terminal outcome.
409/429 требуют нового разрешённого GET перед следующей явной командой; GET429
также полностью соблюдает свой cooldown. Нет автоматических mutation retries.
Identity outage блокирует команды как недоступность сессии, без ложного logout.

Per-job cancellation marker сохраняется в private query cache при client-side
навигации и очищается существующей границей logout/subject. Reload восстанавливает
известный UUID через GET, без повторной команды; серверный cancel-request marker
восстанавливает pending. Долговечное локальное хранение cancel payload не вводится.
Create body/key recovery S3 и запрет replay не изменены.

`e2e/run.mjs` теперь разделяет foundation/library, builder и execution на отдельные
cold fixtures. `ROEHUB_PROOF_STAGE=S4` сохраняет регрессионные evidence отдельно от
принятых S1–S3. Локальная fixture допускает паузу только своего scheduler для
воспроизводимой queued cancellation; API, worker, БД, admission limits и transitions
остаются production implementations. Cleanup возобновляет остановленный scheduler.
Наблюдение короткого running-интервала использует read-only БД, не ручную правку
состояний. Фактические проверки и границы — в
[S4 report](../../../../.codex/delivery/evidence/roehub-backtests-client-v1/S4-execution-and-cancel.md).
S4 не подтверждает S5/results, general Jobs, target-role cutover или deployment.

## Результаты и сохранение стратегии platform client (S5, 2026-09-08)

`results.tsx` и `results-api.ts` подключают summary/top, URL-идентичность варианта,
серверные equity/drawdown, monthly/symbol statistics и trades (50 на страницу,
максимум 10000 страниц). График запрашивает 400 точек при серверном максимуме
1500; финансовые метрики не вычисляются в браузере. ECharts использует v23 tokens,
а каждый график имеет доступную таблицу. OHLC не создаётся. Отмена obsolete reads
и query identity защищают быстрые переключения job/variant/tab/page.

202 materialization и GET429 имеют ограниченный polling по server hints. Ошибка
определяет собственную политику: 401/403/404 останавливают чтения, transport/5xx
допускают явное обновление после указанной задержки; старый cached202 не запускает
polling и не блокирует manual recovery. CSV проверяет Content-Type и HTTP status:
202 JSON остаётся ожиданием, только 200 text/csv скачивается. UI сообщает реальные
row-count/total/max/truncated headers и ограничивает экспорт 100000 строками.

Сохранение стратегии использует bodyless POST source job/variant с отдельным
`Idempotency-Key`. Обязательные свежие readiness reads проверяют source identity.
Compatibility/readiness характеризует возможность торговли, а не допустимость
сохранения: `unsupported_live_evaluator` не запрещает immutable save в действующем
production use case. UI показывает локализованную причину и предупреждение перед
сохранением; run/start не вызывается. Ответ связывается с существующим SSR detail.

`strategy-recovery.ts` хранит только allowlisted per-tab source/key/subject/time
и известную identity. Серверная provenance-дедупликация не имеет TTL job-create.
Текущие current-user/save DTO не подтверждают организацию: replay остаётся
отключённым, unknown outcome сохраняет контекст и предлагает read-only history.
Known refusals требуют явного recheck; logout/subject change очищают recovery.
Недоступное session storage оставляет память с явным предупреждением.

`delete-history.tsx` допускает команду только для terminal job, повторно проверяет
session и удаляет выбранный контекст только после 204. Конфликт требует нового
GET с существующим deadline execution; ambiguous outcome не повторяет DELETE.
404 остаётся absent/not-visible, а не доказательством удаления. Сохраняются
единый S4 polling/cancel reconciliation и S3 create recovery.

Fixture включает штатные strategy CRUD и strategy dashboard composition roots
для доказательства SSR-перехода, без изменения production auth wiring. Runner
добавляет отдельную results fixture; `ROEHUB_PROOF_STAGE=S5` сохраняет предыдущие
регрессионные артефакты в новых каталогах. Фактическое доказательство и ограничения:
[S5 report](../../../../.codex/delivery/evidence/roehub-backtests-client-v1/S5-results-and-strategy.md).
Полное journey acceptance остаётся S6; SSR default, target roles и deployment
не изменены.

## Complete local Backtests verification (S6, 2026-09-08)

The [S6 completion report](../../../../.codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md)
owns the actual T1–T8 outcome and separates `local_journey_verified` from
`target_role_cutover_ready`. `e2e/journey.spec.ts` adds an integrated real
configure/preflight/create/worker/result/trades/CSV/save path and reads its actual
job, strategy and trades through flag-off SSR without migration or recreation.
The same saved strategy renders in the existing SSR detail; flag-on returns to
the same result identity. `e2e/run.mjs` isolates this fifth group with a cold fixture.
`ROEHUB_PROOF_STAGE=S6` keeps all producer evidence unchanged.

S6 aligns result429 manual refresh with the existing automatic deadline, including
missing and short retry hints. Other errors retain explicit manual recovery or a
restricted stop. Both charts and their accessible tables now describe quote
currency/percent and server UTC time/trade-index axes in RU/EN. Financial data and
production API/authz behavior are unchanged. Only local implemented_scope is
updated in the functional registry; broader UI/API dependencies remain.

`WEB_BACKTESTS_CLIENT_ENABLED` stays defaultfalse. Original-organization replay,
workstation extended filters, target roles, general Jobs/Artifacts/ingestion and
execution-detail work remain outside this completed implementation slice.
The old pack's entry/claim machinery was explicitly superseded by the user's
sequential execution instruction; its ledger and accepted prompts are unchanged.

## Текущие источники истины

| Граница | Канонический источник |
|---|---|
| Продуктовый результат и platform constraints | `docs/architecture/platform/roehub-product-transformation-requirements-v1.md` |
| Актуальное функциональное покрытие UI и оставшаяся работа | `roehub-ui-functional-contract-v1.md` и `roehub-ui-functional-registry-v1.json` |
| Принятая local-platform information architecture | `roehub-local-platform-information-architecture-v1.md` и её machine-readable registries |
| Роли, capabilities, маршруты и mutation policy | текущий код, API и принятые server contracts |
| Текущее browser-visible поведение | `apps/web/` и воспроизводимые browser evidence |
| Публичный сайт | public-site registry и `roehub-public-site-identity-release-and-measurement-v1.md` |
| Желаемый визуальный результат | принятый Backtests Workbench v23 и последующие явные решения пользователя |
| Объём реализации | выбранная пользователем задача и её критерии проверки |
| Публичный выпуск | SemVer из `pyproject.toml`, Git tag и GitHub Release |

Исторический документ, прототип или screenshot не становится текущей
authority только потому, что он хранится в репозитории. Текущий код также не
может молча переопределить желаемый пользователем продуктовый смысл.

## Источники для разработки

В пределах выбранной задачи используются:

- принятый product baseline;
- актуальный [функциональный контракт](roehub-ui-functional-contract-v1.md) и
  [реестр функций](roehub-ui-functional-registry-v1.json), включая восстановленные
  из истории требования; первая задача —
  [Backtests client](../../../../.codex/tickets/2026-09-08-roehub-backtests-client.md);
- local-platform information architecture, screen registry и access/route
  contract;
- public-site registry, если сайт входит в авторизованный scope;
- текущие route/API/DTO/runtime sources как evidence, а не как второй product
  owner;
- пользовательская задача и принятые визуальные решения;
- принятый пилот Backtests Workbench v23.

Технические документы ограничивают совместимость и описывают текущую
реализацию. Они не заменяют пользовательский замысел и не требуют создавать
новую программу проектирования. Визуальные требования удалённых исторических
документов не наследуются автоматически.

## Принятая визуальная база

Артефакт
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`
принят пользователем как пилот Backtests Workbench и сохраняется по этому пути.

- Его нельзя удалять или переписывать в рамках общей очистки старой UI
  документации.
- SHA-256: `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- Повторная приёмка пилота через отдельный процесс не требуется.
- Пилот задаёт визуальную базу, но не доказывает реализацию API, доступность
  серверных операций или полноту всех экранов продукта.

Старый `prototypes/roehub-v2/` не является этой защищённой базой. Его
историческое существование фиксируется Git history и не требует сохранения
живой копии в рабочем дереве.

## Граница design-to-code

Реализация выполняется в пределах выбранной пользователем задачи. Изменение
должно:

- покрывать выбранные screens, states, actions и critical journeys;
- сохранять server authority, same-origin boundary, CSRF/recent-auth,
  secret-redaction и unknown-result reconciliation;
- отделять visual fidelity от API, persistence, performance и runtime proof;
- определять responsive и accessibility evidence для изменяемого scope;
- ссылаться на реальные текущие paths и versioned contracts;
- не использовать mock или design surface как доказательство доступности
  операции.

Browser acceptance выполняется на реальном приложении с безопасными fixtures.
Source tests не доказывают browser behavior, а дизайн не доказывает server-side
authorization.

## Граница проверки

Завершение конкретной задачи определяется её объёмом и полученными
доказательствами. Принятый пилот не означает, что весь интерфейс реализован.
Проверка одного экрана не подтверждает работоспособность остальных экранов.

Изменения принятого визуального результата и информационной архитектуры
опираются на явные решения пользователя. Этот документ не назначает новый
процесс разработки.

## Связанные документы

- [`roehub-product-transformation-requirements-v1.md`](../../platform/roehub-product-transformation-requirements-v1.md)
- [`roehub-local-platform-information-architecture-v1.md`](roehub-local-platform-information-architecture-v1.md)
- [`roehub-local-platform-screen-registry-v1.json`](roehub-local-platform-screen-registry-v1.json)
- [`roehub-local-platform-access-and-route-contract-v1.json`](roehub-local-platform-access-and-route-contract-v1.json)
- [`roehub-ui-surface-inventory-v1.md`](roehub-ui-surface-inventory-v1.md)
- [`roehub-public-site-identity-release-and-measurement-v1.md`](../../platform/roehub-public-site-identity-release-and-measurement-v1.md)

## Как проверить

```bash
python -m tools.docs.generate_docs_index --check
python -m tools.docs.generate_project_map --check
pytest -q tests/unit/docs/test_roehub_ui_surface_inventory.py \
  tests/unit/docs/test_roehub_local_platform_information_architecture.py
git diff --check
```
