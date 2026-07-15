# Источники данных, декларативные панели и визуальная система v1

## Статус и область

- Контракт: `RoehubDataFrame/v1`, `DataSourceQuery/v1`,
  `RoehubPanelContribution/v1`, `RoehubAppContribution/v1`.
- Статус: принят Stage `13` для self-hosted OSS platform v1.
- Владелец исполнения: Roehub host; плагин остаётся изолированным процессом.
- Граница доказательств: `N/A`; одноразовые контейнеры, искусственная внешняя
  PostgreSQL и QA-only Web route без production data или credentials.
- Не входят: произвольный JavaScript плагина в основном origin, анонимный
  Grafana embed, production dashboard replacement и library-specific public API.

## Решение

Плагин источника данных не получает DOM и не рисует пользовательский интерфейс.
Он принимает ограниченный read-only query через подписанный plugin RPC и
возвращает библиотечно-независимый `RoehubDataFrame/v1`. Roehub проверяет
организацию, capability, типы, выбранные поля и бюджеты до того, как передать
данные host render adapter.

Панель и приложение являются только декларациями. Их Pydantic-модели и JSON
Schemas не содержат полей для HTML, JavaScript, CSS, URL скрипта или
исполняемого callback. Layout, design tokens, локализация, доступность, состояния
и визуализация принадлежат Roehub.

## Поток данных и границы доверия

1. Browser отправляет `DataSourceQuery/v1` на фиксированный endpoint
   `/api/v1/plugins/data-sources/{instance_id}:query`.
2. Payload не содержит `organization_id`, tenant либо другой authority.
3. Identity adapter выводит ровно одну организацию с `plugins.read` из текущей
   server-side сессии. Ноль организаций даёт `forbidden`, несколько —
   `ambiguous_scope`; client-provided organization не рассматривается.
4. Application service сверяет organization у instance, installation и
   package, статус `enabled`, тип `data-source` и granted `data.read`.
5. Gateway вызывает только специализированный data-source RPC. Organization,
   instance, package digest/version и capabilities передаются в короткоживущей
   service identity Stage `12`, а не в query body.
6. Plugin читает разрешённый внешний источник и возвращает
   `PluginResponse/v1alpha1` с ровно одним `RoehubDataFrame/v1`.
7. Roehub повторно валидирует схему, набор полей, количество строк, байты,
   точки, partial status и redaction. Dimension/measure roles обязаны совпадать
   с query, а panel references — только с выбранными query fields. После этого
   host renderer строит визуализацию и эквивалентную таблицу.

Browser никогда не получает database DSN, publisher/gateway key, OpenBao
reference, authorization header или plugin container address. Audit хранит
только dataset, количество строк/точек и partial flag.

## `DataSourceQuery/v1`

Контракт содержит dataset, уникальные dimensions/measures, ограниченные
filters и четыре явных бюджета. `read_only` имеет единственное допустимое
значение `true`.

| Бюджет | Значение по умолчанию | Жёсткий максимум |
|---|---:|---:|
| Строки | `200` | `1000` |
| Ответ | `262144` байт | `1048576` байт |
| Точки | `1000` | `5000` |
| Время | `3000` мс | `5000` мс |

Внешний timeout оборачивает весь gateway call через `asyncio.timeout`. При
истечении срока coroutine отменяется; plugin fixture дополнительно применяет
PostgreSQL `statement_timeout` и закрывает connection. Host и gateway читают
HTTP response потоково и прекращают чтение сразу после превышения запрошенного
byte budget плюс ограниченного protocol envelope. Полностью буферизовать
неограниченный ответ до проверки запрещено. Ответ, превысивший любой
запрошенный бюджет, отклоняется целиком.

## `RoehubDataFrame/v1`

Frame фиксирует:

- `frame_id`, человекочитаемые title и source/query labels;
- уникальные columns с portable key, label, data type, dimension/measure role,
  unit, scale и nullable policy;
- rows, содержащие ровно объявленные columns;
- timezone-aware `generated_at` и `observed_at`, freshness и age targets;
- ограниченные metadata attributes, notices, partial status и bounded errors.

Поддерживаются `timestamp`, `number`, `integer`, `string`, `boolean` и `null` по
nullable policy. Timestamp обязан иметь явную timezone и unit `timestamp`.
`NaN`, `Infinity`, невалидные timestamp и secret-shaped keys отклоняются.
Перед API response token-like redaction рекурсивно применяется к title, labels,
unit symbols, строковым row values, metadata, notices и bounded errors.

Контракт намеренно не кодирует Pandas, Arrow, Vega, Chart.js либо другую
реализацию. Канонические JSON Schemas живут в `schemas/plugins/`, а Python model
— в `src/trading/integration/data_frame.py`.

## Декларативные contributions

`RoehubPanelContribution/v1` выбирает один из host adapters:

- `trading-time-series` — временные торговые ряды с раздельными unit lanes;
- `analytics-series` — общий временной/аналитический ряд;
- `analytics-table` — табличное представление без графика;
- `research-summary` — host-owned summary с таблицей.

Presentation содержит только ссылки на x/y/table columns и default view.
Series adapters требуют x и хотя бы один y column. Pydantic является
исполняемой semantic authority: x обязан быть выбранной dimension, y —
выбранными measures, а table columns — выбранными query fields. JSON Schemas
повторяют структурные ограничения и явно ссылаются на эти межполевые правила.
Списки уникальны и имеют ограниченную длину. `RoehubAppContribution/v1`
группирует panel contribution IDs по уникальным sections, не встраивая сами
данные или код.

Новая версия контракта обязательна, если потребуется несовместимое поле,
другой trust model либо executable contribution. Последнее не может быть
добавлено в `v1`; для него нужна отдельная origin/sandbox architecture и новый
security review.

## Host-owned визуализация и доступность

Лаборатория `/__qa/plugin-panels` выключена по умолчанию и доступна только при
`ROEHUB_PLUGIN_PANEL_LAB=true`. Она не заменяет production dashboard. Route
использует существующие Roehub tokens, shell, themes и locale catalog.

Host renderer обязан поддерживать:

- `loading`, `empty`, `error`, `degraded`, `partial` и `success`;
- `aria-busy`, live status/alert и заметный focus outline;
- SVG points с `role=button`, описательным `aria-label`, Enter/Space drill-down;
- единицы в badges, scale labels и table headers;
- всегда доступную table alternative;
- внутреннюю прокрутку таблицы без page-level overflow;
- светлую и тёмную темы, а также `prefers-reduced-motion: reduce`.

У валюты и процентов разные вертикальные lanes: значения разных единиц не
показываются на общей вводящей в заблуждение шкале. SVG viewBox вычисляется из
реальной ширины панели после раскрытия content, поэтому подписи остаются
читаемыми на `375` пикселях.

## Сбои и безопасное поведение

| Сбой | Поведение |
|---|---|
| Нет либо неоднозначна session organization | Запрос отклоняется до repository/plugin call. |
| Foreign organization instance | Возвращается одинаковый `data_source.not_found`, данные не раскрываются. |
| Capability/package/status не подходят | `data_source.capability_forbidden`. |
| Gateway timeout | Coroutine отменяется, `data_source.query_timeout`. |
| Невалидная схема/partial mismatch | `data_source.response_invalid`. |
| Лишние columns | `data_source.response_fields_mismatch`. |
| Превышены rows/bytes/points | `data_source.response_too_large`. |
| Источник degraded/partial | Данные показываются с предупреждением и table alternative. |
| Источник error | Предыдущие rows очищаются, content скрывается, отображается alert. |

## Реальная граница проверки

Одноразовая Stage `13` fixture создаёт две случайные организации, отдельную
PostgreSQL `16`, read-only роль с `SELECT` и digest-bound OCI plugin. Плагин
работает с read-only root filesystem, без capabilities, с no-new-privileges,
CPU/memory/PID limits и двумя разделёнными сетями. Proof проходит через
реальные API error handlers, `DataSourceQueryService`, HTTP gateway adapter,
`PluginRpcClient`, plugin container и внешнюю PostgreSQL. Он проверяет signed
bundle, image digest, запрет записи, session-scope denial, row/stream-byte
limits, cooperative и non-cooperative timeout/cancellation, cross-org denial,
declarative contributions и точный cleanup контейнеров/сетей/тома/образа.

Browser proof использует только disposable user, искусственные строки и
same-origin API mock. Он проверяет четыре ширины, все состояния/adapters,
focus/keyboard, темы, reduced motion, единицы/таблицу и отсутствие неожиданных
console/network failures. Это UI-доказательство дополняет, но не подменяет и не
изображает единый browser-to-signed-plugin end-to-end trace; реальная
server/plugin/database цепочка проверяется отдельной контейнерной fixture.

## Совместимость

Основная классификация — `compatible-change`: добавлены новые versioned
контракты и API endpoint, существующие dashboard DTO, persistence schemas,
торговые формулы и browser-visible defaults не заменены. QA route disabled by
default. Добавление `ROEHUB_PLUGIN_GATEWAY_URL`,
`ROEHUB_PLUGIN_PANEL_LAB` и `ROEHUB_PLUGIN_PANEL_LAB_INSTANCE_ID` является
необязательным config extension; production wiring по-прежнему закрывается на
обязательных trust/storage inputs Stage `12`.
