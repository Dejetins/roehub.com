# Этап 13 — источники данных, панели и визуальная система

## Статус

- Этап: `13`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; disposable PostgreSQL/OCI/browser fixtures без
  production data, credentials и внешнего трафика.
- Исключены: arbitrary same-origin plugin JavaScript и production dashboard
  replacement.
- Следующий разрешённый этап: `14`.

## Результат

Добавлены публичные библиотечно-независимые контракты
`RoehubDataFrame/v1`, `DataSourceQuery/v1`,
`RoehubPanelContribution/v1` и `RoehubAppContribution/v1` с Pydantic-моделями и
JSON Schemas. Frame фиксирует types/roles/units/labels, metadata, freshness,
notices, partial state и bounded errors. Невалидные timezone, non-finite numbers,
secret-shaped fields, неоднозначные declarations и лишние properties
отклоняются.

Data-source query service выводит installation/organization только из session
authorization. Клиент не может передать organization authority. До plugin RPC
проверяются organization всех сущностей, enabled status, package type и
`data.read`. После RPC заново проверяются schema, exact fields, column roles,
rows, bytes, points и partial status. HTTP body читается потоково и обрывается
при превышении запрошенного byte budget плюс ограниченного protocol envelope.
Redaction покрывает labels, units, строковые rows, metadata, notices и errors;
audit содержит только агрегаты.

Добавлены специализированные API/gateway adapters и endpoint
`/api/v1/plugins/data-sources/{instance_id}:query`. Timeout отменяет coroutine;
query всегда read-only и имеет максимумы `1000` rows, `1048576` bytes, `5000`
points и `5000` ms.

## Декларативный Web UI

Панель и приложение не способны представить HTML, script URL либо callback.
Pydantic проверяет межполевой контракт: x выбирается только из query dimensions,
y — из query measures, table columns — из выбранных query fields; JSON Schemas
фиксируют структурные ограничения и явно документируют semantic authority.
Roehub host реализует adapters для trading series, general analytics,
analytics table и research summary. QA-only route `/__qa/plugin-panels`
выключен по умолчанию, использует текущий shell/design tokens/locales и не
изменяет production dashboard.

Панель покрывает loading/empty/error/degraded/partial/success, выводит units,
раздельные шкалы USD/% и всегда предоставляет таблицу. SVG points доступны по
Tab, имеют focus outline и активируются Enter/Space с drill-down. Ошибка очищает
старые rows. Reduced motion полностью отключает spinner animation.

## Реальная граница проверки

`tests/fixtures/plugins/external_db_data_source/proof.py` создал:

- две случайные организации и искусственные portfolio rows;
- отдельную PostgreSQL `16` с ролью `stage13_reader`, default read-only и только
  `SELECT`;
- подписанный bundle и digest-bound OCI image;
- изолированный non-root container с read-only root filesystem,
  no-new-privileges, dropped capabilities и resource/PID limits;
- отдельные data/gateway networks без общего публичного egress.

Proof прошёл через реальные API error handlers, `DataSourceQueryService`, HTTP
gateway adapter, `PluginRpcClient`, изолированный plugin container и внешнюю
PostgreSQL. Итог:

`{"api_service_gateway_plugin_chain":"passed","cleanup":"passed","declarative_contributions":"passed","external_database":"passed","image_digest_binding":"passed","non_cooperative_timeout":"passed","read_only_role":"passed","row_limit":"passed","schema":"io.roehub.data-source-panel-proof/v1","session_scope_denial":"passed","signed_bundle":"passed","status":"passed","stream_byte_limit":"passed","timeout_cancellation":"passed","two_organization_isolation":"passed"}`

Никакие production базы, secrets, конфигурации или пользовательские данные не
читались и не менялись.

## Браузерное доказательство

Изолированная FastAPI fixture использовала disposable user, искусственный
`RoehubDataFrame/v1` и same-origin API mock. Это доказательство UI-состояний, а
не единый browser-to-signed-plugin end-to-end trace; реальная server/plugin/DB
цепочка доказана отдельной контейнерной fixture. Screenshots сохранены как
локальные игнорируемые evidence artifacts в
`output/playwright/stage13-plugin-panels/`.

- `375`, `768`, `1024`, `1440`: document width совпал с viewport, page overflow
  отсутствует, table остаётся внутри панели, chart viewBox равен фактической
  ширине.
- `loading`: `aria-busy=true`; normal spinner `0.75s`, reduced motion —
  `animation-name: none`, `0s`.
- `empty`: content скрыт, `0` rows; `degraded` и `partial`: content доступен,
  `12` rows и одно предупреждение; `success`: `12` rows без предупреждений;
  `error`: alert с контролируемым сообщением, content скрыт, старые rows очищены.
- Trading/analytics series: `24` keyboard-focusable points и отдельные labels
  `Portfolio PnL (USD)`/`Drawdown (%)`; table/research adapters не создают SVG
  points и сохраняют `12` table rows.
- Tab переводит focus на SVG point с `rgb(138, 216, 242) solid 2px`; Enter
  обновляет drill-down.
- Реальным theme selector проверены `paper` и `graphite`; body colors изменились
  со светлых `rgb(244, 245, 246)`/`rgb(52, 67, 76)` на тёмные
  `rgb(8, 16, 24)`/`rgb(213, 222, 230)`.
- Все `61` зафиксированных fixture requests имели `read_only=true` и не
  содержали organization authority.
- Чистый success reload: `0` console warnings/errors, `0` failed requests и
  `0` HTTP responses `>=400`. Отдельно вызванный error state ожидаемо дал один
  browser console resource error для контролируемого `503`.

## Проверки качества

- Node syntax для panel component, page module и общего API helper — `passed`.
- Draft 2020-12 JSON Schemas и OpenAPI YAML parse — `passed`.
- Целевой `ruff` — `passed`.
- Целевой `pyright` по Stage `13` — `0 errors, 0 warnings`.
- Целевой pytest API/Web/contracts/data-source — `54 passed`, три существующих
  `httpx` warnings.
- Полный `uv run ruff check .` — `passed`.
- Первый полный pytest выявил только stale runtime input inventory:
  `1834 passed, 1 failed`; inventory регенерирован до `146` без значений, его
  check и focused test прошли. Финальный полный pytest — `1840 passed`, четыре
  существующих `httpx` warnings.
- Полный `uv run pyright` не является gate этапа: после исправления Stage `13`
  scope остаются прежние `153 errors, 2 warnings` только в чужих
  `local_artifacts` и exchange cleanup tools.
- Реальный external database/container proof — `passed` с cleanup.
- Browser matrix и accessibility smoke — `passed`.
- Docs index generation/`--check` и project map generation/`--check` — `passed`.
- Единственная независимая проверка дала `Block`; все обязательные замечания
  исправлены локально без второго независимого review.
- Холодная локальная перепроверка после исправлений — `Release after fixes`.
- Финальный `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| DataFrame/query/panel/app | `compatible-change` | Добавлены versioned `v1` контракты без замены существующих DTO. |
| Plugin RPC/OpenAPI | `compatible-change` | Добавлен специализированный bounded query вместо универсального execute. |
| Application ports | `compatible-change` | Добавлены data-source authorization/invoker ports и repository lookup. |
| Persistence | `none` | Новых таблиц и миграций нет. |
| Config/defaults | `compatible-change` | Gateway URL и QA lab inputs additive; lab выключен по умолчанию. |
| Identity | `compatible-change` | Organization выводится из существующей session/RBAC authority. |
| Browser defaults | `none` | Production dashboard не заменён; новая route disabled by default. |
| Request/hash identity | `none` | Stage `12` package/service identities не менялись. |
| Trading formulas | `none` | Торговые вычисления не менялись. |
| Внешние эффекты | `none` | Только disposable fixtures, полностью удалённые после proof. |

На независимой проверке классификация временно была `unknown` из-за дрейфа
schemas/runtime semantics. После синхронизации Pydantic, JSON Schema и OpenAPI,
а также поведенческих schema tests основная классификация подтверждена как
`compatible-change`. Несовместимая эволюция требует новой версии. Executable
same-origin contributions не допускаются этой версией.

## Файлы этапа

Созданы DataFrame/contribution contracts и schemas, data-source application/API
adapters, host panel component/QA route, external DB/browser fixtures, focused
tests, архитектурный контракт и этот отчёт. Изменены extensions ports/repository,
plugin RPC OpenAPI, API/Web wiring, locales, runtime input inventory и
производные архитектурные индексы.

Три новых файла `apps/web/dist` явно возвращены из общего `dist/` ignore через
узкие исключения `.gitignore`; staging для этого не выполнялся. Чужие dirty
изменения сохранены. Commit, push, deploy, staging и production mutation не
выполнялись.

## Независимая проверка

- Режим: ровно одна cold independent review.
- Первоначальный вердикт: `Block`.
- Блокирующие замечания: полная буферизация ответа до byte check; обход point
  limit через ложные column roles; redaction только notices/errors; runtime
  proof в обход реальных API/service/gateway границ.
- Существенные замечания: дрейф JSON Schemas/OpenAPI/Pydantic; panel fields без
  связи с query; browser mock не должен выдаваться за signed-plugin E2E;
  cleanup должен явно проверять OCI image.
- Исправлено: потоковые hard caps в обоих HTTP-клиентах; server-side role и
  point validation; глубокая redaction; semantic panel validation и
  поведенческие schema tests; реальная цепочка
  `API → service → gateway → plugin → PostgreSQL`; явный image cleanup.
- Повторная независимая проверка не запускалась. Холодная локальная
  перепроверка всех замечаний и повторный runtime proof дали
  `Release after fixes`.
- Остаточная граница: browser fixture доказывает host-owned UI отдельно от
  контейнерной server/plugin/database цепочки; единый browser-to-plugin trace в
  Stage `13` не заявляется.
