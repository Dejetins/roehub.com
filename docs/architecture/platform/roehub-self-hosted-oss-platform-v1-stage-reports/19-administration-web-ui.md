# Stage 19 — отдельный административный Web UI

## Статус

- Этап: `19`.
- Состояние: `accepted`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A` — локальные синтетические пользователи и
  типизированный стенд API; production mutation, реальные поставщики,
  выпуск, резервное копирование, deploy и замена аварийного `roehubctl`
  исключены.
- Реальная граница: настоящий Chromium через Playwright открыл работающий
  локальный FastAPI/Jinja server, выполнил HTTP-запросы через same-origin proxy
  и прошёл role/mutation/reconcile/responsive/console scenarios. Это не
  tests-only принятие; синтетическими были только identities и данные стенда.

## Результат кандидата

Добавлена отдельная страница `/admin` в существующем FastAPI/Jinja shell. Она
сохраняет терминальную визуальную систему Roehub и разделяет установку,
организацию, плагины, поставщиков, резервные копии, обновления, сервисные
операции и аудит. Резервные копии до Stage `21` показываются как явно
отложенная возможность: интерфейс не отправляет фиктивную операцию.

Новая граница `/api/v1/admin` возвращает снимок без секретов и исходных данных
поставщиков, принимает только закрытый набор типизированных операций,
проверяет организацию, роль, `recent-auth`, same-origin и `Idempotency-Key`,
а также позволяет читать и сверять `unknown` через управляющий агент Stage
`18`. `operation_id` выводится из organization namespace и
`Idempotency-Key`; тот же ключ обязателен для polling/reconcile, поэтому
операцию нельзя прочитать через URL другой организации. Browser/API не
принимают Docker-команды, произвольное окружение или неограниченные журналы.

## Реализованные сценарии

- `owner` и `admin`: управление ролями и плагинами, разрешённые сервисные
  операции.
- `operator`: диагностика/restart и аудит без управления ролями/плагинами.
- Только реальный `installation_owner`: recovery и release
  install/update/rollback; организационные `admin`/`operator` не могут
  получить эту власть через членство.
- `trader` и `viewer`: административное чтение без привилегированных действий.
- Изменение роли: влияние, точная фраза подтверждения, `recent-auth`, защита
  `last_owner` и запрет назначения владельца администратором.
- Плагин: server-local `bundle_id`, проверка подписанного пакета, точная
  разница текущих/запрошенных/добавляемых/объявленных разрешений и отдельное
  подтверждение. Поля секретов отсутствуют.
- Операция: organization-bound `accepted`/`running`/terminal progress, явный
  `unknown` и ручная сверка до `succeeded` с тем же ключом идемпотентности.
- Поставщики: переход в существующий типизированный владеющий сценарий.
- Резервные копии: понятное `deferred`-состояние до Stage `21` без ложной
  готовности.
- Полные loading, empty, degraded, disabled, error и success состояния.

## Контрактное влияние

| Измерение | Классификация | Обоснование |
|---|---|---|
| API и DTO | `compatible-change` | Добавлены `/api/v1/admin/...`, `io.roehub.admin-snapshot/v1alpha1`, installation-owner indicator и новые read-only endpoints; operation reads требуют исходный `Idempotency-Key`; существующие response shapes вне нового API не изменены. |
| Browser mutation policy | `breaking-change` | Существующие organization mutations теперь fail closed без корректного same-origin `Origin`; браузерный shell совместим, не-browser clients должны передавать допустимый origin. |
| Порты и adapters | `breaking-change` | Внутренние `OrganizationRepository`, `PluginRepository` и `PluginAuthorization` расширены обязательными read/list методами; оба штатных adapters обновлены. |
| Хранение | `none` | Новых таблиц, миграций и persisted defaults нет. |
| Конфигурация | `none` | Новых runtime inputs и зависимостей нет. |
| Идентичность и RBAC | `compatible-change` | Канонические роли и permissions не расширялись; сервер отдельно доказывает `installation_owner`, `recent-auth`, `audit.read` и organization membership. |
| Идемпотентность и hashes | `compatible-change` | Новая admin-operation identity детерминированно выводится как UUIDv5 из organization и валидного `Idempotency-Key`; GET/reconcile повторно доказывают эту связь; существующие hashes не менялись. |
| Межсервисные вызовы | `compatible-change` | API использует только typed `submit/get/reconcile` клиента управляющего агента Stage `18`. |
| Внешние эффекты | `compatible-change` | Новая browser entrypoint может инициировать уже разрешённые операции; Docker остаётся только за локальной границей управляющего агента. |
| Аудит | `compatible-change` | Снимок объединяет redacted identity/plugin events; durable control audit остаётся владельцем Stage `18`. |
| Browser defaults | `compatible-change` | Добавлен пункт `Administration`; темы, локали и shell defaults сохранены. |

## Проверки

- `real browser boundary`: отдельный процесс Chromium против запущенного
  `uvicorn` и настоящих Web assets; typed HTTP `GET/POST/PATCH`, ожидаемый
  `409`, async `202`, polling и reconcile `200` проверены по network log.
- Сфокусированные API/Web/identity/extensions/operations тесты до review:
  `91 passed`, `3 warnings`; post-review authorization follow-up:
  `45 passed`, `3 warnings`.
- Полный `pytest` после исправлений: `1948 passed`, `4 warnings` за `82.09s`;
  предупреждения —
  существующая deprecation диагностика тестового `httpx cookies=`.
- `ruff` для API, Web, identity, extensions и Stage `19` fixtures/tests:
  `passed`.
- Targeted `pyright`: `0 errors, 0 warnings`.
- `node --check apps/web/dist/js/pages/admin.js`: `passed`.
- Browser role matrix: `owner` — 6 operation controls; `admin` и `operator` —
  только diagnostics/restart; `trader` и `viewer` — 0; release/recovery
  выключены без installation ownership.
- `recent-auth` success/failure, `last_owner` HTTP `409`, plugin permission
  diff, confirmation/cancel focus, `unknown` → reconcile → `succeeded` —
  `passed`.
- Console: `0 errors, 0 warnings` для штатных сценариев; единственный failed
  network request — ожидаемый и отображённый `409 last_owner` в отрицательном
  сценарии.
- Post-review real-browser follow-up: audit скрыт без `audit.read`; одинаковый
  organization-bound `Idempotency-Key` фактически присутствовал в
  submit/poll/reconcile; `unknown` завершился `succeeded`; console
  `0 errors, 0 warnings`.
- Redaction surface: `0` password inputs, `0` secret/token/credential-named
  inputs, `0` executable inline scripts.
- Клавиатура: skip-link получает первый focus, `Enter` переводит focus в
  `#main-content`, далее доступен `Reload`; native dialog возвращает control
  после отмены.
- Responsive: `375`, `768`, `1024`, `1440` без document overflow; широкие
  таблицы имеют локальный прокручиваемый контейнер.
- Тёмная `graphite`, светлая `paper`, русский каталог и
  `prefers-reduced-motion: reduce` прошли.
- Browser evidence:
  `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/evidence/19-admin-browser-proof.json`.

## Исправления по реальному браузеру

1. Кнопка отмены в privileged confirmation первоначально попадала в общий
   submit handler и не закрывала native dialog. Обработчик разделён по
   `submitter.value`; повторная проверка прошла.
2. HTML `pattern` для `bundle_id` был несовместим с RegExp `v` и создавал
   console error. Дефис экранирован; повторная консоль чистая.

## Независимая проверка

- Режим: единственный `independent subagent`.
- Первоначальный вердикт: `Block`.
- Исправленные блокирующие замечания:
  1. release/recovery были доступны через обычное организационное
     `operations.execute`; теперь требуют реального `installation_owner` и
     остаются disabled в Web UI для `admin`/`operator`;
  2. GET/reconcile проверяли только path organization; теперь `operation_id`
     повторно вычисляется из organization namespace и обязательного
     `Idempotency-Key`, cross-organization namespace получает `404`;
  3. plugin events возвращались без `audit.read`; теперь snapshot фильтрует
     оба источника административного аудита;
  4. plugin update без permission expansion и rollback полагались на
     disabled browser control; теперь каждый lifecycle submit серверно
     требует `recent-auth` и записывает rejected event.
- Локальная повторная проверка после замечаний: `Release after fixes` —
  `45 passed`, targeted pyright `0`, ruff/node `passed`, полный набор
  `1948 passed`, реальный Chromium role/operation follow-up и clean console.
- Остаточные риски: backup/restore остаются отложены до Stage `21`; реальная
  production authentication и реальные административные эффекты сознательно
  не входят в `N/A` proof boundary.

## Запрещённые действия

Production/current data, реальные учётные данные и заказы не читались и не
изменялись. Commit, push, deploy, staging и production mutation не выполнялись.
