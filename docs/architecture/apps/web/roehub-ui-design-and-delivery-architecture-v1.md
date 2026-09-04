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
- Новый процесс разработки пока не выбран. Прежние этапы, журналы и
  управляющие артефакты не являются условием начала UI-работы.

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

## Текущие источники истины

| Граница | Канонический источник |
|---|---|
| Продуктовый результат и platform constraints | `docs/architecture/platform/roehub-product-transformation-requirements-v1.md` |
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
