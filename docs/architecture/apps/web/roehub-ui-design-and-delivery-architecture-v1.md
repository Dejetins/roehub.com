# Roehub — архитектура проектирования и поставки интерфейса v1

Документ сохраняет принятую целевую Web-архитектуру Roehub и связывает будущие
UI-программы с глобальным `ui-design-program` без наследования старых visual requirements.

## Статус и полномочия

- Статус: `accepted architecture`, обновлена после прекращения прежнего
  design-system/Penpot workflow.
- Документ сохраняет принятые границы продукта, источники фактов и условия
  передачи дизайна в реализацию.
- Документ не задаёт визуальный стиль, дизайн-систему, обязательный
  дизайн-инструмент, план реализации или разрешение на изменение кода,
  публикацию и развёртывание.
- Полные программы проектирования Roehub создаются и исполняются только через
  глобальный skill `ui-design-program`. Репозиторий не дублирует его G0–G6
  control plane собственным workflow.

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
| Желаемый визуальный результат | выбранный в G0 `visual_authority` текущей UI design program |
| Реализационный handoff | принятый G6 `implementation_handoff` текущей UI design program |
| Публичный выпуск | SemVer из `pyproject.toml`, Git tag и GitHub Release |

Исторический документ, прототип или screenshot не становится текущей
authority только потому, что он хранится в репозитории. Текущий код также не
может молча переопределить желаемый пользователем продуктовый смысл.

## Вход в новую UI design program

Перед G0 агент собирает один durable intent baseline из:

- принятого product baseline;
- local-platform information architecture, screen registry и access/route
  contract;
- public-site registry, если сайт входит в авторизованный scope;
- текущих route/API/DTO/runtime sources как evidence, а не как второй product
  owner;
- пользовательского brief и явно выбранного visual direction;
- пилотного или другого visual source, если пользователь выбирает его как
  основу.

G0 обязан зафиксировать purpose, users, outcomes, roles, journeys, included и
excluded scope, platform, responsive range, mobile authorization и ровно один
текущий `visual_authority`. Визуальные темы, tokens, component families,
spacing, typography и motion не наследуются из удалённых исторических
документов.

`ui-design-program` владеет atlas, journeys, families, waves, review boards,
hash-pinned provenance, owner receipts и G0–G6 transitions. Пользователю
показываются законченные визуальные checkpoints; технические hashes и ledger
mechanics остаются внутри control plane.

## Защищённая историческая база

Артефакт
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`
сохраняется как важная историческая база Backtests Workbench.

- Его нельзя удалять или переписывать в рамках общей очистки старой UI
  документации.
- Само наличие файла не делает его автоматически текущим `visual_authority`.
- Если будущий G0 выбирает его как accepted pilot или visual-language anchor,
  программа обязана привязать точный live path, SHA-256 и корректный evidence
  mode.

Старый `prototypes/roehub-v2/` не является этой защищённой базой. Его
историческое существование фиксируется Git history и не требует сохранения
живой копии в рабочем дереве.

## Граница design-to-code

Реализация может начаться только из отдельно авторизованного handoff или
delivery ticket. Handoff должен:

- exact-cover принятые screens, states, actions и critical journeys;
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

## Граница завершения

UI program считается готовой к реализации только после принятого G6 handoff.
Одна принятая screen family, component catalog, route list или pilot не
заменяет полный all-screen atlas и journey coverage.

Изменение принятого визуального результата после G3–G6 проходит через change
impact правила `ui-design-program`; принятая information architecture меняется
только отдельным продуктовым решением.

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
