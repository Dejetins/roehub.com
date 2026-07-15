# Этап 12 — публичная платформа плагинов, API и SDK

## Статус

- Этап: `12`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовые PostgreSQL, внутренние Docker-сети,
  подписанные тестовые пакеты и изолированные контейнеры без production data,
  credentials и внешнего трафика.
- Public marketplace, production plugin install и arbitrary execution plugins
  не входят в доказанный объём.
- Следующий разрешённый этап: `13`.

## Результат

Создан ограниченный контекст `extensions` с версионированными контрактами
`roehub.plugin/v1alpha1`, `roehub.plugin.api/v1alpha1` и
`roehub.plugin.rpc/v1alpha1`. API-процесс не импортирует код плагина. Плагин
исполняется только как отдельный OCI-контейнер за фиксированным RPC-шлюзом.

Модель различает immutable package, organization-scoped installation,
instance, granted permission set, asynchronous operation и immutable audit
event. Установка, обновление, rollback, изменение разрешений, health и runtime
observations проходят через один lifecycle service и repository port.
Management API возвращает `202`, требует `Idempotency-Key` и не содержит
универсального `/execute` либо endpoint для изменения publisher trust.
Cookie-auth mutations дополнительно требуют fail-closed same-origin CSRF
check.

Submission сохраняет полный принятый request snapshot и его SHA-256. Executor
не принимает bundle, permissions, config или instance identity повторно:
единственный вход — `operation_id`. Переход `pending → running` выполняется
compare-and-set, а конкурентный identical submit возвращает исходную
operation. Изменённый persisted payload отклоняется до package/instance write.

В production wiring закрывается при отсутствии
`ROEHUB_PLUGIN_PUBLISHER_KEYS_FILE` или `EXTENSIONS_PG_DSN`. Unsigned mode по
умолчанию выключен, требует явного development manifest и
`ROEHUB_PLUGIN_UNSIGNED_DEVELOPMENT=true`, запрещён в `prod` и `mainnet`.

## Пакет, подпись и доверие

Строгий `roehub.plugin.yaml` проверяется Draft 2020-12 JSON Schema до записи и
до запуска. Валидатор проверяет:

1. SemVer пакета и диапазон совместимости Roehub;
2. `Plugin API` и RPC protocol version;
3. поддерживаемые `linux/amd64` и `linux/arm64` architectures;
4. digest OCI image и SHA-256 конфигурационной схемы, лицензии и SPDX SBOM;
5. SPDX identifier и базовый контракт SPDX `2.3`;
6. запрошенные capabilities и container runtime policy;
7. Ed25519-подпись canonical package digest доверенным publisher key.

Canonical digest строится из детерминированного JSON manifest без detached
signature value и подписывается с отдельным контекстом
`roehub-plugin-package-v1alpha1`. Publisher public keys принадлежат установке и
загружаются из версионированного read-only файла
`PluginPublisherKeys/v1alpha1`; при первой активации проверенный ключ
синхронизируется в installation-scoped PostgreSQL status mirror. Активация и
rollback требуют одновременно совпадения текущего operator-file fingerprint и
`status=trusted`; `revoked` закрывает операцию с audit. `admin` не может
изменить корень доверия через Plugin API.

## Разрешения, идентичность и изоляция

`admin` управляет плагинами только внутри организации, доступной через
Identity authorization adapter. Расширение granted permissions требует
`recent-auth`; сокращение разрешений не требует повторной аутентификации. Оба
случая создают audit event. Foreign-organization administration отклоняется.

Gateway выдаёт service identity сроком не более `60` секунд. Подписанные claims
фиксируют организацию, installation, instance, package digest и granted
capability. RPC authorizer одновременно проверяет identity, instance, package
digest/version, capability и однократный nonce; строковый произвольный dispatch
отсутствует, маршруты описаны versioned OpenAPI.

Runtime policy строится из уже проверенного manifest и не принимает
произвольные mounts или environment. Контейнер запускается:

- от `10001:10001`, с read-only root filesystem;
- с `no-new-privileges`, `cap-drop=ALL`, limits `0.5 CPU`, `128 MB`, `64 PID`;
- с отдельным `tmpfs` для `/tmp`, без Docker socket, host mounts и secret env;
- только во внутренней сети без внешнего egress и без сети PostgreSQL;
- с явным fail-close для заявленного egress, пока не настроен enforcing
  egress gateway.

Runtime сначала получает образ по reference, сверяет content-addressed image id
с подписанным digest и запускает контейнер только по digest. Container inspect
обязан подтвердить тот же `Image`; mutable tag не является execution identity.

## Хранение и миграция

Greenfield-миграция
`migrations/postgres/0017_extensions_plugin_platform_v1alpha1.sql` добавляет:

- `extensions_publisher_keys`;
- `extensions_plugin_packages`;
- `extensions_plugin_installations`;
- `extensions_plugin_instances`;
- `extensions_plugin_operations`;
- `extensions_plugin_events`.

Package хранится отдельно от installation и instance. Composite foreign keys
и repository queries сохраняют organization scope. Idempotency identity
уникальна внутри организации. Audit events защищены от `UPDATE` и `DELETE`.
Конфигурация проходит JSON Schema и отклоняет raw secret-shaped keys; секреты
передаются только как typed references в будущей runtime composition.

Rollback перед claim повторно проверяет signed/development mode, текущий trust
fingerprint, PostgreSQL status и mainnet policy. Затем compare-and-set меняет
ровно ожидаемые current/previous package identities, сохраняет instance и
configuration revision, создаёт отдельную asynchronous operation и audit
event. Миграция не импортирует и не исправляет текущие production rows.

Текущие checksums:

- файл `0017_extensions_plugin_platform_v1alpha1.sql`:
  `b1865fa9b49a1b4c3212796a2e615dcc33f5eaf41e380915066e4163dab7928c`;
- фаза `extensions-plugin-platform-0017`:
  `c6bf595aca5017e2b1a2c86f94ece6cc2c96c4041f53b5fecff34588ed787016`.

## Публичные контракты и инструменты

Добавлены:

- JSON Schema `schemas/plugins/roehub-plugin-manifest-v1alpha1.schema.json`;
- OpenAPI `schemas/plugins/plugin-rpc-v1alpha1.openapi.yaml`;
- conformance fixture `schemas/plugins/conformance/plugin-response.json`;
- Python SDK `sdk/python/roehub_plugin_sdk`;
- TypeScript SDK `sdk/typescript`;
- offline validator `python -m tools.plugins.validate`;
- CLI `roehubctl plugins init|validate|install|update|rollback|doctor`;
- архитектурный контракт `docs/architecture/platform/plugin-api-v1alpha1.md`;
- операторская инструкция `docs/runbooks/plugin-runtime-and-rollback.md`.

Python и TypeScript SDK используют один RPC version, один capability enum и
эквивалентные context/response identities. Conformance test сверяет эти поля в
обоих SDK, OpenAPI и fixture. API OpenAPI содержит `90` путей, среди которых нет
`/execute` и trust-management route.

## Реальная граница проверки

`uv run python -m apps.migrations.verify_storage_runtime` на Docker CLI
`29.6.1`, Engine `29.5.2` и Compose `5.3.1` подняла чистые PostgreSQL `16.14`,
ClickHouse `24.8.14.39` и Redis `7.2.14`. Прошли fresh bootstrap,
interruption/recovery, idempotent rerun, persistent-volume restart, external
readiness, новая фаза `extensions-plugin-platform-0017` и cleanup.

`uv run python -m tools.plugins.verify_runtime` собрала подписанный data-source
fixture, проверила bundle offline, установила package/instance в настоящую
PostgreSQL и запустила container через runtime policy. Доказаны:

1. валидная Ed25519 signature и раздельные package/instance;
2. tag после проверки переназначен на другой image, но container запущен по
   подписанному digest; inspect подтвердил точный `Image`;
3. последовательная и конкурентная management idempotency возвращает одну
   operation;
4. изменённый accepted request snapshot отклоняется по hash, повторный claim
   отклоняется CAS;
5. stale recent-auth и foreign-organization admin отклоняются;
6. publisher key безопасно создаётся из operator trust file, а rollback при
   `revoked` отклоняется;
7. service identity короткоживущая, проверяет полный scope/version и отклоняет
   replay nonce;
8. RPC protocol version согласован, незаявленная capability отклоняется;
9. запись в root filesystem запрещена;
10. PostgreSQL платформы недоступна, внешний egress запрещён;
11. health и metrics возвращают `ready`;
12. зафиксировано `10` audit events;
13. configuration revision достигла `2`, rollback восстановил предыдущий
    package;
14. OCI inspect подтвердил все ограничения runtime policy;
15. одноразовые контейнеры, plugin/storage images, сети и project volumes
    проверенно очищены.

Это real-boundary evidence: проверки исполняли настоящий подписанный fixture в
изолированном Docker container и production-class PostgreSQL constraints, а не
только mocks или статические тесты.

## Проверки качества

- Финальный целевой набор Stage `12` и storage manifest — `24 passed`;
  дополнительный узкий follow-up — `12 passed`.
- Полный `uv run pytest -q` — `1821 passed`, `4` существующих предупреждения
  `httpx` о будущем изменении per-request cookies.
- Полный `uv run ruff check .` — `passed`.
- Целевой `pyright` для Stage `12` — `0 errors, 0 warnings`.
- Полный `uv run pyright` дополнительно запускался, но не является gate этого
  этапа: сохранены `153` ошибки и `2` предупреждения в чужих
  `local_artifacts` и exchange cleanup tools; Stage `12` scope чист.
- Runtime input inventory generation и `--check` — `passed`, `145` имён без
  значений.
- Runbook generation и `--check` — `passed`, шесть русских инструкций и index.
- Project map `--check` — `passed`.
- Plugin API OpenAPI и CLI help для шести команд — `passed`.
- Docker storage lifecycle и signed plugin runtime proof — `passed` с cleanup.
- Docs index generation/`--check` и финальный `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Manifest, API, RPC и SDK | `compatible-change` | Добавлены новые versioned `v1alpha1` контракты без замены существующего публичного API. |
| DTO и application ports | `compatible-change` | Новый bounded context не меняет существующие notification/research/trading DTO и ports. |
| Persistence | `compatible-change` | `0017` добавляет только новые таблицы, ограничения и фазу clean bootstrap. |
| Config/defaults | `compatible-change` | Новые plugin inputs необязательны в development; production wiring закрывается без operator trust file и отдельной DB configuration. |
| Идентичность и hashes | `compatible-change` | Добавлены package digest, service identity и idempotency identities с явными версиями. |
| Межсервисные вызовы | `compatible-change` | Добавлен фиксированный API→extensions→gateway→plugin RPC flow; существующие вызовы не заменены. |
| Permissions/audit | `compatible-change` | Добавлены scoped capabilities, recent-auth expansion и immutable plugin audit. |
| Внешние эффекты | `none` | Только disposable local containers и искусственные строки; production и публичный marketplace не затронуты. |
| Compute/trading formulas | `none` | Торговые формулы и решения не менялись. |
| Browser-visible defaults | `none` | Пользовательский Web UI на этом этапе не добавлялся. |
| Runbook | `compatible-change` | Добавлена отдельная инструкция диагностики и rollback. |

Основная классификация Stage `12` — `compatible-change`. Контракт остаётся
`v1alpha1`; будущая несовместимая эволюция должна получить новую версию и
явную migration policy.

## Файлы этапа

Созданы `extensions` domain/application/adapters, plugin gateway, manifest/RPC
schemas, Python/TypeScript SDK, CLI/offline/runtime tools, signed fixture,
`0017`, focused tests, архитектурный контракт, runbook и этот отчёт. Изменены
API/CLI/storage composition, migration manifest/verifier, generated runtime
inventory, runbook index и тесты производных артефактов.

Чужие изменения в `.codex/PLANS.md`, `local_artifacts`, exchange cleanup tools,
предыдущих этапах и остальных dirty files сохранены. Staging, commit, push,
deploy, production data read и production mutation не выполнялись.

## Независимая проверка

Единственная проверка `independent subagent` завершилась вердиктом `Block`.
Она обнаружила два blocker и пять high/medium групп: mutable image tag не был
связан с signed digest; executor повторно принимал payload; trust file и DB не
имели product sync/revocation boundary; operation claim не был CAS; cookie-auth
mutations не имели CSRF; service identity не проверяла package version/replay;
rollback не перепроверял trust. Дополнительно отчёт переоценивал SDK
conformance и cleanup.

Все замечания исправлены локально. Runtime proof теперь намеренно переназначает
tag до запуска по digest, проверяет inspect, immutable accepted payload,
concurrent idempotency/CAS, trust bootstrap/revocation, полный identity scope и
replay. SDK test сверяет общий enum/context fields, а cleanup проверяет images,
networks и compose volumes. Второй независимый review не запускался. Актуальный
snapshot прошёл локальную холодную перепроверку, реальные Docker/PostgreSQL
proofs, `1821 passed`, полный ruff, целевой pyright, генераторы и diff gate;
итог после исправлений — `Release after fixes`.

## Передача этапу 13

Stage `12` принят. Этап `13` получает подписанный
package/instance lifecycle, `RoehubDataFrame`-совместимую capability boundary,
versioned RPC, SDK и изолированный data-source fixture как основание для
декларативных панелей и design system.
