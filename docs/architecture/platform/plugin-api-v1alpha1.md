# Roehub Plugin API v1alpha1

## Статус и граница

`Plugin API v1alpha1` — публичная экспериментальная граница Stage `12` для
подписанных расширений. Сторонний Python/TypeScript-код не импортируется API,
worker или предметными контекстами Roehub. Backend плагина запускается только
как отдельный OCI-контейнер и взаимодействует через
`roehub.plugin.rpc/v1alpha1`.

Публичные типы первой версии: `data-source`, `panel`, `app` и
`notification-provider`. Marketplace, production install и произвольное
исполнение не входят в доказанную границу.

## Идентичности и источник истины

| Ресурс | Идентичность | Источник истины |
|---|---|---|
| ключ издателя | `installation_id + key_id` | operator-owned `PluginPublisherKeys/v1alpha1` + status mirror `extensions_publisher_keys` |
| пакет | `package_id`, `plugin_id`, SemVer и `package_digest` | `extensions_plugin_packages` |
| установка пакета | `plugin_installation_id + organization_id` | `extensions_plugin_installations` |
| экземпляр | `instance_id + organization_id` | `extensions_plugin_instances` |
| операция | `operation_id`, `organization_id + idempotency_key` | `extensions_plugin_operations` |
| событие аудита | `event_id` | неизменяемая `extensions_plugin_events` |

Пакет неизменяем. Конфигурация и выданные права принадлежат установке и
экземпляру, а не пакету. Обновление переключает установку на новый package id
и сохраняет `previous_package_id`; rollback меняет их местами, не переписывая
пакеты и завершённые операции.

## Подпись и совместимость

`roehub.plugin.yaml` проверяется JSON Schema Draft 2020-12. Канонический
`package_digest` — SHA-256 детерминированного JSON без detached signature
value. Ed25519 подписывает контекст
`roehub-plugin-package-v1alpha1` и digest. До регистрации проверяются:

- доверенный `keyId` и подпись;
- digest OCI-образа и SHA-256 конфигурационной схемы, license и SPDX SBOM;
- SPDX identifier;
- строгий SemVer и диапазон совместимости Roehub;
- `Plugin API v1alpha1` и `roehub.plugin.rpc/v1alpha1`;
- `linux/amd64` либо `linux/arm64`;
- ограниченный перечень capabilities и container runtime policy.

Изменение operator-owned trust file принадлежит только `installation_owner`.
При первой активации проверенный ключ атомарно записывается в installation-
scoped status mirror. Дальнейшая активация требует одновременно прежний
fingerprint в текущем trust file и `status=trusted` в PostgreSQL; несовпадение
или `revoked` закрывает операцию. Публичного admin route для добавления/отзыва
ключа в v1alpha1 нет: обычный organization `admin` не может расширить
доверенную границу.

Unsigned development mode отключён по умолчанию. Он требует одновременно
`metadata.developmentMode: true` и явного
`ROEHUB_PLUGIN_UNSIGNED_DEVELOPMENT=true`; validator и lifecycle service
отказываются включать его при `mainnet`.

## Управляющий API

API выводит организацию из аутентифицированного server-side principal и
использует канонический Identity ACL `plugins.manage`:

- `POST /api/v1/organizations/{organization_id}/plugins/bundles:validate`;
- `POST /api/v1/organizations/{organization_id}/plugins/installations`;
- `POST /api/v1/organizations/{organization_id}/plugins/installations/{plugin_id}:rollback`;
- `GET /api/v1/organizations/{organization_id}/plugins/operations/{operation_id}`.

Изменяющие cookie-auth запросы требуют same-origin CSRF check и
`Idempotency-Key`. Ответ install/update/rollback —
`202 PluginOperation/v1alpha1`; API не выполняет сторонний код. Повтор того же
ключа и request hash возвращает исходную операцию, конфликт payload получает
`plugin.idempotency_conflict`. Расширение permissions относительно текущей
установки требует `recent-auth` и всегда создаёт audit event, включая отказ.

Submission сохраняет полный проверенный request snapshot и его SHA-256.
Executor не принимает bundle, permissions или config повторно: он атомарно
переводит ровно одну `pending` operation в `running`, сверяет сохранённый hash
и выполняет только этот snapshot. Конкурентный claim получает
`plugin.operation_not_pending`.

Generic `/execute`, shell command, Docker arguments, mount или environment
payload в API отсутствуют.

## Сетевой контракт и служебная идентичность

Gateway вызывает только фиксированные capability endpoints:

- `POST /v1alpha1/data-source/query`;
- `POST /v1alpha1/panel/describe`;
- `POST /v1alpha1/app/action`;
- `POST /v1alpha1/notification-provider/send`;
- `GET /v1alpha1/health` и `GET /v1alpha1/metrics`.

Mutating capability требует отдельный idempotency key. Gateway выдаёт
Ed25519-подписанную `PluginServiceIdentity/v1alpha1` не более чем на 60 секунд.
Claims содержат organization, instance, package digest/version, одну
capability, время и одноразовый nonce. Container проверяет подпись, срок,
полный scope и однократное потребление nonce в рамках runtime process. Неверная
версия протокола закрывается, а timeout после возможного
принятия классифицируется gateway как `plugin.rpc_unknown`, без слепого
повтора.

## Контейнерная политика

Runtime container обязан иметь:

- непривилегированный uid не ниже `10000`;
- read-only root filesystem и отдельный ограниченный `tmpfs`;
- `cap-drop ALL` и `no-new-privileges`;
- CPU, memory и PID limits из подписанного manifest;
- только отдельную internal network с allowlisted egress;
- отсутствие Docker socket, platform database network, host mounts и raw
  credentials в environment.

Оркестратор сначала получает образ по `image_reference`, затем сравнивает его
content-addressed image id с подписанным `image_digest` и запускает контейнер
только по digest. Container inspection обязан повторно подтвердить то же поле
`Image`; mutable tag никогда не является execution identity.

Product configuration хранится в PostgreSQL и проверяется подписанной JSON
Schema. Секреты не являются config: plugin runtime получает только
краткоживущую identity и разрешает typed OpenBao reference внутри своей
доверенной границы.

## Отказы, наблюдаемость и восстановление

Operation имеет состояния `pending`, `running`, `succeeded`, `failed` и
`unknown`. `unknown` не означает безопасный повтор: оператор сначала сверяет
durable operation/event state и состояние контейнера. Health/metrics не
содержат raw response или секреты. Наблюдение фиксируется событием
`plugin.runtime.observed`; event rows неизменяемы на уровне PostgreSQL.

Rollback перед claim повторно проверяет signed/explicit-development mode,
текущий operator trust fingerprint, PostgreSQL `trusted` status и запрет
unsigned package для `mainnet`. Затем compare-and-set переключает установку на
точно сохранённый предыдущий package id и не
удаляет новый пакет, чтобы сохранить доказательство и возможность анализа.
Порядок восстановления описан в
`docs/runbooks/plugin-runtime-and-rollback.md`.

## Совместимость

| Измерение | Классификация | Обоснование |
|---|---|---|
| Manifest/RPC/API/SDK | `compatible-change` | Добавлены новые versioned v1alpha1 surfaces; прежнего публичного контракта не было. |
| Persistence | `compatible-change` | Миграция `0017` добавляет отдельные таблицы и не изменяет существующие строки. |
| Identity/RBAC | `compatible-change` | Переиспользуются существующие роли, `plugins.manage`, membership и recent-auth. |
| Config/defaults | `compatible-change` | Unsigned mode добавлен выключенным; mainnet конструктивно запрещён. |
| Secret boundary | `compatible-change` | Новых raw secret inputs нет; используется Stage `08` plugin scope. |
| Retry/idempotency | `compatible-change` | Добавлены scoped keys и `unknown`; существующие операции не меняются. |
| Trading/compute/browser | `none` | Формулы, исполнение и UI defaults не затронуты. |
