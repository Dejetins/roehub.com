# Хранилище артефактов v1

## Статус и область

- Контракты: `ArtifactStore/v1`, `ArtifactManifest/v1`, `ArtifactBackup/v1`.
- Статус: принят Stage `14` self-hosted OSS platform v1.
- Реализация по умолчанию: `local_cas` во внешнем долговечном каталоге хоста.
- Альтернативная реализация: path-style S3-compatible storage за тем же портом.
- Каталог, владение, квоты, закрепления и аренды: PostgreSQL.
- Секреты S3: только typed OpenBao reference `storage`; разрешённое значение
  живёт внутри доверенной границы адаптера и не сериализуется.
- Не входят: импорт текущего корпуса RL/backtest, чтение production-каталогов,
  HTTP API хранилища и замена существующих path-shaped потребителей до их
  отдельных этапов.

## Решение

`ArtifactStore/v1` разделяет неизменяемые байты и изменяемый каталог:

1. байты адресуются только `sha256:<64 lowercase hex>` и лежат вне контейнера;
2. `ArtifactManifest/v1` связывает нормализованные относительные пути с
   digest, размером и media type;
3. манифест подписан Ed25519 доверенным издателем и проверяется при каждой
   публикации, включая восстановление из резервной копии;
4. PostgreSQL хранит организационное владение, манифесты, квоты, закрепления,
   аренды и кандидатов на сборку мусора;
5. compute consumer получает локальный read-only путь через материализацию и
   не зависит от исходного layout артефакта.

Публичные Pydantic-модели не зависят от библиотеки хранения. Канонические JSON
Schemas находятся в `schemas/artifacts/`, а Python-контракт — в
`src/trading/integration/artifact_store.py`.

Буферизованный v1 ограничивает один blob и суммарный signed bundle значением
`67108864` байт. Это доказанный fail-closed предел для CLI/backup/S3 read, а не
обещание 1 ТиБ. Большие модели обязаны публиковаться manifest из ограниченных
shards; переход к streaming/multipart contract потребует отдельной версии.

## Идентичность и неизменяемость

Digest вычисляется по точным байтам blob. `manifest_digest` вычисляется по
каноническому JSON без поля `signature`; подпись Ed25519 покрывает те же
канонические байты. `bundle_id + version` уникальны внутри организации, а одна
и та же пара не может незаметно указывать на другой манифест.

Канонический unsigned JSON v1 — UTF-8 без BOM и перевода строки, object keys в
лексикографическом порядке Unicode code points, разделители `,` и `:` без
пробелов, JSON escaping без ASCII-транслитерации. `created_at` нормализуется в
UTC с `Z`; metadata допускает только строки, portable integers в диапазоне
`±(2^53-1)` и booleans. Float, `NaN` и `Infinity` запрещены. Репозиторий хранит
golden bytes подписываемого demo payload для межъязыковой сверки.

Метаданные манифеста ограничены portable keys и отклоняют ключи, похожие на
token, password, credential, authorization или secret. Entry path обязан быть
нормализованным относительным POSIX path без `..`, абсолютного пути, обратной
косой черты и каталога вместо файла. Исполняемые payload в v1 запрещены.

Сервис создаётся с одним installation-controlled набором доверенных публичных
ключей. Метод установки не принимает второй набор ключей, поэтому caller не
может подменить trust root. Прямая публикация и restore повторно проверяют
подпись и наличие каждого blob.

## Локальный CAS

`LocalCasBlobStore` размещает объект как
`blobs/sha256/<prefix>/<digest>`. Публикация выполняется через уникальный
incoming-файл, `fsync`, проверку SHA-256 и атомарный hard link. Победитель
конкурентной публикации создаёт canonical inode, остальные проверяют уже
существующий объект. Symlink не открывается благодаря `O_NOFOLLOW`.

Canonical blob и материализованные hard links имеют mode `0440`. Это делает
обычный compute path read-only; Stage `15` дополнительно обязан предоставить
consumer отдельную непривилегированную identity и read-only mount. Перед
чтением проверяется digest. Повторная проверка может использовать fingerprint
`device/inode/size/mtime/ctime`; изменение файла сбрасывает быстрый путь и
обнаруживается как `artifact.digest_mismatch`.

Материализация использует hard link на том же filesystem и потоковую копию с
`fsync` как fallback между filesystem. Cache namespace — SHA-256 от явного
`cache_key`; пользовательская строка не становится частью пути. Удаление blob
удаляет canonical link и все локальные materialization links этого digest.

## S3-compatible adapter

S3-адаптер использует path-style requests и AWS Signature Version 4. Endpoint,
bucket и region являются обычной конфигурацией, а access key и secret key
поступают только после разрешения OpenBao reference типа `storage`.
`S3ResolvedCredentials.__repr__` всегда возвращает redacted представление.
Runtime proof сохраняет искусственную пару в настоящий digest-bound OpenBao,
разрешает её через `OpenBaoSecretResolver` из mode-`0600` service credential
file и только затем создаёт S3 adapter; raw значение в evidence не выводится.

Каждая запись проверяется контрольным чтением, каждое чтение — SHA-256.
Redirects отключены, timeout ограничен 30 секундами. Plain HTTP разрешён только
для `127.0.0.1`, `localhost` и `minio`; удалённый endpoint обязан использовать
HTTPS. Материализация S3 также атомарна и read-only.

## Каталог, владение и квоты

Миграция `0018_artifact_store_v1.sql` добавляет:

- глобальный immutable object catalog;
- физические locations по паре `digest + backend`;
- organization-to-blob ownership;
- organization-scoped manifests и entries;
- отдельную квоту организации;
- pins и time-bounded leases;
- durable GC candidates.

Foreign keys ведут к `identity_organizations`. Manifest entry может ссылаться
только на blob той же организации. Reads всегда требуют `organization_id` и
не находят foreign manifest. Квота считается по уникальным digest организации,
поэтому повторная ссылка не удваивает usage, а межорганизационное физическое
dedupe не даёт межорганизационного чтения.

Публикация каталога сериализуется PostgreSQL advisory lock организации и
выполняется одной транзакцией. До физической записи сервис durable-регистрирует
`digest + size + backend`; поэтому crash и quota rollback не создают невидимый
файл. Превышение квоты откатывает ownership, manifest и entries, после чего
install немедленно запускает orphan cleanup. GC также собирает зарегистрированную
запись, если процесс прервался между регистрацией и записью bytes.

Глобальная content identity содержит только digest и size. `media_type` остаётся
на manifest entry, а backend — на location/ownership. Поэтому первая
организация не может заблокировать другой организации те же байты иным
допустимым media type или storage backend.

## Закрепления, аренды и сборка мусора

Manifest удерживает blobs через entries. После retirement blob остаётся
защищён, пока существует organization pin или непросроченная lease. Сборщик:

1. удаляет просроченные leases;
2. снимает organization ownership только с blob без manifest, pin и активной
   lease;
3. сохраняет durable GC candidate;
4. удаляет физический object;
5. подтверждает удаление metadata, либо отменяет candidate, если ownership
   появился повторно.

GC идемпотентен к отсутствующему физическому object. В v1 его запускает один
операционный исполнитель установки; Stage `18` обязан сериализовать GC
относительно всех publishers в `control-agent`, а не только выбрать одного
GC-лидера. Конкурентная публикация
одинакового bundle доказана, но одновременный publish того же digest и GC не
заявляется как доказанная распределённая граница Stage `14`.

## Резервное копирование и восстановление

`ArtifactBackup/v1` — строгая Pydantic-модель и Draft 2020-12 JSON Schema.
Backup содержит `catalog.json`, его SHA-256 и все owned blobs, удерживаемые
активным manifest или pin. Поэтому состояние «manifest retired, blob удержан
только pin» является корректным и восстанавливаемым. Экспорт сохраняет квоту,
подписанные manifests и pins; leases намеренно не переносятся как временные
runtime claims. Restore:

1. сверяет ожидаемый и записанный digest каталога;
2. валидирует schema и каждый blob;
3. повторно записывает blob через storage adapter;
4. повторно проверяет подпись каждого manifest;
5. одной PostgreSQL-транзакцией публикует quota, ownership, все manifests,
   entries и pins.

Bytes и durable locations записываются до каталожной транзакции. При внешнем
сбое целевая организация остаётся пустой, а повторный restore идемпотентно
переиспользует content-addressed bytes. Инъекция PostgreSQL-сбоя после manifest
insert доказала нулевое частичное состояние и успешный повтор.

Каталог можно восстановить в новую организацию чистой установки. Это не
является импортом текущих production-данных и не читает текущий artifact tree.
Stage `21` включит этот доменный backup в полный installation backup lifecycle.

## Демонстрационный комплект и CLI

Канонический комплект поставляется в wheel из
`trading/resources/artifacts/demo_bundle/`; его точная тестовая копия находится
в `tests/fixtures/artifacts/demo_bundle/`. Комплект содержит два малых payload,
известные digests, подписанный `artifact.bundle.json` и только публичный ключ
издателя. Private key не хранится в репозитории. Команда
`roehubctl artifacts install <bundle>` принимает organization UUID, private
mode-`0600` файл DSN, CAS root, файл доверенных публичных ключей и необязательную
квоту. DSN читается через file descriptor с `O_NOFOLLOW`; значение не попадает
в вывод.

## Сбои и безопасное поведение

| Сбой | Поведение |
|---|---|
| Неизвестный издатель или неверная подпись | Публикация запрещена до записи каталога. |
| Повреждённый bundle, backup или CAS object | `artifact.*_corrupted` либо `artifact.digest_mismatch`; данные не выдаются. |
| Foreign organization manifest | Одинаковый `artifact.manifest_not_found`. |
| Превышение квоты | Транзакция каталога откатывается, зарегистрированный orphan удаляется. |
| Pin или активная lease | GC не снимает ownership и не удаляет bytes. |
| Повторная конкурентная публикация | Один immutable object и один idempotent manifest. |
| S3 timeout/HTTP failure | Ограниченная `artifact.s3_unavailable`; credentials не выводятся. |
| Process restart | Новый процесс читает каталог из PostgreSQL, bytes — из внешнего CAS. |

## Совместимость

Основная классификация Stage `14` — `breaking-change`: установочная
конфигурация меняет значение `artifacts.mode` с промежуточного `local` на
канонический `local_cas`, а fresh schema получает новый storage layout и
organization ownership. При этом `ArtifactStore/v1` и application ports —
новые versioned поверхности (`compatible-change` относительно существующего
кода). Текущие path-shaped backtest/RL consumers не переподключены и не
импортируются; их переход относится к Stages `15`–`17`.

Добавление несовместимого digest algorithm, executable payload, другого trust
model или иной manifest identity требует новой версии контракта.
