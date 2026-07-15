# Этап 14 — хранилище артефактов и демонстрационный комплект

## Статус

- Этап: `14`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовые PostgreSQL/MinIO/OpenBao и временные
  каталоги без production data, credentials или текущего artifact state.
- Исключены: импорт текущего RL/backtest corpus, production mutation и
  переподключение существующих path-shaped consumers.
- Следующий разрешённый этап: `15`.

## Результат

Добавлены публичные библиотечно-независимые контракты `ArtifactStore/v1`,
`ArtifactManifest/v1` и `ArtifactBackup/v1`, их Pydantic-модели и Draft 2020-12
JSON Schemas. Manifest использует SHA-256 content identity, нормализованные
пути и обязательную Ed25519-подпись. Secret-shaped metadata и executable entry
отклоняются. Буферизованный контракт ограничивает один blob и весь bundle
значением `67108864` байт; недоказанный предел 1 ТиБ удалён.

`local_cas` стал каноническим default в `roehub.yaml`. Local adapter публикует
read-only blobs атомарным hard link после `fsync` и digest verification,
поддерживает read-only materialization для mmap-heavy consumers и очищает
materialized links вместе с canonical object. S3-compatible adapter использует
SigV4, bounded timeout, HTTPS для удалённых endpoints и только typed OpenBao
reference типа `storage` для credentials.

Миграция `0018` добавляет content-only object catalog, backend locations,
organization-scoped manifests/ownership, quotas, pins, leases и durable GC
candidates. Она включена в единый Stage `04`
migration manifest/bootstrap/status lifecycle. Команда
`roehubctl artifacts install <bundle>` устанавливает signed bundle через тот же
service, CAS и PostgreSQL catalog; DSN читается только из private regular file.

Демонстрационный комплект включён в wheel, содержит два искусственных payload общим размером
`202` байта, известные digests, signed manifest и публичный ключ. Private key в
репозитории отсутствует. Текущие артефакты Roehub не читались и не копировались.

## Реальная граница проверки

`tests/fixtures/artifacts/runtime_proof.py` создаёт чистую PostgreSQL `16`,
применяет полный migration lifecycle до `0018`, создаёт три искусственные
организации и запускает MinIO и OpenBao из digest-bound images:

- image: `minio/minio:RELEASE.2025-04-22T22-12-26Z`;
- digest:
  `sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e`.
- OpenBao image: `ghcr.io/openbao/openbao:2.5.4`;
- OpenBao digest:
  `sha256:436eaf9778cad75507ff70ea26ace30dcbe15606e619ac3823495663d7f7c115`.

Proof проходит через реальную CLI-команду, local CAS, PostgreSQL repository и
S3 adapter и настоящий OpenBao resolver. Проверены 16 конкурентных установок
одного bundle, новый Python-процесс над прежними PostgreSQL/CAS,
cross-organization denial, серия quota rollback без физического роста,
прерывание после durable регистрации, pin/lease/GC, corruption rejection,
pinned-only backup, транзакционный restore с инъекцией сбоя и повтором, а также
S3 put/get/materialize/delete. Все контейнеры, volume и network удалены; точный
cleanup проверен через Docker.

Последний результат:

`{"artifact_store_contract":"passed","atomic_concurrent_publish":"passed","backup_restore":"passed","catalog_postgresql":"passed","cleanup":"passed","cli_bundle_install":"passed","corruption_rejection":"passed","cross_organization_denial":"passed","demo_bundle_signature":"passed","gc_pin_lease":"passed","image_digest_binding":"passed","interrupted_orphan_cleanup":"passed","local_cas_process_restart":"passed","materialization_benchmark":{"copy_median_ms":1.371,"materialize_median_ms":0.771,"materialize_over_copy_ratio":0.562,"payload_bytes":8388608,"samples":30,"warmups":5},"openbao_s3_credentials":"passed","quota":"passed","quota_orphan_cleanup":"passed","restore_atomicity":"passed","s3_compatible_minio":"passed","schema":"io.roehub.artifact-store-runtime-proof/v1","status":"passed"}`

## Доказательство производительности

На одном хосте и одном 8 MiB payload сравнивались полная `shutil.copyfile` и
`LocalCasBlobStore.materialize`; обе серии имели пять прогревов и 30 измерений.
Медиана materialization — `0.771 ms`, copy baseline — `1.371 ms`, отношение —
`0.562`. Измерение доказывает только локальный same-filesystem hot path через
hard link на этой машине; оно не является обещанием S3 latency или Stage `24`
macOS ML/RL throughput.

## Проверки качества

- Генерация и проверка трёх artifact JSON Schemas — `passed`.
- Целевой `ruff` — `passed`.
- Целевой `pyright` — `0 errors, 0 warnings`.
- Целевой pytest contracts/CAS/S3/CLI/migration — `23 passed`.
- Реальный PostgreSQL/MinIO/OpenBao proof — `passed` с cleanup.
- Stage `03` installation schema/golden/runtime — `36 passed`; deterministic
  generation и реальные `docker compose config`/config-consumer для
  `base/trading/ml` прошли.
- Полный PostgreSQL/ClickHouse/Redis storage lifecycle — `passed`: fresh
  bootstrap, interrupted recovery, idempotent rerun, persistent restart,
  external readiness и cleanup.
- Packaged wheel — `passed`: четыре demo bundle files присутствуют в настоящем
  `roehub-0.1.0-py3-none-any.whl`.
- Полный `uv run ruff check .` — `passed`.
- Полный pytest — `1853 passed`, четыре прежних `httpx` warnings.
- Полный `uv run pyright` не является gate этапа: `153 errors, 2 warnings`
  остаются только в чужих `local_artifacts` и двух exchange cleanup tools;
  Stage `14` scope — чистый.
- Docs index generation/`--check`, project map generation/`--check`, runtime
  input inventory и финальный `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| `ArtifactStore/v1`, manifest и backup | `compatible-change` | Добавлены новые versioned contracts и schemas. |
| Application ports/service | `compatible-change` | Новый порт не заменяет текущие artifact services. |
| Persistence | `breaking-change` | Fresh schema получает `0018` и organization ownership. |
| Config/defaults | `breaking-change` | `artifacts.mode=local` заменён на канонический `local_cas`; S3 требует region и OpenBao ref. |
| Identity | `compatible-change` | Catalog использует принятую Stage `05` organization authority. |
| Request/hash identity | `compatible-change` | Новая SHA-256 identity пока не переключает прежних path-shaped consumers. |
| Service calls | `compatible-change` | CLI и будущие consumers вызывают новый application service; существующие вызовы не переключены. |
| Внешние эффекты | `none` | Только disposable PostgreSQL/MinIO/OpenBao и временные каталоги с полным cleanup. |
| Audit/runbooks | `none` | Операционный запуск GC/backup будет включён в Stages `18`,`20`,`21`. |
| Browser defaults | `none` | Web UI не менялся. |

Основная классификация — `breaking-change`, запланированная для greenfield v1.
Legacy converter, dual-read и импорт текущих путей отсутствуют по решению
`A07`.

## Независимая проверка

- Режим: ровно одна cold independent review.
- Первоначальный вердикт: `Block`.
- Блокирующие замечания: quota rollback оставлял невидимые физические blobs;
  заявленный `ArtifactBackup/v1` не имел модели/schema и не сохранял pinned-only
  state; restore был многошаговым и невозобновляемым.
- Существенные замечания: буферизованный 1 TiB contract, глобальные
  media/backend metadata, недостаточно нормативная канонизация, отсутствие
  OpenBao/process-restart proof, schema drift и demo только в test fixture.
- Исправлено: durable object locations и quota/interruption orphan cleanup;
  строгие backup model/schema и pinned-only proof; единая restore transaction с
  fault injection и retry; 64 MiB blob/bundle limits; content-only global
  metadata; UTC/finite portable canonical JSON и golden bytes; настоящий
  OpenBao resolver; отдельный restart process; schema extensions и demo в wheel.
- Повторная независимая проверка не запускалась. После финальных gates выполнена
  холодная локальная перепроверка всех замечаний.
- Остаточный риск: distributed publish того же digest против GC не входит в
  Stage `14`. До Stage `18` GC не подключается к operational entrypoint; Stage
  `18` обязан сериализовать его относительно всех publishers, а не только
  обеспечить одного GC-лидера.

## Холодная локальная перепроверка после исправлений

- Режим: `cold self-review fallback` после единственной независимой проверки.
- Проверено: каждый blocker и High/Medium finding сопоставлен с кодом, schema,
  SQL constraint, unit test, реальным PostgreSQL/MinIO/OpenBao proof, wheel и
  отчётом.
- Вердикт: `Release after fixes` для Stage `14`.
- Blocker устранены: невидимых quota/interruption orphans нет; pinned-only
  backup валиден; restore атомарен и повторяем после fault injection.
- Существенные замечания устранены: 64 MiB bounded contract, content-only
  global identity, нормативная канонизация и golden bytes, real OpenBao,
  отдельный process restart, schema constraints/extensions и packaged demo.
- Follow-up: distributed publish-vs-GC serialization остаётся явным
  обязательством Stage `18`; GC до этого не имеет operational entrypoint.

## Файлы и ограничения выполнения

Созданы contract/models/schemas, local CAS и S3 adapters, service/ports/catalog,
migration `0018`, generator, packaged signed demo bundle, CLI use case, runtime proof,
focused tests, архитектурный документ и этот отчёт. Обновлены installation
config/schema/golden hashes и единый migration lifecycle.

Чужие dirty изменения сохранены. Staging, commit, push, deploy и production
mutation не выполнялись. MinIO и OpenBao images оставлены как установленные
зависимости для следующих повторяемых proof; одноразовые runtime resources
отсутствуют.

## Передача Stage 15

После принятия Stage `14` этап `15` получает immutable signed manifest,
organization-scoped catalog, local materialization и S3-compatible storage.
Stage `15` обязан запускать consumer под непривилегированной identity с
read-only materialization, не передавать S3/OpenBao credentials заданию и
публиковать результат через тот же signed/digest contract.
