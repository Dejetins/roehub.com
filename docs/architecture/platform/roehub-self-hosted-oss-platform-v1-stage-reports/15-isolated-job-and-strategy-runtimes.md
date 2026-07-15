# Этап 15 — изолированные задания и среды стратегий

## Статус

- Этап: `15`.
- Статус: `accepted`; независимая follow-up review дала
  `Release after fixes`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовые PostgreSQL и OCI containers,
  синтетические данные и Stage `14` demo bundle.
- Исключены: native MPS promise, mainnet order submit, production training
  cost, production data/credentials и Stage `17` Compose topology.
- Следующий разрешённый этап: `16`.

## Результат

Добавлены публичные `JobEnvelope/v1` и `JobResultManifest/v1`, Pydantic-модели и
Draft 2020-12 schemas. Генератор выражает доступные schema ограничения secret
keys, canonical JSON, уникальности input, paths и outcome; rejection parity
покрыта тестами только для выразимых ограничений. Четыре model-only инварианта —
canonical byte limit config, общий UTF-8 byte limit command, уникальность output
path и aggregate output bytes — отмечены extension fields. Отдельный тест
доказывает, что raw Draft validator их принимает, а Pydantic отклоняет; schema
прямо называет `model_validate` обязательной semantic enforcement boundary.
Envelope связывает
job/attempt/org, semantic key,
capability, immutable image digest/runtime, canonical config, input artifact
digests, exact limits, UTC deadline и argv. Secret-shaped keys, float,
непереносимые integers, mutable image tags и networking отклоняются.

Host-owned capability registry включает backtest, optimize, history import,
report, artifact transform, ML/RL training/inference и custom strategy. Ни одна
capability не имеет exchange access. Custom strategy result допускает только
строгие `signal`/`intent`; direct order fields отклоняются.

`TrustedRuntimeAuthority` до submit и повторно перед исполнением связывает
capability, runtime name/version, image digest, точный command digest и, для
custom strategy, canonical package digest Stage `12`. Custom strategy grant
принимает только установленный `PluginPackage`, включённую organization-scoped
`PluginInstallation`, текущий host trust root publisher fingerprints и точное
совпадение package/image/version; unsigned, disabled, cross-organization или
untrusted package отклоняется. Корень scratch задаётся
host composition root; произвольный путь из envelope или caller не принимается.
Plugin trust resolver вызывается при каждом `authorize`, включая повторную
проверку перед execute, поэтому отзыв publisher key или отключение installation
после submit закрывает запуск.

Миграция `0019` добавляет organization-scoped semantic jobs и attempts в
PostgreSQL. Catalog поддерживает idempotent submit, конкурентный
`FOR UPDATE SKIP LOCKED` claim, heartbeat, durable cancel, terminal result,
retry того же semantic spec и восстановление lost worker новым процессом.
DB triggers запрещают менять envelope, identity и terminal rows. Все
двухтабличные переходы используют порядок `job → attempt`; cancel/finish
линеаризованы. Recovery получает отдельный owner lease в состоянии
`recovering`, очищает OCI boundary и только затем фиксирует terminal result.
Порог lost-worker heartbeat и срок recovery lease разделены; свежую lease
нельзя немедленно перехватить. Cancel marker имеет приоритет как при recovery
claim, так и перед terminal commit.
Redis не является источником истины. Миграция включена в полный Stage `04`
manifest/bootstrap/readiness lifecycle.

`OciJobRunner` запускает только local image ID `sha256:<hex>` с
`--pull never`, UID/GID `65532:65532`, read-only root, dropped capabilities,
`no-new-privileges`, `network=none`, exact CPU/RAM/PID/tmpfs/output/time limits
и единственным read-only input bind mount. Output находится в одноразовом
Docker volume на `tmpfs` с byte/inode quota; только ограниченный host-owned
keeper с read-only mount удерживает его после завершения job. Runner ждёт
реальной остановки PID 1, после которой Docker уничтожает фоновые процессы, и
только затем запускает именованный exporter с временным writable bind mount.
Keeper lifetime превышает максимальный job wall-time; running state проверяется
при создании и перед export, иначе публикация закрывается. Keeper/exporter используют отдельный обязательный host-controlled
`utility_image_digest`, а не job/plugin image. Заданию не доступен writable
control/status channel. Docker logging выключен,
а control calls ограничены отдельным timeout. Job, keeper и volume inspect
проверяются; Docker socket и secret-shaped env отклоняются.

`JobAttemptExecutor` сам получает organization-scoped manifests из
`ArtifactStore/v1`, материализует entries в read-only input, запускает OCI и
публикует bounded non-executable outputs как подписанный
`ArtifactManifest/v1`. Digest повторно проверяется перед публикацией. Input,
output scratch, container и volume удаляются после terminal result; result
signing key остаётся только в доверенной host boundary.

## Реальная граница проверки

`tests/fixtures/jobs/runtime_proof.py` создаёт чистую PostgreSQL `16`, применяет
полный migration lifecycle до `0019`, создаёт две синтетические организации,
устанавливает Stage `14` signed demo bundle в local CAS и запускает реальные
containers из immutable Alpine image:

- tag для получения уже установленного image: `alpine:3.22`;
- проверенный image ID:
  `sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce`.

Проверены success и signed artifact result, deterministic replay, strict
strategy intent, timeout, cancel, crash, retry, memory/PID/output/inode
exhaustion, cross-organization denial, read-only artifact input,
non-root/rootfs/network policy, отсутствие Docker socket и полная очистка.
Добровольный `exit 137` классифицирован как crash; фоновый процесс не смог
изменить output после остановки PID 1. Worker с живыми job/keeper containers и
volume был завершён через `SIGKILL`; новый процесс получил recovery lease,
сверил labels, удалил orphan exporter/job/keeper, volume и scratch, после чего
сохранил `crashed`. Двенадцать конкурентных cancel/finish гонок завершились без
deadlock и с линейным итогом. Отдельно проверены cancel/recovery race, запрет
перехвата свежей recovery lease и reclaim после её истечения; прямые изменения
envelope и terminal row были отклонены PostgreSQL.

Последний результат имеет `schema=io.roehub.job-runtime-proof/v1` и
`status=passed`. Полная versioned запись сохранена в
[`evidence/15-isolated-job-runtime-proof.json`](evidence/15-isolated-job-runtime-proof.json).

Полный `apps.migrations.verify_storage_runtime` после `0019` также прошёл:
fresh bootstrap, interrupted recovery, idempotent rerun, persistent-volume
restart, external readiness, все ранее принятые organization/auth/OIDC/research/
trading/notification probes и cleanup.

## Доказательство производительности

Baseline и hardened path использовали тот же image, тот же synthetic integer
compute и UID. Hardened path дополнительно включает quota-backed output volume,
ограниченный exporter и полную policy/lifecycle runner; PostgreSQL claim и
Artifact Store publication в этот узкий benchmark не входят. После пяти
прогревов выполнено по 20 измерений.

- baseline median: `122.491 ms`;
- hardened median: `444.336 ms`;
- отношение: `3.627` при fail-closed бюджете не выше `4.0`;
- результат обоих путей:
  `sha256:75f206dc943bc3e65701052f1a3a6b3c7bc6f77eeca150cf24f38155e59674fb`.

Это доказательство измеряет container lifecycle overhead на текущем Colima
host. Оно не является обещанием MPS, provider latency или production ML/RL
throughput; target-platform compute matrix остаётся Stage `24`.

## Проверки качества

- Job schema generation и `--check` — `passed`; CI drift gate добавлен.
- Целевой `ruff` — `passed`.
- Целевой `pyright` — `0 errors, 0 warnings`.
- Целевой pytest contracts/schema/migration/storage — `29 passed`.
- Реальный PostgreSQL/OCI/ArtifactStore proof — `passed` с cleanup.
- Полный storage lifecycle — `passed` с cleanup.
- Полный `uv run ruff check .` — `passed`.
- Полный pytest — `1872 passed`, четыре прежних `httpx` warnings.
- Полный `uv run pyright` не является gate этапа: прежние
  `153 errors, 2 warnings` остаются только в чужих `local_artifacts` и двух
  exchange cleanup tools; Stage `15` scope чистый.
- Docs index generation/`--check`, project map generation/`--check`, runtime
  input inventory (`146`) и `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| `JobEnvelope/v1`, result и schemas | `compatible-change` | Добавлены новые versioned public contracts. |
| Capability/application boundary | `compatible-change` | Новая host-owned registry не меняет текущие domain algorithms. |
| Persistence | `breaking-change` | Greenfield schema получает organization-scoped `0019`. |
| Config/defaults | `none` | User-facing `roehub.yaml` на этом этапе не менялся. |
| Job/request identity | `breaking-change` | Semantic job + attempt/digest заменяют legacy child-process identity для self-hosted v1. |
| Service calls | `breaking-change` | Greenfield scheduler/worker должен использовать PostgreSQL claim и OCI executor; Stage `17` собирает topology. |
| External effects | `none` | Только disposable PostgreSQL/containers и local temp CAS с cleanup. |
| Secrets/trust | `compatible-change` | Host signer остаётся в trusted composition boundary; job не получает secret material. |
| Browser defaults | `none` | Web UI не менялся. |

Основная классификация — `breaking-change`, ожидаемая для greenfield v1. Legacy
job alias, dual-read и импорт прежнего runtime state отсутствуют по `A07`.

## Независимая проверка

- Режим: одна cold independent review и follow-up того же reviewer после
  исправлений.
- Первоначальный вердикт: `Block`.
- Исправлены: host disk/inode/log bounds, реальный orphan recovery, lock order
  и cancel/finish race, schema parity, runtime/package authority, DB
  immutability, failure classification, writable status/TOCTOU boundary,
  cancel/recovery race, recovery lease cutoff и недостающие реальные proofs.
- Follow-up verdict: `Release after fixes`; оставшихся blockers нет.
- Локальная повторная проверка: `29 passed`, scoped ruff/pyright, real OCI/DB
  proof и полный storage lifecycle — `passed`.
- Остаточные неблокирующие риски: production Compose wiring dynamic plugin trust
  resolver и utility image digest принадлежит Stage `17`; recovery lease должна
  превышать худшее время cleanup; target-host ML/RL performance принадлежит
  Stage `24`; Artifact GC serialization относительно result publishers —
  Stage `18`. Raw JSON Schema остаётся только structural projection, поэтому
  все consumers обязаны вызывать `model_validate`.

## Файлы и ограничения выполнения

Созданы public models/schemas, schema generator, PostgreSQL catalog, OCI runner,
capability registry, host executor/result publisher, migration `0019`, runtime/
restart proofs, versioned proof result, focused tests, архитектурный документ и
этот отчёт. Обновлены
Artifact Store host API, migration lifecycle/verifier/manifest, CI schema gate,
platform plan, docs index/project map и центральный ledger.

Чужие dirty изменения сохранены. Staging, commit, push, deploy, production
mutation и paid compute не выполнялись. Одноразовые containers и временные
каталоги удалены; установленные Docker images сохранены для повторяемых proof.

## Передача Stage 16

После принятия отчёта Stage `16` получает
durable successful job result и strict strategy `signal`/`intent`, но не
exchange credential, direct order submit или право обходить server-side risk,
idempotency и reconciliation gates.
