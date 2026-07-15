# Изолированная среда заданий Roehub v1

## Назначение

`JobEnvelope/v1` является единственной новой границей запуска тяжёлого или
стороннего вычислительного кода для greenfield-установки. Контрольный процесс
сохраняет задание и попытку в PostgreSQL, материализует только явно указанные
манифесты `ArtifactStore/v1`, запускает digest-bound OCI image и принимает
результат обратно как подписанный `ArtifactManifest/v1`.

Существующие вычислительные алгоритмы backtest, optimize, ML/RL и strategy не
переносят mutable state в runner. Они остаются владельцами доменной логики и
подключаются к общей среде через capability adapter. Stage `17` отвечает за
полную Compose-топологию producers/scheduler/worker; direct Python child paths
прежней системы не являются контрактом self-hosted v1.

## Контракты

`JobEnvelope/v1` содержит:

- `job_id`, `attempt_id`, номер попытки и server-derived `organization_id`;
- смысловой idempotency key и capability;
- только `sha256:<hex>` image identity, runtime/plugin version и, для custom
  strategy, canonical package digest принятого Stage `12` пакета;
- канонический config snapshot без float и secret-shaped keys;
- упорядоченные уникальные digests входных artifact manifests;
- CPU, RAM, PID, wall-time, tmpfs и output limits;
- UTC deadline, argv без host shell и `network=none`.

Канонические bytes дают `envelope_digest`. `semantic_spec_digest` исключает
только идентификаторы конкретной попытки и deadline: retry сохраняет тот же
`job_id`, image, config, inputs, limits и command, но получает новый
`attempt_id` и последовательный номер.

`JobResultManifest/v1` связывает job/attempt/org, исходный envelope digest,
итог, exit/error code, bounded output descriptors, подписанный artifact
manifest и, для custom strategy, строгий список `signal`/`intent`. Поля заказа,
exchange connection или credentials модель не принимает.

JSON Schemas Draft 2020-12 находятся в `schemas/jobs/` и воспроизводятся
командой:

```bash
uv run python -m tools.jobs.generate_schemas --check
```

Schema является переносимой структурной проекцией, но не самостоятельной
security boundary. Стандартный Draft 2020-12 не выражает четыре агрегатных
инварианта: canonical byte limit config, общий UTF-8 byte limit command,
уникальность output по path и суммарный размер outputs. Они помечены extension
fields; тесты отдельно показывают, что обычный Draft validator принимает такие
payloads, а обязательный Pydantic boundary отклоняет. Semantic enforcement
boundary явно записана в schema как `x-roehub-enforcement-boundary` и
выполняется через соответствующий `model_validate`.

## Capabilities

Host-owned registry включает `backtest`, `optimize`, `history_import`,
`report`, `artifact_transform`, `ml_training`, `ml_inference`, `rl_training`,
`rl_inference` и `custom_strategy`. Ни одна capability не имеет exchange
access. Импорт истории в этой границе получает уже загруженный входной
артефакт; внешний egress выключен. Расширение сетевой политики возможно только
как отдельный versioned контракт с реальным enforcement proof.

## Долговечное состояние

Миграция `0019` добавляет `job_runtime_jobs` и `job_runtime_attempts` с
organization-scoped composite keys. PostgreSQL хранит immutable envelope,
semantic spec digest, image digest, deadline, worker claim/heartbeat,
cancel request, результат и artifact manifest. `FOR UPDATE SKIP LOCKED`
разрешает конкурентный claim; Redis может быть wake-up transport, но не
источник истины.

Триггеры PostgreSQL запрещают менять identity/envelope и terminal rows прямым
`UPDATE`. Все переходы, затрагивающие обе таблицы, блокируют сначала job, затем
attempt. Если cancel фиксируется раньше finish, итогом становится `canceled`;
если finish фиксируется первым, поздняя отмена отклоняется. Recovery переводит
stale attempt в `recovering` с owner и lease timestamp, поэтому два recovery
процесса не владеют одним OCI-ресурсом одновременно. Порог потерянного worker
и срок recovery lease передаются раздельно: свежую lease нельзя немедленно
перехватить. Cancel marker, зафиксированный до claim либо до завершения
recovery, всегда переводит результат в `canceled`.

Истёкшая queued attempt становится `timed_out`. Потерянная running attempt
после heartbeat threshold становится `crashed`; новый процесс видит состояние
без памяти прежнего worker. Retry разрешён только после `failed`, `crashed`,
`timed_out` или `resource_exhausted` и только с тем же semantic spec digest.
После `succeeded` или cancel request retry запрещён.

## OCI-политика

Host runner перед `start` сверяет реальный container inspect:

- image ID равен digest из envelope, pull по tag запрещён;
- UID/GID `65532:65532`, read-only root, `cap-drop=ALL`,
  `no-new-privileges`;
- `network=none`, единственный read-only input bind mount, bounded tmpfs и
  одноразовый output volume на `tmpfs`;
- memory/memory-swap, NanoCPU и PID limits равны envelope;
- output volume имеет точную byte/inode quota, Docker logging выключен;
- Docker socket, DSN, token, password, credential и secret env отсутствуют.

Входной mount read-only и строится host executor непосредственно из
organization-scoped `ArtifactStore/v1`. Недоверенный container не получает
writable host path: output остаётся внутри quota-backed volume. Отдельный
host-owned keeper с read-only mount удерживает `tmpfs` после завершения
основного container. Runner ждёт реальной остановки PID 1, после которой Docker
уничтожает фоновые процессы, и только затем запускает именованный ограниченный
exporter. Keeper lifetime строго превышает максимальный job wall-time, а его
running state проверяется при создании и непосредственно перед export;
остановившийся keeper закрывает публикацию. Keeper/exporter запускаются из отдельного обязательного
host-controlled `utility_image_digest`, а не из недоверенного job/plugin image.
Job не получает writable control/status channel и не может подделать
exit code либо продолжить менять output во время копирования. Принимаются только
обычные неисполняемые файлы, не более 256 entries и установленного aggregate
byte limit; чтение bounded и защищено от symlink/TOCTOU. После подписанной
публикации либо terminal failure exporter, job, keeper, volume и input/output
scratch удаляются. Docker control calls имеют отдельный hard timeout, а
неоднозначный cleanup оставляет attempt для lease-protected recovery.

`TrustedRuntimeAuthority` до submit и повторно перед исполнением требует
точного совпадения capability, runtime name/version, image digest, command
digest и plugin package digest. Grant custom strategy строится только из
установленного Stage `12` `PluginPackage`, включённой organization-scoped
`PluginInstallation` и текущего host trust root publisher fingerprints;
unsigned, disabled, cross-organization, untrusted либо не совпадающий
package/image/version отклоняется. Корень scratch передаётся composition root и
для каждой попытки выводится только как `<runtime_root>/<attempt_id>`.
Plugin trust resolver вызывается заново при каждом `authorize`, поэтому
отзыв publisher key либо отключение installation между submit и execute
закрывает запуск; статический grant сам по себе недостаточен.

## Доказательства

`tests/fixtures/jobs/runtime_proof.py` использует чистую PostgreSQL `16`,
полный migration lifecycle до `0019`, реальный Alpine image с ID
`sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce`,
local CAS и подписанный Stage `14` demo bundle. Проверены success,
deterministic replay, custom strategy intent, timeout, cancel, crash, retry,
memory/PID/output/inode exhaustion, конкурентные cancel/finish, прямой запрет
изменения PostgreSQL rows, cross-organization denial и signed artifact result
publication. Отдельно доказано, что добровольный `exit 137` остаётся crash,
фоновый процесс не меняет output после остановки PID 1, cancel выигрывает
recovery, а свежая recovery lease не перехватывается вторым owner. Для restart
recovery worker действительно уничтожается через `SIGKILL` во время живых job
и keeper containers; новый Python-процесс удаляет orphan exporter/job/keeper,
output volume и scratch до фиксации результата.

Последнее сравнимое измерение использует одинаковые synthetic integer compute,
image и UID. Output path намеренно различается: baseline использует host bind,
а hardened path — quota volume, keeper и exporter; эта разница входит в
измеренный overhead. После пяти прогревов выполнено 20 измерений каждого пути:
минимальный OCI baseline имеет медиану `122.491 ms`, hardened lifecycle —
`444.336 ms`, отношение `3.627`; SHA-256 результата одинаков. Это измерение
покрывает container lifecycle overhead на текущем Colima host и не обещает
MPS, provider latency или production training throughput.

Версионированный результат доказательного запуска сохранён в
`roehub-self-hosted-oss-platform-v1-stage-reports/evidence/15-isolated-job-runtime-proof.json`.

## Совместимость и передача

Основная классификация — `breaking-change`: greenfield schema получает `0019`,
а прежние локальные child-process/job identities не являются aliases нового
контракта. Public models/schemas и capability registry — `compatible-change`;
browser defaults и внешние provider effects не меняются.

Stage `16` получает только validated strategy intent, никогда exchange secret
или прямой order submit. Stage `17` подключает producers/scheduler/worker к
единой контейнерной топологии и OpenBao-resolved host signing identity. Stage
`18` должен сериализовать operational Artifact GC относительно всех job/result
publishers. Stage `24` отдельно измеряет representative ML/RL на целевых Linux
архитектурах и macOS M3 Pro.
