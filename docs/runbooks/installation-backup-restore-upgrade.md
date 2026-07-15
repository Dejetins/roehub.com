# Резервное копирование, восстановление, обновление и откат установки

## Назначение и граница

Эта инструкция описывает единый аварийный жизненный цикл Roehub через
host-side `control-agent` и `roehubctl`. Web и API для него не требуются.
Операции разрешены только `installation_owner`, используют закрытый
`io.roehub.installation-backup-policy/v1alpha1` и не принимают произвольные
команды оболочки.

Восстановление всегда выполняется в новый пустой каталог или новую установку.
Восстановление поверх источника, непустой цели, символической ссылки или
непроверенного комплекта завершается отказом.

## Владельцы состояния

Подписанный манифест обязан содержать ровно восемь владельцев:

| Владелец | Режим согласованности | Ограничение |
|---|---|---|
| `release_config` | `application_quiesced` | Версия релиза и снимок сгенерированной конфигурации должны относиться к одному окну остановки записей. |
| `postgresql` | `database_snapshot` | Каноническое транзакционное состояние приложения. |
| `clickhouse` | `database_snapshot` | Диапазоны времени и число строк сверяются после импорта. |
| `redis_checkpoint` | `durable_checkpoint` | Это восстанавливаемая контрольная точка, но не единственный источник истины. |
| `openbao` | `encrypted_raft_snapshot` | Raft snapshot уже зашифрован операторским `age`-ключом; recovery/unseal values в комплект и отчёты не входят. |
| `artifacts` | `content_addressed_snapshot` | Blob и каталог сверяются по SHA-256. |
| `plugin_operation_audit` | `application_quiesced` | Package/instance, операции и аудит относятся к тому же окну согласованности. |
| `observability` | `bounded_history_snapshot` | Ограниченная история помогает расследованию, но не является продуктовым источником истины. |

Отсутствие любого владельца, повтор владельца или неизвестный режим
согласованности блокирует резервную копию до создания ciphertext.

## Предварительная проверка

1. Убедиться, что `control-agent` запущен с `--backup-policy` и что policy-файл,
   operator identity, signing key и каталоги имеют владельца установки и не
   доступны для записи группе или остальным пользователям.
2. Проверить, что `age`, `docker version` и `docker compose version` доступны
   host-side сервису.
3. Проверить typed capture coordinator, который создаёт согласованные
   snapshot-файлы по точным именам из policy внутри эффекта control agent.
   `source_root`, `backup_root` и `restore_root` обязаны быть абсолютными,
   отдельными и не вложенными друг в друга. `source_root` имеет режим `0700`,
   snapshot-файлы — `0600`. Plaintext staging автоматически удаляется только
   после configured-key signature/decryptability verification; при отмене он
   остаётся digest-bound материалом для resume и удаляется после успеха.
   Постоянным backup-артефактом является только каталог с файлами `*.age` и
   подписанным манифестом.
4. Не хранить age identity и signing private key рядом с backup destination.
   Публичный verification key и recipient не дают возможности расшифровать
   состояние.
5. Зафиксировать уникальные `backup_id` и `operation_id`. Повтор смысловой
   операции использует тот же `backup_id`; новый эффект с другим содержимым не
   должен переиспользовать существующий идентификатор.

## Создание копии

```bash
roehubctl backup \
  --profile base \
  --subject-id backup-20260714-001 \
  --operation-id 00000000-0000-4000-8000-000000002101
```

Успех означает, что все восемь ciphertext-файлов записаны, их SHA-256 и
plaintext SHA-256 связаны с versioned manifest, а detached Ed25519 signature
проверяется operator public key. `backup-progress.json` должен иметь
`state=completed`; `latest-verified.json` обновляется только после успешной
проверки полного комплекта.

Не копировать в тикеты или отчёты содержимое snapshot, ciphertext, ключи,
credential-файлы, DSN и необработанный вывод хранилищ.

## Отмена, частичный отказ и возобновление

Отмена адресует `operation_id` выполняемой операции, а не `backup_id`:

```bash
roehubctl backup-cancel \
  --profile base \
  --subject-id 00000000-0000-4000-8000-000000002101 \
  --operation-id 00000000-0000-4000-8000-000000002102
```

Для восстановления используется `restore-cancel` с тем же правилом. Marker
создаётся отдельно от Web/API. Выполняющий процесс проверяет его во время
потокового шифрования/расшифровки и между владельцами состояния, фиксирует
`state=cancelled` и не публикует полный manifest/result.

После частичного отказа не удалять progress-файл и уже записанные ciphertext.
Повторить операцию с тем же `subject-id`, но новым `operation-id`: каждый
готовый ciphertext повторно сверяется по digest, после чего продолжаются только
отсутствующие владельцы. Несовпадение digest блокирует возобновление.

## Восстановление в новую установку

```bash
roehubctl restore \
  --profile base \
  --subject-id backup-20260714-001 \
  --operation-id 00000000-0000-4000-8000-000000002103
```

До расшифровки проверяются identity подписи, detached signature, manifest hash,
полнота владельцев, форма каталога и digest каждого ciphertext. Каждый
plaintext-файл записывается атомарно и повторно сверяется с manifest. Затем
typed state coordinator импортирует PostgreSQL, ClickHouse и Redis checkpoint,
восстанавливает OpenBao только на свежем storage, публикует
content-addressed artifacts и release/config, после чего сверяет:

- строки пользователей, конфигурации, plugin/operation state и аудита;
- число строк и временной диапазон ClickHouse;
- Redis checkpoint с учётом того, что Redis не источник истины;
- SHA-256 артефактов, конфигурации и всех восьми plaintext snapshot;
- версию релиза и installation fingerprint.

`restore.completed` публикуется только после полной сверки и состояния
`ready`. Частично восстановленная цель не становится текущей установкой.

## Обновление `N-1 → N` и откат

`update` и `rollback` fail closed, если `latest-verified.json` отсутствует,
подпись/расшифровываемость не проходят, версия источника не совпадает с
установленной или переход отсутствует в owner-protected
`io.roehub.installation-release-policy/v1alpha1`.

1. Создать и проверить полный pre-upgrade backup версии `N-1`.
2. Для необратимой миграции добавить digest отдельного forward recovery plan в
   trusted release policy. Одной резервной копии недостаточно.
3. Восстановить `N-1` в отдельную цель, применить миграции `N` и только после
   всех проверок атомарно изменить marker установленной версии.
4. При отказе до commit возобновить ту же цель по progress-файлу. При
   несовпадении состояния не продолжать.
5. Для отката восстановить подтверждённый `N-1` backup ещё в одну свежую цель;
   не перезаписывать частично обновлённую установку.

## RPO, RTO и доказательство

`observed_rpo_seconds` — максимальный возраст любого owner snapshot на границе
завершения quiesce window. `observed_rto_seconds` измеряется от начала restore
до полного импорта, сверки и `ready`. Эти числа описывают конкретную репетицию
и не являются обещанием SLA.

Локальная воспроизводимая проверка без production data:

```bash
uv run python -m tools.backup.verify_runtime \
  --project-prefix roehub-stage21-proof
```

Успешный результат обязан содержать `status=passed`, восемь зашифрованных
владельцев, `manifest_signature=verified`, совпадение БД/диапазонов/digest,
работающий мониторинг при остановленных Web/API, реальную concurrent отмену и
resume backup/restore, отказ обновления до commit, возобновление, rollback и
нулевые остаточные Docker-ресурсы. Принятие является `real-boundary`, а не
tests-only.

## Когда остановиться

Немедленно остановить жизненный цикл при неполном owner coverage,
незашифрованном постоянном backup, ошибке подписи/digest, непустой цели,
неизвестной согласованности, несовпадении данных, отсутствии rollback или
попытке выдать измерение одной репетиции за SLA. Production restore, commit,
push, deploy и greenfield production launch требуют отдельного разрешения.
