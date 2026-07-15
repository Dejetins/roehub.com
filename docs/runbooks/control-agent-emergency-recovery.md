# Аварийное управление через `control-agent` и `roehubctl`

## Назначение

Этот ранбук описывает локальный аварийный контур новой self-hosted установки.
Он остаётся доступен, когда Web UI, API и PostgreSQL остановлены. Основной API
не получает Docker socket и не запускает команды Docker: единственная такая
реализация находится в `apps/control_agent/`.

Для бизнеса это означает, что отказ панели администрирования или базы данных
не лишает владельца установки возможности увидеть состояние и восстановить
разрешённую topology. Цена этого разделения — наличие отдельного host-service,
его локального журнала и двух закрытых файлов служебной идентичности.

## Предварительные условия

- профиль сгенерирован в `configs/installation/generated/<profile>/`;
- `control-policy.json`, `compose.yaml` и `generation-manifest.json` не
  изменялись после генерации;
- runtime images установлены и совпадают с digest из release manifest;
- файлы API и `installation_owner` identity существуют только на основной
  машине, имеют режим `0600` и не находятся в репозитории;
- каталог сокета и аварийного журнала имеет режим `0700`.

Значения identity нельзя передавать в командной строке, журнале, диагностике
или отчёте. Клиент читает host-local файл, создаёт одноразовое HMAC-утверждение
со сроком действия не более 60 секунд и отправляет только его через Unix socket.

## Запуск host-service

Пример состава аргументов для installation unit:

```bash
roehub-control-agent \
  --profile-root /opt/roehub/config/generated/base \
  --trusted-release-manifest /opt/roehub/releases/0.1.0/release-metadata.json \
  --profile base \
  --project roehub-installation-base \
  --socket /var/run/roehub/control-agent.sock \
  --job-socket /var/run/roehub/job-control.sock \
  --journal /var/lib/roehub/control-agent/operations.jsonl \
  --api-token-file /etc/roehub/control-agent-api.identity \
  --owner-token-file /etc/roehub/control-agent-owner.identity \
  --job-token-file /etc/roehub/control-agent-job-runtime.identity
```

Installer обязан создавать identity-файлы локально и не печатать их
содержимое. Systemd/launchd packaging и ротация identity входят в release
lifecycle следующих этапов; ручное копирование значений запрещено.

## Безопасные команды владельца

Проверить generated policy без изменения Docker:

```bash
roehubctl validate-config \
  --profile-root /opt/roehub/config/generated/base \
  --trusted-release-manifest /opt/roehub/releases/0.1.0/release-metadata.json
```

Показать только очищенную effective-конфигурацию:

```bash
roehubctl effective \
  --path /opt/roehub/config/generated/base/effective-config.redacted.json
```

Диагностировать topology при остановленных API/PostgreSQL:

```bash
roehubctl doctor --profile base
```

Остановить известные сервисы. Для безопасного повтора нужно повторно передать
тот же `operation_id`; новый UUID означает новую операторскую операцию:

```bash
roehubctl stop \
  --profile base \
  --service web \
  --service api \
  --service postgresql \
  --operation-id 00000000-0000-4000-8000-000000000181
```

Восстановить разрешённую topology из уже установленного release manifest:

```bash
roehubctl recover --profile base
```

Восстановить предыдущую версию из последней проверенной pre-upgrade копии:

```bash
roehubctl rollback --profile base --release-version 0.1.0
```

Полный `N-1 → N`, state backup и state restore реализованы в Stage `21` через
закрытые host backup/release policies и typed state coordinator. Когда
`control-agent` запущен с `--backup-policy`, команды `backup`, `restore`,
`backup-cancel`, `restore-cancel`, `update` и `rollback` маршрутизируются в
типизированный recovery handler; terminal restore означает полный импорт и
`ready`, а не только расшифровку. Без policy/coordinator state-changing recovery
fail closed. Подготовка, fresh-target guard, проверка подписи и
расшифровываемости, возобновление и откат описаны в
[отдельной инструкции](installation-backup-restore-upgrade.md). Произвольная
команда оболочки не принимается ни в одном режиме.

## Неизвестный результат

Если Docker timeout, разрыв или ошибка receipt произошли после начала эффекта,
журнал получает `unknown`. Нельзя повторять такую операцию с новым
`operation_id`. Нужно вызвать сверку по исходному идентификатору через
control-agent API; сервис проверит topology либо signed backup/restore/release
result, installation fingerprint и request binding, после чего зафиксирует
`succeeded` либо оставит `unknown` без повтора эффекта.

## Проверка журнала и аудит

`operations.jsonl` — append-only hash-chain с `fsync`, последовательностью и
`previous_hash`. Он не зависит от PostgreSQL. После возврата API адаптер читает
события после последнего принятого `sequence` и идемпотентно записывает их в
основной аудит по `entry_hash`.

Повреждение строки, разрыв hash-chain, небезопасные права, symlink или
незавершённая последняя запись блокируют операции. В журнал не входят identity,
значения environment, command output, DSN, cookies или provider payloads.

## Запрещённые обходы

- монтировать Docker socket в API, Web UI, worker или plugin;
- выполнять `docker`, `docker compose` или shell из API;
- редактировать generated Compose для аварийной операции;
- передавать image, mount, environment или command внутри запроса;
- вслепую повторять `unknown` с новым `operation_id`;
- хранить аварийный журнал только в PostgreSQL.
