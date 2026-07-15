# OpenBao: секреты, ротация и восстановление

## Назначение и граница

Этот документ описывает greenfield-контур OpenBao для self-hosted Roehub v1.
Историческая native-инструкция `docs/runbooks/exchange-secret-management.md`
остаётся доказательством прежней системы и не является руководством для новой
установки. Секреты, токены и recovery material прежней установки не
импортируются.

OpenBao — единственный владелец значений ключей бирж, Telegram, OIDC, плагинов
и storage credentials. PostgreSQL и API/UI хранят либо передают только opaque
reference. В evidence разрешены классы состояния, версии, digest зашифрованного
backup и policy outcome; запрещены значения, ciphertext, HMAC input, AppRole
credential, unseal share и recovery identity.

Stage `08` вводит этот контракт платформы и переводит OIDC, exchange и storage
границы. Перевод существующего notification-dispatcher/Telegram worker с raw
env на typed reference принадлежит Stage `11`. Пока Stage `11` не принят, этот
старый worker path не является разрешённым runnable path greenfield v1.

## Артефакты и владельцы

| Контракт | Канонический артефакт | Владелец |
|---|---|---|
| Контейнер и persistent state | `infra/docker/openbao-embedded.compose.yml`, volume `openbao-data` | `control-agent` |
| Конфигурация Raft и audit | `infra/openbao/config/openbao.hcl` | platform security |
| Service policies | `infra/openbao/policies/` | platform security |
| Init/unseal custody | `configs/openbao/bootstrap.yaml` | `installation_owner`, quorum `2` из `3` |
| Зашифрованный backup | `configs/openbao/backup.yaml` | `control-agent`; recovery identity — `installation_owner` |
| Runtime readiness | `ops.roehub.io/v1`, runbook `auth.openbao-unavailable` | operator |

Образ закреплён manifest digest и запускается не в `dev`-режиме. Raft хранится
на volume вне writable layer контейнера. Audit device задаётся декларативно в
HCL, пишет только HMAC-защищённые записи с `log_raw=false` на отдельный volume.

## Инициализация и unseal

Production bootstrap создаёт `3` Shamir shares с threshold `2`. Каждый share и
начальный административный credential шифруются заранее переданными публичными
PGP-ключами владельцев и записываются в owner-controlled files с mode `0600`.
Они не выводятся в terminal, JSON evidence, CI log или чат.

После первого unseal автоматизированный bootstrap:

1. включает KV v2 на mount `kv` с retention `5` версий и Transit;
2. создаёт ключ `roehub-exchange-credentials` без export/plaintext backup;
3. загружает политики из `infra/openbao/policies/`;
4. создаёт отдельные AppRole и выдаёт одноразовые response-wrapped SecretID в
   mode-`0600` files с wrap TTL `5m`;
5. разворачивает каждый wrapping token ровно один раз, выдаёт короткоживущий
   token `15m` с максимумом `30m`, без `default` policy, и проверяет `renew-self`;
6. отзывает initial administrative credential.

Одноразовый verifier Stage `08` создаёт три независимых временных PGP keyring,
получает `3` PGP-зашифрованных shares с threshold `2`, расшифровывает только два
share в памяти, PGP-расшифровывает начальный administrative credential и
отзывает его после bootstrap. Перезапуск проверяется тем же quorum `2` из `3`.
Схема `1/1` используется только для временной инициализации уже нового пустого
recovery volume перед force restore; после восстановления она недействительна и
не является параметром восстановленного storage.

## Служебные границы

| Идентичность | Разрешено | Явно не разрешено |
|---|---|---|
| `api` | health и metadata под `kv/metadata/roehub/*` | KV values, Transit decrypt, root/unseal |
| `identity` | `kv/data/roehub/oidc/*` read | Telegram, exchange, plugins |
| `notification-dispatcher` | `kv/data/roehub/telegram/*` read | OIDC, exchange, plugins |
| `exchange-execution` | exchange KV read и ограниченный Transit encrypt/decrypt/HMAC | другие secret kinds и operator paths |
| `plugin-runtime` | один точный `organization_id/instance_id` path | соседние организации и экземпляры |
| `secret-operator` | create/update/read, soft-delete/undelete и metadata | permanent destroy |
| `backup-recovery` | Raft snapshot read | KV values, policy mutation, force restore |

Force restore на новом пустом storage принадлежит только
`installation_owner` с одноразовым административным credential новой пустой
инсталляции. Он не входит в service policy.

## Типизированные ссылки

Каноническая форма:

`openbao://<mount>/<root>/<kind>/<resource>?version=<positive-int>#<field>`

Поддерживаемые `kind`: `exchange`, `telegram`, `oidc`, `plugins`, `storage`.
Fragment с именем field обязателен. `version` необязателен: отсутствие означает
latest non-deleted version. Resolver проверяет configured root и ожидаемый kind,
отклоняет traversal/escaping, читает service credential из абсолютного regular
file с mode `0600` на каждый запрос и возвращает объект с redacted `repr/str`.

Примеры ссылок без значений:

- `openbao://kv/roehub/oidc/provider-a#client_secret`;
- `openbao://kv/roehub/telegram/org-a#bot_token`;
- `openbao://kv/roehub/exchange/org-a/connection-a?version=2#credential`;
- `openbao://kv/roehub/plugins/org-a/instance-a#credential`.

## Ротация и rollback

1. `secret-operator` записывает полную новую KV v2 version.
2. Consumer без `version` получает latest после следующего чтения token file и
   reference; процесс не требует перезапуска.
3. Bounded canary подтверждает provider operation без записи значения в evidence.
4. При ошибке latest version выполняется soft-delete, а предыдущая version
   остаётся доступной по pinned reference.
5. Undelete возвращает удалённую version. Rollback копирует выбранную прежнюю
   version в новую version, сохраняя audit trail; номер версии не уменьшается.
6. Permanent destroy запрещён общей operator policy и требует отдельного
   security-owner решения вне автоматического этапа.

## Зашифрованный backup

Backup destination обязан быть отдельным от `openbao-data` volume. В каталоге
backup сохраняются только `age` ciphertext и sidecar с timestamp, размером и
SHA-256 ciphertext. Recovery identity хранится отдельно у
`installation_owner`; public recipient разрешён в конфигурации.

Пример запуска с host-local files:

```bash
uv run python infra/openbao/snapshot.py backup \
  --address http://127.0.0.1:18200 \
  --credential-path /run/secrets/roehub-openbao-backup-token \
  --recipient-path /etc/roehub/openbao/backup-recipient.txt \
  --destination /var/lib/roehub/backups/openbao/openbao.snap.age
```

Команда отказывается перезаписывать существующий файл и публикует результат
атомарно с mode `0600`. Raw Raft snapshot существует только в памяти процесса и
не записывается на диск.

## Восстановление на новой пустой установке

До начала должны быть доказаны: пустой новый volume, digest ciphertext,
доступность owner-held recovery identity и исходный unseal quorum. Если volume
не пустой либо происхождение backup не подтверждено, операция останавливается.

1. Остановить consumers и сохранить secret-free readiness timeline.
2. Создать новый пустой `openbao-data` volume и временно инициализировать его.
3. Передать временный administrative credential через mode-`0600` file.
4. Выполнить explicit force restore только потому, что storage новый и его seal
   отличается:

```bash
uv run python infra/openbao/snapshot.py restore \
  --address http://127.0.0.1:18200 \
  --credential-path /run/secrets/roehub-openbao-recovery-admin \
  --recovery-path /secure/offline/openbao-backup.agekey \
  --source /var/lib/roehub/backups/openbao/openbao.snap.age \
  --force-new-storage
```

5. После restore временный credential новой пустой установки становится
   недействителен. Обязательно перезапустить контейнер OpenBao, чтобы процесс
   заново загрузил восстановленную конфигурацию seal из Raft.
6. После перезапуска unseal выполняется исходным quorum восстановленного
   snapshot. Share временной схемы нового storage применять нельзя.
7. Проверить `unsealed`, AppRole policy denial/allow и одну pinned version без
   вывода значения. Затем запустить consumers.

Force endpoint нельзя использовать для in-place rollback или обхода ошибки seal
compatibility. Recovery material нельзя пересылать в чат либо добавлять в
incident evidence.

## `ops.roehub.io/v1`

Resolver публикует только `DependencyReadiness` с `classification`:
`uninitialized`, `sealed`, `unsealed` либо `unavailable`, boolean `ready` и
`runbook_id=auth.openbao-unavailable`. При любом состоянии кроме `unsealed`
secret-dependent операция fail closed; raw-env fallback отсутствует.

После инцидента собираются только health status, sanitized policy outcome,
ciphertext digest backup, version transition и подтверждение отсутствия
forbidden material в audit/application/verification output.
