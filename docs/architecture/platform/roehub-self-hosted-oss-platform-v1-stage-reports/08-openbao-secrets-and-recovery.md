# Этап 08 — OpenBao, ссылки на секреты и восстановление

## Статус

- Этап: `08`.
- Статус: `accepted`; единственная независимая проверка сначала дала `Block`,
  все обязательные замечания исправлены и локальная повторная проверка прошла.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Proof boundary: `N/A`; только одноразовый OpenBao, искусственные значения и
  зашифрованная временная резервная копия. Production OpenBao, реальные
  биржевые/Telegram/OIDC credentials и recovery material не читались и не
  изменялись.
- Blocker: отсутствует.
- Следующий разрешённый этап: `09`.

## Реальная граница проверки

Принятие не основано только на тестах. На настоящем Docker Engine `29.5.2`
одноразовый контейнер OpenBao `2.5.4` прошёл PGP-инициализацию `3/2`, состояния
`uninitialized → sealed → unsealed`, policy denials, ротацию KV v2, AppRole
response wrapping/renewal, перезапуск Raft, зашифрованный `age` snapshot и
восстановление на новый пустой volume.
Verifier удалил контейнеры, тома и временные материалы после проверки.
Класс доказательства: `real-boundary` Docker/OpenBao, а не принятие только по
модульным тестам.
Тип доказательства: `runtime`, реальная граница Docker/OpenBao; это не
`tests-only` acceptance.

Команда: `uv run python infra/openbao/verify_runtime.py`.

Безопасный результат:

```json
{"api_transit_decrypt":"denied","api_value_access":"denied","approle_one_time_unwrap":"passed","approle_response_wrapping":"passed","backup_source_digest_verified":true,"bootstrap_credential_revoked":"passed","cleanup":"passed","compose_config":"passed","encrypted_backup":true,"forbidden_output_scan":"passed","fresh_storage_guard":"passed","fresh_volume_force_restore":"passed","image_digest_pinned":true,"initial_admin_pgp_encrypted":true,"initialization_states":["uninitialized","sealed","unsealed"],"live_reference_rotation":"passed","ops_status":{"apiVersion":"ops.roehub.io/v1","kind":"DependencyReadiness","metadata":{"id":"openbao"},"status":{"classification":"unsealed","ready":true,"runbook_id":"auth.openbao-unavailable"}},"production_pgp_bootstrap":"passed","production_unseal_shares":3,"production_unseal_threshold":2,"raft_restart_persistence":"passed","restored_config_reload":"passed","rollback_new_version":"passed","schema":"io.roehub.openbao-runtime-proof/v1","service_identities":7,"service_token_renewal":"passed","shared_broad_tokens":false,"soft_delete_undelete":"passed","status":"passed","typed_kinds":["exchange","oidc","plugins","storage","telegram"],"version_rotation":[1,2,3],"wrapped_file_delivery":"passed"}
```

Verifier дополнительно подтвердил семь уникальных AppRole/token identities без
broad/shared token, одноразовое разворачивание wrapping token, `renew-self`,
точную границу plugin instance, Transit только для exchange identity, soft
delete/undelete, rollback содержимого новой версией, отсутствие запрещённых
данных в stdout/stderr, audit, metadata и backup tree и полную очистку среды.
Три независимых временных PGP keyring получили зашифрованные shares; для unseal
использованы ровно `2` из `3`. Временная схема `1/1` существовала только на новом
пустом recovery volume до force restore и не стала схемой восстановленного Raft.

## Результат

Добавлен закреплённый по manifest digest контейнер OpenBao `2.5.4` в
не-dev режиме с Raft на отдельном томе, декларативным HMAC-защищённым audit
device и закрытой по умолчанию моделью служебных политик. Реализованы семь
раздельных идентичностей: API, exchange execution, identity, notification
dispatcher, конкретный plugin instance, secret operator и backup recovery.

Значения секретов больше не являются конфигурационным контрактом приложений.
Каноническая ссылка имеет форму
`openbao://<mount>/<root>/<kind>/<resource>?version=<positive-int>#<field>`.
Типы `exchange`, `telegram`, `oidc`, `plugins` и `storage`, корневой namespace,
версия и поле проверяются до обращения к OpenBao. Возвращаемое значение имеет
редактированное строковое представление и не сериализуется API/UI.

Учетные данные служебной идентичности передаются только через абсолютный
mode-`0600` файл без symlink; файл перечитывается перед каждым запросом. Это
позволяет ротацию без удержания значения в конфигурационном объекте. OIDC,
exchange и storage границы Stage `08` не имеют raw-env fallback. Перевод
существующего notification-dispatcher/Telegram worker принадлежит Stage `11`;
до его принятия этот legacy path не является разрешённым runnable path
greenfield v1. API не получает права чтения значений или Transit decrypt, а
`exchange-control` не принимает даже путь к API Transit credential.

Каноническая эксплуатационная инструкция:
`docs/runbooks/openbao-secrets-and-recovery.md`.

## Контейнер и хранение

- Образ:
  `ghcr.io/openbao/openbao:2.5.4@sha256:436eaf9778cad75507ff70ea26ace30dcbe15606e619ac3823495663d7f7c115`;
  manifest поддерживает `linux/amd64` и `linux/arm64`.
- Контейнер запускается как `100:1000`, с `read_only`, `cap_drop: ALL`,
  `no-new-privileges`, ограниченным `tmpfs` и loopback publish.
- Raft хранится в `openbao-data`, audit — в отдельном `openbao-audit`; writable
  layer не является источником восстановления.
- Listener использует HTTP только внутри локальной container boundary;
  опубликованный порт ограничен `127.0.0.1`. Stage `17` обязан подключить к
  service network только разрешённых потребителей и не открывать порт наружу.
- OpenBao UI отключён; audit задаётся конфигурацией до первого запроса,
  `log_raw=false`, mode `0600`.

Официальная документация подтверждает поддерживаемые container registries и
запуск `server` ([установка OpenBao](https://openbao.org/docs/install/)), Raft
storage и snapshots ([storage](https://openbao.org/docs/configuration/storage/),
[Raft operations](https://openbao.org/docs/next/commands/operator/raft/)), а
также декларативную audit-конфигурацию
([audit RFC](https://openbao.org/docs/rfcs/config-audit-devices/)).

## Инициализация, владение и служебные политики

Production policy фиксирует `3` Shamir shares с threshold `2`. Shares и
начальный административный credential должны заранее шифроваться публичными
PGP-ключами владельцев и записываться в owner-controlled mode-`0600` files.
Initial administrator отзывается после bootstrap. Ничего из этого не выводится
в terminal, CI, report или чат.

Service authentication использует AppRole с response wrapping; runtime token
доставляется в `tmpfs`-файл, имеет TTL `15m`, maximum TTL `30m` и не получает
policy `default`. Wrapping token имеет TTL `5m`, разворачивается ровно один раз,
а service token успешно проходит `renew-self`. Каждая identity имеет
самостоятельный role/token и ровно одну политику. Начальный административный
credential PGP-зашифрован и отзывается после bootstrap. Модель следует
deny-by-default policy semantics и документированному AppRole workflow
([policies](https://openbao.org/docs/next/concepts/policies/),
[AppRole](https://openbao.org/docs/auth/approle/)).

| Идентичность | Разрешено | Явно не разрешено |
|---|---|---|
| `api` | health и KV metadata | KV values, Transit decrypt, root/unseal |
| `exchange-execution` | exchange KV, encrypt/decrypt/HMAC одного Transit key | OIDC, Telegram, plugin, storage values |
| `identity` | OIDC KV | exchange/Telegram/plugin/storage values, decrypt |
| `notification-dispatcher` | Telegram KV | остальные kinds и decrypt |
| `plugin-runtime` | один exact organization/instance path | wildcard organization/plugin access |
| `secret-operator` | create/update/read, soft delete/undelete | irreversible version destroy |
| `backup-recovery` | Raft snapshot read | service values и force restore |

Принудительное восстановление нового пустого storage не включено в постоянную
service policy: его можно явно вызвать только recovery-командой с
`--force-new-storage` и отдельной recovery authority.

## Ротация и ссылки

`SecretReference` отклоняет неверную схему, неизвестный kind, другой root,
path traversal, percent-encoded traversal, лишние query-параметры, нулевую
версию и ссылку без поля. Resolver обращается к KV v2 endpoint и выбирает
положительную версию, если она зафиксирована в ссылке.

Runtime proof создал версии `1` и `2`, выполнил soft delete и undelete версии,
затем rollback содержимого как новую версию `3`. История сохраняется; version
destroy не входит в operator policy. Это соответствует KV v2 versioned data и
delete/undelete semantics
([KV v2](https://openbao.org/docs/2.3.x/secrets/kv/kv-v2/)).

## Резервное копирование и восстановление

`infra/openbao/snapshot.py` получает стандартный Raft snapshot с отдельной
mode-`0600` credential boundary, не записывает plaintext snapshot на диск и
передаёт байты через stdin в `age`. Результат атомарно сохраняется на отдельном
томе с mode `0600`; sidecar содержит только время, размер ciphertext и SHA-256
ciphertext.

Restore сначала проверяет sidecar, размер и SHA-256 ciphertext, затем
расшифровывает snapshot только в памяти. Обычный endpoint используется для
штатного восстановления, а force endpoint — только при явном
`--force-new-storage` и после fail-closed проверки свежести mounts, auth methods
и policies. Runtime proof удалил исходный том, инициализировал новый пустой
Raft, выполнил force restore, обязательно перезапустил контейнер для загрузки
восстановленной seal-конфигурации и подтвердил исходные versions/policies после
unseal исходным quorum `2` из `3`. Все временные контейнеры, тома, credentials,
recovery identity и backup удалены.

## Состояния и `ops.roehub.io/v1`

`OpenBaoReadiness` различает `uninitialized`, `sealed`, `unsealed` и
`unavailable`. Только `unsealed` имеет `ready=true`; остальные состояния
закрывают secret-dependent операции и направляют к
`auth.openbao-unavailable`. Канонический runbook обновлён до revision `2`,
получил русский locale/render и три problem mappings.

## Проверки

- Docker Engine `29.5.2`, Compose `5.3.1`, `age 1.3.1` — доступны.
- `docker compose -f infra/docker/openbao-embedded.compose.yml config --quiet`
  — `passed`.
- OpenBao runtime/recovery/forbidden-output proof — `passed` на реальном
  контейнере и новых Docker volumes.
- Scoped `ruff` — `passed`.
- Scoped `pyright` — `0 errors, 0 warnings`.
- Focused Stage `08` suite — `74 passed`.
- Расширенная API/identity/exchange/platform/infra regression —
  `468 passed`.
- Runtime input inventory — `passed`, `138` имён без значений.
- Runbook generation/check — `passed`: 6 русских документов, 13 problem
  mappings, 21 ещё не перенесённая historical инструкция.
- Docs index generation/check — `passed`.
- Project map generation/check — `passed`, 5 артефактов.
- Финальный `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Secret reference DTO | `breaking-change` | Значение заменено строгой typed opaque ссылкой с обязательным field и optional version. |
| Common resolver API | `compatible-change` | Добавлен общий fail-closed resolver/credential-file contract. |
| OIDC runtime config | `breaking-change` | Raw client credential заменён `IDENTITY_OIDC_CLIENT_SECRET_REF` и отдельным file credential. |
| Exchange Transit config | `breaking-change` | Raw token inputs заменены mode-`0600` file paths; API Transit credential исключён из exchange-control. |
| Installation schema/generated config | `breaking-change` | Storage refs получили обязательный field и OpenBao file-auth contract. |
| Persistence | `breaking-change` | OpenBao Raft/audit/backup получают отдельные durable volumes и lifecycle. |
| Identity/authorization | `breaking-change` | Shared broad tokens запрещены; каждая служба имеет отдельную policy/TTL. |
| Request/cache/resource identity | `none` | Hash identity продукта не менялась; изменены только golden config hashes. |
| Межсервисные вызовы | `compatible-change` | Появился общий KV v2 resolver; существующий Transit adapter сохраняет interface. |
| Внешние эффекты | `compatible-change` | Только disposable OpenBao/backup; production mutation отсутствует. |
| Audit/runbook/operations | `compatible-change` | Добавлены безопасные состояния, audit и recovery actions. |
| Browser/API/UI output | `none` | Значения, ciphertext и credentials не возвращаются наружу. |

Основная классификация Stage `08` — `breaking-change`: raw secret inputs,
shared authority и container-local recovery намеренно несовместимы с
greenfield v1.

## Файлы этапа

Созданы:

- `infra/docker/openbao-embedded.compose.yml`;
- `infra/openbao/config/openbao.hcl`, семь policy-файлов,
  `infra/openbao/snapshot.py`, `infra/openbao/verify_runtime.py`;
- `configs/openbao/{bootstrap,backup,service-identities}.yaml`;
- `src/trading/platform/secrets/{__init__,reference,openbao,transport}.py`;
- tests для secret reference/resolver/assets/runtime boundaries;
- `docs/runbooks/openbao-secrets-and-recovery.md` и этот report.

Изменены:

- installation schema/config/golden hashes/runtime inventory;
- OIDC wiring и architecture contract;
- exchange-control Transit/API client credential boundaries;
- OpenBao canonical/locale/generated runbook и runbook index/test;
- platform plan и generated docs/project-map outputs.

Удалённых tracked-файлов нет. `.codex/PLANS.md`, license/governance,
предыдущие stages и unrelated dirty changes сохранены. Commit, staging, push,
deploy и production mutation не выполнялись.

## Холодная проверка

- Режим: единственная проверка `independent subagent`; после исправлений —
  только локальная повторная проверка, без второго независимого review.
- Первоначальный вердикт: `Block`.
- Исправлены обязательные замечания: исключена сериализация `SecretValue`;
  запрещены HTTP redirects с передачей `X-Vault-Token`; OIDC credential теперь
  разрешается перед каждым token POST; AppRole доказывает response wrapping,
  одноразовый unwrap и renewal; cleanup закрывает проверку при ошибке teardown;
  credential/snapshot files читаются через один `O_NOFOLLOW` descriptor;
  force restore проверяет fresh storage и digest; runtime proof использует PGP
  `3/2`, не содержит критических `assert`; scope raw-env утверждения сужен до
  реально переведённых consumers.
- Локальный итог после исправлений: `Release after fixes`; `ruff`, `pyright`,
  `74` focused tests, `468` expanded tests и повторный Docker/OpenBao proof
  прошли.
- Остаточные риски: сетевую топологию consumers формирует Stage `17`; общую
  lifecycle-репетицию backup/restore повторит Stage `21`; миграция старого
  Telegram worker остаётся обязательной частью Stage `11`.

## Передача Stage 09

Stage `09` получает typed reference contract, пять secret kinds, API denial,
раздельные service identities, AppRole wrapping/renewal, file credential
rotation и fail-closed `ops.roehub.io/v1` readiness. Значения секретов не
являются входом Stage `09`; production OpenBao и legacy data не затрагиваются.

## Повторная проверка для этапа 22

После разрешения владельца OpenBao повторно собран как
`2.5.4-roehub-licensed-qr.1`: нелицензированный
`github.com/yeqown/reedsolomon` заменён MIT-лицензированным
`github.com/skip2/go-qrcode`, а JWT QR-функция сохранена. Точный image digest —
`sha256:610395fc927391e2cfa4e082ba9cb520a8359b2c14591a9ff63378bf0c52225b`.
Module graph, ELF build info, `TestPrintQR`, повторная OCI-сборка и реальный
OpenBao runtime proof прошли. Изменение внутренней QR-библиотеки имеет
классификацию `compatible-change`; container identity — `breaking-change` до
первого опубликованного релиза.
