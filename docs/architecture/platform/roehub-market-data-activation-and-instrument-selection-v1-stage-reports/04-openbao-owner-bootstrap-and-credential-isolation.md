# Этап 04 — OpenBao: owner bootstrap и изоляция учётных данных

## Результат

Этап принят на границе `disposable_runtime_and_owner_handoff`. Добавлена
отдельная команда `roehubctl openbao-owner-init`, не основанная на runtime
verifier и не использующая legacy-схему `1/1`.

Команда `initialize` принимает ровно три разных public PGP recipient,
инициализирует новую пустую установку с `3` shares и порогом `2`, а затем
сохраняет только PGP-шифрованные `unseal-share-*.pgp` и
`initial-admin.pgp` в private delivery-каталоге. Она не создаёт и не
расшифровывает private PGP key, unseal share или token. Повторный безопасный
запуск с уже подготовленным delivery-каталогом выполняет только проверку
состояния и возвращает `already_initialized`.

После owner unseal команда `provision-services` создаёт KV v2, Transit,
AppRole, ограниченные policy и семь статических service identity. Для каждой
identity она сохраняет отдельные `role-id` и response-wrapped one-time
`wrapped-secret-id`; TTL wrapper не превышает `5m`. После успешной выдачи
initial administrator credential отзывается.

`plugin-runtime` намеренно не имеет общего AppRole: его policy и credential
должны создаваться только для точного `organization_id/instance_id` при
установке plugin instance. Это исключает межорганизационный универсальный
plugin token.

## Доказательство на реальной границе (`runtime smoke`)

Изолированный Docker Desktop `runtime smoke` использовал уже сохранённый
offline OpenBao image с `pull_policy: never`; внешний registry не запрашивался.
Проверка подтверждает:

- состояния `uninitialized`, `sealed`, `unsealed`;
- три PGP recipient и threshold `2-of-3`;
- PGP-шифрование initial admin material;
- `8` policy identities в disposable proof, включая instance-scoped plugin
  policy; owner-команда выдаёт `7` только статических identities;
- response wrapping, одноразовый unwrap, delivery mode `0600`, renewal
  short-lived service token и отзыв bootstrap credential;
- запрет API на доступ к KV values и Transit decrypt, отсутствие shared broad
  token;
- cleanup контейнеров, сетей и volume после проверки.

Подробное sanitized evidence:
[`04-openbao-owner-bootstrap-proof.json`](evidence/04-openbao-owner-bootstrap-proof.json).
В нём отсутствуют PGP fingerprint, private key, unseal share, SecretID,
wrapping token, administrator token и provider payload.

## Контрактное влияние

| Поверхность | Класс | Переход и откат |
|---|---|---|
| CLI | `compatible-change` | Добавлена opt-in команда; существующие CLI-маршруты не изменены. |
| Persisted schema | `none` | Миграции и пользовательские данные не затронуты. |
| OpenBao owner bootstrap | `breaking-change` | Только новая empty installation; после `init` запрещён повтор без проверенного delivery-каталога. |
| Service credentials | `breaking-change` | После явного owner provisioning сервис получает только свой wrapped delivery; rollback — оставить OpenBao `sealed` и удалить неиспользованный tmpfs delivery. |
| Browser/API | `none` | UI и публичные API не изменялись. |
| Logging/evidence | `compatible-change` | Результаты содержат только состояние, количество и policy outcome; секретные значения исключены. |

## Проверки

- `21 passed` для owner-init CLI, OpenBao assets и secret resolver;
- Ruff — успешно;
- Pyright — `0 errors, 0 warnings`;
- `docker compose ... config --quiet` с offline override — успешно;
- отдельный `runtime smoke` OpenBao — успешно;
- JSON evidence, генератор индекса документации, генератор runtime topology и
  `git diff --check` — успешно.

## Handoff владельцу

Durable PGP public material, приватные ключи получателей, расшифрованные shares
и initial admin token не создавались в этой рабочей среде. Реальная
инициализация новой установки выполняется владельцем по
[runbook](../../../runbooks/openbao-secrets-and-recovery.md) через approved
local source. Секреты не нужно и нельзя передавать в чат.

Следующий разрешённый этап: `05` — пересборка локального release candidate,
greenfield lifecycle, браузерная проверка и scoped remote sync. Исходный
blocker native `linux/amd64` остаётся внешним и не снимается этим этапом.

## Холодная самостоятельная проверка

`cold self-review fallback`: проверены новая CLI boundary, режимы файлов,
идемпотентный stop gate, политики, AppRole delivery, runtime evidence,
документация и очистка Docker. Вердикт: `Release after fixes` — offline
compose override добавлен после подтверждённой ошибки доступа к registry.
Остаточный риск: durable owner custody и instance-specific plugin credential
требуют последующего явного действия владельца; production не изменялся.
