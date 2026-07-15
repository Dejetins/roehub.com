# Локальная аутентификация, сессии и восстановление v1

## Пользовательский результат

Чистая самостоятельно разворачиваемая установка Roehub не зависит от
Keycloak. Первый владелец создаётся локальной одноразовой операцией, привязывает
ключ доступа WebAuthn и после этого входит по ключу доступа как основному
способу. Регистрация извне закрыта. Опциональный пароль и одноразовые коды
восстановления являются резервными способами; TOTP в v1 не включён и не имеет
скрытого или частично настроенного endpoint.

Первичная настройка одновременно создаёт `installation_owner`, первую
организацию и членство `owner`, поэтому сохраняет принятые в Stage `05`
инварианты владения. Импорт существующих production users, Keycloak subjects,
сессий или credentials отсутствует.

## Граница доверия

Application service в `identity` владеет WebAuthn ceremonies, fallback
проверками, rate limiting, recovery и выпуском сессии. Inbound FastAPI adapter
владеет same-origin/CSRF policy и cookie. PostgreSQL adapter атомарно хранит
одноразовые bootstrap/challenge/recovery переходы и неизменяемый аудит.
Browser получает только публичные WebAuthn options, opaque cookie и восемь
кодов восстановления один раз непосредственно после bootstrap.

Сервер никогда не хранит:

- исходное значение bootstrap-файла;
- исходный пароль;
- исходные recovery codes;
- private key ключа доступа;
- provider token в локальном режиме.

Сервер хранит SHA-256 одноразового bootstrap-значения и subject, Argon2id
пароля/recovery codes, публичный WebAuthn credential, challenge SHA-256,
server-side session и безопасный аудит без credential payload.

## Первичная настройка

Локальная команда:

```bash
IDENTITY_PG_DSN=<host-local-reference> roehubctl local-auth-bootstrap \
  --output-file /safe/local/path/bootstrap.txt
```

создаёт один активный ticket сроком на 15 минут. Новый вызов инвалидирует
предыдущий. Файл создаётся эксклюзивно с mode `0600`; stdout содержит только
статус, путь и время истечения. Значение не принимается через argv/stdout и не
записывается в журнал.

Browser загружает этот файл через file input и держит его содержимое только в
памяти на время `POST /auth/local/bootstrap/options`. Успешная WebAuthn
registration в одной PostgreSQL-транзакции:

1. потребляет bootstrap ticket и challenge;
2. создаёт локального пользователя и account;
3. создаёт singleton installation, installation owner, первую organization и
   membership `owner`;
4. сохраняет публичный passkey credential;
5. сохраняет восемь Argon2id recovery hashes;
6. добавляет административный и auth audit;
7. выпускает opaque session.

Повторное использование ticket/challenge и параллельный второй bootstrap
отклоняются advisory lock и условными `UPDATE ... RETURNING`.

## WebAuthn и резервные способы

WebAuthn использует discoverable credential, `residentKey=required` и
`userVerification=required`. Challenge хранится только как SHA-256 и живёт пять
минут. Registration/authentication проверяют challenge, RP ID, exact origin,
user verification, credential public key и sign counter. Dev RP ID —
`localhost`; production требует явные HTTPS origin и совпадающий RP host.

`webauthn==2.5.1` и `argon2-cffi==25.1.0` закреплены в lockfile. Версия WebAuthn
совместима с принятой в проекте веткой `cryptography`; обновление выполняется
отдельным dependency review.

Опциональный password fallback хранится только как Argon2id hash. Если пароль
не задан при bootstrap, password endpoint выдаёт ту же публичную ошибку, что и
неизвестный username или неверное значение. Recovery code одноразовый:
успешное потребление отзывает все предыдущие сессии пользователя и выпускает
новую; replay отклоняется.

Успешный свежий passkey, password или recovery вход создаёт новую сессию и
является свежей аутентификационной церемонией. Если session старше пяти минут,
привилегированное добавление passkey требует `recent-auth` WebAuthn ceremony.
Успешный `recent-auth` создаёт новую сессию, подтверждённо отзывает старую и
компенсирующе отзывает новую при любой ошибке ротации.

## Сессии и CSRF

Session ID — случайный UUID, который адресует серверную запись и не содержит
identity claims. Cookie:

- `HttpOnly` для session;
- `Secure` в production;
- `SameSite=lax`;
- общий ограниченный `Path`;
- срок не больше absolute session TTL.

Server-side запись содержит idle и absolute expiry, revocation timestamp и
пользователя. Отсутствующая, неизвестная, отозванная или истёкшая сессия
возвращает одинаково закрытый `401`.

Все pre-auth POST требуют exact same-origin signal. Все authenticated mutations
дополнительно требуют double-submit `roehub_csrf` cookie/header. Logout
отзывает server-side session и удаляет обе cookie. Auth/options/complete ответы
имеют `Cache-Control: no-store`.

## Rate limiting, ошибки и аудит

Password, recovery и passkey failures учитываются в 15-минутном окне. Пять
неудач блокируют subject на 15 минут. Публичные auth failures имеют единый
`authentication_failed`, чтобы не раскрывать существование username,
credential или recovery code. Bootstrap возвращает отдельные безопасные коды
только для состояния установки и валидации нового пароля.

`identity_local_auth_events` append-only и хранит user ID при его наличии,
SHA-256 subject, фиксированное action/outcome/reason и время. Migration
запрещает чувствительные ключи и `UPDATE`/`DELETE`. Значения credentials,
cookies, tokens и исходный provider payload в аудит не попадают.

## API

Публичная локальная поверхность:

- `GET /auth/local/status`;
- `POST /auth/local/bootstrap/options|complete`;
- `POST /auth/local/passkey/options|complete`;
- `POST /auth/local/password`;
- `POST /auth/local/recovery`;
- `POST /auth/local/logout`.

Аутентифицированная поверхность:

- `GET /auth/current-user` — provider-neutral endpoint;
- `POST /auth/local/passkeys/options|complete`;
- `POST /auth/local/recent-auth/options|complete`.

Local auth всегда включён. OIDC routes появляются только при полном явном
наборе provider settings и остаются дополнительным способом Stage `07`, а не
условием base profile.

## Persistence

Migration `local-auth-0012` добавляет:

- `identity_local_accounts`;
- `identity_webauthn_credentials`;
- `identity_local_bootstrap_tickets`;
- `identity_local_auth_challenges`;
- `identity_local_recovery_codes`;
- `identity_local_auth_rate_limits`;
- `identity_local_auth_events`.

Manifest phase SHA-256 —
`80655fe744ec74df03816d7a7f74ae8ec5e910b5f0c2569d88d2c13a626f2a44`,
а файл `0012_identity_local_auth_v1.sql` —
`89710347fdc39ed2cf7075c11336319ac1b74f81a1395d0e2bccdc309c5650e4`.

## Совместимость

| Поверхность | Классификация | Последствие |
|---|---|---|
| Local-auth API/DTO | `compatible-change` | Добавлены новые routes; старый OIDC контракт не удалён. |
| Base identity semantics | `breaking-change` | Local passkey становится обязательным базовым способом вместо обязательного Keycloak. |
| Identity application ports | `compatible-change` | Добавлены local-auth и user/session lifecycle методы. |
| PostgreSQL persistence | `breaking-change` | Greenfield schema получает обязательную phase `local-auth-0012`. |
| Runtime config | `breaking-change` | Production требует явные `IDENTITY_LOCAL_RP_ID`, `IDENTITY_LOCAL_RP_NAME`, `IDENTITY_LOCAL_ORIGIN`. |
| Cookie/session semantics | `breaking-change` | Server-side rotation, CSRF, expiry и recent-auth становятся обязательными. |
| Organization/RBAC | `compatible-change` | Bootstrap реализует принятые Stage `05` owner invariants. |
| Request hash/cache/resource identity | `none` | Product resource identity не менялась. |
| Browser defaults | `breaking-change` | `/login` показывает passkey первым, fallback вторично, `/register` закрыт. |

Rollback заполненной legacy базы не поддерживается и не нужен для greenfield
v1. Внутри новой установки rollback выполняется восстановлением тома до phase
`local-auth-0012`; backup/restore доказательство относится к Stage `21`.

## Доказательства и передача

Stage `06` доказал migration на реальном PostgreSQL, hash-only bootstrap,
single-active ticket, passkey counter, recovery replay, rate limit и append-only
audit. Реальный Chromium с виртуальным CTAP2 authenticator доказал bootstrap,
passkey registration/login, password fallback, recovery, CSRF, logout,
session rotation, expiry и закрытую регистрацию на desktop/mobile layout.

Stage `07` добавит provider-neutral OIDC linking/degradation, сохраняя локального
владельца как независимый fallback. Stage `08` перенесёт остальные secret
references и operational recovery в OpenBao. Stage `19` добавит полный
административный browser UI.
