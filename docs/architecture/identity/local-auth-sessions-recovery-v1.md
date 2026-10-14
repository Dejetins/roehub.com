# Локальная аутентификация, сессии и восстановление v1

**Уточнение целевой архитектуры от 2026-07-16:** этот документ сохраняется как
доказательство реализованного local-auth контура. Требования к будущему
пользовательскому первому запуску и host-local recovery зафиксированы в
[`roehub-product-transformation-requirements-v1.md`](../platform/roehub-product-transformation-requirements-v1.md).
Они не выбирают конкретную реализацию; до отдельного route/API cutover текущий
контракт остаётся совместимым runtime.

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

Local auth всегда включён. С 2026-10-13 внешний OIDC удалён: provider settings,
login/link/callback/status routes больше не поддерживаются. Локальные сессии,
восстановление, CSRF и recent-auth сохраняются.

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
| Local-auth API/DTO | `compatible-change` | Локальные routes сохранены; удаление OIDC в 2026-10-13 — отдельный `breaking-change`. |
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

Исторические Stage `07`, `08`, `19` не являются текущим планом исполнения.
OIDC из Stage `07` удалён 2026-10-13. OpenBao и локальный административный UI
сохранены; локальная аутентификация является действующим способом входа.

## Устройства и управление входами (2026-10-10)

В настройках безопасности единицей списка является **профиль браузера текущего
аккаунта**, а не каждая выдача session cookie. Физическое устройство нельзя
достоверно определить по IP или User-Agent: два браузера, приватное окно и профиль
после удаления cookie могут отображаться отдельными устройствами. Название
браузера/ОС — информационная подпись, не фактор авторизации.

### Идентичность и жизненный цикл

- Отдельная `roehub_browser` cookie (`HttpOnly`, `SameSite=lax`, `Path=/`, `Secure`
  в production, срок один год) переживает logout. Она не предоставляет доступ.
  База хранит только SHA-256 маркера; публичный `device_id` — отдельный UUID.
- `identity_browser_devices` принадлежит identity context. `(user_id, device_key)`
  уникален; устройство ссылается на последнюю выданную сессию. История сессий
  сохраняется в `identity_sessions`; bearer `session_id` никогда не входит в DTO
  устройств. Старый `GET /ui/account/sessions` сохраняет прежний контракт.
- Все успешные local-auth flows (bootstrap, passkey, password, recovery,
  recent-auth) регистрируют устройство **до** выдачи auth cookie.
  При ошибке регистрации новая сессия компенсирующе отзывается; cookie не выдаётся.
- Регистрация и owner-scoped отзыв устройства сериализуются блокировкой строки
  пользователя. При повторном входе в том же профиле предыдущие сессии этого
  устройства отзываются атомарно, а `device_id` сохраняется. В остальных профилях
  вход остаётся действителен. Одновременные входы оставляют одну действующую сессию;
  ответ проигравшего конкурентного входа может содержать уже отозванную cookie и
  получает обычный `401`, без автоматического восстановления старой сессии.
- Авторизованные запросы обновляют `last_seen_at` с точностью до минуты.
  Запись условная: отозванные/истёкшие сессии не оживают. Время создания и оба
  срока действия не меняются; recent-auth и TTL-политика остаются прежними.
  Это время обращения с действующей сессией, включая фоновые запросы, не датчик
  присутствия человека онлайн.

### API и интерфейс

- `GET /ui/account/devices?limit=20&cursor=<device UUID>`: `items`, `next_cursor`,
  `legacy_active_count`. Размер страницы 1–50, стабильный keyset по UUID DESC.
  Элемент: `device_id`, `label`, `last_seen_at`, `expires_at`, `active`, `is_current`.
  Список ограничен владельцем из principal; `is_current` сравнивает внутреннюю
  ссылку на сессию, а не IP/маркер клиента. Ответ `Cache-Control: no-store`.
- `POST /ui/account/devices/{device_id}/revoke`: идемпотентный отзыв всех входов
  устройства владельца; чужой/неизвестный ID не раскрывает его существование.
  Ответ `{revoked: true, current: boolean}`. Для текущего устройства удаляются
  session/CSRF cookie, клиент очищает приватный кэш и переходит на `/login`.
- `POST /ui/account/devices/legacy/revoke`: отзывает только неизвестные старые
  входы владельца, сохраняя доказанную текущую сессию. Оба POST требуют same-origin
  и double-submit CSRF; дополнительных полномочий организации или recent-auth
  для завершения собственных входов не требуется.
- При первом чтении доказанная текущая legacy-сессия получает устройство.
  Остальные старые записи не объединяются по догадке: показывается одна группа
  «Ранние входы» с количеством ещё действующих входов и общим завершением.
  Их исторический `last_seen_at` не представляется как измеренная активность.
- Вкладка «Устройства»: браузер/ОС, «Это устройство», последняя активность в UTC,
  «Действителен»/«Завершён», действие завершения. Действующий вход не означает
  онлайн. Повторный login обновляет строку, а не добавляет новую.
- Применяется общий read-snapshot contract: тихое обновление, недоступность команд
  на retained/error snapshot, немедленное скрытие приватных строк при deny чтения
  **или команды**, отсутствие автоматического повтора команды с неизвестным
  результатом. Вкладка «События» сохраняет существующее назначение.

### Совместимость, порядок обновления и проверка

Миграция `0032_identity_browser_devices_v1.sql`, фаза
`identity-browser-devices-0032`, применяется через текущий storage lifecycle и
checksum manifest **до** обновления API. Она расширяет схему и не переписывает
старые входы. Затем обновляются все issuers/resolvers и UI. Откат приложения без
отката схемы допускает прежний login, но старое приложение не гарантирует
группировку/отзыв на повторном входе; это ограничение отката, не новая гарантия.

API DTO и persistence: `compatible-change` (новые endpoints/таблица/nullable FK,
старые endpoints сохранены). Application session port: `breaking-change` для
реализаций порта, оба поддерживаемых адаптера обновлены совместно.
Auth lifecycle: намеренный `breaking-change` — прежняя cookie того же профиля
больше не остаётся действительной после повторного входа. TTL, user/org scopes,
CSRF и прочие recent-auth gates: `none`.

Проверки: `tests/unit/apps/api/test_browser_devices.py`, существующие identity
route/use-case tests, `tests/runtime/test_browser_devices_postgres.py` (реальный
PostgreSQL, disposable schema, конкурентные login, owner isolation, activity,
legacy revoke, pagination и повторное применение миграции), UI tests
`session-devices.test.tsx` и `settings-page.test.tsx`. PostgreSQL-тест требует
`IDENTITY_PG_DSN`; пропуск без DSN не доказывает SQL-поведение.
