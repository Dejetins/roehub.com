# Универсальный OIDC-провайдер аутентификации v1

## Пользовательский результат

Чистая установка Roehub по умолчанию запускается без внешнего провайдера и
сохраняет локальный вход владельца. При явной настройке внешний OIDC становится
дополнительным способом входа: Keycloak, Pocket ID и другие совместимые
провайдеры проходят через один `AuthenticationProvider/v1`, а не получают
собственную доменную модель.

Внешняя идентичность не является пользователем Roehub. Проверенная тройка
`(provider_id, issuer, subject)` указывает на стабильный внутренний `user_id`.
Новый пользователь и членство создаются только по действующему приглашению;
существующий локальный пользователь может привязать OIDC только из свежей
аутентифицированной сессии. Конфликты отклоняются без замены связи.

Текущие production users, `keycloak_subject`, пароли, TOTP-секреты, сессии и
provider credentials не читаются и не импортируются.

## Граница `AuthenticationProvider/v1`

Application port владеет двумя операциями:

1. построить authorization URL с новым `state`, `nonce` и PKCE `S256`;
2. один раз обменять code и вернуть только `VerifiedExternalIdentity` после
   проверки issuer, подписи, audience, времени, nonce и subject.

HTTP adapter получает discovery document и JWKS, принимает только `RS256`,
проверяет exact issuer и HTTPS endpoints. HTTP разрешён только явному
изолированному dev fixture и запрещён production wiring. Redirect URI задаётся
ровно одной настройкой клиента и передаётся одинаково в authorization и token
requests.

Заявленные пределы сети являются максимумами, а не рекомендациями:

| Бюджет | Значение по умолчанию | Допустимое изменение |
|---|---:|---|
| соединение | 3 секунды | только уменьшить |
| ответ | 10 секунд | только уменьшить |
| общий deadline | 15 секунд | только уменьшить |

Discovery/JWKS GET выполняет не больше трёх попыток, то есть не больше двух
повторов, только после transport error, `429` или `5xx`. Между попытками есть
ограниченный jitter, каждая попытка остаётся внутри общего deadline. Cache
учитывает `no-store` и `max-age`, ограничивая TTL одним часом. Неизвестный `kid`
разрешает одно принудительное обновление JWKS.

Каждая сетевая операция выполняется за отменяемой для вызывающего кода
wall-clock границей: после исчерпания общего бюджета caller получает закрытый
отказ, даже если transport ещё завершает одно уже начатое чтение в daemon
worker. Поздний результат игнорируется и не может создать сессию. Transport
timeout самого worker также ограничен остатком бюджета; новая попытка token POST
не создаётся.

Token POST выполняется ровно один раз. Timeout, transport error, `429` или
`5xx` после отправки означают `token_result_unknown`: попытка потребляется,
сессия не создаётся, повтор POST запрещён, пользователь начинает новую
церемонию. Перед внешним POST repository атомарно переводит попытку в
`exchange_started_at`; при двух одновременных callback только один получает
claim и достигает провайдера.

Постоянный provider adapter не удерживает значение client credential. Он
получает resolver callback и вызывает его непосредственно перед каждым
единственным token POST, уже внутри общего deadline. `SecretValue` живёт только
в локальной области этого запроса и не сериализуется. Поэтому один и тот же
экземпляр provider видит новую версию OpenBao при следующей церемонии без
перезапуска процесса; недоступный resolver закрывает запрос до внешнего POST.

## Церемония и сессия

`OidcAuthenticationService` хранит одноразовую попытку 10 минут. Browser cookie
содержит только opaque `attempt_id`; исходные state, nonce и provider payload в
cookie не попадают. В PostgreSQL сохраняются SHA-256 state/nonce, server-side
PKCE verifier, назначение `login|link`, optional linking user и безопасный
`next_path`.

Callback выполняет проверки в таком порядке:

1. действующая неприменённая попытка и binding к provider/issuer;
2. constant-time сравнение state hash;
3. для linking — совпадение свежей callback session с `linking_user_id`;
4. атомарный claim права на единственный token exchange;
5. единственный token exchange;
6. подпись и claims;
7. повторная проверка текущего времени и атомарное отображение или provisioning;
8. новая opaque Roehub session только для успешного login.

Linking не меняет текущую локальную сессию. Route `/auth/oidc/link` требует,
чтобы session была создана не более пяти минут назад. Один пользователь может
иметь не больше одной identity одного provider/issuer, а один внешний subject —
не больше одного пользователя.

## Приглашение и атомарность

Для неизвестного subject обязателен проверенный email и хотя бы одно
непросроченное `identity_invitations` с exact SHA-256 recipient. PostgreSQL
adapter в одной транзакции:

1. блокирует попытку и подходящие приглашения;
2. создаёт внутреннего пользователя;
3. создаёт external identity link;
4. создаёт memberships с ролями приглашений;
5. переводит приглашения в `accepted`;
6. потребляет попытку и записывает аудит.

Без приглашения транзакция откатывается: orphan user и membership не остаются.
Issuer/subject conflict, повторная привязка provider к другому subject и попытка
захвата уже связанного subject получают закрытый отказ.

## API и конфигурация

При полном явном наборе OIDC-настроек доступны:

- `GET /auth/oidc/status`;
- `GET /auth/oidc/login`;
- `GET /auth/oidc/link`;
- `GET /auth/oidc/callback`.

`GET /auth/current-user` остаётся provider-neutral. Локальные `/auth/local/*`
всегда собираются независимо. В `roehub.yaml` значение по умолчанию — только:

```yaml
oidc:
  enabled: false
```

Включённый installation input требует `provider_id`, `display_name`, HTTPS
`issuer`, `client_id`, `client_secret_ref` и HTTPS `redirect_uri`. Stage `08`
заменил временную raw-инъекцию на `IDENTITY_OIDC_CLIENT_SECRET_REF`,
`OPENBAO_ADDR`, `ROEHUB_OPENBAO_ROOT` и защищённый файл служебной идентичности
`ROEHUB_IDENTITY_OPENBAO_TOKEN_FILE`. API разрешает ссылку с типом `oidc`
только внутри доверенной границы перед созданием provider adapter; ссылка и
значение не возвращаются через API/UI и не копируются в отчёты или
доказательства. Timeout fields необязательны и могут только ужесточить пределы
3/10/15 секунд; недоступный или sealed OpenBao блокирует только сборку
включённого OIDC provider, не создавая raw-env fallback.

API и Web production entrypoints сохраняют access logs, но перед форматированием
заменяют весь query OIDC callback на `?redacted`. Поэтому authorization code и
state не попадают ни в Web, ни в API access log; proxy error также не возвращает
сырой URL или transport payload.

## Хранение и аудит

Migration phase `oidc-provider-0013` добавляет:

- `identity_oidc_login_attempts`;
- `identity_external_identities`;
- append-only `identity_oidc_auth_events`.

`identity_oidc_login_attempts.exchange_started_at` является durable claim:
пустое значение допускает один exchange, установленное — запрещает повтор даже
до записи окончательного результата. Истёкшая во время сетевого обмена попытка
отклоняется по новому `completed_at` и не создаёт user/session.

Manifest SHA-256 —
`65a95708d424346a486add22367fc7dd113ee499e9faa5e03f533cd2b1e05d4f`, файл
`0013_identity_oidc_provider_v1.sql` —
`3a1c1d713fdf920eebd420ef66294635eb47cec0892103c2277dc18fc4892628`.

Subject и verified email сохраняются только как SHA-256. Audit содержит
provider ID, internal user ID при наличии, subject hash, bounded
action/outcome/reason и время. Триггер запрещает `UPDATE` и `DELETE`.
Authorization codes, tokens, state, nonce, cookies, credentials и raw provider
responses не входят в аудит.

## Деградация и наблюдаемость

Provider adapter публикует низкокардинальные метрики:

- `identity_oidc_provider_requests_total`;
- `identity_oidc_provider_request_duration_seconds`;
- `identity_oidc_provider_last_success_unixtime`.

Метки ограничены `provider_id`, bounded операцией и безопасным классом
результата. Три и более transport/HTTP/unknown/deadline/validation failures за
10 минут, сохраняющиеся пять минут, поднимают `OidcProviderUnavailable`.
Связанная инструкция
`docs/runbooks/generated/ru/identity.oidc-provider-unavailable.md` построена из
`ops.roehub.io/v1`.

Отказ внешнего провайдера не меняет локальную конфигурацию, не отзывает
действующие Roehub sessions и не блокирует локальный вход. Если одновременно
недоступен локальный контур, это уже более широкий identity incident, а не
изолированная OIDC-деградация.

## Совместимость

| Поверхность | Классификация | Последствие |
|---|---|---|
| `AuthenticationProvider/v1` и OIDC API/DTO | `compatible-change` | Добавлена provider-neutral optional поверхность. |
| Identity semantics | `breaking-change` | External subject больше не создаёт пользователя без invitation/linking. |
| PostgreSQL persistence | `breaking-change` | Greenfield lifecycle получает обязательную phase `oidc-provider-0013`. |
| Runtime config | `breaking-change` | Старые provider-specific env keys заменены `client_secret_ref`, `OPENBAO_ADDR`, `ROEHUB_OPENBAO_ROOT` и mode-`0600` файлом identity credential; raw secret env больше не является контрактом OIDC. |
| Session/cookie semantics | `compatible-change` | Сохраняется opaque Roehub session; OIDC не передаёт browser token. |
| Organization/RBAC | `compatible-change` | Provisioning принимает только роли существующих invitations. |
| Request hash/cache/resource identity | `none` | Product resource identity не меняется. |
| Межсервисные вызовы | `compatible-change` | Появляется bounded исходящий HTTPS OIDC call только при включённом provider. |
| Внешние эффекты | `compatible-change` | Единственный внешний эффект — OIDC ceremony; current production не затрагивается. |
| Audit/metrics/runbook | `compatible-change` | Добавлены redacted события, метрики, alert и инструкция. |
| Browser defaults | `compatible-change` | Local passkey остаётся первым; OIDC показывается только при полной настройке. |

Основная классификация Stage `07` — `breaking-change` для старой
provider-specific identity/persistence/config модели. Для чистой установки это
запланированный новый контракт без legacy migration и dual read.

## Доказательства и передача

Stage `07` проверяет provider contract модульными fault injections, реальную
PostgreSQL migration/provisioning/linking границу и headless Chromium через
одноразовый сетевой OIDC provider. Browser evidence хранит только безопасные
статусы и изображения страницы до входа; trace, network dump и storage state не
создаются.

Stage `08` переносит credential references и восстановление секретов в OpenBao;
его runtime proof подтверждает ротацию client credential на одном живом
provider adapter без raw-env fallback.
Stage `19` добавляет административный browser UI управления invitations и
identity links. Stage `20` расширяет общий health/incident контур, сохраняя уже
принятый OIDC alert и `ops.roehub.io/v1` runbook.
