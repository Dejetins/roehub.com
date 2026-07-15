# Этап 07 — универсальная интеграция OIDC-провайдера

## Статус

- Этап: `07`.
- Статус: `accepted`; все функциональные, runtime, browser и documentation
  gates пройдены.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Proof boundary: `N/A`; disposable OIDC/API/PostgreSQL/browser без production
  identity store, credentials и пользователей.
- Blocker: отсутствует.
- Следующий разрешённый этап: `08`.

## Результат

Реализован provider-neutral OIDC Authorization Code Flow через
`AuthenticationProvider/v1`: verified discovery/issuer/JWKS, state, nonce, PKCE
`S256`, строгие redirect/subject bindings, ограниченные timeout/retry/cache и
единственный token POST. Keycloak и Pocket ID являются обычными совместимыми
провайдерами новой установки, а не обязательными контейнерами или migration
sources.

Внешняя тройка `(provider_id, issuer, subject)` отображается на стабильный
внутренний `user_id`. Неизвестная identity получает пользователя и membership
только атомарно с действующим invitation; существующий пользователь связывает
identity только из свежей аутентифицированной сессии. Duplicate, issuer/subject
conflict и takeover отклоняются.

Локальный passkey/recovery контур Stage `06` остаётся обязательной независимой
основой. Чистая установка имеет `oidc.enabled=false`; внешний provider
появляется только после полного явного набора настроек и не отключает local
auth.

Канонический контракт:
`docs/architecture/identity/oidc-authentication-provider-v1.md`.

## Реализованные границы

- Добавлены `AuthenticationProvider/v1`, `OidcIdentityRepository` и
  `OidcAuthenticationService`.
- HTTP adapter принимает только verified `RS256`, exact issuer/audience/azp,
  time claims, nonce и bounded subject.
- Максимальные budgets — connect `3`, response `10`, overall `15` секунд;
  конфигурация может только уменьшить их; slow-stream wall-clock tests доказали
  жёсткую границу для discovery, JWKS и token POST.
- Discovery/JWKS GET делает максимум две повторные попытки с bounded jitter;
  cache учитывает `no-store`/`max-age` и ограничен одним часом.
- Token POST после неизвестного результата не повторяется, попытка завершается
  без Roehub session.
- Durable `exchange_started_at` атомарно допускает к token POST только один из
  одновременных callback; завершение использует новое текущее время после сети.
- Routes `/auth/oidc/status|login|link|callback` не зависят от
  provider-specific URL или DTO.
- Browser cookie OIDC-попытки содержит только opaque attempt UUID; state,
  nonce, code и provider payload не сериализуются.
- Web и API production entrypoints маскируют весь callback query как
  `?redacted` до форматирования Uvicorn access log; proxy transport error не
  отражает сырой URL.
- PostgreSQL adapter выполняет provisioning, invitation acceptance, linking и
  аудит в одной транзакции.
- `/login` показывает внешний provider как вторичный способ после local
  passkey и только при успешном `/auth/oidc/status`.
- Добавлены redacted metrics, включая deadline/semantic validation failures,
  alert `OidcProviderUnavailable` и связанная инструкция `ops.roehub.io/v1`.

## Persistence evidence

Phase manifest:

- `oidc-provider-0013` SHA-256:
  `65a95708d424346a486add22367fc7dd113ee499e9faa5e03f533cd2b1e05d4f`;
- `0013_identity_oidc_provider_v1.sql` SHA-256:
  `3a1c1d713fdf920eebd420ef66294635eb47cec0892103c2277dc18fc4892628`.

Финальный `uv run python -m apps.migrations.verify_storage_runtime` на реальных
PostgreSQL `16.14`, ClickHouse `24.8.14.39` и Redis `7.2.14` вернул:

- `fresh_bootstrap=passed`;
- `idempotent_rerun=passed`;
- `interrupted_recovery=passed`;
- `persistent_volume_restart=passed`;
- `external_readiness=passed`;
- `oidc.invitation_provisioning=passed`;
- `oidc.stable_subject_mapping=passed`;
- `oidc.uninvited_provisioning=rejected` без orphan user;
- `oidc.authenticated_linking=passed`;
- `oidc.subject_takeover=rejected`;
- `oidc.hash_only_identity=passed`;
- `oidc.audit_immutable=passed`;
- `oidc.single_exchange_claim=passed` на двух одновременных PostgreSQL claims;
- Stage `05` organization constraints и Stage `06` local-auth proof остались
  зелёными;
- `cleanup=passed`.

Raw subject, email, UUID, DSN и credentials из proof не извлекались.

## Реальная браузерная проверка

Режим: реальный headless Chromium через pinned Playwright CLI. Три процесса
подняты только на loopback: настоящий Web SSR `:8000`, identity API `:8010` и
одноразовый сетевой OIDC provider `:9010` с discovery, JWKS, PKCE и настоящей
RSA-подписью. Client и local-login credentials генерировались только в памяти
родительского shell. Provider access log был отключён, а Web/API использовали
production redaction config. Production authentication не затрагивалась.

Проверено на финальной версии fixture:

1. `/login` одновременно показывает local passkey и вторичный
   `Disposable OIDC`; provider не заменяет local UI.
2. Настоящий local password flow через Web форму и `/auth/local/password`
   создал сессию при режиме provider `outage`; `/auth/current-user=200` и
   `/auth/local/status=200`.
3. Новый OIDC login при том же outage вернул `503`, а local session после него
   осталась действующей (`/auth/current-user=200`).
4. OIDC login по действующему invitation создал внутреннего пользователя,
   membership и действующую opaque session; безопасные результаты —
   `current_user_status=200`, `invitation_provisioned=true`.
5. Authenticated linking сохранил local owner session; последующий OIDC login
   по тому же subject вернул тот же internal user.
6. Привязка уже занятого subject из другой session вернула `409`, а исходная
   secondary session осталась действующей.
7. Logout через production `/auth/local/logout` отозвал server-side session;
   последующий `/auth/current-user` вернул `401`.
8. Повтор callback из browser history вернул `400` и не создал session.
9. Fault `token_unknown` вернул `503`, оставил `/auth/current-user=401` и
   provider counter подтвердил ровно `exchange_calls=1`.
10. Реальный Uvicorn startup с access log записал callback только как
    `/auth/oidc/callback?redacted`; контрольный query marker отсутствовал.
11. Desktop `1440x900` и полный mobile `390x844` layout визуально проверены:
   обе auth-границы различимы, overlap/cutoff нет.

Sanitized visual artifacts находятся в ignored output:

- `output/playwright/stage07-oidc/login-desktop.png`;
- `output/playwright/stage07-oidc/login-mobile-full.png`.

Trace, video, HAR, network dump и storage state не создавались. После закрытия
browser/server sessions автоматический поиск подтвердил отсутствие raw
code/state/nonce/session/token и disposable credential values в
`.playwright-cli` и временных server logs. Предварительный прогон с несовпавшим
loopback hostname не принимался; его provider process и все grants были
уничтожены до финального прогона. Console содержит ожидаемые `401` после logout
и `404/502` product-dashboard endpoints, отсутствующие в identity-only fixture;
они не использовались как доказательство product dashboard.

## Fault injection и эксплуатация

Модульные provider tests дополнительно доказали:

- discovery transport timeout и initial GET + не более двух retry;
- JWKS timeout с тремя GET attempts и без повторного token POST;
- malformed signing key закрыто отклоняется без refresh loop;
- stale/unknown signing key вызывает ровно одно JWKS refresh;
- nonce mismatch и неверные claims закрыто отклоняются;
- token timeout имеет `token_result_unknown` и один POST;
- concurrent callback допускают ровно один provider exchange;
- попытка, истёкшая во время сети, не создаёт session;
- slow discovery/JWKS/token stream отклоняется в пределах wall-clock budget;
- malformed/semantic/deadline failures получают bounded metrics outcomes.

Добавлен `OidcProviderUnavailable`: минимум три
transport/HTTP/unknown/deadline/validation failures за 10 минут, сохраняющиеся
пять минут. `promtool 3.13.1` проверил весь файл: `SUCCESS: 24 rules found`.
Docker Hub дважды не отдал образ (`EOF`, затем `TLS handshake timeout`), поэтому
по пользовательскому разрешению установлен Homebrew `prometheus 3.13.1`;
service не запускался.

`tools.docs.generate_runbooks` сгенерировал шесть русских инструкций и index;
новая `identity.oidc-provider-unavailable` связана с alert. В доказательствах
запрещены authorization, code, cookie, credential, nonce, session, state и
token fields.

## Проверки

- focused reviewer-fix suite — `49 passed`;
- расширенный identity/API/Web/migration/config regression — `424 passed`,
  четыре те же warnings;
- scoped `ruff` — `passed`;
- scoped `pyright` — `0 errors, 0 warnings`;
- runtime input inventory generation/check — `passed`, `135` имён без значений;
- disposable Docker storage/OIDC runtime proof — `passed`;
- `promtool check rules` — `SUCCESS: 24 rules found`;
- runbook generation/check, monitoring assets и `16 passed` — `passed`;
- browser local-login/OIDC/link/conflict/replay/unknown/outage/access-log/
  mobile-desktop proof — `passed`;
- docs index, project map и `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| OIDC API/DTO | `compatible-change` | Добавлена optional provider-neutral поверхность. |
| Identity semantics | `breaking-change` | External identity требует invitation или authenticated linking. |
| Application ports | `compatible-change` | Добавлены `AuthenticationProvider/v1` и OIDC repository без provider-specific domain. |
| PostgreSQL persistence | `breaking-change` | Greenfield lifecycle получает `oidc-provider-0013` и новые unique bindings. |
| Runtime config | `breaking-change` | Generic provider settings заменяют старые Keycloak-specific inputs; Stage `07` использует injected env credential, а разрешение secret reference относится к Stage `08`. |
| Session/cookie/CSRF | `compatible-change` | Сохраняется opaque local session; OIDC token не попадает в browser. |
| Organization/RBAC | `compatible-change` | Provisioning принимает только роли существующего invitation. |
| Request hash/cache/resource identity | `none` | Product resource namespace не менялся. |
| Межсервисные вызовы | `compatible-change` | Появляется bounded outbound OIDC call только при enabled provider. |
| Внешние эффекты | `compatible-change` | Только disposable/provider ceremony; production mutation отсутствует. |
| Audit/metrics/runbook | `compatible-change` | Добавлены hash-only audit, redacted metrics, alert и инструкция. |
| Browser defaults | `compatible-change` | Local passkey остаётся первым, OIDC вторичен и optional. |

Основная классификация Stage `07` — `breaking-change` относительно прежней
Keycloak-specific persistence/config модели, но ожидаемая для greenfield v1.
Legacy import, dual-read и current identity migration отсутствуют.

## Холодная проверка

- Режим: единственная `independent subagent` проверка, затем `local follow-up
  after independent review` согласно repository artifact gate.
- Первоначальный вердикт: `Block`.
- Исправлено:
  1. token POST защищён durable atomic claim; service/DB concurrency proof
     подтверждает один exchange;
  2. overall budget стал жёсткой wall-clock границей, покрытой slow-stream
     discovery/JWKS/token tests;
  3. Uvicorn access logs Web/API маскируют callback query, а proxy error не
     отражает transport URL;
  4. deadline и semantic validation failures включены в metrics/alert;
  5. настоящий local-login browser flow доказан во время OIDC outage;
  6. migration file/phase hashes обновлены и весь Docker proof выполнен заново;
  7. `completed_at` перечитывается после сети, а документация разделяет
     injected env Stage `07` и OpenBao reference Stage `08`.
- Локальная повторная проверка: `49 passed`, `424 passed`, ruff/pyright,
  real Docker/PostgreSQL, promtool/runbook, Chromium и access-log scan прошли.
- Итоговый вердикт: `Release after fixes`.
- Остаточные риски: OpenBao secret reference lifecycle относится к Stage `08`;
  admin UI invitations/identity links — к Stage `19`; общий identity incident
  dashboard и product dashboard browser APIs — к Stage `20` и последующим UI
  этапам.

## Файлы этапа

Созданы:

- provider/OIDC application ports, service и HTTP adapter;
- in-memory/PostgreSQL OIDC repositories;
- migration `0013`, runtime probe и Stage `07` tests;
- shared Uvicorn access-log redaction для production entrypoints;
- `tools/qa/oidc_browser_app.py`;
- OIDC architecture doc, canonical/locale/generated runbook и этот report.

Изменены:

- identity exports/API routes/wiring и monitoring;
- API/Web process entrypoints и безопасная Web proxy error boundary;
- migration manifest/lifecycle/bootstrap/verifier;
- Web login template/application/locales;
- installation schema и runtime input inventory;
- Prometheus rules, runbook index/generator test;
- superseded Keycloak status, main plan и generated docs/project-map outputs.

Удалённых файлов нет. `.codex/PLANS.md`, unrelated `local_artifacts`, license,
governance и другие foreign changes сохранены. Commit, push, deploy и
production mutation не выполнялись.

## Передача Stage 08

После принятия Stage `07` этап `08` должен реализовать OpenBao secret reference
lifecycle, policy, recovery и реальные failure drills. OIDC client credential
нельзя переносить в `roehub.yaml`, generated config, env inventory values,
доказательства или чат; только ссылка и approved secret boundary.
