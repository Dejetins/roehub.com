# Identity Exchange Connections v1 — журнал итераций

Статус: активный рабочий журнал для staged rollout.

Документ является единым handoff-источником между stages архитектуры
`identity-exchange-connections-live-trading-v1`. Каждый executor обязан обновлять
его после валидации своего stage и до direct-main delivery в `main`.

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage `00-07` обновляет этот документ до финального отчета. |
| Источник фактов | Записываются только проверенные факты из тестов, runtime calls, DB evidence, browser QA, Prometheus/Monit или явно помеченные blockers. |
| Секреты | Нельзя записывать API secrets, passphrase, ciphertext, HMAC, raw exchange error body, user tokens или session cookies. |
| Следующие stages | Каждый stage обязан заполнить секцию "Что обязательно знать дальше". |
| Direct-main delivery | После успешной validation stage доставляется напрямую в `main`: `git pull --ff-only origin main`, scoped staging, commit на `main`, `git push origin main`, контроль CI/deploy. |
| Запрет stage-веток | Отдельная branch или draft PR на stage не создаются; если direct-main delivery невозможен, stage помечается `blocked`. |
| Blocked state | Если stage не принят, следующий stage не стартует; blocker фиксируется в таблице и финальном отчете. |

## Stage Status

| Stage | Статус | Stage report | Ключевой результат | Blocker |
|---|---|---|---|---|
| 00 baseline | accepted | `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md` | Current `/api/exchange-keys`, `/settings`, `identity_exchange_keys`, secret-safe response shape, and `market_type=spot|futures` baseline frozen. | None |
| 01 security baseline | accepted; direct-main chain delivered | `docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md` | Legacy exchange-key mutations now fail closed on CSRF/same-origin, require recent Keycloak-backed Roehub session, write redacted exchange audit events, and extend audit event schema. | None |
| 02 exchange-control process | accepted; direct-main deploy and Mac Studio supervision evidence complete | `docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md` | `exchange-control` runtime boundary, service identity, `/health/ready`, `/metrics`, Prometheus target, launchd service, Monit service and controlled restart are implemented and verified on Mac Studio. | None |
| 03 secret engine transit | pending | `docs/architecture/identity/exchange-connections-stage-reports/03-secret-engine-transit.md` | TBD | TBD |
| 04 connections/backfill | pending | `docs/architecture/identity/exchange-connections-stage-reports/04-connections-credential-versions-backfill.md` | TBD | TBD |
| 05 Binance/Bybit validation | pending | `docs/architecture/identity/exchange-connections-stage-reports/05-binance-bybit-validation.md` | TBD | TBD |
| 06 settings UI | pending | `docs/architecture/identity/exchange-connections-stage-reports/06-settings-ui.md` | TBD | TBD |
| 07 production readiness | pending | `docs/architecture/identity/exchange-connections-stage-reports/07-production-readiness.md` | TBD | TBD |

## Что Обязательно Знать Дальше

| Stage | Факт / решение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| 00 | Current compatibility surface is `POST/GET/DELETE /api/exchange-keys`; public API response fields are only `key_id`, `exchange_name`, `market_type`, `label`, `permissions`, masked `api_key`, `created_at`, `updated_at`. | Stage 01 must preserve this legacy surface while adding security gates; Stage 04 must keep rollback/projection compatibility when adding `exchange_connections`. | `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md` |
| 00 | Current `market_type` v1 is `spot|futures` in SQL, API DTOs, application/domain validation, and `/settings` UI. | Stage 04/05 must not introduce `linear|inverse` as persisted/API enum values without an explicit migration and compatibility decision. | `migrations/postgres/0003_identity_exchange_keys_v1.sql`; `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`; `apps/web/templates/fragments/account/exchange_keys.html` |
| 00 | Current duplicate/delete errors are deterministic: duplicate active key returns `409 exchange_key_already_exists`; missing/foreign/already-deleted delete returns `404 exchange_key_not_found`. | Stage 01+ must not silently change existing client-visible error contracts. | `tests/unit/apps/api/test_identity_exchange_keys_routes.py`; `src/trading/contexts/identity/application/use_cases/exchange_keys_errors.py` |
| 00 | Authenticated runtime smoke through Keycloak public edge returned `/settings=200`, `/api/exchange-keys=200`, empty exchange-key list, and no forbidden response fields. | Stage 01 has a concrete authenticated baseline to compare security-gate behavior against. | Stage 00 report runtime evidence table |
| 01 | Deterministic security errors are now `csrf_required` for missing/cross-origin mutation context and `recent_auth_required` for stale/missing recent auth. | Stage 02+ credential add/rotate/delete/disable hooks must reuse these literals and preserve ordering: CSRF/same-origin gate before recent-auth before mutation/audit. | `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`; `tests/unit/apps/api/test_identity_exchange_keys_routes.py` |
| 01 | Recent-auth source is local Roehub session `created_at`, created by the Keycloak-backed login callback; accepted window is 10 minutes. | Stage 02 can harden or replace the policy with Keycloak step-up claims later, but must not weaken fail-closed behavior for credential mutations. | `src/trading/contexts/identity/application/ports/current_user.py`; `src/trading/contexts/identity/adapters/outbound/security/current_user/roehub_session_current_user.py` |
| 01 | Audit schema accepts `exchange_key_created`, `exchange_key_deleted`, `exchange_connection_created`, `exchange_connection_validated`, `exchange_connection_validation_failed`, `exchange_credential_rotated`, `exchange_connection_disabled`, `exchange_connection_deleted`. | Stage 02+ can emit future exchange events without another enum migration for these names. | `migrations/postgres/0007_identity_exchange_audit_events_v1.sql`; `tests/unit/apps/migrations/test_identity_exchange_audit_events_sql.py` |
| 01 | Current create/delete audit metadata intentionally excludes API secrets, passphrases, ciphertext, HMAC, fingerprints and raw exchange errors. | Future stages must keep audit metadata redacted and use stable IDs/state/reason codes instead of raw secret material. | `tests/unit/apps/api/test_identity_exchange_keys_routes.py` |
| 01 | `ROEHUB_ENV=prod` rejects the documented static dev-only exchange-key KEK. | Stage 02/03 product-ready secret work cannot rely on the dev fallback and must keep production fail-closed behavior. | `apps/api/wiring/modules/identity.py`; `tests/unit/apps/api/test_identity_wiring_module.py` |
| 02 | Service identity is exactly `exchange-control`; invalid identities fail config validation. | Stage 03 Transit ACL work can bind policy to a concrete runtime principal instead of the API process. | `src/trading/contexts/exchange_control/application/service_identity.py`; `tests/unit/contexts/exchange_control/test_exchange_control_runtime.py`; Stage 02 report service identity table |
| 02 | Runtime contract is local-only `127.0.0.1:9205`, `GET /health/ready`, and `/metrics`; prod config rejects non-9205 ports. | Stage 03/05 must preserve this boundary and should not route exchange validation back through opaque `apps/api` code paths. | `apps/exchange_control/main/main.py`; `src/trading/contexts/exchange_control/adapters/inbound/http/app.py`; Stage 02 report health and metrics tables |
| 02 | `exchange_control_active`, `exchange_connection_validation_total`, and `exchange_connection_status` are exported with bounded labels and no user, connection, credential, API key, or raw error labels. | Stage 05 can extend metrics with real validation outcomes without introducing secret-bearing labels. | `src/trading/contexts/exchange_control/adapters/inbound/http/app.py`; local `/metrics` smoke in Stage 02 report |
| 02 | Real exchange validation remains disabled and fails config validation if enabled before Stage 5. | Stage 03/04 must not introduce Binance/Bybit calls or decrypt paths while preparing Transit and schema work. | `ExchangeControlRuntimeConfig`; `rg` no-external-call evidence in Stage 02 report |
| 02 | Mac Studio target-runtime evidence is accepted: `up{job="exchange-control"} = 1`, `roehub_exchange_control OK`, and controlled Monit restart changed PID `22544 -> 23125` while `/health/ready` and `/metrics` stayed healthy. | Stage 03 can start Transit ACL work against a live supervised service identity. | Stage 02 report Prometheus and Monit tables |
| 03 | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD |

## Контракты И Миграции

| Stage | API / DTO | Persistence | Config / env | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|
| 00 | `none`: no API/DTO behavior changed; baseline records existing `/api/exchange-keys`. | `none`: `identity_exchange_keys` remains current v1 table; no migrations added or modified. | `none`: no config/env changes. | Authenticated Keycloak smoke passed via public edge; no deploy/restart performed for docs-only Stage 00. | Legacy compatibility preserved; rollback is documentation-only because runtime behavior was not changed. |
| 01 | `compatible-change`: legacy `POST/DELETE /api/exchange-keys` still exist, but mutation preconditions now reject missing/cross-origin CSRF context and stale sessions with deterministic payloads. DTO fields unchanged. | `compatible-change`: `0007_identity_exchange_audit_events_v1.sql` additively extends `identity_audit_events_type_check`; bootstrap applies it after `0006`. | `compatible-change`: `ROEHUB_ENV=prod` now rejects the static dev-only `IDENTITY_EXCHANGE_KEYS_KEK_B64`; dev/test fallback remains allowed. | No production deploy in this stage; local route, migration, lint, type, docs and direct-main delivery gates are the acceptance surface. | Legacy `/api/exchange-keys` response shape and duplicate/delete contracts are preserved; rollback removes Stage 1 code plus 0007 only if no exchange audit rows have been emitted. |
| 02 | `none`: existing `apps/api` public routes and DTOs are unchanged; new internal operational HTTP endpoints are additive. | `none`: no database migration or table shape changed. | `compatible-change`: new operational env vars `ROEHUB_EXCHANGE_CONTROL_SERVICE_IDENTITY`, `ROEHUB_EXCHANGE_CONTROL_BIND_HOST`, `ROEHUB_EXCHANGE_CONTROL_METRICS_PORT`, `ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED`; prod requires `127.0.0.1:9205` and disabled validation. | `compatible-change`: new `exchange-control` process, Prometheus job, launchd plist, Monit config and monitoring runbook checks. | Rollback removes the new process/config/docs/tests; no data rollback is required because no persistence changed. |
| 03 | TBD | TBD | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD | TBD | TBD |

## Direct-Main Delivery

После успешной validation stage не создает отдельную ветку и не открывает
draft PR. Доставка идет только в `main`. Если stage выполняет runtime
deploy/restart/smoke на target environment, это фиксируется в колонке
`Deploy/runtime status`. Если runtime deploy не относится к stage, пишется
`not applicable`.

| Stage | Branch | Commit | Push status | CI/deploy status | Deploy/runtime status | Notes |
|---|---|---|---|---|---|---|
| 00 | `main` | `dda694b7`; present in `origin/main` at `0a5386fa` | pushed to `origin/main`; old stage branch deleted | covered by direct-main chain through `0a5386fa` | not applicable; docs/runtime-smoke baseline only | Previous draft PR handoff is superseded by direct-main history. |
| 01 | `main` | `9d405202`; present in `origin/main` at `0a5386fa` | pushed to `origin/main`; old stage branch deleted | covered by direct-main chain through `0a5386fa` | not applicable; local-dev/security stage only | Previous draft PR handoff is superseded by direct-main history. |
| 02 | `main` | `0a5386fa` | pushed to `origin/main`; old stage branch deleted | GitHub workflows watched in Stage 02 report; current local `main` is synchronized to `origin/main` | Deployed on Mac Studio: `exchange-control` launchd/Monit/Prometheus evidence accepted. | Direct `main` delivery includes Stage 0, Stage 1 and Stage 2 chain. |
| 03 | `main` | TBD | TBD | TBD | TBD | TBD |
| 04 | `main` | TBD | TBD | TBD | TBD | TBD |
| 05 | `main` | TBD | TBD | TBD | TBD | TBD |
| 06 | `main` | TBD | TBD | TBD | TBD | TBD |
| 07 | `main` | TBD | TBD | TBD | TBD | TBD |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence |
|---|---|---|---|---|
| 00 | None | N/A | Stage 1 can start from frozen baseline after direct-main ledger handoff. | Stage report plus passing gates. |
| 01 | None | N/A | Stage 2 can start after direct-main delivery evidence is recorded. | Stage 1 report plus focused route/migration/static gates. |
| 02 | None | N/A | Stage 3 can start Transit ACL work from accepted `exchange-control` supervision boundary. | Stage 02 report plus Mac Studio Prometheus, Monit, restart and smoke evidence. |
