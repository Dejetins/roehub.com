# Identity Exchange Connections v1 — журнал итераций

Статус: активный рабочий журнал для staged rollout.

Документ является единым handoff-источником между stages архитектуры
`identity-exchange-connections-live-trading-v1`. Каждый executor обязан обновлять
его после валидации своего stage и до publish/deploy handoff через
`github:yeet`.

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage `00-07` обновляет этот документ до финального отчета. |
| Источник фактов | Записываются только проверенные факты из тестов, runtime calls, DB evidence, browser QA, Prometheus/Monit или явно помеченные blockers. |
| Секреты | Нельзя записывать API secrets, passphrase, ciphertext, HMAC, raw exchange error body, user tokens или session cookies. |
| Следующие stages | Каждый stage обязан заполнить секцию "Что обязательно знать дальше". |
| Publish/deploy | После успешной validation stage выполняет `github:yeet`; branch, commit, draft PR и deploy/runtime status записываются здесь. |
| Blocked state | Если stage не принят, следующий stage не стартует; blocker фиксируется в таблице и финальном отчете. |

## Stage Status

| Stage | Статус | Stage report | Ключевой результат | Blocker |
|---|---|---|---|---|
| 00 baseline | accepted | `docs/architecture/identity/exchange-connections-stage-reports/00-baseline-current-state.md` | Current `/api/exchange-keys`, `/settings`, `identity_exchange_keys`, secret-safe response shape, and `market_type=spot|futures` baseline frozen. | None |
| 01 security baseline | accepted; draft PR handoff complete | `docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md` | Legacy exchange-key mutations now fail closed on CSRF/same-origin, require recent Keycloak-backed Roehub session, write redacted exchange audit events, and extend audit event schema. | None |
| 02 exchange-control process | blocked on target-runtime supervision evidence | `docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md` | Local `exchange-control` runtime boundary, service identity, `/health/ready`, `/metrics`, Prometheus config, launchd config and Monit config are implemented. | Mac Studio Prometheus/Monit/controlled-restart evidence is not available in this local environment. |
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
| 02 | Target-runtime acceptance is blocked until Mac Studio Prometheus, Monit and controlled restart evidence are captured. | Stage 03 cannot claim a fully accepted predecessor until supervision evidence exists. | Stage 02 report Prometheus and Monit tables |
| 03 | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD |

## Контракты И Миграции

| Stage | API / DTO | Persistence | Config / env | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|
| 00 | `none`: no API/DTO behavior changed; baseline records existing `/api/exchange-keys`. | `none`: `identity_exchange_keys` remains current v1 table; no migrations added or modified. | `none`: no config/env changes. | Authenticated Keycloak smoke passed via public edge; no deploy/restart performed for docs-only Stage 00. | Legacy compatibility preserved; rollback is documentation-only because runtime behavior was not changed. |
| 01 | `compatible-change`: legacy `POST/DELETE /api/exchange-keys` still exist, but mutation preconditions now reject missing/cross-origin CSRF context and stale sessions with deterministic payloads. DTO fields unchanged. | `compatible-change`: `0007_identity_exchange_audit_events_v1.sql` additively extends `identity_audit_events_type_check`; bootstrap applies it after `0006`. | `compatible-change`: `ROEHUB_ENV=prod` now rejects the static dev-only `IDENTITY_EXCHANGE_KEYS_KEK_B64`; dev/test fallback remains allowed. | No production deploy in this stage; local route, migration, lint, type, docs and GitHub handoff gates are the acceptance surface. | Legacy `/api/exchange-keys` response shape and duplicate/delete contracts are preserved; rollback removes Stage 1 code plus 0007 only if no exchange audit rows have been emitted. |
| 02 | `none`: existing `apps/api` public routes and DTOs are unchanged; new internal operational HTTP endpoints are additive. | `none`: no database migration or table shape changed. | `compatible-change`: new operational env vars `ROEHUB_EXCHANGE_CONTROL_SERVICE_IDENTITY`, `ROEHUB_EXCHANGE_CONTROL_BIND_HOST`, `ROEHUB_EXCHANGE_CONTROL_METRICS_PORT`, `ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED`; prod requires `127.0.0.1:9205` and disabled validation. | `compatible-change`: new `exchange-control` process, Prometheus job, launchd plist, Monit config and monitoring runbook checks. | Rollback removes the new process/config/docs/tests; no data rollback is required because no persistence changed. |
| 03 | TBD | TBD | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD | TBD | TBD |

## Publish / Deploy Handoff

`github:yeet` в этом плане означает обязательный GitHub publish handoff после
успешной validation: scope review, targeted staging, commit, push и draft PR.
Если конкретный stage также выполняет runtime deploy/restart/smoke на target
environment, это фиксируется в колонке `Deploy/runtime status`. Если stage не
выполняет production deploy, это явно записывается как `not applicable`.

| Stage | Branch | Commit | Draft PR | Checks before push | Deploy/runtime status | Notes |
|---|---|---|---|---|---|---|
| 00 | `codex/identity-exchange-stage0-baseline` | `dda694b7` | [Draft PR #22](https://github.com/Dejetins/roehub.com/pull/22) | `gh --version && gh auth status`; focused pytest; docs index check; market-type grep; secret grep; authenticated runtime smoke | No production deploy; runtime smoke used existing public edge and logged out afterward. | Targeted staging included only Stage 00 docs/index files; unrelated `.codex/*` and architecture prompt-manager changes were left unstaged. |
| 01 | `codex/identity-exchange-stage1-security-baseline` | `9d405202` | [Draft PR #23](https://github.com/Dejetins/roehub.com/pull/23) | `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py`; `uv run pytest -q tests/unit/apps/migrations`; `uv run ruff check apps/api src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/migrations`; `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api`; `python -m tools.docs.generate_docs_index --check`; `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE" logs output .playwright-cli \|\| true`; `gh --version && gh auth status`. | No production deploy; local-dev stage only. | Stacked on Stage 0 draft PR #22 because Stage 0 docs are not merged to `main` yet; targeted staging included only Stage 1 code, tests, migration, and docs/index files. |
| 02 | `codex/identity-exchange-stage2-exchange-control` | `02a85e61` | [Draft PR #24](https://github.com/Dejetins/roehub.com/pull/24) | `gh --version && gh auth status`; `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api`; `uv run ruff check apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`; `uv run pyright apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api`; local `curl` health/metrics; `plutil -lint`; no-external-call grep; `python -m tools.docs.generate_docs_index --check`. | Local-dev runtime passed; Mac Studio Prometheus/Monit/controlled restart blocked because Prometheus `127.0.0.1:9090` is not running here and `/opt/homebrew/opt/monit/bin/monit` is absent. | Stacked on Stage 1 branch because Stage 0/1 draft PRs are not merged to `main` yet; targeted staging included only Stage 2 code, ops configs, docs and tests. |
| 03 | TBD | TBD | TBD | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD | TBD | TBD | TBD |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence |
|---|---|---|---|---|
| 00 | None | N/A | Stage 1 can start from frozen baseline after PR handoff. | Stage report plus passing gates. |
| 01 | None | N/A | Stage 2 can start after GitHub draft PR handoff completes. | Stage 1 report plus focused route/migration/static gates. |
| 02 | Mac Studio supervision evidence unavailable in local checkout: Prometheus is not listening on `127.0.0.1:9090`, Monit binary `/opt/homebrew/opt/monit/bin/monit` is absent, and controlled restart was not executed to avoid restarting unrelated local services. | Stage-blocking for Stage 3 acceptance | Deploy/install Stage 2 branch on Mac Studio, reload `com.roehub.exchange-control`, verify `up{job="exchange-control"}`, verify `roehub_exchange_control`, run controlled restart, then update this row/report. | Local health/metrics and static ops config checks passed in Stage 02 report. |
