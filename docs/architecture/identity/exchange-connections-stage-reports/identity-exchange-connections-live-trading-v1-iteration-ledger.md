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
| 01 security baseline | pending | `docs/architecture/identity/exchange-connections-stage-reports/01-security-baseline.md` | TBD | TBD |
| 02 exchange-control process | pending | `docs/architecture/identity/exchange-connections-stage-reports/02-exchange-control-process.md` | TBD | TBD |
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
| 01 | TBD | TBD | TBD |
| 02 | TBD | TBD | TBD |
| 03 | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD |

## Контракты И Миграции

| Stage | API / DTO | Persistence | Config / env | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|
| 00 | `none`: no API/DTO behavior changed; baseline records existing `/api/exchange-keys`. | `none`: `identity_exchange_keys` remains current v1 table; no migrations added or modified. | `none`: no config/env changes. | Authenticated Keycloak smoke passed via public edge; no deploy/restart performed for docs-only Stage 00. | Legacy compatibility preserved; rollback is documentation-only because runtime behavior was not changed. |
| 01 | TBD | TBD | TBD | TBD | TBD |
| 02 | TBD | TBD | TBD | TBD | TBD |
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
| 00 | `pending github:yeet` | `pending github:yeet` | `pending github:yeet` | `gh --version && gh auth status`; focused pytest; docs index check; market-type grep; secret grep; authenticated runtime smoke | No production deploy; runtime smoke used existing public edge and logged out afterward. | To be finalized after targeted staging, push, and draft PR creation. |
| 01 | TBD | TBD | TBD | TBD | TBD | TBD |
| 02 | TBD | TBD | TBD | TBD | TBD | TBD |
| 03 | TBD | TBD | TBD | TBD | TBD | TBD |
| 04 | TBD | TBD | TBD | TBD | TBD | TBD |
| 05 | TBD | TBD | TBD | TBD | TBD | TBD |
| 06 | TBD | TBD | TBD | TBD | TBD | TBD |
| 07 | TBD | TBD | TBD | TBD | TBD | TBD |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence |
|---|---|---|---|---|
| 00 | None | N/A | Stage 1 can start from frozen baseline after PR handoff. | Stage report plus passing gates. |
