# Roehub — активация рыночных данных и выбор инструментов v1: журнал этапов

- `plan_doc`: `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/`
- `stage_ledger`: `docs/architecture/platform/roehub-market-data-activation-and-instrument-selection-v1-stage-reports/roehub-market-data-activation-and-instrument-selection-v1-stage-ledger.md`
- `execution_mode`: `goal_driven`
- `ledger_status`: `completed`
- `current_stage`: `05`
- `branch`: `codex/market-data-egress-instrument-onboarding`
- `updated_at`: `2026-07-15`

## Правила

Следующий этап выполняется только после `accepted` predecessor. Обновлять журнал после validation и до stage report. `blocked` не пропускается. Не создавать дополнительные ветки/worktree/stash, не включать чужие изменения, не выполнять commit/push/deploy без отдельной пользовательской authority. Секреты и чувствительные provider-данные не сохранять.

| Этап | Статус | Prompt | Предшественник | Доказательство | Blocker | Следующий разрешённый |
|---|---|---|---|---|---|---|
| `00` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/00-contract-baseline-and-repair-route.md` | `N/A` | architecture/contract baseline and source/config audit | none | `01` |
| `01` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/01-controlled-egress-and-market-readiness.md` | `00` | Docker + exchange `BTCUSDT` runtime | none | `02` |
| `02` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/02-user-instrument-selection-and-onboarding.md` | `01` | PostgreSQL/API/browser | none | `03` |
| `03` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/03-artifact-capacity-and-first-publish.md` | `02` | bounded artifact runtime | none | `04` |
| `04` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/04-openbao-owner-bootstrap-and-credential-isolation.md` | `03` | disposable three-recipient `2-of-3` runtime + owner custody handoff | durable PGP custody is explicit owner action after stage acceptance | `05` |
| `05` | `accepted` | `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/05-release-rebuild-local-lifecycle-and-handoff.md` | `04` | Docker Desktop `linux/arm64`, normal ingress, real browser onboarding, live `BTCUSDT` readiness, bounded artifact publish and cgroup memory | original self-hosted Stage `24` remains externally blocked on native `linux/amd64`; current code additionally requires signed Stage `22`/`23` recertification | none |

## Изменения журнала

| Дата | Этап | Изменение | Evidence |
|---|---|---|---|
| 2026-07-15 | `00` | Создан отдельный repair plan: исходный Stage `24` остаётся исторически `blocked`, новый маршрут не ослабляет его AMD64/ingress требования. | User-authorized branch; current compose/log/metrics audit; original Stage `24` report |
| 2026-07-15 | `00` | Зафиксированы network, startup-race, whitelist, selection, coverage, artifact-memory и OpenBao custody контракты; этап принят. | `00-contract-baseline-and-repair-route.md`; schema/source audit; contract matrix |
| 2026-07-15 | `00` | Независимая cold-head проверка нашла docs-index, tenant-isolation, catalog-refresh, publisher-scope и OpenBao-threshold defects; все исправлены до продолжения Stage `01`. | independent subagent verdict `Block`; local follow-up: `generate_docs_index --check`, `git diff --check` passed |
| 2026-07-15 | `01` | Добавлены schema-validated egress network только для scheduler/WS, refreshable WS subscriptions и runtime readiness verifier. | `01-controlled-egress-and-market-readiness.md`; `evidence/01-market-readiness-proof.json`; focused tests and Docker proof |
| 2026-07-15 | `02` | Добавлены organization selections/audit/history bounds/catalog state, global effective collector set, coverage/artifact inventory, API и browser chooser; CSV исключён из generated runtime policy. | `02-user-instrument-selection-and-onboarding.md`; `evidence/02-market-data-selection-runtime-proof.json`; PostgreSQL marker, API/browser/mobile proof, `118 passed` |
| 2026-07-15 | `03` | Publisher ограничен одним worker и бюджетом `768 MiB` при контейнерном лимите `1 GiB`; единственный ручной publish `BTCUSDT` из `10080` свечей успешно переключил pointer без OOM. Первичный scheduler bootstrap ограничен 7 днями, периодическая историческая достройка отключена до отдельного решения о расширении. | `03-artifact-capacity-and-first-publish.md`; `evidence/03-artifact-capacity-proof.json`; cgroup peak `715964416` bytes, pointer manifest `933973047920870ef45c958c658fa1d4e8b106b22cc68def0e36846e7b1a0154` |
| 2026-07-15 | `04` | Добавлена отдельная идемпотентная owner-команда OpenBao: ровно три public PGP recipient, encrypted custody delivery, затем семь статических least-privilege AppRole с response-wrapped one-time SecretID; initial admin отзывается после provisioning. | `04-openbao-owner-bootstrap-and-credential-isolation.md`; `evidence/04-openbao-owner-bootstrap-proof.json`; isolated Docker Desktop runtime proof passed, cleanup passed; durable PGP custody остаётся owner action |
| 2026-07-15 | `05` | Пересобран локальный `linux/arm64` runtime и последовательно пересоздан профиль `trading` без удаления volume. Устранён file-backed ClickHouse credential gap в CLI publisher; bounded `BTCUSDT` publish, normal ingress, browser onboarding и readiness проверены на реальной границе. | `05-release-rebuild-local-lifecycle-and-handoff.md`; `evidence/05-local-runtime-proof.json`; `evidence/05-local-runtime-readiness-proof.json`; `124 passed`, Ruff, Pyright, Node checks |
