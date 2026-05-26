# <Название плана> — журнал выполнения stages

Единый handoff-документ для выполнения плана через prompt pack или staged agent workflow.

## Статус Документа

- plan_doc: `<docs/architecture/... или другой source-of-truth plan>`
- prompt_pack: `.codex/agents/generated/<pack_folder>/`
- ledger_status: `active | completed | blocked | superseded | archived`
- current_stage: `<stage_id>`
- updated_at: `YYYY-MM-DD`
- owner: `<agent/person/team>`

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот документ после validation и до финального отчета. |
| Источник фактов | Пишите только проверенные факты: tests, runtime calls, DB evidence, browser QA, CI, benchmark, deploy/smoke или явно помеченные blockers. |
| Статусы | Используйте только `pending`, `in_progress`, `accepted`, `blocked`, `skipped`, `superseded`. |
| Следующий stage | Заполняйте "Что обязательно знать дальше" так, чтобы следующий executor не перечитывал весь чат. |
| Blocked state | Если stage не принят, следующий зависимый stage не стартует, кроме repair/unblock/supersede prompt. |
| Секреты | Не записывайте secrets, tokens, cookies, passphrases, ciphertext, raw provider errors, HMAC, API keys или credentials. |
| Publish/deploy | Если stage публикуется, деплоится или проверяется на runtime, фиксируйте branch/commit/PR/checks/deploy/smoke. |
| Validation depth | Для нетривиального stage тесты являются локальным gate, но не acceptance. Фиксируйте real-boundary или end-to-end evidence. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|
| 00 <name> | pending | `.codex/agents/generated/<pack>/00-...md` | `docs/architecture/<area>/<stage-reports>/00-...md` | TBD | TBD | TBD | no |
| 01 <name> | pending | `.codex/agents/generated/<pack>/01-...md` | `docs/architecture/<area>/<stage-reports>/01-...md` | TBD | TBD | TBD | no |

## Что Обязательно Знать Дальше

| Stage | Факт / решение / ограничение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| 00 | TBD | TBD | TBD |
| 01 | TBD | TBD | TBD |

## Контракты, Миграции И Совместимость

| Stage | API / DTO | Persistence | Config / env | Browser-visible | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|---|
| 00 | none / compatible-change / breaking-change / unknown | TBD | TBD | TBD | TBD | TBD |
| 01 | none / compatible-change / breaking-change / unknown | TBD | TBD | TBD | TBD | TBD |

## Проверки И Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Result | Evidence path / note | Tests-only exception | Residual risk |
|---|---|---|---|---|---|---|
| 00 | TBD | TBD | TBD | TBD | TBD | TBD |
| 01 | TBD | TBD | TBD | TBD | TBD | TBD |

## Publish / Deploy Handoff

| Stage | Branch | Commit | PR | Checks before push | Deploy/runtime status | Notes |
|---|---|---|---|---|---|---|
| 00 | TBD | TBD | TBD | TBD | TBD | TBD |
| 01 | TBD | TBD | TBD | TBD | TBD | TBD |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| TBD | TBD | low / medium / high / blocker | TBD | TBD | no |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| YYYY-MM-DD | 00 | Created ledger from template. | `.codex/agents/stage_execution_ledger_template.md` |
