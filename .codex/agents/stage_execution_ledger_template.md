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
| Proof boundary | Для Mac Studio, target-host, deploy, runtime или production smoke фиксируйте один из labels: `target_host_readiness_pre_main`, `read_only_existing_runtime_smoke`, `post_main_production_runtime_proof`, или `N/A`. |
| File manifest | Каждый stage фиксирует created/modified/deleted/outside_expected_paths; для outside paths указывайте причину. |
| Branch / worktree / stash | По умолчанию stage работает на `main`; branch, worktree, stash или отдельная папка допустимы только при явном user approval и должны быть записаны ниже. |
| Docs index | Если менялись Markdown docs, фиксируйте `tools.docs.generate_docs_index` или причину, почему проверка не применима. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Proof boundary | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|---|
| 00 <name> | pending | `.codex/agents/generated/<pack>/00-...md` | `docs/architecture/<area>/<stage-reports>/00-...md` | TBD | N/A | TBD | TBD | no |
| 01 <name> | pending | `.codex/agents/generated/<pack>/01-...md` | `docs/architecture/<area>/<stage-reports>/01-...md` | TBD | N/A | TBD | TBD | no |

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

| Stage | Local gates | Real-boundary / e2e evidence | Proof boundary | Result | Evidence path / note | Tests-only exception | Residual risk |
|---|---|---|---|---|---|---|---|
| 00 | TBD | TBD | N/A | TBD | TBD | TBD | TBD |
| 01 | TBD | TBD | N/A | TBD | TBD | TBD | TBD |

## File Manifest

| Stage | Created | Modified | Deleted | Outside expected paths | Outside-path justification |
|---|---|---|---|---|---|
| 00 | none | TBD | none | none | N/A |
| 01 | none | TBD | none | none | N/A |

## Publish / Deploy Handoff

| Stage | Branch / worktree / stash status | Commit | PR | Checks before push | Deploy/runtime status | Docs index evidence | Notes |
|---|---|---|---|---|---|---|---|
| 00 | `main`; no branch/worktree/stash unless explicitly approved | TBD | TBD | TBD | TBD | TBD | TBD |
| 01 | `main`; no branch/worktree/stash unless explicitly approved | TBD | TBD | TBD | TBD | TBD | TBD |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| TBD | TBD | low / medium / high / blocker | TBD | TBD | no |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| YYYY-MM-DD | 00 | Created ledger from template. | `.codex/agents/stage_execution_ledger_template.md` |
