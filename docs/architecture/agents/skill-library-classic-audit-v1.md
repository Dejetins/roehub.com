# Skill Library Classic Audit v1

Статус: `historical audit; execution retired`.

Дата: 2026-07-07.

Исторический пакет исполнения:

- `plan_doc`: `docs/architecture/agents/skill-library-classic-audit-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/skill-library-classic-audit-v1/`
- `stage_ledger`: `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/skill-library-classic-audit-v1-stage-ledger.md`
- `execution_mode`: `superseded`
- intended_agent_model: `gpt-5.5`
- reasoning_effort: `xhigh`
- Этот пакет не выбирает текущую работу и не разрешает Goal. Используйте
  глобальный delivery contract и current ticket; ссылки ниже сохранены только
  как audit evidence.

## Цель

Сделать отдельный полный классический аудит всей локальной библиотеки skills/plugins:

- каждый найденный `SKILL.md` должен попасть в inventory;
- каждый skill должен быть проверен основной моделью и минимум одним clean-context subagent;
- по каждому skill должен быть итог: назначение, сильные стороны, проблемы, риски, конкретные предложения улучшений, приоритет и рекомендуемое следующее действие;
- исходные skill/plugin файлы не изменяются в этом цикле. Это audit-only задача.

## Бизнес-смысл

Локальная библиотека skills/plugins определяет, как Codex выбирает инструменты,
читает контекст, проверяет результат, работает с секретами и завершает отчеты.
Классический аудит нужен до auto-improve-итераций, чтобы увидеть общий уровень
качества библиотеки: какие skills уже достаточно надежны, какие требуют
точечного усиления, какие стоит разделить, объединить или переписать. Итоговый
backlog дает понятный список улучшений по каждому skill, но не меняет сами
skills без отдельного решения.

## Что Считается Библиотекой

Stage `00` должен сформировать фактический список. Default search roots:

| Root | Назначение |
|---|---|
| `/Users/daniildegtyarev/.codex/skills` | пользовательские и локально установленные skills |
| `/Users/daniildegtyarev/.codex/plugins/cache` | skills, поставляемые установленными plugins |
| `/Users/daniildegtyarev/.codex/skills/.system` | system skills, если доступны для чтения |

Если любой configured root недоступен или не может быть проверен, Stage `00`
должен остановиться как `blocked`, если пользователь явно не согласовал reduced
scope. Неполный охват без такого approval не может открыть Stage `01`.

Stage `00` должен дедуплицировать найденные `SKILL.md` по canonical/resolved
path до назначения `skill_id`: `/Users/daniildegtyarev/.codex/skills/.system`
может быть вложенным подмножеством `/Users/daniildegtyarev/.codex/skills`.

## Пайплайн

| Stage | Prompt | Purpose | Acceptance |
|---|---|---|---|
| `00` | `.codex/agents/generated/skill-library-classic-audit-v1/00-inventory-and-batch-plan.md` | Полная инвентаризация `SKILL.md`, hash/metadata, canonical-path dedupe, классификация и batch plan для subagents. | Все configured roots проверены; у каждого canonical skill есть `skill_id`, path, hash, type, batch id; недоступный root блокирует stage без explicit reduced-scope approval. |
| `01` | `.codex/agents/generated/skill-library-classic-audit-v1/01-subagent-batch-audits.md` | Read-only аудит каждого batch основной моделью и subagents. | Для каждого skill есть main-model review, subagent review, `subagent_evidence_ref`, hash-drift check и coverage reconciliation row или явный blocker. |
| `02` | `.codex/agents/generated/skill-library-classic-audit-v1/02-consolidated-improvement-backlog.md` | Свести результаты в общий backlog улучшений по каждому skill. | Финальный отчет содержит `what_works`, предложение улучшений, приоритет, риски, coverage status и next action по каждому skill. |

## Метод Аудита

Классический аудит не делает auto-improve mutation и не выбирает champion. Он
проверяет текущий skill как инструкционный контракт:

1. Назначение и activation boundary.
2. When-to-use / when-not-to-use.
3. Context acquisition and stop conditions.
4. Workflow clarity for a fresh context window.
5. Tool routing and plugin/MCP boundaries.
6. Local-only, secret redaction and persistence rules.
7. Verification gates and evidence quality.
8. Failure/blocker behavior.
9. Output/report shape.
10. Contradictions, stale instructions and over-broad scope.

## Subagent Контракт

Subagents используются как clean-context reviewers, а не как авторы правок:

- subagent получает только один batch и bounded instructions;
- subagent не редактирует файлы;
- subagent возвращает structured report по каждому skill;
- главный executor сверяет subagent report со своим review;
- если main review и subagent review конфликтуют, итог должен явно показать конфликт и выбрать conservative recommendation;
- если subagents недоступны, Stage `01` блокируется, потому что задача пользователя требует subagent coverage.

## Выходные Артефакты

| Artifact | Path |
|---|---|
| inventory/batch report | `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md` |
| batch audit report | `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/01-subagent-batch-audits.md` |
| consolidated backlog | `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md` |

Дополнительный local raw-state каталог не создается в этом плане. Если
исполнителю нужен отдельный scratch/local-state path, это требует явного user
approval и lifecycle rule до записи файлов. Durable docs should contain
summaries, hashes, file paths and recommendations, not large copied skill bodies.

## Минимальная Схема По Каждому Skill

Every final per-skill row must include:

- `skill_id`
- `source`: `user_skill | system_skill | plugin_skill | unknown`
- `path`
- `sha256`
- `batch_id`
- `skill_type`
- `what_works`
- `main_model_verdict`
- `subagent_verdict`
- `subagent_evidence_ref`
- `top_findings`
- `improvement_proposals`
- `priority`: `P0 | P1 | P2 | P3`
- `risk_if_unchanged`
- `recommended_next_action`: `leave_as_is | rewrite_prompt_contract | add_examples | tighten_routing | split_skill | merge_or_deprecate | needs_manual_decision`

`improvement_proposals` is required for every skill. If no material issue is
found, the proposal should be an explicit low-priority no-op or optional polish
recommendation, not a blank field.

## Coverage Reconciliation Schema

Stage `01` and Stage `02` must include a per-skill coverage reconciliation table
with:

- `skill_id`
- `batch_id`
- `inventory_sha256`
- `review_sha256`
- `hash_drift_status`: `same | changed | blocked`
- `main_review_status`: `done | blocked`
- `subagent_review_status`: `done | blocked`
- `subagent_evidence_ref`
- `clean_context_input_scope`
- `coverage_status`: `covered | blocked`

If `review_sha256` differs from `inventory_sha256`, the affected skill is blocked
until the inventory is refreshed or the user explicitly approves reviewing the
changed file version.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Roehub public API | `none` | Audit-only docs/prompt pack. |
| Roehub persistence/schema | `none` | No DB changes. |
| Runtime/deploy | `none` | No production runtime or retired-host deploy. |
| Local skill/plugin files | `none` in this plan | Source skill files are read-only. Improvements are proposals only. |
| Local Codex workflow | `compatible-change` | Adds a repeatable audit process and reports. |

## Операционные Аспекты

| Surface | Решение |
|---|---|
| Service calls | `N/A` for Roehub runtime. Subagents are used as read-only review workers inside Codex, not as product service calls. |
| Secrets/redaction | Skill bodies and reports must not include secrets, tokens, cookies, env dumps or raw provider payloads. |
| Retry/idempotency | Re-running a subagent review creates a new local evidence record; do not overwrite old evidence without noting supersession. |
| Alerts/monitoring | `N/A` for production alerts. Stage status in the ledger is the operational signal. |
| Runbook | This plan, prompt pack and ledger are the local runbook. |

## Риски

- Библиотека skills/plugins может быть большой. Stage `00` должен batch the work
  and keep every skill covered, not shrink scope silently.
- Configured roots overlap. Stage `00` must deduplicate by canonical/resolved
  path so `.system` skills are not counted twice.
- Subagent output can be inconsistent. Main executor must resolve conflicts
  conservatively and record disagreements.
- Plugin cache paths may change across plugin versions. Reports must include
  observed path and hash.
- Audit proposals are not implementation approval. Applying changes to global
  skills/plugins needs a separate explicit task.
