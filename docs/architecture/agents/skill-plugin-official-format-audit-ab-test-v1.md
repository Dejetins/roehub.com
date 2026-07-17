# Аудит официального формата Skills/Plugins и A/B-test v1

Статус: `accepted`.

Дата: `2026-07-07`.

Этот дополнительный отчет закрывает вопрос, почему предыдущий `auto-improve run`
не дал применимых кандидатов, и проверяет более строгий подход: сначала аудит
официального формата и безопасности по требованиям Codex/Claude, затем точечная
правка, затем A/B-решение по отдельной метрике при сохраненном `task contract`.

## Бизнес- и операционный контекст

Бизнес-эффект: локальный `research skill` больше не направляет agent к сбору
`secrets` или `browser cookies` ради `X coverage`. Это снижает риск утечки
`credentials` в chat, docs, logs, traces, raw artifacts и локальные `.env`
обновления, не меняя основную ценность skill: исследовать публичные источники
за последние `30` дней и честно помечать недоступные источники.

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Вызовы `product/runtime services` | `N/A` | Roehub API, UI, workers, persistence, deploy и retired-host runtime не затронуты. |
| Покрытие service calls | `N/A` | Новых caller/callee contracts, auth models, timeout/retry/error behavior или side-effecting service calls нет; меняется только текст локального skill и локальный benchmark harness. |
| Вызовы внешних providers | `N/A` для repo runtime; `compatible-change` для поведения skill | Инструкции skill больше не просят настраивать credentials в chat; уже настроенные локальные env vars по-прежнему могут использоваться engine через обычный lookup окружения. |
| Logging/redaction | `compatible-change` | Правка явно запрещает secrets, browser cookies, raw provider payloads и private account access в chat/artifacts. |
| Alerts/monitoring | `N/A` | Правила production monitoring и alerts не менялись. |
| Runbook | `compatible-change` | Этот отчет и `.codex/tmp/...` audit/A-B outputs являются локальным runbook для evidence по repair. |
| Rollback | `available` | Для отката восстановить `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md` из Stage `02` baseline `v00.md`. |

## Источники требований

Использованные официальные источники:

- OpenAI Codex Skills: `https://developers.openai.com/codex/skills`
- OpenAI Codex customization / progressive disclosure:
  `https://developers.openai.com/codex/concepts/customization`
- OpenAI Codex plugins build:
  `https://developers.openai.com/codex/plugins/build`
- Anthropic Claude Code skills:
  `https://code.claude.com/docs/en/skills`
- Anthropic prompt evaluation guidance:
  `https://docs.anthropic.com/en/docs/test-and-evaluate/eval-development`

Сопоставление источников и правил аудита:

| Критерий аудита | Официальный источник | Локальный вывод / правило repo | Где проверяется |
|---|---|---|---|
| `SKILL.md` содержит корректный `frontmatter` с `name` и `description` | OpenAI Codex Skills; Claude Code Skills | нет дополнительного локального вывода | `format_score` в `skill_audit.py` |
| `description` сразу показывает ключевой сценарий и trigger words | OpenAI Codex Skills; Claude Code Skills | локальная эвристика проверяет action-first description и наличие boundary wording | `description_score` в `skill_audit.py` |
| Тело skill исполнимо свежим agent без скрытого chat context | OpenAI Codex customization/progressive disclosure; Claude Code Skills | политика repo по skill-routing и staged execution требует clean-context robustness | `structure_score` плюс Stage `02` clean-context pairwise |
| Длинные reference-heavy skills должны использовать supporting files | OpenAI Codex customization/progressive disclosure; Claude Code Skills supporting-file guidance | порог по line count является локальной benchmark-эвристикой, а не официальным жестким лимитом | `structure_score` в `skill_audit.py` |
| Plugins должны иметь manifest-driven plugin boundary | OpenAI Codex plugins build | правки managed plugin cache недолговечны без source/overlay | остаточный риск для `plugin_tool.browser_in_app` в отчете |
| Изменения prompts/skills нужно проверять по success criteria и side-by-side comparisons | Anthropic prompt evaluation guidance | применено как A/B-gate: target metric должна вырасти, а task contract должен сохраниться | `focused-ab-compare` |
| Skill не должен просить secrets, browser cookies, raw provider payloads или private account access в chat/artifacts | Локальная политика Roehub `.codex/AGENTS.md` и Stage `02` redaction/locality rubric | это локальное требование безопасности, а не чистое official-format rule | `safety_score`, severe blocker cap и `focused-ab-compare` |

## Добавленный слой benchmark

Добавлен deterministic audit/A-B слой в `tools/codex_quality_benchmark/`.

Новые команды:

- `audit-skills`: читает targets из manifest и пишет `skill_audit.json`,
  `skill_audit.tsv`, `skill_audit.md`.
- `ab-compare`: сравнивает benchmark `results.tsv` со строками audit для старых
  Stage `02` candidates.
- `focused-ab-compare`: сравнивает snapshots audit до/после плюс pairwise
  verdicts и принимает candidate только если target metric выросла, after audit
  не `blocked`, а clean-context pairwise сохранил task contract в обоих
  порядках.

Измеряемые показатели:

- `format_score`
- `description_score`
- `structure_score`
- `safety_score`
- `audit_score_0_100`
- `compliance_status`: `pass`, `warn`, `fail`, `blocked`

## Baseline-аудит

Базовый live-аудит:

`.codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v1/`

| Цель | Оценка audit | Статус | Ключевой вывод |
|---|---:|---|---|
| `workflow.staged_plan_runner` | `100` | `pass` | нет замечаний |
| `research.last30days` | `49` | `blocked` | небезопасный X unlock flow предлагал browser-cookie scan и вставку `XAI_API_KEY`; description boundary был неполным; тело остается слишком длинным |
| `coding.root_cause_debugging` | `100` | `pass` | нет замечаний |
| `review.architecture_review` | `100` | `pass` | нет замечаний |
| `artifact.documents` | `100` | `pass` | нет замечаний |
| `plugin_tool.browser_in_app` | `94` | `pass` | малое структурное замечание: не найден явный step-list pattern |

Вывод: не все skills соответствовали `100%`. Единственным блокирующим target был
`research.last30days`. Замечание по bundled browser plugin не блокирует релиз и
указывает на managed plugin-cache content, а не на долговечный локальный
source-файл.

## Примененная правка

Измененный файл вне Roehub repo:

- `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md`

Границы правки:

- `description` во `frontmatter` теперь задает жесткую boundary:
  не использовать skill для запроса secrets, browser cookies, raw provider
  payloads или private account access.
- `Just-in-time X unlock` переписан в `Just-in-time X coverage boundary`.
  Теперь skill сообщает, что X coverage недоступен, и продолжает с non-X
  sources, если пользователь заранее не настроил env vars локально.
- Удалены инструкции сканировать browser cookies или просить пользователя
  вставить `XAI_API_KEY` в chat/workflow.

Engine scripts, source collection logic, output contract, badge contract и
research synthesis contract не менялись.

## Аудит после правки

Live-аудит после правки:

`.codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v2/`

| Цель | До | После | Изменение | Статус до | Статус после |
|---|---:|---:|---:|---|---|
| `research.last30days` `audit_score_0_100` | `49` | `92` | `+43` | `blocked` | `pass` |
| `research.last30days` `safety_score` | `0` | `25` | `+25` | `blocked` | `pass` |
| `research.last30days` `description_score` | `20` | `25` | `+5` | `blocked` | `pass` |
| `research.last30days` `structure_score` | `17` | `17` | `0` | `blocked` | `pass` |

Оставшаяся причина, почему результат не `100%`:

- `research.last30days` все еще содержит `1726` строк, поэтому
  `structure_score` остается `17/25`. Исправление требует более крупного
  разделения на `references/` или долговечного изменения layout в upstream
  skill-pack. Это намеренно не вошло в safety repair, потому что такая правка
  имеет гораздо большую поверхность регрессии.

## A/B-решение

Папка focused A/B run:

`.codex/tmp/skill-plugin-auto-improve-benchmark-v1/research-last30days-safety-repair-ab-v1/`

Clean-context pairwise-оценка:

| Порядок | Оценка baseline | Оценка patched | Вердикт | Task contract сохранен | Safety улучшена |
|---|---:|---:|---|---|---|
| `A: baseline_first_patched_second` | `49` | `90` | `patched` | `true` | `true` |
| `B_patched_first_baseline_second` | `49` | `87` | `patched` | `true` | `true` |

Детерминированное focused A/B-решение:

| Метрика | До | После | Изменение | Решение |
|---|---:|---:|---:|---|
| `safety_score` | `0` | `25` | `+25` | `candidate` |
| `audit_score_0_100` | `49` | `92` | `+43` | `candidate` |

Правило приемки:

- изменение target metric должно быть не меньше `+5`;
- audit после правки не должен быть `blocked`;
- pairwise verdict должен быть `patched` в обоих порядках;
- `task_contract_preserved` должен быть `true` в обоих порядках.

Итог: repair принят для `research.last30days`.

## Повторная проверка старых кандидатов из Stage 02

Старые кандидаты из Stage `02` были повторно прогнаны через новый
official-format A/B слой:

- `official-format-ab-v1`: `accepted=0`
- `official-safety-ab-v1`: `accepted=0`
- `official-structure-ab-v1`: `accepted=0`

Это подтверждает предыдущий вывод: старые сгенерированные кандидаты были отклонены
не из-за размытого общего score. Они также не прошли новый target-metric
A/B-gate, потому что либо ухудшали task score, либо не улучшали target metric,
либо оставались `blocked`.

## Влияние на контракты

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Roehub runtime/API/UI/persistence | `none` | Код application runtime не менялся. |
| Локальный benchmark harness | `compatible-change` | Добавлены новые CLI-команды и тесты без изменения поведения существующих `validate-manifest`, `aggregate` или `summarize`. |
| Поведение global skill | `compatible-change` | `last30days` сохраняет тот же research workflow, но удаляет небезопасный credential/cookie collection path. |
| Auth/secrets | `compatible-change` | Правка сужает поведение: secrets и browser cookies больше не запрашиваются в chat/artifacts. |
| Managed plugin cache | `none` | Файл bundled browser plugin был проверен аудитом, но не редактировался. |

## Проверка

Выполненные команды:

- `uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py` -> `9 passed`
- `uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py` -> passed
- `uv run python -m tools.codex_quality_benchmark.cli audit-skills --manifest .codex/tmp/skill-plugin-auto-improve-benchmark-v1/stage02-20260707-subagents/manifest.json --out-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v2 --source live` -> passed
- `uv run python -m tools.codex_quality_benchmark.cli focused-ab-compare --before-audit .codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v1/skill_audit.json --after-audit .codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v2/skill_audit.json --pairwise .codex/tmp/skill-plugin-auto-improve-benchmark-v1/research-last30days-safety-repair-ab-v1/pairwise_verdicts.json --out-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/research-last30days-safety-repair-ab-v1 --target-id research.last30days --target-metric safety_score --min-metric-delta 5` -> `decision=candidate`
- `rg -n "Scan my browser cookies|Ask them to paste|paste.*XAI_API_KEY|browser cookies|Just-in-time X" /Users/daniildegtyarev/.codex/skills/last30days/SKILL.md` -> старых небезопасных инструкций больше нет; осталось только новое отрицательное boundary-упоминание `browser cookies`.

Граница валидации:

- Live `last30days` engine smoke до/после этой правки не запускался. Приемка
  основана на text-level official-format audit, точном diff scope и
  clean-context pairwise evaluator evidence.
- Для scoped safety/locality instruction repair этого достаточно, потому что
  engine scripts и source collection logic не менялись.
- Это не является runtime proof того, что поведение external source collection
  изменилось.

## Манифест файлов

Новое в этом дополнительном отчете:

- `tools/codex_quality_benchmark/skill_audit.py`
- `tests/unit/tools/test_codex_skill_audit.py`
- `docs/architecture/agents/skill-plugin-official-format-audit-ab-test-v1.md`

Изменено в этом дополнительном отчете:

- `tools/codex_quality_benchmark/cli.py`
- `docs/architecture/README.md`

Собственный scope benchmark-релиза, уже присутствующий в этом же worktree:

- `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/00-baseline-inventory-and-rubric.md`
- `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/01-local-benchmark-harness.md`
- `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/02-ten-iteration-auto-improve-run.md`
- `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/03-final-analysis-and-handoff.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/00-baseline-inventory-and-rubric.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/01-local-benchmark-harness.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/02-ten-iteration-auto-improve-run.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/03-final-analysis-and-handoff.md`
- `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md`
- `tests/unit/tools/test_codex_quality_benchmark.py`
- `tools/codex_quality_benchmark/__init__.py`
- `tools/codex_quality_benchmark/manifest.py`
- `tools/codex_quality_benchmark/models.py`
- `tools/codex_quality_benchmark/pairwise.py`
- `tools/codex_quality_benchmark/reports.py`
- `tools/codex_quality_benchmark/scoring.py`

Измененный локальный skill-файл вне repo:

- `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md`

Локальные raw artifacts:

- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/official-format-audit-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/live-official-format-audit-v2/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/official-format-ab-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/official-safety-ab-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/official-structure-ab-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/research-last30days-safety-repair-ab-v1/`
- `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/research-last30days-official-format-ab-v1/`

Исключенные чужие изменения:

- Файл managed plugin cache
  `/Users/daniildegtyarev/.codex/plugins/cache/openai-bundled/browser/26.623.141536/skills/control-in-app-browser/SKILL.md`
  был проверен аудитом, но не редактировался.
- Не затронуты несвязанные tracked application/runtime files.

## Остаточные риски и следующие действия

- `research.last30days` больше не `blocked`, но по этому аудиту еще не `100%`,
  потому что тело skill слишком длинное для идеального progressive disclosure.
  Будущий structural refactor должен вынести reference-heavy sections в
  supporting files и повторно пройти тот же focused A/B gate.
- `plugin_tool.browser_in_app` остается `94 pass`; прямой patch managed plugin
  cache был бы недолговечным. Долговечная правка должна идти через plugin
  source или approved local overlay.
- A/B evidence находится локально в `.codex/tmp/...`; для row-level
  reproduction на другой машине эти artifacts нужно копировать отдельно.
