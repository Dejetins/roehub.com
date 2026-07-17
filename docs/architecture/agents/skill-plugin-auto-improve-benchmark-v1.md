# Skill/Plugin Auto-Improve Benchmark v1

Статус: `historical benchmark; execution retired`.

Дата: 2026-07-07.

Исторический пакет исполнения:

- `plan_doc`: `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/`
- `stage_ledger`: `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/skill-plugin-auto-improve-benchmark-v1-stage-ledger.md`
- `execution_mode`: `superseded`
- intended_agent_model: `gpt-5.5`
- reasoning_effort: `xhigh`
- Этот пакет не выбирает текущую работу и не разрешает Goal. Используйте
  глобальный delivery contract и current ticket; ссылки ниже сохранены только
  как benchmark evidence.

## Цель

Создать локальный инструмент и процесс, который позволяет аудировать и улучшать
локальные Codex skills/plugins по единой методологии, близкой к
`crimeacs/auto-improve`: mutation -> independent evaluation -> pairwise
champion gate -> keep/discard -> iteration log.

Нужный результат:

- каждый вариант каждого проверяемого файла имеет `version_id`, `sha256` и
  `score_0_100`;
- для каждого target есть baseline `v00` и ровно 10 iteration-attempt rows:
  успешные candidate attempts, `no_op` attempts или `blocked` attempts;
- оценки разных версий считаются по одинаковой рубрике и одинаковым кейсам;
- clean-context тестирование выполняется через Codex subagents или другой явно
  изолированный evaluator, но все Python scripts, manifests, TSV/JSONL и
  итоговые отчеты сохраняются только локально на этой машине;
- изменения в исходные skills/plugins применяются только после явного keep gate
  и только в scoped файлы.

## Методологический источник

Исполнитель должен перед Stage `00` заново открыть и записать наблюдаемый источник
методологии:

- `https://github.com/crimeacs/auto-improve`
- `https://raw.githubusercontent.com/crimeacs/auto-improve/main/README.md`
- `https://raw.githubusercontent.com/crimeacs/auto-improve/main/criteria/README.md`

На момент создания плана сеть для `git ls-remote` не дала надежный SHA без
зависания, поэтому stage prompt требует зафиксировать observed source URL, дату
чтения и, если доступен, commit SHA. Без SHA Stage `00` может быть `accepted`,
если README/criteria были прочитаны и методология явно записана как source URL
snapshot, но Stage `01` должен пометить это как residual reproducibility risk.

Ключевые принципы, которые переносим:

- отдельный evaluator не должен быть тем же контекстом, который мутировал файл;
- лучший candidate каждого раунда сравнивается с текущим champion pairwise;
- keep разрешен только при строгом выигрыше candidate над champion;
- malformed patch или неполная версия не портит исходный файл, а получает
  `discarded`;
- rubric фиксируется до сравнения версий, состоит из независимых dimensions и
  суммируется до 100;
- 10 итераций являются обязательным evidence grid. Ранняя остановка реальных
  edits допустима только если оставшиеся iteration rows явно записаны как
  `no_op` или `blocked` с причиной, score/hash текущего champion и теми же
  comparable metrics.

## Бизнес-смысл

Локальные skills/plugins сильно влияют на качество ответов Codex: routing,
контекстные бюджеты, local-only ограничения, redaction, verification depth и
формат отчетов. Без измеримого benchmark улучшение skill-файла легко превращается
в вкусовое переписывание. Этот план делает изменения сравнимыми: один target,
одна рубрика, фиксированные clean-context кейсы, score каждой версии и журнал
причин keep/discard.

## Охват

Входит:

- инвентаризация локальных skills и plugin skills на текущей машине;
- классификация target files по типам;
- локальный benchmark harness для manifest, version snapshots, rubric,
  candidate score aggregation, pairwise decisions и summary reports;
- 10 improvement iterations на выбранном ограниченном наборе targets;
- clean-context response tests через subagents;
- локальные отчеты `TSV`, `JSONL`, `Markdown` и stage reports;
- итоговая рекомендация, какие skill/plugin изменения можно применить, оставить
  как candidate или отклонить.

Не входит:

- изменение production runtime Roehub;
- изменение public API, persistence, ClickHouse/Postgres schema, browser UI,
  deploy pipeline или retired-host runtime;
- установка внешних сервисов или отправка секретов наружу;
- автоматический `git commit`, `push`, branch, worktree или stash;
- массовое переписывание всех skills/plugins без target manifest;
- blind acceptance от самого mutator-контекста.

## Local-Only И Clean-Context Контракт

Все скрипты, файлы, manifests, сырые ответы, score tables и summaries должны
создаваться на текущей машине. Рекомендуемые пути:

| Тип артефакта | Путь |
|---|---|
| durable plan/ledger/reports | `docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/` |
| prompt pack | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/` |
| Python harness | `tools/codex_quality_benchmark/` |
| optional CLI wrapper | `scripts/codex_quality_benchmark/` |
| raw local run state, not committed | `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/` |

Subagents разрешены как cloud clean-context evaluators только для sanitized prompt
and answer evaluation. Главный executor обязан:

- не передавать subagents секреты, cookies, tokens, private keys, env dumps,
  raw provider payloads или sensitive local paths beyond what is needed;
- копировать subagent verdicts в локальные JSON/TSV artifacts;
- не считать subagent filesystem state source of truth;
- если пользователь потребует zero off-machine content, переключиться на
  local-only evaluator mode и записать `subagent_clean_context=false`.

Python scripts не должны сами обращаться к внешним LLM APIs по умолчанию. Если
будет добавлен provider adapter, он должен быть opt-in, redacted, local-configured
и не должен хранить raw secrets in repo artifacts.

## Target Types

Каждый target получает один `skill_type`, чтобы изменения были сравнимыми внутри
типа:

| Type | Примеры | Единая суть улучшений |
|---|---|---|
| `workflow_skill` | staged execution, publish/deploy, prompt-manager | tighter routing, state sources, blocker behavior, final report contract |
| `research_skill` | last30days, web/source research | source discipline, recency, citation limits, partial-source caveats |
| `coding_skill` | backend gates, root cause, performance | reproduction, scoped fix, tests, evidence and regression guard |
| `review_skill` | architecture-review, production-risk-review | findings-first, fact vs inference, severity and smallest fix |
| `artifact_skill` | documents, pdf, spreadsheets, presentations | render/openability gates, artifact verification, visual QA |
| `plugin_tool_skill` | browser, GitHub, product-design plugins | tool routing, auth/redaction, clean handoff, local persistence boundaries |

Для каждого type десять iteration approaches одинаковы по сути:

1. `routing_precision`: уточнить when-to-use / when-not-to-use.
2. `context_budget`: ограничить чтение и stop conditions.
3. `input_output_contract`: зафиксировать входы, выходы, artifacts.
4. `failure_blockers`: описать fail-closed и blocker reporting.
5. `verification_depth`: добавить evidence surfaces and edge cases.
6. `clean_context`: сделать prompt executable without hidden chat memory.
7. `locality_redaction`: усилить local-only, secrets and raw payload rules.
8. `examples`: добавить minimal positive/negative examples where useful.
9. `consistency`: убрать противоречия, дубли, stale instructions.
10. `compression_final`: сократить без потери routing, safety and evidence.

Executor может адаптировать конкретные edits под type, но обязан сохранить один
approach label and comparable metrics across all targets in that iteration.

## Метрики

Основной score: `score_0_100`, сумма dimensions:

| Dimension | Points | Что оценивает |
|---|---:|---|
| Routing precision | 15 | skill/plugin clearly activates only for right tasks |
| Context economy | 10 | reads enough, avoids broad preload and hidden memory |
| Task execution clarity | 15 | steps are actionable for a fresh Codex window |
| Safety and locality | 15 | local-only, secrets, provider payloads, branch/worktree boundaries |
| Verification depth | 15 | evidence, tests, edge cases and real-boundary checks where relevant |
| Clean-context robustness | 10 | works without author chat history |
| Failure behavior | 10 | blockers, malformed candidates, partial evidence and rollback are clear |
| Output/report quality | 10 | final report shape, score logging and handoff are consistent |

Дополнительные comparable metrics:

- `dimension_scores_json`
- `pairwise_verdict`: `candidate | champion | tie | blocked | not_run`
- `candidate_vs_champion`: `2-0 | 1-1 | 0-2 | not_run`
- `eval_cases_total`
- `eval_cases_passed`
- `contract_violations`
- `secret_redaction_violations`
- `locality_violations`
- `estimated_context_tokens`
- `patch_apply_status`
- `score_delta`

Keep rule:

- candidate may replace champion only when it has higher average score and wins
  both pairwise orderings against champion;
- if pairwise verdict is `champion` or `tie`, keep champion;
- `not_run` is valid only for baseline `v00`, malformed/no-candidate `no_op`
  rows, or blocked rows where pairwise evaluation could not be safely run;
- if any severe safety/locality/redaction violation appears, candidate is
  discarded regardless of numeric score;
- if evaluator outputs are incomplete, mark `blocked` or rerun the evaluator,
  but do not keep by inference.

## Целевая Архитектура Harness

Планируемый local module:

```text
tools/codex_quality_benchmark/
  __init__.py
  cli.py
  models.py
  manifest.py
  scoring.py
  pairwise.py
  reports.py
```

Планируемый local run layout:

```text
.codex/tmp/skill-plugin-auto-improve-benchmark-v1/<run_id>/
  manifest.json
  rubric.md
  targets/<target_id>/v00.md
  targets/<target_id>/v01.md
  evaluations/<target_id>/<version_id>/<case_id>.json
  pairwise/<target_id>/<iteration_id>.json
  results.tsv
  events.jsonl
  summary.md
```

Durable summary may be copied to:

```text
docs/architecture/agents/skill-plugin-auto-improve-benchmark-v1-stage-reports/
  benchmark-summary-<run_id>.md
```

Raw local state under `.codex/tmp/` must not be committed.

## Операционные Аспекты

| Surface | Решение |
|---|---|
| Roehub runtime / deploy | `N/A`: план не меняет production services, retired-host runtime, API, UI, workers или deploy workflow. |
| Service calls | `N/A` для Roehub продукта. Clean-context Codex subagents используются как evaluator boundary, но Python harness не вызывает external LLM APIs по умолчанию. |
| Auth/secrets | `N/A` для product auth. Секреты, tokens, cookies, raw provider payloads и env dumps запрещены в prompts, ledgers, reports, subagent packets и local run artifacts. |
| Retry/idempotency | `N/A` для side effects. Повтор evaluator call допускается только как новый local event с новым `evaluation_id`; keep/discard не перезаписывается без trace. |
| Alerts/monitoring | `N/A` для production alerts. Операционный сигнал плана - stage status in ledger: `blocked`, `accepted`, `accepted_for_learning`, `completed`. |
| Runbook | Этот plan_doc плюс stage prompts являются runbook для локального benchmark. Отдельный runtime runbook не нужен. |

## План Внедрения

| Stage | Prompt | Purpose | Acceptance |
|---|---|---|---|
| `00` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/00-baseline-inventory-and-rubric.md` | Read repo rules, audit local skill/plugin inventory, freeze target manifest, rubric, eval cases and auto-improve source snapshot. | Ledger has target manifest summary, rubric dimensions, source snapshot and Stage `01` allowed. |
| `01` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/01-local-benchmark-harness.md` | Implement local Python harness and deterministic report schemas. | Harness validates fixtures, writes local TSV/JSONL/summary, and never requires external APIs by default. |
| `02` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/02-ten-iteration-auto-improve-run.md` | Run exactly 10 improvement iteration attempts with clean-context subagent evaluation and pairwise keep/discard. | Every target has `v00` plus iteration rows `1..10`; no-op/blocked rows are explicit; raw state is local; champion decisions are recorded. |
| `03` | `.codex/agents/generated/skill-plugin-auto-improve-benchmark-v1/03-final-analysis-and-handoff.md` | Produce final benchmark report, apply only approved scoped improvements if requested, and close or block the plan. | Final report explains winners, rejected approaches, residual risks, and whether local skill edits are proposed/applied. |

## Validation Ladder

Stage `00`:

- docs/index check after Markdown updates;
- manifest completeness and target classification review;
- no source skills/plugins edited.

Stage `01`:

- `uv run ruff check tools/codex_quality_benchmark`;
- focused pytest for manifest/scoring/pairwise/report fixtures;
- sample local run that produces `results.tsv`, `events.jsonl`, `summary.md`.

Stage `02`:

- local score aggregation gate;
- clean-context evaluator outputs for fixed cases;
- pairwise `2-0` keep/discard record per accepted candidate;
- secret/locality violation scan on artifacts;
- reproducibility rerun of score aggregation from saved JSON.

Stage `03`:

- docs index check;
- final report score table and file manifest;
- no raw secrets, provider payloads or uncommitted raw `.codex/tmp` state in durable docs;
- if applying edits to real skill/plugin files, scoped diff review only and no broad git staging.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Roehub public API | `none` | Plan targets local Codex skills/plugins and local harness only. |
| Roehub persistence/schema | `none` | No DB schema or persisted app contract changes. |
| Runtime/deploy | `none` | No retired-host deploy or production smoke in scope. |
| Local Codex workflow | `compatible-change` | Adds optional local benchmark and reports. |
| Skill/plugin files | `unknown` until Stage `03` | Stage `02` proposes versions; Stage `03` must classify actual edited files. |
| Secrets/redaction | `compatible-change` | Strengthens redaction and local-only requirements. |

## Риски И Открытые Вопросы

- Subagent evaluation is useful for clean context but is not fully local compute.
  This is acceptable only because user explicitly suggested subagents; raw scripts
  and stored artifacts stay local.
- If subagents are unavailable, Stage `02` must block or use a clearly labeled
  local evaluator fallback as `accepted_for_learning` only. Fallback evidence
  must not create apply-ready candidates; Stage `03` may summarize/defer/reject
  but cannot apply skill/plugin edits from fallback-only evidence.
- Full inventory of every installed plugin may be too large for one run. Stage
  `00` must cap target count or split into batches before Stage `02`.
- Scores are evaluator-model dependent. The benchmark is comparable inside one
  run only if model, rubric, cases and pairwise gate stay fixed.
- Direct editing of global skill files under `/Users/daniildegtyarev/.codex/skills`
  is outside Roehub git. Stage `03` must separate repo-durable reports from
  local user-profile edits and ask for explicit approval before applying global
  skill changes.
