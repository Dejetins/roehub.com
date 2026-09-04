> Historical copy edited on 2026-09-04 to remove retired tooling references.
> Original results and totals describe the original run, not current validation.
> Unmodified originals: `/Users/daniildegtyarev/Documents/Codex/2026-09-04/new-chat/outputs/removed-skills-backup-2026-09-04.tar.gz`.

# Полный all-skills аудит официального формата и A/B-test v1

Статус: `accepted`.

Дата: `2026-07-07`.

Этот отчет расширяет предыдущую точечную правку до полного набора установленных `skills`. Проверка была выполнена не только по первым `6` целевым `skills`, а по всему найденному `inventory`: локальные `global/system skills` и `skills` из установленного `plugin cache`.

Цель проверки: для каждого `skill` получить воспроизводимую A/B-строку, где видно, изменился ли целевой показатель `audit_score_0_100`, сохранился ли общий `task contract` по статусу `audit`, и нужно ли применять `candidate` или оставить `baseline`.

## Область проверки

- Всего найдено skills: `61`.
- Локальные/system/global skills: `23`.
- Skills из managed plugin cache: `38`.
- Источник `inventory`: `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v2/skill_inventory.json`.
- `Baseline audit`: `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v1/skill_audit.json`.
- `After audit`: `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v2/skill_audit.json`.
- A/B-решения: `.codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-ab-v1/all_skills_ab_decisions.json`.

## Методика

Методика осталась той же, что в предыдущем отчете, но применена ко всему `inventory`:

1. `audit-all-skills` собирает все `SKILL.md` под `/Users/daniildegtyarev/.codex/skills` и `/Users/daniildegtyarev/.codex/plugins/cache`.
2. Для каждого skill считается `format_score`, `description_score`, `structure_score`, `safety_score`, общий `audit_score_0_100` и `compliance_status`.
3. Точечные правки делаются только там, где можно улучшить `routing/discovery` или `safety boundary` без переписывания `workflow body`.
4. `all-skills-ab-compare` сравнивает baseline и after по целевой метрике `audit_score_0_100`.
5. `Candidate` принимается только если целевая метрика выросла, `after audit` не `blocked`, а общий статус не ухудшился.
6. Если `skill` уже был `pass` и метрика не выросла, строка получает `baseline_retained`.

Это не `runtime smoke` самих `plugin tools`. Это A/B на уровне текста `SKILL.md`: официальный формат, `safety/routing` и границы применения.

## Сводка результата

| Показатель | До | После |
|---|---:|---:|
| `pass` | `56` | `61` |
| `warn` | `5` | `0` |
| `fail` | `0` | `0` |
| `blocked` | `0` | `0` |

| A/B-решение | Количество |
|---|---:|
| `baseline_retained` | `54` |
| `candidate` | `7` |

Принято `candidates`: `7`. Из них локальных `system/global`: `3`, из `managed plugin-cache`: `4`.

После правок все `61` skills имеют `compliance_status: pass`. При этом не все получили `100/100`: ниже `100` осталось из-за длинных тел `skills`, `nonportable plugin names` или малых эвристических замечаний по `description/executable-step`, которые уже не блокируют `audit`.

## Принятые A/B `candidates`

| Цель | `managed_cache` | До | После | Изменение | Статус до | Статус после | Что изменено |
|---|---:|---:|---:|---:|---|---|---|
| `global.numba-jit-performance` | `false` | `95` | `100` | `5` | `pass` | `pass` | `frontmatter` `description` начинается с `Use for ...`, чтобы ключевой сценарий был в начале описания. |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.jobs` | `true` | `87` | `92` | `5` | `warn` | `pass` | `frontmatter` `description` переписан в формате `action-first`; добавлена граница против чисто локальных или не-Hugging Face workloads. |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.llm-trainer` | `true` | `87` | `92` | `5` | `warn` | `pass` | `frontmatter` `description` переписан в формате `action-first`; добавлена граница против generic prompt tuning/local-only LLM work. |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.paper-publisher` | `true` | `87` | `92` | `5` | `warn` | `pass` | `frontmatter` `description` переписан в формате `action-first`; добавлена граница против generic academic writing без Hugging Face Hub publishing. |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.vision-trainer` | `true` | `82` | `92` | `10` | `warn` | `pass` | `frontmatter` `description` переписан в формате `action-first`; добавлена граница против general image editing/local-only CV tasks. |
| `system.skill-creator` | `false` | `95` | `100` | `5` | `pass` | `pass` | `frontmatter` `description` переписан в формате `action-first` `Use when ...`; добавлена граница к `skill-installer`. |

## Полный список A/B по всем skills

| Цель | `scope` | `managed_cache` | До | После | Изменение | Статус после | Решение | Остаточные `findings` после |
|---|---|---:|---:|---:|---:|---|---|---|
| `global.architecture-design` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.architecture-review` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.backend-performance-evidence` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.backend-quality-gates` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.browser-qa-evidence` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.contract-impact-analysis` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.data-analytics-methodology` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.last30days` | `global_skill` | `false` | `92` | `92` | `0` | `pass` | `baseline_retained` | `skill_body_too_long` |
| `global.numba-jit-performance` | `global_skill` | `false` | `95` | `100` | `5` | `pass` | `candidate` | `none` |
| `global.playwright` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.pre-ship-gate` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.production-risk-review` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.prompt-manager` | `global_skill` | `false` | `92` | `92` | `0` | `pass` | `baseline_retained` | `skill_body_too_long` |
| `global.publish-ci-deploy` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.root-cause-debugging` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.staged-plan-runner` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `global.topological-data-analysis` | `global_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-bundled.browser.26.623.141536.control-in-app-browser` | `plugin_skill` | `true` | `94` | `94` | `0` | `pass` | `baseline_retained` | `missing_executable_steps` |
| `plugin.openai-bundled.chrome.26.623.141536.control-chrome` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-bundled.computer-use.1.0.857.computer-use` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated-remote.github.0.1.6-2841cf9749ae.gh-address-comments` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.github.0.1.6-2841cf9749ae.gh-fix-ci` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `missing_invocation_boundary` |
| `plugin.openai-curated-remote.github.0.1.6-2841cf9749ae.github` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated-remote.github.0.1.6-2841cf9749ae.yeet` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated-remote.product-design.0.1.48.audit` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated-remote.product-design.0.1.48.design-qa` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.get-context` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.ideate` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `missing_invocation_boundary` |
| `plugin.openai-curated-remote.product-design.0.1.48.image-to-code` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.index` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated-remote.product-design.0.1.48.prototype` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.research` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.share` | `plugin_skill` | `true` | `90` | `90` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded`, `missing_invocation_boundary` |
| `plugin.openai-curated-remote.product-design.0.1.48.url-to-code` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated-remote.product-design.0.1.48.user-context` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated.github.d6169bef.gh-address-comments` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated.github.d6169bef.gh-fix-ci` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `missing_invocation_boundary` |
| `plugin.openai-curated.github.d6169bef.github` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated.github.d6169bef.yeet` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.cli` | `plugin_skill` | `true` | `90` | `90` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded`, `missing_invocation_boundary` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.community-evals` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.datasets` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `missing_invocation_boundary` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.gradio` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `missing_invocation_boundary` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.jobs` | `plugin_skill` | `true` | `87` | `92` | `5` | `pass` | `candidate` | `skill_body_too_long` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.llm-trainer` | `plugin_skill` | `true` | `87` | `92` | `5` | `pass` | `candidate` | `skill_body_too_long` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.paper-publisher` | `plugin_skill` | `true` | `87` | `92` | `5` | `pass` | `candidate` | `skill_body_too_long` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.papers` | `plugin_skill` | `true` | `90` | `90` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded`, `missing_invocation_boundary` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.trackio` | `plugin_skill` | `true` | `95` | `95` | `0` | `pass` | `baseline_retained` | `key_use_case_not_front_loaded` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.transformers.js` | `plugin_skill` | `true` | `92` | `92` | `0` | `pass` | `baseline_retained` | `skill_body_too_long` |
| `plugin.openai-curated.hugging-face.b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04.vision-trainer` | `plugin_skill` | `true` | `82` | `92` | `10` | `pass` | `candidate` | `skill_body_too_long` |
| `plugin.openai-primary-runtime.documents.26.630.12135.documents` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-primary-runtime.pdf.26.630.12135.pdf` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `plugin.openai-primary-runtime.presentations.26.630.12135.presentations` | `plugin_skill` | `true` | `92` | `92` | `0` | `pass` | `baseline_retained` | `nonportable_name`, `description_too_short` |
| `plugin.openai-primary-runtime.spreadsheets.26.630.12135.spreadsheets` | `plugin_skill` | `true` | `97` | `97` | `0` | `pass` | `baseline_retained` | `nonportable_name` |
| `plugin.openai-primary-runtime.template-creator.26.630.12135.template-creator` | `plugin_skill` | `true` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `system.imagegen` | `system_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `system.openai-docs` | `system_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `system.plugin-creator` | `system_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |
| `system.skill-creator` | `system_skill` | `false` | `95` | `100` | `5` | `pass` | `candidate` | `none` |
| `system.skill-installer` | `system_skill` | `false` | `100` | `100` | `0` | `pass` | `baseline_retained` | `none` |

## Остаточные риски и ограничения

- Правки `managed plugin-cache` применены локально в `/Users/daniildegtyarev/.codex/plugins/cache/...`; они могут быть перезаписаны при `reinstall/update`. Для долговечного результата нужен `upstream repair` или `reinstall-safe overlay`.
- `global.last30days`, `global.prompt-manager`, и часть Hugging Face plugin skills остаются ниже `100/100` из-за `skill_body_too_long`. Это требует отдельного `progressive-disclosure refactor` в `references/`, но не блокирует текущий `all-skills audit`, потому что статус уже `pass`.
- `plugin.openai-primary-runtime.presentations...` и `plugin.openai-primary-runtime.spreadsheets...` имеют малые `format findings` (`description_too_short` или `nonportable_name`), но это `managed runtime skill naming/content`, поэтому прямой долговечный repair не делался.
- `Runtime smoke` конкретных `tools/plugins` не запускался; проверялись контракты `SKILL.md`, `discovery text`, `structure` и `safety/locality wording`.

## Контрактное влияние

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Roehub runtime/API/UI/persistence | `none` | Application runtime не менялся. |
| Benchmark harness | `compatible-change` | Добавлены `audit-all-skills` и `all-skills-ab-compare`; существующие команды не ломались. |
| Local/system/global skills | `compatible-change` | Менялись только `description` во `frontmatter`; `workflow body` не переписывался. |
| Managed plugin cache | `compatible-change` локально, `unknown` долговечно | Локальный кэш улучшен и проверен, но `reinstall/update` может перезаписать файлы. |
| Secrets/redaction | `compatible-change` | `Boundary wording` стал строже; новых требований к secrets нет. |

## Проверка

Выполнено:

- `uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py` -> `11 passed`
- `uv run ruff check tools/codex_quality_benchmark tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py` -> passed
- `uv run python -m tools.codex_quality_benchmark.cli audit-all-skills --out-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v2 --run-id all-skills-20260707-v2` -> `inventory=61 rows=61`
- `uv run python -m tools.codex_quality_benchmark.cli all-skills-ab-compare --before-audit .codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v1/skill_audit.json --after-audit .codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v2/skill_audit.json --inventory .codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-audit-v2/skill_inventory.json --out-dir .codex/tmp/skill-plugin-auto-improve-benchmark-v1/all-skills-ab-v1 --target-metric audit_score_0_100 --min-metric-delta 1` -> `rows=61 accepted=7`
- `uv run python -m tools.docs.generate_docs_index --check` -> `OK`
- `git diff --check -- tools/codex_quality_benchmark/skill_audit.py tools/codex_quality_benchmark/cli.py tests/unit/tools/test_codex_skill_audit.py docs/architecture/agents/skill-plugin-all-skills-official-format-ab-test-v1.md docs/architecture/README.md` -> passed

## Манифест измененных файлов

Roehub repo:

- `tools/codex_quality_benchmark/skill_audit.py`
- `tools/codex_quality_benchmark/cli.py`
- `tests/unit/tools/test_codex_skill_audit.py`
- `docs/architecture/agents/skill-plugin-all-skills-official-format-ab-test-v1.md`
- `docs/architecture/README.md` после генерации индекса

Вне Roehub repo:

- `/Users/daniildegtyarev/.codex/skills/.system/skill-creator/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/numba-jit-performance/SKILL.md`
- `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/jobs/SKILL.md`
- `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/llm-trainer/SKILL.md`
- `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/paper-publisher/SKILL.md`
- `/Users/daniildegtyarev/.codex/plugins/cache/openai-curated/hugging-face/b1986b3d3da5bb8a04d3cb1e69af5a29bb5c2c04/skills/vision-trainer/SKILL.md`
