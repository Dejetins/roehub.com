# Stage 00 — Contract And Catalog Foundation

Статус: `accepted`.

Дата: `2026-07-09`.

## Результат

Создан доказуемый фундамент единой системы скиллов без изменения managed
plugin cache и без активации dormant plugins.

- Immutable audit baseline: `85/85` (`S001–S085`).
- Recursive current filesystem inventory: `96/96`.
- Supplemental Figma cache skills: `11/11`, все
  `cache_only/not_exposed/preserve_dormant`.
- Exact ownership: `96/96`; Stage `00–05` baseline coverage
  `2+5+13+11+32+22=85`, без пропусков и пересечений.
- Bootstrap repairs: `S065` и `S066` — `implemented/structural_pass`.
- Остальные baseline rows: `83 pending`; они принадлежат следующим stages, а
  не объявлены исправленными заранее.

## Что реализовано

1. `skill-spec/v1`, `skill-result/v1`, `skill-catalog/v1` и
   `skill-contract-case-result/v1` имеют formal JSON Schemas и общий parser.
2. Recursive discovery теперь видит hidden/nested dependency skills, включая
   Playwright paths без стандартного `skills/<name>` layout.
3. Catalog builder соединяет immutable audit, backlog и текущий filesystem,
   проверяет unaccounted drift, relations, enums и counts.
4. Canonical global catalog находится в
   `/Users/daniildegtyarev/.codex/skill-system/catalog-v1.json`; repo snapshot
   `.codex/skill-system/catalog-v1.json` имеет тот же SHA-256
   `4ab3a68c101b4d61701686dfb0063709c72fa8569a92998e36b6764a1dc16145`.
5. Resolver доказан для aliases `Presentations`/`Spreadsheets` и deprecated ID
   `S005→S020` без заявления, что каталог физически фильтрует loader.
6. `.codex/skill-system/ownership-v1.json` задаёт exact source/effective paths,
   stage, operation, before hash, discovery, exposure и activation policy.
7. Rollback-v1 хранит content-addressed before blobs для `S065/S066`, absent
   state для новых references и global catalog, и проходит restore-plan verify.
8. Deterministic fixtures покрывают budget, target, destination, visibility,
   authority, unknown provider state, secret evidence, read-only intent, dirty
   main, capability absence и alias resolution.
9. `S065 plugin-creator` теперь проверяет не только manifest, но и source,
   marketplace, install и fresh-process discovery; managed cache остаётся
   read-only.
10. `S066 skill-creator` использует один v1 schema contract, exact ownership,
    deterministic fixtures и clean-context forward testing без answer leakage.

## Evidence

| Gate | Результат |
|---|---|
| recursive `audit-all-skills` | `96` inventory rows; `94 pass`, `2 warn`, `0 fail/blocked` |
| contract fixtures | `15/15 passed` |
| standard `quick_validate.py` for S065/S066 | `2/2 valid` через repo `uv` runtime |
| v1 skill-spec validation | `2/2 valid` |
| rollback manifest | valid; `5` recovery entries; blob hashes match filenames |
| catalog/global parity | identical bytes and SHA-256 |
| scoped `ruff` | passed |
| required unit tests | `26 passed` |
| docs index | passed |
| `git diff --check` | passed |

Durable evidence:

- `.codex/skill-system/evidence/stage-00-current-audit/`
- `.codex/skill-system/evidence/stage-00-contract-fixtures.json`
- `.codex/skill-system/rollback/manifest-v1.json`
- `.codex/skill-system/ownership-v1.json`

Системный `python3` не содержит `PyYAML`, поэтому первый direct запуск
`quick_validate.py` завершился environmental `ModuleNotFoundError`. Повтор через
зафиксированный repo `uv` runtime прошёл. Это не скрыто и не классифицировано
как дефект скилла.

## Контрактное влияние

| Dimension | Classification | Обоснование |
|---|---|---|
| Roehub API/UI/data/runtime | `none` | продуктовый код и runtime не затронуты |
| skill metadata schema | `compatible-change` | v1 добавлен в разрешённый nested `metadata`; legacy skills пока остаются в catalog |
| skill result/report schema | `compatible-change` | новый versioned envelope; неизвестная major version fail closed |
| authoring behavior S065/S066 | `compatible-change` | capability сохранена, validation/ownership/reload gates усилены |
| managed plugin discovery | `none` в Stage 00 | cache не изменялся, overlay не создавался и не устанавливался |
| global routing behavior | `unknown` до Stage 06/07 | catalog создан, но AGENTS consumer и fresh-process proof ещё не подключены |
| logs/evidence/redaction | `compatible-change` | добавлены стабильные sanitized evidence fields и secret-value scan |

## File manifest

Created:

- `tools/codex_quality_benchmark/skill_contract.py`
- `tools/codex_quality_benchmark/skill_catalog.py`
- `tools/codex_quality_benchmark/skill_contract_fixtures.py`
- `tools/codex_quality_benchmark/schemas/*.schema.json` — `4` schemas
- `tests/unit/tools/test_codex_skill_contract.py`
- `tests/unit/tools/test_codex_skill_catalog.py`
- `tests/unit/tools/test_codex_skill_contract_fixtures.py`
- `.codex/skill-system/` catalog, policy, ownership, fixtures, evidence и rollback
- `/Users/daniildegtyarev/.codex/skill-system/catalog-v1.json`
- `/Users/daniildegtyarev/.codex/skills/.system/plugin-creator/references/skill-system-contract-v1.md`
- `/Users/daniildegtyarev/.codex/skills/.system/skill-creator/references/skill-system-contract-v1.md`
- этот Stage `00` report.

Modified:

- `tools/codex_quality_benchmark/skill_audit.py`
- `tests/unit/tools/test_codex_skill_audit.py`
- `/Users/daniildegtyarev/.codex/skills/.system/plugin-creator/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/.system/skill-creator/SKILL.md`
- stage ledger.

Deleted: none.

Outside expected paths: none. Global catalog and два system skill trees были
явно перечислены в Stage `00` ownership.

Foreign changes excluded: все Roehub application/UI/runtime файлы, existing
benchmark artifacts и остальные shared-main changes.

Mixed files:

- `tools/codex_quality_benchmark/skill_audit.py` и
  `tests/unit/tools/test_codex_skill_audit.py` существовали как untracked
  foreign benchmark work; Stage `00` владеет только recursive-discovery hunks и
  nested-path regression test.
- `docs/architecture/README.md` содержит foreign changes; Stage `00` только
  запустил generator и не присваивает себе чужие hunks.

## Proof boundary и handoff

Roehub product runtime/API/browser/deploy: `N/A`. Реальная граница Stage `00` —
локальный filesystem inventory, parser/resolver, штатный validator, rollback и
global/repo catalog parity.

Ledger обновлён до создания этого отчёта. Stage `01` разрешён только для
`S067,S075,S078,S081,S085` с exact ownership и rollback-before-mutation.

Residual risks:

- catalog пока не подключён к global/repo AGENTS; это Stage `06`;
- overlay и fresh-process discovery ещё не реализованы; это Stages `02–07`;
- `83` baseline rows закономерно остаются pending и не должны считаться
  исправленными до их профильных gates.
