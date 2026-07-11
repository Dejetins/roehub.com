# Stage 01 — Wave 0 Direct Critical Repairs

Статус: `accepted`.

Дата: `2026-07-09`.

## Результат

Критические direct-source repairs завершены для
`S067,S075,S078,S081,S085`.

| ID | Реализованное усиление | Итог |
|---|---|---|
| `S067` | provenance/ref review, destination collision stop, immutable system skills, no credential collection or downloaded-code execution | `implemented/forward_pass` |
| `S075` | valid v1 frontmatter, portable root, execution/synthesis/security one-hop refs, no policy override or cookie path, paid/save boundaries | `implemented/forward_pass` |
| `S078` | quoted valid description, strict report-only mode, shared-main scope evidence, empty file manifest | `implemented/forward_pass` |
| `S081` | route-conditional prerequisites, deploy relevance, `shipped-no-runtime`, exact proof labels, bounded redacted cleanup | `implemented/forward_pass` |
| `S085` | compact stack-aware router, web/mobile/acceptance refs, opt-in persist/install, browser/device proof | `implemented/forward_pass` |

Root sizes после progressive disclosure: `103`, `118`, `85`, `167`, `118`
строк соответственно; все ниже `500`.

## Evidence

- Standard `quick_validate.py`: `5/5 valid`.
- `skill-spec/v1`: `5/5 valid`.
- Deterministic contract fixtures: `23/23 passed`; добавлены проверки existing
  installer destination, system target, report-only mutation, irrelevant
  runtime action, unknown UI stack, unrequested persistence, cookie access и
  policy override.
- Recursive current audit: `96/96` discovered; все пять Stage `01` roots имеют
  `audit_score_0_100=100`, status `pass`, findings `0`.
- Required Python gates: `26 passed`; scoped `ruff` passed.
- Rollback: `16` entries, content-addressed blobs verified.
- Global/repo catalog parity: SHA-256
  `dacb9cc8726aa6f86816f5b55dca177d8acfd820009d30efa551c79d11626262`.
- Docs index и `git diff --check`: passed.

Durable evidence:

- `.codex/skill-system/evidence/stage-01-current-audit/`
- `.codex/skill-system/evidence/stage-01-contract-fixtures.json`
- `.codex/skill-system/evidence/stage-01-semantic-gates.json`
- `.codex/skill-system/rollback/manifest-v1.json`

Live install, GitHub publish, paid research, provider mutation, browser action,
Mac Studio или production deploy не выполнялись. Их отсутствие — safety proof
для instruction audit, а не заявление о runtime acceptance этих систем.

## Контрактное влияние

| Dimension | Classification | Migration |
|---|---|---|
| Roehub application API/UI/data | `none` | product code не менялся |
| skill frontmatter/result | `compatible-change` | v1 nested metadata + common result envelope |
| `S067` overwrite behavior | `breaking-change`, safety-required | existing/system destinations теперь fail closed |
| `S075` formatting/provider behavior | `breaking-change`, safety-required | platform/tool citation rules win; cookies/secrets never collected; paid/save choices explicit |
| `S078` mutation behavior | `breaking-change`, safety-required | readiness gate больше не выполняет publish mutations |
| `S081` delivery routing | `compatible-change` | irrelevant prereqs removed; explicit no-runtime terminal added |
| `S085` default stack/persistence | `breaking-change`, safety-required | stack inferred; persistence/install require opt-in |
| external side effects | `none` in this stage | instructions changed, live actions не запускались |

## File manifest

Modified:

- `/Users/daniildegtyarev/.codex/skills/.system/skill-installer/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/last30days/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/pre-ship-gate/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/publish-ci-deploy/SKILL.md`
- `/Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/SKILL.md`
- `.codex/skill-system/catalog-v1.json`
- `.codex/skill-system/ownership-v1.json`
- `.codex/skill-system/fixtures/skill-contract-cases-v1.json`
- `.codex/skill-system/rollback/manifest-v1.json`
- `tools/codex_quality_benchmark/skill_catalog.py`
- `tools/codex_quality_benchmark/skill_contract_fixtures.py`
- stage ledger.

Created:

- `/Users/daniildegtyarev/.codex/skills/last30days/references/execution.md`
- `/Users/daniildegtyarev/.codex/skills/last30days/references/synthesis.md`
- `/Users/daniildegtyarev/.codex/skills/last30days/references/security.md`
- `/Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/references/web.md`
- `/Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/references/mobile.md`
- `/Users/daniildegtyarev/.codex/skills/ui-ux-pro-max/references/acceptance.md`
- Stage `01` evidence artifacts и этот report.

Deleted: none.

Outside expected paths: none. Все source/reference paths были перечислены в
ownership-v1 до mutation.

Foreign changes excluded: весь Roehub product/runtime scope, managed plugin
cache и unrelated shared-main files.

Mixed files: benchmark/catalog modules были частью существующего untracked
benchmark work; Stage `01` владеет только fixture/rollback/status hunks.

## Handoff

Ledger обновлён до создания отчёта. Stage `02` разрешён для точных `13` P0
managed IDs. Он создаёт только resolver resources и catalog deprecations;
managed cache mutation, plugin installation и dormant activation запрещены.
