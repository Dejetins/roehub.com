# Stage 07 — полная проверка

Статус: `accepted`.

Дата: `2026-07-09`.

## Итог

Полная система прошла structural, semantic, contract и real-boundary runtime
проверки:

- audit baseline: `85/85` terminal, `0` pending, `0` blocked;
- loader-candidate inventory: `96/96`, unexplained drift `0`;
- implemented effective contracts: `78/78`;
- deprecated: `7/7` с canonical resolution;
- supplemental: `11/11` classified как
  `inventory_only/preserve_dormant`;
- installed resource-only plugin не добавил публичных skill names.

## Real-boundary runtime evidence

Validation не является tests-only. Отдельный свежий процесс:

```bash
codex exec --ephemeral -s read-only +  -C /Users/daniildegtyarev/Projects/roehub.com
```

проверил фактическую Codex plugin/catalog/loader boundary:

- plugin `codex-skill-system-overrides@personal` установлен и enabled;
- source/installed manifests и все `55` resolver resources совпадают;
- manifest объявляет `0` skills, top-level skill files: `0`;
- repo/global catalogs побайтно совпадают и schema-valid;
- resolver прошёл public/internal/dormant/alias/deprecated cases;
- отсутствующая capability завершилась fail-closed с exit code `1`;
- loader candidates: `96`, supplemental: `11`;
- intersection installed `resources/skills` с loader candidates: `0`;
- fresh process увидел точный список из `31` skills;
- Figma/Hugging Face/artifact-template/override names: `0`;
- expected/observed public exposure delta: `0/0`;
- файловых, plugin, network, browser, account, paid или production mutations
  не было;
- секреты, cookies, headers, env и provider payloads не читались и не
  сохранялись.

Durable proof:

- `.codex/skill-system/evidence/stage-07-fresh-process.json`.

Roehub product runtime/API/browser/deploy proof: `N/A`, потому что изменена
локальная Codex skill-discovery система, а не Roehub application/runtime.

## Effective contracts

Для всех `78` implemented entries:

- standard `quick_validate.py`: `78/78`;
- `skill-spec/v1`: `78/78`;
- deterministic structural audit: `78 × 100/100`;
- effective SHA-256 совпадает с catalog: `78/78`;
- `role`, `visibility`, `owner`, `mutability`, `side-effect-class` и
  `primary-output` совпадают с effective frontmatter: `78/78`.

Пять исходно invalid logical skills
`S018,S056,S057,S075,S078` имеют valid effective implementation `5/5`.
Восемь исходно длинных roots
`S013,S014,S015,S018,S019,S075,S080,S085` теперь имеют compact root не длиннее
`500` строк и/или точные one-hop references.

## Catalog, relations и ownership

- catalog schema: pass;
- repo/global SHA-256:
  `1767adcd1beffa982217ab9111e602849988f00d83c02ec3c95d4db269fea2f3`;
- behavior metadata source: effective v1 frontmatter;
- ownership rows: `96`, broad ownership: `false`;
- dangling edges: `0`;
- public duplicate targets: `0`;
- unresolved aliases: `0`;
- duplicate/deprecated families resolve:
  `S005→S020`, `S006→S021`, `S007→S022`, `S008→S023`,
  `S059/S060/S061→S077`;
- Office aliases resolve:
  `Presentations→S056`, `Spreadsheets→S057`.

## Source integrity и recovery

- managed baseline source hashes: `62/62 unchanged`;
- supplemental source hashes: `11/11 unchanged`;
- rollback manifest: `102` valid content-addressed entries;
- implemented effective rollback coverage: `78/78`;
- integration-path rollback coverage: `10/10`;
- managed cache ручным редактированием не затрагивался.

## Result contracts и fixtures

Provider decisions для всех `96` catalog rows:

- `skill_body`: `20`;
- `overlay_adapter`: `31`;
- `orchestrator`: `7`;
- `artifact_adapter`: `20`;
- `not_applicable`: `18` (deprecated + supplemental inventory-only).

Missing provider/evidence: `0`. Representative emitted
`skill-result/v1` envelopes для четырёх активных provider classes прошли
schema validation `4/4`.

Deterministic contract fixtures: `61/61`.

## Supplemental diagnostic

Диагностический запуск standard validator на `S086–S096` дал `1/11 pass` и
`10/11 fail`. Это upstream Figma managed-cache sources, добавленные после
immutable audit baseline:

`S086,S087,S088,S090,S091,S092,S093,S094,S095,S096`.

Они не являются effective implementations, не установлены в свежем CLI
process, имеют `inventory_only/preserve_dormant`, их hashes сохранены, а
managed cache не редактировался. Поэтому это не скрытый failure acceptance, а
явная terminal classification по плану. Их будущая активация требует отдельного
installed-plugin audit.

## Quality gates

```text
ruff: pass
pyright: 0 errors, 0 warnings
pytest: 30 passed
live fixtures: 61/61
prompt YAML: 9/9
skill-system JSON: pass
plugin validation: pass
source/installed plugin diff: empty
rollback verify: pass
docs index: pass
git diff --check: pass
```

`pyright` первоначально нашёл два type annotation defects в foundation
`skill_contract.py`. После узкого исправления `Any/Iterable` gate повторён и
завершился `0 errors`; focused regression tests: `15 passed`.

## Контрактное влияние

Overall: `compatible-change`.

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Roehub runtime/API/UI/data | `none` | продукт не затронут |
| skill spec/result consumers | `compatible-change` | versioned v1 schemas |
| canonical names | `compatible-change` | aliases и deprecated resolver сохранены |
| external/paid autonomy | `breaking-change` | safety-required explicit authority/target/budget |
| plugin public surface | `compatible-change` | fresh-process delta `0` |
| global routing policy | `compatible-change` | deterministic resolver + effective path |
| persisted schema | `none` | не затронута |
| request/cache identity | `none` | не затронута |

## Диагностика вне scope

Fresh CLI сообщил, что три существующих repo agent role TOML
`architect.toml`, `promt_manager.toml`, `software_engineer.toml` содержат
неподдерживаемое поле `reasoning_language` и игнорируются. Это не skill files,
не входит в ownership-v1 и не повлияло на проверяемый skill surface, поэтому
Stage 07 их не менял. Finding сохранён в validation evidence как
`unrelated_pre_existing`.

## Доказательства

- `.codex/skill-system/evidence/stage-07-validation.json`;
- `.codex/skill-system/evidence/stage-07-fresh-process.json`;
- `.codex/skill-system/evidence/stage-06-plugin-reload-receipt.json`;
- `.codex/skill-system/rollback/manifest-v1.json`.

## Граница доказательства

Доказано локальное Codex runtime discovery/routing после reinstall. Не
утверждается, что каталог фильтрует loader, и не утверждается pickup текущим
desktop task. Desktop/CLI surfaces могут различаться; после plugin/cache/loader
updates fresh-process proof нужно повторять.

Shared dirty `main` сохранён; staging, commit, push, deploy и production
mutation не выполнялись.

Следующий разрешённый этап: `08-closure`.
