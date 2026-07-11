# Stage 02 — Wave 0 Managed Overlay

Статус: `accepted`.

Дата: `2026-07-09`.

## Результат

Критические managed-plugin рекомендации реализованы долговечным source bundle
без прямого cache patch и без изменения текущей discovery surface.

- Deprecated/canonicalized: `S008→S023`, `S059→S077`, `S060→S077`,
  `S061→S077`.
- Dormant corrected resources: `S013,S014,S019,S030`.
- Public logical corrected resources: `S023,S047,S048,S052`.
- Internal corrected resource: `S053`.
- Plugin source:
  `/Users/daniildegtyarev/plugins/codex-skill-system-overrides`.
- Top-level `skills/`: absent. Все corrections лежат в
  `resources/skills/<ID>/SKILL.md` и не добавляют duplicate exposed names.
- Plugin installation: не выполнялась; это Stage `06`.

## Реализованные P0 boundaries

| Family | Усиление |
|---|---|
| HF Jobs/training | exact target/revisions, hardware, timeout, persistence, visibility, submit authority, cost cap, small smoke, secret references only, unknown-state reconcile |
| generic GitHub publish | repository policy first, Roehub route, exact staging, no automatic branch/dependency install |
| legal memorandum | jurisdiction, as-of date, current primary authority, citations, facts/inference split, counsel-review boundary |
| Product Design ideation | selection contract before generation, durable result IDs, configurable count, private-reference consent |
| image-to-code | exact selected target, licensed assets, opt-in dependencies/artifacts/deploy, real browser gate |
| URL clone | ownership/permission, access boundary, route/state manifest, capture stop budget, asset provenance |
| saved design context | design-token terminology, namespace, fresh consent, PII/secret scan, retention and delete path |

## Evidence

- Plugin validator: passed through repository `uv` runtime.
- Standard + v1 validation: `9/9` resources valid.
- Structural audit: `9/9 pass`, score `100` each.
- Contract fixtures: `30/30 passed`.
- Rollback-v1: `26` entries verified, including absent-before plugin/resource
  paths.
- Catalog/global parity SHA-256:
  `33f4d4be08e92214669b487451164643e3c444c4d7189bdc54acd6ef2c30a6fd`.
- Required Python gates: `26 passed`; `ruff`, docs index and
  `git diff --check` passed.
- `codex plugin list`: `codex-skill-system-overrides` absent; `figma` and
  `hugging-face` remain `not installed`.
- Managed cache writes: `0`.

Первый direct запуск plugin validator через system `python3` снова показал
environmental отсутствие `PyYAML`; повтор через `uv` passed. Это не дефект
plugin source.

Durable evidence:

- `.codex/skill-system/evidence/stage-02-overlay-audit/`
- `.codex/skill-system/evidence/stage-02-contract-fixtures.json`
- `.codex/skill-system/evidence/stage-02-semantic-gates.json`
- `.codex/skill-system/rollback/manifest-v1.json`

## Контрактное влияние

| Dimension | Classification | Обоснование |
|---|---|---|
| Roehub runtime/API/UI/data | `none` | product/runtime не менялись |
| managed cache | `none` | read-only source evidence, zero writes |
| effective skill contracts | `compatible-change` for capability; `breaking-change` for unsafe defaults | resolver resources сохраняют задачи и fail closed на cost/auth/privacy/ownership gaps |
| plugin public surface | `none` в Stage 02 | bundle не установлен и top-level `skills/` отсутствует |
| dormant plugins | `none` | HF/templates/Figma не активированы |
| canonical duplicate IDs | `breaking-change` with migration | deprecated IDs разрешаются на S023/S077 через catalog |
| external/paid behavior | `breaking-change`, safety-required | material choices и unknown-state reconciliation обязательны |

## File manifest

Created outside repo:

- `/Users/daniildegtyarev/plugins/codex-skill-system-overrides/.codex-plugin/plugin.json`
- девять exact files under
  `/Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/`.

Created/modified in repo:

- Stage `02` audit, fixture и semantic evidence;
- `.codex/skill-system/catalog-v1.json`;
- `.codex/skill-system/ownership-v1.json`;
- `.codex/skill-system/fixtures/skill-contract-cases-v1.json`;
- `.codex/skill-system/rollback/manifest-v1.json`;
- `tools/codex_quality_benchmark/skill_contract_fixtures.py`;
- stage ledger и этот report.

Deleted: none.

Outside expected paths: none. Marketplace, Codex config/cache и managed plugin
cache не изменялись.

Foreign changes excluded: все Roehub product/runtime files и unrelated
shared-main work.

Mixed files: benchmark fixture runner и catalog являются частью общей untracked
benchmark системы; Stage `02` владеет только P0 fixture/status hunks.

## Handoff

Ledger обновлён до отчёта. Stage `03` разрешён только для direct IDs
`S063,S064,S068,S069,S072,S076,S077,S079,S080,S082,S083` с
rollback-before-mutation. Plugin install и AGENTS integration по-прежнему
отложены до Stage `06`.
