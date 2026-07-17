# Единая система скиллов — полный план внедрения Wave 0–3 v1

План полностью реализует `85` рекомендаций classic-аудита и одновременно
контролирует текущий рекурсивный filesystem inventory, который после появления
Figma cache skills содержит `96` canonical paths. Baseline аудита неизменяем;
новые `S086–S096` классифицируются отдельно и не считаются автоматически
включёнными или публичными.

## Статус и исполняемые артефакты

- status: `completed`
- date: `2026-07-09`
- execution_mode: `goal_driven`
- intended_agent_model: `gpt-5.5`
- reasoning_effort: `xhigh`
- plan_doc: `docs/architecture/agents/skill-library-wave0-full-implementation-v1.md`
- prompt_pack_dir: `.codex/agents/generated/skill-library-wave0-full-implementation-v1/`
- stage_ledger: `docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/skill-library-wave0-full-implementation-v1-stage-ledger.md`
- audit_source: `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/02-consolidated-improvement-backlog.md`
- audit_inventory: `docs/architecture/agents/skill-library-classic-audit-v1-stage-reports/00-inventory-and-batch-plan.md`
- `GOAL.md`: не создаётся; durable state — план, prompt pack и ledger.

Завершено `2026-07-10`: stages `00–08` приняты, immutable audit baseline
закрыт `85/85`, текущий loader-candidate inventory классифицирован `96/96`,
обязательных `pending/blocked` строк нет. Итоговая сверка и границы
доказательства находятся в
`.codex/skill-system/evidence/stage-08-final-reconciliation.json` и отчёте
`docs/architecture/agents/skill-library-wave0-full-implementation-v1-stage-reports/08-closure.md`.

## Цель

1. Закрыть каждую строку `S001–S085` статусом `implemented`,
   `canonicalized`, `deprecated` или `accepted_no_change` с проверяемым evidence.
2. Обнаруживать current inventory рекурсивно, включая hidden dependency paths,
   и классифицировать все `96/96` записей без смешения с audit baseline.
3. Реализовать `skill-spec/v1`, `skill-result/v1`, parser, resolver, ownership,
   rollback и deterministic contract fixtures.
4. Исправлять user/system sources напрямую, а managed plugin improvements
   поставлять долговечным personal plugin overlay без прямого cache patch.
5. Не активировать dormant/cache-only skills, не расширять публичный список
   и не менять side effects без явного activation policy.
6. Доказать применение каталога через глобальный runtime policy, resolver и
   отдельный fresh Codex process после plugin reload boundary.

## Наблюдаемое состояние

- Immutable audit baseline: `85` IDs (`18 user`, `5 system`, `62 plugin`).
- Baseline priorities: `18 P0`, `45 P1`, `12 P2`, `10 P3`.
- Baseline validator failures: `S018,S056,S057,S075,S078`.
- Baseline long roots: `S013,S014,S015,S018,S019,S075,S080,S085`.
- Baseline duplicates: `S005/S020`, `S006/S021`, `S007/S022`, `S059/S060`.
- Current recursive inventory: `96` canonical paths; additions `S086–S096`,
  removals `0`.
- Existing `tools/codex_quality_benchmark` is the only harness and must be
  extended. Its discovery must change from one-level globbing to recursive,
  hidden-path-aware discovery.
- `figma` and `hugging-face` are not installed; their cache-only entries remain
  dormant. Existing public plugin surface contains `17` baseline plugin IDs.
- Personal marketplace is `/Users/daniildegtyarev/.agents/plugins/marketplace.json`.

## Supplemental current inventory

| ID | name | lines | sha256 | activation |
|---|---|---:|---|---|
| `S086` | `figma-code-connect` | 527 | `206d6db8304704019ee64e23128428af8f8815e2a114a99087ec2a300c194e9f` | `preserve_dormant` |
| `S087` | `figma-create-new-file` | 80 | `82e0a018692d3d009b5d2c83ecce9203a4ce7dfbdb98b7b13a4f7e5b11832bc2` | `preserve_dormant` |
| `S088` | `figma-generate-design` | 491 | `2915993a955d4684ea3ae3ce98af4e673134870f2b412e661d9114387980fa42` | `preserve_dormant` |
| `S089` | `figma-generate-diagram` | 112 | `7297638dbf2130d1b37bfd85df65d290a4dd0b11a11b54abafcf7d07565922b5` | `preserve_dormant` |
| `S090` | `figma-generate-library` | 370 | `38d9381d4fb089233eec0a9af829a6444fbe7c8413cfcfbb89d1992a5f08b7fc` | `preserve_dormant` |
| `S091` | `figma-implement-motion` | 145 | `115e37a86d0528cc7674b9f94efa9e87f7d61d2a8b4569d3cab1c8459efd84a6` | `preserve_dormant` |
| `S092` | `figma-swiftui` | 36 | `868c4defe2c0854f5a3e27594d3fa0f20347c86a9b15d103483b7488bfa8affe` | `preserve_dormant` |
| `S093` | `figma-use-figjam` | 64 | `16b15da304f777e6bdc79bf425da73fa1ce0ea41af726cab8ddd6f40180acd74` | `preserve_dormant` |
| `S094` | `figma-use-motion` | 80 | `d1aeb004f2fccd74b368a5ce6d30775ef3d9c17a28a4d68ff8a36a47b71ac2ec` | `preserve_dormant` |
| `S095` | `figma-use-slides` | 217 | `f317b2f08ba1015f7255eca631ebddfe209c70e3046967f4ac6ce6102ad56f79` | `preserve_dormant` |
| `S096` | `figma-use` | 439 | `6845635c0c79717feae41f1cd372cafbe9596a982d2aade18f2a7dac4c127a96` | `preserve_dormant` |

Все пути `S086–S096` находятся под
`/Users/daniildegtyarev/.codex/plugins/cache/openai-curated-remote/figma/2.0.14/skills/`.

## Охват и не-цели

В охват входят direct user/system skill sources, personal overlay source,
global/repo catalog policy, exact ownership manifest, rollback snapshot store,
schemas, resolver, fixtures, tests, AGENTS routing, reports и docs continuity.

Не входят Roehub application/runtime/deploy, retired-host mutation, Git
commit/push, branch/worktree/stash, managed cache mutation, secrets/cookies/raw
browser state/provider payloads и автоматическая установка dormant plugins.

## Архитектурные решения

### D1. Два инвентаря

`audit_baseline` всегда содержит ровно `S001–S085`. `filesystem_inventory`
строится рекурсивно с hidden paths и содержит все текущие canonical paths.
Добавления получают supplemental IDs и policy decision; они не влияют на
закрытие baseline, но текущий inventory обязан сходиться `96/96`.

### D2. Реальный потребитель каталога

Canonical runtime catalog:
`/Users/daniildegtyarev/.codex/skill-system/catalog-v1.json`.
Repo snapshot: `.codex/skill-system/catalog-v1.json`; их SHA-256 должен
совпадать. `tools/codex_quality_benchmark/skill_catalog.py resolve-skill`
реализует deterministic resolution. Глобальный `AGENTS.md` требует resolver
для aliases/conflicts/duplicate names. Overlay может хранить read-only copy,
но не является authority. Каталог не объявляется loader-фильтром: он управляет
выбором, а fresh-process proof проверяет фактическую discovery surface.

### D3. Состояние discovery и activation

Каждая запись имеет `discovery_state`, `installed_enabled`, `session_exposed`,
`activation_policy`. Closed values:

- `discovery_state`: `direct`, `plugin_public`, `plugin_internal`,
  `cache_only`, `dependency_duplicate`, `missing`;
- `installed_enabled`: `true|false|unknown`;
- `session_exposed`: `public|internal|not_exposed|unknown`;
- `activation_policy`: `preserve_public`, `preserve_internal`,
  `preserve_dormant`, `deprecate`, `activate_overlay`.

Only these baseline plugin IDs may remain public logical skills whose effective
copy is an overlay resource:
`S001,S004,S020,S021,S022,S023,S044,S047,S048,S049,S052,S054,S055,S056,S057,S058,S062`.
Internal Product Design helpers `S045,S046,S050,S051,S053` remain internal.
HF `S009–S019`, templates `S024–S043`, disabled `S002/S003` and supplemental
Figma stay dormant. All managed corrected copies live under plugin
`resources/skills/`, never top-level `skills/`; therefore installing the
resource bundle cannot add a second session-exposed skill name. The global
resolver loads the selected resource contract explicitly.

### D4. Canonicalization

- `S005→S020`, `S006→S021`, `S007→S022`, `S008→S023`;
- `S059→S077`, `S060→S077`, `S061→S077` with trace behavior incorporated
  into the canonical direct Playwright contract;
- `Presentations→presentations`, `Spreadsheets→spreadsheets` retain aliases.

### D5. Bootstrap authoring skills

`S065` and `S066` are repaired in Stage `00` after exact snapshots and before
any bulk skill authoring. They are removed from Stage `03`.

### D6. Exact ownership and recovery

Stage `00` generates `.codex/skill-system/ownership-v1.json`. Every ID records
exact source, effective path, stage, operation, before hash and allowed
secondary touches. Later stages fail closed on mismatch and may not use broad
directory ownership.

Before each direct existing-file edit, content is stored in
`.codex/skill-system/rollback/blobs/<sha256>.md`; manifest:
`.codex/skill-system/rollback/manifest-v1.json`. New files record
`before_state=absent`. A verifier checks content addresses and restore plans;
snapshots are scanned for forbidden secret patterns before persistence.

### D7. Deterministic forward fixtures

Fixture manifest:
`.codex/skill-system/fixtures/skill-contract-cases-v1.json`.
Runner: `tools/codex_quality_benchmark/skill_contract_fixtures.py`.
Schema:
`tools/codex_quality_benchmark/schemas/skill-contract-case-result-v1.schema.json`.
Required negative/positive cases cover budget, target, destination, visibility,
authority, unknown provider state, secret evidence, read-only intent, dirty
main, capability absence and alias resolution with expected
`blocked|completed`. Durable results are sanitized JSON under
`.codex/skill-system/evidence/`.

### D8. Result-contract traceability

Every record has `result_contract_provider`:
`skill_body|overlay_adapter|orchestrator|artifact_adapter|not_applicable` and
`result_contract_evidence`. Stage `07` validates representative emitted
`skill-result/v1` envelopes, not just the schema.

### D9. Reload boundary

After overlay reinstall, Stage `06` records plugin name, marketplace, version,
loaded paths and enters `fresh_process_required`. Stage `07` may proceed only
with a separate sanitized
`codex exec --ephemeral -s read-only -C /Users/daniildegtyarev/Projects/roehub.com`
discovery/routing proof. Pickup by the current desktop task is not claimed.

## Machine-readable contracts

Required catalog fields:

- identity/provenance: `skill_id`, `logical_name`, `canonical_name`,
  `source_kind`, `canonical_path`, `source_sha256`, `baseline_membership`;
- effective state: `effective_path`, `effective_sha256`,
  `implementation_channel`, `discovery_state`, `installed_enabled`,
  `session_exposed`, `activation_policy`;
- behavior: `role`, `visibility`, `owner`, `mutability`,
  `side_effect_class`, `primary_output`, `aliases`, `companions`, `conflicts`,
  `supersedes`, `result_contract_provider`, `result_contract_evidence`;
- execution: `stage`, `recommended_action`, `implementation_status`,
  `verification_status`, `evidence_refs`.

`implementation_channel`:
`direct|overlay|dormant_overlay|deprecated|no_change|supplemental`.
`implementation_status`:
`pending|implemented|canonicalized|deprecated|accepted_no_change|classified|blocked`.
`verification_status`:
`pending|structural_pass|family_pass|forward_pass|inventory_only|blocked`.

## Stage ownership

| Stage | Exact IDs | Основной результат |
|---|---|---|
| `00` | all records + bootstrap `S065,S066` | recursive inventory, schemas, resolver, catalog, ownership, rollback, fixtures |
| `01` | `S067,S075,S078,S081,S085` | direct P0 repairs |
| `02` | `S008,S013,S014,S019,S023,S030,S047,S048,S052,S053,S059,S060,S061` | P0 public/internal/dormant overlay or deprecation without dormant activation |
| `03` | `S063,S064,S068,S069,S072,S076,S077,S079,S080,S082,S083` | direct P1 repairs |
| `04` | remaining plugin `P1` IDs from ownership manifest | public/internal/dormant P1 overlay |
| `05` | `S070,S071,S073,S074,S084` plus remaining `P2/P3` rows | domain acceptance and uniform polish |
| `06` | effective entries only | global/repo policy, plugin install, catalog parity, reload receipt |
| `07` | baseline `85` + current inventory `96` | full validators, fixtures, fresh-process proof, reconciliation |
| `08` | all | ledger/backlog/docs closure |

The exact per-stage path list is generated once in ownership-v1 and is a hard
precondition for Stages `01–08`. Prompt manifests name their fixed direct paths
and the bounded overlay subtree; the ownership record is authoritative for
per-ID secondary files.

## Порядок выполнения каждого stage

Единый порядок, без противоречия между report и ledger:

1. проверить previous gate, hashes и ownership;
2. создать/проверить rollback snapshots;
3. выполнить scoped changes;
4. провести validation и real-boundary evidence;
5. обновить catalog и durable evidence;
6. обновить ledger;
7. сформировать stage report из принятого ledger/evidence;
8. вернуть user response и продолжить только при `next_allowed=true`.

## Валидация

```bash
uv run ruff check tools/codex_quality_benchmark tests/unit/tools
uv run pytest -q tests/unit/tools/test_codex_quality_benchmark.py tests/unit/tools/test_codex_skill_audit.py tests/unit/tools/test_codex_skill_contract.py tests/unit/tools/test_codex_skill_catalog.py tests/unit/tools/test_codex_skill_contract_fixtures.py
uv run python -m tools.docs.generate_docs_index --check
git diff --check
```

Real boundaries: recursive inventory, `quick_validate.py` for every effective
skill, catalog resolver, rollback verification, deterministic fixtures,
`validate_plugin.py`, `codex plugin list`, global/repo hash parity and sanitized
fresh-process discovery/routing proof. Live paid/external/production mutation
is forbidden.

## Контрактное влияние

| Dimension | Classification | Migration |
|---|---|---|
| Roehub runtime/API/UI/data | `none` | не затрагивается |
| Skill schema/result consumers | `compatible-change` for v1, `unknown` for external consumers | versioned schema and fail on unknown major |
| Canonical names | `breaking-change` without aliases | aliases and resolver migration window |
| External/paid autonomy | `breaking-change`, safety-required | explicit mode/target/budget/authority |
| Plugin public surface | `compatible-change` only if activation matrix preserved | no dormant activation; fresh-process proof |
| Global routing policy | `compatible-change` with resolver fallback | global catalog + repo snapshot parity |

## Критерии полного завершения

- Stages `00–08` accepted; ledger `completed`.
- Audit baseline `85/85` terminal and current inventory `96/96` classified with
  `0` unexplained additions/removals.
- Exact ownership `85/85` baseline + `11/11` supplemental, no broad ownership.
- Rollback manifest verifies every direct/outside-repo mutation.
- Zero invalid effective entries, public duplicates, dangling edges or
  unresolved aliases.
- Five baseline-invalid logical skills have valid effective implementations;
  eight long-root decisions and four duplicate groups verified.
- Dormant HF/templates/Figma and disabled browser adapters remain unexposed.
- Every record has result-contract provider/evidence decision.
- Required fixtures and representative emitted result envelopes pass.
- Overlay validates, installs, appears in `codex plugin list`; fresh process
  proves expected discovery/routing without claiming current-session reload.
- Global catalog and repo snapshot hashes match; global and repo AGENTS are
  current.
- All `P0/P1` evidence and `P2/P3` semantic acceptance rows are durable.
- Docs index, lint, tests and `git diff --check` pass.
- No required row is pending, silently deferred or marked complete only from
  absence of errors.

## Откат и риски

Direct recovery uses only verified content-addressed snapshots and the exact
rollback manifest; overlay recovery reinstalls the preceding cachebuster or
removes only `codex-skill-system-overrides`. Managed cache is never reset.

Residual upstream risk remains: provider updates can change cache paths or
public exposure, desktop and CLI discovery can differ, and a later plugin
upgrade can require catalog reconciliation. These risks are monitored through
recursive inventory drift, catalog parity and fresh-process fixtures rather
than hidden by optimistic status.
