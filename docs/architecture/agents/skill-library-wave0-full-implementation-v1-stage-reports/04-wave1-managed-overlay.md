# Stage 04 — managed overlay Wave 1

Статус: `accepted`.

Дата: `2026-07-09`.

## Итог

Закрыты все 32 managed P1 записи этапа:

- создано 29 corrected resource-контрактов;
- `S005→S020`, `S006→S021`, `S007→S022` deprecated без дублирующих файлов;
- managed plugin cache не изменялся;
- resource-only plugin не устанавливался;
- `preserve_public`, `preserve_internal` и `preserve_dormant` сохранены.

Проверки:

- official validator: `29/29`;
- `skill-spec/v1`: `29/29`;
- structural audit: `29 × 100/100`;
- deterministic fixtures: `53/53`;
- semantic/source/deprecation gates: `32/32`;
- cache source hashes: `32/32` unchanged;
- plugin validation: pass;
- resolver aliases `Presentations→S056`, `Spreadsheets→S057`: pass;
- duplicate resolution `S005→S020`: pass;
- repo/global catalog:
  `783d0a67871eccf785d4ca750d8af7674f6aa030fbfd57b86551777af27cbfbd`.

## Исправленные семейства

### Browser и визуальные adapters

- `S002 control-chrome`: named profile/tab/domain scope, connector-first,
  запрет auth bypass и raw profile/storage evidence.
- `S003 computer-use`: fresh AX-state loop, наследование platform confirmation
  policy, узкая terminal boundary и redacted UI evidence.
- `S004 visualize`: higher communication rules имеют приоритет, writable path
  обнаруживается из host contract, Mermaid/HTML выбираются по требуемой
  fidelity, CSP/responsive/accessibility проверяются явно.

### Hugging Face

- `S009` использует установленный и проверенный `hf`, live help, dry-run,
  destructive/paid gates и secret redaction вместо `curl | bash`.
- `S010` разделяет `local`, `provider` и `remote_job`, фиксирует revision,
  seed, versions, corpus и trust decision.
- `S011` разделяет read-only viewer/download и publish с destination,
  visibility и authority.
- `S015/S016` отделяют чтение papers от index/link/authorship/visibility writes
  и требуют primary-source citations, preview/diff, rollback и PII safety.
- `S017` ограничивает poll/relaunch budget, run identity и Space/webhook privacy.
- `S018` получил valid metadata, version/model/license/privacy gates,
  сопоставимый benchmark и корректный `pipe.dispose()` lifecycle.

### GitHub

- Старые cache duplicates `S005/S006/S007` устранены на уровне каталога.
- `S020` сохраняет thread-aware GraphQL, least-privilege fallback и redacted
  excerpts; local fix scope отделён от reply/resolve/publish.
- `S021` ограничивает и очищает CI logs; уже запрошенный scoped fix не требует
  повторного approval.
- `S022` читает `AGENTS.md` до маршрутизации; Roehub publish идёт в
  `publish-ci-deploy`, generic route — только при отсутствии repo override.

### Предметные artifact templates

Каждый corrected template читает retained manifest/reference из catalog
`source_path`, а policy — из resource overlay:

- `S027`: experiment design, uncertainty, power/MDE, multiplicity и causal label;
- `S028`: recalc, errors, totals, scenarios, runway и openability;
- `S029`: provenance, assumptions, sensitivity и unresolved-data flags;
- `S037`: opportunity grain, stages/probabilities, duplicates и forecast tie-out;
- `S041`: обязательный architecture-design/review, contracts, rollout,
  validation и cold review;
- `S043`: three-statement integration, roll-forwards, balance/cash tie-outs и
  openability.

### Product Design

- `S045 design-qa`: `report_only` и `fix_authorized`, caller-owned report path,
  repo asset policy вместо абсолютного запрета форматов.
- `S049 index`: явные public/internal edges, capability discovery, lazy
  user-context и один focused route.
- `S050 research`: time/stop budget, privacy, dedupe, citations, quote limits и
  evidence/inference confidence.
- `S051 share`: `disposable_preview` отделён от `repository_delivery`,
  readiness/rollback обязательны, repo orchestrator имеет приоритет.

### Office artifacts и Roehub prototype

- `S054 documents`: compact router, workspace dependency loader, конечный
  render/a11y/privacy цикл.
- `S055 pdf`: caller-owned paths, dependency authority, Unicode-safe generation,
  encrypted/active/malformed content gate.
- `S056 presentations` и `S057 spreadsheets`: lowercase canonical names с
  legacy aliases; разрешены необходимые native vector diagrams; inspect/execute,
  authority, visual/formula/openability gates согласованы.
- `S058 template-creator`: informed retention, hidden metadata/PII scan, scrub
  option и verified temp cleanup.
- `S062 backtests-live-prototype`: plugin-relative source, exact cwd, build,
  real-browser, console/network acceptance; production Roehub не затрагивается.

## Совместимость

Contract impact: `compatible-change`.

- Public canonical names сохранены; uppercase Office names стали aliases.
- Dormant/internal ресурсы не стали session-exposed.
- Resource roots используют `skill-spec/v1` и `skill-result/v1`.
- Read-only intent больше не переходит неявно в write/paid/deploy действия.
- Provider/repository/template assets продолжают жить в исходных packages;
  resource overlay не дублирует и не теряет их.

## Доказательства и проверки

```text
quick_validate.py: 29/29
skill-spec/v1: 29/29
resource audit: 29 x 100/100
contract fixtures: 53/53
semantic gates: 32/32
cache hashes: 32/32 unchanged
plugin validate: pass
uv run ruff check ...: pass
uv run pytest -q ...: 26 passed
docs index check: pass
git diff --check: pass
YAML/JSON/whitespace: pass
rollback verify: 69 entries
plugin list: codex-skill-system-overrides absent
```

Evidence:

- `.codex/skill-system/evidence/stage-04-overlay-audit/`
- `.codex/skill-system/evidence/stage-04-contract-fixtures.json`
- `.codex/skill-system/evidence/stage-04-semantic-gates.json`
- `.codex/skill-system/rollback/manifest-v1.json`

## Файлы этапа

Созданы:

- `/Users/daniildegtyarev/plugins/codex-skill-system-overrides/resources/skills/<ID>/SKILL.md`
  для `S002,S003,S004,S009,S010,S011,S015,S016,S017,S018,S020,S021,S022,S027,S028,S029,S037,S041,S043,S045,S049,S050,S051,S054,S055,S056,S057,S058,S062`;
- Stage 04 audit, fixture и semantic evidence;
- этот отчёт.

Изменены:

- repo/global `catalog-v1.json`;
- `ownership-v1.json`;
- fixture manifest;
- rollback manifest;
- stage ledger.

Удалённых файлов нет. Managed cache, plugin installation state и plugin
manifest не менялись.

## Граница доказательства

Proof boundary: corrected resource contracts, resolver и activation policy.
Roehub product runtime, browser, API, deploy и production не запускались:
этап изменяет инструкции, а не продукт.

Shared dirty `main` сохранён; чужие application/test/redesign/prototype changes
не staged, не committed и не pushed. Mixed `docs/architecture/README.md` и
benchmark surfaces изменялись только принадлежащими плану hunks.

Следующий разрешённый этап: `05-wave2-wave3-completion`.
