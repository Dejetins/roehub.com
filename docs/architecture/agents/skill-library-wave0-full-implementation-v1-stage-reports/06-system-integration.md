# Stage 06 — системная интеграция

Статус: `accepted`.

Дата: `2026-07-09`.

Gate: `fresh_process_required`.

## Итог

Единая система подключена к реальным global/repo точкам маршрутизации:

- canonical authority:
  `/Users/daniildegtyarev/.codex/skill-system/catalog-v1.json`;
- repo snapshot имеет тот же SHA-256:
  `1767adcd1beffa982217ab9111e602849988f00d83c02ec3c95d4db269fea2f3`;
- global и repo `AGENTS.md` требуют resolver для aliases, deprecated names,
  duplicates и conflicts и последующее чтение `effective_path`;
- каталог явно зафиксирован как selection policy, а не loader filter;
- `78/78` доступных effective v1-контрактов являются источником
  `role/visibility/owner/mutability/side-effect/primary-output`;
- relations нормализованы в canonical ID и проверены `96/96`.

## Plugin install

Выполнен cachebuster/reinstall workflow `plugin-creator`:

- marketplace: `personal`;
- plugin: `codex-skill-system-overrides@personal`;
- version: `0.1.0+codex.20260709224522`;
- source:
  `/Users/daniildegtyarev/plugins/codex-skill-system-overrides`;
- installed cache:
  `/Users/daniildegtyarev/.codex/plugins/cache/personal/codex-skill-system-overrides/0.1.0+codex.20260709224522`;
- status: `installed, enabled`;
- source и installed manifest SHA-256 совпадают:
  `aa407c1c48cf2418700e228781d713cce0985c4b38c43e26ec814803189cc22e`.

Plugin manifest не объявляет `skills`. Установленный bundle содержит `55`
resolver resources под `resources/skills/` и `0` top-level skill files.
Ожидаемая разница публичной discovery surface равна `0`.

После установки рекурсивный audit сначала увидел бы resource-контракты как
ложные loader candidates. Исправлен общий discovery helper: undeclared
`resources/skills` resource-only plugin не считаются exposed skills, но hidden
dependency paths продолжают обнаруживаться. Фактический loader-candidate
inventory после установки остаётся `96`.

## Global/repo policy

Добавлена единая политика:

1. разрешить неоднозначное имя через `resolve-skill`;
2. проверить activation/discovery state;
3. читать выбранный `effective_path`;
4. использовать `canonical_path` только как provenance;
5. сохранять platform → applicable AGENTS → explicit user → resolved skill
   precedence;
6. не считать dormant/cache-only записи доступными только из-за присутствия в
   каталоге.

## Доказательства

```text
plugin source validation: pass
marketplace entry: unique, exact local source
codex plugin add: pass
codex plugin list: installed, enabled
source/installed tree diff: empty
declared plugin skills: 0
resolver resources: 55
expected public exposure delta: 0
effective metadata parity: 78/78
relation integrity: 96/96
loader-candidate inventory after install: 96
resolver aliases/deprecations: pass
catalog repo/global parity: pass
ruff: pass
pytest: 30 passed
rollback: 101 entries, valid
docs index: pass
git diff --check: pass
```

Durable receipt:

- `.codex/skill-system/evidence/stage-06-plugin-reload-receipt.json`.

## Совместимость

Contract impact: `compatible-change`.

- Roehub runtime/API/UI/data: `none`.
- Public skill names: `compatible-change` через aliases/resolver.
- Effective behavior metadata: `compatible-change`; исторические audit labels
  заменены данными выбранного v1-контракта.
- Plugin public surface: ожидаемый delta `0`, потому что plugin resource-only.
- Loader behavior: каталог его не фильтрует; discovery helper исключает только
  undeclared resource contracts, а не обычные или hidden dependency skills.
- Fresh current-task pickup: `unknown` до отдельного процесса; это не скрыто и
  вынесено в обязательный Stage `07` gate.

## Файлы этапа

Изменены ожидаемые integration paths:

- `/Users/daniildegtyarev/.codex/AGENTS.md`;
- `.codex/AGENTS.md`;
- global/repo `catalog-v1.json`;
- `.codex/skill-system/policy-v1.json`;
- plugin manifest cachebuster;
- `/Users/daniildegtyarev/.agents/plugins/marketplace.json`;
- stage ledger и этот отчёт.

Необходимые secondary integration paths вне исходного краткого manifest:

- `tools/codex_quality_benchmark/skill_catalog.py`;
- `tools/codex_quality_benchmark/skill_discovery.py`;
- `tools/codex_quality_benchmark/skill_audit.py`;
- `tools/codex_quality_benchmark/schemas/skill-catalog-v1.schema.json`;
- `tests/unit/tools/test_codex_skill_catalog.py`;
- `tests/unit/tools/test_codex_skill_discovery.py`.

Причина: без effective-metadata projection и исключения undeclared resource
contracts установленный plugin делал бы каталог семантически устаревшим, а
filesystem inventory ложным. Все эти пути добавлены в rollback manifest до
финальной интеграционной мутации.

CLI также изменил только собственное installed plugin state. Managed cache не
редактировался вручную.

## Граница доказательства

Текущий desktop task не объявляется перезагруженным. Stage `06` доказывает
валидный source, marketplace/install state, catalog routing и отсутствие
manifest-declared skills. Полное discovery/routing доказательство должно быть
получено отдельным sanitized fresh Codex process на Stage `07`.

Roehub product runtime, browser, API, deploy и production не затрагивались.

Следующий разрешённый этап: `07-full-validation`.
