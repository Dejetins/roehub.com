# Stage 03 — прямые исправления Wave 1

Статус: `accepted`.

Дата: `2026-07-09`.

## Итог

Исправлены все 11 прямых P1-скиллов этапа:
`S063,S064,S068,S069,S072,S076,S077,S079,S080,S082,S083`.

- Все корневые `SKILL.md` переведены на `skill-spec/v1` и сокращены до
  `86–121` строк.
- Официальный validator: `11/11` valid.
- Формальная JSON Schema: `11/11` pass.
- Structural audit: `11/11` со score `100/100`.
- Contract fixtures: `38/38`.
- Семантические gates: `11/11`.
- Recursive inventory: `96`, из них `94 pass`, `2 warn`, `0 fail`,
  `0 blocked`. Два warn относятся к другим, ещё не обработанным stages.
- Rollback manifest: `40` записей, проверен.
- Repo/global catalog идентичны:
  `09b20f965890c4556da81ca7f6b51a3071e976e352fbc29ae6418909200e2c7f`.

## Реализованные контракты

| ID | Результат |
|---|---|
| `S063 imagegen` | Разделены `built_in` и `cli_fallback`; действующий tool contract имеет приоритет; production-logo остаётся в существующей vector/design system; терминальный tool output не требует конфликтующего пост-ответа. |
| `S064 openai-docs` | Добавлено обнаружение доступных capabilities, официальный fallback и `bounded uncertainty`; docs-only режим запрещает неявную установку MCP, SDK или изменение global/project config. |
| `S068 architecture-design` | Portable core отделён от `references/roehub-profile.md`; один разрешённый read-only reviewer является terminal role. |
| `S069 architecture-review` | Закреплены `fact/inference/proposal/unknown`, единая шкала findings и общий `skill-result/v1`; designated reviewer никогда не делегирует review. |
| `S072 browser-qa-evidence` | Результат называется `browser_qa_readiness`, содержит `proof_boundary` и не выдаётся за общий release/ship verdict. |
| `S076 numba` | Добавлены runtime/version lock, deterministic corpus, сопоставимый benchmark contract и владение JIT/cache artifacts. |
| `S077 playwright` | Capture отключён во время auth; raw storage/browser state запрещён; evidence destination принадлежит вызывающему workflow. |
| `S079 production-risk-review` | Обязательны applicable `AGENTS.md`, точные base/merge-base/head, confidence/evidence и семимерная contract matrix. |
| `S080 prompt-manager` | 503-строчный root заменён portable core; общий контракт вынесен в `references/prompt-contract.md`, Roehub policy routing — в `references/roehub-profile.md`; новые docs создаются по impact, а не автоматически. |
| `S082 root-cause-debugging` | Разведены `diagnose_only` и `fix_authorized`; diagnosis не изменяет файлы; evidence проходит обязательную sanitization. |
| `S083 staged-plan-runner` | `inspect_status` строго read-only; перед execution обязателен однозначный stage schema; fallback по имени файла, порядку папки или chat memory запрещён. |

## Совместимость

Contract impact: `compatible-change`.

- Canonical names, positive triggers и основные задачи сохранены.
- Metadata и result envelope унифицированы.
- Уточнены режимы, полномочия, side effects, blockers и evidence boundaries.
- Поведение, которое ранее могло неявно изменять config, код или ledger,
  теперь корректно возвращает `blocked` без явного authority.
- Roehub-специфичная политика не копируется в portable roots и читается после
  текущего `.codex/AGENTS.md`.

## Проверки

Пройдены:

```text
official quick_validate.py: 11/11
skill-spec/v1: 11/11
structural audit: 11 x 100/100
contract fixtures: 38/38
semantic gates: 11/11
uv run ruff check ...: pass
uv run pytest -q ...: 26 passed
python -m tools.docs.generate_docs_index --check: pass
git diff --check: pass
YAML/JSON parse and external-file whitespace scan: pass
rollback verify: pass
catalog parity: pass
```

Evidence:

- `.codex/skill-system/evidence/stage-03-current-audit/`
- `.codex/skill-system/evidence/stage-03-contract-fixtures.json`
- `.codex/skill-system/evidence/stage-03-semantic-gates.json`
- `.codex/skill-system/rollback/manifest-v1.json`

## Файлы этапа

Созданы:

- `/Users/daniildegtyarev/.codex/skills/architecture-design/references/roehub-profile.md`
- `/Users/daniildegtyarev/.codex/skills/prompt-manager/references/prompt-contract.md`
- `/Users/daniildegtyarev/.codex/skills/prompt-manager/references/roehub-profile.md`
- три Stage 03 evidence artifacts;
- этот отчёт.

Изменены:

- 11 целевых корневых `SKILL.md`;
- `.codex/skill-system/catalog-v1.json` и глобальная authoritative копия;
- `.codex/skill-system/ownership-v1.json`;
- `.codex/skill-system/fixtures/skill-contract-cases-v1.json`;
- `.codex/skill-system/rollback/manifest-v1.json` и новые content-addressed blobs;
- stage ledger.

Удалённых файлов нет. Managed plugin cache не изменялся. Plugin не
устанавливался.

## Границы доказательства и рабочее дерево

Proof boundary: instruction/contract validation. Roehub API, browser, runtime,
deploy и production не запускались и не требуются для изменения скилл-контрактов.

В shared dirty `main` сохранены чужие изменения приложений, тестов, redesign
docs и prototypes. `tools/codex_quality_benchmark/`, связанные tests и
`docs/architecture/README.md` являются mixed/foreign surfaces; этап изменял
только принадлежащие текущему плану hunks и не выполнял staging, commit или
push.

Следующий разрешённый этап: `04-wave1-managed-overlay`.
