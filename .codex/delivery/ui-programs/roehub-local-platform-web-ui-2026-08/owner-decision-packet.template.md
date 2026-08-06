---
artifact_kind: ui_design_owner_decision_packet
program_id: <program-id>
stage_instance_id: <stage-instance-id>
status: needs_input
question_count: <1-3>
resume_same_stage: true
---

# Нужны решения владельца

## Что уже готово

Кратко опишите готовый результат. Для G3-G6 дайте до трёх визуальных ссылок;
для G0-G2 используйте только короткую non-visual сводку и source-linked delta.
Не показывайте владельцу raw JSON, полный atlas или длинный validator output.

## Вопросы

### 1. <короткий вопрос>

- **<рекомендованный вариант>** — <видимый эффект>.
- <вариант 2> — <видимый эффект>.

## Что произойдёт после ответа

Агент запишет canonical receipt, возобновит тот же stage instance и продолжит
до готового результата. Новое задание, повтор write authorization и magic
acceptance string не требуются.

Пользователь отвечает обычными словами. Идентификаторы, пути, JSON, хеши,
ledger и техническое подтверждение точного артефакта заполняет агент.
