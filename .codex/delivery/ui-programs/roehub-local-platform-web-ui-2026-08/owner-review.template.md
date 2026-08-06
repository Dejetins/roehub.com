---
artifact_kind: ui_design_owner_review
program_id: <program-id>
gate_id: <G3|G4|G5|G6>
validation_profile: <program_ready|family_review_ready|wave_review_ready|handoff_review_ready>
review_revision: <exact-revision>
transition_receipt: <transition-receipt-path>
mobile_scope: <exact-live-mobile-scope>
agent_self_acceptance: prohibited
---

# Готовый результат: <короткое название>

## Что получилось

До семи коротких пунктов о готовом результате. Этот owner-gated шаблон
применяется только к G3-G6; для G0-G2 используйте обычный негейтящий summary.
Сначала показывайте визуальный review board или screenshots.

## Посмотреть (только если есть rendered surface)

- `<главный rendered artifact или screenshot>`
- `<дополнительный G3-G6 artifact при необходимости>`

Не давайте больше пяти ссылок. Raw JSON, hashes, полный inventory, команды и
validator output остаются в durable evidence и открываются только по запросу.

## Автоматические проверки

| Граница | Результат | Evidence |
|---|---|---|
| Текущий gate | `<passed>` | `<receipt>` |
| Browser/raster/accessibility, если применимо | `<passed-or-N/A>` | `<receipt>` |
| Переход или завершение | `<review_ready|ready|complete|blocked>` | `<transition-receipt>` |

## Важные исключения и риски

- `<только материальный риск или none>`

## Нужна только ваша оценка результата

Задайте не более трёх вопросов только когда решение действительно
`owner_required`. Рекомендованный вариант ставьте первым. Если вопросов нет,
напишите: `Нет. После принятия агент продолжит по готовому handoff.`

## Принятие

Вы можете ответить естественно: `принимаю`, `ок`, `вариант 2`, `не
принимаю: ...` или дать ограниченные правки. Magic string, SHA и повторение
revision не требуются, если контекст однозначен.

После ответа агент сам создаёт и проверяет все технические записи. Пользователю
не нужно указывать идентификаторы, владельцев файлов, хеши, пути evidence или
повторно разрешать обычные записи в уже согласованной области программы.
