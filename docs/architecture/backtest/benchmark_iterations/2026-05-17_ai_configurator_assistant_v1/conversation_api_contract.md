# Backtest AI Configurator Assistant v1 — Conversation API Contract

Дата: 2026-05-18.

Статус: Iteration 03 contract.

## Цель

Зафиксировать backend-only conversation API для `/backtests` AI assistant. Контракт
заменяет старый one-shot AI job surface: browser-visible API больше не содержит ручной
`mode` и не публикует retired AI job endpoints.

## Limits

MVP storage limits:

- `retention_days=30`;
- `max_conversations_per_user=50`;
- `max_messages_per_conversation=100`.

История хранится в Roehub DB. LM Studio state не является источником истории.

## Public API

Все routes same-origin и требуют текущего Roehub user context.

### `GET /backtests/ai-config/conversations`

Возвращает owner-scoped список активных, не истекших conversations.

Query:

- `limit`: `1..50`, default `50`.

Response:

- `conversations[]`;
- `limits`.

### `POST /backtests/ai-config/conversations`

Создает один чат и startup assistant message на выбранном platform locale.

Request:

```json
{
  "locale": "en"
}
```

Response status: `201`.

Response содержит:

- `conversation.conversation_id`;
- `conversation.conversation_title`;
- `messages[]` с startup assistant message;
- `status`;
- `limits`.

Fallback title до первого валидного model title: `New backtest chat`.

### `GET /backtests/ai-config/conversations/{conversation_id}/messages`

Возвращает owner-scoped conversation, messages и latest status. Чужой
`conversation_id` возвращает `backtest.ai_config.not_found`, а не раскрывает факт
существования чужого чата.

### `POST /backtests/ai-config/conversations/{conversation_id}/messages`

Добавляет user message, сохраняет assistant response placeholder и run row.

Request:

```json
{
  "message": "Create RSI for BTCUSDT",
  "current_config": {},
  "ui_context": {}
}
```

Response status: `201`.

Response содержит:

- user `message_id`;
- assistant message;
- run `status`;
- `load_action`.

Iteration 03 не вызывает LM Studio. Production gateway возвращает disabled placeholder:

```json
{
  "enabled": false,
  "state": "unavailable",
  "reason": "backend_not_ready",
  "config": null
}
```

### `GET /backtests/ai-config/conversations/{conversation_id}/status`

Возвращает latest run status и `load_action`.

### `GET /backtests/ai-config/conversations/{conversation_id}/load-action`

Возвращает тот же backend-gated `load_action`. В Iteration 03 он всегда disabled, потому
что validation/repair/load gate будет добавлен позже. Будущий `enabled=true` допустим
только при backend state `ready` и наличии validated config.

## Conversation Title

Модель должна возвращать title как `conversation_title`.

Backend policy:

- принимает только первый безопасный model title;
- нормализует whitespace;
- ограничивает длину до 80 chars;
- отклоняет control chars и `<` / `>`;
- если title отсутствует или unsafe, сохраняет fallback `New backtest chat`;
- после первого валидного model title backend не перезаписывает title следующими
  model titles.

## Storage

Migration: `20260518_0014`.

Tables:

- `backtest_ai_conversations`;
- `backtest_ai_conversation_messages`;
- `backtest_ai_conversation_runs`.

Owner isolation:

- каждая table содержит `owner_user_id`;
- repository reads always filter by `owner_user_id`;
- foreign conversation reads map to not-found.

Retention:

- `expires_at = last_write_at + retention_days`;
- list/get filter active rows with `expires_at > now()`;
- physical purge is an ops cleanup concern outside Iteration 03.

## Contract Impact

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | breaking-change | Retired AI job endpoints не возвращаются; новый API conversation-only. |
| Browser-visible mode field | breaking-change | Новый request не содержит `mode`; intent classification переносится на backend/model boundary later. |
| DTO schema | breaking-change | Новые conversation DTOs вместо retired job DTOs для browser-visible assistant API. |
| Persisted schema | compatible-change | Additive tables `backtest_ai_conversations`, `backtest_ai_conversation_messages`, `backtest_ai_conversation_runs`. |
| Config schema | compatible-change | Additive `conversation` limits block with defaults. |
| Request hash/cache identity | none | Backtest job request hash and artifact identities не меняются. |
| Runtime workflow | compatible-change | LM Studio не вызывается; assistant returns disabled placeholder until later iteration. |
