# Iteration 03 Conversation API/storage

Дата: 2026-05-18.

Статус: accepted, direct-main delivery pending.

## Цель

Добавить one-chat conversation API и storage для `/backtests` AI assistant без старых
browser-visible AI job endpoints.

## Предварительный gate

Iteration 02B проверен перед началом:

- `implementation_progress.json`: `02b-context-snapshot.accepted=true`;
- `next_iteration_allowed=true`;
- `iteration_02b_context_snapshot.json`: `pushed_to_main=true`;
- `macstudio_verified=true`;
- recorded accepted commit: `9f0fb780903e3fb4e3341adedfa290c4eaf6ac14`.

## Что изменено

- Добавлен conversation use case `BacktestAiConversationUseCase`.
- Добавлены DTO:
  - `BacktestAiConversation`;
  - `BacktestAiConversationMessage`;
  - `BacktestAiConversationRun`;
  - `BacktestAiLoadAction`;
  - `BacktestAiConversationModelResponse`.
- Добавлен repository port `BacktestAiConversationRepository`.
- Добавлен Postgres adapter `PostgresBacktestAiConversationRepository`.
- Добавлена migration `20260518_0014`.
- Добавлены conversation limits в `configs/{dev,test,prod}/backtest_ai_configurator.yaml`:
  - `retention_days=30`;
  - `max_conversations_per_user=50`;
  - `max_messages_per_conversation=100`.
- Добавлены FastAPI endpoints:
  - `GET /backtests/ai-config/conversations`;
  - `POST /backtests/ai-config/conversations`;
  - `GET /backtests/ai-config/conversations/{conversation_id}/messages`;
  - `POST /backtests/ai-config/conversations/{conversation_id}/messages`;
  - `GET /backtests/ai-config/conversations/{conversation_id}/status`;
  - `GET /backtests/ai-config/conversations/{conversation_id}/load-action`.
- Startup message создается на platform-selected `locale`.
- Browser-visible request не содержит `mode`.
- `load_action` существует как backend-gated placeholder и disabled до будущего
  backend state `ready`.
- Создан contract doc: `conversation_api_contract.md`.

## API contract

Новый public assistant API conversation-only. Retired AI job endpoints не
зарегистрированы.

Title policy:

- модель возвращает `conversation_title`;
- backend валидирует длину и unsafe chars;
- первый валидный model title сохраняется;
- missing/unsafe title дает fallback `New backtest chat`.

## Storage

Migration: `20260518_0014`.

Tables:

- `backtest_ai_conversations`;
- `backtest_ai_conversation_messages`;
- `backtest_ai_conversation_runs`.

Owner isolation реализован через `owner_user_id` в таблицах и owner-scoped repository
methods. Чужой `conversation_id` возвращается как `backtest.ai_config.not_found`.

## Старые endpoint refs

Current active code/API refs:

```text
current_active_hit_count=0
```

Historical retained refs:

- `docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/macstudio_blocker.md`;
- `docs/architecture/backtest/benchmark_iterations/2026-05-12_iteration_08_ai_configurator_load_security/macstudio_blocker.json`.

Они оставлены как historical evidence, не как current contract.

## Контрактное влияние

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | breaking-change | Новый conversation-only API; retired AI job endpoints не возвращены. |
| Browser-visible mode field | breaking-change | Новый request не содержит `mode`; intent будет backend-classified later. |
| DTO schema | breaking-change | Browser-visible assistant DTOs теперь conversation/message/status/load_action. |
| Persisted schema | compatible-change | Additive conversation/message/run tables. |
| Config schema | compatible-change | Additive `conversation` limits block. |
| Request hash/cache identity | none | Backtest jobs/artifact identity не менялись. |
| Runtime workflow | compatible-change | LM Studio не вызывается в Iteration 03; placeholder disabled. |

## Проверки

Completed locally:

```text
uv run pytest -q tests/unit/apps/api tests/unit/contexts/backtest/application/ai_configurator
```

Result: `196 passed`.

```text
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api tests/unit/contexts/backtest/application/ai_configurator
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

## Mac Studio

Pre-delivery smoke ran in isolated temp worktree `/tmp/roehub-iter03-smoke` with the
current local diff applied on top of `2496b1e8e558737138740e6d45e95ee057cdcd50`.

Migration command:

```text
PYTHONPATH=/tmp/roehub-iter03-smoke/src:/tmp/roehub-iter03-smoke \
  /opt/roehub/app/.venv/bin/python -m apps.migrations.main
```

Result:

- `Migration success`;
- `alembic_version=20260518_0014`;
- `public.backtest_ai_conversations=backtest_ai_conversations`;
- `public.backtest_ai_conversation_messages=backtest_ai_conversation_messages`;
- `public.backtest_ai_conversation_runs=backtest_ai_conversation_runs`.

API smoke used FastAPI `TestClient`, real `build_backtest_ai_configurator_use_cases`,
Mac Studio `STRATEGY_PG_DSN`, and fake current user
`00000000-0000-0000-0000-000000000903`.

Result:

```json
{
  "conversation_title": "New backtest chat",
  "limits": {
    "max_conversations_per_user": 50,
    "max_messages_per_conversation": 100,
    "retention_days": 30
  },
  "listed_count_before_cleanup": 1,
  "load_action": {
    "config": null,
    "enabled": false,
    "reason": "backend_not_ready",
    "state": "unavailable"
  },
  "send_status": "awaiting_model",
  "status_endpoint": "awaiting_model"
}
```

Smoke rows were deleted by `owner_user_id` after the route checks.

## Delivery

Direct-main delivery not started yet.

Current marker before Mac Studio and delivery:

- `accepted=true`;
- `next_iteration_allowed=false`;
- `pushed_to_main=false`;
- `macstudio_verified=true` for pre-delivery temp-worktree smoke, final deployed
  commit verification still pending.
