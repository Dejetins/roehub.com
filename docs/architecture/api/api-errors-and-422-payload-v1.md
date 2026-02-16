```md
# STR-EPIC-02 — Strategy API v1 (updated): immutable CRUD + clone + run control + identity + standard errors

Цель: пользователь через API может создать стратегию, клонировать как шаблон/из существующей, запускать/останавливать run’ы. Все правила владения/видимости — доменно-очевидны (use-case), identity подключается через порт `current_user`, ошибки унифицированы через `RoehubError` и общий 422 payload.

## Scope

### Endpoints
- `POST /strategies` — create (immutable)
- `POST /strategies/clone` — clone (template/existing → new strategy)
- `GET /strategies` — list (только owner, по умолчанию без deleted)
- `GET /strategies/{id}` — get (только owner, по умолчанию без deleted)
- `POST /strategies/{id}/run` — создать новый run + перевести в running по правилам
- `POST /strategies/{id}/stop` — остановить активный run (stopping → stopped)
- `DELETE /strategies/{id}` — soft delete (deleted)

### Application / Use-cases (обязательно)
- `ListMyStrategies(current_user)`
- `GetMyStrategy(strategy_id, current_user)`
- `CreateStrategy(payload, current_user)`
- `CloneStrategy(source_strategy_id, overrides, current_user)`
- `RunStrategy(strategy_id, current_user)`
- `StopStrategy(strategy_id, current_user)`
- `DeleteStrategy(strategy_id, current_user)`

Правило: ownership/visibility не “в SQL”, а в use-case (явная доменная семантика).

### Identity integration (обязательно)
- Strategy контекст использует порт `CurrentUser`/`CurrentUserProvider`.
- API слой предоставляет реализацию порта через существующий identity (без копирования JWT логики в strategy).

### Runner semantics (обязательно)
- Warmup считается в runner’е детерминированно по индикаторам стратегии.
- “Второй run” (следующий запуск после остановки) должен быть возможен всегда при отсутствии активного run.

### Run state machine (обязательно)
- `stopped/failed -> run -> running`
- `running -> stop -> stopping -> stopped`
- конкурентность: один активный run на стратегию (fail-fast на гонках/конфликтах)

### Errors (обязательно)
- Единый 422 payload (как в indicators/market_data), детерминированный порядок ошибок.
- Общий механизм ошибок:
  - `RoehubError(code, message, details)` в `src/trading/platform/errors/`
  - доменные ошибки мапятся в RoehubError
  - API слой мапит RoehubError → HTTPException/Response

## Non-goals
- UI realtime и подписки на статусы (это STR-EPIC-04).
- Сложные права доступа (кроме owner).
- Распределённая оркестрация/кластерный runner.

## DoD (Definition of Done)
- Невозможно “обновить” стратегию: только новая через clone.
- Clone работает:
  - переносит indicator params,
  - поддерживает overrides `instrument_id/timeframe` (и только явно разрешённые поля),
  - owner = current_user.
- Ownership/visibility правило реализовано и покрыто тестами как use-case (не как “фильтр в SQL”).
- Run state machine соблюдается и покрыта unit + интеграционными тестами:
  - stopped/failed → run → running
  - running → stop → stopping → stopped
  - повторный запуск после stop (second run) работает
  - один активный run на стратегию (конкурентные запросы корректно дают конфликт/ошибку)
- Warmup считается runner’ом детерминированно и фиксируется в метаданных run.
- Ошибки:
  - RoehubError внедрён,
  - доменные ошибки мапятся в RoehubError,
  - API отдаёт единый payload и детерминированный 422.
- Документация:
  - добавлены 2 архитектурных документа:
    - `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`
    - `docs/architecture/api/api-errors-and-422-payload-v1.md`
  - индекс обновлён: `python -m tools.docs.generate_docs_index`