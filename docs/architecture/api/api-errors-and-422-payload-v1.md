# Ошибки API и payload 422

Статус: active; сверено с обработчиками и тестами 2026-09-06.

[Общие обработчики](../../../apps/api/common/errors.py) регистрируются для
`RoehubError` и `RequestValidationError`. Они возвращают JSON следующей формы:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Validation failed",
    "details": {
      "errors": [
        {"path": "body.name", "code": "missing", "message": "Field required"}
      ]
    }
  }
}
```

`RoehubError.code` определяет HTTP status через `_ROEHUB_STATUS_BY_CODE`.
Например, `auth.required` → 401, `forbidden` → 403, `not_found` → 404,
`conflict` → 409, `validation_error` → 422,
`backtest.rate_limited` → 429, `backtest.artifacts_unavailable` → 503.
Неизвестный код даёт 500; клиент не должен угадывать status по подстроке кода.
Полный mapping поддерживается в обработчике, без дублирования таблицы здесь.

`RequestValidationError` даёт 422 и `error.code = validation_error`.
Каждый элемент `details.errors` содержит только `path`, `code`, `message`:
`loc` превращается в путь с точками (включая `body`, `query` и индексы),
`type` становится кодом, `msg` — сообщением. Список сортируется по
`(path, code, message)`, обеспечивая детерминированный порядок. Исходные
`input` и `ctx` Pydantic не копируются в этот список.

`details` доменных ошибок нормализуются в JSON-совместимые значения;
для `validation_error` применяется тот же формат списка ошибок.
Обработчики не устанавливают этот envelope для любого возможного ответа:
обычные `HTTPException` и route-specific responses нужно проверять отдельно.

Проверка: [tests/unit/apps/api/test_api_error_handlers.py](../../../tests/unit/apps/api/test_api_error_handlers.py),
`python -m pytest -q tests/unit/apps/api/test_api_error_handlers.py`.
