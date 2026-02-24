# Web / API Contract -- Backtest runtime defaults endpoint v1

Документ фиксирует контракт `GET /backtests/runtime-defaults` (через gateway как
`GET /api/backtests/runtime-defaults`) для PR2 web backtests UI.

## Цель

- Дать браузеру один стабильный endpoint для загрузки runtime defaults, которые уже
  загружены и провалидированы API на старте.
- Исключить дублирование YAML-логики в браузере и ненадежные placeholder-дефолты.

## Endpoint

- Method: `GET`
- Path (API module): `/backtests/runtime-defaults`
- Path (browser через gateway): `/api/backtests/runtime-defaults`
- Auth: тот же dependency, что и для `POST /backtests` (cookie-session -> current user).

## Источник данных и fail-fast

- Source of truth: `configs/<env>/backtest.yaml` (production: `configs/prod/backtest.yaml`).
- Загрузка выполняется на старте в wiring:
  - `apps/api/wiring/modules/backtest.py`
  - `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- Endpoint не читает YAML на каждый request; использует предсобранный DTO из startup config.

## Response contract v1

```json
{
  "warmup_bars_default": 200,
  "top_k_default": 300,
  "preselect_default": 20000,
  "top_trades_n_default": 3,
  "execution": {
    "init_cash_quote_default": 10000.0,
    "fixed_quote_default": 100.0,
    "safe_profit_percent_default": 30.0,
    "slippage_pct_default": 0.01,
    "fee_pct_default_by_market_id": {
      "1": 0.075,
      "2": 0.1
    }
  },
  "jobs": {
    "top_k_persisted_default": 300
  }
}
```

## Поля и инварианты

- `warmup_bars_default` <- `backtest.warmup_bars_default`
- `top_k_default` <- `backtest.top_k_default`
- `preselect_default` <- `backtest.preselect_default`
- `top_trades_n_default` <- `backtest.reporting.top_trades_n_default`
- `execution.init_cash_quote_default` <- `backtest.execution.init_cash_quote_default`
- `execution.fixed_quote_default` <- `backtest.execution.fixed_quote_default`
- `execution.safe_profit_percent_default` <- `backtest.execution.safe_profit_percent_default`
- `execution.slippage_pct_default` <- `backtest.execution.slippage_pct_default`
- `execution.fee_pct_default_by_market_id` <- `backtest.execution.fee_pct_default_by_market_id`
- `jobs.top_k_persisted_default` <- `backtest.jobs.top_k_persisted_default`

Детерминизм:

- `fee_pct_default_by_market_id` сериализуется в key-sorted порядке по market id.
- Payload содержит только non-secret значения, нужные для browser prefill/validation hints.

## Использование в UI

- `apps/web/templates/backtests.html` задает data-hook
  `data-api-backtest-runtime-defaults-path="/api/backtests/runtime-defaults"`.
- `apps/web/dist/backtest_ui.js` загружает defaults один раз при инициализации страницы:
  - префилл `Advanced` input `.value`;
  - обновление `execution.fee_pct` при смене market, пока поле не стало user-dirty;
  - подсказка по cap `jobs.top_k_persisted_default`.

## Связанные файлы

- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
