# Web / API Contract -- Backtest runtime defaults endpoint v1

Документ фиксирует контракт `GET /backtests/runtime-defaults` (через same-origin `/api/*` proxy как
`GET /api/backtests/runtime-defaults`) для PR2 web backtests UI.

## Status

- Status: active v1 endpoint contract with additive R0 freeze fields.
- Superseded by target-v2 source-of-truth:
  - `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
  - `docs/architecture/roadmap/base_refactor_plan.md`
  - `docs/architecture/backtest/backtest-v2-benchmarks.md`
- Historical scope kept here:
  - current browser prefill defaults already loaded on API startup,
  - legacy `top_k_*` fields required for backward compatibility.
- R0 freeze addition:
  - endpoint now publishes additive `contracts.*` fields for target-v2 semantics,
  - current request/response/runtime behavior is not broken or silently renamed.

## Цель

- Дать браузеру один стабильный endpoint для загрузки runtime defaults, которые уже
  загружены и провалидированы API на старте.
- Исключить дублирование YAML-логики в браузере и ненадежные placeholder-дефолты.

## Endpoint

- Method: `GET`
- Path (API module): `/backtests/runtime-defaults`
- Path (browser через same-origin proxy): `/api/backtests/runtime-defaults`
- Auth: тот же dependency, что и для `POST /backtests` (cookie-session -> current user).

## Источник данных и fail-fast

- Source of truth: `configs/<env>/backtest.yaml` (production: `configs/prod/backtest.yaml`).
- Загрузка выполняется на старте в wiring:
  - `apps/api/wiring/modules/backtest.py`
  - `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- Endpoint не читает YAML на каждый request; использует предсобранный DTO из startup config.

## Response contract v1 + R0 additive freeze

```json
{
  "warmup_bars_default": 200,
  "top_k_default": 300,
  "preselect_default": 20000,
  "top_trades_n_default": 3,
  "ranking": {
    "primary_metric_default": "total_return_pct",
    "secondary_metric_default": null
  },
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
  },
  "contracts": {
    "request_timeframes": {
      "allowed": ["15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "2d", "3d"],
      "forbidden": ["1m", "5m"]
    },
    "summary": {
      "top_n_default": 100,
      "top_n_max": 300,
      "ranking_metrics": [
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct"
      ],
      "sortable_columns": [
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
        "trade_count",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
        "best_tp_pct",
        "best_sl_pct"
      ]
    },
    "signals": {
      "params_path": "signals.v1.params",
      "params_policy": "default-only"
    },
    "execution": {
      "risk_model": "signal_tf + 1m_risk"
    },
    "launch": {
      "execution_mode": "auto",
      "auto_preflight_enabled": true,
      "auto_fallback_to_background_enabled": true
    }
  }
}
```

## Поля и инварианты

- `warmup_bars_default` <- `backtest.warmup_bars_default`
- `top_k_default` <- `backtest.top_k_default`
- `preselect_default` <- `backtest.preselect_default`
- `top_trades_n_default` <- `backtest.reporting.top_trades_n_default`
- `ranking.primary_metric_default` <- `backtest.ranking.primary_metric_default`
- `ranking.secondary_metric_default` <- `backtest.ranking.secondary_metric_default`
- `execution.init_cash_quote_default` <- `backtest.execution.init_cash_quote_default`
- `execution.fixed_quote_default` <- `backtest.execution.fixed_quote_default`
- `execution.safe_profit_percent_default` <- `backtest.execution.safe_profit_percent_default`
- `execution.slippage_pct_default` <- `backtest.execution.slippage_pct_default`
- `execution.fee_pct_default_by_market_id` <- `backtest.execution.fee_pct_default_by_market_id`
- `jobs.top_k_persisted_default` <- `backtest.jobs.top_k_persisted_default`
- `contracts.request_timeframes.allowed` <- `backtest.contracts.request_timeframes.allowed`
- `contracts.request_timeframes.forbidden` <- `backtest.contracts.request_timeframes.forbidden`
- `contracts.summary.top_n_default` <- `backtest.contracts.summary.top_n_default`
- `contracts.summary.top_n_max` <- `backtest.contracts.summary.top_n_max`
- `contracts.summary.ranking_metrics` <- `backtest.contracts.summary.ranking_metrics`
- `contracts.summary.sortable_columns` <- `backtest.contracts.summary.sortable_columns`
- `contracts.signals.params_path` <- `backtest.contracts.signals.params_path`
- `contracts.signals.params_policy` <- `backtest.contracts.signals.params_policy`
- `contracts.execution.risk_model` <- `backtest.contracts.execution.risk_model`
- `contracts.launch.execution_mode` <- `backtest.contracts.launch.execution_mode`
- `contracts.launch.auto_preflight_enabled` <- `backtest.contracts.launch.auto_preflight_enabled`
- `contracts.launch.auto_fallback_to_background_enabled` <- `backtest.contracts.launch.auto_fallback_to_background_enabled`

Детерминизм:

- `fee_pct_default_by_market_id` сериализуется в key-sorted порядке по market id.
- массивы `contracts.*` сериализуются в YAML-defined order; порядок является частью frozen contract surface.
- Payload содержит только non-secret значения, нужные для browser prefill/validation hints.

## Migration note: `top_k` vs `top_n`

- `top_k_default` и `jobs.top_k_persisted_default` остаются обязательными legacy-полями для текущего v1 runtime/API.
- `contracts.summary.top_n_default` и `contracts.summary.top_n_max` фиксируют target-v2 vocabulary.
- До runtime cutover UI/API не должны silently переименовывать `top_k` поля; mapping должен быть явным и тестируемым.

## Использование в UI

- `apps/web/templates/backtests.html` задает data-hook
  `data-api-backtest-runtime-defaults-path="/api/backtests/runtime-defaults"`.
- `apps/web/dist/backtest_ui.js` загружает defaults один раз при инициализации страницы:
  - префилл `Advanced` input `.value`;
  - префилл ranking selectors (`primary_metric`, `secondary_metric`);
  - обновление `execution.fee_pct` при смене market, пока поле не стало user-dirty;
  - подсказка по cap `jobs.top_k_persisted_default`;
  - чтение frozen target-v2 literals из `contracts.*` без изменения текущего v1 request shape.

## Связанные файлы

- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
