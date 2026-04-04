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
- R9-01 rollout note:
  - `/backtests` launch form consumes `contracts.request_timeframes.allowed`,
    `contracts.summary.ranking_metrics`, `contracts.summary.top_n_default`,
    `contracts.summary.top_n_max`, `contracts.launch.supported_indicator_ids`, and
    `contracts.launch.source_values_by_indicator_id` directly,
  - browser launch no longer requires manual `POST /api/indicators/estimate`,
  - `202 Accepted` with `execution_mode=background_auto` is treated as an explicit launch outcome,
    not as a silent mode switch.
- R1 additive contract:
  - endpoint publishes additive `contracts.*` fields for target-v2 semantics,
  - request TF restrictions and `signals.v1.params = default-only` are enforced in backend,
  - launch form can be driven from backend indicator/source catalog without YAML parsing in browser.
- A1 additive contract:
  - endpoint also publishes additive execution-profile discovery fields:
    `contracts.execution.default_execution_profile` and ordered
    `contracts.execution.available_execution_profiles`,
  - source config for this catalog lives in `backtest.execution_profiles`,
  - A1 introduced the typed catalog as discovery-only surface.
- B3 additive contract:
  - the same startup-validated execution-profile catalog now drives exact request classification,
    persisted-run `execution_profile_mode`, and `/backtests/runs*` progress/ETA weights;
  - each profile payload now also exposes additive `launch_budget` and `progress_weights`,
    so browser/debug tooling can inspect the same source of truth used by launch/history;
  - active runtime exact default remains `exact_small`, while benchmark corpus may still keep
    `exact_baseline=exact_parallel` as evidence anchor; these roles must stay explicit and
    non-interchangeable.

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
      "risk_model": "signal_tf + 1m_risk",
      "default_execution_profile": "exact_small",
      "available_execution_profiles": [
        {
          "mode": "exact_small",
          "shortlist_config": {
            "enabled": false,
            "max_candidates": null
          },
          "parallelism": {
            "stage_a_workers": 1,
            "stage_b_workers": 1
          },
          "feature_flags": {
            "runtime_enabled": true,
            "heuristic_shortlist_enabled": false,
            "parallel_stage_b_enabled": false,
            "family_plugin_enabled": false
          },
          "launch_budget": {
            "max_stage_a_variants_total": 1500,
            "max_stage_b_variants_total": 12000,
            "max_estimated_memory_bytes": 268435456
          },
          "progress_weights": {
            "stage_a": 25,
            "stage_b": 70,
            "finalizing": 5
          },
          "planning_budget_ms": 25
        },
        {
          "mode": "exact_parallel",
          "shortlist_config": {
            "enabled": false,
            "max_candidates": null
          },
          "parallelism": {
            "stage_a_workers": 1,
            "stage_b_workers": 4
          },
          "feature_flags": {
            "runtime_enabled": true,
            "heuristic_shortlist_enabled": false,
            "parallel_stage_b_enabled": true,
            "family_plugin_enabled": false
          },
          "launch_budget": {
            "max_stage_a_variants_total": 25000,
            "max_stage_b_variants_total": 180000,
            "max_estimated_memory_bytes": 1610612736
          },
          "progress_weights": {
            "stage_a": 35,
            "stage_b": 60,
            "finalizing": 5
          },
          "planning_budget_ms": 50
        },
        {
          "mode": "hybrid_conservative",
          "shortlist_config": {
            "enabled": true,
            "max_candidates": 5000
          },
          "parallelism": {
            "stage_a_workers": 1,
            "stage_b_workers": 4
          },
          "feature_flags": {
            "runtime_enabled": false,
            "heuristic_shortlist_enabled": false,
            "parallel_stage_b_enabled": false,
            "family_plugin_enabled": false
          },
          "launch_budget": {
            "max_stage_a_variants_total": 50000,
            "max_stage_b_variants_total": 250000,
            "max_estimated_memory_bytes": 2147483648
          },
          "progress_weights": {
            "stage_a": 50,
            "stage_b": 45,
            "finalizing": 5
          },
          "planning_budget_ms": 75
        },
        {
          "mode": "hybrid_family",
          "shortlist_config": {
            "enabled": true,
            "max_candidates": 2000
          },
          "parallelism": {
            "stage_a_workers": 1,
            "stage_b_workers": 4
          },
          "feature_flags": {
            "runtime_enabled": false,
            "heuristic_shortlist_enabled": false,
            "parallel_stage_b_enabled": false,
            "family_plugin_enabled": false
          },
          "launch_budget": {
            "max_stage_a_variants_total": 75000,
            "max_stage_b_variants_total": 300000,
            "max_estimated_memory_bytes": 2684354560
          },
          "progress_weights": {
            "stage_a": 60,
            "stage_b": 35,
            "finalizing": 5
          },
          "planning_budget_ms": 100
        }
      ]
    },
    "launch": {
      "execution_mode": "auto",
      "auto_preflight_enabled": true,
      "auto_fallback_to_background_enabled": true,
      "supported_indicator_ids": ["ma.sma", "momentum.trix"],
      "source_values_by_indicator_id": {
        "ma.sma": ["close", "hlc3"],
        "momentum.trix": ["close", "hlc3", "ohlc4"]
      }
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
- `contracts.execution.default_execution_profile` <- `backtest.execution_profiles.default`
- `contracts.execution.available_execution_profiles[]` <- ordered `backtest.execution_profiles.profiles`
  with nested fields:
  - `mode`
  - `shortlist_config.enabled`
  - `shortlist_config.max_candidates`
  - `parallelism.stage_a_workers`
  - `parallelism.stage_b_workers`
  - `feature_flags.runtime_enabled`
  - `feature_flags.heuristic_shortlist_enabled`
  - `feature_flags.parallel_stage_b_enabled`
  - `feature_flags.family_plugin_enabled`
  - `launch_budget.max_stage_a_variants_total`
  - `launch_budget.max_stage_b_variants_total`
  - `launch_budget.max_estimated_memory_bytes`
  - `progress_weights.stage_a`
  - `progress_weights.stage_b`
  - `progress_weights.finalizing`
  - `planning_budget_ms`
- `contracts.launch.execution_mode` <- `backtest.contracts.launch.execution_mode`
- `contracts.launch.auto_preflight_enabled` <- `backtest.contracts.launch.auto_preflight_enabled`
- `contracts.launch.auto_fallback_to_background_enabled` <- `backtest.contracts.launch.auto_fallback_to_background_enabled`
- `contracts.launch.supported_indicator_ids` <- ordered ids from `configs/<env>/indicators.yaml`
- `contracts.launch.source_values_by_indicator_id` <- ordered `inputs.source` values per supported indicator

Детерминизм:

- `fee_pct_default_by_market_id` сериализуется в key-sorted порядке по market id.
- массивы `contracts.*` сериализуются в YAML-defined order; порядок является частью frozen contract surface.
- `contracts.execution.available_execution_profiles` сериализуется в YAML-defined profile order;
  этот порядок является частью browser/runtime discovery contract.
- `launch_budget` и `progress_weights` публикуют reviewable server-side hints, но browser не
  выбирает `execution_profile_mode` самостоятельно; launch classification остаётся server-owned.
- `contracts.launch.supported_indicator_ids` сериализуется в детерминированном `indicator_id` порядке.
- `contracts.launch.source_values_by_indicator_id` сериализует ключи в том же порядке, значения `source` — в детерминированном literal order.
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
  - строит timeframe selector из `contracts.request_timeframes.allowed`;
  - строит ranking selectors из `contracts.summary.ranking_metrics`;
  - префилл user-facing `top_n` и кап `top_n_max`, затем явно маппит `top_n -> top_k`;
  - обновление `execution.fee_pct` при смене market, пока поле не стало user-dirty;
  - чтение frozen target-v2 literals из `contracts.*` без изменения текущего v1 request shape;
  - при необходимости показывает explanatory hints about `exact_small`, `exact_parallel`, and
    explicit queued `background_auto`, but не принимает profile decision на client side;
  - построение indicator/source selectors из `contracts.launch.supported_indicator_ids` и
    `contracts.launch.source_values_by_indicator_id`.

## Связанные файлы

- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
