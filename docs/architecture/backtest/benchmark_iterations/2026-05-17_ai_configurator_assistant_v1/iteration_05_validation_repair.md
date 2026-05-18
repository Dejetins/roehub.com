# Iteration 05 Validation/repair/load gate

Дата: 2026-05-18.

Статус: accepted locally, delivery pending.

## Предварительный gate

Iteration 04 проверен перед началом:

- `implementation_progress.json`: `04-prompt-lmstudio.accepted=true`;
- `next_iteration_allowed=true`;
- `iteration_04_prompt_lmstudio.json`: `pushed_to_main=true`;
- `macstudio_verified=true`;
- recorded accepted commit: `30b5153754ee4d8de224a80dd947a009642d225f`.

## Что изменено

- Добавлен backend repair call в `BacktestConfigAgentGateway`.
- `BacktestAiConfigPipeline` теперь записывает generate/repair attempts и делает максимум один repair.
- `LMStudioOpenAICompatibleAdapter` выполняет repair через тот же `POST /v1/chat/completions` runtime.
- `PipelineBacktestAiConversationGateway` строит `load_action` только из backend `ready + validated_config`.
- API DTO больше не включает load action по одному факту наличия assistant text.
- `TRUSTED_CONTEXT_JSON` скрывает 8 no-window indicators до loadable runtime contract.
- Добавлен Mac Studio smoke harness: `scripts/backtest_ai/run_iteration_05_validation_repair_smoke.py`.
- Создан contract doc: `validation_repair_contract.md`.

## Validation/repair

Repair policy:

- `repair_attempts: 1`;
- same LM Studio adapter/runtime;
- unsupported symbols/indicators and no-window execution gaps do not repair into fabricated configs;
- `auto_run_backtest_attempt` blocks at input gate and never reaches core backtest jobs.

Focused tests cover:

- valid ready config;
- unsupported indicator clarification;
- schema failure repair success;
- repair failure after one attempt;
- invalid JSON without unsafe repair;
- prompt injection;
- secret/output injection;
- `auto_run_backtest_attempt`.

## Indicator default audit

Local audit result: `32/40` preflight-valid.

Excluded/hidden model-facing indicators:

```text
momentum.stoch
structure.candle_stats
structure.candle_stats_atr_norm
structure.pivots
trend.psar
volatility.tr
volume.ad_line
volume.obv
```

Reason: current preflight/prepare-pools execution contract does not expose a loadable
`window` axis for these indicators. They are omitted from model-facing prompt context
with reason `hidden_until_no_window_runtime_contract_is_loadable`.

## Проверки

Completed locally:

```text
uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator tests/unit/contexts/backtest/application/services/v2
```

Result: `275 passed`.

```text
uv run ruff check src/trading/contexts/backtest/application/ai_configurator apps/api tests/unit/contexts/backtest/application/ai_configurator
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

## Delivery

Direct-main delivery is pending.

Final acceptance still requires:

- push to `origin/main`;
- green main CI/deploy;
- Mac Studio sync of the accepted commit;
- `scripts/backtest_ai/run_iteration_05_validation_repair_smoke.py` from deployed runtime;
- final evidence update with `pushed_to_main=true`, `macstudio_verified=true`, and `next_iteration_allowed=true`.
