# Навигация агентов и субагентов по карте Roehub

Этот файл задаёт компактный маршрут чтения. Он не заменяет `AGENTS.md`, текущий task prompt или локальные инструкции. Skills выбираются по metadata текущей сессии; пустое поле `skills` сохранено только для совместимости schema v1.

## Обязательный порядок

1. Применить актуальный корневой `AGENTS.md` и относящиеся к задаче scoped instructions.
2. Для cross-context, repository-wide или неясной задачи открыть `project-map.json` и выбрать только релевантные `areas`, `components`, `entrypoints`, `docs` и `agent_routes`.
3. Проверить указанные пути в текущем коде: карта — навигационный индекс, а не доказательство runtime-поведения.
4. При разрешённом делегировании передать субагенту только нужный slice карты, точный outcome, owned paths и proof boundary. Не заставлять субагента читать весь inventory.
5. После добавления/перемещения компонентов выполнить генератор; generated-файлы вручную не редактировать.

## Машиночитаемые запросы

```bash
# Компонент и его зависимости
jq '.components[] | select(.id == "context:backtest")' docs/architecture/project-map/project-map.json

# Маршруты агента
jq '.agent_routes[]' docs/architecture/project-map/project-map.json

# Все файлы области
jq -r '.inventory[] | select(.area == "operations") | .path' docs/architecture/project-map/project-map.json
```

## Маршруты по типу работы

### HTTP API, DTO, auth, UI API

- Компоненты: `app:api`, `context:identity`
- Читать сначала: `apps/api/main/app.py`, `apps/api/routes/`, `apps/api/dto/`, `docs/architecture/api/`

### Current local-platform client UI, routes, locale, loading, charts

- Компоненты: `app:platform-web`, `app:api`
- Читать сначала: `apps/platform-web/src/`, `apps/platform-web/package.json`, `packages/web-contracts/src/`, `docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md`

### Web server/integration and legacy templates

- Компоненты: `app:web`, `app:api`
- Читать сначала: `apps/web/main/app.py`, `apps/web/templates/`, `apps/web/dist/`, `docs/architecture/apps/web/`

### Backtest, benchmark, optimize

- Компоненты: `context:backtest`, `context:backtest_artifacts`, `context:optimize`, `worker:backtest_job_runner`
- Читать сначала: `src/trading/contexts/backtest/`, `apps/worker/backtest_job_runner/`, `docs/architecture/backtest/`

### Market data, candles, ingestion

- Компоненты: `context:market_data`, `worker:market_data_ws`
- Читать сначала: `src/trading/contexts/market_data/`, `apps/worker/market_data_ws/`, `docs/architecture/market_data/`

### Live trading, order, exchange, risk

- Компоненты: `context:live_execution`, `context:exchange_control`, `context:risk`, `app:exchange_execution`, `worker:strategy_live_runner`
- Читать сначала: `src/trading/contexts/live_execution/`, `src/trading/contexts/exchange_control/`, `apps/exchange_execution/`, `docs/architecture/live_execution/`

### RL, ML, training, inference

- Компоненты: `context:rl_trading`, `context:ml`, `worker:rl_trading_trainer`, `worker:rl_trading_inference`
- Читать сначала: `src/trading/contexts/rl_trading/`, `apps/worker/rl_trading_trainer/`, `apps/worker/rl_trading_inference/`, `docs/architecture/ml/`

### Deploy, CI, runtime, monitoring

- Компоненты: `area:operations`
- Читать сначала: `.github/workflows/`, `infra/`, `docs/runbooks/`
