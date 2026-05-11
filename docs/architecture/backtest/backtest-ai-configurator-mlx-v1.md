# Backtest AI Configurator MLX v1

Документ фиксирует целевую production-архитектуру AI configurator для страницы `/backtests` на локальных MLX-моделях, Mac Studio inference host и существующем Roehub Web UI.

Статус: целевой production-план перед реализацией.

Дата фиксации: 2026-05-11.

## Цель

Сделать AI configurator в блоке `/backtests`, который принимает русский или
английский запрос пользователя, собирает конфигурацию backtest strategy из
текущих платформенных возможностей, валидирует ее через backend и возвращает в
чат понятный результат:

- валидная конфигурация доступна как кнопка `Загрузить конфигурацию`;
- форма `/backtests` заполняется, но backtest job не создается автоматически;
- если запрос неполный, AI дает рабочий вариант с явными assumptions и
  уточняет, что можно добавить;
- если запрос нельзя безопасно привести к поддерживаемому конфигу, пользователь
  получает объяснение и ближайшую корректировку запроса;
- все запросы, ответы, validation/repair attempts и факт применения сохраняются
  для будущего дообучения MLX-модели.

Система должна быть production-ready по границам: очередь, лимиты, owner scope,
валидация, repair loop, наблюдаемость, Tailscale-only доступ к inference host и
возможность менять MLX model path через конфиг без переписывания бизнес-логики.

## Контекст

Текущий `/backtests` не является SPA. Он работает как FastAPI SSR + Jinja2 +
plain ES modules:

- `apps/web/templates/pages/backtests.html` - workstation layout и placeholder
  AI block;
- `apps/web/dist/js/pages/backtests.js` - page-scoped JS, сборка текущего
  backtest request payload, polling jobs/results;
- browser-visible API идет через same-origin `/api/...`;
- backend API route не должен получать второй `/api` prefix;
- `apps/api/routes/ui_backtests.py` и
  `apps/api/wiring/modules/ui_backtests.py` отдают
  `/ui/backtests/workstation`;
- `ai_configurator_state` сейчас возвращает `enabled=false`, `state=placeholder`,
  `stage=Stage 10`;
- source of truth для supported indicators и их доступных параметров:
  `configs/prod/indicators.yaml`;
- runtime defaults и validation уже есть в
  `src/trading/contexts/backtest/application/services/v2/preflight.py`;
- текущий backtest create path уже durable queued:
  `POST /backtests/jobs` сохраняет `queued` job, а тяжелый compute не должен
  возвращаться в API request path.

Внешние runtime-ограничения:

- `mlx_lm.server` дает OpenAI-like HTTP API и поддерживает `stream`, но
  официальная документация `mlx-lm` предупреждает, что этот server не
  рекомендуется выставлять как production surface, потому что в нем только
  базовые security checks. Поэтому он допустим только как internal runtime за
  gateway.
- MLX оптимизирован под Apple Silicon unified memory, что делает Mac Studio
  M2 Max 64GB подходящим MVP inference host, но concurrency/context window надо
  подтверждать benchmark evidence, а не задавать "на глаз".

Проверка Tailscale на 2026-05-11:

- `tailscale status --self --json` показывает `BackendState=Running` и `TUN=true`
  на локальной машине;
- Mac Studio peer виден как online/active;
- VPS peer виден в tailnet;
- приватные Tailnet IP/DNS не фиксируются в git-документе. Их нужно брать из
  deployment env или `tailscale status`, чтобы не хардкодить private topology.

## Охват

Входит:

- AI configurator только для страницы `/backtests`;
- режимы `create config from prompt`, `edit current config`, `explain current
  config`, `repair invalid config`, `suggest safer config`;
- русский и английский язык пользовательских запросов;
- только параметры, доступные пользователю на текущей `/backtests` форме:
  exchange, market, symbol, timeframe, period, indicators, indicator params,
  risk/SL/TP, long/short direction, fees, slippage, entry sizing, ranking,
  top_n;
- только MLX-модели и MLX runtime adapter;
- model path задается конфигом как путь к папке модели;
- SSE/status events для `queued`, `preparing_catalog`, `generating`,
  `validating`, `repairing`, `ready` и ошибочных terminal states;
- friendly overload/quota UX с estimated wait, а не raw error code в UI;
- сохранение данных для будущего fine-tuning;
- benchmark/load-test план для 1, 5, 10, 50 и 100 online users;
- MVP на одном Mac Studio M2 Max 64GB inference host.

## Что не входит

Не входит:

- AI configurator для страниц strategies, monitoring, live trading или settings;
- автоматический запуск backtest job после AI-ответа;
- публичный доступ к `mlx_lm.server`;
- remote OpenAI, llama.cpp, GGUF fallback или не-MLX модели;
- multi-host inference scaling в MVP;
- показ chain-of-thought модели пользователю;
- дообучение модели в рамках MVP;
- замена текущего FastAPI SSR/Jinja/plain JS Web UI на React/Vite/SPA.

## Целевая архитектура

```text
Browser / /backtests Web UI
        |
        | same-origin /api/backtests/ai-config/*
        v
VPS Web UI proxy / existing apps/web
        |
        | Tailscale private route to backend API
        v
Existing Backend API / Backtest AI Configurator routes
        |
        | create/read job, SSE events, owner scope, quotas
        v
Postgres durable AI config queue + audit tables
        |
        v
Mac Studio backtest-ai-configurator-worker
        |
        | catalog subset + prompt profile + LLM Gateway
        v
MLX Runtime Adapter
        |
        | loopback only
        v
mlx_lm.server or custom MLX worker
        |
        v
JSON parse -> schema validation -> business preflight -> repair loop
        |
        v
Validated BacktestConfigDraft + assistant message + load button
```

Главное решение: MLX-модель не является source of truth. Source of truth:

- `configs/prod/indicators.yaml`;
- `BacktestRuntimeDefaultsService`;
- `BacktestPreflightService`;
- current artifact manifests для symbols/timeframes;
- tier/guardrail config;
- JSON Schema и business validators.

Модель отвечает за natural-language mapping и user-facing explanation. Backend
отвечает за допустимость, безопасность и финальный payload.

## Направление зависимостей

Целевая форма внутри backend:

```text
apps/api/routes/backtest_ai_config.py
        |
        v
src/trading/contexts/backtest/application/ai_configurator/*
        |
        +--> ports/catalog.py
        +--> ports/llm_gateway.py
        +--> ports/repository.py
        +--> services/pipeline.py
        +--> services/quota.py
        +--> services/validator.py
        |
        v
adapters/outbound/*
        +--> persistence/postgres
        +--> llm/mlx_openai_compatible
        +--> catalog/runtime_defaults
```

Рекомендуемая граница: не создавать общий `ai` bounded context для всего
продукта в v1. Capability привязана к `/backtests`, поэтому application code
живет рядом с backtest context. Heavy LLM execution живет в отдельном process:

```text
apps/worker/backtest_ai_configurator/main.py
```

Так API остается existing public contract owner, а LLM worker можно reload,
restart, profile и benchmark отдельно от `com.roehub.api`.

## Компоненты

### 1) Web UI

`apps/web/templates/pages/backtests.html` должен превратить placeholder AI block в
рабочий chat panel без смены frontend stack.

Минимальные UI обязанности:

- отправить prompt, mode и текущий form snapshot;
- получить `job_id`;
- подписаться на SSE `events` или использовать fallback polling;
- показывать pipeline stages, но не chain-of-thought;
- после terminal `ready` печатать `assistant_message` в чат с typewriter effect;
- показывать кнопку `Загрузить конфигурацию` только если backend вернул
  `validated_config`;
- по клику локально заполнить текущую форму `/backtests` и indicator table;
- отправить feedback event `applied=true|false` для training dataset.

MVP не должен stream-ить сырой JSON из модели в UI. Частичный model output может
не пройти validation и не должен появляться как обещание пользователю.

### 2) Backend API

Browser-visible routes:

```text
POST /api/backtests/ai-config/jobs
GET  /api/backtests/ai-config/jobs/{job_id}
GET  /api/backtests/ai-config/jobs/{job_id}/events
POST /api/backtests/ai-config/jobs/{job_id}/feedback
```

Backend router paths внутри `apps/api`:

```text
POST /backtests/ai-config/jobs
GET  /backtests/ai-config/jobs/{job_id}
GET  /backtests/ai-config/jobs/{job_id}/events
POST /backtests/ai-config/jobs/{job_id}/feedback
```

API process делает только lightweight работу:

- auth/current user;
- owner scope;
- quota/admission check;
- запись durable job;
- чтение статуса и SSE event stream;
- выдача friendly overload/quota response;
- не вызывает MLX напрямую в request path.

### 3) Durable queue и worker

Нужна отдельная очередь, не смешанная с `backtest_jobs`, потому что AI config job
не является compute backtest job и не должен менять `request_hash` backtest jobs.

Target table:

```text
backtest_ai_config_jobs
```

Основные поля:

- `job_id UUID PRIMARY KEY`;
- `owner_user_id UUID NOT NULL`;
- `mode TEXT NOT NULL`;
- `locale TEXT NOT NULL CHECK locale IN ('ru','en')`;
- `state TEXT NOT NULL CHECK state IN (...)`;
- `source_page TEXT NOT NULL DEFAULT 'backtests'`;
- `user_prompt_text TEXT NOT NULL`;
- `current_config_json JSONB NULL`;
- `validated_config_json JSONB NULL`;
- `assistant_message TEXT NULL`;
- `suggestions_json JSONB NOT NULL DEFAULT '[]'`;
- `validation_errors_json JSONB NOT NULL DEFAULT '[]'`;
- `model_id TEXT NULL`;
- `model_path_hash TEXT NULL`;
- `system_prompt_version TEXT NOT NULL`;
- `catalog_snapshot_hash TEXT NOT NULL`;
- `runtime_defaults_hash TEXT NOT NULL`;
- `queued_at`, `started_at`, `finished_at`, `updated_at`;
- `locked_by`, `locked_at`, `lease_expires_at`, `heartbeat_at`, `attempt`;
- `quota_charged BOOLEAN NOT NULL DEFAULT false`;
- `applied_at TIMESTAMPTZ NULL`;
- `user_feedback_json JSONB NULL`.

Сопутствующие таблицы:

```text
backtest_ai_config_events
backtest_ai_config_llm_attempts
backtest_ai_quota_events
```

`backtest_ai_config_llm_attempts` хранит generate/repair attempts:

- prompt profile;
- user prompt;
- compact catalog subset;
- raw model response;
- parsed JSON draft;
- validation errors;
- token estimates;
- latency;
- finish reason;
- success/failure reason.

Это и есть training data source. Для fine-tuning export использовать только
rows с понятным статусом (`ready`, `applied`, `needs_clarification`) и без
service/internal секретов.

### 4) Catalog Resolver

Catalog Resolver строит компактный, платформенно-достоверный subset:

- symbols из artifact manifests / workstation instrument universe;
- timeframes из `BacktestRuntimeDefaultsService`;
- indicators и param specs из `configs/prod/indicators.yaml`;
- allowed sources из runtime defaults;
- risk modes, direction modes, sizing modes, ranking metrics, guardrails;
- tier-specific limits.

Правило prompt budget: модель не получает весь мир, если можно детерминированно
сузить catalog. Для фразы `биток и эфир с RSI и Bollinger` resolver сначала
строит candidates:

```json
{
  "symbols": [
    {"input": "биток", "candidates": ["BTCUSDT"]},
    {"input": "эфир", "candidates": ["ETHUSDT"]}
  ],
  "indicators": [
    {"input": "RSI", "candidates": ["momentum.rsi"]},
    {"input": "Bollinger", "candidates": []}
  ],
  "timeframes": [{"input": "часовик", "candidates": []}]
}
```

Если платформа не поддерживает Bollinger или `1h`, модель не должна выдумывать
их. Backend либо выбирает ближайший допустимый вариант с warning, либо просит
уточнение.

Текущий supported indicator catalog на 2026-05-11 содержит 40 `indicator_id` в
группах `ma`, `trend`, `volatility`, `momentum`, `volume`, `structure`. Для
реализации список берется только из `configs/prod/indicators.yaml`, не из текста
этого документа.

### 5) LLM Gateway

Gateway скрывает runtime details за application port:

```python
class BacktestConfigLLMGateway(Protocol):
    def generate_config(self, request: BacktestConfigLLMRequest) -> BacktestConfigLLMResponse:
        ...

    def repair_config(self, request: BacktestConfigRepairRequest) -> BacktestConfigLLMResponse:
        ...
```

Реализация MVP:

```text
MLXOpenAICompatibleAdapter -> http://127.0.0.1:<port>/v1/chat/completions
```

`mlx_lm.server` остается internal-only за gateway. Если позже нужен полный
control lifecycle, adapter можно заменить на custom MLX worker без изменения API
routes, validators и storage.

### 6) MLX Runtime на Mac Studio

MVP host:

```text
Mac Studio M2 Max 64GB
```

Process layout:

```text
com.roehub.api
com.roehub.backtest-ai-configurator-worker
com.roehub.mlx-runtime.backtest-configurator
```

`mlx-runtime` bind:

```text
host: 127.0.0.1
port: configurable
```

Публично и через Tailscale runtime не открывать. Через Tailscale доступен только
existing backend API / web upstream, который уже делает auth, owner scope, quota
и validation.

Config sketch:

```yaml
backtest_ai_configurator:
  enabled: true
  queue:
    max_queue_size: 50
    lease_seconds: 120
    job_timeout_seconds: 90
    repair_attempts: 1
  model:
    model_id: gemma_4_e2b_it_4bit_local
    provider: mlx
    runtime: mlx_lm_server
    model_path: /Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit
    base_url: http://127.0.0.1:8081/v1
    context_window: 8192
    max_input_tokens: 5500
    max_output_tokens: 900
    temperature: 0.0
    top_p: 0.9
    active_generations: 1
  ux:
    sse_heartbeat_seconds: 15
    typewriter_min_chars_per_second: 30
```

Machine-specific `model_path` можно держать в prod/local config или env override,
но schema должна поддерживать простой folder path.

### 7) Model registry и reload

Нужен registry, но не нужно держать много моделей в памяти.

```yaml
models:
  backtest_config_default:
    provider: mlx
    runtime: mlx_lm_server
    model_path: /Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit
    base_url: http://127.0.0.1:8081/v1
    context_window: 8192
    max_output_tokens: 900
    active_generations: 1

  backtest_config_candidate:
    provider: mlx
    runtime: mlx_lm_server
    model_path: /Users/daniildegtyarev/.lmstudio/models/mlx-community/<other-model>
    base_url: http://127.0.0.1:8082/v1
    context_window: 8192
    max_output_tokens: 900
    active_generations: 1
    enabled: false
```

MVP policy:

- одна активная модель;
- maintenance reload для смены модели;
- rolling switch допустим, если второй runtime поднят на другом port и worker
  переключается через config reload;
- аварийный fallback: reload той же модели, а не remote fallback;
- standby same-model instance можно использовать только как runtime резерв, не
  как самостоятельный "собеседник".

Repair не требует "общения моделей между собой". Orchestrator делает так:

```text
draft model call
    -> validator errors
    -> repair prompt profile with original draft + errors
    -> same MLX instance or standby same-model instance
    -> revised JSON
```

Модели не вызывают друг друга напрямую. Все решения о retry/repair принимает
backend pipeline.

## Pipeline

Общий pipeline:

```text
1. API принимает prompt + mode + current_config
2. Auth, owner scope, quota/admission
3. Создается durable AI config job
4. Worker claim job
5. Normalize language and intent
6. Domain gate: только /backtests config intent
7. Load current runtime defaults and current form config
8. Catalog candidate lookup
9. Build compact prompt
10. MLX generate
11. Parse strict JSON
12. JSON Schema validation
13. Map to BacktestConfigDraft shape
14. Business validation через BacktestPreflightService
15. Repair loop <= 1 attempt
16. Persist final state and audit
17. UI получает assistant_message + validated_config
18. User clicks "Загрузить конфигурацию"
19. Browser fills current /backtests form
20. Feedback event persists applied=true
```

### UX policy для уточнений

Правило для лучшего UX:

- если есть безопасный default, backend возвращает valid config и явно пишет
  assumptions;
- если пользователь не упомянул stop loss, это не blocker. Default:
  `risk.mode=none`, а assistant message предлагает добавить `tp_sl_grid`;
- если пользователь просит "сделай безопаснее", pipeline может выбрать
  `tp_sl_grid`, но только если TP/SL grid проходит validation;
- если пользователь просит unsupported timeframe, indicator или symbol, backend
  пробует ближайший supported вариант и пишет correction warning;
- если ближайшего supported варианта нет, возвращается `needs_clarification` без
  `validated_config`;
- невозможный или unsupported config никогда не показывается с кнопкой
  `Загрузить конфигурацию`.

## Режимы

### create config from prompt

Вход: prompt + optional empty/default config.

Выход: полный `validated_config` для заполнения формы.

### edit current config

Вход: prompt + current form snapshot.

Выход: полный `validated_config`, не patch-only. UI проще и безопаснее применяет
полный snapshot.

### explain current config

Вход: current form snapshot + optional prompt.

Выход: `assistant_message`, `status=explanation`, без load button. Model не
переписывает config.

### repair invalid config

Вход: current form snapshot + validation errors.

Выход: исправленный `validated_config` или `needs_clarification`.

### suggest safer config

Вход: current form snapshot + user risk preference.

Выход: консервативный `validated_config`, например smaller sizing, explicit
TP/SL grid, lower variants count, если эти параметры доступны платформе.

## API contracts

### Create job

```http
POST /api/backtests/ai-config/jobs
```

Request:

```json
{
  "mode": "create",
  "locale": "ru",
  "message": "Собери конфиг для BTC и ETH на RSI за 2023 год",
  "current_config": null,
  "ui_context": {
    "page": "backtests",
    "runtime_defaults_hash": "optional-client-observed-hash"
  }
}
```

Response accepted:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "events_url": "/api/backtests/ai-config/jobs/{job_id}/events",
  "estimated_wait_seconds": 8,
  "message": "Запрос поставлен в очередь. Ожидаемое время ответа около 8 секунд."
}
```

Quota/capacity response:

```json
{
  "job_id": null,
  "status": "capacity_delayed",
  "estimated_wait_seconds": 90,
  "message": "AI configurator сейчас под высокой нагрузкой. Попробуйте примерно через 1-2 минуты.",
  "retry_after_seconds": 90
}
```

UI не должен показывать пользователю только HTTP status. Даже если backend
использует `429` для API semantics, payload обязан содержать user-facing
message.

### Read job

```http
GET /api/backtests/ai-config/jobs/{job_id}
```

Terminal ready:

```json
{
  "job_id": "uuid",
  "status": "ready",
  "mode": "create",
  "assistant_message": "Я собрал валидный конфиг для BTCUSDT на 15m. Stop loss не добавлен, потому что вы его не просили; его можно добавить отдельным запросом.",
  "validated_config": {
    "coordinates": {"exchange": "binance", "market_type": "spot", "symbol": "BTCUSDT"},
    "timeframe": "15m",
    "time_range": {"start": "2023-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
    "indicators": [
      {"indicator_id": "momentum.rsi", "sources": ["close"], "window": {"start": 7, "stop": 28, "step": 7}}
    ],
    "risk": {"mode": "none"},
    "execution": {
      "direction_mode": "long_short_reversal",
      "fee_rate": 0.00075,
      "slippage_rate": 0.0001,
      "initial_cash_quote": 10000,
      "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10},
      "profit_lock": {"enabled": false},
      "close_on_end": true
    },
    "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
    "top_n": 100
  },
  "load_action": {"enabled": true, "label": "Загрузить конфигурацию"},
  "warnings": [],
  "suggestions": ["Добавить stop loss / take profit grid"]
}
```

Needs clarification:

```json
{
  "job_id": "uuid",
  "status": "needs_clarification",
  "assistant_message": "Я не нашел поддерживаемый индикатор Bollinger Bands в текущем каталоге. Уточните индикатор или выберите один из доступных: volatility.atr, volatility.stddev, volatility.hv.",
  "validated_config": null,
  "load_action": {"enabled": false},
  "validation_errors": [
    {"path": "indicators.0.indicator_id", "code": "unsupported_indicator"}
  ]
}
```

### SSE events

```http
GET /api/backtests/ai-config/jobs/{job_id}/events
```

Event names:

```text
queued
preparing_catalog
assembling_prompt
generating
validating_json
validating_business
repairing
ready
needs_clarification
failed
heartbeat
```

Example event:

```text
event: validating_business
data: {"job_id":"...","message":"Проверяю конфигурацию по правилам /backtests","progress":70}
```

Это observable stages, не reasoning trace.

## Model output contract

Модель возвращает один strict JSON object без Markdown:

```json
{
  "schema_version": 1,
  "mode": "create",
  "status": "config_ready",
  "assistant_message": "Короткий текст для пользователя.",
  "assumptions": ["Если пользователь не указал период, использован 2023 год."],
  "warnings": [],
  "config": {
    "coordinates": {"exchange": "binance", "market_type": "spot", "symbol": "BTCUSDT"},
    "timeframe": "15m",
    "time_range": {"start": "2023-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
    "indicators": [],
    "risk": {"mode": "none"},
    "execution": {},
    "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
    "top_n": 100
  },
  "suggestions": []
}
```

Backend не доверяет этому JSON напрямую. Он:

1. проверяет JSON Schema;
2. нормализует в existing `BacktestConfigDraftResponse` / backtest request shape;
3. вызывает business validation;
4. repair-ит один раз при ошибке;
5. только после этого отдает `validated_config` в UI.

## Prompt policy

Системный prompt должен быть versioned и hash-based:

```text
system_prompt_version: backtest-ai-configurator-v1
system_prompt_hash: sha256(...)
```

Минимальный policy:

```text
Ты Backtest AI Configurator для Roehub /backtests.
Разрешенная тема: только сбор, редактирование, объяснение и исправление конфигурации backtest.
Нельзя отвечать на общие вопросы, новости, программирование, инвестиционные советы вне конфигурации backtest, личные темы и любые запросы вне /backtests.
Нельзя использовать значения, которых нет в allowed catalog.
Нельзя выдумывать symbols, indicators, timeframes, risk modes, sizing modes.
Если пользователь просит unsupported значение, предложи ближайший supported вариант только если он есть в candidates.
Если valid config невозможен, верни status=needs_clarification и объясни, что уточнить.
Верни только JSON по заданной schema. Никакого Markdown.
Пользовательский язык ответа: русский или английский в соответствии с request locale.
```

Prompt injection defense не ограничивается prompt. Обязательные backend gates:

- pre-LLM domain gate для явно off-topic prompts;
- compact allowed catalog в prompt;
- schema validation;
- business validation;
- no load button без valid config;
- audit всех violations.

## Context window

MVP target:

```yaml
context_window: 8192
max_output_tokens: 900
max_input_tokens: 5500
repair_attempts: 1
temperature: 0.0
```

Budget:

```text
system + policy prompt:          ~800-1200 tokens
JSON schema summary:             ~700-1000 tokens
current config snapshot:         ~300-800 tokens
catalog candidates/subset:       ~1000-2200 tokens
user message + recent context:   ~300-1200 tokens
reserved completion:             900 tokens
safety margin:                   800-1200 tokens
```

Правило:

```text
max_input_tokens = min(5500, context_window - max_output_tokens - safety_margin)
```

Если quality низкое:

1. улучшать prompt examples и repair profile;
2. расширять relevant catalog subset;
3. тестировать более качественную MLX-модель;
4. только потом увеличивать context window.

Если latency/p95 высокий:

1. уменьшать catalog subset;
2. уменьшать max_output_tokens;
3. снижать active_generations;
4. выбирать меньшую MLX-модель;
5. увеличивать queue feedback вместо параллелизма без evidence.

## Лимиты и admission control

Существующий `PaidLevel` поддерживает `base|free|pro|ultra`. AI quota должна
быть отдельной от backtest job quota.

Quota windows, требуемые продуктом:

- requests per week;
- requests per 5 hours.

Начальные значения для настройки, не hard-coded product truth:

| Tier | Requests / 5h | Requests / week | Max queued per user | Max active user jobs |
| --- | ---: | ---: | ---: | ---: |
| `free` | 3 | 10 | 1 | 1 |
| `base` | 6 | 25 | 2 | 1 |
| `pro` | 15 | 75 | 3 | 1 |
| `ultra` | 40 | 200 | 5 | 1 |

Global MVP defaults:

```yaml
queue:
  max_queue_size: 50
  max_active_generations: 1
  request_timeout_sec: 90
  queue_timeout_sec: 180
  repair_attempts: 1
```

Concurrency policy для Mac Studio:

- стартовать с `max_active_generations=1` для любой новой модели;
- для 1-2B / E2B тестировать `2`, `4`, `6` active generations;
- для 4B тестировать `1`, `2`, затем только при хорошем memory pressure `3-4`;
- не ship-ить concurrency выше `1` без Mac Studio load evidence;
- при queue saturation UI получает friendly message и estimated wait.

Estimated wait formula для MVP:

```text
estimated_wait_seconds =
  ceil((queue_position / observed_completed_jobs_per_second) + current_generation_eta)
```

Если observed throughput еще нет, использовать conservative fallback из
config, например `60-90 sec`, и быстро корректировать после первых production
metrics.

## Validation и repair

Validation layers:

1. JSON parse;
2. JSON Schema;
3. allowed enum/catalog validation;
4. `BacktestPreflightService` business validation;
5. tier/guardrail limits;
6. final UI shape validation.

Repair loop:

```yaml
repair_attempts: 1
```

Repair prompt получает:

- исходный model draft;
- compact validation errors;
- allowed values для ошибочных paths;
- исходный user prompt;
- current config snapshot.

Если repair успешен, status `ready` и warnings объясняют correction. Если repair
неуспешен, status `needs_clarification`, без кнопки загрузки.

## Хранение данных для дообучения

Поскольку пользовательский prompt и model response должны сохраняться для
future fine-tuning, хранение является explicit product behavior.

Сохранять:

- raw user prompt;
- normalized prompt;
- mode and locale;
- current_config snapshot;
- catalog snapshot hash и compact catalog subset;
- system prompt version/hash;
- model_id and model_path_hash;
- raw LLM output;
- parsed draft;
- validation errors;
- repair prompt/output;
- final validated config;
- assistant message;
- applied feedback;
- latency/token estimates.

Не сохранять:

- secrets, tokens, API keys, env dumps;
- Tailscale IP/DNS в training rows;
- full runtime logs с private network topology.

Для будущего fine-tuning сделать отдельный export use case, который выбирает
только training-safe rows и помечает:

```text
quality_label: applied | rejected | clarification | repaired | failed_validation
```

Лучшие supervised samples для первого fine-tune:

- `applied=true`;
- `status=ready`;
- `repair_attempts=0`;
- низкая validation warning count;
- user did not immediately edit many fields after load.

## Observability

Structured log на job:

```json
{
  "event": "backtest_ai_config_job_completed",
  "job_id": "uuid",
  "owner_user_id": "uuid",
  "mode": "create",
  "locale": "ru",
  "tier": "pro",
  "model_id": "gemma_4_e2b_it_4bit_local",
  "queue_wait_ms": 4210,
  "llm_latency_ms": 8900,
  "validation_latency_ms": 120,
  "repair_attempts": 0,
  "prompt_tokens_est": 3800,
  "completion_tokens_est": 640,
  "status": "ready",
  "total_latency_ms": 14300
}
```

Metrics:

- `backtest_ai_config_jobs_total{status,mode,tier,model_id}`;
- `backtest_ai_config_queue_depth`;
- `backtest_ai_config_active_generations`;
- `backtest_ai_config_queue_wait_seconds`;
- `backtest_ai_config_llm_latency_seconds`;
- `backtest_ai_config_total_latency_seconds`;
- `backtest_ai_config_validation_failures_total{code}`;
- `backtest_ai_config_repair_attempts_total{result}`;
- `backtest_ai_config_quota_rejections_total{tier,window}`;
- `backtest_ai_config_capacity_rejections_total`;
- `backtest_ai_config_applied_total`;
- `backtest_ai_config_model_reload_total{result}`.

Mac Studio host metrics:

- MLX runtime RSS / physical footprint;
- `memory_pressure`;
- swap activity from `vm_stat`;
- CPU/GPU utilization where available;
- process restarts;
- Tailscale peer reachability from VPS to backend API.

## Benchmark и нагрузочное тестирование

Acceptance evidence должно сниматься на Mac Studio, потому что локальные tests
не отвечают на вопрос MLX throughput/memory pressure.

### Уровень 1: чистый inference benchmark

Цель: подобрать safe `active_generations`, `max_output_tokens`,
`context_window`, model size.

Матрица:

| Model class | Active generations | Context | Output |
| --- | --- | --- | --- |
| E2B / 1-2B 4bit | 1, 2, 4, 6 | 8192 | 900 |
| 3-4B 4bit | 1, 2, 3, 4 | 8192 | 900 |
| chosen default | accepted value | 8192 | 900 |

Метрики:

- time to first token, если runtime streaming используется;
- total generation latency;
- tokens/sec per request;
- aggregate tokens/sec;
- finish_reason;
- JSON parse success rate;
- RSS/physical footprint;
- memory pressure;
- swap deltas;
- process stability after 100+ generations.

### Уровень 2: API pipeline benchmark

Endpoint:

```text
POST /api/backtests/ai-config/jobs
GET  /api/backtests/ai-config/jobs/{job_id}
GET  /api/backtests/ai-config/jobs/{job_id}/events
```

Сценарии online users:

| Scenario | Users | Spawn | Think time | Run time | Expected behavior |
| --- | ---: | ---: | --- | --- | --- |
| S1 | 1 | 1/s | 5-20s | 10m | no queue, validates quality |
| S5 | 5 | 1/s | 20-90s | 15m | light concurrent real usage |
| S10 | 10 | 2/s | 30-120s | 20m | normal MVP target |
| S50 | 50 | 2/s | 60-180s | 30m | queue behavior and p95 |
| S100 | 100 | 2/s | 120-300s | 45m | overload UX, quota/capacity |

Важно: 50/100 users это online users, не 50/100 simultaneous generations.
Нагрузка должна имитировать чат: пользователь отправляет запрос раз в несколько
минут.

Prompt mix:

- create simple BTC config;
- create multi-symbol request where platform supports only one symbol per form;
- unsupported timeframe correction;
- unsupported indicator clarification;
- edit current config to add TP/SL;
- repair invalid current config;
- suggest safer config;
- off-topic prompt rejection.

Acceptance targets для MVP:

| Metric | Target |
| --- | ---: |
| final valid config rate for supported prompts | >= 98% |
| hallucinated symbol/indicator after validation | 0 |
| repair rate | < 25% |
| p50 total latency S10 | <= 10s |
| p95 total latency S10 | <= 30s |
| p95 queue wait S50 | reported accurately, no silent timeout |
| quota/capacity UI responses | 100% friendly message |
| memory pressure under S50 | no sustained swap growth |
| worker crash/restart recovery | queued/running jobs recover or terminal-fail deterministically |

### Уровень 3: soak test

Run:

```text
50 online users
2-6 hours
realistic prompt mix
chosen production model
chosen active_generations
```

Проверить:

- memory growth;
- MLX runtime reload behavior;
- queue depth recovery;
- stale leases;
- SSE disconnect/reconnect;
- DB table growth and indexes;
- training data row completeness.

### Harness

Не обязательно добавлять Locust как production dependency. Допустимые варианты:

- отдельный dev-only Locust environment;
- `scripts/backtest_ai/run_configurator_load_test.py` на `httpx.AsyncClient`;
- k6 снаружи Mac Studio.

Генератор нагрузки не должен жить на Mac Studio, чтобы не съедать ресурсы
inference host.

## Rollout plan

### Stage 0 - Contract freeze

- Создать этот документ как source of truth.
- Зафиксировать route names, storage tables, model config schema, prompt version.
- Проверить docs index.

### Stage 1 - Storage, quota, DTOs

- Добавить migrations для `backtest_ai_config_*` tables.
- Добавить application DTOs and repositories.
- Добавить quota service для 5h/week windows по `PaidLevel`.
- Tests: repository, quota, owner scope.

### Stage 2 - API shell + fake worker

- Добавить routes `POST/GET/SSE/feedback`.
- Worker пока fake deterministic, без MLX.
- UI получает job statuses и может отрисовать friendly high-load/quota states.
- Tests: API DTO shape, auth/owner scope, rate limit response, SSE smoke.

### Stage 3 - Catalog resolver + validators

- Подключить `BacktestRuntimeDefaultsService`, `configs/prod/indicators.yaml`,
  artifact symbols.
- Добавить schema validator и business validation через `BacktestPreflightService`.
- Tests: unsupported symbol/indicator/timeframe, correction/clarification.

### Stage 4 - MLX runtime adapter

- Добавить model registry config.
- Добавить `MLXOpenAICompatibleAdapter`.
- Добавить launchd target для internal `mlx_lm.server`.
- Добавить worker process `backtest-ai-configurator-worker`.
- Smoke: one prompt -> valid config on Mac Studio.

### Stage 5 - Prompt profiles + repair loop

- Добавить versioned system prompt.
- Добавить generate profile, repair profile, explain profile.
- Добавить raw attempt audit.
- Tests: parse failure, schema failure, business validation failure, repair once.

### Stage 6 - Web UI integration

- Включить AI block через `ai_configurator_state.enabled=true`.
- Добавить chat input, mode handling, SSE status timeline, typewriter effect.
- Добавить `Load configuration` action that fills current form.
- Browser QA: ru/en locale, console/network clean, no raw codes.

### Stage 7 - Observability and training export

- Metrics/logging.
- Admin-safe training export command/view.
- Scrub checks for secrets and private infra fields.

### Stage 8 - Mac Studio load evidence

- Run S1/S5/S10/S50/S100.
- Pick accepted model, active generation count and queue limits.
- Record benchmark summary under `docs/architecture/backtest/benchmark_iterations/`.

### Stage 9 - Production rollout

- Feature flag default off.
- Enable for admin/internal users.
- Watch metrics, quota, memory, validation quality.
- Rollout to paid tiers by config.
- Rollback: disable feature flag, stop worker, keep existing `/backtests` form.

## Контрактное влияние

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API | compatible-change | Additive `/backtests/ai-config/*` endpoints. Existing `/backtests/jobs` unchanged. |
| Browser-visible behavior | compatible-change | Placeholder AI block becomes enabled; no auto job creation. |
| DTO schema | compatible-change | New DTOs. Existing workstation response may flip `ai_configurator_state.enabled` to true. |
| Persisted schema | compatible-change | New tables only. No change to `backtest_jobs` request hash semantics. |
| Config schema | compatible-change | New `backtest_ai_configurator` config section. |
| Request hash / cache identity | none | AI config is applied to form; final backtest job hash remains produced by existing backtest create flow. |
| Runtime workflow | compatible-change | New worker/runtime processes; existing API/web still operate if disabled. |
| Performance risk | unknown until benchmark | MLX latency/concurrency must be accepted on Mac Studio evidence. |

## Связанные файлы

- `apps/web/templates/pages/backtests.html` - AI block and current form controls.
- `apps/web/dist/js/pages/backtests.js` - current form state, request payload,
  job/result UI.
- `apps/api/routes/ui_backtests.py` - workstation read model route.
- `apps/api/wiring/modules/ui_backtests.py` - `ai_configurator_state`,
  `instrument_universe`, `indicator_catalog`.
- `apps/api/routes/backtests.py` - existing runtime defaults, preflight and jobs API.
- `src/trading/contexts/backtest/application/services/v2/preflight.py` -
  runtime defaults and business validation source.
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py` -
  public defaults and guardrails DTOs.
- `configs/prod/indicators.yaml` - supported indicator catalog.
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` -
  canonical backtest service contract.
- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md` -
  existing Mac Studio worker/queue production plan.
- `docs/runbooks/mac-studio-native-backend-operations.md` - native operations
  and service reload context.

## Как проверить реализацию

Focused backend gates:

```bash
uv run ruff check apps/api src/trading/contexts/backtest tests/unit/apps/api
uv run pyright
uv run pytest -q tests/unit/apps/api/test_backtests_routes.py
uv run pytest -q tests/unit/apps/api/test_ui_backtests_routes.py
uv run pytest -q tests/unit/apps/web/test_app_routes.py
```

New expected tests:

```text
tests/unit/contexts/backtest/application/ai_configurator/
tests/unit/apps/api/test_backtest_ai_config_routes.py
tests/unit/apps/web/test_backtests_ai_configurator.py
```

Docs:

```bash
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Runtime/Mac Studio:

```bash
tailscale status --self --json
memory_pressure
vm_stat
```

Load evidence:

```text
S1, S5, S10, S50, S100 benchmark summaries recorded with:
- commit sha
- model_id
- model_path hash
- active_generations
- context_window
- max_output_tokens
- Mac Studio memory pressure
- queue p50/p95
- total latency p50/p95/p99
- final valid config rate
```

## Риски и решения

- Риск: модель уверенно выдает unsupported config.
  Решение: no load button before business validation; repair once; then
  clarification only.

- Риск: prompt injection переводит модель на другие темы.
  Решение: pre-LLM domain gate, hard system prompt, strict JSON schema,
  no free-form actions.

- Риск: 100 online users создают long queue.
  Решение: per-tier weekly/5h quota, queue size limit, estimated wait and
  friendly capacity state.

- Риск: Mac Studio начинает swap.
  Решение: conservative `active_generations=1` until benchmark, memory pressure
  gate before raising concurrency.

- Риск: training data содержит private infra or secrets.
  Решение: no env dumps, no Tailnet details, export scrubber, explicit
  training-safe rows.

- Риск: второй repair model усложняет orchestration.
  Решение: v1 uses same model instance with repair prompt profile; standby same
  model only for runtime availability.

- Риск: UI показывает internal reasoning.
  Решение: only observable stages and final assistant message, no chain-of-thought.

## Открытые параметры перед implementation

Эти значения должны быть конфигом, а не кодом:

- exact tier quota values;
- chosen MLX model path;
- accepted `active_generations`;
- queue timeout;
- model reload procedure;
- prompt examples included in system/developer prompt;
- data retention period for raw training rows.

Все они могут стартовать с defaults из этого документа, но production acceptance
должен зафиксировать фактические Mac Studio benchmark values.

## Источники

- `mlx-lm` HTTP server docs: `https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md`
- Apple Open Source MLX project page: `https://opensource.apple.com/projects/mlx/`
