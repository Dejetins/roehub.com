# Web UI v1 -- Tests + Docs Index (WEB-EPIC-07)

Документ фиксирует контракт WEB-EPIC-07: минимальный набор unit/smoke тестов для Milestone 6
(Web UI + gateway + market-data reference API) и правила поддержки docs index.

## Цель

- Закрепить Milestone 6 контракт тестами (минимально), без внешних сервисов.
- Гарантировать, что docs index (`docs/architecture/README.md`) всегда актуален.

## Контекст

Milestone 6 добавил:

- `apps/web` (SSR + HTMX) + auth UX.
- same-origin gateway (Nginx) с `/api/*` префиксом и `/assets/*` статикой.
- Market-data reference API endpoints для UI (`/market-data/*`).
- Strategy/backtest/backtest-jobs UI flows, которые работают browser-side через `/api/*`.

Нам нужно:

- иметь smoke-level уверенность в том, что ключевые страницы под login gate и содержат нужные
  "hooks" для browser JS,
- иметь unit-level уверенность, что reference endpoints защищены auth и детерминированы,
- проверять, что инфраструктурные контракты delivery (nginx strip `/api`, compose profile `ui`)
  не дрейфуют,
- не ломать CI из-за не обновленного индекса docs.

## Scope

### 1) Unit tests: API reference endpoints

Покрываем контракты `GET /market-data/markets` и `GET /market-data/instruments`:

- auth-only (401 без principal)
- deterministic mapping/payload
- `q`/`limit` semantics

### 2) Smoke-level web route tests (без внешних сервисов)

Покрываем минимально:

- login gate для protected pages (401 -> redirect на `/login?next=...`)
- базовый SSR render страниц и наличие:
  - data-* hooks,
  - JS entrypoints (`/assets/*.js`),
  - literal `/api/*` paths

Цель этих smoke тестов:

- убедиться, что web app реально "прошивает" browser-side flow в нужные API пути,
- исключить регрессию open-redirect guard.

### 3) Docs index

- Любое добавление/изменение `.md` в `docs/**` требует обновления индекса:
  - `python -m tools.docs.generate_docs_index`
- CI должен проверять отсутствие drift:
  - `python -m tools.docs.generate_docs_index --check`

## Non-goals

- Полные e2e UI тесты (Playwright) в v1.
- Интеграционные тесты с реальным Postgres/ClickHouse/Nginx.

## Implementation notes (как именно фиксируем контракт)

### 1) Тесты должны быть детерминированными и без внешних сервисов

- Используем FastAPI `TestClient`.
- Для web login gate подменяем internal current-user API adapter в `app.state`.
- Для API routes используем фейковые use-cases и deterministic auth dependency stubs.

### 2) Infra smoke tests допускаются (как усиление Milestone 6)

Хотя в scope EPIC-07 формально входят только API reference и web smoke, мы допускаем
добавление shape assertions для:

- Nginx gateway config (наличие `location ^~ /api/` и strip semantics).
- Docker compose `ui` profile (наличие сервисов и port publishing только у gateway).

Причина:

- эти контракты напрямую влияют на работоспособность UI в браузере.

## Связанные файлы (реализация/тесты)

API reference tests:
- `tests/unit/apps/api/test_market_data_reference_routes.py`
- `tests/unit/apps/api/test_market_data_reference_wiring_module.py`

Web smoke tests:
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_security.py`
- `tests/unit/apps/web/test_api_client.py`

Infra smoke tests (optional усиление):
- `tests/unit/infra/test_gateway_nginx_config.py`
- `tests/unit/infra/test_ui_compose_profile.py`
- `tests/unit/apps/migrations/test_bootstrap_decisions.py`

Docs index:
- `docs/architecture/README.md`
- `tools/docs/generate_docs_index.py`

## Quality gates

Локально:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

CI:

- `.github/workflows/ci.yml` выполняет:
  - migrations fail-fast
  - ruff
  - pyright
  - docs index `--check`
  - pytest

## Риски / что дальше

- Unit/smoke тесты не заменяют e2e браузерные проверки (настоящий Telegram login widget,
  gateway routing, реальные cookies). В v1 это принимаемый риск.
- Если понадобится повысить уверенность, следующий шаг — добавить Playwright smoke (login bypass
  в dev или тестовый bot) как отдельный EPIC.
