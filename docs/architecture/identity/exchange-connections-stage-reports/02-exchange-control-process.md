# Stage 2: Exchange-control process и service identity

Дата проверки: 2026-05-24.

Статус: blocked on target-runtime supervision evidence. Local-dev runtime, tests,
health и metrics приняты; Mac Studio Prometheus/Monit/restart evidence не получен
в этой среде.

## Scope

Stage 2 добавляет минимальный production-shaped runtime boundary для
`exchange-control` до любых реальных Binance/Bybit validation calls.

Изменено:

- добавлен отдельный `apps.exchange_control` FastAPI runtime entrypoint;
- service identity жестко зафиксирована как `exchange-control`;
- `GET /health/ready` возвращает readiness payload на `127.0.0.1:9205`;
- `/metrics` экспортирует `exchange_control_active` и secret-safe
  `exchange_connection_*` series;
- production config fail-fast требует `127.0.0.1:9205` и disabled real exchange
  validation;
- добавлены Prometheus, launchd и Monit configs;
- monitoring runbook обновлен под `roehub_exchange_control`.

## Service Identity

| Contract | Expected result | Observed result | Blocker |
|---|---|---|---|
| Service identity | Runtime principal name is exactly `exchange-control`. | `ExchangeControlServiceIdentity(name="exchange-control")` accepted; any other name raises `ValueError`. | None |
| Stage 3 Transit ACL handoff | Stage 3 can bind Transit policy to service identity `exchange-control`. | Identity is exported from `src/trading/contexts/exchange_control/application/service_identity.py`. | None |
| Production bind contract | Prod runtime uses only `127.0.0.1:9205`. | `ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "prod"})` resolves host `127.0.0.1`, port `9205`; port override `9206` fails. | None |
| Real exchange validation | Real exchange validation remains disabled until Stage 5. | `ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED=true` fails config validation. | None |

## Health Evidence

| Endpoint | Command | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| `GET /health/ready` | `curl -fsS http://127.0.0.1:9205/health/ready` | Ready response with `service_identity="exchange-control"`. | Passed locally: `{"status":"ready","service":"exchange-control","service_identity":"exchange-control","checks":[{"name":"service_identity","status":"ready"},{"name":"external_exchange_validation","status":"ready"}]}`. | None |

## Metrics Evidence

| Endpoint | Command | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| `/metrics` | `curl -fsS http://127.0.0.1:9205/metrics \| rg 'exchange_control_active\|exchange_connection_'` | Metrics endpoint exposes `exchange_control_active` and secret-safe `exchange_connection_*` series. | Passed locally: `exchange_control_active 1.0`, `exchange_connection_validation_total{exchange="none",reason="stage_2_no_real_exchange_calls",result="disabled"} 0.0`, `exchange_connection_status{exchange="none",status="validation_disabled"} 0.0`. | None |
| Secret-safe labels | `rg -n "binance\|bybit\|ccxt\|pybit\|Client\\(\|requests\\.\|httpx\\.\|aiohttp\|api_key\|api_secret\|decrypt" apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control \|\| true` | No runtime exchange adapters, no credential/decrypt path, no raw secret labels. | No runtime matches; only test assertions checked that `api_key` and `connection_id` are absent from `/metrics`. | None |

## Prometheus Evidence

| Config / command | Expected result | Observed result | Blocker |
|---|---|---|---|
| `infra/macos/prometheus/prometheus.prod.yml` | Scrape job `exchange-control` targets `127.0.0.1:9205`. | Config contains `job_name: exchange-control` with target `127.0.0.1:9205`. | None |
| `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="exchange-control"}'` | Prometheus query returns target sample for `up{job="exchange-control"}` on target runtime. | Failed locally: `curl: (7) Failed to connect to 127.0.0.1 port 9090`. | Prometheus is not running in this local checkout/runtime, so target-runtime scrape evidence is blocked. |

## Monit И Restart Evidence

| Config / command | Expected result | Observed result | Blocker |
|---|---|---|---|
| `plutil -lint infra/macos/launchd/com.roehub.exchange-control.plist` | launchd plist is valid. | Passed: `OK`. | None |
| `infra/scripts/monit/roehub-exchange-control.monitrc` | Monit service name is `roehub_exchange_control`; start/stop goes through `launchctl_service_control.sh`. | Config present with health and metrics probes on port `9205`. | None |
| `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary \| rg 'roehub_exchange_control'` | Monit sees `roehub_exchange_control` as `Running/Accessible` on target runtime. | Failed locally: `/opt/homebrew/opt/monit/bin/monit` is absent. | Monit is not installed/configured in this local environment. |
| Controlled restart | Monit or launchd restart succeeds and `/health/ready` returns ready afterward. | Not executed against production launchd/Monit because target supervision is unavailable here and `scripts/macos/reload_launchd_services.sh prod` would restart unrelated local services. | Stage 2 remains blocked for Mac Studio acceptance until deployed supervision is installed and restart evidence is captured. |

## Verification Commands

| Command | Outcome |
|---|---|
| `gh --version && gh auth status` | Passed: `gh version 2.85.0`; authenticated to `github.com` as `Dejetins`. |
| `uv run pytest -q tests/unit/contexts/exchange_control` | Passed: `4 passed in 0.29s`. |
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed: `152 passed in 7.57s`. |
| `uv run ruff check apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed. |
| `uv run pyright apps/api apps/exchange_control src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed: `0 errors, 0 warnings, 0 informations`; pyright reported an available version update only. |
| `python -m tools.docs.generate_docs_index --check` | Initial check failed because `docs/architecture/README.md` needed the Stage 2 report entry; `python -m tools.docs.generate_docs_index` updated it; rerun passed. |
| `curl -fsS http://127.0.0.1:9205/health/ready` | Passed against local `uv run python -m apps.exchange_control.main.main --host 127.0.0.1 --port 9205`. |
| `curl -fsS http://127.0.0.1:9205/metrics \| rg 'exchange_control_active\|exchange_connection_'` | Passed against local runtime. |
| `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="exchange-control"}'` | Blocked locally: Prometheus was not listening on `127.0.0.1:9090`. |
| `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary \| rg 'roehub_exchange_control'` | Blocked locally: Monit binary path is absent. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `none` | Existing `apps/api` HTTP routes and DTO payloads are unchanged. |
| Operational HTTP contract | `compatible-change` | New internal service exposes additive `GET /health/ready` and `/metrics` on `127.0.0.1:9205`. |
| Port contract | `compatible-change` | New runtime principal `exchange-control` becomes the required Stage 3/5 boundary. |
| DTO schema | `none` | No persisted or public DTO schema is changed. |
| Persisted schema | `none` | No database migration or table shape is changed in Stage 2. |
| Config schema | `compatible-change` | New operational env vars are introduced: `ROEHUB_EXCHANGE_CONTROL_SERVICE_IDENTITY`, `ROEHUB_EXCHANGE_CONTROL_BIND_HOST`, `ROEHUB_EXCHANGE_CONTROL_METRICS_PORT`, `ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED`. |
| Cache/request identity | `none` | No cache key, request hash, idempotency key, or persistence identity behavior changes. |
| Ops / rollout gate | `compatible-change` | New Prometheus job, launchd plist, Monit config and runbook checks are added; Stage 3 cannot proceed until target-runtime restart evidence is captured. |

## Stage 3 Handoff Facts

- Stage 3 Transit ACL design must use service identity `exchange-control`.
- The process contract is `127.0.0.1:9205`, `GET /health/ready`, `/metrics`.
- `exchange_control_active` is the mandatory liveness metric.
- `exchange_connection_validation_total` and `exchange_connection_status` are
  present with bounded labels and currently encode disabled validation only.
- Real exchange validation, decrypt, Binance and Bybit calls are absent from the
  Stage 2 runtime.
- Stage 3 is blocked until Mac Studio Prometheus target evidence, Monit summary
  evidence and controlled restart evidence are captured after deploy/install.
