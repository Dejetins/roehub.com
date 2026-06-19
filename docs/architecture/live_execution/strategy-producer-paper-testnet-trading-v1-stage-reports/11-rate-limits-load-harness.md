# Stage 11: Rate Limits And Load Harness

Статус: `in_progress`.

Дата старта: `2026-06-20`.

## Pre-Start

User required before start: nothing.

Stage `10` gate: `accepted` в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; Stage `11` may start. Blockers: `none`; Next stage allowed: `yes`.

## Scope

Stage `11` добавляет контролируемый load harness для десятков/сотен `testnet`-mode strategies. Цель не в том, чтобы нагружать Binance/Bybit testnet endpoints, а в том, чтобы доказать через accepted `live_execution` boundary, что Roehub:

- создает `strategy_signal` source events и accepted risk intents для `testnet`;
- публикует execution requests через dispatch/backpressure policy;
- обрабатывает их в `exchange-execution` через controlled testnet adapter;
- применяет per-exchange limiter перед adapter calls;
- измеряет queue lag, limiter waits, submit/status/cancel latency, ack/fill/reconciliation, retry/DLQ counters, CPU/RSS.

Acceptance run использует controlled adapter в `testnet` environment, без реальных provider HTTP requests. Это сохраняет форму Stage `09` money boundary и не создает риск DDoS testnet endpoints. Реальные exchange credentials, chat secrets, raw provider payloads, cookies, tokens, signed payloads и ciphertext в harness/report не используются и не записываются.

## Concrete File List Before Edits

Prompt указал broad paths. До implementation scope сужен до:

| Path | Planned action | Reason |
|---|---:|---|
| `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` | no planned edit | Existing Stage `06`/`10` producer metrics already expose source-event latency; harness records `strategy_signal` source events directly through `live_execution` to avoid broad producer runtime rewiring. |
| `apps/exchange_execution/main/app.py` | modify | Expose limiter wait metrics from `ExchangeExecutionProcessService` in `/metrics`. |
| `apps/exchange_execution/load_harness.py` | create | Controlled testnet-mode strategy load harness and CLI. |
| `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | modify | Enforce configured per-exchange rate limiter before adapter calls and emit wait observations. |
| `infra/macos/prometheus/rules/live-execution-stage17.rules.yml` | modify | Add operational alerts for sustained limiter wait and high submit latency. |
| `tests/unit/apps/exchange_execution/test_app.py` | modify | Lock metric exposure/wiring. |
| `tests/unit/apps/exchange_execution/test_load_harness.py` | create | Lock harness summary, thresholds, mode mix, retry/DLQ probe behavior, and no-mainnet invariant. |
| `tests/unit/contexts/live_execution/test_exchange_execution_process.py` | modify | Lock limiter wait enforcement/callback in the exchange-execution use case. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/11-rate-limits-load-harness.md` | create/update | Stage report, thresholds, evidence, manifest, handoff. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage status, evidence, publish/deploy handoff. |
| `docs/architecture/README.md` | check/update if generated index changes | Required docs index gate after Markdown changes. |

Files outside prompt expected paths: none planned.

## Pass / Fail Thresholds

Thresholds are defined before interpreting load results.

| Metric | Pass threshold |
|---|---:|
| Testnet-mode strategy count | `>= 120` in acceptance run. |
| Mode mix | `testnet=100%`; `paper` may appear only in supporting runs, not acceptance. |
| Mainnet submits | `0`. |
| Main load submitted orders | `submitted_count == strategy_count`. |
| Main load guard/adapter/quarantine failures | `0`. |
| Main load retry/DLQ | `retry_count=0`, `dlq_count=0`. |
| Controlled backpressure probe | `retry_count >= 1`, `request_count=0`, no order side effect. |
| Controlled retry-budget probe | `dlq_count <= 1`, no order side effect, proves bounded retry exhaustion. |
| Redis pending after cleanup | `0`. |
| Redis max pending during batches | `<= exchange read_count`. |
| Queue lag | `p95 <= 15s`, `p99 <= 20s`. |
| Signal-to-source latency | `p99 <= 250ms`. |
| Source-to-intent latency | `p99 <= 250ms`. |
| Risk latency | `p99 <= 50ms`. |
| Dispatch latency | `p99 <= 500ms`. |
| Controlled adapter submit/status/cancel latency | `p99 <= 25ms` per operation, excluding limiter wait. |
| Limiter wait evidence | `total_wait_seconds > 0`, `p99_wait <= 250ms`, waits recorded per exchange. |
| Exchange process ack/fill latency | `p95 <= 15s`, `p99 <= 20s`. |
| Fills/reconciliation | `reconciliation pending=0`; fill count may be `0` for controlled cancel path, but status/cancel rows must be recorded. |
| CPU | `process_cpu_seconds <= 20s` for acceptance run. |
| Memory | `max_rss_delta_mb <= 256MB` where RSS is available. |
| Secret leakage | Report contains no raw secrets, cookies, tokens, ciphertext, signed payloads, provider credential values, or raw provider payloads. |

## Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `none` | No public HTTP API payload or route changes. |
| Port contract | `compatible-change` | `ExchangeExecutionProcessService` gains optional limiter wait callback/injected timing helpers; existing callers keep working. |
| DTO schema | `none` | No DTO change. |
| Persisted schema | `none` | No migration or table shape change. |
| Config schema | `none` | Existing `rate_limit.per_second` and `rate_limit.burst` config becomes enforced; schema/defaults unchanged. |
| Request hash / cache / persistence identity | `none` | Idempotency hashes, client order id derivation, dispatch ids and persisted identities unchanged. |
| Service-call timeout/retry/error semantics | `compatible-change` | Adapter calls may wait before submit/status/cancel/private-stream/clock calls to respect configured per-exchange limiter; no blind retry behavior added. |
| External side effects | `compatible-change` | Real testnet/mainnet side-effect safety improves by throttling; harness uses controlled adapter only. |
| Logs/metrics/audit/redaction | `compatible-change` | Adds limiter wait metrics and stage report evidence; no sensitive values. |
| Alerts/runbooks | `compatible-change` | Adds Prometheus alert expressions for sustained limiter wait/high submit latency. |
| Benchmark/rollout gate | `compatible-change` | Stage `11` adds a reusable controlled benchmark gate and thresholds. |
| Browser-visible behavior | `none` | No UI/browser behavior changed. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/11-rate-limits-load-harness.md` | none | none | Stage `11` report and required pre-start/scope/threshold record. | none: documentation artifact. |
| `apps/exchange_execution/load_harness.py` | none | none | Controlled `testnet`-mode load harness exercising source event, risk, dispatch, exchange-execution, limiter, controlled adapter, reconciliation, and threshold reporting. | compatible-change: new opt-in operational benchmark CLI; no production default change. |
| none | `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | none | Enforce existing per-exchange `rate_limit` config before adapter clock/private-stream/submit/status/cancel calls and emit wait observations. | compatible-change: existing config is now enforced; adapter calls may wait instead of overrun. |
| none | `apps/exchange_execution/main/app.py` | none | Expose limiter wait counter/histogram through exchange-execution Prometheus metrics. | compatible-change: additive metrics. |
| none | `infra/macos/prometheus/rules/live-execution-stage17.rules.yml` | none | Add warning alerts for sustained limiter waits and high native adapter latency. | compatible-change: additive alert/runbook triggers. |
| `tests/unit/apps/exchange_execution/test_load_harness.py` | none | none | Focused harness regression for pass/fail summary, mode mix, limiter waits, backpressure probe, retry-budget probe, and no main DLQ/retry. | none: test-only. |
| none | `tests/unit/apps/exchange_execution/test_app.py`; `tests/unit/contexts/live_execution/test_exchange_execution_process.py` | none | Lock metric wiring and limiter enforcement with deterministic fake clock/sleeper. | none: test-only. |
| none | `tests/unit/infra/test_monitoring_assets.py` | none | Keep the repo-managed Prometheus alert inventory aligned with the two additive Stage `11` rules. This is outside the prompt's possible secondary paths but directly required by the expected `infra/macos/prometheus` touch. | none: test-only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; `docs/architecture/README.md` | none | Planned final stage handoff and docs index after validation. | none: documentation handoff. |

Files outside prompt expected paths: `tests/unit/infra/test_monitoring_assets.py`, justified because the stage intentionally adds Prometheus rules under an expected primary path and the repository locks that managed alert inventory in the infra test suite.

## Validation Log

| Check | Result | Evidence |
|---|---:|---|
| Stage `10` accepted gate | passed | Ledger row marks Stage `10` `accepted`, blockers `none`, next stage allowed `yes`. |
| Focused tests | passed | `uv run pytest -q tests/unit/apps/exchange_execution/test_load_harness.py tests/unit/apps/exchange_execution/test_app.py tests/unit/contexts/live_execution/test_exchange_execution_process.py` -> `18 passed`. |
| Focused lint | passed | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/apps/exchange_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py` -> `All checks passed!`. |
| Focused pyright | passed | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/apps/exchange_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py` -> `0 errors`. |
| Controlled load run | passed | `uv run python -m apps.exchange_execution.load_harness --strategies 120 --read-count 25 --rate-limit-per-second 60 --rate-limit-burst 20 --pretty` -> `passed=true`, `violations=[]`. |
| Prompt ruff gate | passed | `uv run ruff check apps src/trading/contexts/live_execution tests` -> `All checks passed!`. |
| Prompt pyright gate | passed | `uv run pyright apps src/trading/contexts/live_execution tests` -> `0 errors`. |
| Prompt unit gate | passed | `uv run pytest -q tests/unit/apps tests/unit/contexts/live_execution` -> `391 passed, 3 warnings`; warnings are existing `httpx` deprecations in web route tests. |
| Docs index gate | passed | `python -m tools.docs.generate_docs_index --check` -> `docs/architecture/README.md is up-to-date`. |
| Broad pre-publish ruff | passed | `uv run ruff check .` -> `All checks passed!`. |
| Broad pre-publish pyright | passed | `uv run pyright` -> `0 errors`. |
| Broad pre-publish pytest | passed after test inventory repair | First run failed only because `tests/unit/infra/test_monitoring_assets.py` intentionally enumerated Prometheus alert names and did not yet include the two new Stage `11` alerts. After updating the managed-alert inventory test and the non-testnet harness invariant, `uv run pytest -q -ra` -> `1257 passed, 3 warnings`. |

## Load Evidence

Command:

```bash
uv run python -m apps.exchange_execution.load_harness --strategies 120 --read-count 25 --rate-limit-per-second 60 --rate-limit-burst 20 --pretty
```

Run result: `passed=true`, `violations=[]`, duration `9.353047s`.

| Metric | Result | Threshold | Decision |
|---|---:|---:|---|
| Testnet strategy count | `120` | `>=120` | pass |
| Mode mix | `testnet=120`, `paper=0` | `testnet=100%` | pass |
| Requests / submitted / acked | `120 / 120 / 120` | `submitted == strategy_count` | pass |
| Main guard / adapter / quarantined | `0 / 0 / 0` | `0` | pass |
| Main retry / DLQ | `0 / 0` | `0 / 0` | pass |
| Orders by exchange | `binance=60`, `bybit=60` | balanced representative mix | pass |
| Orders by environment | `testnet=120` | no mainnet | pass |
| Fills / reconciliation | `fills=120`, `matched=120`, `pending=0` | pending `0` | pass |
| Redis pending final / max pending | `0 / 25` | final `0`, max `<=25` | pass |
| Queue lag | p95 `8843.307ms`, p99 `9241.272ms` | p95 `<=15000ms`, p99 `<=20000ms` | pass |
| Signal-to-source | p99 `0.051ms` | `<=250ms` | pass |
| Source-to-intent | p99 `0.083ms` | `<=250ms` | pass |
| Risk | p99 `0.004ms` | `<=50ms` | pass |
| Dispatch | p99 `0.032ms` | `<=500ms` | pass |
| Controlled adapter submit/status/cancel | p99 `2ms / 1ms / 1ms` | `<=25ms` | pass |
| Limiter waits | total `7.435842s`, p99 `16.052ms`, max `16.470ms` | total `>0`, p99 `<=250ms` | pass |
| Limiter waits by exchange | `binance=3.720955s`, `bybit=3.714887s` | per-exchange evidence | pass |
| CPU / RSS delta | `0.117934s` / `1.28125MB` | `<=20s` / `<=256MB` | pass |

Controlled probes:

| Probe | Result | Evidence |
|---|---:|---|
| Backpressure | pass | `result=retry`, `reason=dispatch_backpressure`, `retry_count=1`, `request_count=0`, `dlq_count=0`. |
| Retry budget | pass | `result=dlq`, `reason=retry_budget_exhausted`, `request_count=0`, `dlq_count=0`, no order side effect. |

Safety notes:

- No real Binance/Bybit HTTP requests were sent by this load run.
- No mainnet environment or mainnet adapter path was created.
- No raw secrets, cookies, tokens, ciphertext, signed payloads, provider credential values, or raw provider payloads were read or written.
- `paper` was not used as acceptance evidence.
- Controlled in-memory state drained to Redis pending `0`; no cleanup of real orders/positions was applicable.

## Publish / Deploy

Pending direct `main` delivery. Stage must not be marked `accepted` until CI/deploy evidence, Mac Studio checkout sync, runtime smoke, and branch cleanup evidence are recorded.

## Blockers

| Blocker | Severity | Owner / next action | Acceptance impact |
|---|---|---|---|
| main delivery, CI/deploy, Mac Studio host sync/runtime smoke pending | blocker | Complete direct-main publish/deploy, sync Mac Studio checkout, run runtime smoke, and update this report/ledger before acceptance. | Stage remains `in_progress`; Stage `12` may not start. |

## Handoff

Local implementation and validation are complete. Next handoff is direct `main` delivery plus CI/deploy/Mac Studio runtime proof before acceptance.
