# Stage 11: Source-Aware Risk Gate

Stage 11 adds a producer-neutral risk gate to execution ingress. Every
persisted `ExecutionIntent` now receives a durable `accepted` or `rejected`
risk result before any future Redis dispatch stage can see it. Missing risk
state fails closed as `risk_state_unavailable`.

Date: 2026-05-31.

Status: accepted. Direct-main delivery, CI/deploy and Mac Studio
post-deploy runtime evidence are complete.

## Scope

Included:

- source-aware risk evaluation for `strategy_signal`, `manual_request`,
  `ml_agent_decision` and `ops_test`;
- stable accepted/rejected reason codes for connection, custody, source policy,
  compatibility, market data, config guard, account projection, position
  ownership, capital reservation, paper accounting, profile/run/binding,
  recent-auth, ML policy, kill switch, environment and size/limit checks;
- durable intent state transition to `status=accepted|rejected`,
  `risk_status=accepted|rejected` and `risk_reason=<stable code>`;
- additive `execution_risk_audit_events` table with bounded metadata
  `dispatch=no-dispatch`;
- authenticated API request support for a bounded risk context snapshot;
- bounded API metrics:
  `execution_risk_gate_total{source_type,result,reason}` and
  `execution_risk_gate_latency_seconds{source_type,result}`;
- focused domain, API-route and migration tests.

Out of scope:

- no Redis `execution.requests.v1` dispatch;
- no exchange-execution process, adapter call, credential decrypt or order
  submit;
- no mainnet order submission;
- no browser-visible surface change;
- no direct exchange SDK/API/secrets access from Strategy, ML, browser UI or
  API producers.

Deviation:

- Stage 11 currently evaluates a bounded risk context snapshot supplied at the
  execution API boundary. The fail-closed default is `risk_state_unavailable`.
  This preserves the dispatch safety boundary but does not yet wire every prior
  readiness table into an automatic assembler. The next implementation stage
  must replace diagnostic snapshot input with repository-backed assembly before
  opening dispatch beyond controlled probes.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused tests | `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_source_aware_risk_gate_sql.py` | `19 passed`; rerun after expanded source-aware cases: `23 passed`. |
| Required ruff | `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy src/trading/contexts/identity apps tests` | Passed. |
| Required pyright | `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy src/trading/contexts/identity apps tests` | `0 errors`. |
| Required unit scope | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/strategy tests/unit/contexts/identity tests/unit/apps` | `407 passed, 3 warnings`. |

## Runtime Boundary Evidence

Local API boundary proof used a real FastAPI router, request validation,
Roehub error handlers, API metrics middleware and the in-memory execution
repository. This proves API/DTO/use-case/persistence-port behavior without
Postgres or Redis services on the workstation.

| Surface | Evidence | Result |
|---|---|---|
| API accepted path | Real `POST /ui/execution/source-events` + `POST /ui/execution/intents` for `strategy_signal` with all risk checks true returned HTTP `201`, `risk_status=accepted`, `risk_reason=risk_gate_accepted`. | Pass. |
| API rejected paths | Real calls returned durable rejected intents for incompatible variant, missing feed, stale feed, config mismatch, stale account projection, insufficient capital, ownership conflict, inactive connection, blocked profile, inactive run, missing binding, missing manual recent-auth, ML policy missing and kill switch closed. | Pass. |
| Unsupported order model | Real `POST /ui/execution/intents` with `take_profit` returned HTTP `422`, code `execution.unsupported_order_model`, reason `tp_sl_not_supported`; no accepted intent was produced. | Pass. |
| Durable state | In-memory repository after the API probe contained `15` intents and `15` risk audit events: `1` accepted, `14` rejected. | Pass. |
| No dispatch | Every risk audit event included bounded metadata `dispatch=no-dispatch`; no Redis publisher exists in Stage 11 wiring. | Pass. |
| Metrics | `/metrics` exposed `execution_risk_gate_total` and `execution_risk_gate_latency_seconds`; unsupported-order metric still recorded `tp_sl_not_supported`. | Pass. |

Boundary command summary:

```text
api_cases 15 accepted 1 rejected 14
db_intents 15 audit_events 15 no_dispatch True
unsupported_order_model 422 tp_sl_not_supported
sample_reasons risk_gate_accepted,strategy_variant_incompatible,market_data_missing,market_data_stale,exchange_config_mismatch
metrics_present True True
```

Mac Studio production proof used the deployed API, persisted Roehub session
auth, Postgres, Redis and the live `/metrics` endpoint on `127.0.0.1:8000`.
The temporary smoke session was revoked after the probe.

| Surface | Evidence | Result |
|---|---|---|
| API accepted path | Real authenticated `POST /ui/execution/source-events` + `POST /ui/execution/intents` for `strategy_signal` with all risk checks true returned HTTP `201`, `risk_status=accepted`, `risk_reason=risk_gate_accepted`. | Pass. |
| API rejected paths | Real authenticated calls returned durable rejected intents for incompatible variant, missing feed, stale feed, config mismatch, stale account projection, insufficient capital, ownership conflict, inactive connection, blocked profile, inactive run, missing binding, missing manual recent-auth, ML policy missing and kill switch closed. | Pass. |
| Unsupported order model | Real authenticated `POST /ui/execution/intents` with `take_profit` returned HTTP `422`, reason `tp_sl_not_supported`; no intent row was created for that unsupported order. | Pass. |
| Durable state | Postgres contained `15` Stage 11 smoke intents grouped as `1` accepted and `14` rejected, with the expected stable risk reasons. | Pass. |
| Audit/no-dispatch | Postgres contained `15` `execution_risk_audit_events` rows and `BOOL_AND(metadata_json->>'dispatch' = 'no-dispatch') = true`. | Pass. |
| Redis | Redis `SCAN "*execution*"` returned `0` keys and `XINFO STREAM execution.requests.v1` returned `ResponseError`, proving the dispatch stream is absent. | Pass. |
| Metrics | `/metrics` exposed `execution_risk_gate_total` and `execution_risk_gate_latency_seconds`. | Pass. |

Production boundary command summary:

```text
api_cases 15 accepted 1 rejected 14
audit {'audit_count': 15, 'no_dispatch': True}
redis_execution_key_count 0 dispatch_stream ResponseError
metrics_present True
unsupported_order_model 422 tp_sl_not_supported
```

## Delivery Evidence

| Evidence | Result |
|---|---|
| Commits | `edc8e955` implementation; `089c895c` Alembic lint repair. |
| Initial CI | Run `26719536539` failed static lint on two Alembic `E501` lines; migrations and tests had passed. |
| Final CI | Run `26719584962` passed static lint/type/docs, migrations and all test shards. |
| Deploy | Publish App Image `26719665047`, Deploy Backend `26719665048` and Deploy Web `26719665044` completed successfully for `089c895c`. |
| Mac Studio smoke | `bash scripts/macos/smoke_prod.sh` succeeded on `/opt/roehub/app`: API health, Redis `PONG`, Postgres service, launchd services, Monit services and Tailscale backend state were healthy. |

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | `POST /ui/execution/intents` accepts optional `risk_context` and returns `accepted` or `rejected` risk status. Missing context fails closed instead of leaving new intents `not_evaluated`. |
| Ports/DTO | compatible-change | `CreateExecutionIntentCommand` and API DTOs gain optional risk context; repository port gains risk audit append. |
| Persistence | compatible-change | Adds accepted/rejected values to existing intent constraints and creates additive `execution_risk_audit_events`. Existing Stage 10 `recorded/not_evaluated` rows remain valid. |
| Redis | none | No Redis dispatch path is added; rejected intents cannot be published by current code. |
| Config | none | No new environment variable or feature flag. |
| Runtime/Ops | compatible-change | Adds bounded result counter and latency histogram. |
| UI/browser | none | No visible page or browser workflow changed. |
| External side effects | none | No exchange SDK/API call, credential decrypt, signed payload or order submit. |
| Logs/redaction | compatible-change | Audit metadata is bounded and contains no user, strategy, connection, API key, token, cookie, Authorization header, signed payload, ciphertext or passphrase labels. |

## Rollback

Revert the Stage 11 code path or stop including the UI execution router. The
new audit table and accepted/rejected intent rows are inert without Stage 12
dispatch. Existing rejected rows must be retained as safety audit evidence.

## Handoff To Stage 12

Stage `12` must dispatch only intents with `risk_status=accepted` and
`status=accepted`. Rejected intents and `recorded/not_evaluated` legacy Stage
10 rows are terminal for dispatch purposes. Stage `12` must keep Redis as
transport only and must prove `no-dispatch` for every rejected risk reason.
