# Stage 10E: Trading CJM Production Readiness

Дата проверки: 2026-05-27.

Статус: partial; production readonly rejection/not-active path, Active-only
readiness, runtime ops, docs, and local gates are accepted. Full
trading-ready success proof is blocked because env-backed trade credentials are
absent on Mac Studio: `ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY`,
`ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_KEY`, and matching secret env vars are not
present. No orders were placed.

Scope: evidence/readiness stage for the trading-only `/settings` exchange
connection CJM after Stage 10D reclassified existing non-trading active rows.
Implementation changes were not required.

## Evidence Matrix

| Surface | Requirement | Evidence | Verdict |
|---|---|---|---|
| Prerequisite | Stage 10D accepted before Stage 10E starts. | Ledger marks 10D accepted with dry-run -> execute -> API/DB/audit/metrics evidence. | Pass. |
| Browser CJM | Authenticated production `/settings` proves no read/trade selector, `Connect and validate`, mainnet default, advanced testnet, Active/History only, `Disconnect`, `Re-check`, and password-manager hardening. | `output/playwright/settings-stage10e-readonly-production-20260526T223834Z.json` has `readonly_path_accepted=true`; screenshot `output/playwright/settings-stage10e-readonly-production-20260526T223834Z.png`. Bundle contains `Disconnect` and `Re-check`; no active row existed, so active-row action visibility is covered by Stage 10C and blocked from fresh production proof by missing trade credentials. | Partial pass. |
| Readonly rejection | Env-backed readonly key must be rejected/not active. | Playwright used host-local `ROEHUB_TEST_BYBIT_READONLY_API_KEY/SECRET` without printing values. POST response was `status=disabled`, `status_reason=auto_validation_failed`, `effective_capability=none`, `connection_readiness=rejected`, reason `read_only_not_supported`; History contained the attempt, Active did not. | Pass. |
| Active readiness | Active list contains only `Ready for trading` rows. | Public API curl returned `{"count":0,"all_ready":true,"items":[]}`. DB `exchange_connections WHERE status='active'` returned `active_count=0`. | Pass. |
| Trade-ready proof | Run connect -> active `Ready for trading` when env-backed trade credentials exist. | Mac Studio env check: `ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY`, `ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_SECRET`, `ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_KEY`, `ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_SECRET`, `ROEHUB_E2E_BINANCE_MAINNET_TRADE_API_KEY`, and `ROEHUB_E2E_BINANCE_MAINNET_TRADE_API_SECRET` are missing. | Blocked. |
| Runtime ops | API/DB/audit/metrics/Prometheus/Monit/OpenBao evidence. | API/DB/metrics/Prometheus/Monit/OpenBao calls passed. Audit has redacted auto-validation evidence under accepted event type `exchange_connection_validated` with `validation_mode=auto_validation`, plus `exchange_connection_reclassified`; the prompt's suggested `exchange_connection_auto_validated` / `exchange_connection_rejected` event names are not the current accepted schema. | Pass with note. |
| Secret safety | No secrets in artifacts/reports. | Playwright summary stores only redacted booleans; API key/secret values are not written. Grep matches are field-name-only (`api_secret_redacted`, `password_inputs`) and historical `.playwright-cli` password text, not secret values. | Pass with explained matches. |

## Browser / Playwright Evidence

| Check | Sanitized result |
|---|---|
| Target | `https://roehub.com/settings` with temporary server-side session. |
| Artifact | `output/playwright/settings-stage10e-readonly-production-20260526T223834Z.json`. |
| Screenshot | `output/playwright/settings-stage10e-readonly-production-20260526T223834Z.png`. |
| Selector checks | `permissions_controls=0`, `password_inputs=0`, `active_tabs=1`, `history_tabs=1`, `disabled_tabs=0`, `archived_tabs=0`. |
| Visible / bundle checks | `Connect and validate=true`, `mainnet_default=true`, `testnet_advanced_visible=true`, `Disconnect=true`, `Re-check=true`. |
| Create payload | `contains_permissions=false`; keys are `api_key`, `api_secret`, `environment`, `exchange_name`, `label`, `market_type`; stored artifact only records `api_key_redacted=true` and `api_secret_redacted=true`. |
| Readonly response | HTTP `201`; durable row `disabled/auto_validation_failed`; `connection_readiness_reason=read_only_not_supported`. |
| Active proof inside browser | `active_api_summary.count=0`, `all_ready_for_trading=true`, `contains_readonly_attempt=false`. |
| Cleanup | Temporary `stage10e-%` sessions were revoked; final active temporary-session count is `0`. |

## Runtime Evidence

| Surface | Command | Sanitized result |
|---|---|---|
| Public Active API | `curl -fsS "https://roehub.com/api/ui/account/exchange-connections?status=active" -H "Cookie: roehub_session_id=<redacted>" \| jq -c ...` | `{"count":0,"all_ready":true,"items":[]}`. |
| DB active rows | `SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE status='active' ORDER BY created_at DESC LIMIT 20;` | `active_count=0`, `rows=[]`; no secret-bearing columns selected. |
| DB readonly attempts | `SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE label LIKE 'prod_readonly_reject_%' ORDER BY created_at DESC LIMIT 5;` | Latest Stage 10E rows are `disabled/auto_validation_failed`, `effective_capability=none`, `connection_readiness=rejected`, reason `read_only_not_supported`. |
| Audit | Recent `exchange_connection%` audit query. | Latest readonly attempts emit `exchange_connection_validated` with metadata `{"exchange":"bybit","operation":"create","result":"rejected","reason":"read_only_not_supported","validation_mode":"auto_validation"}`; Stage 10D `exchange_connection_reclassified` row remains present. Metadata is redacted. |
| Prompt audit query | `event_type IN ('exchange_connection_auto_validated','exchange_connection_rejected','exchange_connection_reclassified','exchange_connection_disabled')`. | Returns reclassification evidence; auto-validation evidence is under `exchange_connection_validated` in the accepted Stage 10B contract. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics \| rg 'exchange_connection_auto_validation_total\|exchange_connection_trading_readiness_total\|exchange_connection_reclassification_total'`. | `exchange_connection_auto_validation_total{exchange="bybit",result="rejected",reason="read_only_not_supported"} 2.0`; `exchange_connection_trading_readiness_total{exchange="bybit",result="rejected",reason="read_only_not_supported"} 3.0`; `exchange_connection_reclassification_total{source="stage10d",result="disabled",reason="read_only_not_supported"} 1.0`. |
| Prometheus | `curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22exchange-control%22%7D' \| jq`. | `status=success`, `up{job="exchange-control"}=1`. |
| Monit | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary \| rg 'roehub_exchange_control\|roehub_openbao'`. | `roehub_exchange_control OK`, `roehub_openbao OK`. |
| OpenBao | `curl -fsS "$OPENBAO_ADDR/v1/sys/health" \| jq '{sealed,initialized}'`. | `{"sealed":false,"initialized":true}`; no token printed. |
| No-order grep | `rg -n "place_order|create_order|submit_order|cancel_order|exchange[-_ ]execution|order placement|order ledger|simulate order" src/trading/contexts/exchange_control apps/api apps/web tools tests/... \|\| true`. | No matches. |

## Validation

| Gate | Result |
|---|---|
| Required pytest | Passed: `117 passed`, `3 warnings` from existing httpx cookie deprecation. |
| Required ruff | Passed: `All checks passed!`. |
| Required pyright | Passed: `0 errors`; pyright printed only the upstream new-version notice. |
| Docs index | Passed after report creation: `python -m tools.docs.generate_docs_index --check`. |
| Authenticated Playwright readonly rejection | Passed; `readonly_path_accepted=true`. |
| Authenticated Playwright trade-ready flow | Blocked: no env-backed trade credentials present. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No code or payload shape changes in Stage 10E. |
| Internal API / ports | `none` | Evidence-only stage; no command contract changes. |
| Persistence schema | `none` | No migration or storage schema changes. Readonly attempts remain durable disabled rows by accepted Stage 10B behavior. |
| Config / secret contract | `unknown` for future full acceptance | Required trade env names are absent, so full trading-ready proof cannot be completed. |
| Browser-visible behavior | `none` | No UI code changed; production browser evidence confirms accepted Stage 10C behavior for readonly path. |
| Trading execution | `none` | No order placement, exchange-execution, order simulation, or order ledger code path was added or called. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before evidence collection. |
| Commit / push | `dd93c736 Record stage 10e production readiness`, pushed to `origin/main`. | Pass. |
| CI | GitHub Actions CI `26479411653`. | Success; docs-only path ran docs index drift check and skipped non-applicable broad jobs. |
| Deploy | Deploy Backend `26479427111`, Deploy Web `26479427098`, Publish App Image `26479427110`. | Success. |

## Residual Risk / Handoff

| Risk | Status | Next action |
|---|---|---|
| Full trade-ready production success proof | Blocked. | Add approved env-backed trade credentials under `ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_KEY` / `ROEHUB_E2E_BYBIT_MAINNET_TRADE_API_SECRET` or `ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_KEY` / `ROEHUB_E2E_BYBIT_TESTNET_TRADE_API_SECRET`, then rerun Stage 10E trade-ready Playwright without placing orders. |
| Active-row `Disconnect` / `Re-check` browser visibility | Partial. | Labels are present in the deployed bundle and Stage 10C proved rendering; Stage 10E had no active trading-ready row because trade credentials are missing. |
| Audit event naming | Accepted with note. | Current accepted implementation records auto-validation as `exchange_connection_validated` with `validation_mode=auto_validation`; future cleanup may add alias event types only through a separate compatibility stage. |
| Future Stage 11 | Blocked until full Stage 10E acceptance decision. | Do not start strategy binding guard as accepted full readiness while trade-ready proof remains missing. |
