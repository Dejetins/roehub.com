# Stage 09E: Lifecycle Production Readiness

Дата проверки: 2026-05-26.

Статус: accepted; direct-main delivery pending after this readiness report commit.

Scope: production readiness for accepted Stage 09 lifecycle hardening. Stage 09E
does not add public API, persistence schema, trading execution, order placement,
exchange-execution, physical delete, or new exchange support.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisites | Stage 09A-09D must be accepted before 09E starts. | Iteration ledger marks 09A, 09B, 09C and 09D as accepted with direct-main delivery and Mac Studio/runtime evidence. | Accepted. | None. |
| Browser lifecycle | Authenticated `/settings` must prove `create -> validate -> disable -> archive -> assert hidden`. | Playwright artifact `output/playwright/settings-stage09e-lifecycle-20260526T185500Z.json` has `accepted=true`; screenshot `output/playwright/settings-stage09e-archived-panel-20260526T185500Z.png`. | Accepted. | 09E used a temporary server-side smoke app session because no `ROEHUB_E2E_USERNAME/PASSWORD` was available locally; Keycloak password login remains covered by Stage 08. |
| Default visibility | Archived e2e row must be hidden from default UI/API list and visible only through explicit archive/history path. | Default API after archive returned `label_present=false`; active UI filter hid the row; archived filter and `status=archived` returned the row with status `archived`. | Accepted. | None. |
| Permission semantics | `permission_mismatch` or readonly mismatch must not be presented as normal trade readiness. | Validate returned `invalid_credentials`, `requested_permissions=read`, `exchange_permissions=unknown`, `effective_permissions=none`; artifact asserts mismatch is not reported as trade-ready. | Accepted. | External validation used dummy credentials, so no live exchange-valid credential readiness is claimed. |
| Runtime evidence | API, DB, audit, metrics, Prometheus, Monit and OpenBao-relevant checks must be captured without secrets. | DB row is archived with lifecycle timestamps; audit has `exchange_connection_archived`; metrics expose archive/cleanup/mismatch counters; Prometheus `up{job="exchange-control"}=1`; Monit reports `roehub_exchange_control OK` and `roehub_openbao OK`; OpenBao `sealed=False`. | Accepted. | Direct internal capabilities curl returned 403 with the host env token; lifecycle API path itself succeeded through deployed `apps/api -> exchange-control`. |
| Secret safety | Reports and artifacts must not leak API secrets, ciphertext, HMAC, tokens or cookies. | Stage 09E artifact stores only sanitized statuses, label, redacted connection ref and screenshot path; secret inputs were cleared after create. Secret grep is recorded in Validation. | Accepted. | None. |

## Browser And API Evidence

| Check | Evidence | Result |
|---|---|---|
| Authenticated settings load | `https://roehub.com/settings`; `/api/auth/current-user` returned `200`; production settings APIs returned `200`. | Pass. |
| Add connection | UI add form submitted label `e2e_stage09_20260526T185459Z` with dummy Binance credentials. | `POST /api/ui/account/exchange-connections` returned `201`, lifecycle `active`, default permission `read`, validation `skipped_external_validation`. |
| Secret clearing | Add form secret input lengths after submit were `[0, 0]`. | Pass. |
| Validate | UI validate action called `POST .../validate`. | `200`, `invalid_credentials`, reason `exchange_rejected_credentials_400`, `effective_permissions=none`. |
| Disable | UI prompt accepted `DISABLE`; `POST .../disable`. | `200`, lifecycle `disabled`, `disabled_at` present; active/default UI no longer showed the label. |
| Explicit disabled history | UI disabled filter and `GET ...?status=disabled`. | Label present only as `disabled`. |
| Archive | UI prompt accepted `ARCHIVE`; `POST .../archive`. | `200`, lifecycle `archived`, `archived_at` present. |
| Assert hidden | Active/default UI and `GET /api/ui/account/exchange-connections` after archive. | Label absent; item statuses empty. |
| Explicit archived history | UI archived filter and `GET ...?status=archived`. | Label present only as `archived`. |
| Console/network | Artifact captured API response statuses and console summary. | Console errors `0`, warnings `0`; relevant account API calls returned `200/201`. |

## DB And Audit Evidence

| Surface | Query shape | Sanitized result | Verdict |
|---|---|---|---|
| Lifecycle row | `exchange_connections WHERE label='e2e_stage09_20260526T185459Z'`. | `620c6814...`, status `archived`, status_reason `user_archived`, validation `invalid_credentials`, requested `read`, exchange `unknown`, effective `none`, `disabled_at` present, `archived_at` present. | Pass. |
| Archive audit | `identity_audit_events` joined through redacted `metadata_json.connection_id`. | `exchange_connection_archived`, connection `620c6814...`, previous `disabled`, new `archived`, reason `user_archived`, secret marker check `false`. | Pass. |
| Temporary auth session cleanup | Active recent sessions for smoke subject after run. | Count `0` for non-revoked sessions created in the last 15 minutes. | Pass. |

## Metrics And Ops Evidence

| Surface | Evidence | Result |
|---|---|---|
| Exchange-control health | `curl http://127.0.0.1:9205/health/ready`. | `status=ready`, service identity `exchange-control`, external validation `ready`, Transit cipher `ready`. |
| Archive metric | `curl http://127.0.0.1:9205/metrics`. | `exchange_connection_archive_total{exchange="binance",reason="user_archived",result="archived"} 1.0`. |
| Cleanup metric | Same scrape. | `exchange_connection_cleanup_total{result="stage_09a_no_cleanup_attempt",source="none"} 0.0`; Stage 09D cleanup metric remains separately accepted. |
| Permission mismatch metric | Same scrape. | `exchange_permission_mismatch_total{effective="none",exchange="none",requested="read"} 0.0`; no mismatch was created by dummy invalid credentials. |
| Prometheus | `query=exchange_connection_archive_total`. | Prometheus API status `success`, result count `2`. |
| Prometheus target | `query=up{job="exchange-control"}`. | Prometheus API status `success`, value `1`. |
| Monit | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary`. | `roehub_exchange_control OK`, `roehub_openbao OK`, Keycloak and other Roehub processes OK. |
| OpenBao | `curl http://127.0.0.1:8200/v1/sys/health`. | `openbao_sealed=False`, initialized `True`, standby `False`. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `93 passed`, 3 known httpx cookie deprecation warnings. | Local run on 2026-05-26. |
| `uv run ruff check src/trading/contexts/exchange_control apps/api apps/web tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed. | Local run on 2026-05-26. |
| `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | Passed: `0 errors`. | Local run on 2026-05-26. |
| `python -m tools.docs.generate_docs_index --check` | Passed after adding this report and regenerating docs index. | Local run on 2026-05-26. |
| Authenticated Playwright | Passed. | `output/playwright/settings-stage09e-lifecycle-20260526T185500Z.json`; screenshot path above. |
| Runtime evidence | Passed with one caveat. | DB/audit/metrics/Prometheus/Monit/OpenBao evidence tables above; direct capabilities curl with env token returned `403`, while deployed API lifecycle calls succeeded. |
| Secret artifact grep | Passed. | Added-line scan over Stage 09E docs and sanitized Playwright artifact returned `introduced_secret_value_scan_matches=0`. |
| No-order/no-delete grep | Passed. | `git diff --name-only -- apps src tests tools migrations infra scripts .github` returned no changed runtime/code paths; Stage 09E diff is docs-only. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before evidence collection. |
| Commit / push | Pending. | Stage 09E docs will be committed and pushed directly to `origin/main`; no stage branch or PR. |
| CI / deploy | Pending. | Must be watched after direct-main push. |
| Post-deploy runtime | Pending. | Must confirm Mac Studio deployed revision and smoke after push. |

## Residual Risk And Handoff

| Risk | Impact | Follow-up |
|---|---|---|
| Keycloak password login was not re-executed in 09E. | The browser was authenticated with a temporary server-side smoke session, not a password credential flow. | Keep Stage 08 as accepted Keycloak login evidence; add `ROEHUB_E2E_USERNAME/PASSWORD` to an approved local secret channel if future stages must re-prove password login. |
| Dummy exchange validation returned `invalid_credentials`. | Lifecycle readiness is proven; live exchange-valid credential readiness is not claimed by 09E. | Use host-local real readonly credentials only through an approved secret channel for future exchange validation stages. |
| Direct capabilities curl with the current host env token returned `403`. | Does not block lifecycle proof because deployed `apps/api` successfully called exchange-control create/validate/disable/archive. | Reconcile host env token drift separately if operators need direct curl capabilities smoke outside deployed `apps/api`. |
