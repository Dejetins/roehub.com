# Stage 09B: API UI List Archive

Дата проверки: 2026-05-26.

Статус: accepted; direct-main delivered; CI/deploy and Mac Studio runtime
evidence complete for implementation commit `80b1dacf`.

Scope ограничен public account facade и `/settings` list semantics. Stage 09B
exposes the Stage 09A lifecycle foundation through active-only defaults,
explicit disabled/archived history filters, `POST .../archive` for disabled
connections, and active-only limits. Permission mismatch semantics, controlled
cleanup of old `stage08_*` rows, and trading execution remain out of scope.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| 09A prerequisite | Stage 09A must be accepted before Stage 09B starts. | Iteration ledger marks 09A as `accepted; direct-main delivered; Mac Studio runtime evidence complete`, and 09A report status is accepted. | Accepted. | None. |
| API list defaults | `GET /api/ui/account/exchange-connections` returns active rows by default; `status=active`, `status=disabled`, `status=archived`, and `status=all` are explicit filters. | API route tests prove default equals active-only, disabled/archived are hidden from the default list, and explicit filters return the expected lifecycle rows. | Accepted. | Cursor/limit remain pass-through placeholders, unchanged from prior facade behavior. |
| Archive command | Public archive is `POST .../archive`, allowed for disabled owned rows and rejected for active rows. | API route tests cover active archive returning `exchange_connection_not_disabled`, disabled archive returning `archived` with `archived_at`, and archived rows rejecting rotate/validate as not found. | Accepted. | UI exposes archive only in disabled history; direct API callers can still retry archived archive idempotently through 09A semantics. |
| Security gates | Archive mutation must fail closed on missing/cross-origin context before command execution and require recent auth. | Route calls same-origin/CSRF before recent-auth and before exchange-control client use; focused test proves missing origin returns `csrf_required` and archive client is not called. | Accepted. | Existing create/rotate/disable same-origin behavior is unchanged. |
| UI behavior | `Connected Exchange APIs` shows active rows only; disabled/archived are available through explicit history filters. | Browser QA on local `/settings` shows Active tab with only active row, Disabled tab with archive action, archive prompt, empty disabled list after archive, and Archived tab with archived rows. | Accepted. | Production-authenticated 09E will run the full create -> validate -> disable -> archive proof after cleanup stages. |
| Limits | Account limits count only `status == "active"` rows. | API route tests assert `exchange_connections_used` and `api_keys_used` exclude disabled/archived rows. | Accepted. | None. |
| No DELETE | Stage 09B must not add a public DELETE endpoint or imply physical deletion. | Web route tests assert no `DELETE` action in settings JS; route implementation adds only `POST .../archive`; 09A no-hard-delete persistence remains unchanged. | Accepted. | physical hard delete запрещен remains a Stage 09 invariant. |

## API UI Evidence

| Surface | Change | Evidence |
|---|---|---|
| DTO | `ExchangeConnectionResponse.status` now includes `archived`; response includes nullable `archived_at`. | `apps/api/dto/ui_account.py`; API tests assert archived response shape. |
| List route | Added `status` query filter with default `active`; `all` is explicit and not the default. | `tests/unit/apps/api/test_ui_account_routes.py::test_ui_account_exchange_connections_default_active_filter_archive_and_limits`. |
| Limits route | Counts only rows where `status == "active"` for `exchange_connections_used` and API keys. | Same API test covers active count after disabled -> archived transition. |
| Archive facade | Added `POST /api/ui/account/exchange-connections/{connection_id}/archive` via same-origin web proxy to apps/api `/ui/.../archive`; no DELETE route added. | API test covers disabled success and active deterministic rejection; web route test checks `/archive` action and no `DELETE` client action. |
| Audit | User-facing archive calls `record_exchange_connection_archive` with redacted metadata only: connection id, exchange, market, environment, previous/new status and reason. | API test asserts exactly one `exchange_connection_archived` event after disabled -> archived. |
| Settings UI | Added accessible status tabs for Active, Disabled and Archived history; default JS request is `status=active`. | Web route test checks SSR controls; browser QA confirms runtime DOM and network requests. |

## Browser Evidence

| Check | Evidence | Result |
|---|---|---|
| Local browser target | `http://127.0.0.1:8765/settings` with mocked same-origin API upstream and authenticated current-user adapter. | Pass. |
| Default active list | Playwright network showed `GET /api/ui/account/exchange-connections?status=active => 200`; Active tab rendered only `active_main`. | Pass. |
| Disabled history | Disabled tab requested `status=disabled`, rendered `disabled_old`, and exposed only `Archive connection`; rotate/validate/disable were absent for disabled row. | Pass. |
| Archive mutation | Prompt accepted `ARCHIVE`; network showed `POST /api/ui/account/exchange-connections/.../archive => 200`; Disabled tab became empty. | Pass. |
| Archived history | Archived tab requested `status=archived` and rendered the newly archived row plus existing archived row with no action buttons. | Pass. |
| Console | `playwright-cli console warning` returned 0 warnings/errors. | Pass. |
| Screenshot | `output/playwright/stage09b-settings-archived.png`. | Captured archived-history state. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds `status` list filter, `archived` status literal, nullable `archived_at`, and `POST .../archive`; default list changes to active-only by design for Stage 09B. |
| Browser-visible behavior | `compatible-change` | `/settings` now treats the main `Connected Exchange APIs` table as active-only and moves disabled/archived rows behind explicit status tabs. |
| Internal API contract | `none` | Stage 09B reuses the local-only archive command and capability from 09A; no internal endpoint shape changes. |
| Port contract | `none` | No exchange-control application/repository port is changed beyond accepted 09A archive support. |
| Persisted schema | `none` | No migration or storage invariant changes are added in 09B. |
| Audit schema | `none` | `exchange_connection_archived` was added in 09A; 09B only calls the prepared redacted writer. |
| Config / env | `none` | No new runtime variables or feature flags. |
| Metrics / ops | `none` | Archive/cleanup metrics contract remains the 09A exchange-control metric surface. |
| Trading execution | `none` | No order placement, exchange execution, order ledger, or signal-to-execution path is added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/contexts/exchange_control` | Passed: `72 passed`, 3 known httpx cookie deprecation warnings. | Local run on 2026-05-26. |
| `uv run ruff check apps/api apps/web src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/exchange_control` | Passed. | Local run on 2026-05-26. |
| `uv run pyright apps/api src/trading/contexts/exchange_control tests/unit/apps/api tests/unit/contexts/exchange_control` | Passed: `0 errors`. | Local run on 2026-05-26. |
| `python -m tools.docs.generate_docs_index --check` | Passed. | Docs index updated for the 09B report and checked on 2026-05-26. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | `80b1dacf Add exchange connection archive UI semantics`; pushed `3141d402..80b1dacf` to `origin/main`. | Pass. |
| CI / deploy | CI `26422618351` success; Deploy Backend `26422665374` success; Publish App Image `26422665365` success; Deploy Web `26422665361` and `26422695699` success. | Pass. |
| Mac Studio runtime smoke | `cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh`. | Pass; backend API smoke, expected unauthenticated 401, Redis PONG and Tailscale state were healthy. |
| Exchange-control readiness | `curl -fsS http://127.0.0.1:9205/health/ready`. | Pass; service ready, service identity ready, external validation ready, Transit secret cipher ready. |
| Deployed archive route evidence | `/opt/roehub/app/apps/api/routes/ui_account.py` contains `/ui/account/exchange-connections/{connection_id}/archive`. | Pass. |

## Residual Risk And Stage 09C Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Permission semantics remain old `permissions` behavior. | 09C | Add requested/exchange/effective permission fields and deterministic mismatch semantics without overloading lifecycle status. |
| Old disabled `stage08_*` and e2e rows are not cleaned up in 09B. | 09D | Run controlled cleanup/backfill through archive command/API only; no physical deletes. |
| Production authenticated browser proof is deferred. | 09E | Prove create -> validate -> disable -> archive -> hidden default list on production after cleanup/prep stages. |
