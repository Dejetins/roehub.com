# Backtest Futures Funding And Short Direction Policy v1 - Stage 08 Final Verification And Delivery

Final verification, documentation closure and delivery-readiness record for the funding and futures-only short policy line.

Date: 2026-06-23

Status: accepted.

Execution branch policy: `main` by default; no branches, worktrees, local
workflow folders or stashes unless the user explicitly requests that exact
workflow.

`User required before start: nothing`

## Scope

Requested scope:

- confirm Stages `00` through `07` are accepted before final verification edits;
- run broad local gates;
- run Mac Studio browser/runtime proof for the backtest funding and
  futures-only short CJM;
- run Mac Studio Prometheus/runtime proof for `market-data-scheduler` funding
  freshness metrics and alert assets;
- run pre-ship readiness review;
- update this final stage report, the stage ledger and docs index.

Out of scope unless explicitly requested by the user:

- no production code changes;
- no new feature scope;
- no branch, worktree, stash, temporary checkout, auxiliary folder or publish
  workflow;
- no commit, push, CI wait, deployment or production sync in this turn.

## Previous-Stage Gate

The Stage `08` prompt requires every required prior stage, `00` through `07`, to
be `accepted` or explicitly removed from scope before final verification edits.

Observed ledger state before this report was created:

| Stage | Ledger status | Gate result |
| --- | --- | --- |
| `00` | `accepted` | satisfied |
| `01` | `accepted` | satisfied |
| `02` | `accepted` | satisfied |
| `03` | `accepted` | satisfied |
| `04` | `accepted` | satisfied |
| `05` | `accepted` | satisfied |
| `06` | `accepted` | satisfied |
| `07` | `accepted` | satisfied |

Decision: previous-stage gate is satisfied. Stage `08` may proceed with final
verification.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/08-final-verification-and-delivery.md`

Modified:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`

Deleted:

- none

Production code changes:

- none.

## Stage Acceptance Summary

Decision: Stage `08` is accepted as final verification and delivery-readiness
closure.

- Stages `00` through `07` are accepted in the stage ledger.
- The current checkout is `main` at
  `aaee1bfeccbbfb08bd5994b5f00bf4c135659ec0`, matching the clean Mac Studio
  checkout.
- The Stage `08` diff is docs-only: this report, the stage ledger and generated
  docs index.
- Broad local gates passed.
- Mac Studio runtime, Prometheus, scheduler funding metrics, production smoke
  and authenticated production browser evidence passed.
- No publish, commit, push, CI wait, deployment or production sync was requested
  or performed in this turn.

## Risk Burn-Down

| Risk | Evidence to burn down | Current state |
| --- | --- | --- |
| Prior-stage gate drift | Ledger and accepted stage reports `00` through `07` | prior stages accepted |
| Local regression | `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` | passed |
| Docs index drift | `python -m tools.docs.generate_docs_index --check` | passed |
| Browser/CJM regression | Authenticated production browser proof for `/backtests` | passed |
| Funding freshness observability regression | Mac Studio `127.0.0.1:9202/metrics`, Prometheus query and alert asset proof | passed |
| Delivery overclaim | Pre-ship review and explicit no-publish boundary | passed |

## Validation

Broad local gates:

- `uv run ruff check .`: passed.
- `uv run pyright`: passed with `0 errors, 0 warnings, 0 informations`; pyright
  printed only the existing newer-version notice.
- `uv run pytest -q -ra`: passed with `1321 passed, 3 warnings in 62.27s`.
  Warnings are existing `httpx` per-request cookie deprecation warnings in
  `tests/unit/apps/web/test_app_routes.py`.
- `python -m tools.docs.generate_docs_index --check`: passed.

Additional final checks:

- `node --check apps/web/dist/js/pages/backtests.js`: passed.
- `git diff --check`: passed.

## Browser Runtime Evidence

Authenticated production browser QA used the pinned local Playwright CLI wrapper
against `https://roehub.com/backtests`. Login used the smoke Keycloak account
`smoke_e2e_keycloak`; the password was read from the Mac Studio host-local
`ROEHUB_SMOKE_E2E_PASSWORD` env source and was not printed, saved, screenshot, or
written to repository artifacts.

Observed evidence:

- Page URL: `https://roehub.com/backtests`.
- Page title: `Backtests | Roehub`.
- Authenticated `GET /api/backtests/runtime-defaults`: `200`.
- Runtime policy confirmed `spot.short` and `spot.long_short_reversal` reject
  with `short_direction_requires_futures_market`, while futures allows
  `long_short_reversal`.
- Initial browser-visible state: `Spot` and `Long only`.
- Selecting `Long/Short` switched the market to `Futures` and showed the
  short-direction policy message.
- Results view rendered `50` job rows; opening the first job rendered `10`
  variant rows.
- Result table/detail rendered `GROSS RETURN` and `NET FUNDING RETURN`.
- Mobile viewport `390x844`: `scrollWidth=390`, `innerWidth=390`,
  `rootWidth=374`; no horizontal overflow.
- Post-auth console errors: `0`.
- Failed requests: `0`.
- API `4xx/5xx` responses after the authenticated page load: `0`.

Tooling note: the Keycloak submit button still hit the known Playwright pointer
stability timeout, so the final login submit used DOM `.click()` on `#kc-login`.
The product assertions used the shipped page, routes and browser JS handlers.

## Funding Metrics And Alert Evidence

Mac Studio runtime proof used `ssh macstudio`; remote git commands ran only
under `/Users/daniildegtyarev/Projects/roehub.com`, and runtime checks used
`/opt/roehub/app`.

Runtime state:

- Mac Studio checkout: clean `main` at
  `aaee1bfeccbbfb08bd5994b5f00bf4c135659ec0`.
- Runtime API health: `GET http://127.0.0.1:8000/health` returned
  `{"status":"ok"}`.
- `bash scripts/macos/smoke_prod.sh` passed from `/opt/roehub/app`.
- Runtime hash parity matched the checkout for:
  - `src/trading/shared_kernel/direction_policy.py`;
  - `apps/web/dist/js/pages/backtests.js`;
  - `apps/api/dto/backtests.py`;
  - `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`;
  - `src/trading/contexts/backtest/application/services/v2/tp_sl_funding.py`;
  - `infra/macos/prometheus/rules/market-data-funding.rules.yml`;
  - `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`.

Scheduler funding metrics on Mac Studio `127.0.0.1:9202`:

- `scheduler_funding_catchup_instruments_total` exported Binance futures
  statuses `ok=432`, `not_due=136`; Bybit futures statuses `ok=3`,
  `not_due=651`, `skipped_missing_interval=36`.
- `scheduler_funding_catchup_rows_written_total` exported Bybit futures
  `3.0` for the current sample.
- `scheduler_funding_catchup_lag_seconds{status="ok"}` exported Binance
  futures `23614` and Bybit futures `2019`.
- `scheduler_funding_catchup_last_success_timestamp_seconds` exported
  `1782167624.035` for Binance and Bybit futures, i.e.
  `2026-06-23T01:33:44.035+03:00`.
- `scheduler_funding_catchup_universe_instruments` exported Binance futures
  `with_interval=568`, `missing_interval=0`; Bybit futures
  `with_interval=654`, `missing_interval=36`.

Prometheus proof:

- Prometheus target `market-data-scheduler` is `up` at
  `http://127.0.0.1:9202/metrics`.
- Prometheus API query for
  `scheduler_funding_catchup_last_success_timestamp_seconds` returned `2`
  series.
- Prometheus API query for
  `scheduler_funding_catchup_lag_seconds{status="ok"}` returned `2` series.
- Prometheus API query for
  `scheduler_funding_catchup_universe_instruments{status="missing_interval"}`
  returned `2` series: Binance `0`, Bybit `36`.
- Runtime alert asset
  `/opt/roehub/app/infra/macos/prometheus/rules/market-data-funding.rules.yml`
  is present and contains `MarketDataFundingCatchupErrors`,
  `MarketDataFundingNoRecentSuccess`, `MarketDataFundingLagHigh` and
  `MarketDataFundingMissingIntervals`.
- Runtime Prometheus config references
  `/opt/roehub/config/prometheus.rules/market-data-funding.rules.yml`.

## Delivery Evidence

Pre-ship gate:

- Branch/diff: `main...origin/main`; changed paths are exactly
  `docs/architecture/README.md`,
  `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/08-final-verification-and-delivery.md`
  and
  `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`.
- Intent vs delivered: final verification, docs closure and delivery-readiness
  evidence for the funding/short policy line; no production code changes.
- Checks: broad local gates, docs check, JS syntax check, diff whitespace check,
  Mac Studio runtime/Prometheus proof and authenticated production browser QA
  passed.
- Contract gate: no code/API/schema/config/identity changes; docs/evidence-only
  ledger/report update.
- Docs drift: docs index regenerated and checked.
- Release artifact need: no version, changelog, migration or rollback artifact
  is needed for this docs-only closure.
- Ship verdict: ready for scoped docs-only review. Publishing was not requested
  and was not performed.

## Contract Impact

Stage `08` is docs/evidence only:

- public API contract: `none`;
- port contract: `none`;
- DTO schema: `none`;
- persisted schema: `none`;
- config schema: `none`;
- request hash, cache key or persistence identity semantics: `none`;
- service-call auth, timeout, retry or error semantics: `none`;
- external side-effect idempotency and unknown-state semantics: `none`;
- logs, metrics, traces, audit, ledger, report or redaction semantics:
  `compatible-change` for documentation and evidence only;
- alert or runbook semantics: `none`;
- browser-visible behavior: `none`; runtime checks validate existing delivered
  behavior.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `08` report, stage ledger updates, docs index entry,
previous-stage gate, verification evidence, Mac Studio path/proof boundary,
browser auth/redaction handling, pre-ship scope and contract classification.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release
Blockers fixed: report updated from pending/in-progress to accepted only after
all broad gates, Mac Studio runtime/Prometheus proof, authenticated production
browser QA, pre-ship review, docs index and cold self-review passed; delivery
wording explicitly states no publish/commit/push/deploy was requested or
performed.
Local follow-up check: completed
Residual risks: independent subagent review was not used because the available
subagent tool policy permits spawning only when the user explicitly asks for
delegation; this receipt is therefore the repository-required cold self-review
fallback.

## Residual Risks

- This Stage `08` turn does not create a commit, push to GitHub, wait on CI, or
  perform a fresh deployment because publishing was not explicitly requested.
- Bybit still reports `36` futures funding instruments with missing interval
  metadata; this is monitored through the accepted
  `scheduler_funding_catchup_universe_instruments{status="missing_interval"}`
  metric and alert rule.
- The production browser proof is assertion-based; no screenshot artifact was
  recorded.
