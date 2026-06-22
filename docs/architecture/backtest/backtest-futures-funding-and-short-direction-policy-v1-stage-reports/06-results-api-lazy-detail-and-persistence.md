# Backtest Futures Funding And Short Direction Policy v1 - Stage 06 Results API Lazy Detail And Persistence

Expose persisted funding metrics through result APIs, lazy detail and cache
identity while keeping top rows summary-only.

Date: 2026-06-22

Status: accepted.

Proof boundary: this report records local implementation gates, local and Mac
Studio `TestClient` route smoke, direct `main` delivery, green GitHub CI/deploy,
`/opt/roehub/app` runtime hash parity and production smoke.

Execution branch policy: `main` by default; do not create branches, worktrees,
local workflow folders or stashes unless the user explicitly requests them.

`User required before start: nothing`

## Scope

Requested scope:

- expose funding metrics in top, variant and lazy-detail read models;
- expose funding event overlay data for lazy detail;
- include `funding_manifest_hash` in lazy detail cache identity;
- keep old no-funding jobs readable;
- keep top rows summary-only and avoid full trade tape persistence in top rows.

Out of scope:

- no browser UI changes;
- no launch policy changes;
- no Postgres schema migration unless the additive JSON/cache contract proves
  insufficient.

## Previous-Stage Gate

The Stage `06` prompt requires Stage `05` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state:

- Stage `05` status: `accepted`.
- Acceptance boundary: implementation gates only; no post-main CI/deploy or
  production runtime proof is claimed for Stage `05`.
- Accepted evidence includes shared lazy-detail/TP-SL exit semantics, net funding
  metrics, deterministic funding-window tests, required local gates and a Mac
  Studio target-host benchmark with a transient candidate patch.

Decision: previous-stage gate is satisfied. Stage `06` implementation may
proceed.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/06-results-api-lazy-detail-and-persistence.md`

Modified:

- `src/trading/contexts/backtest/application/dto/backtest_jobs.py`
- `src/trading/contexts/backtest/application/dto/__init__.py`
- `src/trading/contexts/backtest/application/ports/lazy_trades_cache.py`
- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py`
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `apps/api/dto/backtests.py`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`
- `tests/unit/apps/api/test_backtests_routes.py`
- `tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py`
- `tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py`

Deleted:

- none

Manifest note: the checkout already contained dirty Stage `05` implementation
files and unrelated architecture ledger/report edits before Stage `06` started.
This Stage `06` report lists only files touched for this stage on top of that
existing state.

## API Contract

Stage `06` adds only optional/additive response fields:

- top and variant responses now expose `funding_manifest_hash`;
- top and variant responses now expose a compact `funding` object with:
  - `funding_manifest_hash`;
  - `included`;
  - `metrics.total_return_pct_net_of_funding`;
  - `metrics.funding_return_pct`;
  - `metrics.funding_pnl_quote`;
  - `metrics.funding_events_count`;
  - data quality, stable warning codes, adjustment scope and effective ranking
    metadata when persisted in the top row;
- lazy-detail responses expose the same `funding_manifest_hash` and `funding`
  object;
- lazy-detail `chart_overlay` can now include compact `funding_events` overlay
  records.

Existing `summary_metrics` payloads remain unchanged and still carry the raw
funding metric names for current consumers. Existing jobs without funding fields
remain readable: the new funding object resolves to `included=false`, null
`funding_manifest_hash`, empty `metrics` and empty `warning_codes`.

Contract classification:

- Public API contract: `compatible-change`.
- DTO schema: `compatible-change`.
- Port contract: `compatible-change` for `BacktestLazyTradesCacheKey`.
- Persisted schema: `none`; no Postgres migration.
- Config schema: `none`.
- Request hash semantics: `none`.
- Cache key / persistence identity semantics: `compatible-change`; funding jobs
  include `funding_manifest_hash` in lazy-detail cache identity.
- Service-call auth/timeout/retry/error semantics: `none`.
- External side effects/idempotency: `none`.
- Logs/metrics/traces/audit/redaction: `none`.
- Alert/runbook semantics: `none`.
- Browser-visible behavior: `none`; Stage `07` owns UI.

## Persistence And Cache Contract

Top rows remain summary-only. Stage `06` does not store full trades in
`BacktestJobTopVariant.trades_json` and does not add a database column.

Lazy detail cache identity now includes `funding_manifest_hash` when present.
For jobs without a funding manifest, the digest omits the null funding field so
legacy no-funding cache identity remains stable. The public cache key fields
still expose `funding_manifest_hash` explicitly as null or the concrete hash.

The filesystem lazy-trades cache still persists trade rows separately and strips
full trade markers/segments from metadata. It now preserves only compact
`chart_overlay.funding_events` metadata on cache hits so funding overlays survive
materialization without storing a full trade tape in top rows.

## Lazy Detail

Lazy detail merges funding summary metrics from the persisted top row into the
recomputed detail summary. This keeps the existing exact trade recompute path
for gross detail while exposing the Stage `04`/`05` net funding metrics already
persisted in top-row summary JSON.

For futures jobs with `execution.funding.mode != off` and a selected
`funding_manifest_hash`, lazy detail loads funding arrays through the existing
artifact array port and builds compact overlay events using the accepted window:

```text
entry_time < funding_time <= exit_time
```

If funding overlay arrays are unavailable, lazy detail does not fail the base
detail response. It reports stable warning code `funding_events_unavailable` in
the funding object and leaves `funding_events` empty.

## Validation

Focused checks:

- `uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/apps/api/test_backtests_routes.py`
  passed with `95 passed`.
- `uv run ruff check src/trading/contexts/backtest/application/dto/backtest_jobs.py src/trading/contexts/backtest/application/dto/__init__.py src/trading/contexts/backtest/application/ports/lazy_trades_cache.py src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py apps/api/dto/backtests.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/apps/api/test_backtests_routes.py`
  passed.
- `uv run pyright src/trading/contexts/backtest/application/dto/backtest_jobs.py src/trading/contexts/backtest/application/dto/__init__.py src/trading/contexts/backtest/application/ports/lazy_trades_cache.py src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py apps/api/dto/backtests.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/apps/api/test_backtests_routes.py`
  passed with `0 errors`.

Required gates:

- `uv run ruff check apps/api src/trading/contexts/backtest tests`: passed.
- `uv run pyright apps/api src/trading/contexts/backtest tests`: passed with
  `0 errors`.
- `uv run pytest -q tests/unit/apps/api tests/unit/contexts/backtest`: passed
  with `602 passed`.

Delivery gates:

- Pre-publish broad gate `uv run ruff check .`: passed.
- Pre-publish broad gate `uv run pyright`: passed with `0 errors`.
- Pre-publish broad gate `uv run pytest -q -ra`: passed with `1319 passed,
  3 warnings`.
- Pre-publish docs gate `uv run python -m tools.docs.generate_docs_index --check`:
  passed.
- Scoped commit `d165d18e951b5470e0b275a4d30c73c3d5ae1321` delivered Stage
  `05` and Stage `06` funding scope to `main`.
- GitHub CI run `27985018813`: passed.
- GitHub Deploy Backend run `27985269835`: passed.
- GitHub Publish App Image run `27985269851`: passed.
- GitHub Deploy Web runs `27985269883` and `27985281326`: passed.
- Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com`
  fast-forwarded to `d165d18e951b5470e0b275a4d30c73c3d5ae1321`; remaining
  dirty files were unrelated ML/backfill files and were not part of this
  delivery.
- Runtime tree `/opt/roehub/app` hash parity matched commit `d165d18e` for:
  - `apps/api/dto/backtests.py`
    (`bf81bf0f3f5eeff707205ed9dfc9d6d66a8df31ecf747fd3d1c8486f747c00eb`);
  - `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
    (`0a8ba8c0126a49a3064ea296672c1c66eeb4a3d55ea1bbbc53428127c89996a3`);
  - `src/trading/contexts/backtest/application/services/v2/tp_sl_funding.py`
    (`c5ceab1d1ef8f72b33053656b64ec9a15a089f90af1e7408b289f18c01b5fb83`).
- `bash scripts/macos/smoke_prod.sh` passed in `/opt/roehub/app`.
- Runtime API smoke on Mac Studio returned `/health` `200` and
  `/auth/current-user` `401 missing_session_id`, proving the deployed API
  process is reachable and auth still fails closed without a session.
- Target-host route smoke on Mac Studio checkout passed:
  `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py::test_backtest_results_routes_expose_funding_fields_without_top_trades`
  with `1 passed`.

Regression coverage added:

- cache key digest changes when `funding_manifest_hash` changes;
- cache key fields expose `funding_manifest_hash`;
- filesystem lazy cache persists compact `funding_events` overlay metadata;
- lazy detail cache key uses artifact `funding_manifest_hash`;
- funding overlay uses `entry_time < funding_time <= exit_time`;
- top, variant and lazy-detail routes expose funding fields while top rows omit
  trades.

## Real-Boundary Evidence

Local API route smoke:

- `tests/unit/apps/api/test_backtests_routes.py::test_backtest_results_routes_expose_funding_fields_without_top_trades`
  exercises the real FastAPI router through `TestClient` for:
  - `GET /backtests/jobs/{job_id}/top`;
  - `GET /backtests/jobs/{job_id}/variants/{variant_key}`;
  - `POST /backtests/jobs/{job_id}/variants/{variant_key}/trades`.
- The smoke verifies `total_return_pct_net_of_funding`,
  `funding_manifest_hash`, stable funding warning codes and lazy-detail
  `chart_overlay.funding_events`.

Mac Studio changed-code runtime proof:

- Commit `d165d18e951b5470e0b275a4d30c73c3d5ae1321` is on `main`.
- GitHub CI/deploy workflows for that commit passed.
- Deploy Backend synced the source bundle into `/opt/roehub/app`, ran bootstrap,
  migrations, launchd reload and backend smoke.
- Runtime hash parity proved the deployed Stage `06` DTO and lazy-detail files
  match the committed source.
- `smoke_prod.sh`, API health/current-user smoke and target-host route smoke
  passed.

## Cold-Head Review Receipt

Cold-head review: completed.

Mode: cold self-review fallback.

Review scope:

- additive API fields;
- lazy cache identity and legacy no-funding cache compatibility;
- top-row summary-only persistence boundary;
- lazy-detail funding overlay semantics;
- route smoke and required quality gates;
- proof-boundary wording.

Review instructions: repository artifact review gate for architecture docs,
plans, prompt packs, prompt files, agent instructions, or skill-routing changes;
perform one cold independent review when available, otherwise perform and label a
cold self-review fallback before reporting ready/final.

Verdict: pass.

Blockers fixed:

- initial cache-key design risked hiding `funding_manifest_hash` when absent;
  cache key fields now expose it explicitly while digest omits null for legacy
  no-funding identity compatibility;
- filesystem cache initially dropped all overlay metadata; it now preserves
  compact `funding_events` without storing full trade markers/segments;
- lazy detail initially exposed recomputed gross-only summaries; it now merges
  persisted funding summary metrics from the selected top row.

Local follow-up check: completed through focused tests, focused ruff/pyright and
the required broad ruff/pyright/unit gates.

Residual risks:

- independent subagent review was not used in this turn; this receipt is a cold
  self-review fallback;
- Mac Studio changed-code runtime proof was collected after delivery to `main`;
- lazy-detail funding PnL per event is presented as overlay estimate when
  mark-price artifacts are available; authoritative net metrics remain the
  persisted summary metrics from the accepted Stage `04`/`05` scoring path.

## Residual Risks

- Stage `06` is accepted after direct `main` delivery, green GitHub CI/deploy
  and Mac Studio post-main production runtime proof.
- Existing materialized no-funding lazy-detail bundles keep their legacy digest
  because null `funding_manifest_hash` is omitted from identity; funding-enabled
  bundles use the new digest.
- The new API fields are additive, but downstream UI display remains Stage `07`.
