# Backtest Futures Funding And Short Direction Policy v1 - Stage 07 Futures-only Short Policy API And CJM

Enforce the futures-only policy for short-like directions across backtest API
preflight/create, strategy launch, scenario matrix and the backtests browser CJM.

Date: 2026-06-23

Status: implemented locally; post-main Mac Studio browser/runtime proof pending.

Proof boundary: local code, local gates and local changed-code browser evidence
are complete. This stage is not accepted until the same revision is on `main`,
relevant CI/deploy is green, `/opt/roehub/app` is synced from that revision, and
Mac Studio browser/runtime proof covers the changed code.

Execution branch policy: `main` by default; no branches, worktrees, local
workflow folders or stashes unless the user explicitly requests that exact
workflow.

`User required before start: nothing`

## Previous-Stage Gate

The Stage `07` prompt requires Stage `06` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state:

- Stage `06` status: `accepted`.
- Acceptance boundary: post-main production runtime proof is recorded in the
  stage ledger.

Decision: previous-stage gate is satisfied. Stage `07` implementation may
proceed.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/07-futures-only-short-policy-api-and-cjm.md`
- `src/trading/shared_kernel/direction_policy.py`

Modified:

- `apps/api/routes/backtests.py`
- `apps/api/routes/strategies.py`
- `apps/api/wiring/modules/ui_backtests.py`
- `apps/web/dist/css/pages/backtests.css`
- `apps/web/dist/js/pages/backtests.js`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`
- `apps/web/templates/pages/backtests.html`
- `docs/architecture/README.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`
- `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py`
- `tests/unit/apps/api/test_strategies_routes.py`
- `tests/unit/apps/api/test_ui_backtests_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/contexts/strategy/application/test_strategy_use_cases.py`

Deleted:

- none

## Policy Contract

- Public API contract: `breaking-change` for new `spot` plus `short` or
  `long_short_reversal` requests; they fail with
  `short_direction_requires_futures_market`.
- Existing persisted spot short-like jobs remain readable and immutable.
- Strategy launch is rejected unless the source snapshot and launch request are
  futures-compatible before strategy/profile/run creation.
- Scenario matrix now marks paper and testnet spot short-like rows blocked,
  not paper-only.
- DTO schema, persisted schema, config schema and request hash semantics:
  `none`.
- External side effects/idempotency: `compatible-change`; invalid requests fail
  before side-effecting launch creation.
- Browser-visible behavior: `breaking-change` for invalid spot short-like
  configuration and launch paths.

## CJM Changes

- Backtests workstation defaults to `spot + long_only` instead of the invalid
  `spot + long_short_reversal` combination.
- The browser consumes the server-provided
  `direction_market_compatibility` policy. Selecting `long_short_reversal` or
  `short` while on spot switches to futures when the futures option exists, and
  otherwise falls back to an allowed spot direction with the shared reason text.
- Results variant table and selected detail show gross `total_return_pct` and
  net `total_return_pct_net_of_funding` side by side.
- Selected detail shows funding included/degraded/unavailable state plus visible
  warning/detail text such as `missing_leading_coverage`.
- Old spot short-like job launch modal shows the policy reason and a
  `Switch to Futures config` CTA; it does not submit a strategy launch from the
  invalid spot variant.

## Validation

Final prompt-listed gates after browser-found fixes:

- `node --check apps/web/dist/js/pages/backtests.js` passed.
- `uv run ruff check apps src/trading/contexts/strategy src/trading/contexts/backtest tests` passed.
- `uv run pyright apps src/trading/contexts/strategy src/trading/contexts/backtest tests` passed with `0 errors, 0 warnings, 0 informations`; pyright reported only a newer-version notice.
- `uv run pytest -q tests/unit/apps tests/unit/contexts/strategy tests/unit/contexts/backtest` passed with `832 passed, 3 warnings`. Warnings are existing httpx per-request cookie deprecation warnings in web route tests.
- `python -m tools.docs.generate_docs_index --check` passed.
- `git diff --check` passed.

Additional gates run before the final browser-found JS/CSS fixes:

- `uv run ruff check .` passed.
- `uv run pyright` passed with `0 errors, 0 warnings, 0 informations`.
- `uv run pytest -q` passed with `1321 passed, 3 warnings`.

## Browser Evidence

Local changed-code browser QA used a real Chrome browser through Playwright
against a local SSR fixture server at `http://127.0.0.1:8017/backtests`. The
fixture served the real templates/assets and an in-memory API proxy for one old
spot `long_short_reversal` job with funding metrics and degraded warning data.

Assertions passed:

- Initial browser-visible config: direction `Long only`, market `Spot`.
- Selecting `Long/Short` switched market to `Futures` and displayed
  `short_direction_requires_futures_market` policy text.
- Result table and selected detail showed `GROSS RETURN`, `NET FUNDING RETURN`,
  gross `12.5%`, net `10.8%`, `Included with warnings`, and
  `missing_leading_coverage`.
- Old spot Long/Short job launch modal showed `Switch to Futures config`, the
  policy reason, and no launch submission.
- Clicking the CTA switched the configurator to `Futures` and closed the modal.
- Console errors: `0`.
- Failed requests: `0`.
- 4xx/5xx responses: `0`.

Screenshots:

- `/tmp/roehub-stage07-results.png`
- `/tmp/roehub-stage07-launch-modal.png`

Mac Studio changed-code browser/runtime proof was not run. Per `.codex/AGENTS.md`,
pre-main/local evidence cannot be reported as post-main production proof; this
is deferred to delivery after the revision reaches `main`, CI/deploy is green
and `/opt/roehub/app` is synced from that revision.

## Cold-head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `07` report, stage ledger status, file manifest, policy
contract, browser evidence boundary, validation evidence and residual acceptance
boundary.
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: browser QA found funding warning codes were title-only and old
spot Long/Short launch modal policy state was overwritten after form reset; both
were fixed and revalidated.
Local follow-up check: completed
Residual risks: Mac Studio changed-code browser/runtime proof remains pending
until post-main delivery; local browser fixture is changed-code evidence but not
production acceptance evidence.

## Residual Risks

- Stage `07` is not accepted yet because post-main Mac Studio browser/runtime
  proof is pending.
- The local browser fixture proved the CJM against deterministic API payloads;
  production proof must repeat the flow against the deployed target runtime and
  authenticated session.
- Existing old jobs are readable and launch-blocked by policy, but production
  old-job coverage still needs target-runtime browser evidence after delivery.
