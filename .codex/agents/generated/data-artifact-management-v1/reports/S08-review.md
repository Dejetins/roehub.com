# S08 independent production-risk review

Reviewer: `s08_security_review`. Date: 2026-10-08. Procedure:
`production-risk-review`; independent read-only product review required by AGENTS
for the installation-owner boundary. No recursive reviewer or product edits.

## Findings and disposition

No unresolved material findings in the final reviewed source.

- **P1, resolved:** the original unknown-operation guard existed only in a URL
  parameter; ordinary Settings navigation removed it and allowed a new cleanup
  key without reconciliation. The final implementation persists only a minimal
  subject/organization-scoped operation identity in sessionStorage before sending
  durable commands. Navigation restores the guard; unavailable recovery storage
  fails closed. No path, capability token or response body is persisted there.
  The navigation regression test verifies no replay/new mutation.
- **P2, resolved:** late command completion could recreate owner-only cache data
  after denial/unmount, including across the awaited query cancellation. The final
  code aborts on denial/unmount, checks current denial/mount/abort state after both
  asynchronous boundaries, clears the owner cache and latches denial independently
  of the removed query. The late-reply revocation test covers cache erasure. Final
  denied browser evidence also shows an explicit denied state without server path
  or a perpetual initial-loading state.

## Reviewed boundary

Read storage-api, storage-settings and CSS, settings-page/client-routes integration,
storage-settings tests, shared requestJson/read-snapshot/query-cache mechanisms,
S06 report, actual storage HTTP route, typed commands/results and storage service.
Also reviewed backtest-readiness, builder and builder-api integration and focused
readiness tests against the existing preflight/recipe projection.

Installation ownership is checked separately from organization operations rights;
the backend remains authoritative. Owner reads occupy a separate privilege cache.
Commands use the trusted root identifier, expected version, fresh idempotency key
and probe/preview token. The browser submits no arbitrary server path. Existing
same-origin request handling and backend recent-auth/origin checks remain in place.
Preview and confirmation are separate; target edits invalidate the preview;
expired/version-mismatched tokens cannot enable confirmation. Relocation is a
reported outcome, with no new move/delete/activation behavior.

Backtests distinguishes source watermark from published NPY bounds, preserves
date admission and existing submission semantics, and consumes the existing
preflight input_readiness projection. Missing projection remains unavailable;
ready is explicitly pending worker payload validation. No scoring, rank or
financial calculation change was observed in these owned UI changes.

## Evidence inspected and limits

- `S08-checks.json`: recorded successful focused frontend run, **105 tests in 7
  files**, typecheck, candidate build and scoped Ruff. Build includes the recorded
  large-bundle warning; no performance conclusion follows. Reviewer inspected
  evidence and tests, without independently rerunning the suite.
- `S08-api-negative.json`: actual route responses for organization admin GET/POST
  403, missing Origin 403, arbitrary path 422 and stale version 409.
- Browser text artifacts: `S08-first-configured.txt`,
  `S08-relocation-required.txt`, `S08-storage-denied-final.txt`,
  `S08-storage-unavailable.txt`, `S08-backtest-materialization.txt`,
  `S08-beyond-npy.txt`, `S08-backtest-journey.txt`. These support the displayed
  configured/paused, relocation, denied/unavailable, NPY lag, on-demand preparation,
  out-of-bounds refusal and existing queued-history UI states.
- Browser fixture inspection confirms synthetic authentication and job persistence,
  real disposable PG/filesystem storage service through a direct `LocalControl`
  adapter, and native NPY/preflight execution over deterministic synthetic candles.
  The browser run does not prove production authentication, signed Unix transport,
  provider ingestion, scheduler execution or a completed worker backtest.
- Executor reports a separate focused backend run of **28 passed / 93 deselected**,
  including real PG/Unix storage and native ready/materialization cases. This is
  executor-reported here; it must be bound to final stage evidence separately.
  S06's accepted real Unix/PG/files proof is predecessor evidence, not a browser
  claim or proof of a deployed installation.

Recovery requires separate owner reconciliation after an unknown result; this UI
does not implement automatic cleanup recovery. Session recovery is a local UX
guard, while durable backend ownership/quarantine provides the data-safety boundary.
Review finds no reason to broaden scope into S06 transport or numerical semantics.
UI/API consumption is `compatible-change`; scoring/rank/NPY schemas are `none`;
production installation and provider behavior remain `unknown`.

This review supports the reviewed local S08 changes within these evidence limits.
It does not accept the stage ledger, execute S09, authorize publication or establish
production readiness. Final report, evidence bindings and lifecycle checks remain
the executor's responsibility.
