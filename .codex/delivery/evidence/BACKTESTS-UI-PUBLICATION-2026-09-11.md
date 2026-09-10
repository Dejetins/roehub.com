# Accepted Backtests UI publication

Pull request: https://github.com/Dejetins/roehub.com/pull/33

Product-owner acceptance: 2026-09-11. Authorized operation: publish the complete
Backtests implementation through `codex/backtests-ui-accepted`, merge into `main`,
then delete the technical branch. [Accepted iteration log](../../../docs/architecture/apps/web/backtests-ui-iteration-log.md).

Scope: platform client and web contracts; gated web integration; job-pinned candle
API and aggregation; QA fixtures, tests and CI; associated architecture, ticket,
iteration evidence and generated project map. Local runtime data, credentials,
dependency/build output and the unrelated `.codex/AGENTS.md` change are excluded.

Independent read-only release review found no material blocker in the sampled
security/data boundaries. The review checked authorization before artifact reads,
transport/session/recovery behavior, bootstrap/build-path checks, local QA
isolation and artifact hygiene. Its identified compact-spec coverage gap is
addressed by including `compact-builder.spec.ts` in the standard e2e runner.

Local verification before publication:
- `ruff check .` and repository Pyright: passed.
- Documentation index/project-map drift checks: passed.
- Focused web/API/candle/preview-session pytest suites: 98 passed.
- Current frontend unit suite: 167 passed; typecheck and build passed.
- Existing real-browser evidence covers the accepted UI, actual-price charts,
  viewport expansion, motion, focus and desktop/mobile layouts. This release
  reuses that evidence; GitHub runs the isolated e2e fixtures separately.

The GitHub PR and its checks are the publication-state authority. Merge is gated
on successful relevant checks. The technical branch is removed only after merge.
The repository selects no production installation target/runbook, so the intended
terminal status is `shipped-no-runtime`, not deployed. Target-role cutover remains
an explicit separate dependency (`target_role_cutover_ready=false`).

CI reconciliation: registered the published first-party screenshot hashes, regenerated
the runtime-input inventory, and added candle coverage to the CI shard matrix.
Updated browser assertions for the accepted embedded workspace and made the
disposable session-expiry fault exceed the preview session lifetime. Focused
release inventory and routing regression suite: 396 passed.

Browser regression found an uncaught `InvalidStateError: ViewTransition opt-in
disabled` during report-to-saved-strategy navigation. The client shell opted in
inline while the SSR shell waited for external CSS. The SSR head now declares
the same opt-in before stylesheets. The real-results e2e retains its no-page-error
assertion and the complete CSV/save/return/delete journey as regression coverage.

Automated PR review disposition: the suggestion to substitute API `return_pct`
for monthly account return is not adopted. `build_monthly_stats_read_model` in
`result_series.py` sums individual trade percentages; that is not return on the
opening monthly account balance. The accepted UI derives that percentage from
server-provided cash/P&L, and explicitly shows unavailable when starting cash is
missing instead of substituting a different statistic. `report-ui.test.tsx` covers
cross-year balances and the missing-cash case. No invented percentage is shown.
