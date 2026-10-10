# S07 independent production-risk review

Reviewer: `s07_security_review`. Date: 2026-10-08. Review-only; no product,
ledger, acceptance or publication mutations. No recursive reviewer.

## Scope and baseline

Reviewed new Navigator `artifacts-api.ts`, `data-artifacts.tsx`,
`artifact-profile.tsx`, `data-mode.tsx`, `data-artifacts.css`, and the owned
`data-page.tsx` delta against `S07-before-data-page.tsx.txt`. Consulted the actual
S06 DTO/service, management profile contracts, request client, subject/session
boundary and shared snapshot mechanism. Existing unrelated dirty paths are not
part of the reviewed change.

## Findings sent to executor

1. **P2, high confidence — invalid TP/SL text can save a different value.**
   `artifact-profile.tsx`, `levels()` (initial review line 34) ignores nonnumeric
   input while the uncontrolled input continues displaying it. With `1, x`, Save
   remains available and can submit the previous valid value together with other
   edits. Keep raw text and reject invalid grids before submitting.
2. **P2, high confidence — draft/unknown command departures lack a guard.**
   `data-artifacts.tsx` imports but does not render `DirtyGuard`. A route departure
   discards an edited profile or removes the only visible command-recovery key
   while the server outcome is unknown. Guard draft/pending/unknown departures,
   while allowing internal `artifactkey` bookkeeping without a discard prompt.
3. **P2, high confidence — reads ignore Retry-After.**
   `data-artifacts.tsx` applies server cooldown only in command `failure()`.
   Inventory/default/detail refresh and history retry can immediately send a new
   request following 429/503 with Retry-After. Apply cooldown to affected reads
   and their retry/navigation controls; stop polling until eligible.

Additional hardening observation: history/events render cached data on the render
that first observes a 403/404, before the effect clears the parent scope. Suppress
those data in the denied render, consistent with the main snapshots.

## Verified code properties

- Artifacts use subject-scoped query keys; Data remounts at subject changes and
  uses the existing session shutdown/cache-clearing boundary.
- Main 403/404 reads synchronously hide snapshots; command denials clear artifact
  queries and capabilities. No owner-only storage endpoint/cache is introduced.
- Commands carry bounded selected coordinates and server control versions. Stale,
  retained, fetching, conflicted and unknown-outcome states lock command controls.
- Unknown POST responses retain their key and expose same-key GET recovery.
  There is no automatic POST retry or blind replacement key for unresolved work.
- The actual API matches the client endpoints and nullable policy-only outcomes;
  profile and mode are mutually exclusive in submitted patches, schedule null
  restores inheritance, omission preserves schedule, and policy-only results do
  not start a request poll.
- Runtime indicator catalog and stored exclusions drive drafts. Progress uses
  server counts only, with unknown totals left unknown. No privileged paths,
  foreign work details or elapsed-time percentages are synthesized.
- Inventory and request-history continuations remain available for empty pages
  with a next cursor (inventory paging requires selecting one market).

## Verification limits and compatibility

At initial review, executor implementation/tests and disposable browser proof
were still in progress. No acceptance/readiness verdict is issued. Focused tests
should cover the above findings, subject/403/404 clearing, stale CAS, unknown
command recovery, policy-only outcomes, mixed patches and pagination. Final
browser desktop/narrow/keyboard/candle/loading/error/isolation evidence belongs
to the executor's S07 report.

API consumption and Navigator route additions: `compatible-change`. Existing
candle implementation delta is only mode navigation/component wrapping; actual
candle runtime regression proof remains required. Platform-client migration,
production runtime and deployment remain outside this review.

## Follow-up review 1

Re-read executor revisions after the initial findings, including `GridEditor`,
focus restoration and selection lock while editing.

- Departure guard: **resolved in code**. `ArtifactDraftGuard` covers editing,
  pending and unresolved keys, excludes internal artifactkey-only URL changes,
  and handles both SPA and document departure. Selection checkboxes are locked
  while editing; URL selection changes pass through the guard.
- Denied history render: **resolved in code**. History/events synchronously return
  no data on denied reads before propagating parent cache clearing.
- Focus restoration now waits for the editor to close and pending to settle.
- TP/SL validation: **partially resolved, still open**. Raw values and blocking
  validation were added, but edits to the other level field while either field is
  invalid do not update its draft value. Example: make TP invalid, change SL to a
  new valid value, then repair TP. Save becomes enabled but submits old SL.
  Parse current raw values for all touched level fields at save time.
- New `GridEditor`: **P2, high confidence**. Errors are stored in one parent
  string. Make parameter A invalid, then make a valid edit to B: B clears the
  common error while A still displays invalid raw text. Save can submit the old
  value of A. Track validation per parameter (including mounted/search-hidden
  fields), and block submission until all relevant raw inputs are valid.
- Read cooldown: **partially resolved, still open**. Manual main refresh/history
  retry respect a cooldown, but history/event queries still allow focus refetch,
  and history/event/page selection controls can trigger reads during cooling.
  A read-only cooldown also displays command `cooldown=0` as a 1970 deadline.
  Apply consistent disabled/retry behavior and show the actual read deadline.

Executor reports 59 focused tests and TypeScript passing, with real API/PG
browser profile/cadence proof underway. Those reports are not independently
rerun or treated as final stage acceptance here. New findings were sent before
this report update; no product source was changed by the reviewer.

## Follow-up review 2

- TP/SL cross-field scenario: **resolved in code**; each independently valid
  field updates its own draft while aggregate invalid state blocks Save.
- Grid validation collision: **resolved in code**; errors are keyed by indicator,
  parameter kind and name. A different parameter cannot clear the error.
- Main read cooldown and timestamp: **resolved in code**; read deadline feeds
  query eligibility, controls and displayed maximum deadline. History/events
  window-focus refetch is disabled; event/pagination controls obey cooldown.
- One remaining cooldown scenario sent to executor: after history returns 503
  with Retry-After, close and reopen history before expiry. The open toggle still
  changes query `enabled` false to true and triggers a refetch. Block reopen while
  cooling or include its deadline in query eligibility.
- Subject/unmount cleanup now cancels and removes the artifact cache and aborts
  the active request controller. No cross-subject capability or owner storage
  cache was introduced. Risk-level disclosure and last-request metadata do not
  widen the data/API boundary.

Inspected regression source for independent grid errors, malformed TP and the
TP-invalid/SL-edit repair sequence. Executor reports 62 passing tests; final test
and browser gate evidence remains the executor's responsibility.

## Final code-review disposition

Verified the final narrow history change: the closed history toggle is disabled
while its read cooldown is active (`disabled={!open&&cooling>Date.now()}`). Users
can close the section, but reopening cannot trigger an early enabled-transition
refetch. The last reported cooldown bypass is **resolved in code**.

All material findings raised in this review are now resolved in the inspected
source. No unresolved material code finding remains within the reviewed S07
boundary. This is an independent code-review conclusion, not stage acceptance,
release readiness, a claim of completed browser proof, or a deployment verdict.
The executor must bind final focused tests and real disposable-browser evidence
to the final stage report/receipt. No product or ledger mutations were made by
the reviewer.
