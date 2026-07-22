---
evidence_id: ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20
ticket_id: ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20
status: accepted
verdict: pass_with_explicit_waivers
observed_at: 2026-07-20T23:10:40Z
historical_observations_preserved: true
archive_reverified_at: 2026-07-22T11:45:56Z
supplemental_captures_verified_at: 2026-07-22T12:24:32Z
---

# Linear reference completion evidence

## Verdict

All live-browser observations in this document were made on `2026-07-20` and are preserved as historical evidence. On `2026-07-22`, this execution reverified the supplied archive and all listed PNG hashes without opening, querying, or changing Linear.

`pass_with_explicit_waivers` for sanitized reference evidence only.

This evidence accepts the reference pack as sufficient input for the next
repository ticket. It does not prove Roehub runtime, product, Penpot, release,
or deployment readiness.

The `2026-07-22` amendment records three additional user-supplied project
overview states and formalizes functional-structure equivalence. It does not
turn Linear screenshots into Roehub designs or authorize literal replication.

## Historical 2026-07-20 pre-write gates

The table below is preserved historical context, not a claim about current Linear or repository state. The current execution did not query or modify Linear.

| Gate | Result |
|---|---|
| Linear ROE-8 status | `Todo` |
| Repository ticket status before execution | `ready` |
| Repository dependencies | none; `depends_on: []` |
| Archive exists | pass |
| Archive SHA-256 | `eb7b0ab070f64d553baafacefa90fdb2e87e51bc174c63db9af73bc77f8e41c2` |
| Local `main` | `6f64518ba0df0b0ca72944abeadd10b5f8775f69` |
| `origin/main` | `6f64518ba0df0b0ca72944abeadd10b5f8775f69` |
| Active owned-path overlap | none found |
| Authenticated reference boundary | separate agent-created Chrome tab in the existing authenticated session |
| Other browser sessions | not closed or changed |

The parallel authorization worktree was inspected only for owned-path overlap;
its active changes were disjoint. No Custometry W19 result was read or reused.

## Evidence obtained

- All `16` supplied PNG hashes match the manifest.
- All supplied PNGs are `2560 x 1440`; PNG `pHYs` and EXIF X/Y density
  metadata are absent, and browser chrome is included.
- Archive CSS viewport and DPR are explicitly unknown rather than inferred.
- Live Chrome observation: `1512 x 790` CSS viewport, DPR `2`, Chrome
  `150.0.7871.129`, `prefers-reduced-motion=false`.
- Live shell geometry: `305px` navigation, `1199px` main surface, `7px`
  `col-resize` target.
- `Cmd+K`, non-sensitive command search, Escape, issue-options popover,
  deterministic popover focus return, direct route loading, and Browser Back
  URL restoration were observed.
- Sanitized accessibility evidence is retained as role counts only.
- Every prior manifest gap is either observed or explicitly waived with impact
  in `docs/architecture/ui/linear-workspace-reference-measurements-v1.md`.
- Three supplemental PNGs were copied to the local reference folder outside
  Git. Their hashes, pixel dimensions, and sanitized functional-state tags are
  recorded in the manifest: project properties, milestones/progress, labels,
  activity, tabs, resources, and contextual side-panel states.
- The Roehub transition specification now requires an analogous functional
  block or an explicit justified omission for each selected reference block;
  literal geometry, taxonomy, copy, assets, and unsupported concepts are out of
  scope.

## Redaction and prohibited-capture check

- No reference PNG is tracked.
- No live screenshot, recording, trace, accessibility snapshot, HAR, browser
  profile, cookie, token, local/session storage, or authentication state is
  tracked.
- Only hashes, numeric measurements, role counts, route descriptions, and
  waiver rationale are committed.
- Workspace member data and account content are not reproduced in the evidence.

## Explicit waivers

1. The four Roehub theme IDs are not derived from the dark-only Linear source.
2. State-changing command execution was not performed.
3. Sidebar dragging was not performed because it could persist a shared account
   preference during parallel work.
4. Modal, drawer, route, and popover recordings were not retained.
5. Error, stale, forbidden, and session-expired states were not fabricated by
   changing authorization, cookies, or network/session state.
6. Linear motion timings are not claimed: no active Web Animation or non-zero
   computed CSS duration was exposed at the observation point.

Each waiver moves the corresponding behavior to later Roehub design or runtime
acceptance; none is treated as proven product behavior.

## Acceptance mapping

| Criterion | Evidence |
|---|---|
| Source hashes verified | archive and all capture hashes match |
| Viewport/scale metadata | archive pixel dimensions and absent density metadata recorded; unavailable CSS scale explicitly unknown; live CSS viewport/DPR recorded separately |
| Command/keyboard/focus | selected safe paths observed; unsafe execution and broad traversal waived |
| Sidebar geometry | live geometry and cursor recorded; drag recording waived |
| Route/pane/modal/popover evidence | static archive states plus live route/popover observations; recordings waived |
| Required server/UI states | loading and empty observed; unsafe or unavailable states waived |
| Accessibility | sanitized role counts and focus transitions recorded |
| Component geometry/motion | geometry measured; motion timing waived to the shared standard |
| Functional interpretation | analogous Roehub blocks or explicit omissions are required; literal replication is prohibited |
| Prohibited captures absent | pass |
| `git diff --check` | recorded by terminal verification before commit |

## Proof boundary

The accepted result is a sanitized reference description. Its 2026-07-20 live
observations remain historical; the archive and 16 PNG hashes were reverified
on 2026-07-22 without a Linear session. Three additional user-supplied PNGs are
identified by hash and retained only in the local reference folder, bringing
the described set to 19 captures without tracking third-party images. It may
unblock the next repository ticket only after this amendment reaches the common
`main` baseline. It does not make ROE-13 or ROE-15 ready and does not authorize
their execution or status changes.

## 2026-07-22 amendment verification

- `jq` semantic assertions pass for `19` manifest captures, exactly `3`
  supplemental captures, the accepted reference cluster, and the formal
  functional-equivalence contract.
- The three local supplemental PNGs match the recorded SHA-256 values and are
  available under
  `/Users/daniildegtyarev/.codex/visualizations/2026/07/22/019f872a-e059-7ba3-a7dd-66302c23da27/linear-reference/`.
- `git ls-files` confirms that none of the supplemental PNGs is tracked.
- `python .codex/hooks/tests/run_tests.py` passes all `11` active hook
  regressions.
- `python -m tools.docs.generate_docs_index --check` reports the architecture
  index up to date.
- `python -m tools.docs.generate_project_map --check` reports all `5`
  project-map artifacts up to date.
- `python tools/release/oss_metadata.py --check` passes.
- `python -m tools.ci.route_changes ci --changed-files <(...)` classifies the
  amendment as documentation-only: `code=false`, `docs=true`,
  `run_migrations=false`, and an empty test matrix.
- `git diff --check` passes.
- Compatibility classification: `compatible-change` for the reference and
  planning contract; runtime, API, persistence, authorization, and deployment
  behavior are unchanged.
