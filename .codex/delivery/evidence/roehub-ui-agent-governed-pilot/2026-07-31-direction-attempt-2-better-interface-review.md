# Direction attempt 2 — better-interface review

## Scope and Coverage

- Mode: `full`.
- Artifact: `Roehub Graphite Workspace` final raster after two bounded repairs.
- Local identity:
  `/Users/daniildegtyarev/.codex/generated_images/019fb4fa-5eba-7fa0-a94f-ab5e05510b4d/exec-442576f8-7587-4124-954c-f9553be012de.png`.
- Actual raster size: `1586 x 992`; aspect ratio is suitable for normalized `1440 x 900`
  direction review.
- Scope boundary: static visual language and brief-content inventory. This review does not claim
  Figma structure, runtime semantics, browser behaviour, or production accessibility.

| Domain | Evidence inspected | Result |
|---|---|---|
| Accessibility | Visible control labels, status text/icon redundancy, selected/degraded state cues, static hit silhouettes | Clear within static direction boundary |
| Layout | Workspace shell, page identity, contextual toolbar, active filters, list hierarchy, one job row, and contextual pane | Clear |
| Writing | Page/view labels, filter labels, refresh labels, degraded copy, lifecycle copy, and pane properties | Clear |
| Typography | Hierarchy, Inter-like rendering, metadata floor, tabular metric alignment, and dense-row scanning | Clear |
| Colors | Graphite surface steps, restrained Roehub brand cue, selected state, completion and degraded semantics | Clear visually; exact contrast deferred to tokens |
| UI | Continuous panels, restrained radius, icon consistency, selection treatment, separator use, and absence of decorative effects | Clear |

## Findings

No actionable interface findings within the declared static direction-review boundary.

## Considered but Rejected

| Location | Candidate | Rejected because |
|---|---|---|
| Job list | Restore the rounded container from the first repair | A flat list row preserves workspace continuity and avoids dashboard-card styling |
| Toolbar | Expose every filter as a permanently labelled field | Active chips plus one filter entry point retain state visibility with substantially less noise |
| Selected row | Use a copper/orange outline for stronger selection | The neutral surface change and leading/status cues are sufficient; an accent outline would overload the brand colour |
| Detail pane | Add chart or trade preview to reduce empty space | Those surfaces are outside the accepted pilot content and would turn visual polish into product invention |

## Verification

Passed:

- inspected the final raster at its original `1586 x 992` size;
- matched all required toolbar fields: query, state, exchange, market type, symbol, launched date
  range, manual refresh, auto-refresh preset, and refresh status;
- matched all required row data: identity, market, setup, six result metrics, lifecycle, created
  time, and degraded freshness;
- matched the required selected-job pane identity, context, completion, freshness, timestamp, and
  close control;
- confirmed status is not communicated by colour alone;
- confirmed the hard anti-pattern gate has no observed violation;
- confirmed the first generic directions and intermediate repair remain rejected sources;
- applied all six owner skills through `better-interface` after the final repair.

Reference material inspected:

- repository `linear-workspace-ui-transition-standard-v1.md`;
- repository `linear-workspace-reference-measurements-v1.md`;
- official Linear redesign reference:
  `https://linear.app/now/how-we-redesigned-the-linear-ui`;
- current official Linear list and filter images downloaded temporarily from `webassets.linear.app`
  for inspection only; they are not project assets and are not committed.

Not verified and outside this checkpoint:

- exact token contrast, gamut, light themes, and variable bindings;
- accessible names, keyboard traversal, focus return, screen-reader output, zoom/reflow,
  localization extremes, motion, and performance;
- any Figma node, component, style, variable, instance, or publication state.

## Verdict

Approve

## Product-owner override

The static gate above did not constitute product acceptance. On `2026-07-31` the product owner
explicitly rejected `DIR-001` completely because it did not capture the intended design concept.
The verdict is therefore retained only as evidence that the agent-authored gate was insufficient;
it cannot authorize reuse of the raster, frame, doctrine v1, or any visible pattern from them.
