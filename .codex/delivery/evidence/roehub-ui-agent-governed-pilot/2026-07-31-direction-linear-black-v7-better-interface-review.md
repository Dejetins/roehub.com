# Linear-black Backtests Workbench v7 — full interface-craft review

## Boundary

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Status: `awaiting_product_owner_direction_selection`.
- Doctrine: `docs/architecture/ui/roehub-linear-black-backtests-workbench-doctrine-v7.md`.
- Visual authority: the product-owner-preferred v6 topology and black Linear-style language.
- Repair scope: control geometry, icon geometry, typography, refresh interaction, chart-type
  selector, and the platform-status line only.
- Deterministic specimen:
  `specimens/2026-07-31-linear-black-workbench-v7-repair-2.html`.
- Specimen SHA-256: `02c54d7f17f980cda7e88a36c77a512a33015b3c2dd2f395cb87247b7fb15368`.
- Review raster: `images/2026-07-31-linear-black-workbench-v7-repair-2.png`
  (`1672 × 941`).
- Raster SHA-256: `8def0c403dad0bf599510aade9dd4e192ebfe3c9658a61fe35769f5a67c4ddb6`.
- Rendering method: local deterministic HTML, CSS, and inline SVG; no image-generation pass.
- Review method: holistic interface-craft audit across accessibility, layout, writing,
  typography, colors, and UI. The standalone `better-interface` orchestrator was unavailable in
  the active skill catalog, so its available domain gates were run directly.
- Proof boundary: screenshot-led direction review plus local specimen DOM and keyboard evidence.
  This is not canonical Figma structure, Roehub runtime, responsive, permission, API, production
  data, screen-reader, or implementation proof.

## Repair 2 decision

Repair 2 is the final bounded repair permitted by the ticket cycle. It preserves the complete v6
composition and replaces unreliable generated geometry with one deterministic control component:

- `--control-h: 28px` for Search, refresh, interval, notification, Jobs actions, result actions,
  analysis tabs, chart-type buttons, expand, timeframe, and overflow;
- `14px × 14px` SVG icon boxes on a square `0 0 24 24` view box with one `1.5px` stroke strategy;
- normal-width system text using the SF/Inter-compatible stack, `font-stretch: 100%`,
  `12px/14px` compact controls, tabular numeric values, and no text transforms;
- a true `22px` edge-to-edge status line;
- closed default refresh state with icon-only immediate refresh and `15s` menu trigger;
- line, candlestick, and area chart-type controls before expand, `1h`, and overflow.

## Holistic review

| Area | Repair 2 evidence | Verdict |
|---|---|---|
| Accessibility | native buttons; accessible names for icon-only controls; `aria-expanded` and menu roles; selected-state semantics; visible `2px` focus ring; `28px` targets; arrow-key navigation and roving tabindex for menus, tabs, and chart type; Escape restores focus; status meaning is textual | pass within local specimen; screen readers, zoom/reflow, forced colors, and production integration remain unverified |
| Layout | every targeted control measures exactly `28px`; every targeted SVG box measures `14px × 14px`; `Filters`, `New backtest`, overflow, `Save`, `Export CSV`, `Create strategy`, and result overflow share `top=69`, `bottom=97`, `y-centre=83`; each tab and chart-toolbar row has one shared y-centre | pass |
| Writing | resting refresh chrome contains no visible `Refresh` or `Auto`; interval remains `15s`; menu contains `Off`, `5s`, `15s`, `30s`, `1m`, separator, and `Refresh now`; all platform states and counts are explicit | pass |
| Typography | computed compact-control style is `12px/14px`, weight `500`, `font-stretch: 100%`, `transform: none`; full job names and monthly totals have no DOM text clipping; all changing values use tabular figures | pass |
| Colors | neutral-black surfaces and restrained violet, green, amber, and red semantics are preserved; sampled WCAG contrast ratios range from `5.24:1` to `16.19:1`; color is not the only status cue | pass within sampled raster pairs |
| UI | one control radius and lattice; one icon stroke strategy; selected Equity and Line states are explicit; no gradients, glassmorphism, inflated pills, or mixed icon aspect ratios were introduced | pass |

No actionable interface-craft findings remain in the named raster at the review viewport.

## Geometry evidence

Browser-observed values at `1672 × 941`:

| Lattice | Controls | Observed geometry |
|---|---|---|
| Global header | Search, refresh, `15s`, notification | `height=28`, `top=13.5`, `bottom=41.5`, `y-centre=27.5` |
| Jobs and result headers | Filters, New backtest, both overflows, Save, Export CSV, Create strategy | `height=28`, `top=69`, `bottom=97`, `y-centre=83` |
| Analysis tabs | Overview, Equity, Drawdown, Monthly, Trades | `height=28`, `top=115`, `bottom=143`, `y-centre=129` |
| Chart toolbar | Line, Candlestick, Area, expand, `1h`, overflow | `height=28`, `top=222`, `bottom=250`, `y-centre=236` |
| Platform status | complete status inventory | `height=22`, `top=919`, `bottom=941`, `scrollWidth=clientWidth=1672` |

The browser reported exactly one targeted control height (`28`) and exactly one targeted icon box
(`14x14`). The clipping audit returned an empty set for job titles, job metadata, queue status,
variant parameters, and monthly-return cells.

## Contrast evidence

| Pair | WCAG ratio |
|---|---:|
| compact control text / neutral control | `16.19:1` |
| primary action text / violet primary | `5.24:1` |
| selected tab text / violet selected | `6.09:1` |
| metric labels / metric surface | `7.24:1` |
| job metadata / job surface | `7.33:1` |
| status text / application chrome | `8.09:1` |
| positive values / table surface | `9.39:1` |
| negative values / table surface | `5.91:1` |

## Retained-content gate

- exact navigation inventory retained: pass;
- `Jobs 6` with `Running 2`, `Queued 1`, and `Completed 3`: pass;
- Filters and `New backtest` remain inside Jobs: pass;
- `Variants 12` and ten visible variant rows: pass;
- modes remain `Overview`, `Equity`, `Drawdown`, `Monthly`, and `Trades`: pass;
- actions remain `Save`, `Export CSV`, `Create strategy`, and overflow: pass;
- the Equity chart, current curve shape, buy/sell markers, Monthly returns, and Parameters remain:
  pass;
- complete platform-status fixture remains on one baseline: pass;
- no `Symbol`, `Run strategy`, `Notebooks`, `Market`, Risk summary, or Recent trades: pass;
- black neutral Linear visual language retained: pass by visual observation.

## Interaction evidence

- Refresh menu opens from `Auto-refresh interval: 15 seconds`, focuses selected `15s`, exposes the
  required seven menu entries including the separator, and reports `aria-expanded=true`.
- Arrow keys move focus within the menu; Escape closes it, reports `aria-expanded=false`, and
  restores focus to the interval trigger.
- ArrowRight from selected Equity moves the selected state, roving `tabIndex=0`, and focus to
  Drawdown; the test restored Equity afterwards.
- ArrowRight from selected Line moves the checked state, roving `tabIndex=0`, and focus to
  Candlestick; the test restored Line afterwards.
- The default review raster was captured after reload with the refresh menu closed, Equity
  selected, and Line selected.

## Bounded repair audit

1. The first v7 image-generation attempt is `rejected_by_product_owner` and remains historical
   comparison evidence only.
2. Repair 1 improved density but remained `rejected_before_owner_review`: tabs and chart controls
   were visibly taller than `Save` / `Filters`, and the type still appeared condensed.
3. Repair 2 did not invoke image generation. It reconstructed the unchanged screen in deterministic
   HTML/CSS/SVG and fixed only the v7 doctrine surfaces.
4. The final repair preserved the functional inventory and passed the automatic geometry,
   clipping, keyboard, contrast, console, repository-validator, and focused-test gates.
5. No further automatic repair remains in this cycle. The next state change requires an explicit
   product-owner decision naming this raster.

## Repository verification

- `uv run python -m tools.docs.generate_docs_index --check`: pass; index is up to date.
- `uv run python -m tools.design.validate_roehub_ui_delivery`: pass.
- `uv run python -m tools.delivery.validate_roehub_delivery_model`: pass.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_delivery_model.py`:
  `14 passed in 0.15s`.
- `git diff --check`: pass.
- Playwright local specimen console: `0` errors, `0` warnings.

## Residual risk

- Product-owner acceptance is pending and cannot be inferred from these checks.
- Exact Figma node construction, published component bindings, canonical file/page/node identity,
  design-library publication, and the later pilot checkpoints are not attempted.
- Roehub runtime behavior, real data, permissions, network requests, responsive behavior, 200%
  zoom/reflow, touch ergonomics, screen readers, and browser support remain unverified.

## Verdict

Interface-craft gate: `Pass`.

Product-owner checkpoint: `awaiting_product_owner_direction_selection`.
