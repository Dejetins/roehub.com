# Linear-black Backtests Workbench v8 — interface review

## Boundary

- Date: `2026-07-31`.
- Checkpoint: `direction_selection`.
- Direction ID: `linear_black_backtests_workbench_v8`.
- Status: `awaiting_product_owner_direction_selection`.
- Owner refinement:
  `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v8-owner-refinement.md`.
- Deterministic specimen:
  `specimens/2026-07-31-linear-black-workbench-v8.html`.
- Specimen SHA-256: `5a817da8031e5b866e1425df07d59f1e518d61143872a05776ec32fc09aa7707`.
- Review raster: `images/2026-07-31-linear-black-workbench-v8.png` (`1672 × 941`).
- Raster SHA-256: `91e095a664f6f4aee1bdfd1bce026f91ac33423611b72d2f8a111ccec42e997d`.
- Rendering method: local deterministic HTML, CSS, and inline SVG; no image generation.
- Review method: `better-layout`, `better-ui`, `better-typography`, `better-colors`,
  `better-accessibility`, and `browser-qa-evidence` with the `playwright-cli` mechanic.
- Proof boundary: local raster, DOM geometry, contrast, accessible-name, keyboard, clipping,
  console, and repository evidence only. No canonical Figma, product runtime, responsive,
  permission, API, production-data, screen-reader, or implementation claim.

## Owner-refinement gate

| # | Requirement | Observed result | Verdict |
|---:|---|---|---|
| 1 | `roehub` without leading mark; Search and Notifications directly after it; no chevron | brand group contains text plus two borderless icon-only buttons; no preceding mark or chevron exists | pass |
| 2 | align `Backtests` under `Jobs` | browser reports `page-title.left=146` and `Jobs.left=146` | pass |
| 3 | exactly four Jobs buttons: refresh, classic filter, `+`, overflow | Jobs toolbar contains four `28 × 28` icon-only controls; Filters uses a horizontal-adjustments glyph | pass |
| 4 | analysis tabs below KPI summary | DOM and visual order is detail header → KPI summary → analysis toolbar | pass |
| 5 | remove visible Buy/Sell/Buy-and-hold legend | `.legend` count is `0`; no visible legend block remains | pass |
| 6 | Create strategy first; icon-only diskette Save and down-arrow Export CSV | result action order and accessible names match exactly; Save and Export have empty visible text | pass |
| 7 | extend workspace to platform status | `workspace.bottom=919`, `detail.bottom=919`, and `status.top=919` | pass |
| 8 | one-decimal Return/Drawdown; no Monthly semantic fills | KPI, variant, and monthly percentages use one decimal; positive, negative, and total monthly cells all compute to `rgb(21, 26, 31)` | pass |
| 9 | variants as tall as jobs | all job rows and all variant rows measure `50px` | pass |
| 10 | visible Failed job state | `Failed 1` group includes `Failed · invalid period` plus an x-circle icon | pass |
| 11 | lighter neutral-gray working surfaces with correct contrast | panels use `#171c21` → `#13181d`; sampled contrast ranges from `5.24:1` to `15.42:1` | pass |

## Geometry evidence

| Lattice | Controls | Browser-observed geometry |
|---|---|---|
| Platform brand group | Search, Notifications | `height=28`, `top=13.5`, `bottom=41.5`, `y-centre=27.5` |
| Jobs toolbar | Refresh, Filters, New backtest, overflow | each `28 × 28`, `top=69`, `bottom=97`, `y-centre=83` |
| Result actions | Create strategy, Save, Export CSV, overflow | each `height=28`, `top=69`, `bottom=97`, `y-centre=83`; icon-only actions are `28 × 28` |
| Analysis toolbar | five tabs plus chart-type, expand, timeframe, overflow | each `height=28`, `top=169`, `bottom=197`, `y-centre=183` |
| Workspace | Jobs, Variants, selected result | `top=56`, `bottom=919`, exactly meets the `22px` status line |

- Targeted control heights: one unique value, `28`.
- Non-header toolbar icon boxes: one unique value, `14x14`.
- Job-row heights: one unique value, `50`.
- Variant-row heights: one unique value, `50`.
- Text clipping audit across job identity/meta/state, variants, Monthly returns, and detail identity:
  empty set.
- Body and viewport both measure `1672 × 941`; no page overflow.

## Contrast evidence

| Pair | WCAG ratio |
|---|---:|
| control text / neutral control | `13.74:1` |
| primary-action text / violet primary | `5.24:1` |
| panel text / gray panel | `15.42:1` |
| muted text / gray panel | `8.56:1` |
| job metadata / job surface | `6.60:1` |
| platform status / status chrome | `7.59:1` |
| positive value / neutral monthly surface | `9.50:1` |
| negative value / neutral monthly surface | `6.70:1` |
| Failed explanation / job surface | `9.24:1` |

Color remains supportive rather than exclusive: positive/negative values retain signed numbers,
and the Failed state includes explicit text and an icon.

## Accessibility and interaction evidence

- Browser accessibility snapshot exposes Search, Notifications, Refresh jobs, Filters,
  New backtest, Save, Export CSV, both overflow controls, chart type, expand, and timeframe with
  descriptive names.
- All icon-only controls are native buttons and meet the doctrine's `28 × 28` compact target.
- Activating Refresh jobs updates the stable polite live region to `Jobs refreshed`.
- ArrowRight from selected Equity moves selected state, focus, and roving `tabIndex=0` to Drawdown;
  the test restored Equity afterwards.
- ArrowRight from checked Line moves checked state, focus, and roving `tabIndex=0` to Candlestick;
  the test restored Line afterwards.
- Focus-visible styling remains a `2px` outline with offset.
- Browser console after final reload: `0` errors, `0` warnings.
- Screen readers, forced colors, 200% zoom/reflow, touch ergonomics, and production integration were
  not tested because this is a fixed-viewport direction specimen.

## Retained-content gate

- navigation inventory retained: pass;
- Running, Queued, Completed, and the new Failed job state are visible: pass;
- `Variants 12` and ten visible rows retained: pass;
- KPI summary, Equity chart and markers, Monthly returns, and Parameters retained: pass;
- `Overview`, `Equity`, `Drawdown`, `Monthly`, and `Trades` retained: pass;
- `Create strategy`, Save, Export CSV, and overflow retained in the requested order: pass;
- platform-status inventory and one-line baseline retained: pass;
- no forbidden `Symbol`, `Run strategy`, `Notebooks`, `Market`, Risk summary, or Recent trades:
  pass.

## Repository verification

- `uv run python -m tools.docs.generate_docs_index --check`: pass; index is up to date.
- `uv run python -m tools.design.validate_roehub_ui_delivery`: pass.
- `uv run python -m tools.delivery.validate_roehub_delivery_model`: pass.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_delivery_model.py`:
  `14 passed in 0.14s`.
- `git diff --check`: pass.

## Residual risk

- Product-owner acceptance is pending and cannot be inferred from automated or visual checks.
- Exact Figma node construction, published library bindings, and later pilot checkpoints are not
  attempted.
- Roehub runtime behavior, real data, permissions, responsive layouts, performance, release, and
  production behavior remain unverified.

## Verdict

Interface-craft and local browser QA gate: `Pass`.

Product-owner checkpoint: `awaiting_product_owner_direction_selection`.
