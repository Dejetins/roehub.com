# Roehub Overview — portfolio analytics requirements v1

## Authority and delivery boundary

- Status: accepted product direction and page requirements, 2026-09-26.
- Authority: the owner accepted the portfolio analytics concept and clarified
  Total, exchange, instrument and custom strategy portfolios, then requested
  requirements preparation before a separate implementation step.
- Canonical route: `/dashboard`; navigation label: Overview.
- This specification supersedes installation readiness, recent research work,
  guided technical next actions and the single-selected-strategy workstation
  as the purpose of Overview. Historical implementation evidence remains history.
- Current task: documentation only. No client, API, persistence, trading or
  authorization implementation is claimed or authorized by this document alone.
- Existing security contracts remain authoritative. These requirements do not
  grant portfolio access or introduce new trading permissions.

## OV-01 — Purpose

Overview is the primary page for monitoring portfolio capital, performance,
strategy contributions, exchange/instrument exposure and recent trading activity.
It answers: what is the portfolio worth, how has it performed, what produced
that result, where is risk concentrated, and what changed through recent trades?
Technical service health, research-job queues and setup checklists are not
primary page modules. Explain missing financial data where it affects a number;
link to the relevant recovery surface only when useful.

## OV-02 — Portfolio entity and selection

A portfolio is an identified analytics entity with a defined membership scope.
It can span multiple strategies, exchanges, accounts and instruments. It is not
restricted to a set of accounts, and selecting or saving it does not move funds,
change strategy allocation, start/stop strategies or submit trading commands.

The selector supports:

| Type | Membership |
|---|---|
| Total | All strategies and their eligible financial records in the user's authorized scope and selected Live/Paper mode |
| Exchange | Strategy activity and attributable capital for the selected exchange across eligible accounts |
| Instrument | Strategy activity and attributable capital for the selected instrument; cross-exchange when instrument identities are economically compatible |
| Custom | An explicitly saved set of strategies, potentially across exchanges and instruments |

- Built-in scopes update as eligible strategies/activity change. Custom scopes
  change through explicit membership edits; they do not silently include newly
  created strategies.
- Users can create, name, edit membership of, rename and delete their custom
  portfolios. Deleting a portfolio removes its analytics definition only.
- Persist custom definitions and restore the selected portfolio after reload
  when it remains accessible. Provide a clear fallback when it no longer exists.
- Membership selection exposes strategy identity, exchange and instrument;
  identical display names must not cause ambiguous selection.
- Archived/stopped strategies retain their attributable historical results.
- Exchange/instrument scopes include only matching financial records, not the
  entire result of a multi-exchange or multi-instrument strategy.
- A strategy appears once per custom membership. Overlapping portfolios are
  alternative views; never sum their totals into Total.
- Common-account cash, manual trading and other unattributed balances/results
  have an explicit reconciliation bucket in account-backed Total/exchange views.
  Never silently assign them to a strategy or include all account equity in each
  custom/instrument portfolio. Show the coverage/basis of each total.

## OV-03 — Shared context

One visible context controls all page modules: portfolio, Live/Paper mode,
reporting currency and historical period. Optional exchange/account/instrument/
strategy drill-down filters are visible and resettable without editing membership.
Support day, week, month, year-to-date, all available history and a custom range.
Current Equity, positions and exposure are labeled as current/as-of; period P&L,
returns, contributions and historical activity use the selected range. A period
change must not pretend current positions are historical holdings.
Live, Paper and backtest results are never combined in portfolio totals.

## OV-04 — Summary

A compact summary presents current Equity, net period P&L, period return,
current drawdown, gross/net exposure relative to capital, and available funds
where the selected scope supports a meaningful value. Unsupported values remain
unavailable with a concise reason, never zero or an invented allocation.
P&L detail separates realized result, change in unrealized result, commissions,
funding and other supported costs/income with no double counting. Current open
unrealized P&L is distinct from its change over the reporting period.

## OV-05 — Capital and performance history

- One primary ECharts chart with Equity, Return and Drawdown modes.
- Default: selected portfolio aggregate. Optional comparisons by exchange,
  account or strategy, with selectable series and legible legends/tooltips.
- Absolute values explain monetary scale; normalized percentage performance
  supports comparison across different capital sizes.
- Mark external deposits/withdrawals and internal transfers on Equity history.
- Return uses a documented cash-flow-adjusted time-weighted methodology (TWR).
  Derive performance drawdown from the cash-flow-adjusted wealth series so a
  withdrawal does not appear as a trading loss. Show current and period maximum
  drawdown with explicit labels; distinguish their time windows.
- Provide compact monthly returns, including clearly marked partial months and
  unavailable intervals. Align chart, monthly values and summary methodology.
- Consistent timestamps, timezone, valuation basis and reporting currency apply
  across all series. Gaps are not zero values or fabricated continuous history.

## OV-06 — Strategy contributions

A sortable table presents strategy identity, attributed capital and share,
net period P&L, own return, contribution to portfolio return in percentage points,
period drawdown and current exposure. Exchange/instrument context is inspectable.
Capital share, own return and contribution are distinct concepts with separate
labels and calculation bases. Do not use share of net total profit as the main
contribution metric; it becomes unstable around zero total profit.
Select a row to inspect its history and related positions/trades, with a direct
link to Strategies detail. Preserve portfolio context on return.
Show manual/unattributed results and costs separately where applicable so the
breakdown reconciles to the total. Missing capital attribution prevents a valid
strategy return; do not substitute full account equity or infer arbitrary weights.

## OV-07 — Allocation and risk

Exchange/account breakdown: Equity, share of portfolio capital, net period P&L,
available funds, margin usage where applicable, and open exposure.
Instrument breakdown: gross and net exposure, Long/Short components, concentration
and cross-strategy overlap. Economically distinct contracts remain distinguishable.
Offsetting positions must not hide gross leverage, costs or margin obligations.
Risk notices concern money and exposure: configured concentration/drawdown limits
and meaningful margin constraints. Do not invent thresholds or a synthetic health
score. Exchange margin rules remain exchange/account-specific; unavailable metrics
are not inferred from unrelated balances.

## OV-08 — Positions and recent trades

- Positions and Recent trades are adjacent/tabbed views below the main analysis.
- Positions: instrument, strategy, exchange/account, side, quantity, current
  notional, unrealized P&L and exposure share. Aggregate by instrument with
  drill-down to constituent positions, preserving Long/Short visibility.
- Recent trades: timestamp, instrument, strategy, exchange/account, operation
  (open/increase/reduce/close), quantity, price, fees and realized P&L when incurred.
- Group partial fills by order with an expandable execution breakdown; do not
  equate one fill or order with an entire completed position lifecycle.
- Support bounded pagination, linked context filters and detail navigation.
  These views are observational; manual trading commands are not required here.

## OV-09 — Financial integrity and data quality

- Equity means valued assets minus liabilities with open-position P&L accounted
  for exactly once under the relevant account model. No universal spot/derivative
  shortcut and no double counting of collateral and derivative notional.
- An internal transfer is neutral for a portfolio containing both ends, but is
  a flow for a scope containing only one end. Transfer fees remain costs.
- Strategy allocation, reallocation and shared-account attribution require
  explicit historical records/rules. Present their basis; do not invent history.
- Report currency conversion at appropriate historical valuation times. USD,
  USDT and other assets retain distinct identities; no implicit 1:1 assumption.
- Totals reconcile to strategy/manual/unattributed components and supported
  costs/valuation effects. Expose unexplained differences, never distribute them
  silently among strategies. Period contribution linking must reconcile to the
  aggregate return rather than summing incompatible standalone returns.
- Retain the last valid values on transient failure with as-of time and partial
  coverage indication. Missing an exchange must not create a fictitious loss.
- Server-side access filtering applies before aggregation, series and membership
  choices; no unauthorized data may leak through totals or hidden constituents.
- Missing history, incomplete attribution and insufficient observations remain
  explicit. Financial totals are not marked complete merely because HTTP succeeded.

## OV-10 — Interaction and visual requirements

Use the current refined Backtests and Strategies surfaces, compact controls,
colors, typography, chart patterns and normal motion. The old v23 specimen is
not a visual authority. Preserve RU/EN, keyboard access, 820/1024/1440 layouts,
zoom and reduced-motion support from the shared UI contract.

Keep summary, primary chart, contribution analysis, allocation/risk and activity
in that reading order. Avoid a wall of equally prominent charts and metrics.
Refresh financial data in place without resetting period, filters, chart state or
scroll. Switching uncached scopes keeps layout stable and displays a spinner with
localized “Loading data…” / “Загрузка данных…”. Prevent stale-response races: data
from the previous scope must never appear as the new scope's totals. Partial
failures remain local to affected modules. No routine technical status banners,
redundant page labels or duplicated refresh controls.

Required states: initial loading, background refresh, ready, empty portfolio,
no matching activity, unavailable metric/history, partial/degraded coverage,
stale last-valid data, retryable error, permission-filtered data, custom portfolio
editing/saving/validation failure and deleted/inaccessible selection recovery.

## OV-11 — Scope limits

Detailed Sharpe/Sortino, correlation matrices and distribution analysis may be
future drill-down work; they are not required first-screen modules. Benchmark
comparison, portfolio optimization, automated rebalancing, cross-user sharing and
manual order entry are not implied by this specification. Shell search,
notifications and technical monitoring keep their separate product requirements.

## OV-12 — Implementation entry decisions and proof

The accepted product behavior above is settled. Before implementing affected
calculations/persistence, the next task must specify:

1. Portfolio identity/storage and APIs, membership revisions, and historical
   behavior after membership edits (recomputed selected basket versus effective-
   dated composition), clearly labeled to the user.
2. Source of historical capital allocations and strategy attribution on shared
   accounts; scope eligibility for cash, available funds and instrument returns.
3. Financial event/snapshot sources, coverage, valuation timestamps, costs,
   currency conversion, transfer matching, TWR/contribution/drawdown algorithms.
4. Authorized read/mutation contracts for personal portfolio definitions using
   existing policy; page requirements do not decide new role grants.

These are explicit technical/domain entry decisions, not claims that existing
DashboardSummaryResponse already supports portfolio analytics. The old dashboard
has partial strategy/run reads, but its financial series and several panels are
unavailable. A visual fixture cannot prove financial correctness or integration.

Acceptance scenarios for the implementation:

- Total, exchange, instrument and a custom cross-exchange/cross-instrument basket
  produce correctly scoped, deduplicated summary/series/contributions/activity.
- Custom portfolio create/edit/rename/delete survives reload without mutating
  strategies/trading; deleted or inaccessible selections recover cleanly.
- Deposits/withdrawals, internal transfers, fees, funding, unrealized changes and
  currency valuation reconcile without false profit or drawdown.
- Shared-account/multi-instrument strategies cannot double-count account capital;
  unavailable attribution is explicit; membership-history policy is verified.
- Stopped/archived strategies remain in relevant history; overlapping custom
  portfolios, partial fills and offsetting positions are handled correctly.
- Unauthorized constituents never affect exposed aggregates. Live/Paper/backtest
  separation is verified against real authorized integration boundaries.
- Cold/cached switching, concurrent refresh, incomplete exchange history, source
  failure, currency/period changes and retry preserve coherent scope and layout.
- Real browser verification covers RU/EN, keyboard, target widths and reduced
  motion in the accepted current style. Financial checks use deterministic event
  scenarios; runtime proof remains separate from synthetic visual demonstration.

## Documentation change evidence

2026-09-26: replaced the old Overview purpose in the functional contract, current
registry, baseline IA/screen pointer and UI-HOME plan. Historical source inventory,
old implementation reports and access policy remain evidence, not competing page
requirements. Review mode: cold self-review; requirements consistent with the
owner's accepted concept. Proof boundary: documentation only; runtime compatibility
is unchanged. Remaining risk: the four implementation entry decisions above.
