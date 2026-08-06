---
doc: analog-platform-ux-research
version: "1.0"
status: complete
language: en
research_date: 2026-08-06
time_horizon: current_public_sources
---

# Roehub Backtest-to-Live UX Research

## Research question

How should Roehub organize a simple but trustworthy journey from compact
Backtest setup through large variant-result analysis to strategy creation and
paper/live operation, while preserving a Linear-like low-noise interface?

This review uses current official product documentation. It evaluates workflow
patterns, not visual fidelity or business claims.

## Evidence summary

### QuantConnect

QuantConnect separates the Backtest result into runtime statistics, charts,
statistics, reports, orders/logs, and source files. Live deployment is a
separate guided configuration step for brokerage, node, data provider, initial
state, notifications, and restart behavior. Live results preserve an
out-of-sample Backtest curve for reconciliation.

Useful Roehub pattern: result inspection and deployment are connected but not
collapsed into one action. Deployment has its own preflight and live operation
keeps simulation comparison visible.

Sources:

- [Backtest results](https://www.quantconnect.com/docs/v2/local-platform/backtesting/results)
- [Deploy live algorithms](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/getting-started)
- [Live results](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/results)
- [Backtest/live reconciliation](https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/reconciliation)

### TradingView

TradingView layers one strategy report into Overview, Performance, Trades
analysis, Risk/performance ratios, and List of trades. The Overview intentionally
shows a small decision set and comparison chart; full metrics and every trade
remain available in dedicated views.

Useful Roehub pattern: progressively disclose analytical depth instead of
rendering every KPI, chart, and trade on the first screen.

Sources:

- [Strategy Report overview](https://www.tradingview.com/support/solutions/43000764138-tradingview-strategy-report-how-to-start/)
- [Pine strategy report concepts](https://www.tradingview.com/pine-script-docs/concepts/strategies/)

### StrategyQuant

StrategyQuant treats generated/tested strategies as ranked Databanks. A
Databank stores a bounded top set rather than unlimited generated strategies,
supports configurable column views and sorting, opens one result on demand,
and provides explicit Retest and Portfolio actions. Its documented workflow
moves from generation to evaluation and then robustness/retest work before use.

Useful Roehub pattern: separate the huge search population from the retained
candidate set; make ranking visible and configurable; treat selection as the
start of validation, not the end.

Sources:

- [Databanks](https://strategyquant.com/doc/strategyquant/databank/)
- [Recommended strategy workflow](https://strategyquant.com/doc/strategyquant/workflow/)

### MetaTrader 5 Strategy Tester

MetaTrader optimization exposes sortable runs, optional columns, filters for
failed/weak passes, and a double-click path from one optimization result to a
detailed single test. Forward testing re-runs a best subset on a separate
period and makes Backtest/forward comparison explicit.

Useful Roehub pattern: mass results need filtering and a deliberate
single-result drill-down; holdout/forward evidence should sit beside ranking so
that a profitable but fragile variant is not visually promoted.

Sources:

- [Strategy optimization](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization)
- [Strategy testing and forward period](https://www.metatrader5.com/en/terminal/help/algotrading/testing)

### TrendSpider

TrendSpider uses a no-code tester to create and run a strategy, then exposes
`Launch as Strategy Bot` only after the test completes. The selected strategy
is pre-populated in bot configuration. Its verification guidance uses a cloned,
read-only strategy snapshot with a locked historical date and recommends
observing control notifications before attaching execution.

Useful Roehub pattern: promotion begins from an immutable tested snapshot;
activation is pre-populated but still has a separate verification phase.

Sources:

- [Strategy Tester](https://help.trendspider.com/kb/strategy-tester/understanding-strategy-tester-from-trendspider)
- [Creating and managing Strategy Bots](https://help.trendspider.com/kb/trading-bots/trading-bots)

## Recommended Roehub synthesis

| Journey moment | Adopted pattern | Roehub adaptation |
|---|---|---|
| Configure | task-oriented quick setup | Four compact stages with irrelevant options hidden and advanced assumptions inspectable |
| Run | persistent job identity | Queue and execution progress separated; safe background execution and recovery |
| Triage | layered result report | Overview first, then Variants, Compare, selected Variant detail, and raw evidence |
| Select | bounded ranked candidate set | Retain top 10–30, show ranking rationale, shortlist candidates, never equate rank with readiness |
| Validate | forward/robustness checks | Holdout, market/timeframe, cost and parameter sensitivity as promotion evidence |
| Promote | action from tested result | `Create strategy` from immutable variant snapshot; no blank live strategy creation |
| Preflight | separate deployment configuration | Connection, paper/live mode, risk, resources, capability, recent-auth, and condition-diff review |
| Operate | live result plus simulation comparison | Live workspace with origin link, safe stop, unknown-result recovery, and reconciliation |

## Risks and rejected patterns

- Reject KPI walls: they increase comparison cost and make weak signals look
  equally important.
- Reject profit-only ranking: it rewards overfitting and hides robustness.
- Reject an all-fields wizard: the owner described a small input set; steps
  should clarify decisions, not manufacture ceremony.
- Reject one-click live launch: pre-population is valuable, but connection,
  permission, risk, and simulation/live differences need an explicit gate.
- Reject permanent three-column detail at every width: use list → inspector →
  focused detail transformations based on available space and task.
- Reject visual copying of competitor dashboards: the durable value is their
  workflow decomposition and evidence hierarchy.

## Evidence confidence

- High confidence: progressive result layering, ranked/filtered variant triage,
  separate deployment preflight, immutable test provenance, and live/backtest
  reconciliation are supported by multiple primary sources.
- Medium confidence: exact shortlist size and compare limit are Roehub-specific
  design decisions inferred from the owner's stated 10–30 retained variants.
- Not established here: user frequency, conversion impact, or runtime fitness
  of any specific Roehub control. Those require G3–G6 testing and implementation
  evidence.
