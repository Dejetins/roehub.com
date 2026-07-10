import { useMemo, useState } from "react";
import {
  Pulse,
  ArrowRight,
  Bell,
  Brain,
  ChartLineUp,
  CheckCircle,
  ClockCounterClockwise,
  CloudWarning,
  Code,
  Database,
  DownloadSimple,
  EyeSlash,
  Gear,
  GitBranch,
  Hourglass,
  Info,
  Lightning,
  LockKey,
  MagnifyingGlass,
  Pause,
  Play,
  Plus,
  RocketLaunch,
  ShieldCheck,
  SlidersHorizontal,
  Sparkle,
  Stop,
  TestTube,
  TrendDown,
  TrendUp,
  UserCircle,
  Warning,
  XCircle,
} from "@phosphor-icons/react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  EmptyState,
  EntityLink,
  ErrorState,
  Field,
  FilterButton,
  InspectorRow,
  LoadingState,
  Metric,
  MoreButton,
  PageHeader,
  Panel,
  PortfolioChart,
  ProgressBar,
  SearchField,
  Select,
  StatusBadge,
  Table,
  Tabs,
  Toolbar,
} from "./components.jsx";
import {
  connections,
  dataHealth,
  entities,
  executions,
  experiments,
  models,
  notifications,
  portfolioSeries,
  positions,
  programMapGroups,
  signals,
  strategies,
  themes,
} from "./data.js";

const spark = [
  { value: 12 }, { value: 18 }, { value: 16 }, { value: 24 }, { value: 22 }, { value: 31 }, { value: 38 },
];

const compactTooltip = {
  background: "var(--surface-raised)",
  border: "1px solid var(--border-strong)",
  borderRadius: 6,
  color: "var(--text)",
  fontSize: 12,
};

function toneForState(state) {
  if (["Running", "Healthy", "Completed", "Production", "Active"].includes(state)) return "success";
  if (["Delayed", "Queued", "Candidate", "Degraded"].includes(state)) return "warning";
  if (["Stopped", "Restricted", "Error"].includes(state)) return "danger";
  return "info";
}

const strategyColumns = (onOpen) => [
  { key: "name", label: "Strategy", render: (row) => <button className="table-primary" onClick={(event) => { event.stopPropagation(); onOpen?.(row); }}>{row.name}</button> },
  { key: "venue", label: "Venue" },
  { key: "state", label: "State", render: (row) => <StatusBadge tone={toneForState(row.state)}>{row.state}</StatusBadge> },
  { key: "pnl", label: "P&L (USD)", render: (row) => <span className={row.pnl.startsWith("-") ? "text-danger" : "text-success"}>{row.pnl}</span> },
  { key: "pnlPct", label: "P&L (%)", render: (row) => <span className={row.pnlPct.startsWith("-") ? "text-danger" : "text-success"}>{row.pnlPct}</span> },
  { key: "drawdown", label: "Drawdown" },
  { key: "updated", label: "Updated" },
];

export function OverviewPage({ onNavigate, onSelectStrategy }) {
  const [chartRange, setChartRange] = useState("1d");
  const [experimentState, setExperimentState] = useState("all");
  const visibleExperiments = experiments.filter((item) => experimentState === "all" || item.state.toLowerCase() === experimentState);
  return (
    <div className="page page-overview">
      <PageHeader
        title="Portfolio overview"
        description="Real-time portfolio health and strategy performance"
        actions={(
          <>
            <time className="button button-secondary" dateTime="2026-07-10">10 Jul 2026</time>
            <button className="button button-primary" onClick={() => onNavigate("backtests")}><Play size={15} weight="fill" />Run backtest</button>
          </>
        )}
      />

      <div className="metric-grid metric-grid-five">
        <Metric label="Net equity (USD)" value="1,248,732" delta="+3.42%" tone="success" data={spark} meta="Fresh 12s ago" />
        <Metric label="Day P&L (USD)" value="+18,573" delta="+1.51%" tone="success" data={spark.slice().reverse()} meta="Fresh 12s ago" />
        <Metric label="Exposure" value="1.27" delta="+0.18" tone="info" data={spark.slice(1)} meta="32.3% of equity" />
        <Metric label="Max drawdown (30D)" value="-6.35%" delta="-0.42%" tone="danger" data={spark.map((item) => ({ value: 48 - item.value }))} meta="Below 7% soft limit" />
        <Metric label="Active runs" value="7" delta="of 12" data={spark.slice(2)} meta="1 degraded source" />
      </div>

      <Panel
        title="Equity & P&L"
        className="overview-chart-panel"
        action={(
          <Toolbar>
            <Tabs compact label="Chart range" active={chartRange} onChange={setChartRange} items={[{ id: "1h", label: "1H", disabled: true }, { id: "4h", label: "4H", disabled: true }, { id: "1d", label: "1D" }, { id: "1w", label: "1W", disabled: true }, { id: "1m", label: "1M", disabled: true }]} />
            <MoreButton />
          </Toolbar>
        )}
      >
        <PortfolioChart />
      </Panel>

      <div className="content-grid content-grid-two">
        <Panel title="Active strategies" action={<FilterButton>Filter</FilterButton>}>
          <Table columns={strategyColumns(onSelectStrategy)} rows={strategies} rowKey="id" onRowClick={onSelectStrategy} />
          <EntityLink onClick={() => onNavigate("strategies")}>View all strategies</EntityLink>
        </Panel>
        <Panel title="Recent experiments" action={<Tabs compact label="Experiment state" active={experimentState} onChange={setExperimentState} items={[{ id: "all", label: "All" }, { id: "queued", label: "Queued" }, { id: "running", label: "Running" }]} />}>
          <Table
            rows={visibleExperiments}
            rowKey="id"
            columns={[
              { key: "name", label: "Experiment", render: (row) => <span className="table-primary">{row.name}</span> },
              { key: "state", label: "State", render: (row) => <StatusBadge tone={toneForState(row.state)}>{row.state}</StatusBadge> },
              { key: "progress", label: "Progress", render: (row) => row.progress ? <ProgressBar value={row.progress} /> : "—" },
              { key: "score", label: "Score" },
              { key: "owner", label: "Owner" },
              { key: "updated", label: "Updated" },
            ]}
          />
          <EntityLink onClick={() => onNavigate("backtests")}>View all experiments</EntityLink>
        </Panel>
        <Panel title="Signals & anomalies" action={<FilterButton>Timeline</FilterButton>}>
          <Table
            rows={signals}
            columns={[
              { key: "time", label: "Time" },
              { key: "event", label: "Event", render: (row) => <span><i className={`signal-dot signal-${row.level}`} />{row.event}</span> },
              { key: "strategy", label: "Strategy" },
              { key: "symbol", label: "Symbol" },
              { key: "side", label: "Side", render: (row) => <span className={row.side === "Buy" ? "text-success" : row.side === "Sell" ? "text-danger" : ""}>{row.side}</span> },
              { key: "impact", label: "Impact" },
            ]}
          />
          <EntityLink onClick={() => onNavigate("live")}>View all signals</EntityLink>
        </Panel>
        <Panel title="Data health" action={<MoreButton />}>
          <Table
            rows={dataHealth}
            rowKey="source"
            columns={[
              { key: "source", label: "Source" },
              { key: "status", label: "Status", render: (row) => <StatusBadge tone={toneForState(row.status)}>{row.status}</StatusBadge> },
              { key: "latency", label: "Latency" },
              { key: "updated", label: "Last update" },
            ]}
          />
          <EntityLink onClick={() => onNavigate("connections")}>Manage connections</EntityLink>
        </Panel>
      </div>
    </div>
  );
}

function StrategyLibrary({ onSelect }) {
  const [query, setQuery] = useState("");
  const [state, setState] = useState("all");
  const filtered = strategies.filter((item) => item.name.toLowerCase().includes(query.toLowerCase()) && (state === "all" || item.state.toLowerCase() === state));
  return (
    <div className="workspace-stack">
      <Toolbar className="workspace-toolbar">
        <SearchField value={query} onChange={setQuery} placeholder="Search strategies, symbols or venues…" />
        <Select value={state} onChange={setState} aria-label="Filter strategy state">
          <option value="all">All states</option><option value="running">Running</option><option value="stopped">Stopped</option>
        </Select>
        <button className="button button-primary" disabled title="Strategy creation uses the production contract"><Plus size={15} />New strategy</button>
      </Toolbar>
      <div className="strategy-library-grid">
        {filtered.map((strategy) => (
          <article key={strategy.id} className="strategy-card">
            <div className="strategy-card-top"><StatusBadge tone={toneForState(strategy.state)}>{strategy.state}</StatusBadge><MoreButton /></div>
            <button className="strategy-card-main" onClick={() => onSelect(strategy)}>
              <h3>{strategy.name}</h3>
              <p>{strategy.model} · {strategy.venue} · {strategy.symbols}</p>
              <dl><div><dt>P&L</dt><dd className={strategy.pnl.startsWith("-") ? "text-danger" : "text-success"}>{strategy.pnl}</dd></div><div><dt>Drawdown</dt><dd>{strategy.drawdown}</dd></div><div><dt>Updated</dt><dd>{strategy.updated}</dd></div></dl>
            </button>
          </article>
        ))}
      </div>
    </div>
  );
}

function StrategyAnalytics({ onSelect }) {
  return (
    <div className="content-grid content-grid-analytics">
      <Panel title="Strategy performance" className="analytics-chart" action={<FilterButton>30 days</FilterButton>}><PortfolioChart compact /></Panel>
      <Panel title="Comparison">
        <Table rows={strategies.slice(0, 4)} rowKey="id" onRowClick={onSelect} columns={[
          { key: "name", label: "Strategy" }, { key: "pnlPct", label: "Return", render: (row) => <span className="text-success">{row.pnlPct}</span> }, { key: "drawdown", label: "Drawdown" }, { key: "model", label: "Model" },
        ]} />
      </Panel>
      <Panel title="Long / short attribution">
        <div className="chart-medium" role="img" aria-label="Long and short attribution by strategy">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={[{ name: "Mean rev", long: 62, short: 38 }, { name: "Momentum", long: 74, short: 26 }, { name: "RL Alpha", long: 58, short: 42 }, { name: "Market making", long: 51, short: 49 }]}><CartesianGrid stroke="var(--chart-grid)" vertical={false} /><XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} /><YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} /><Tooltip contentStyle={compactTooltip} /><Bar dataKey="long" fill="var(--positive)" /><Bar dataKey="short" fill="var(--negative)" /></BarChart></ResponsiveContainer>
        </div>
      </Panel>
      <Panel title="Risk contribution">
        <div className="risk-list">{strategies.slice(0, 4).map((row, index) => <div key={row.id}><span>{row.name}</span><ProgressBar value={[31, 28, 24, 17][index]} /></div>)}</div>
      </Panel>
    </div>
  );
}

function StrategyRuntime({ onToast }) {
  const [running, setRunning] = useState(true);
  return (
    <div className="content-grid content-grid-runtime">
      <Panel title="Runtime control" action={<StatusBadge tone={running ? "success" : "danger"}>{running ? "Running" : "Stopped"}</StatusBadge>}>
        <div className="runtime-summary"><div><strong>Momentum Breakout v7</strong><span>Bybit · BTCUSDT, ETHUSDT, SOLUSDT</span></div><div className="runtime-actions"><button className="button button-secondary" onClick={() => { setRunning(!running); onToast(running ? "Strategy stopped" : "Strategy started"); }}>{running ? <Stop size={15} weight="fill" /> : <Play size={15} weight="fill" />}{running ? "Stop" : "Start"}</button><button className="button button-secondary" disabled={!running} onClick={() => onToast("Strategy restart requested and reconciled")}><ClockCounterClockwise size={15} />Restart</button></div></div>
        <div className="readiness-grid"><InspectorRow label="Environment" value="Testnet" /><InspectorRow label="Run state" value={running ? "Running" : "Stopped"} tone={running ? "success" : "danger"} /><InspectorRow label="Checkpoint" value="12:27:42" /><InspectorRow label="Observed gap" value="0.8s" /><InspectorRow label="Latest signal" value="BTCUSDT Buy" /><InspectorRow label="Execution" value="Reconciled" tone="success" /></div>
      </Panel>
      <Panel title="Open positions"><Table rows={positions} rowKey="symbol" columns={[{ key: "symbol", label: "Symbol" }, { key: "side", label: "Side", render: (row) => <span className={row.side === "Long" ? "text-success" : "text-danger"}>{row.side}</span> }, { key: "size", label: "Size" }, { key: "entry", label: "Entry" }, { key: "mark", label: "Mark" }, { key: "pnl", label: "P&L", render: (row) => <span className="text-success">{row.pnl}</span> }, { key: "roe", label: "ROE" }]} /></Panel>
      <Panel title="Latest executions"><Table rows={executions} rowKey="time" columns={[{ key: "time", label: "Time" }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side", render: (row) => <span className={row.side === "Buy" ? "text-success" : "text-danger"}>{row.side}</span> }, { key: "price", label: "Price" }, { key: "qty", label: "Qty" }, { key: "fee", label: "Fee" }]} /></Panel>
      <Panel title="Runtime health"><div className="check-list"><span><CheckCircle weight="fill" />Data feed healthy</span><span><CheckCircle weight="fill" />Model loaded</span><span><CheckCircle weight="fill" />Risk limits active</span><span><CheckCircle weight="fill" />Execution reconciled</span><span className="text-warning"><Warning weight="fill" />1 warning requires review</span></div></Panel>
    </div>
  );
}

function StrategyModels({ onNavigate }) {
  return (
    <div className="content-grid content-grid-models">
      <Panel title="Active model" className="model-hero">
        <div className="model-heading"><Brain size={32} weight="duotone" /><div><StatusBadge tone="success">Production</StatusBadge><h3>RL Alpha v14</h3><p>PPO policy · Crypto Uni v3 · activated 09 Jul 2026</p></div></div>
        <div className="metric-grid metric-grid-three"><Metric label="Evaluation score" value="0.86" delta="+0.04" tone="success" /><Metric label="Sharpe (30D)" value="1.61" delta="+0.18" tone="success" /><Metric label="Max drawdown" value="0.92%" /></div>
      </Panel>
      <Panel title="Training pipeline"><div className="pipeline"><span className="is-done"><Database size={18} />Dataset<small>Crypto Uni v3</small></span><ArrowRight size={16} /><span className="is-done"><Code size={18} />Features<small>84 certified</small></span><ArrowRight size={16} /><span className="is-active"><Brain size={18} />Train<small>PPO · 12M steps</small></span><ArrowRight size={16} /><span><TestTube size={18} />Evaluate<small>8 gates</small></span><ArrowRight size={16} /><span><RocketLaunch size={18} />Activate<small>Manual</small></span></div></Panel>
      <Panel title="Model registry"><Table rows={models} rowKey="name" columns={[{ key: "name", label: "Model", render: (row) => <span className="table-primary">{row.name}</span> }, { key: "family", label: "Family" }, { key: "stage", label: "Stage", render: (row) => <StatusBadge tone={toneForState(row.stage)}>{row.stage}</StatusBadge> }, { key: "score", label: "Score" }, { key: "dataset", label: "Dataset" }, { key: "updated", label: "Updated" }]} /></Panel>
      <Panel title="Promotion readiness"><div className="check-list"><span><CheckCircle weight="fill" />Methodology parity</span><span><CheckCircle weight="fill" />Backtest evaluation</span><span><CheckCircle weight="fill" />Reward drift check</span><span className="text-warning"><Hourglass weight="fill" />Paper soak · 18h remaining</span></div><button className="button button-secondary" onClick={() => onNavigate("models")}>Open full model registry</button></Panel>
    </div>
  );
}

export function StrategiesPage({ onSelectStrategy, onNavigate, onToast }) {
  const [tab, setTab] = useState("library");
  const tabs = [{ id: "library", label: "Library" }, { id: "analytics", label: "Analytics" }, { id: "runtime", label: "Runtime control" }, { id: "models", label: "RL / ML" }];
  return (
    <div className="page strategies-page">
      <PageHeader title="Strategies" description="Design, compare, operate and promote trading strategies" actions={<button className="button button-primary" disabled title="Strategy creation uses the production contract"><Plus size={15} />New strategy</button>} />
      <Tabs label="Strategy workspace" items={tabs} active={tab} onChange={setTab} />
      {tab === "library" && <StrategyLibrary onSelect={onSelectStrategy} />}
      {tab === "analytics" && <StrategyAnalytics onSelect={onSelectStrategy} />}
      {tab === "runtime" && <StrategyRuntime onToast={onToast} />}
      {tab === "models" && <StrategyModels onNavigate={onNavigate} />}
    </div>
  );
}

function BacktestConfigure({ onRun, onToast }) {
  const [symbols, setSymbols] = useState(["BTCUSDT", "ETHUSDT", "SOLUSDT"]);
  const [instrumentQuery, setInstrumentQuery] = useState("");
  const [config, setConfig] = useState({ direction: "long-short", exchange: "binance", market: "futures", timeframe: "1h" });
  const [preflight, setPreflight] = useState("ready");
  const toggleSymbol = (symbol) => setSymbols((current) => current.includes(symbol) ? current.filter((item) => item !== symbol) : [...current, symbol]);
  return (
    <div className="backtest-config-grid">
      <Panel title="Configuration" className="backtest-form-panel">
        <div className="form-grid">
          <Field label="Name"><input defaultValue="RL Alpha v14 · 1h" /></Field>
          <Field label="Direction"><Select value={config.direction} onChange={(value) => setConfig((current) => ({ ...current, direction: value }))}><option value="long-short">Long / Short</option><option value="long-only">Long only</option><option value="short-only">Short only</option></Select></Field>
          <Field label="Exchange"><Select value={config.exchange} onChange={(value) => setConfig((current) => ({ ...current, exchange: value }))}><option value="binance">Binance</option><option value="bybit">Bybit</option></Select></Field>
          <Field label="Market"><Select value={config.market} onChange={(value) => setConfig((current) => ({ ...current, market: value }))}><option value="futures">Futures</option><option value="spot">Spot</option></Select></Field>
          <Field label="Timeframe"><Select value={config.timeframe} onChange={(value) => setConfig((current) => ({ ...current, timeframe: value }))}><option value="1h">1h</option><option value="15m">15m</option><option value="4h">4h</option></Select></Field>
          <Field label="Initial capital"><div className="input-suffix"><input type="number" defaultValue="100000" /><span>USDT</span></div></Field>
          <Field label="Start"><input type="date" defaultValue="2023-01-01" /></Field>
          <Field label="End"><input type="date" defaultValue="2026-07-10" /></Field>
          <Field label="Taker fee"><div className="input-suffix"><input type="number" defaultValue="0.075" step="0.001" /><span>%</span></div></Field>
          <Field label="Slippage"><div className="input-suffix"><input type="number" defaultValue="0.01" step="0.01" /><span>%</span></div></Field>
        </div>
        <div className="advanced-row"><button className="text-button" disabled title="Advanced options are deferred in this prototype"><SlidersHorizontal size={15} />Advanced options</button><span>Risk mode: Dynamic · Entry sizing: Equal weight</span></div>
      </Panel>

      <Panel title="Instruments" action={<span className="panel-meta">{symbols.length} selected</span>}>
        <SearchField value={instrumentQuery} onChange={setInstrumentQuery} placeholder="Search instruments…" />
        <div className="instrument-list">{["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"].filter((symbol) => symbol.toLowerCase().includes(instrumentQuery.toLowerCase())).map((symbol) => <label key={symbol}><input type="checkbox" checked={symbols.includes(symbol)} onChange={() => toggleSymbol(symbol)} /><span className="instrument-mark">{symbol.slice(0, 1)}</span><span><strong>{symbol}</strong><small>Futures · USDT perpetual</small></span></label>)}</div>
      </Panel>

      <Panel title="Indicators & parameter space" action={<button className="button button-secondary button-compact" disabled title="Indicator editing is deferred in this prototype"><Plus size={14} />Add indicator</button>}>
        <Table rows={[
          { id: "dema", name: "DEMA", from: 5, to: 200, step: 5, source: "close", combinations: 40 },
          { id: "rsi", name: "RSI", from: 7, to: 35, step: 2, source: "close", combinations: 15 },
          { id: "atr", name: "ATR", from: 5, to: 50, step: 5, source: "hlc3", combinations: 10 },
        ]} columns={[{ key: "name", label: "Indicator", render: (row) => <span className="table-primary">{row.name}</span> }, { key: "from", label: "From" }, { key: "to", label: "To" }, { key: "step", label: "Step" }, { key: "source", label: "Source" }, { key: "combinations", label: "Variants" }]} />
        <div className="parameter-summary"><span>Parameter combinations</span><strong>6,000</strong><span>Estimated duration</span><strong>18m</strong></div>
      </Panel>

      <aside className="run-dock">
        <div><StatusBadge tone={preflight === "ready" ? "success" : "warning"}>{preflight === "ready" ? "Ready" : "Checking"}</StatusBadge><span>3 symbols · 6,000 variants · Futures</span></div>
        <button className="button button-secondary" onClick={() => { setPreflight("checking"); window.setTimeout(() => { setPreflight("ready"); onToast("Preflight passed: data, capital and connections ready"); }, 600); }}>Run preflight</button>
        <button className="button button-primary" onClick={onRun}><Play size={15} weight="fill" />Run optimization</button>
      </aside>
    </div>
  );
}

function BacktestQueue({ onOpenResults }) {
  const [query, setQuery] = useState("");
  const rows = experiments.filter((row) => row.name.toLowerCase().includes(query.toLowerCase()));
  return (
    <div className="workspace-stack">
      <Toolbar className="workspace-toolbar"><SearchField value={query} onChange={setQuery} placeholder="Search queue…" /><FilterButton>State</FilterButton><button className="button button-secondary" disabled title="Queue mutation is available only through the production contract"><Pause size={15} />Pause queue</button></Toolbar>
      <div className="metric-grid metric-grid-four"><Metric label="Running" value="2" delta="6 workers" tone="info" /><Metric label="Queued" value="1" meta="Starts in ~8m" /><Metric label="Completed today" value="38" delta="96.4% passed" tone="success" /><Metric label="Compute usage" value="72%" meta="3.8 GB memory" /></div>
      <Panel title="Backtest queue" action={<MoreButton />}>
        <Table rows={rows} rowKey="id" onRowClick={(row) => row.state === "Completed" && onOpenResults(row)} columns={[
          { key: "id", label: "Job" }, { key: "name", label: "Strategy", render: (row) => <span className="table-primary">{row.name}</span> }, { key: "state", label: "State", render: (row) => <StatusBadge tone={toneForState(row.state)}>{row.state}</StatusBadge> }, { key: "progress", label: "Progress", render: (row) => <ProgressBar value={row.progress} /> }, { key: "owner", label: "Owner" }, { key: "updated", label: "Updated" }, { key: "actions", label: "", render: () => <MoreButton /> },
        ]} />
      </Panel>
      <Panel title="Worker activity"><div className="worker-grid">{["compute-01", "compute-02", "compute-03", "compute-04"].map((worker, index) => <div key={worker}><div><StatusBadge tone={index === 3 ? "neutral" : "success"}>{index === 3 ? "Idle" : "Busy"}</StatusBadge><strong>{worker}</strong></div><ProgressBar value={[82, 61, 74, 0][index]} /><span>{index === 3 ? "Waiting for job" : `${[318, 241, 276][index]} variants/min`}</span></div>)}</div></Panel>
    </div>
  );
}

const resultSeries = portfolioSeries.map((point, index) => ({ ...point, candidate: Math.round(point.equity * (1 + index * 0.0015)), baseline: point.equity }));

function BacktestResults() {
  const [metric, setMetric] = useState("return");
  return (
    <div className="workspace-stack">
      <div className="result-head"><div><StatusBadge tone="success">Completed</StatusBadge><h2>BT-8413 · Funding Arb v3</h2><p>01 Jan 2023 — 10 Jul 2026 · Binance Futures · 6,000 variants</p></div><div><button className="button button-secondary" disabled title="Export is connected in production"><DownloadSimple size={15} />Export</button><button className="button button-primary" disabled title="Promotion requires production readiness checks"><RocketLaunch size={15} />Promote variant</button></div></div>
      <div className="metric-grid metric-grid-five"><Metric label="Total return" value="+84.6%" delta="+11.2% vs baseline" tone="success" /><Metric label="Sharpe" value="1.74" delta="+0.21" tone="success" /><Metric label="Max drawdown" value="-8.2%" delta="-1.4%" tone="success" /><Metric label="Win rate" value="62.4%" /><Metric label="Trades" value="1,284" meta="0.9/day" /></div>
      <Panel title="Candidate vs baseline" className="result-chart" action={<Tabs compact label="Result metric" active={metric} onChange={setMetric} items={[{ id: "return", label: "Return" }, { id: "drawdown", label: "Drawdown" }, { id: "risk", label: "Risk" }]} />}>
        <div className="chart-large" role="img" aria-label="Candidate equity curve outperformed baseline during the backtest period."><ResponsiveContainer width="100%" height="100%"><LineChart data={resultSeries}><CartesianGrid stroke="var(--chart-grid)" vertical={false} /><XAxis dataKey="time" tick={{ fill: "var(--text-muted)", fontSize: 11 }} interval={3} /><YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} /><Tooltip contentStyle={compactTooltip} /><Legend /><Line dataKey="candidate" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} /><Line dataKey="baseline" stroke="var(--chart-secondary)" strokeDasharray="4 4" dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div>
      </Panel>
      <div className="content-grid content-grid-two">
        <Panel title="Top variants"><Table rows={[{ id: 1, variant: "#1432", return: "+84.6%", sharpe: "1.74", drawdown: "-8.2%", score: "0.91" }, { id: 2, variant: "#2911", return: "+81.2%", sharpe: "1.70", drawdown: "-7.9%", score: "0.89" }, { id: 3, variant: "#482", return: "+79.8%", sharpe: "1.68", drawdown: "-8.4%", score: "0.87" }]} columns={[{ key: "variant", label: "Variant" }, { key: "return", label: "Return", render: (row) => <span className="text-success">{row.return}</span> }, { key: "sharpe", label: "Sharpe" }, { key: "drawdown", label: "Drawdown" }, { key: "score", label: "Score" }]} /></Panel>
        <Panel title="Selected parameters"><div className="parameter-grid"><InspectorRow label="DEMA period" value="42" /><InspectorRow label="RSI period" value="14" /><InspectorRow label="ATR period" value="20" /><InspectorRow label="Funding threshold" value="0.014%" /><InspectorRow label="Leverage" value="2x" /><InspectorRow label="Risk mode" value="Dynamic" /></div></Panel>
      </div>
    </div>
  );
}

export function BacktestsPage({ onToast }) {
  const [tab, setTab] = useState("configure");
  const run = () => { setTab("queue"); onToast("Backtest BT-8422 added to the queue"); };
  return (
    <div className="page">
      <PageHeader title="Backtests" description="Configure, queue, compare and promote research runs" actions={<button className="button button-primary" onClick={() => setTab("configure")}><Plus size={15} />New backtest</button>} />
      <Tabs label="Backtest workspace" active={tab} onChange={setTab} items={[{ id: "configure", label: "Configure" }, { id: "queue", label: "Queue", count: 3 }, { id: "results", label: "Results" }]} />
      {tab === "configure" && <BacktestConfigure onRun={run} onToast={onToast} />}
      {tab === "queue" && <BacktestQueue onOpenResults={() => setTab("results")} />}
      {tab === "results" && <BacktestResults />}
    </div>
  );
}

export function LivePage({ onSelectStrategy, onToast }) {
  const [paused, setPaused] = useState(false);
  const [stream, setStream] = useState("all");
  return (
    <div className="page">
      <PageHeader
        title="Live operations"
        description="Read-only operational view of strategies, positions, executions and market data"
        actions={(
          <>
            <StatusBadge tone={paused ? "warning" : "success"}>{paused ? "Paused locally" : "Live · 12s"}</StatusBadge>
            <button className="button button-secondary" onClick={() => { setPaused(!paused); onToast(paused ? "Live updates resumed" : "Live updates paused locally"); }}>
              {paused ? <Play size={15} weight="fill" /> : <Pause size={15} weight="fill" />}{paused ? "Resume" : "Pause"}
            </button>
          </>
        )}
      />
      <div className="metric-grid metric-grid-five">
        <Metric label="Open positions" value="4" delta="3 long · 1 short" />
        <Metric label="Unrealized P&L" value="+2,163" delta="+0.74%" tone="success" />
        <Metric label="Orders (1h)" value="38" meta="0 rejected" />
        <Metric label="Median latency" value="146 ms" delta="-18 ms" tone="success" />
        <Metric label="Active alerts" value="2" delta="1 warning" tone="warning" />
      </div>
      <div className="content-grid live-grid">
        <Panel title="Positions" className="live-positions" action={<FilterButton>Portfolio</FilterButton>}>
          <Table rows={positions} rowKey={(row) => `${row.venue}-${row.symbol}-${row.side}`} columns={[{ key: "symbol", label: "Symbol" }, { key: "venue", label: "Venue" }, { key: "side", label: "Side", render: (row) => <span className={row.side === "Long" ? "text-success" : "text-danger"}>{row.side}</span> }, { key: "size", label: "Size" }, { key: "entry", label: "Entry" }, { key: "mark", label: "Mark" }, { key: "pnl", label: "P&L", render: (row) => <span className="text-success">{row.pnl}</span> }, { key: "roe", label: "ROE" }]} />
        </Panel>
        <Panel title="Strategy health" className="live-health" action={<MoreButton />}>
          <div className="health-stack">
            {strategies.slice(0, 4).map((strategy, index) => (
              <button key={strategy.id} onClick={() => onSelectStrategy(strategy)}>
                <span><i className={`health-light ${index === 3 ? "is-warning" : ""}`} /><strong>{strategy.name}</strong></span>
                <small>{index === 3 ? "Market stream delayed 4.8s" : `${strategy.venue} · fresh ${strategy.updated}`}</small>
                <StatusBadge tone={index === 3 ? "warning" : "success"}>{index === 3 ? "Degraded" : "Healthy"}</StatusBadge>
              </button>
            ))}
          </div>
        </Panel>
        <Panel title="Execution stream" className="live-executions" action={<Tabs compact label="Execution stream filter" active={stream} onChange={setStream} items={[{ id: "all", label: "All" }, { id: "fills", label: "Fills" }, { id: "events", label: "Events" }]} />}>
          <Table rows={executions} rowKey="time" columns={[{ key: "time", label: "Time" }, { key: "strategy", label: "Strategy" }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side", render: (row) => <span className={row.side === "Buy" ? "text-success" : "text-danger"}>{row.side}</span> }, { key: "price", label: "Price" }, { key: "qty", label: "Quantity" }, { key: "fee", label: "Fee" }]} />
        </Panel>
        <Panel title="Data freshness" className="live-data" action={<StatusBadge tone="warning">1 delayed</StatusBadge>}>
          <div className="data-freshness-list">{dataHealth.map((source) => <div key={source.source}><span><strong>{source.source}</strong><small>Updated {source.updated}</small></span><span><b>{source.latency}</b><StatusBadge tone={toneForState(source.status)}>{source.status}</StatusBadge></span></div>)}</div>
        </Panel>
      </div>
    </div>
  );
}

export function ModelsPage() {
  const [selected, setSelected] = useState(models[1]);
  return (
    <div className="page">
      <PageHeader title="Model registry" description="Train, evaluate and promote RL / ML models with explicit readiness gates" actions={<button className="button button-primary" disabled title="Training creation requires the production contract"><Plus size={15} />New training run</button>} />
      <div className="models-layout">
        <Panel title="Models" action={<FilterButton>Stage</FilterButton>}>
          <Table rows={models} rowKey="name" selectedKey={selected.name} onRowClick={setSelected} columns={[{ key: "name", label: "Model", render: (row) => <span className="table-primary">{row.name}</span> }, { key: "family", label: "Family" }, { key: "stage", label: "Stage", render: (row) => <StatusBadge tone={toneForState(row.stage)}>{row.stage}</StatusBadge> }, { key: "score", label: "Score" }, { key: "dataset", label: "Dataset" }, { key: "updated", label: "Updated" }]} />
        </Panel>
        <Panel title="Evaluation" className="model-evaluation">
          <div className="model-heading"><Brain size={32} weight="duotone" /><div><StatusBadge tone={toneForState(selected.stage)}>{selected.stage}</StatusBadge><h3>{selected.name}</h3><p>{selected.family} · {selected.dataset}</p></div></div>
          <div className="metric-grid metric-grid-three"><Metric label="Composite score" value={selected.score} delta="+0.03" tone="success" /><Metric label="Sharpe" value="1.61" /><Metric label="Drawdown" value="-0.92%" /></div>
          <PortfolioChart compact />
        </Panel>
        <Panel title="Promotion contract" className="model-contract">
          <div className="check-list"><span><CheckCircle weight="fill" />Dataset lineage certified</span><span><CheckCircle weight="fill" />Backtest parity passed</span><span><CheckCircle weight="fill" />Reward drift below limit</span><span className="text-warning"><Hourglass weight="fill" />Paper soak · 18h remaining</span></div>
          <button className="button button-primary" disabled><RocketLaunch size={15} />Promote after soak</button>
        </Panel>
      </div>
    </div>
  );
}

export function ConnectionsPage({ onToast }) {
  const [selected, setSelected] = useState(connections[0]);
  return (
    <div className="page">
      <PageHeader title="Connections" description="Exchange, market-data and research providers without exposing stored credentials" actions={<button className="button button-primary" disabled title="Credentials are accepted only by the production secret form"><Plus size={15} />Add connection</button>} />
      <div className="connections-layout">
        <Panel title="Providers" className="connections-table" action={<button className="button button-secondary" disabled title="The production audit endpoint owns this action"><ShieldCheck size={15} />Security log</button>}>
          <Table rows={connections} rowKey="venue" selectedKey={selected.venue} onRowClick={setSelected} columns={[{ key: "venue", label: "Provider", render: (row) => <span className="table-primary">{row.venue}</span> }, { key: "environment", label: "Environment" }, { key: "status", label: "Status", render: (row) => <StatusBadge tone={toneForState(row.status)}>{row.status}</StatusBadge> }, { key: "latency", label: "Latency" }, { key: "scopes", label: "Scopes" }, { key: "updated", label: "Checked" }]} />
        </Panel>
        <Panel title="Connection details" className="connection-details">
          <div className="connection-brand"><Database size={28} weight="duotone" /><div><h3>{selected.venue}</h3><p>{selected.environment} environment</p></div></div>
          <div className="parameter-grid"><InspectorRow label="Status" value={selected.status} tone={toneForState(selected.status)} /><InspectorRow label="Latency" value={selected.latency} /><InspectorRow label="Scopes" value={selected.scopes} /><InspectorRow label="Credentials" value="Stored securely" /><InspectorRow label="Last rotation" value="28 Jun 2026" /><InspectorRow label="Used by" value="4 strategies" /></div>
          <div className="panel-button-row"><button className="button button-secondary" onClick={() => onToast(`${selected.venue}: connection test passed`)}><Lightning size={15} />Test connection</button><button className="button button-secondary" disabled title="Configuration is connected in production"><Gear size={15} />Configure</button></div>
        </Panel>
      </div>
    </div>
  );
}

export function SettingsPage({ activeTheme, onThemeChange }) {
  const [density, setDensity] = useState("comfortable");
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [defaults, setDefaults] = useState({ exchange: "binance", market: "futures", timezone: "moscow" });
  return (
    <div className="page">
      <PageHeader title="Settings" description="Workspace appearance, behavior, notifications and access policy" />
      <div className="settings-grid">
        <Panel title="Appearance" className="settings-wide">
          <div className="setting-block"><div><strong>Interface theme</strong><p>Six semantic palettes from very dark to warm light.</p></div><div className="theme-settings-grid">{themes.map((theme) => <button key={theme.id} className={activeTheme === theme.id ? "is-active" : ""} onClick={() => onThemeChange(theme.id)}><span style={{ background: theme.swatch }} /><strong>{theme.name}</strong><small>{theme.tone}</small></button>)}</div></div>
          <div className="setting-row"><div><strong>Information density</strong><p>Controls table row height and workspace spacing.</p></div><Tabs compact label="Information density" active={density} onChange={setDensity} items={[{ id: "compact", label: "Compact" }, { id: "comfortable", label: "Comfortable" }, { id: "relaxed", label: "Relaxed" }]} /></div>
        </Panel>
        <Panel title="Notifications">
          <div className="setting-row"><div><strong>Operational alerts</strong><p>Drawdown, execution and data health.</p></div><button className={`switch ${notificationsEnabled ? "is-on" : ""}`} aria-pressed={notificationsEnabled} onClick={() => setNotificationsEnabled(!notificationsEnabled)}><span /></button></div>
          <div className="setting-row"><div><strong>Experiment updates</strong><p>Queue, completion and evaluation events.</p></div><button className="switch is-on" aria-pressed="true" disabled title="Notification persistence is connected in production"><span /></button></div>
          <div className="setting-row"><div><strong>Quiet hours</strong><p>Only critical alerts from 23:00 to 07:00.</p></div><button className="button button-secondary" disabled title="Quiet hours are deferred in this prototype">Configure</button></div>
        </Panel>
        <Panel title="Access & security">
          <div className="check-list"><span><ShieldCheck weight="fill" />Keycloak single sign-on active</span><span><CheckCircle weight="fill" />Multi-factor policy enforced</span><span><CheckCircle weight="fill" />Session last verified 12:18</span></div>
          <button className="button button-secondary" disabled title="Sessions are provided by the production identity API">View active sessions</button>
        </Panel>
        <Panel title="Workspace defaults">
          <div className="form-grid form-grid-single"><Field label="Default exchange"><Select value={defaults.exchange} onChange={(value) => setDefaults((current) => ({ ...current, exchange: value }))}><option value="binance">Binance Futures</option><option value="bybit">Bybit Futures</option></Select></Field><Field label="Default market"><Select value={defaults.market} onChange={(value) => setDefaults((current) => ({ ...current, market: value }))}><option value="futures">Futures</option><option value="spot">Spot</option></Select></Field><Field label="Timezone"><Select value={defaults.timezone} onChange={(value) => setDefaults((current) => ({ ...current, timezone: value }))}><option value="moscow">Europe / Moscow</option><option value="utc">UTC</option></Select></Field></div>
        </Panel>
      </div>
    </div>
  );
}

export function ProgramMapPage({ onNavigate }) {
  return (
    <div className="page">
      <PageHeader title="Program map" description="Every workspace, shared surface, UI state and domain entity in Roehub v2" actions={<StatusBadge tone="info">System blueprint</StatusBadge>} />
      <div className="program-map-grid">
        {programMapGroups.map((group, index) => <Panel key={group.title} title={group.title} className={`map-group map-group-${index + 1}`}><ol>{group.items.map((item) => <li key={item}><span>{String(index + 1).padStart(2, "0")}.{String(group.items.indexOf(item) + 1).padStart(2, "0")}</span><strong>{item}</strong></li>)}</ol></Panel>)}
        <Panel title="Domain entities" className="map-entities"><div className="entity-map">{entities.map((entity) => <button key={entity.name} onClick={() => onNavigate(entity.name === "Backtest" ? "backtests" : entity.name.includes("model") ? "models" : entity.name === "Exchange connection" ? "connections" : "strategies")}><strong>{entity.name}</strong><span>{entity.links}</span><ArrowRight size={14} /></button>)}</div></Panel>
        <Panel title="Interaction flow" className="map-flow"><div className="flow-track"><button onClick={() => onNavigate("backtests")}><TestTube size={18} /><strong>Backtest</strong><small>Configure and compare</small></button><ArrowRight size={18} /><button onClick={() => onNavigate("strategies")}><GitBranch size={18} /><strong>Strategy</strong><small>Promote and operate</small></button><ArrowRight size={18} /><button onClick={() => onNavigate("live")}><Pulse size={18} /><strong>Live</strong><small>Observe runtime</small></button><ArrowRight size={18} /><button onClick={() => onNavigate("settings")}><Gear size={18} /><strong>Settings</strong><small>Connections and policy</small></button></div></Panel>
      </div>
    </div>
  );
}

export function StateGalleryPage() {
  const [errorVisible, setErrorVisible] = useState(true);
  return (
    <div className="page">
      <PageHeader title="Interface states" description="Reusable operational states for every Roehub workspace" />
      <div className="state-gallery">
        <Panel title="Loading"><LoadingState rows={4} /></Panel>
        <Panel title="Empty"><EmptyState icon={<Database size={28} weight="duotone" />} title="No research runs yet" description="Create a backtest to begin comparing strategy variants." action={<button className="button button-primary" disabled title="This gallery demonstrates the state only"><Plus size={15} />New backtest</button>} /></Panel>
        <Panel title="Error">{errorVisible ? <ErrorState description="The compute service returned an invalid response. Configuration was preserved." onRetry={() => setErrorVisible(false)} /> : <div className="state-success"><CheckCircle size={28} weight="fill" /><h3>Workspace restored</h3><p>The latest valid snapshot is now visible.</p></div>}</Panel>
        <Panel title="Stale data"><div className="state-callout state-stale"><CloudWarning size={28} weight="duotone" /><div><StatusBadge tone="warning">Stale · 45m</StatusBadge><h3>Glassnode data is delayed</h3><p>Live strategies continue using exchange data. On-chain features are held at the last certified value.</p><button className="button button-secondary" disabled title="This gallery demonstrates the state only">Inspect source</button></div></div></Panel>
        <Panel title="Restricted"><div className="state-callout"><LockKey size={28} weight="duotone" /><div><StatusBadge tone="danger">Restricted</StatusBadge><h3>Production activation unavailable</h3><p>The backend capability does not allow activation.</p><button className="button button-secondary" disabled title="Access requests use the organization identity workflow">Request access</button></div></div></Panel>
        <Panel title="Success & disabled"><div className="state-callout state-good"><CheckCircle size={28} weight="fill" /><div><StatusBadge tone="success">Verified</StatusBadge><h3>Connection test passed</h3><p>All required scopes and market streams are available.</p><button className="button button-primary" disabled>Already connected</button></div></div></Panel>
      </div>
    </div>
  );
}

export function LoginPage({ onLogin, activeTheme, onThemeChange }) {
  return (
    <main className="login-screen" data-theme={activeTheme}>
      <section className="login-brand">
        <div className="login-logo"><ChartLineUp size={24} weight="bold" /><strong>Roehub</strong></div>
        <div className="login-statement"><p className="eyebrow">Quantitative operations workspace</p><h1>Research, operate and understand every strategy from one calm system.</h1><p>Native workspaces for backtests, models, live execution and data health — designed to stay legible under pressure.</p></div>
        <div className="login-status"><StatusBadge tone="success">All systems operational</StatusBadge><span>EU West · 14 ms</span></div>
      </section>
      <section className="login-form-wrap">
        <div className="login-theme-row"><span>Interface</span><Select value={activeTheme} onChange={onThemeChange} aria-label="Login theme">{themes.map((theme) => <option key={theme.id} value={theme.id}>{theme.name}</option>)}</Select></div>
        <section className="login-form">
          <div><p className="eyebrow">Secure workspace</p><h2>Sign in to Roehub</h2><p>Use your Keycloak account to continue.</p></div>
          <div className="auth-destination"><ShieldCheck size={20} weight="duotone" /><div><strong>Keycloak single sign-on</strong><span>After sign-in: Portfolio overview</span></div><StatusBadge tone="success">Ready</StatusBadge></div>
          <button className="button button-primary button-large" type="button" onClick={onLogin}><LockKey size={16} />Continue with Keycloak</button>
          <div className="login-options"><button type="button" className="text-button" disabled title="Use the organization identity workflow">Request access</button><button type="button" className="text-button" disabled title="Help is linked by the production support surface">Need help?</button></div>
          <p className="login-note"><ShieldCheck size={15} />You will be redirected to Keycloak. Roehub does not receive or store your password.</p>
        </section>
      </section>
    </main>
  );
}
