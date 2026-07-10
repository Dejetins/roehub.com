export const themes = [
  { id: "abyss", name: "Abyss", tone: "Very dark", swatch: "#071019" },
  { id: "graphite", name: "Graphite Dusk", tone: "Dark", swatch: "#101923" },
  { id: "slate", name: "Slate Balance", tone: "Dim", swatch: "#263240" },
  { id: "frost", name: "Frost", tone: "Soft light", swatch: "#e9eef4" },
  { id: "paper", name: "Paper", tone: "Bright", swatch: "#f8fafc" },
  { id: "sand", name: "Warm Sand", tone: "Warm light", swatch: "#f2eee6" },
];

export const portfolioSeries = [
  { time: "00:00", equity: 1002, benchmark: 1000, drawdown: -0.2 },
  { time: "01:00", equity: 1018, benchmark: 1006, drawdown: -0.8 },
  { time: "02:00", equity: 1032, benchmark: 1012, drawdown: -1.2 },
  { time: "03:00", equity: 1044, benchmark: 1010, drawdown: -0.7 },
  { time: "04:00", equity: 1058, benchmark: 1021, drawdown: -1.6 },
  { time: "05:00", equity: 1081, benchmark: 1026, drawdown: -2.4 },
  { time: "06:00", equity: 1070, benchmark: 1034, drawdown: -1.9 },
  { time: "07:00", equity: 1094, benchmark: 1040, drawdown: -2.8 },
  { time: "08:00", equity: 1121, benchmark: 1048, drawdown: -3.7 },
  { time: "09:00", equity: 1112, benchmark: 1060, drawdown: -2.6 },
  { time: "10:00", equity: 1140, benchmark: 1074, drawdown: -2.1 },
  { time: "11:00", equity: 1168, benchmark: 1081, drawdown: -3.1 },
  { time: "12:00", equity: 1159, benchmark: 1090, drawdown: -4.2 },
  { time: "13:00", equity: 1184, benchmark: 1102, drawdown: -3.6 },
  { time: "14:00", equity: 1210, benchmark: 1116, drawdown: -2.4 },
  { time: "15:00", equity: 1228, benchmark: 1128, drawdown: -1.9 },
  { time: "16:00", equity: 1216, benchmark: 1142, drawdown: -2.8 },
  { time: "17:00", equity: 1238, benchmark: 1155, drawdown: -1.7 },
  { time: "18:00", equity: 1252, benchmark: 1164, drawdown: -1.2 },
  { time: "19:00", equity: 1264, benchmark: 1176, drawdown: -0.9 },
  { time: "20:00", equity: 1258, benchmark: 1182, drawdown: -1.4 },
  { time: "21:00", equity: 1276, benchmark: 1195, drawdown: -0.8 },
  { time: "22:00", equity: 1291, benchmark: 1204, drawdown: -0.6 },
  { time: "23:00", equity: 1284, benchmark: 1213, drawdown: -1.1 },
];

export const strategies = [
  { id: "mean-reversion", name: "Mean Reversion v3", venue: "Binance", symbols: "BTC, ETH", state: "Running", pnl: "+8,314.25", pnlPct: "+0.67%", drawdown: "1.22%", updated: "12s ago", model: "Classic" },
  { id: "momentum", name: "Momentum Breakout v7", venue: "Bybit", symbols: "BTC, ETH, SOL", state: "Running", pnl: "+10,912.44", pnlPct: "+0.88%", drawdown: "1.85%", updated: "18s ago", model: "Classic" },
  { id: "rl-alpha", name: "RL Alpha v14", venue: "Binance", symbols: "BTC, ETH, SOL", state: "Running", pnl: "+4,521.72", pnlPct: "+0.36%", drawdown: "0.92%", updated: "9s ago", model: "RL" },
  { id: "market-making", name: "Market Making v2", venue: "Binance", symbols: "SOL, DOGE", state: "Running", pnl: "+1,023.42", pnlPct: "+0.21%", drawdown: "0.55%", updated: "22s ago", model: "Classic" },
  { id: "funding-arb", name: "Funding Arbitrage v3", venue: "Bybit", symbols: "BTC, ETH", state: "Stopped", pnl: "-245.11", pnlPct: "-0.04%", drawdown: "0.18%", updated: "1m ago", model: "Classic" },
];

export const experiments = [
  { id: "BT-8421", name: "RL Alpha v14", state: "Running", progress: 72, score: "0.86", owner: "quant_trader", updated: "12:27" },
  { id: "BT-8419", name: "Momentum Breakout v7", state: "Running", progress: 45, score: "0.71", owner: "aria_quant", updated: "12:21" },
  { id: "BT-8418", name: "Mean Reversion v5", state: "Queued", progress: 0, score: "—", owner: "nik_research", updated: "12:19" },
  { id: "BT-8413", name: "Funding Arb v3", state: "Completed", progress: 100, score: "0.64", owner: "quant_trader", updated: "11:58" },
  { id: "BT-8409", name: "RL Alpha v13", state: "Completed", progress: 100, score: "0.83", owner: "aria_quant", updated: "11:32" },
];

export const signals = [
  { time: "12:26:12", event: "Entry signal", strategy: "RL Alpha v14", symbol: "ETHUSDT", side: "Buy", impact: "High", level: "success" },
  { time: "12:14:08", event: "Funding rate spike", strategy: "—", symbol: "BTCUSDT", side: "—", impact: "Medium", level: "warning" },
  { time: "11:47:31", event: "Exit signal", strategy: "Momentum Breakout v7", symbol: "SOLUSDT", side: "Sell", impact: "High", level: "success" },
  { time: "11:21:09", event: "Strategy restarted", strategy: "Mean Reversion v3", symbol: "BTCUSDT", side: "—", impact: "Low", level: "info" },
];

export const dataHealth = [
  { source: "Binance (Futures)", status: "Healthy", latency: "1.2s", updated: "12:27:56" },
  { source: "Bybit (Futures)", status: "Healthy", latency: "1.6s", updated: "12:27:55" },
  { source: "CoinGecko (Prices)", status: "Healthy", latency: "2.1s", updated: "12:27:55" },
  { source: "Glassnode (On-chain)", status: "Delayed", latency: "45m", updated: "11:42:10" },
];

export const positions = [
  { symbol: "BTCUSDT", venue: "Bybit", side: "Long", size: "0.842", entry: "109,452.1", mark: "110,235.4", pnl: "+659.22", roe: "+0.60%" },
  { symbol: "ETHUSDT", venue: "Bybit", side: "Long", size: "6.452", entry: "2,583.12", mark: "2,612.72", pnl: "+191.10", roe: "+0.86%" },
  { symbol: "SOLUSDT", venue: "Binance", side: "Long", size: "256.00", entry: "154.21", mark: "158.37", pnl: "+1,063.04", roe: "+2.69%" },
  { symbol: "BTCUSDT", venue: "Binance", side: "Short", size: "0.381", entry: "110,892.0", mark: "110,235.4", pnl: "+250.01", roe: "+0.59%" },
];

export const executions = [
  { time: "12:27:51", strategy: "Momentum Breakout v7", symbol: "BTCUSDT", side: "Buy", price: "110,182.2", qty: "0.210", fee: "2.31" },
  { time: "12:27:48", strategy: "RL Alpha v14", symbol: "ETHUSDT", side: "Buy", price: "2,612.35", qty: "1.500", fee: "0.78" },
  { time: "12:27:35", strategy: "Mean Reversion v3", symbol: "SOLUSDT", side: "Sell", price: "158.29", qty: "32.000", fee: "0.32" },
  { time: "12:27:21", strategy: "Momentum Breakout v7", symbol: "BTCUSDT", side: "Sell", price: "110,311.4", qty: "0.120", fee: "1.41" },
];

export const notifications = [
  { id: 1, title: "Drawdown approaching soft limit", detail: "Momentum Breakout v7 · 6.72% of 7%", time: "12:27", level: "warning", unread: true },
  { id: 2, title: "Backtest BT-8421 is running", detail: "72% complete · ETA 00:07:21", time: "12:26", level: "info", unread: true },
  { id: 3, title: "Bybit stream reconnected", detail: "Market data healthy again", time: "12:22", level: "success", unread: false },
  { id: 4, title: "Order rejected", detail: "Insufficient balance · Binance Futures", time: "12:18", level: "danger", unread: false },
];

export const models = [
  { name: "RL Alpha v14", family: "PPO", stage: "Production", score: "0.86", dataset: "Crypto Uni v3", updated: "12m ago" },
  { name: "RL Alpha v15-rc2", family: "PPO", stage: "Candidate", score: "0.89", dataset: "Crypto Uni v4", updated: "38m ago" },
  { name: "DQN Article Baseline", family: "DQN", stage: "Research", score: "0.42", dataset: "Article demo", updated: "2d ago" },
  { name: "Signal Ensemble v6", family: "Ensemble", stage: "Production", score: "0.81", dataset: "Signals v8", updated: "4h ago" },
];

export const connections = [
  { venue: "Binance Futures", environment: "Testnet", status: "Healthy", latency: "128 ms", scopes: "Read, Trade", updated: "12s ago" },
  { venue: "Bybit Futures", environment: "Testnet", status: "Healthy", latency: "164 ms", scopes: "Read, Trade", updated: "18s ago" },
  { venue: "Glassnode", environment: "Production", status: "Delayed", latency: "45 m", scopes: "Read", updated: "45m ago" },
];

export const programMapGroups = [
  {
    title: "Global shell",
    items: ["Top toolbar", "Global rail", "Context navigator", "Document tabs", "Inspector", "Activity drawer", "Status bar"],
  },
  {
    title: "Primary workspaces",
    items: ["Authentication", "Portfolio overview", "Strategy library", "Strategy analytics", "Runtime control", "RL / ML lab", "Backtest configure", "Backtest queue", "Backtest results", "Live monitoring", "Connections", "Settings"],
  },
  {
    title: "Cross-cutting surfaces",
    items: ["Command search", "Notification center", "User menu", "Theme picker", "Confirm dialog", "Filters", "Saved views", "Export", "Freshness states"],
  },
  {
    title: "UI states",
    items: ["Loading", "Empty", "Error", "Stale data", "Restricted", "Disabled", "Success", "Degraded"],
  },
];

export const entities = [
  { name: "Strategy", links: "Run, backtest, model, signals" },
  { name: "Strategy run", links: "Positions, orders, executions, events" },
  { name: "Backtest", links: "Variants, queue job, result" },
  { name: "Backtest variant", links: "Parameters, score, comparison" },
  { name: "Position", links: "Orders, fills, P&L" },
  { name: "Order", links: "Position, execution, venue" },
  { name: "Execution", links: "Order, fee, realized P&L" },
  { name: "Exchange connection", links: "Venue, credentials, readiness" },
  { name: "Alert", links: "Source, severity, acknowledgement" },
  { name: "Event", links: "Actor, entity, timestamp" },
  { name: "RL / ML model", links: "Dataset, training run, evaluation, activation" },
];
