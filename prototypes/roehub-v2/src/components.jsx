import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  CaretDown,
  CaretRight,
  CheckCircle,
  CircleNotch,
  DotsThree,
  Funnel,
  MagnifyingGlass,
  Warning,
  X,
} from "@phosphor-icons/react";
import { portfolioSeries, themes } from "./data.js";

export function IconButton({ label, children, className = "", ...props }) {
  return (
    <button className={`icon-button ${className}`} aria-label={label} title={label} {...props}>
      {children}
    </button>
  );
}

export function StatusBadge({ children, tone = "neutral", dot = true }) {
  return (
    <span className={`status-badge status-${tone}`}>
      {dot && <span className="status-dot" aria-hidden="true" />}
      {children}
    </span>
  );
}

export function Panel({ title, action, children, className = "", collapsible = false }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <header className="panel-header">
          <div className="panel-heading">
            {collapsible && <CaretDown size={14} aria-hidden="true" />}
            {title && <h2>{title}</h2>}
          </div>
          {action}
        </header>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, description, actions, children }) {
  return (
    <header className="page-header">
      <div className="page-title-group">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
      {children}
    </header>
  );
}

export function Metric({ label, value, delta, tone = "neutral", meta, data, icon }) {
  return (
    <article className="metric-card">
      <div className="metric-topline">
        <span>{label}</span>
        {icon}
      </div>
      <div className="metric-value-row">
        <strong>{value}</strong>
        {delta && <span className={`metric-delta text-${tone}`}>{delta}</span>}
      </div>
      {data && (
        <div className="metric-spark" aria-hidden="true">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <Line type="monotone" dataKey="value" stroke="var(--accent)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {meta && <small>{meta}</small>}
    </article>
  );
}

export function Tabs({ items, active, onChange, label, compact = false }) {
  return (
    <div className={`tabs ${compact ? "tabs-compact" : ""}`} role="tablist" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.id}
          role="tab"
          aria-selected={active === item.id}
          disabled={item.disabled}
          title={item.disabled ? "This range is not available in the prototype dataset" : undefined}
          className={active === item.id ? "is-active" : ""}
          onClick={() => !item.disabled && onChange(item.id)}
        >
          {item.label}
          {item.count !== undefined && <span className="tab-count">{item.count}</span>}
        </button>
      ))}
    </div>
  );
}

export function ProgressBar({ value, label }) {
  return (
    <div className="progress-wrap" aria-label={label || `${value}% complete`}>
      <div className="progress-track">
        <span className="progress-value" style={{ width: `${value}%` }} />
      </div>
      <span>{value}%</span>
    </div>
  );
}

export function Table({ columns, rows, rowKey = "id", onRowClick, selectedKey, className = "", empty = "No data" }) {
  return (
    <div className={`table-scroll ${className}`}>
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr><td colSpan={columns.length} className="table-empty">{empty}</td></tr>
          ) : rows.map((row, index) => {
            const rowIdentity = typeof rowKey === "function" ? rowKey(row, index) : row[rowKey] ?? "row";
            const key = `${rowIdentity}-${index}`;
            return (
              <tr
                key={key}
                className={`${onRowClick ? "is-clickable" : ""} ${selectedKey === rowIdentity ? "is-selected" : ""}`}
                tabIndex={onRowClick ? 0 : undefined}
                onClick={() => onRowClick?.(row)}
                onKeyDown={(event) => {
                  if (!onRowClick || !["Enter", " "].includes(event.key)) return;
                  event.preventDefault();
                  onRowClick(row);
                }}
              >
                {columns.map((column) => <td key={column.key}>{column.render ? column.render(row) : row[column.key]}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Toolbar({ children, className = "" }) {
  return <div className={`toolbar ${className}`}>{children}</div>;
}

const tooltipStyle = {
  background: "var(--surface-raised)",
  border: "1px solid var(--border-strong)",
  borderRadius: "6px",
  color: "var(--text)",
  fontSize: 12,
};

export function PortfolioChart({ compact = false }) {
  const last = portfolioSeries.at(-1);
  return (
    <div className={`chart-stack ${compact ? "chart-compact" : ""}`}>
      <div className="chart-legend" aria-hidden="true">
        <span><i className="legend-line legend-equity" />Net equity (USDT)</span>
        <span><i className="legend-line legend-benchmark" />Benchmark (BTCUSDT)</span>
        <span><i className="legend-marker marker-buy" />Trades</span>
        <span><i className="legend-marker marker-sell" />Sells</span>
      </div>
      <div className="main-chart" role="img" aria-label={`Net equity rose from 1.002M to ${last.equity / 1000}M USDT while the benchmark ended at ${last.benchmark / 1000}M.`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={portfolioSeries} margin={{ top: 10, right: 8, left: -14, bottom: 0 }}>
            <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} interval={3} />
            <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} domain={[980, 1320]} tickFormatter={(value) => `${(value / 1000).toFixed(2)}M`} />
            <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "var(--text-muted)" }} />
            <Line type="monotone" dataKey="benchmark" stroke="var(--chart-secondary)" strokeDasharray="4 4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="equity" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <ReferenceDot x="04:00" y={1058} r={4} fill="var(--positive)" stroke="var(--surface)" />
            <ReferenceDot x="17:00" y={1238} r={4} fill="var(--negative)" stroke="var(--surface)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {!compact && (
        <div className="drawdown-chart" role="img" aria-label="Thirty day drawdown stayed between zero and minus 4.2 percent.">
          <span className="drawdown-label">Drawdown (30D)</span>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={portfolioSeries} margin={{ top: 2, right: 8, left: 0, bottom: 0 }}>
              <Area type="monotone" dataKey="drawdown" stroke="var(--accent-muted)" fill="var(--chart-fill)" fillOpacity={0.5} strokeWidth={1} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      <details className="chart-data-fallback">
        <summary>View chart data</summary>
        <table>
          <thead><tr><th>Time</th><th>Equity</th><th>Benchmark</th><th>Drawdown</th></tr></thead>
          <tbody>{portfolioSeries.slice(-5).map((point) => <tr key={point.time}><td>{point.time}</td><td>{point.equity}K</td><td>{point.benchmark}K</td><td>{point.drawdown}%</td></tr>)}</tbody>
        </table>
      </details>
    </div>
  );
}

export function Field({ label, children, hint, error, className = "" }) {
  return (
    <label className={`field ${className}`}>
      <span className="field-label">{label}</span>
      {children}
      {hint && !error && <small>{hint}</small>}
      {error && <small className="field-error">{error}</small>}
    </label>
  );
}

export function Select({ value, onChange, children, ...props }) {
  return (
    <span className="select-wrap">
      <select value={value} onChange={(event) => onChange?.(event.target.value)} {...props}>{children}</select>
      <CaretDown size={14} aria-hidden="true" />
    </span>
  );
}

export function SearchField({ value, onChange, placeholder = "Search…", label = "Search" }) {
  return (
    <label className="search-field">
      <MagnifyingGlass size={16} aria-hidden="true" />
      <span className="sr-only">{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} />
      {value && <IconButton label="Clear search" onClick={() => onChange("")}><X size={14} /></IconButton>}
    </label>
  );
}

export function EmptyState({ icon, title, description, action }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function LoadingState({ rows = 5 }) {
  return (
    <div className="loading-state" aria-label="Loading" aria-busy="true">
      <div className="loading-title"><CircleNotch size={18} className="spin" /> Loading workspace</div>
      {Array.from({ length: rows }).map((_, index) => <span className="skeleton-row" key={index} />)}
    </div>
  );
}

export function ErrorState({ title = "Workspace unavailable", description, onRetry }) {
  return (
    <div className="error-state" role="alert">
      <Warning size={28} weight="duotone" />
      <h3>{title}</h3>
      <p>{description || "The service did not return a valid response. Your changes were not submitted."}</p>
      <button className="button button-secondary" onClick={onRetry}>Retry</button>
    </div>
  );
}

export function ThemePicker({ activeTheme, onChange, onClose }) {
  return (
    <div className="popover theme-popover" role="dialog" aria-label="Choose interface theme">
      <div className="popover-header">
        <div><strong>Interface theme</strong><span>Six calibrated brightness levels</span></div>
        <IconButton label="Close theme picker" onClick={onClose}><X size={16} /></IconButton>
      </div>
      <div className="theme-grid">
        {themes.map((theme) => (
          <button key={theme.id} className={`theme-option ${activeTheme === theme.id ? "is-active" : ""}`} onClick={() => onChange(theme.id)}>
            <span className="theme-swatch" style={{ background: theme.swatch }} />
            <span><strong>{theme.name}</strong><small>{theme.tone}</small></span>
            {activeTheme === theme.id && <CheckCircle size={18} weight="fill" />}
          </button>
        ))}
      </div>
    </div>
  );
}

export function FilterButton({ children, onClick, title }) {
  const unavailable = typeof onClick !== "function";
  return <button className="button button-secondary button-compact" disabled={unavailable} title={title || (unavailable ? "Not connected in this prototype" : undefined)} onClick={onClick}><Funnel size={14} />{children}</button>;
}

export function MoreButton({ label = "More options", onClick }) {
  return <IconButton label={label} disabled={!onClick} title={!onClick ? "No additional actions in this prototype" : label} onClick={onClick}><DotsThree size={18} weight="bold" /></IconButton>;
}

export function EntityLink({ children, onClick }) {
  return <button className="entity-link" onClick={onClick}>{children}<ArrowRight size={13} /></button>;
}

export function InspectorRow({ label, value, tone }) {
  return <div className="inspector-row"><span>{label}</span><strong className={tone ? `text-${tone}` : ""}>{value}</strong></div>;
}

export function NavTreeItem({ icon, label, count, active, onClick, nested = false }) {
  return (
    <button className={`nav-tree-item ${active ? "is-active" : ""} ${nested ? "is-nested" : ""}`} aria-label={label} onClick={onClick}>
      {icon}
      <span>{label}</span>
      {count !== undefined && <small>{count}</small>}
      {active && <CaretRight size={12} className="nav-active-caret" />}
    </button>
  );
}
