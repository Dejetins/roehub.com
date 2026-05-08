import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { createPoller } from "../core/poller.js";
import { t } from "../core/locale.js";

const DEFAULT_ENDPOINT = "/api/ui/strategies/dashboard";
const DEFAULT_CLONE_PATH = "/api/strategies/clone";
const DEFAULT_REFRESH_PRESET = "15s";
const REFRESH_PRESETS = {
  off: 0,
  "10s": 10000,
  "15s": 15000,
  "30s": 30000,
  "1m": 60000,
  "5m": 300000,
};
const SVG_NS = "http://www.w3.org/2000/svg";

const metricLabelKeys = {
  total_return: "strategies.metric.total_return",
  best_sharpe: "strategies.metric.best_sharpe",
  max_drawdown: "strategies.metric.max_drawdown",
  profit_factor: "strategies.metric.profit_factor",
  win_rate: "strategies.metric.win_rate",
  trades: "strategies.metric.trades",
  avg_hold: "strategies.metric.avg_hold",
  exposure: "strategies.metric.exposure",
  avg_trade: "strategies.metric.avg_trade",
  best_month: "strategies.metric.best_month",
  worst_month: "strategies.metric.worst_month",
  profitable_months: "strategies.metric.profitable_months",
};

const breakdownLabelKeys = {
  win_rate: "strategies.metric.win_rate",
  trades: "strategies.metric.trades",
  return: "strategies.metric.total_return",
  profit_factor: "strategies.metric.profit_factor",
  avg_trade: "strategies.metric.avg_trade",
  avg_holding_time: "strategies.risk.avg_holding_time",
  avg_bars_in_trade: "strategies.risk.avg_bars_in_trade",
  commissions: "strategies.risk.commissions",
  execution_paid: "strategies.risk.execution_paid",
  max_consecutive_losses: "strategies.risk.max_consecutive_losses",
  worst_trade: "strategies.risk.worst_trade",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function localTime(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "--";
  }
  return new Intl.DateTimeFormat(document.documentElement.lang || "en", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function valueOrUnavailable(value) {
  if (value === null || value === undefined || value === "" || value === "Unavailable") {
    return t("common.unavailable");
  }
  return value;
}

function financialClass(direction) {
  if (direction === "positive") {
    return "rh-financial--positive";
  }
  if (direction === "negative") {
    return "rh-financial--negative";
  }
  return "rh-financial--neutral";
}

function directionFromNumber(value) {
  const number = Number(value);
  if (number > 0) {
    return "positive";
  }
  if (number < 0) {
    return "negative";
  }
  return "neutral";
}

function panelStatusText(state, reason, source = null) {
  const stateKey = {
    ready: "strategies.panel.ready",
    empty: "strategies.panel.empty",
    degraded: "strategies.panel.degraded",
    unavailable: "strategies.panel.unavailable",
  }[state || "unavailable"];
  const base = t(stateKey || "strategies.panel.unavailable");
  const sourceSuffix = source ? ` / ${source}` : "";
  return reason ? `${base}${sourceSuffix}: ${reason}` : `${base}${sourceSuffix}`;
}

function normalizeSearchValue(value) {
  return String(value ?? "").trim().toLowerCase();
}

function savedRowMatchesQuery(row, query) {
  if (!query) {
    return true;
  }
  return [
    row.strategy_id,
    row.name,
    row.version,
    row.status,
    row.latest_activity,
  ].some((value) => normalizeSearchValue(value).includes(query));
}

function clearSvg(svg) {
  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }
}

function appendSvg(svg, tagName, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tagName);
  Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
  svg.appendChild(node);
  return node;
}

function drawPlaceholderChart(svg, { negative = false } = {}) {
  clearSvg(svg);
  const viewBox = (svg.getAttribute("viewBox") || "0 0 360 120").split(/\s+/).map(Number);
  const width = viewBox[2] || 360;
  const height = viewBox[3] || 120;
  const padding = Math.max(16, Math.round(width * 0.04));
  for (let index = 0; index <= 8; index += 1) {
    const x = padding + ((width - padding * 2) * index) / 8;
    appendSvg(svg, "line", {
      x1: x,
      y1: padding,
      x2: x,
      y2: height - padding,
      stroke: "var(--rh-chart-grid-line)",
      "stroke-width": 1,
    });
  }
  for (let index = 0; index <= 4; index += 1) {
    const y = padding + ((height - padding * 2) * index) / 4;
    appendSvg(svg, "line", {
      x1: padding,
      y1: y,
      x2: width - padding,
      y2: y,
      stroke: "var(--rh-chart-grid-line)",
      "stroke-width": 1,
    });
  }
  const mid = height * (negative ? 0.68 : 0.46);
  appendSvg(svg, "polyline", {
    points: [
      `${padding},${mid}`,
      `${width * 0.24},${mid + (negative ? 14 : -12)}`,
      `${width * 0.42},${mid + (negative ? -4 : 10)}`,
      `${width * 0.62},${mid + (negative ? 18 : -8)}`,
      `${width - padding},${mid + (negative ? 5 : -18)}`,
    ].join(" "),
    fill: "none",
    stroke: negative ? "var(--rh-financial-negative)" : "var(--rh-accent)",
    "stroke-width": 2,
    "stroke-dasharray": "6 6",
  });
}

function renderSelected(root, selected) {
  setText("[data-selected-name]", selected?.name || t("common.unavailable"), root);
  setText("[data-selected-version]", selected?.version || "--", root);
  setText("[data-selected-exchange]", selected?.exchange || t("common.unavailable"), root);
  setText("[data-selected-symbols]", (selected?.symbols || []).join(", ") || "--", root);
  setText("[data-selected-capital]", valueOrUnavailable(selected?.capital_usdt), root);
  setText("[data-selected-commission]", valueOrUnavailable(selected?.commission_percent), root);
  setText("[data-selected-period]", valueOrUnavailable(null), root);
  setText("[data-selected-direction]", selected?.direction || "--", root);
  setText("[data-selected-timeframe]", selected?.timeframe || "--", root);
  setText("[data-selected-slippage]", valueOrUnavailable(selected?.slippage_percent), root);
  setText("[data-selected-run-state]", selected?.run_state || selected?.status || "--", root);
  setText("[data-selected-updated]", localTime(selected?.latest_update), root);
  setText("[data-selected-status]", selected?.status || t("strategies.status.unknown"), root);
  setText("[data-command-state]", selected?.state === "ready" ? t("strategies.status.ready") : t("strategies.status.degraded"), root);
}

function renderMetrics(root, metricGrid) {
  const target = qs("[data-strategy-metrics]", root);
  if (!target) {
    return;
  }
  const items = metricGrid?.items || [];
  target.innerHTML = items
    .map((metric) => {
      const label = t(metricLabelKeys[metric.key] || metric.label || metric.key);
      return `
        <article class="strategies-stat">
          <span class="strategies-stat__label">${escapeHtml(label)}</span>
          <strong class="strategies-stat__value ${financialClass(metric.direction)}">${escapeHtml(valueOrUnavailable(metric.formatted))}</strong>
          <span class="strategies-stat__source">${escapeHtml(metric.source)} / ${escapeHtml(metric.status)}</span>
        </article>
      `;
    })
    .join("");
}

function renderSelector(root, selector, savedQuery = "") {
  const selectedId = selector?.selected_strategy_id || "";
  const selectedRow = (selector?.items || []).find((row) => row.strategy_id === selectedId);
  setText("[data-selector-current]", selectedRow?.name || selectedId || "--", root);
  const menu = qs("[data-strategy-selector-menu]", root);
  if (menu) {
    const rows = selector?.items || [];
    if (!rows.length) {
      menu.innerHTML = `<span class="rh-menu-item strategies-empty-block">${escapeHtml(panelStatusText(selector?.state, selector?.degradation_reason, selector?.source))}</span>`;
    } else {
      menu.innerHTML = rows
        .map((row) => `
          <button
            class="rh-menu-item"
            type="button"
            role="option"
            aria-selected="${row.strategy_id === selectedId ? "true" : "false"}"
            data-select-strategy="${escapeHtml(row.strategy_id)}"
          >
            <span>${escapeHtml(row.name)}</span>
            <span class="rh-dropdown__value">${escapeHtml(row.version || "--")} / ${escapeHtml(row.status)}</span>
          </button>
        `)
        .join("");
    }
  }

  const saved = qs("[data-strategy-saved-rows]", root);
  if (!saved) {
    return;
  }
  const rows = selector?.items || [];
  const query = normalizeSearchValue(savedQuery);
  const filteredRows = rows.filter((row) => savedRowMatchesQuery(row, query));
  if (!rows.length) {
    saved.innerHTML = `
      <tr><td class="strategies-empty-row" colspan="3">
        ${escapeHtml(panelStatusText(selector?.state, selector?.degradation_reason, selector?.source))}
      </td></tr>
    `;
    return;
  }
  if (!filteredRows.length) {
    saved.innerHTML = `
      <tr><td class="strategies-empty-row" colspan="3">
        ${escapeHtml(t("strategies.saved.no_matches"))}
      </td></tr>
    `;
    return;
  }
  saved.innerHTML = filteredRows
    .map((row) => `
      <tr data-saved-strategy-id="${escapeHtml(row.strategy_id)}">
        <td>${escapeHtml(row.name)}</td>
        <td>${escapeHtml(row.version || "--")}</td>
        <td>${escapeHtml(localTime(row.latest_activity))}</td>
      </tr>
    `)
    .join("");
}

function renderChart(root, chart) {
  const svg = root.querySelector("[data-strategy-chart]");
  if (svg instanceof SVGElement) {
    drawPlaceholderChart(svg);
  }
  setText("[data-chart-symbol]", chart?.symbol || "--", root);
  setText(
    "[data-chart-state]",
    panelStatusText(chart?.state, chart?.degradation_reason, chart?.source),
    root,
  );
}

function renderMonthly(root, monthly) {
  const head = qs("[data-monthly-head]", root);
  const body = qs("[data-monthly-rows]", root);
  if (!head || !body) {
    return;
  }
  const columns = monthly?.columns || [];
  head.innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  if (!monthly?.rows?.length) {
    body.innerHTML = `
      <tr><td class="strategies-empty-row" colspan="${Math.max(columns.length, 1)}">
        ${escapeHtml(panelStatusText(monthly?.state, monthly?.degradation_reason, monthly?.source))}
      </td></tr>
    `;
    return;
  }
  body.innerHTML = monthly.rows
    .map((row) => `
      <tr>${columns.map((column) => `<td>${escapeHtml(row[column] ?? "--")}</td>`).join("")}</tr>
    `)
    .join("");
}

function renderStatTiles(root, monthly) {
  const target = qs("[data-stat-tiles]", root);
  if (!target) {
    return;
  }
  target.innerHTML = (monthly?.summary || [])
    .map((metric) => `
      <article class="strategies-tile">
        <span class="strategies-tile__label">${escapeHtml(t(metricLabelKeys[metric.key] || metric.key))}</span>
        <strong class="strategies-tile__value ${financialClass(metric.direction)}">${escapeHtml(valueOrUnavailable(metric.formatted))}</strong>
        <span class="strategies-tile__source">${escapeHtml(metric.source)} / ${escapeHtml(metric.status)}</span>
      </article>
    `)
    .join("");
}

function renderLongShort(root, panel) {
  const target = qs("[data-long-short-rows]", root);
  if (!target) {
    return;
  }
  const rows = panel?.rows || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="4">${escapeHtml(panelStatusText(panel?.state, panel?.degradation_reason, panel?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(t(breakdownLabelKeys[row.key] || row.label || row.key))}</td>
        <td>${escapeHtml(valueOrUnavailable(row.long_value))}</td>
        <td>${escapeHtml(valueOrUnavailable(row.short_value))}</td>
        <td class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.total_value))}</td>
      </tr>
    `)
    .join("");
}

function renderRisk(root, panel) {
  const target = qs("[data-risk-execution]", root);
  if (!target) {
    return;
  }
  target.innerHTML = (panel?.rows || [])
    .map((row) => `
      <div>
        <dt>${escapeHtml(t(breakdownLabelKeys[row.key] || row.label || row.key))}</dt>
        <dd class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.total_value))}</dd>
      </div>
    `)
    .join("");
}

function renderSeries(root, panel, selector, { negative = false } = {}) {
  const svg = root.querySelector(selector.chart);
  if (svg instanceof SVGElement) {
    drawPlaceholderChart(svg, { negative });
  }
  setText(
    selector.state,
    panelStatusText(panel?.state, panel?.degradation_reason, panel?.source),
    root,
  );
}

function renderDays(root, panel) {
  const target = qs("[data-best-worst-days]", root);
  if (!target) {
    return;
  }
  if (!panel?.best_days?.length && !panel?.worst_days?.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="4">${escapeHtml(panelStatusText(panel?.state, panel?.degradation_reason, panel?.source))}</td></tr>`;
    return;
  }
  const maxRows = Math.max(panel.best_days.length, panel.worst_days.length);
  target.innerHTML = Array.from({ length: maxRows }, (_, index) => {
    const best = panel.best_days[index] || {};
    const worst = panel.worst_days[index] || {};
    return `
      <tr>
        <td>${escapeHtml(best.date || "--")}</td>
        <td class="${financialClass(best.direction)}">${escapeHtml(best.formatted || "--")}</td>
        <td>${escapeHtml(worst.date || "--")}</td>
        <td class="${financialClass(worst.direction)}">${escapeHtml(worst.formatted || "--")}</td>
      </tr>
    `;
  }).join("");
}

function renderHours(root, panel) {
  const target = qs("[data-hourly-results]", root);
  if (!target) {
    return;
  }
  const rows = panel?.items || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="3">${escapeHtml(panelStatusText(panel?.state, panel?.degradation_reason, panel?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.hour_bucket)}</td>
        <td>${escapeHtml(valueOrUnavailable(row.win_rate_percent))}</td>
        <td class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.pnl_percent))}</td>
      </tr>
    `)
    .join("");
}

function renderTrades(root, trades) {
  const target = qs("[data-trade-rows]", root);
  if (!target) {
    return;
  }
  const items = trades?.items || [];
  if (!items.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="14">${escapeHtml(panelStatusText(trades?.state, trades?.degradation_reason, trades?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.row_number)}</td>
        <td>${escapeHtml(item.symbol)}</td>
        <td class="${item.side === "short" ? "rh-financial--negative" : "rh-financial--positive"}">${escapeHtml(item.side)}</td>
        <td>${escapeHtml(localTime(item.entry_time))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.entry))}</td>
        <td>${escapeHtml(localTime(item.exit_time))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.exit))}</td>
        <td class="${financialClass(directionFromNumber(item.pnl_percent))}">${escapeHtml(valueOrUnavailable(item.pnl_percent))}</td>
        <td class="${financialClass(directionFromNumber(item.pnl_usdt))}">${escapeHtml(valueOrUnavailable(item.pnl_usdt))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.bars))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.hold_time))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.phase))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.reason))}</td>
        <td>${escapeHtml(valueOrUnavailable(item.note))}</td>
      </tr>
    `)
    .join("");
}

function renderSymbols(root, symbols) {
  const target = qs("[data-symbol-results]", root);
  if (!target) {
    return;
  }
  const rows = symbols?.items || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="5">${escapeHtml(panelStatusText(symbols?.state, symbols?.degradation_reason, symbols?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.symbol)}</td>
        <td>${escapeHtml(valueOrUnavailable(row.trades))}</td>
        <td>${escapeHtml(valueOrUnavailable(row.win_rate_percent))}</td>
        <td class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.pnl_percent))}</td>
        <td class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.pnl_usdt))}</td>
      </tr>
    `)
    .join("");
}

function renderFooter(summary) {
  const footer = summary?.footer_status || {};
  setText("[data-footer-connection]", footer.connection_status || "--", document);
  setText("[data-footer-data]", footer.data_status || "--", document);
  setText("[data-footer-api]", footer.api_label || "--", document);
  setText("[data-footer-latency]", footer.latency_ms ? `${footer.latency_ms} ms` : "--", document);
  setText("[data-footer-capital]", valueOrUnavailable(footer.capital_usdt), document);
  setText("[data-footer-time]", localTime(footer.server_time), document);
}

function renderFreshness(summary) {
  const retry = summary?.retry_after_seconds;
  const sources = summary?.sources || [];
  const available = sources.filter((source) => source.status === "available").length;
  if (retry) {
    setText(
      "[data-strategies-freshness]",
      t("strategies.refresh.rate_limited", { seconds: retry }),
      document,
    );
    return;
  }
  setText(
    "[data-strategies-freshness]",
    t("strategies.refresh.freshness", {
      status: summary?.refresh_status || "unknown",
      sources: `${available}/${sources.length}`,
      time: localTime(summary?.generated_at),
    }),
    document,
  );
}

function renderDashboard(root, summary, state = {}) {
  renderSelected(root, summary.selected_strategy);
  renderSelector(root, summary.strategy_selector, state.savedQuery);
  renderChart(root, summary.chart);
  renderMetrics(root, summary.metric_grid);
  renderMonthly(root, summary.monthly_stats);
  renderStatTiles(root, summary.monthly_stats);
  renderLongShort(root, summary.long_short);
  renderRisk(root, summary.risk_execution);
  renderSeries(root, summary.drawdown, {
    chart: "[data-drawdown-chart]",
    state: "[data-drawdown-state]",
  }, { negative: true });
  renderSeries(root, summary.equity_curve, {
    chart: "[data-equity-chart]",
    state: "[data-equity-state]",
  });
  renderDays(root, summary.best_worst_days);
  renderHours(root, summary.hourly_results);
  renderTrades(root, summary.trades);
  renderSymbols(root, summary.symbol_results);
  renderFooter(summary);
  renderFreshness(summary);
  setText("[data-strategies-refresh-status]", summary.refresh_status || t("refresh.idle"), document);
}

function openCreate(root) {
  const panel = qs("[data-strategy-create-panel]", root);
  if (panel) {
    panel.hidden = false;
    root.dataset.strategyMode = "create";
  }
}

function closeCreate(root) {
  const panel = qs("[data-strategy-create-panel]", root);
  if (panel) {
    panel.hidden = true;
    root.dataset.strategyMode = "dashboard";
  }
}

function buildCreatePayload(form, state) {
  const symbol = String(form.elements.symbol?.value || "").trim().toUpperCase();
  const marketId = Number(form.elements.market_id?.value || 0);
  const instrumentKey = String(form.elements.instrument_key?.value || "").trim();
  const signalTemplate = String(form.elements.signal_template?.value || "").trim();
  const indicators = JSON.parse(String(form.elements.indicators?.value || "[]"));
  if (!symbol || !Number.isInteger(marketId) || marketId <= 0 || !instrumentKey) {
    throw new Error(t("strategies.create.invalid_identity"));
  }
  if (!Array.isArray(indicators)) {
    throw new Error(t("strategies.create.invalid_indicators"));
  }
  return {
    instrument_id: {
      market_id: marketId,
      symbol,
    },
    instrument_key: instrumentKey,
    market_type: state.createMarketType,
    timeframe: state.createTimeframe,
    indicators,
    ...(signalTemplate ? { signal_template: signalTemplate } : {}),
  };
}

function setBrowserStrategyState(strategyId) {
  if (!strategyId) {
    return;
  }
  const url = new URL(window.location.href);
  url.pathname = "/strategies";
  url.searchParams.set("strategy_id", strategyId);
  url.searchParams.delete("mode");
  window.history.replaceState({}, "", `${url.pathname}${url.search}`);
}

function initStrategies(root) {
  const endpoint = root.dataset.dashboardEndpoint || DEFAULT_ENDPOINT;
  const state = {
    selectedStrategyId: root.dataset.initialStrategyId || null,
    selectorState: "all",
    createMarketType: "spot",
    createTimeframe: "1m",
    savedQuery: "",
  };
  let activeRequest = null;
  let poller = null;
  let delayedStart = null;
  let lastSummary = null;
  let manualRefreshBlockedUntilMs = 0;
  let manualRefreshGateTimer = null;

  function isManualRefreshBlocked() {
    return manualRefreshBlockedUntilMs > Date.now();
  }

  function manualRefreshRetrySeconds() {
    return Math.max(1, Math.ceil((manualRefreshBlockedUntilMs - Date.now()) / 1000));
  }

  function setRunning(isRunning) {
    qsa("[data-strategies-refresh]", document).forEach((button) => {
      if (button instanceof HTMLButtonElement) {
        button.disabled = isRunning;
      }
    });
  }

  function syncManualRefreshGate(summary) {
    if (manualRefreshGateTimer) {
      window.clearTimeout(manualRefreshGateTimer);
      manualRefreshGateTimer = null;
    }
    const rawNextAllowed = summary?.next_allowed_refresh_at
      || summary?.refresh_control?.next_allowed_refresh_at
      || null;
    const nextAllowedMs = rawNextAllowed ? Date.parse(rawNextAllowed) : Number.NaN;
    manualRefreshBlockedUntilMs = Number.isFinite(nextAllowedMs) && nextAllowedMs > Date.now()
      ? nextAllowedMs
      : 0;
    if (isManualRefreshBlocked()) {
      manualRefreshGateTimer = window.setTimeout(() => {
        manualRefreshGateTimer = null;
        manualRefreshBlockedUntilMs = 0;
        if (lastSummary) {
          renderFreshness({ ...lastSummary, retry_after_seconds: null });
        }
        setRunning(false);
      }, Math.max(1, manualRefreshBlockedUntilMs - Date.now()));
    }
    setRunning(false);
  }

  function dashboardUrl(reason) {
    const params = new URLSearchParams();
    params.set("refresh", reason);
    params.set("state", state.selectorState);
    if (state.selectedStrategyId) {
      params.set("strategy_id", state.selectedStrategyId);
    }
    return `${endpoint}?${params.toString()}`;
  }

  async function loadDashboard(reason = "auto") {
    if (reason === "manual" && isManualRefreshBlocked()) {
      setText("[data-strategies-refresh-status]", "rate_limited", document);
      setText(
        "[data-strategies-freshness]",
        t("strategies.refresh.rate_limited", { seconds: manualRefreshRetrySeconds() }),
        document,
      );
      return lastSummary;
    }
    if (activeRequest) {
      return activeRequest;
    }
    setRunning(true);
    activeRequest = apiFetch(dashboardUrl(reason))
      .then((summary) => {
        lastSummary = summary;
        root.dataset.strategiesLoaded = "true";
        root.dataset.strategiesLastRefresh = reason;
        state.selectedStrategyId = summary?.selected_strategy?.strategy_id || state.selectedStrategyId;
        renderDashboard(root, summary, state);
        syncManualRefreshGate(summary);
        return summary;
      })
      .catch((error) => {
        setText("[data-strategies-refresh-status]", error.code || "failed", document);
        setText("[data-strategies-freshness]", error.message || t("strategies.refresh.failed"), document);
        throw error;
      })
      .finally(() => {
        activeRequest = null;
        setRunning(false);
      });
    return activeRequest;
  }

  function stopAutorefresh() {
    if (delayedStart) {
      window.clearTimeout(delayedStart);
      delayedStart = null;
    }
    if (poller) {
      poller.stop();
      poller = null;
    }
  }

  function setAutorefresh(presetKey) {
    stopAutorefresh();
    const intervalMs = REFRESH_PRESETS[presetKey] ?? 0;
    setText("[data-strategies-refresh-current]", presetKey, document);
    qsa("[data-strategies-refresh-preset]", document).forEach((item) => {
      item.setAttribute(
        "aria-selected",
        item.dataset.strategiesRefreshPreset === presetKey ? "true" : "false",
      );
    });
    if (intervalMs <= 0) {
      root.dataset.strategiesAutorefresh = "off";
      return;
    }
    root.dataset.strategiesAutorefresh = presetKey;
    poller = createPoller(() => loadDashboard("auto"), {
      intervalMs,
      hiddenTabPause: true,
    });
    delayedStart = window.setTimeout(() => {
      delayedStart = null;
      poller?.start();
    }, intervalMs);
  }

  function closeStatusRefreshMenu() {
    const trigger = qs("#strategies-refresh-trigger", document);
    const menu = qs("#strategies-refresh-menu", document);
    const dropdown = trigger?.closest("[data-rh-dropdown]");
    if (trigger instanceof HTMLElement) {
      trigger.setAttribute("aria-expanded", "false");
    }
    if (menu instanceof HTMLElement) {
      menu.hidden = true;
    }
    if (dropdown instanceof HTMLElement) {
      dropdown.dataset.open = "false";
    }
  }

  function positionStatusRefreshMenu() {
    const trigger = qs("#strategies-refresh-trigger", document);
    const menu = qs("#strategies-refresh-menu", document);
    if (!(trigger instanceof HTMLElement) || !(menu instanceof HTMLElement) || menu.hidden) {
      return;
    }
    const triggerRect = trigger.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const left = Math.min(Math.max(8, triggerRect.left), window.innerWidth - menuRect.width - 8);
    const top = Math.max(8, triggerRect.top - menuRect.height - 6);
    menu.style.position = "fixed";
    menu.style.insetInlineStart = `${left}px`;
    menu.style.insetInlineEnd = "auto";
    menu.style.insetBlockStart = `${top}px`;
    menu.style.insetBlockEnd = "auto";
  }

  qsa("[data-strategies-refresh]", document).forEach((button) => {
    button.addEventListener("click", () => {
      loadDashboard("manual").catch(() => null);
    });
  });
  qs("#strategies-refresh-trigger", document)?.addEventListener("click", () => {
    window.requestAnimationFrame(positionStatusRefreshMenu);
  });
  qsa("[data-strategies-refresh-preset]", document).forEach((item) => {
    item.addEventListener("click", () => {
      setAutorefresh(item.dataset.strategiesRefreshPreset || "off");
      closeStatusRefreshMenu();
    });
  });
  window.addEventListener("resize", positionStatusRefreshMenu);
  window.addEventListener("scroll", positionStatusRefreshMenu, true);
  qs("[data-saved-search]", root)?.addEventListener("input", (event) => {
    if (event.target instanceof HTMLInputElement) {
      state.savedQuery = event.target.value;
      if (lastSummary) {
        renderSelector(root, lastSummary.strategy_selector, state.savedQuery);
      }
    }
  });
  root.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const strategyOption = event.target.closest("[data-select-strategy]");
    if (strategyOption instanceof HTMLElement) {
      state.selectedStrategyId = strategyOption.dataset.selectStrategy || null;
      setBrowserStrategyState(state.selectedStrategyId);
      loadDashboard("manual").catch(() => null);
      return;
    }
    const stateOption = event.target.closest("[data-strategy-state]");
    if (stateOption instanceof HTMLElement) {
      state.selectorState = stateOption.dataset.strategyState === "active" ? "active" : "all";
      setText("[data-state-current]", state.selectorState, root);
      loadDashboard("manual").catch(() => null);
      return;
    }
    const createOpen = event.target.closest("[data-strategy-create-open]");
    if (createOpen) {
      openCreate(root);
      return;
    }
    const createClose = event.target.closest("[data-strategy-create-close]");
    if (createClose) {
      closeCreate(root);
      return;
    }
    const market = event.target.closest("[data-create-market]");
    if (market instanceof HTMLElement) {
      state.createMarketType = market.dataset.createMarket || "spot";
      setText("[data-create-market-current]", state.createMarketType, root);
      return;
    }
    const timeframe = event.target.closest("[data-create-timeframe]");
    if (timeframe instanceof HTMLElement) {
      state.createTimeframe = timeframe.dataset.createTimeframe || "1m";
      setText("[data-create-timeframe-current]", state.createTimeframe, root);
      return;
    }
  });

  qs("[data-strategy-load]", root)?.addEventListener("click", () => {
    loadDashboard("manual").catch(() => null);
  });
  qs("[data-strategy-run]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    const path = root.dataset.apiRunPathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    apiFetch(path, { method: "POST" }).then(() => loadDashboard("manual")).catch(() => null);
  });
  qs("[data-strategy-stop]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    const path = root.dataset.apiStopPathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    apiFetch(path, { method: "POST" }).then(() => loadDashboard("manual")).catch(() => null);
  });
  qs("[data-strategy-clone]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    apiFetch(root.dataset.apiClonePath || DEFAULT_CLONE_PATH, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ source_strategy_id: state.selectedStrategyId }),
    })
      .then((strategy) => {
        state.selectedStrategyId = strategy.strategy_id;
        setBrowserStrategyState(strategy.strategy_id);
        return loadDashboard("manual");
      })
      .catch(() => null);
  });
  qs("[data-strategy-delete]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId || !window.confirm(t("strategies.confirm_delete"))) {
      return;
    }
    const path = root.dataset.apiDeletePathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    apiFetch(path, { method: "DELETE" })
      .then(() => {
        state.selectedStrategyId = null;
        window.history.replaceState({}, "", "/strategies");
        return loadDashboard("manual");
      })
      .catch(() => null);
  });

  const form = qs("[data-strategy-create-form]", root);
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const status = qs("[data-strategy-create-status]", root);
    try {
      const payload = buildCreatePayload(form, state);
      if (status) {
        status.textContent = t("strategies.create.submitting");
      }
      apiFetch(root.dataset.apiCreatePath, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      })
        .then((strategy) => {
          state.selectedStrategyId = strategy.strategy_id;
          setBrowserStrategyState(strategy.strategy_id);
          closeCreate(root);
          return loadDashboard("manual");
        })
        .catch((error) => {
          if (status) {
            status.textContent = error.message || t("strategies.create.failed");
          }
        });
    } catch (error) {
      if (status) {
        status.textContent = error.message || t("strategies.create.failed");
      }
    }
  });

  window.__roehubStrategies = {
    loadDashboard,
    setAutorefresh,
    openCreate: () => openCreate(root),
    get activeRequest() {
      return activeRequest;
    },
    get lastSummary() {
      return lastSummary;
    },
  };

  if (root.dataset.initialMode === "create") {
    openCreate(root);
  }
  loadDashboard("initial")
    .catch(() => null)
    .finally(() => setAutorefresh(DEFAULT_REFRESH_PRESET));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    const root = qs("[data-strategies-root]");
    if (root) {
      initStrategies(root);
    }
  });
} else {
  const root = qs("[data-strategies-root]");
  if (root) {
    initStrategies(root);
  }
}
