import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { createPoller } from "../core/poller.js";
import { t } from "../core/locale.js";

const SUMMARY_ENDPOINT = "/api/ui/dashboard/summary";
const REFRESH_PRESETS = {
  off: 0,
  "10s": 10000,
  "15s": 15000,
  "30s": 30000,
  "1m": 60000,
  "5m": 300000,
};
const DEFAULT_REFRESH_PRESET = "15s";
const SVG_NS = "http://www.w3.org/2000/svg";

const metricLabelKeys = {
  total_pnl: "dashboard.metric.total_pnl",
  unrealized_pnl: "dashboard.metric.unrealized_pnl",
  realized_pnl: "dashboard.metric.realized_pnl",
  roi: "dashboard.metric.roi",
  win_rate: "dashboard.metric.win_rate",
  open_positions: "dashboard.metric.open_positions",
  equity: "dashboard.metric.equity",
  max_drawdown: "dashboard.metric.max_drawdown",
  exposure: "dashboard.metric.exposure",
  trades_today: "dashboard.metric.trades_today",
  uptime: "dashboard.metric.uptime",
};

const sourceStatusKeys = {
  available: "dashboard.source.available",
  degraded: "dashboard.source.degraded",
  unavailable: "dashboard.source.unavailable",
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
    second: "2-digit",
  }).format(date);
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

function panelStatusText(state, reason) {
  if (state === "ready") {
    return t("dashboard.panel.ready");
  }
  if (state === "empty") {
    return t("dashboard.panel.empty");
  }
  if (state === "degraded") {
    return reason || t("dashboard.panel.degraded");
  }
  return reason || t("dashboard.panel.unavailable");
}

function rowStateClass(status) {
  if (status === "live") {
    return "dashboard-row-dot--live";
  }
  if (status === "degraded" || status === "unknown") {
    return "dashboard-row-dot--degraded";
  }
  return "";
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

function drawChart(svg, points) {
  clearSvg(svg);
  const width = 720;
  const height = 250;
  const padding = 28;
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
  for (let index = 0; index <= 5; index += 1) {
    const y = padding + ((height - padding * 2) * index) / 5;
    appendSvg(svg, "line", {
      x1: padding,
      y1: y,
      x2: width - padding,
      y2: y,
      stroke: "var(--rh-chart-grid-line)",
      "stroke-width": 1,
    });
  }

  const usablePoints = points.filter((point) => Number.isFinite(Number(point.equity)));
  if (usablePoints.length < 2) {
    appendSvg(svg, "line", {
      x1: padding,
      y1: height / 2,
      x2: width - padding,
      y2: height / 2,
      stroke: "var(--rh-line-muted)",
      "stroke-width": 2,
      "stroke-dasharray": "8 8",
    });
    return;
  }

  const values = usablePoints.map((point) => Number(point.equity));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const path = usablePoints
    .map((point, index) => {
      const x = padding + ((width - padding * 2) * index) / (usablePoints.length - 1);
      const y = height - padding - ((Number(point.equity) - min) / spread) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  appendSvg(svg, "path", {
    d: path,
    fill: "none",
    stroke: "var(--rh-accent)",
    "stroke-width": 2,
  });
}

function drawSparkline(values) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "dashboard-sparkline");
  svg.setAttribute("viewBox", "0 0 72 30");
  if (!Array.isArray(values) || values.length < 2) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", "4");
    line.setAttribute("y1", "15");
    line.setAttribute("x2", "68");
    line.setAttribute("y2", "15");
    line.setAttribute("stroke", "var(--rh-line-muted)");
    line.setAttribute("stroke-dasharray", "4 4");
    svg.appendChild(line);
    return svg.outerHTML;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const path = values
    .map((value, index) => {
      const x = 4 + (64 * index) / (values.length - 1);
      const y = 26 - ((value - min) / spread) * 22;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  const line = document.createElementNS(SVG_NS, "path");
  line.setAttribute("d", path);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--rh-financial-positive)");
  line.setAttribute("stroke-width", "1.6");
  svg.appendChild(line);
  return svg.outerHTML;
}

function renderSelected(root, snapshot) {
  setText("[data-selected-name]", snapshot?.name || t("common.unavailable"), root);
  setText("[data-selected-version]", snapshot?.version || "--", root);
  setText("[data-selected-exchange]", snapshot?.exchange || t("common.unavailable"), root);
  setText("[data-selected-symbols]", (snapshot?.symbols || []).join(", ") || "--", root);
  setText("[data-selected-timeframe]", snapshot?.timeframe || "--", root);
  setText("[data-selected-mode]", snapshot?.mode || "--", root);
  setText("[data-selected-capital]", snapshot?.capital || t("common.unavailable"), root);
  setText("[data-selected-leverage]", snapshot?.leverage || t("common.unavailable"), root);
  setText("[data-selected-updated]", localTime(snapshot?.latest_update), root);
  setText("[data-selected-status]", snapshot?.status || t("dashboard.status.unknown"), root);
}

function renderChart(root, series) {
  const svg = root.querySelector("[data-dashboard-chart]");
  if (svg instanceof SVGElement) {
    drawChart(svg, series?.points || []);
  }
  setText(
    "[data-chart-state]",
    panelStatusText(series?.state, series?.degradation_reason),
    root,
  );
}

function renderMetrics(root, metrics) {
  const target = qs("[data-dashboard-metrics]", root);
  if (!target) {
    return;
  }
  target.innerHTML = (metrics || [])
    .map((metric) => {
      const label = t(metricLabelKeys[metric.key] || metric.label || metric.key);
      const status = t(sourceStatusKeys[metric.status] || "dashboard.source.unavailable");
      return `
        <article class="dashboard-metric">
          <span class="dashboard-metric__label">${escapeHtml(label)}</span>
          <strong class="dashboard-metric__value ${financialClass(metric.direction)}">
            ${escapeHtml(metric.formatted)}
          </strong>
          <span class="dashboard-metric__source">${escapeHtml(metric.source)} / ${escapeHtml(status)}</span>
        </article>
      `;
    })
    .join("");
}

function renderPositions(root, positions) {
  const tbody = qs("[data-open-positions]", root);
  if (!tbody) {
    return;
  }
  const items = positions?.items || [];
  if (!items.length) {
    tbody.innerHTML = `
      <tr><td class="dashboard-empty-row" colspan="9">
        ${escapeHtml(panelStatusText(positions?.state, positions?.degradation_reason))}
      </td></tr>
    `;
    return;
  }
  tbody.innerHTML = items
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.symbol)}</td>
        <td>${escapeHtml(item.side)}</td>
        <td>${escapeHtml(item.entry ?? "--")}</td>
        <td>${escapeHtml(item.mark ?? "--")}</td>
        <td class="${financialClass(item.pnl > 0 ? "positive" : item.pnl < 0 ? "negative" : "neutral")}">
          ${escapeHtml(item.pnl ?? "--")}
        </td>
        <td>${escapeHtml(item.roe_percent ?? "--")}</td>
        <td>${escapeHtml(item.leverage ?? "--")}</td>
        <td>${escapeHtml(localTime(item.opened_at))}</td>
        <td><button class="rh-button rh-button--compact" type="button" ${item.can_close ? "" : "disabled"}>${escapeHtml(t("dashboard.action.close"))}</button></td>
      </tr>
    `)
    .join("");
}

function renderExecutions(root, executions) {
  const tbody = qs("[data-recent-executions]", root);
  if (!tbody) {
    return;
  }
  const items = executions?.items || [];
  if (!items.length) {
    tbody.innerHTML = `
      <tr><td class="dashboard-empty-row" colspan="8">
        ${escapeHtml(panelStatusText(executions?.state, executions?.degradation_reason))}
      </td></tr>
    `;
    return;
  }
  tbody.innerHTML = items
    .map((item) => `
      <tr>
        <td>${escapeHtml(localTime(item.timestamp))}</td>
        <td>${escapeHtml(item.symbol)}</td>
        <td class="${item.side === "sell" ? "rh-financial--negative" : "rh-financial--positive"}">${escapeHtml(item.side)}</td>
        <td>${escapeHtml(item.price ?? "--")}</td>
        <td>${escapeHtml(item.quantity ?? "--")}</td>
        <td>${escapeHtml(item.fee ?? "--")}</td>
        <td class="${financialClass(item.realized_pnl > 0 ? "positive" : item.realized_pnl < 0 ? "negative" : "neutral")}">${escapeHtml(item.realized_pnl ?? "--")}</td>
        <td>${escapeHtml(item.reason || "--")}</td>
      </tr>
    `)
    .join("");
}

function renderHealth(root, healthRisk) {
  const target = qs("[data-health-risk]", root);
  if (!target) {
    return;
  }
  const checks = healthRisk?.checks || [];
  if (!checks.length) {
    target.innerHTML = `<div class="dashboard-empty-block">${escapeHtml(panelStatusText(healthRisk?.state, healthRisk?.degradation_reason))}</div>`;
    return;
  }
  target.innerHTML = checks.slice(0, 4)
    .map((check) => `
      <div class="dashboard-health-row">
        <span>${escapeHtml(check.label)}</span>
        <strong>${escapeHtml(check.value)}</strong>
        <span class="dashboard-progress"><span style="width: ${Math.round((check.ratio || 0) * 100)}%"></span></span>
      </div>
    `)
    .join("");
}

function renderAlerts(root, alerts) {
  const target = qs("[data-alerts]", root);
  if (!target) {
    return;
  }
  const items = alerts?.items || [];
  if (!items.length) {
    target.innerHTML = `<div class="dashboard-empty-block">${escapeHtml(panelStatusText(alerts?.state, alerts?.degradation_reason))}</div>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <div class="dashboard-alert-row">
        <strong>${escapeHtml(localTime(item.timestamp))} / ${escapeHtml(item.severity)}</strong>
        <span>${escapeHtml(item.message)}</span>
      </div>
    `)
    .join("");
}

function renderAllocation(root, allocation) {
  const target = qs("[data-symbol-allocation]", root);
  if (!target) {
    return;
  }
  const items = allocation?.items || [];
  if (!items.length) {
    target.innerHTML = `<div class="dashboard-empty-block">${escapeHtml(panelStatusText(allocation?.state, allocation?.degradation_reason))}</div>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <div class="dashboard-allocation-row">
        <strong>${escapeHtml(item.symbol)}</strong>
        <span class="${financialClass(item.direction)}">${escapeHtml(item.pnl ?? "--")}</span>
        <span class="dashboard-progress"><span style="width: ${Math.round((item.bar_ratio || 0) * 100)}%"></span></span>
      </div>
    `)
    .join("");
}

function renderStrategyList(root, strategyList) {
  const totals = strategyList?.totals || {};
  const totalsTarget = qs("[data-strategy-totals]", root);
  if (totalsTarget) {
    const totalItems = [
      ["dashboard.strategy_list.total_running", totals.running ?? 0],
      ["dashboard.strategy_list.total_symbols", totals.symbols ?? 0],
      ["dashboard.strategy_list.total_strategies", totals.strategies ?? 0],
      ["dashboard.strategy_list.total_positions", totals.open_positions ?? "--"],
      ["dashboard.strategy_list.total_degraded", totals.degraded ?? 0],
    ];
    totalsTarget.innerHTML = totalItems
      .map(([labelKey, value]) => `
        <div class="dashboard-total">
          <span>${escapeHtml(t(labelKey))}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `)
      .join("");
  }

  const rowsTarget = qs("[data-strategy-rows]", root);
  if (!rowsTarget) {
    return;
  }
  const rows = strategyList?.items || [];
  if (!rows.length) {
    rowsTarget.innerHTML = `<div class="dashboard-empty-block">${escapeHtml(panelStatusText(strategyList?.state, strategyList?.degradation_reason))}</div>`;
    return;
  }
  rowsTarget.innerHTML = rows
    .map((row) => `
      <article class="dashboard-strategy-row" data-strategy-id="${escapeHtml(row.strategy_id)}">
        <span class="dashboard-row-dot ${rowStateClass(row.status)}" aria-hidden="true"></span>
        <span class="dashboard-row-main">
          <strong>${escapeHtml(row.name)}</strong>
          <span class="dashboard-row-meta">${escapeHtml(row.version || "--")}</span>
        </span>
        <span class="dashboard-row-meta">${escapeHtml(row.exchange || "--")} / ${escapeHtml((row.symbols || []).join(", ") || "--")}</span>
        <span class="${financialClass(row.pnl > 0 ? "positive" : row.pnl < 0 ? "negative" : "neutral")}">${escapeHtml(row.pnl ?? "--")}</span>
        ${drawSparkline(row.mini_sparkline)}
      </article>
    `)
    .join("");
}

function renderFooter(root, footer) {
  setText("[data-footer-system]", footer?.system_status || "--", root);
  setText("[data-footer-account]", footer?.account_tier || "--", root);
  setText("[data-footer-mode]", footer?.mode || "--", root);
  setText("[data-footer-api]", footer?.api_label || "--", root);
  setText("[data-footer-latency]", footer?.latency_ms ? `${footer.latency_ms} ms` : "--", root);
  setText("[data-footer-time]", localTime(footer?.server_time), root);
}

function renderFreshness(root, summary) {
  const retry = summary?.retry_after_seconds;
  const sourceCount = Array.isArray(summary?.sources) ? summary.sources.length : 0;
  const degradedCount = (summary?.sources || []).filter((source) => source.status !== "available").length;
  if (retry) {
    setText(
      "[data-dashboard-freshness]",
      t("dashboard.refresh.rate_limited", { seconds: retry }),
      root,
    );
    return;
  }
  setText(
    "[data-dashboard-freshness]",
    t("dashboard.refresh.freshness", {
      status: summary?.refresh_status || "unknown",
      sources: `${sourceCount - degradedCount}/${sourceCount}`,
      time: localTime(summary?.generated_at),
    }),
    root,
  );
}

function renderSummary(root, summary) {
  renderSelected(root, summary.selected_strategy_snapshot);
  renderChart(root, summary.equity_pnl_series);
  renderMetrics(root, summary.metric_grid);
  renderPositions(root, summary.open_positions);
  renderExecutions(root, summary.recent_executions);
  renderHealth(root, summary.health_risk);
  renderAlerts(root, summary.alerts);
  renderAllocation(root, summary.symbol_allocation);
  renderStrategyList(root, summary.strategy_list);
  renderFooter(root, summary.footer_status);
  renderFreshness(root, summary);
  setText(
    "[data-dashboard-refresh-status]",
    summary.refresh_status || t("refresh.idle"),
    root,
  );
}

function initDashboard(root) {
  const endpoint = root.dataset.summaryEndpoint || SUMMARY_ENDPOINT;
  const refreshButton = qs("[data-dashboard-refresh]", root);
  const loading = qs("[data-dashboard-loading]", root);
  let activeRequest = null;
  let poller = null;
  let delayedStart = null;
  let lastSummary = null;

  function setRunning(isRunning) {
    if (refreshButton instanceof HTMLButtonElement) {
      refreshButton.disabled = isRunning;
    }
    if (loading) {
      loading.hidden = !isRunning && Boolean(lastSummary);
    }
  }

  async function loadSummary(reason = "auto") {
    if (activeRequest) {
      return activeRequest;
    }
    setRunning(true);
    const separator = endpoint.includes("?") ? "&" : "?";
    activeRequest = apiFetch(`${endpoint}${separator}refresh=${encodeURIComponent(reason)}`)
      .then((summary) => {
        lastSummary = summary;
        root.dataset.dashboardLoaded = "true";
        root.dataset.dashboardLastRefresh = reason;
        renderSummary(root, summary);
        return summary;
      })
      .catch((error) => {
        setText("[data-dashboard-refresh-status]", error.code || "failed", root);
        setText("[data-dashboard-freshness]", error.message || t("dashboard.refresh.failed"), root);
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
    setText("[data-dashboard-refresh-current]", presetKey, root);
    qsa("[data-dashboard-refresh-preset]", root).forEach((item) => {
      item.setAttribute("aria-selected", item.dataset.dashboardRefreshPreset === presetKey ? "true" : "false");
    });
    if (intervalMs <= 0) {
      root.dataset.dashboardAutorefresh = "off";
      return;
    }
    root.dataset.dashboardAutorefresh = presetKey;
    poller = createPoller(() => loadSummary("auto"), {
      intervalMs,
      hiddenTabPause: true,
    });
    delayedStart = window.setTimeout(() => {
      delayedStart = null;
      poller?.start();
    }, intervalMs);
  }

  refreshButton?.addEventListener("click", () => {
    loadSummary("manual").catch(() => null);
  });
  qsa("[data-dashboard-refresh-preset]", root).forEach((item) => {
    item.addEventListener("click", () => {
      setAutorefresh(item.dataset.dashboardRefreshPreset || "off");
    });
  });

  window.__roehubDashboard = {
    loadSummary,
    setAutorefresh,
    get activeRequest() {
      return activeRequest;
    },
    get lastSummary() {
      return lastSummary;
    },
  };

  loadSummary("initial")
    .catch(() => null)
    .finally(() => setAutorefresh(DEFAULT_REFRESH_PRESET));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    const root = qs("[data-dashboard-root]");
    if (root) {
      initDashboard(root);
    }
  });
} else {
  const root = qs("[data-dashboard-root]");
  if (root) {
    initDashboard(root);
  }
}
