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

function pctOrUnavailable(value) {
  if (value === null || value === undefined || value === "") {
    return t("common.unavailable");
  }
  const numberValue = Number(value);
  if (Number.isNaN(numberValue)) {
    return value;
  }
  return `${(numberValue * 100).toFixed(2)}%`;
}

function durationText(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return "--";
  }
  const totalSeconds = Math.max(0, Math.round(Number(seconds)));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const remainderSeconds = totalSeconds % 60;
  if (minutes < 60) {
    return remainderSeconds ? `${minutes}m ${remainderSeconds}s` : `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainderMinutes = minutes % 60;
  return remainderMinutes ? `${hours}h ${remainderMinutes}m` : `${hours}h`;
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
  const actions = selected?.actions || {};
  const runState = selected?.run_state || null;
  const canRestart = ["starting", "warming_up", "running"].includes(runState);
  const canManual = ["starting", "warming_up", "running"].includes(runState);
  setText("[data-selected-name]", selected?.name || t("common.unavailable"), root);
  setText("[data-selected-version]", selected?.version || "--", root);
  setText("[data-selected-exchange]", selected?.exchange || t("common.unavailable"), root);
  setText("[data-selected-market]", selected?.market_type || "--", root);
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
  setButtonDisabled(root, "[data-strategy-run]", actions.can_run !== true);
  setButtonDisabled(root, "[data-strategy-stop]", actions.can_stop !== true);
  setButtonDisabled(root, "[data-strategy-restart]", !canRestart);
  setButtonDisabled(root, "[data-strategy-manual-entry]", !canManual);
  setButtonDisabled(root, "[data-strategy-manual-exit]", !canManual);
}

function setActionStatus(root, value) {
  setText("[data-strategy-action-status]", value || "--", root);
}

function actionErrorText(error) {
  const payload = error?.payload?.error || {};
  const details = payload.details || {};
  const reason = details.reason || payload.code || error?.code || "request_failed";
  return `${payload.message || error?.message || t("strategies.actions.failed")}: ${reason}`;
}

function setButtonDisabled(root, selector, disabled) {
  const button = qs(selector, root);
  if (button instanceof HTMLButtonElement) {
    button.disabled = disabled;
  }
}

function renderRuntimeStatus(root, runtimeStatus) {
  const gapStatus = runtimeStatus?.observed_latency_gap_status || "unavailable";
  const gapValue = runtimeStatus?.observed_latency_gap_seconds;
  setText("[data-runtime-environment]", runtimeStatus?.environment || "unknown", root);
  setText(
    "[data-runtime-producer]",
    `${runtimeStatus?.producer_status || "unknown"}: ${runtimeStatus?.producer_reason || "--"}`,
    root,
  );
  setText("[data-runtime-run]", runtimeStatus?.run_state || "--", root);
  setText("[data-runtime-started]", localTime(runtimeStatus?.run_started_at), root);
  setText("[data-runtime-updated]", localTime(runtimeStatus?.run_updated_at), root);
  setText("[data-runtime-checkpoint]", localTime(runtimeStatus?.checkpoint_ts_open), root);
  setText("[data-runtime-warmup]", runtimeStatus?.warmup_progress || "--", root);
  setText("[data-runtime-mainnet]", runtimeStatus?.mainnet_available ? "available" : t("common.unavailable"), root);
  setText("[data-runtime-latest-signal]", localTime(runtimeStatus?.latest_signal_at), root);
  setText("[data-runtime-latest-source]", localTime(runtimeStatus?.latest_source_event_at), root);
  setText("[data-runtime-latest-execution]", localTime(runtimeStatus?.latest_execution_update_at), root);
  setText(
    "[data-runtime-latency-gap]",
    gapStatus === "observed" ? durationText(gapValue) : "--",
    root,
  );
  const panel = qs(".strategies-runtime-status", root);
  if (panel instanceof HTMLElement) {
    panel.dataset.readiness = runtimeStatus?.producer_status || "unknown";
    panel.dataset.environment = runtimeStatus?.environment || "unknown";
  }
}

function renderLiveProfile(root, profile) {
  const readiness = profile?.readiness_status || "blocked";
  const reason = profile?.readiness_reason || "live_profile_unavailable";
  setText("[data-profile-readiness]", `${readiness}: ${reason}`, root);
  setText("[data-profile-mode]", profile?.mode || "monitor_only", root);
  setText("[data-profile-exchange]", profile?.exchange_connection_id || "--", root);
  setText(
    "[data-profile-sizing]",
    `${profile?.sizing_method || "fixed_quote"} / ${valueOrUnavailable(profile?.sizing_value)}`,
    root,
  );
  setText(
    "[data-profile-limits]",
    `${t("strategies.profile.max_orders")}: ${valueOrUnavailable(profile?.max_orders_per_run)} / ${t("strategies.profile.max_notional")}: ${valueOrUnavailable(profile?.max_notional_per_run)}`,
    root,
  );
  setText("[data-profile-reason]", reason, root);
  setText("[data-profile-updated]", localTime(profile?.updated_at), root);
  root.dataset.liveProfileMode = profile?.mode || "monitor_only";
  const panel = qs(".strategies-live-profile", root);
  if (panel instanceof HTMLElement) {
    panel.dataset.readiness = readiness;
  }
}

function manualIdempotencyKey(action, strategyId) {
  const randomPart = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  return `manual-${action}-${strategyId}-${randomPart}`;
}

function manualStatusText(payload) {
  const status = payload?.status || "unknown";
  const reason = payload?.outcome_reason || payload?.risk_reason || "--";
  return `${status}: ${reason}`;
}

function renderCompatibilityReadiness(root, readiness) {
  const compatibilityState = readiness?.compatibility_state || "not_launchable";
  const compatibilityReason = (readiness?.compatibility_reason_codes || [readiness?.launch_blocked_reason || "--"])[0];
  const marketDataState = readiness?.market_data_state || "pending";
  const marketDataReason = (readiness?.market_data_reason_codes || [readiness?.launch_blocked_reason || "--"])[0];
  setText("[data-profile-compatibility]", `${compatibilityState}: ${compatibilityReason}`, root);
  setText("[data-profile-market-data]", `${marketDataState}: ${marketDataReason}`, root);
}

function renderMarketReadiness(root, readiness) {
  const items = Array.isArray(readiness?.items) ? readiness.items : [];
  const firstBlocked = items.find((item) => item?.readiness_state !== "ready");
  const summaryState = firstBlocked?.readiness_state || (items.length ? "ready" : "pending");
  const summaryReason = firstBlocked?.reason_codes?.[0] || readiness?.degradation_reason || "btcusdt_market_ready";
  setText("[data-market-readiness]", `${summaryState}: ${summaryReason}`, root);
  setText("[data-market-symbol]", readiness?.symbol || "BTCUSDT", root);
  setText(
    "[data-market-threshold]",
    readiness?.freshness_threshold_seconds === null || readiness?.freshness_threshold_seconds === undefined
      ? "--"
      : `${readiness.freshness_threshold_seconds}s`,
    root,
  );
  setText("[data-market-checked]", localTime(readiness?.checked_at), root);
  ["binance:spot", "binance:futures", "bybit:spot", "bybit:futures"].forEach((marketKey) => {
    const row = items.find((item) => `${item.exchange_name}:${item.market_type}` === marketKey);
    const detail = row
      ? `${row.readiness_state}: ${row.reason_codes?.[0] || "--"} / ${row.stream_state} / min ${valueOrUnavailable(row.min_notional)}`
      : "missing: reference_market_missing";
    setText(`[data-market-row="${marketKey}"]`, detail, root);
  });
  const panel = qs(".strategies-market-readiness", root);
  if (panel instanceof HTMLElement) {
    panel.dataset.readiness = summaryState;
  }
}

function renderExchangeAccountReadiness(root, readiness) {
  const status = readiness?.status || "degraded";
  const reason = (readiness?.reason_codes || [readiness?.degradation_reason || "--"])[0];
  setText("[data-account-readiness]", `${status}: ${reason}`, root);
  setText("[data-account-connection]", readiness?.exchange_connection_id || "--", root);
  setText("[data-account-instrument]", readiness?.instrument_key || "--", root);
  setText(
    "[data-account-age]",
    readiness?.age_seconds === null || readiness?.age_seconds === undefined
      ? "--"
      : `${readiness.age_seconds}s`,
    root,
  );
  setText(
    "[data-account-config]",
    readiness?.config_guard_result_id
      ? `${readiness.ready_for_risk ? "verified" : status}`
      : "verify-only pending",
    root,
  );
  setText("[data-account-reason]", reason, root);
  setText("[data-account-checked]", localTime(readiness?.checked_at), root);
  const panel = qs(".strategies-account-readiness", root);
  if (panel instanceof HTMLElement) {
    panel.dataset.readiness = status;
  }
}

function renderPaperAccounting(root, accounting) {
  const completeness = accounting?.pnl_complete ? "complete" : "incomplete";
  const reason = accounting?.completeness_reason || "paper_accounting_unavailable";
  setText("[data-paper-completeness]", `${completeness}: ${reason}`, root);
  setText("[data-paper-reserved]", valueOrUnavailable(accounting?.reserved_budget), root);
  setText("[data-paper-position]", valueOrUnavailable(accounting?.position_quantity), root);
  setText("[data-paper-entry]", valueOrUnavailable(accounting?.average_entry_price), root);
  setText("[data-paper-equity]", valueOrUnavailable(accounting?.equity), root);
  setText(
    "[data-paper-pnl]",
    `${valueOrUnavailable(accounting?.realized_pnl)} / ${valueOrUnavailable(accounting?.unrealized_pnl)}`,
    root,
  );
  setText(
    "[data-paper-fees]",
    `${valueOrUnavailable(accounting?.fee_total)} / ${valueOrUnavailable(accounting?.funding_total)}`,
    root,
  );
  setText(
    "[data-paper-models]",
    `${valueOrUnavailable(accounting?.fee_model)} / ${valueOrUnavailable(accounting?.funding_model)}`,
    root,
  );
  setText("[data-paper-updated]", localTime(accounting?.updated_at), root);
  const panel = qs(".strategies-paper-accounting", root);
  if (panel instanceof HTMLElement) {
    panel.dataset.readiness = accounting?.pnl_complete ? "ready" : "degraded";
  }
}

function renderMetrics(root, metricGrid) {
  const target = qs("[data-strategy-metrics]", root);
  if (!target) {
    return;
  }
  const items = metricGrid?.items || [];
  if (!items.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="3">${escapeHtml(panelStatusText(metricGrid?.state, metricGrid?.degradation_reason, metricGrid?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = items
    .map((metric) => {
      const label = t(metricLabelKeys[metric.key] || metric.label || metric.key);
      return `
        <tr>
          <td>${escapeHtml(label)}</td>
          <td class="${financialClass(metric.direction)}">${escapeHtml(valueOrUnavailable(metric.formatted))}</td>
          <td class="strategies-source-cell">${escapeHtml(metric.source)} / ${escapeHtml(metric.status)}</td>
        </tr>
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
      <tr
        data-saved-strategy-id="${escapeHtml(row.strategy_id)}"
        tabindex="0"
        aria-selected="${row.strategy_id === selectedId ? "true" : "false"}"
      >
        <td>${escapeHtml(row.name)}</td>
        <td>${escapeHtml(row.version || "--")}</td>
        <td>${escapeHtml(localTime(row.latest_activity))}</td>
      </tr>
    `)
    .join("");
}

function syncTabGroup(root, selector, activeValue, dataName) {
  qsa(selector, root).forEach((button) => {
    const isActive = button.dataset[dataName] === activeValue;
    button.classList.toggle("strategies-tab--active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

function syncStatWorkspace(root, activeMode) {
  syncTabGroup(root, "[data-stat-mode]", activeMode, "statMode");
  qsa("[data-stat-view]", root).forEach((panel) => {
    if (panel instanceof HTMLElement) {
      panel.hidden = panel.dataset.statView !== activeMode;
    }
  });
}

function renderChart(root, summary, state) {
  const svg = root.querySelector("[data-strategy-chart]");
  const chart = summary?.chart || {};
  const chartMode = state.chartMode || "trades";
  if (svg instanceof SVGElement) {
    drawPlaceholderChart(svg, { negative: chartMode === "drawdown" });
  }
  syncTabGroup(root, "[data-chart-mode]", chartMode, "chartMode");
  setText("[data-chart-symbol]", chart?.symbol || "--", root);
  const sourcePanel = chartMode === "equity"
    ? summary?.equity_curve
    : chartMode === "drawdown"
      ? summary?.drawdown
      : chart;
  setText(
    "[data-chart-state]",
    panelStatusText(sourcePanel?.state, sourcePanel?.degradation_reason, sourcePanel?.source),
    root,
  );
  const legend = qs("[data-chart-legend]", root);
  if (legend instanceof HTMLElement) {
    legend.hidden = chartMode !== "trades";
  }
}

function renderMonthly(root, monthly) {
  const head = qs("[data-monthly-head]", root);
  const body = qs("[data-monthly-rows]", root);
  if (!head || !body) {
    return;
  }
  const columns = monthly?.columns || [];
  head.innerHTML = `
    <tr>
      <th>${escapeHtml(t("strategies.monthly.period"))}</th>
      <th>${escapeHtml(t("strategies.monthly.value"))}</th>
    </tr>
  `;
  if (!monthly?.rows?.length) {
    body.innerHTML = `
      <tr><td class="strategies-empty-row" colspan="2">
        ${escapeHtml(panelStatusText(monthly?.state, monthly?.degradation_reason, monthly?.source))}
      </td></tr>
    `;
    return;
  }
  const valueColumns = columns.filter((column) => column !== "year");
  body.innerHTML = monthly.rows
    .flatMap((row) => valueColumns.map((column) => {
      const year = row.year ? `${row.year} ` : "";
      return `
        <tr>
          <td>${escapeHtml(`${year}${column}`)}</td>
          <td class="${financialClass(directionFromNumber(row[column]))}">${escapeHtml(valueOrUnavailable(row[column]))}</td>
        </tr>
      `;
    }))
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
  const rows = panel?.rows || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="3">${escapeHtml(panelStatusText(panel?.state, panel?.degradation_reason, panel?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(t(breakdownLabelKeys[row.key] || row.label || row.key))}</td>
        <td class="${financialClass(row.direction)}">${escapeHtml(valueOrUnavailable(row.total_value))}</td>
        <td class="strategies-source-cell">${escapeHtml(panel?.source || "--")} / ${escapeHtml(panel?.state || "--")}</td>
      </tr>
    `)
    .join("");
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

function renderSignalJournal(root, signalJournal) {
  const target = qs("[data-signal-journal-rows]", root);
  if (!target) {
    return;
  }
  setText(
    "[data-signal-journal-state]",
    signalJournal?.state || t("strategies.panel.unavailable"),
    root,
  );
  const items = signalJournal?.items || [];
  if (!items.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="8">${escapeHtml(panelStatusText(signalJournal?.state, signalJournal?.degradation_reason, signalJournal?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <tr data-signal-id="${escapeHtml(item.signal_id)}">
        <td>${escapeHtml(localTime(item.created_at || item.bar_ts_close))}</td>
        <td>${escapeHtml(item.mode)}</td>
        <td>${escapeHtml(item.outcome)}</td>
        <td>${escapeHtml(item.side ? `${item.signal_action}/${item.side}` : item.signal_action)}</td>
        <td>${escapeHtml(valueOrUnavailable(item.reference_price))}</td>
        <td>${escapeHtml(`${localTime(item.bar_ts_open)} -> ${localTime(item.bar_ts_close)}`)}</td>
        <td>${escapeHtml(item.reason_code)}</td>
        <td>${escapeHtml(item.source_message_id)}</td>
      </tr>
    `)
    .join("");
}

function renderExecutionOutcomes(root, executionOutcomes) {
  const target = qs("[data-execution-outcome-rows]", root);
  if (!target) {
    return;
  }
  setText(
    "[data-execution-outcomes-state]",
    executionOutcomes?.state || t("strategies.panel.unavailable"),
    root,
  );
  const items = executionOutcomes?.items || [];
  if (!items.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="8">${escapeHtml(panelStatusText(executionOutcomes?.state, executionOutcomes?.degradation_reason, executionOutcomes?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <tr data-source-event-id="${escapeHtml(item.source_event_id)}">
        <td>${escapeHtml(item.strategy_signal_id || item.source_event_ref)}</td>
        <td>${escapeHtml(`${item.source_type}: ${item.outcome} / ${item.outcome_reason}`)}<br><span class="strategies-source-cell">${escapeHtml(localTime(item.source_event_received_at))}</span></td>
        <td>${escapeHtml(item.intent_id ? `${item.intent_status || "--"} / ${item.risk_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.order_status ? `${item.order_status} / ${item.order_status_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.fill_count === null || item.fill_count === undefined ? "--" : `${item.fill_count} / ${localTime(item.latest_fill_at)}`)}</td>
        <td>${escapeHtml(item.reconciliation_status ? `${item.reconciliation_status} / ${item.reconciliation_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.latency_gap_status === "observed" ? durationText(item.latency_gap_seconds) : "--")}</td>
        <td>${escapeHtml(item.notification_event_type ? `${item.notification_event_type} / ${item.notification_reason || "--"}` : "--")}</td>
      </tr>
    `)
    .join("");
}

function renderOutcomeRows(root, selector, outcomes) {
  const target = qs(selector, root);
  if (!target) {
    return;
  }
  const items = outcomes?.items || [];
  if (!items.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="8">${escapeHtml(panelStatusText(outcomes?.state, outcomes?.degradation_reason, outcomes?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = items
    .map((item) => `
      <tr data-source-event-id="${escapeHtml(item.source_event_id)}">
        <td>${escapeHtml(item.strategy_signal_id || item.source_event_ref)}</td>
        <td>${escapeHtml(`${item.source_type}: ${item.outcome} / ${item.outcome_reason}`)}<br><span class="strategies-source-cell">${escapeHtml(localTime(item.source_event_received_at))}</span></td>
        <td>${escapeHtml(item.intent_id ? `${item.intent_status || "--"} / ${item.risk_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.order_status ? `${item.order_status} / ${item.order_status_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.fill_count === null || item.fill_count === undefined ? "--" : `${item.fill_count} / ${localTime(item.latest_fill_at)}`)}</td>
        <td>${escapeHtml(item.reconciliation_status ? `${item.reconciliation_status} / ${item.reconciliation_reason || "--"}` : "--")}</td>
        <td>${escapeHtml(item.latency_gap_status === "observed" ? durationText(item.latency_gap_seconds) : "--")}</td>
        <td>${escapeHtml(item.notification_event_type ? `${item.notification_event_type} / ${item.notification_reason || "--"}` : "--")}</td>
      </tr>
    `)
    .join("");
}

function renderRlMlTab(root, rlMl) {
  const model = rlMl?.model_status || {};
  const slots = rlMl?.ticker_slots || {};
  const modes = rlMl?.modes || {};
  const risk = rlMl?.risk_config || {};
  const operator = rlMl?.operator_controls || {};
  setText("[data-rl-state]", panelStatusText(rlMl?.state, rlMl?.degradation_reason, rlMl?.source), root);
  setText("[data-rl-model-state]", panelStatusText(model.state, model.degradation_reason, model.source), root);
  setText("[data-rl-model-family]", model.model_family || "--", root);
  setText("[data-rl-model-champion]", model.champion_model_id || t("common.unavailable"), root);
  setText("[data-rl-model-registry]", model.registry_status || "--", root);
  setText("[data-rl-model-activation]", model.activation_status || "--", root);
  setText("[data-rl-model-calibration]", model.calibration_pack_id || t("common.unavailable"), root);
  setText("[data-rl-model-artifact-root]", model.artifact_root || "--", root);
  setText("[data-rl-mode-state]", panelStatusText(modes.state, modes.degradation_reason, modes.source), root);
  setText("[data-rl-active-mode]", modes.active_mode || "monitor_only", root);
  setText(
    "[data-rl-mode-options]",
    (modes.options || [])
      .map((option) => `${option.mode}:${option.enabled ? "enabled" : "blocked"}`)
      .join(" / ") || "--",
    root,
  );
  setText("[data-rl-mode-reason]", modes.degradation_reason || "--", root);
  setText("[data-rl-risk-state]", panelStatusText(risk.state, risk.degradation_reason, risk.source), root);
  setText("[data-rl-risk-sizing]", risk.sizing_policy || "--", root);
  setText("[data-rl-risk-gate]", `${risk.risk_gate_status || "--"} / ${risk.policy_status || "--"}`, root);
  setText("[data-rl-risk-base-size]", valueOrUnavailable(risk.base_quote_notional), root);
  setText(
    "[data-rl-risk-max-notional]",
    `${valueOrUnavailable(risk.max_position_notional)} / ${valueOrUnavailable(risk.max_exposure_notional)} / ${valueOrUnavailable(risk.max_turnover_notional)}`,
    root,
  );
  setText(
    "[data-rl-risk-loss-drawdown]",
    `${valueOrUnavailable(risk.max_daily_loss_notional)} / ${pctOrUnavailable(risk.max_drawdown_pct)}`,
    root,
  );
  setText(
    "[data-rl-risk-confidence]",
    `${pctOrUnavailable(risk.min_confidence)} / ${pctOrUnavailable(risk.min_expected_pnl_pct)}`,
    root,
  );
  setText("[data-rl-risk-synthetic-exits]", renderSyntheticExitText(risk.synthetic_exit_rules || []), root);
  setText("[data-rl-risk-reasons]", (risk.validation_reasons || []).join(" / ") || "--", root);
  setText("[data-rl-risk-notes]", (risk.notes || []).join(" / ") || "--", root);
  setText("[data-rl-operator-state]", panelStatusText(operator.state, operator.degradation_reason, operator.source), root);
  setText("[data-rl-operator-reason]", operator.degradation_reason || "--", root);
  renderRlOperatorControls(root, operator);
  renderRlTickerSlots(root, slots);
  setText(
    "[data-rl-outcomes-state]",
    rlMl?.source_event_outcomes?.state || t("strategies.panel.unavailable"),
    root,
  );
  renderOutcomeRows(root, "[data-rl-outcome-rows]", rlMl?.source_event_outcomes);
}

function renderSyntheticExitText(rules) {
  if (!rules.length) {
    return "--";
  }
  return rules
    .map((rule) => `${rule.rule_type}:${pctOrUnavailable(rule.trigger_pct)}:${rule.creates_intent_action}`)
    .join(" / ");
}

function renderRlOperatorControls(root, operator) {
  const target = qs("[data-rl-operator-controls]", root);
  if (!target) {
    return;
  }
  const controls = operator?.controls || [];
  if (!controls.length) {
    target.innerHTML = `<span class="strategies-empty-block">${escapeHtml(panelStatusText(operator?.state, operator?.degradation_reason, operator?.source))}</span>`;
    return;
  }
  target.innerHTML = controls
    .filter((control) => control.visible !== false)
    .map((control) => `
      <button
        class="rh-button rh-button--secondary rh-button--compact"
        type="button"
        data-rl-operator-action="${escapeHtml(control.action)}"
        title="${escapeHtml(control.blocked_reason || "")}"
        ${control.enabled ? "" : "disabled"}
      >
        ${escapeHtml(control.label || control.action)}
      </button>
    `)
    .join("");
}

function renderRlTickerSlots(root, slots) {
  setText("[data-rl-slots-state]", panelStatusText(slots?.state, slots?.degradation_reason, slots?.source), root);
  setText("[data-rl-paid-level]", slots?.paid_level || "--", root);
  setText("[data-rl-product-label]", slots?.product_label || "--", root);
  setText(
    "[data-rl-live-slots]",
    `${valueOrUnavailable(slots?.live_slots_used)} / ${valueOrUnavailable(slots?.live_slots_allowed)}`,
    root,
  );
  setText("[data-rl-slots-reason]", slots?.degradation_reason || "--", root);
  const target = qs("[data-rl-slot-rows]", root);
  if (!target) {
    return;
  }
  const rows = slots?.items || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td class="strategies-empty-row" colspan="6">${escapeHtml(panelStatusText(slots?.state, slots?.degradation_reason, slots?.source))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => `
      <tr>
        <td>${escapeHtml(row.symbol)}</td>
        <td>${escapeHtml(row.exchange_name)}</td>
        <td>${escapeHtml(row.market_type)}</td>
        <td>${escapeHtml(row.mode)}</td>
        <td>${escapeHtml(row.slot_state)}</td>
        <td>${escapeHtml(row.readiness_reason)}</td>
      </tr>
    `)
    .join("");
}

function syncStrategiesMode(root, activeMode) {
  qsa("[data-strategies-mode]", root).forEach((button) => {
    const isActive = button.dataset.strategiesMode === activeMode;
    button.classList.toggle("strategies-tab--active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  qsa("[data-strategies-mode-panel]", root).forEach((panel) => {
    if (panel instanceof HTMLElement) {
      panel.hidden = panel.dataset.strategiesModePanel !== activeMode;
    }
  });
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
  renderRuntimeStatus(root, summary.runtime_status);
  renderLiveProfile(root, summary.live_profile);
  renderCompatibilityReadiness(root, summary.compatibility_readiness);
  renderMarketReadiness(root, summary.market_readiness);
  renderExchangeAccountReadiness(root, summary.exchange_account_readiness);
  renderPaperAccounting(root, summary.paper_accounting);
  renderSelector(root, summary.strategy_selector, state.savedQuery);
  renderChart(root, summary, state);
  syncStatWorkspace(root, state.statMode || "overall");
  renderMetrics(root, summary.metric_grid);
  renderMonthly(root, summary.monthly_stats);
  renderLongShort(root, summary.long_short);
  renderRisk(root, summary.risk_execution);
  renderHours(root, summary.hourly_results);
  renderSignalJournal(root, summary.signal_journal);
  renderExecutionOutcomes(root, summary.execution_outcomes);
  renderRlMlTab(root, summary.rl_ml);
  renderTrades(root, summary.trades);
  syncStrategiesMode(root, state.activeMode || "classic");
  renderFooter(summary);
  renderFreshness(summary);
  setText("[data-strategies-refresh-status]", summary.refresh_status || t("refresh.idle"), document);
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
    savedQuery: "",
    chartMode: "trades",
    statMode: "overall",
    activeMode: "classic",
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

  function selectSavedStrategy(strategyId) {
    if (!strategyId) {
      return;
    }
    state.selectedStrategyId = strategyId;
    setBrowserStrategyState(strategyId);
    loadDashboard("initial").catch(() => null);
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
      selectSavedStrategy(strategyOption.dataset.selectStrategy || "");
      return;
    }
    const savedRow = event.target.closest("[data-saved-strategy-id]");
    if (savedRow instanceof HTMLElement) {
      selectSavedStrategy(savedRow.dataset.savedStrategyId || "");
      return;
    }
    const stateOption = event.target.closest("[data-strategy-state]");
    if (stateOption instanceof HTMLElement) {
      state.selectorState = stateOption.dataset.strategyState === "active" ? "active" : "all";
      setText("[data-state-current]", state.selectorState, root);
      loadDashboard("initial").catch(() => null);
      return;
    }
    const chartMode = event.target.closest("[data-chart-mode]");
    if (chartMode instanceof HTMLElement) {
      state.chartMode = chartMode.dataset.chartMode || "trades";
      if (lastSummary) {
        renderChart(root, lastSummary, state);
      }
      return;
    }
    const statMode = event.target.closest("[data-stat-mode]");
    if (statMode instanceof HTMLElement) {
      state.statMode = statMode.dataset.statMode || "overall";
      syncStatWorkspace(root, state.statMode);
      return;
    }
    const strategiesMode = event.target.closest("[data-strategies-mode]");
    if (strategiesMode instanceof HTMLElement) {
      state.activeMode = strategiesMode.dataset.strategiesMode || "classic";
      syncStrategiesMode(root, state.activeMode);
      return;
    }
  });

  root.addEventListener("keydown", (event) => {
    if (!(event.target instanceof HTMLElement)) {
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const savedRow = event.target.closest("[data-saved-strategy-id]");
    if (savedRow instanceof HTMLElement) {
      event.preventDefault();
      selectSavedStrategy(savedRow.dataset.savedStrategyId || "");
    }
  });

  qs("[data-strategy-run]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    const path = root.dataset.apiRunPathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    setActionStatus(root, t("strategies.actions.run"));
    apiFetch(path, { method: "POST" })
      .then(() => {
        setActionStatus(root, t("strategies.status.ready"));
        return loadDashboard("initial");
      })
      .catch((error) => setActionStatus(root, actionErrorText(error)));
  });
  qs("[data-strategy-stop]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    const path = root.dataset.apiStopPathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    setActionStatus(root, t("strategies.actions.stop"));
    apiFetch(path, { method: "POST" })
      .then(() => {
        setActionStatus(root, t("strategies.status.ready"));
        return loadDashboard("initial");
      })
      .catch((error) => setActionStatus(root, actionErrorText(error)));
  });
  qs("[data-strategy-restart]", root)?.addEventListener("click", () => {
    if (!state.selectedStrategyId) {
      return;
    }
    const path = root.dataset.apiRestartPathTemplate.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    setActionStatus(root, t("strategies.actions.restart"));
    apiFetch(path, { method: "POST" })
      .then(() => {
        setActionStatus(root, t("strategies.status.ready"));
        return loadDashboard("initial");
      })
      .catch((error) => setActionStatus(root, actionErrorText(error)));
  });

  function executeManualAction(action) {
    if (!state.selectedStrategyId) {
      return;
    }
    const template = action === "entry"
      ? root.dataset.apiManualEntryPathTemplate
      : root.dataset.apiManualExitPathTemplate;
    const path = template.replace("{strategy_id}", encodeURIComponent(state.selectedStrategyId));
    const selector = action === "entry"
      ? "[data-strategy-manual-entry]"
      : "[data-strategy-manual-exit]";
    const idempotencyKey = manualIdempotencyKey(action, state.selectedStrategyId);
    setButtonDisabled(root, selector, true);
    setActionStatus(
      root,
      action === "entry"
        ? t("strategies.actions.manual_entry_pending")
        : t("strategies.actions.manual_exit_pending"),
    );
    apiFetch(path, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify({ client_request_id: idempotencyKey }),
    })
      .then((payload) => {
        setActionStatus(root, manualStatusText(payload));
        return loadDashboard("initial");
      })
      .catch((error) => setActionStatus(root, actionErrorText(error)))
      .finally(() => {
        if (lastSummary) {
          renderSelected(root, lastSummary.selected_strategy);
        }
      });
  }

  qs("[data-strategy-manual-entry]", root)?.addEventListener("click", () => {
    executeManualAction("entry");
  });
  qs("[data-strategy-manual-exit]", root)?.addEventListener("click", () => {
    executeManualAction("exit");
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
        return loadDashboard("initial");
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
        return loadDashboard("initial");
      })
      .catch(() => null);
  });

  window.__roehubStrategies = {
    loadDashboard,
    setAutorefresh,
    get activeRequest() {
      return activeRequest;
    },
    get lastSummary() {
      return lastSummary;
    },
  };

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
