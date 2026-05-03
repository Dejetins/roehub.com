import { apiRequest } from "../core/api.js";
import { qs, ready } from "../core/dom.js";
import { formatDateTime } from "../core/formatters.js";
import { getCurrentLocale, translate } from "../core/locale.js";
import { createPoller } from "../core/poller.js";

const DEFAULT_ENDPOINT = "/api/ui/dashboard/summary";
const DEFAULT_POLL_INTERVAL_MS = 12000;
const EMPTY_VALUE = "--";
const SOURCE_TONES = Object.freeze({
  available: "success",
  degraded: "warning",
  unavailable: "neutral",
});

let lastSummary = null;

ready(() => {
  const root = qs("[data-dashboard]");
  if (!root) {
    return;
  }

  const endpoint = root.dataset.dashboardEndpoint || DEFAULT_ENDPOINT;
  const pollIntervalMs = Number(root.dataset.pollIntervalMs) || DEFAULT_POLL_INTERVAL_MS;
  const poller = createPoller(
    ({ signal }) => loadSummary({ root, endpoint, signal }),
    {
      intervalMs: pollIntervalMs,
      hiddenIntervalMs: pollIntervalMs,
      maxBackoffMs: 30000,
    },
  );

  document.addEventListener("roehub:locale-change", () => {
    if (lastSummary) {
      renderDashboard(root, lastSummary);
    }
  });

  poller.start({ immediate: true });
});

async function loadSummary({ root, endpoint, signal }) {
  setDashboardBusy(root, true);
  try {
    const summary = await apiRequest(endpoint, {
      method: "GET",
      redirectOnUnauthorized: true,
      signal,
      timeoutMs: 10000,
    });
    lastSummary = summary;
    renderDashboard(root, summary);
  } finally {
    setDashboardBusy(root, false);
  }
}

function renderDashboard(root, summary) {
  setText(root, "[data-dashboard-generated-at]", formatDate(summary.generated_at));
  setStatus(root, "dashboard.status.ready", "success");
  renderSource(root, "account", summary.account?.source);
  renderSource(root, "strategies", summary.strategies?.source);
  renderSource(root, "backtests", summary.backtests?.source);
  renderSource(root, "alerts", summary.alerts?.source);

  setText(root, '[data-dashboard-value="account.plan"]', summary.account?.paid_level);
  setText(root, '[data-dashboard-value="account.user_id"]', summary.account?.user_id);
  setText(root, '[data-dashboard-value="strategies.total"]', formatCount(summary.strategies?.total_count));
  setText(root, '[data-dashboard-value="strategies.active"]', formatCount(summary.strategies?.active_count));
  setText(root, '[data-dashboard-value="backtests.active"]', formatCount(summary.backtests?.active_count));

  renderStrategies(root, summary.strategies);
  renderBacktests(root, summary.backtests);
  renderAlerts(root, summary.alerts);
}

function renderSource(root, key, source) {
  const badge = qs(`[data-dashboard-source="${key}"]`, root);
  const panel = qs(`[data-dashboard-panel="${key}"]`, root);
  if (!badge || !source) {
    return;
  }

  const status = source.status || "unavailable";
  badge.textContent = translate(`dashboard.source.${status}`);
  badge.title = source.message || "";
  badge.className = `rh-status-badge rh-status-badge--${SOURCE_TONES[status] || "neutral"}`;
  if (panel) {
    panel.dataset.sourceStatus = status;
  }
}

function renderStrategies(root, panel) {
  const list = qs('[data-dashboard-list="strategies"]', root);
  if (!list) {
    return;
  }
  list.replaceChildren();
  if (!panel?.items?.length) {
    list.appendChild(emptyMessage(emptyKeyForSource(panel?.source, "dashboard.strategies.empty")));
    return;
  }
  panel.items.forEach((item) => {
    list.appendChild(
      listItem({
        title: item.name || item.strategy_id,
        meta: [item.instrument_key, item.timeframe, item.state].filter(Boolean).join(" / "),
        side: item.state || EMPTY_VALUE,
      }),
    );
  });
}

function renderBacktests(root, panel) {
  const list = qs('[data-dashboard-list="backtests"]', root);
  if (!list) {
    return;
  }
  list.replaceChildren();
  if (!panel?.items?.length) {
    list.appendChild(emptyMessage(emptyKeyForSource(panel?.source, "dashboard.backtests.empty")));
    return;
  }
  panel.items.forEach((item) => {
    list.appendChild(
      listItem({
        title: [item.symbol, item.state].filter(Boolean).join(" / ") || item.job_id,
        meta: [item.timeframe, item.risk_mode, item.primary_metric].filter(Boolean).join(" / "),
        side: `${item.progress_percent}%`,
      }),
    );
  });
}

function renderAlerts(root, panel) {
  const list = qs('[data-dashboard-list="alerts"]', root);
  if (!list) {
    return;
  }
  list.replaceChildren();
  if (!panel?.items?.length) {
    list.appendChild(emptyMessage(emptyKeyForSource(panel?.source, "dashboard.alerts.empty")));
    return;
  }
  panel.items.forEach((item) => {
    list.appendChild(
      listItem({
        title: item.title || item.alert_id,
        meta: [item.severity, formatDate(item.created_at)].filter(Boolean).join(" / "),
        side: item.severity || EMPTY_VALUE,
      }),
    );
  });
}

function listItem({ title, meta, side }) {
  const row = document.createElement("div");
  row.className = "rh-dashboard-list__item";

  const body = document.createElement("div");
  const titleNode = document.createElement("div");
  const metaNode = document.createElement("div");
  const sideNode = document.createElement("div");

  titleNode.className = "rh-dashboard-list__title";
  metaNode.className = "rh-dashboard-list__meta";
  sideNode.className = "rh-dashboard-progress";

  titleNode.textContent = title || EMPTY_VALUE;
  metaNode.textContent = meta || EMPTY_VALUE;
  sideNode.textContent = side || EMPTY_VALUE;

  body.append(titleNode, metaNode);
  row.append(body, sideNode);
  return row;
}

function emptyMessage(key) {
  const node = document.createElement("p");
  node.className = "rh-dashboard-list__empty";
  node.textContent = translate(key);
  return node;
}

function emptyKeyForSource(source, availableKey) {
  if (source?.status === "degraded") {
    return "dashboard.panel.degraded";
  }
  if (source?.status === "unavailable") {
    return "dashboard.panel.unavailable";
  }
  return availableKey;
}

function setStatus(root, key, tone) {
  const status = qs("[data-dashboard-status]", root);
  if (!status) {
    return;
  }
  status.textContent = translate(key);
  status.className = `rh-status-badge rh-status-badge--${tone}`;
}

function setDashboardBusy(root, busy) {
  root.setAttribute("aria-busy", String(busy));
  const loading = qs("[data-dashboard-loading]", root);
  if (loading) {
    loading.dataset.hidden = String(!busy);
  }
}

function setText(root, selector, value) {
  const node = qs(selector, root);
  if (node) {
    node.textContent = value || EMPTY_VALUE;
  }
}

function formatCount(value) {
  return typeof value === "number" ? String(value) : EMPTY_VALUE;
}

function formatDate(value) {
  if (!value) {
    return EMPTY_VALUE;
  }
  return formatDateTime(value, { locale: getCurrentLocale(), empty: EMPTY_VALUE });
}
