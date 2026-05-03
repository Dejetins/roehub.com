import { apiRequest } from "../core/api.js";
import { delegate, qs, qsa, ready, setBusy } from "../core/dom.js";
import { financialClassName, formatDateTime } from "../core/formatters.js";
import { getCurrentLocale, translate } from "../core/locale.js";
import { notify } from "../core/notifications.js";
import { createPoller } from "../core/poller.js";
import { createEventStream } from "../core/sse.js";
import { drawTimeSeries } from "../charts/timeseries.js";

const DEFAULT_MONITOR_ENDPOINT = "/api/ui/strategies/monitor";
const DEFAULT_POLL_INTERVAL_MS = 10000;
const ACTIVE_STATES = Object.freeze(["starting", "warming_up", "running", "stopping"]);
const SOURCE_TONES = Object.freeze({
  available: "success",
  degraded: "warning",
  unavailable: "neutral",
});

ready(() => {
  const root = qs("[data-monitoring-page]");
  if (!root) {
    return;
  }
  initMonitoring(root);
});

function initMonitoring(root) {
  const state = {
    monitor: null,
    snapshot: null,
    positions: null,
    fills: null,
    equity: null,
    selectedStrategyId: null,
    filterState: "all",
    events: [],
    stream: null,
    lastEventId: "",
    snapshotInFlight: false,
    mutationInFlight: false,
  };
  const monitorEndpoint = root.dataset.monitorEndpoint || DEFAULT_MONITOR_ENDPOINT;
  const pollIntervalMs = Number(root.dataset.pollIntervalMs) || DEFAULT_POLL_INTERVAL_MS;
  const poller = createPoller(
    ({ signal }) => refreshMonitoring({ root, state, monitorEndpoint, signal }),
    {
      intervalMs: pollIntervalMs,
      hiddenIntervalMs: pollIntervalMs,
      maxBackoffMs: 30000,
    },
  );

  delegate(root, "click", "[data-monitoring-state-filter]", (event, element) => {
    event.preventDefault();
    state.filterState = element.getAttribute("data-monitoring-state-filter") || "all";
    qsa("[data-monitoring-state-filter]", root).forEach((button) => {
      const active = button === element;
      button.classList.toggle("rh-segmented__button--active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    void refreshMonitoring({ root, state, monitorEndpoint });
  });

  delegate(root, "click", "[data-monitoring-strategy-id]", (event, element) => {
    event.preventDefault();
    const strategyId = element.getAttribute("data-monitoring-strategy-id");
    if (!strategyId || strategyId === state.selectedStrategyId) {
      return;
    }
    state.selectedStrategyId = strategyId;
    renderMonitor(root, state);
    openStream({ root, state });
    void loadSelectedDetails({ root, state });
  });

  delegate(root, "click", "[data-monitoring-action]", async (event, element) => {
    event.preventDefault();
    await runControlAction({ root, state, element });
  });

  delegate(root, "click", "[data-monitoring-mobile-tab]", (event, element) => {
    event.preventDefault();
    setMobileTab(root, element.getAttribute("data-monitoring-mobile-tab") || "detail");
  });

  document.addEventListener("roehub:locale-change", () => {
    renderAll(root, state);
  });
  document.addEventListener("roehub:theme-change", () => {
    renderEquity(root, state.equity);
  });
  window.addEventListener("beforeunload", () => closeStream(state));

  setMobileTab(root, "detail");
  poller.start({ immediate: true });
}

async function refreshMonitoring({ root, state, monitorEndpoint, signal }) {
  clearError(root);
  const requestUrl = new URL(monitorEndpoint, window.location.origin);
  requestUrl.searchParams.set("state", state.filterState);
  const payload = await apiRequest(`${requestUrl.pathname}${requestUrl.search}`, {
    method: "GET",
    redirectOnUnauthorized: true,
    signal,
    timeoutMs: 10000,
  });
  state.monitor = payload;
  if (!state.selectedStrategyId && payload.selected_strategy_id) {
    state.selectedStrategyId = payload.selected_strategy_id;
    openStream({ root, state });
  }
  renderMonitor(root, state);
  if (state.selectedStrategyId) {
    await loadSelectedDetails({ root, state, signal });
  }
}

async function loadSelectedDetails({ root, state, signal }) {
  if (!state.selectedStrategyId || state.snapshotInFlight) {
    return;
  }
  state.snapshotInFlight = true;
  try {
    const strategyId = state.selectedStrategyId;
    const [snapshot, positions, fills, equity] = await Promise.all([
      apiRequest(renderPath(root.dataset.snapshotPathTemplate, strategyId), {
        redirectOnUnauthorized: true,
        signal,
        timeoutMs: 10000,
      }),
      apiRequest(renderPath(root.dataset.positionsPathTemplate, strategyId), {
        redirectOnUnauthorized: true,
        signal,
        timeoutMs: 10000,
      }),
      apiRequest(renderPath(root.dataset.fillsPathTemplate, strategyId), {
        redirectOnUnauthorized: true,
        signal,
        timeoutMs: 10000,
      }),
      apiRequest(renderPath(root.dataset.equityPathTemplate, strategyId), {
        redirectOnUnauthorized: true,
        signal,
        timeoutMs: 10000,
      }),
    ]);
    if (state.selectedStrategyId !== strategyId) {
      return;
    }
    state.snapshot = snapshot;
    state.positions = positions;
    state.fills = fills;
    state.equity = equity;
    renderDetails(root, state);
  } catch (error) {
    showError(root, error);
  } finally {
    state.snapshotInFlight = false;
  }
}

async function runControlAction({ root, state, element }) {
  if (!state.selectedStrategyId || state.mutationInFlight) {
    return;
  }
  const action = element.getAttribute("data-monitoring-action");
  const template = action === "run" ? root.dataset.runPathTemplate : root.dataset.stopPathTemplate;
  if (!template) {
    return;
  }
  state.mutationInFlight = true;
  setBusy(element, true);
  clearError(root);
  try {
    const payload = await apiRequest(renderPath(template, state.selectedStrategyId), {
      method: "POST",
      redirectOnUnauthorized: true,
      timeoutMs: 10000,
    });
    notify(translate(action === "run" ? "monitoring.notify.run" : "monitoring.notify.stop"), {
      tone: "info",
    });
    appendEvent(state, {
      title: payload.state || action,
      meta: formatDate(payload.updated_at || payload.started_at),
    });
    await refreshMonitoring({
      root,
      state,
      monitorEndpoint: root.dataset.monitorEndpoint || DEFAULT_MONITOR_ENDPOINT,
    });
  } catch (error) {
    showError(root, error);
  } finally {
    state.mutationInFlight = false;
    setBusy(element, false);
  }
}

function openStream({ root, state }) {
  closeStream(state);
  if (!state.selectedStrategyId || !root.dataset.streamPath) {
    return;
  }
  const streamUrl = new URL(root.dataset.streamPath, window.location.origin);
  streamUrl.searchParams.set("strategy_id", state.selectedStrategyId);
  if (state.lastEventId) {
    streamUrl.searchParams.set("last_event_id", state.lastEventId);
  }
  setConnection(root, "monitoring.status.connecting", "warning");
  state.stream = createEventStream(`${streamUrl.pathname}${streamUrl.search}`, {
    withCredentials: true,
    onOpen: () => setConnection(root, "monitoring.status.live", "success"),
    onError: () => {
      setConnection(root, "monitoring.status.polling", "warning");
      closeStream(state);
      void refreshMonitoring({
        root,
        state,
        monitorEndpoint: root.dataset.monitorEndpoint || DEFAULT_MONITOR_ENDPOINT,
      });
    },
    onDowngrade: () => setConnection(root, "monitoring.status.polling", "warning"),
    events: {
      status: () => setConnection(root, "monitoring.status.live", "success"),
      fallback: () => {
        setConnection(root, "monitoring.status.polling", "warning");
        closeStream(state);
      },
      "strategy.metric": (event) => handleStreamEvent({ root, state, event }),
      "strategy.event": (event) => handleStreamEvent({ root, state, event }),
    },
  });
}

function closeStream(state) {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

function handleStreamEvent({ root, state, event }) {
  state.lastEventId = event.lastEventId || state.lastEventId;
  let payload = null;
  try {
    payload = JSON.parse(event.data);
  } catch (_error) {
    return;
  }
  const fields = payload?.payload || {};
  appendEvent(state, {
    title: fields.event_type || fields.metric_type || payload.kind || "event",
    meta: [fields.value, fields.ts].filter(Boolean).join(" / "),
  });
  renderEvents(root, state.events);
  void loadSelectedDetails({ root, state });
}

function renderAll(root, state) {
  renderMonitor(root, state);
  renderDetails(root, state);
}

function renderMonitor(root, state) {
  const monitor = state.monitor;
  setText(root, "[data-monitoring-generated-at]", formatDate(monitor?.generated_at));
  renderSource(root, "monitor", monitor?.source);
  const list = qs("[data-monitoring-strategy-list]", root);
  if (!list) {
    return;
  }
  list.replaceChildren();
  const items = Array.isArray(monitor?.items) ? monitor.items : [];
  if (items.length === 0) {
    list.appendChild(messageNode(translate("monitoring.state.empty")));
    return;
  }
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rh-monitoring-strategy";
    button.dataset.monitoringStrategyId = item.strategy_id;
    button.dataset.state = item.state || "idle";
    if (item.strategy_id === state.selectedStrategyId) {
      button.setAttribute("aria-current", "true");
    }

    const dot = document.createElement("span");
    dot.className = "rh-monitoring-strategy__status";
    dot.setAttribute("aria-hidden", "true");

    const body = document.createElement("span");
    const name = document.createElement("span");
    const meta = document.createElement("span");
    name.className = "rh-monitoring-strategy__name";
    meta.className = "rh-monitoring-strategy__meta";
    name.textContent = item.name || item.strategy_id;
    meta.textContent = [item.instrument_key, item.timeframe, formatLag(item.lag_seconds)]
      .filter(Boolean)
      .join(" / ");
    body.append(name, meta);

    const side = document.createElement("span");
    side.className = "rh-monitoring-strategy__state";
    side.textContent = item.state || "idle";
    button.append(dot, body, side);
    list.appendChild(button);
  });
}

function renderDetails(root, state) {
  const snapshot = state.snapshot;
  setText(root, "[data-monitoring-strategy-name]", snapshot?.name || translate("monitoring.detail.empty_title"));
  setText(
    root,
    "[data-monitoring-strategy-meta]",
    snapshot
      ? [snapshot.spec?.instrument_key, snapshot.spec?.timeframe].filter(Boolean).join(" / ")
      : translate("monitoring.detail.empty_body"),
  );
  setText(root, '[data-monitoring-value="state"]', snapshot?.run?.state || "--");
  setText(root, '[data-monitoring-value="lag_seconds"]', metricValue(snapshot, "lag_seconds"));
  setText(
    root,
    '[data-monitoring-value="checkpoint_ts_open"]',
    metricValue(snapshot, "checkpoint_ts_open"),
  );
  setText(root, '[data-monitoring-value="updated_at"]', formatDate(snapshot?.run?.updated_at));
  setActionState(root, snapshot?.run?.state || "idle");
  renderPositions(root, state.positions);
  renderFills(root, state.fills);
  renderEquity(root, state.equity);
  renderEvents(root, state.events);
}

function renderPositions(root, positions) {
  renderSource(root, "positions", positions?.source);
  renderRows({
    root,
    selector: '[data-monitoring-list="positions"]',
    items: positions?.items,
    emptyKey: "monitoring.positions.empty",
    formatter: (item) => ({
      title: [item.symbol, item.side].filter(Boolean).join(" / "),
      meta: [item.quantity, item.entry_price].filter(Boolean).join(" / "),
      value: item.unrealized_pnl,
    }),
  });
}

function renderFills(root, fills) {
  renderSource(root, "fills", fills?.source);
  renderRows({
    root,
    selector: '[data-monitoring-list="fills"]',
    items: fills?.items,
    emptyKey: "monitoring.fills.empty",
    formatter: (item) => ({
      title: [item.symbol, item.side].filter(Boolean).join(" / "),
      meta: [item.price, item.quantity, formatDate(item.created_at)].filter(Boolean).join(" / "),
      value: item.realized_pnl,
    }),
  });
}

function renderRows({ root, selector, items, emptyKey, formatter }) {
  const list = qs(selector, root);
  if (!list) {
    return;
  }
  list.replaceChildren();
  const rows = Array.isArray(items) ? items : [];
  if (rows.length === 0) {
    list.appendChild(messageNode(translate(emptyKey)));
    return;
  }
  rows.forEach((item) => {
    const formatted = formatter(item);
    const row = document.createElement("div");
    row.className = "rh-monitoring-row";
    const title = document.createElement("div");
    const meta = document.createElement("div");
    title.className = "rh-monitoring-event__title";
    meta.className = "rh-monitoring-event__meta";
    title.textContent = formatted.title || "--";
    meta.textContent = formatted.meta || "--";
    if (formatted.value !== undefined && formatted.value !== null) {
      title.classList.add(financialClassName(formatted.value));
    }
    row.append(title, meta);
    list.appendChild(row);
  });
}

function renderEquity(root, equity) {
  renderSource(root, "equity", equity?.source);
  const canvas = qs("[data-monitoring-equity-chart]", root);
  if (!canvas) {
    return;
  }
  drawTimeSeries(canvas, equity?.items || [], {
    emptyText: translate("monitoring.equity.empty"),
  });
}

function renderEvents(root, events) {
  const list = qs("[data-monitoring-events]", root);
  const count = qs("[data-monitoring-event-count]", root);
  if (count) {
    count.textContent = String(events.length);
  }
  if (!list) {
    return;
  }
  list.replaceChildren();
  if (events.length === 0) {
    list.appendChild(messageNode(translate("monitoring.events.empty")));
    return;
  }
  events.forEach((item) => {
    const row = document.createElement("div");
    row.className = "rh-monitoring-event";
    const title = document.createElement("div");
    const meta = document.createElement("div");
    title.className = "rh-monitoring-event__title";
    meta.className = "rh-monitoring-event__meta";
    title.textContent = item.title || "event";
    meta.textContent = item.meta || "--";
    row.append(title, meta);
    list.appendChild(row);
  });
}

function appendEvent(state, item) {
  state.events = [item, ...state.events].slice(0, 20);
}

function renderSource(root, key, source) {
  const badge = qs(`[data-monitoring-source="${key}"]`, root);
  if (!badge || !source) {
    return;
  }
  const status = source.status || "unavailable";
  badge.textContent = translate(`monitoring.source.${status}`);
  badge.title = source.message || "";
  badge.className = `rh-status-badge rh-status-badge--${SOURCE_TONES[status] || "neutral"}`;
}

function setActionState(root, runState) {
  const canRun = !ACTIVE_STATES.includes(runState);
  const runButton = qs('[data-monitoring-action="run"]', root);
  const stopButton = qs('[data-monitoring-action="stop"]', root);
  if (runButton) {
    runButton.disabled = !canRun;
  }
  if (stopButton) {
    stopButton.disabled = canRun;
  }
}

function setConnection(root, key, tone) {
  const status = qs("[data-monitoring-connection]", root);
  if (!status) {
    return;
  }
  status.textContent = translate(key);
  status.className = `rh-status-badge rh-status-badge--${tone}`;
}

function setMobileTab(root, tab) {
  root.dataset.mobileTab = tab;
  qsa("[data-monitoring-mobile-tab]", root).forEach((button) => {
    const active = button.getAttribute("data-monitoring-mobile-tab") === tab;
    button.classList.toggle("rh-monitoring-tab--active", active);
    button.setAttribute("aria-selected", String(active));
  });
}

function metricValue(snapshot, key) {
  const metric = Array.isArray(snapshot?.metrics)
    ? snapshot.metrics.find((item) => item.key === key)
    : null;
  return metric?.value || "--";
}

function setText(root, selector, value) {
  const node = qs(selector, root);
  if (node) {
    node.textContent = value || "--";
  }
}

function messageNode(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

function showError(root, error) {
  const banner = qs("#monitoring-error-banner", root);
  if (!banner) {
    return;
  }
  banner.textContent = error?.message || translate("js.error.network");
  banner.classList.remove("rh-hidden");
}

function clearError(root) {
  const banner = qs("#monitoring-error-banner", root);
  if (!banner) {
    return;
  }
  banner.textContent = "";
  banner.classList.add("rh-hidden");
}

function renderPath(template, value) {
  return String(template || "").replace("{strategy_id}", encodeURIComponent(value));
}

function formatDate(value) {
  if (!value) {
    return "--";
  }
  return formatDateTime(value, { locale: getCurrentLocale(), empty: "--" });
}

function formatLag(value) {
  if (typeof value !== "number") {
    return "";
  }
  return `${value}s`;
}
