import { apiRequest } from "../core/api.js";
import { ready, setBusy } from "../core/dom.js";
import { formatDateTime } from "../core/formatters.js";
import { getCurrentLocale, translate } from "../core/locale.js";
import { notify } from "../core/notifications.js";
import { createPoller } from "../core/poller.js";

const PAGE_SELECTOR = "[data-backtests-history-page]";
const ACTIVE_STATES = new Set(["queued", "running"]);
const STATE_TONES = Object.freeze({
  queued: "info",
  running: "warning",
  succeeded: "success",
  failed: "danger",
  cancelled: "neutral",
});

ready(() => {
  const root = document.querySelector(PAGE_SELECTOR);
  if (!root) {
    return;
  }
  initHistoryPage(root);
});

function initHistoryPage(root) {
  const paths = {
    jobs: requireData(root, "apiJobsPath"),
    counters: requireData(root, "apiCountersPath"),
  };
  const nodes = {
    body: root.querySelector("[data-backtests-history-body]"),
    refresh: root.querySelector("[data-backtests-refresh]"),
    loadMore: root.querySelector("[data-backtests-load-more]"),
    stateFilter: root.querySelector("[data-backtests-state-filter]"),
    error: root.querySelector("[data-backtests-error]"),
    activeCount: root.querySelector("[data-backtests-active-count]"),
    pageCount: root.querySelector("[data-backtests-page-count]"),
    cursorState: root.querySelector("[data-backtests-cursor-state]"),
  };
  if (Object.values(nodes).some((node) => node === null)) {
    return;
  }

  const state = {
    items: [],
    nextCursor: null,
    loading: false,
    pageSize: Number(root.dataset.pageSize || 25),
  };
  const poller = createPoller(
    async ({ signal }) => {
      if (hasActiveJobs(state.items)) {
        await refreshFirstPage({ paths, nodes, state, signal, silent: true });
      }
      await refreshCounters({ paths, nodes, signal, silent: true });
    },
    { intervalMs: 5000, hiddenIntervalMs: 15000 },
  );

  nodes.refresh.addEventListener("click", async () => {
    await refreshFirstPage({ paths, nodes, state });
  });
  nodes.loadMore.addEventListener("click", async () => {
    await loadNextPage({ paths, nodes, state });
  });
  nodes.stateFilter.addEventListener("change", async () => {
    await refreshFirstPage({ paths, nodes, state });
  });
  nodes.body.addEventListener("click", async (event) => {
    const button = event.target instanceof Element
      ? event.target.closest("[data-backtest-action]")
      : null;
    if (!(button instanceof HTMLElement)) {
      return;
    }
    const action = button.dataset.backtestAction;
    const jobId = button.dataset.jobId || "";
    if (action === "cancel" && jobId) {
      await cancelJob({ paths, nodes, state, jobId, button });
    }
    if (action === "copy" && jobId) {
      await navigator.clipboard?.writeText(jobId);
      notify(translate("backtests.history.copied"), { tone: "info" });
    }
  });
  window.addEventListener("beforeunload", () => poller.stop());

  void refreshFirstPage({ paths, nodes, state }).then(() => {
    poller.start({ immediate: false });
  });
}

async function refreshFirstPage({ paths, nodes, state, signal, silent = false }) {
  state.items = [];
  state.nextCursor = null;
  await Promise.all([
    loadJobs({ paths, nodes, state, cursor: null, append: false, signal, silent }),
    refreshCounters({ paths, nodes, signal, silent }),
  ]);
}

async function loadNextPage({ paths, nodes, state }) {
  if (!state.nextCursor) {
    return;
  }
  await loadJobs({ paths, nodes, state, cursor: state.nextCursor, append: true });
}

async function loadJobs({ paths, nodes, state, cursor, append, signal, silent = false }) {
  if (state.loading) {
    return;
  }
  state.loading = true;
  clearError(nodes);
  if (!silent) {
    renderLoading(nodes, append);
  }
  try {
    const params = new URLSearchParams({ limit: String(state.pageSize) });
    if (cursor) {
      params.set("cursor", cursor);
    }
    const filteredState = nodes.stateFilter.value;
    if (filteredState) {
      params.set("state", filteredState);
    }
    const payload = await apiRequest(`${paths.jobs}?${params.toString()}`, { signal });
    const items = Array.isArray(payload?.items) ? payload.items.map(asRecord) : [];
    state.items = append ? [...state.items, ...items] : items;
    state.nextCursor = typeof payload?.next_cursor === "string" ? payload.next_cursor : null;
    renderRows({ nodes, state });
  } catch (error) {
    if (!silent) {
      showError(nodes, error);
    }
  } finally {
    state.loading = false;
    syncPagination(nodes, state);
  }
}

async function refreshCounters({ paths, nodes, signal, silent = false }) {
  try {
    const counters = await apiRequest(paths.counters, { signal });
    const active = Number(counters?.active_jobs ?? 0);
    const max = Number(counters?.max_active_jobs ?? 0);
    nodes.activeCount.textContent = translate("backtests.history.active_count", {
      count: active,
      max,
    });
    nodes.activeCount.className = `rh-status-badge ${
      counters?.can_create === false ? "rh-status-badge--warning" : "rh-status-badge--neutral"
    }`;
  } catch (error) {
    nodes.activeCount.textContent = translate("backtests.history.active_unknown");
    if (!silent) {
      showError(nodes, error);
    }
  }
}

async function cancelJob({ paths, nodes, state, jobId, button }) {
  clearError(nodes);
  setBusy(button, true);
  try {
    await apiRequest(`${paths.jobs}/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    notify(translate("backtests.history.cancel_requested"), { tone: "info" });
    await refreshFirstPage({ paths, nodes, state });
  } catch (error) {
    showError(nodes, error);
  } finally {
    setBusy(button, false);
  }
}

function renderRows({ nodes, state }) {
  nodes.body.replaceChildren();
  if (state.items.length === 0) {
    nodes.body.append(emptyRow(translate("backtests.history.empty")));
    return;
  }
  state.items.forEach((job) => nodes.body.append(buildJobRow(job)));
}

function buildJobRow(job) {
  const row = document.createElement("tr");
  const request = asRecord(job.request);
  const coordinates = asRecord(request.coordinates);
  const progress = asRecord(job.progress);
  appendTextCell(row, formatDate(job.created_at));

  const stateCell = document.createElement("td");
  stateCell.append(buildStateBadge(String(job.state || "")));
  row.append(stateCell);

  const requestCell = document.createElement("td");
  requestCell.className = "rh-backtests-request-cell";
  const symbol = document.createElement("strong");
  symbol.textContent = [coordinates.symbol, request.timeframe].filter(Boolean).join(" / ") || "-";
  const range = document.createElement("span");
  range.textContent = [
    asRecord(request.time_range).start,
    asRecord(request.time_range).end,
  ].filter(Boolean).join(" -> ");
  const ranking = document.createElement("span");
  ranking.textContent = [
    asRecord(job.ranking).primary_metric,
    asRecord(job.ranking).direction,
    job.requested_top_n ? `top_n=${job.requested_top_n}` : "",
  ].filter(Boolean).join(" / ");
  requestCell.append(symbol, range, ranking);
  row.append(requestCell);

  appendTextCell(
    row,
    `${progress.pipeline_stage || "-"} ${Number(progress.percent ?? 0)}%`,
  );
  const hashCell = document.createElement("td");
  hashCell.className = "rh-backtests-code";
  hashCell.textContent = compactHash(job.request_hash);
  row.append(hashCell);

  const actions = document.createElement("td");
  actions.append(
    buildLink(`/backtests/${encodeURIComponent(String(job.job_id || ""))}`, translate("backtests.action.view")),
    buildButton("copy", translate("backtests.action.copy_id"), job.job_id, "rh-button--ghost"),
  );
  if (ACTIVE_STATES.has(String(job.state || ""))) {
    actions.append(buildButton("cancel", translate("backtests.action.cancel"), job.job_id, "rh-button--ghost"));
  }
  row.append(actions);
  return row;
}

function buildStateBadge(state) {
  const badge = document.createElement("span");
  const tone = STATE_TONES[state] || "neutral";
  badge.className = `rh-status-badge rh-status-badge--${tone}`;
  badge.textContent = state || "-";
  return badge;
}

function buildLink(href, label) {
  const link = document.createElement("a");
  link.className = "rh-button rh-button--ghost";
  link.href = href;
  link.textContent = label;
  return link;
}

function buildButton(action, label, jobId, extraClass = "") {
  const button = document.createElement("button");
  button.className = `rh-button ${extraClass}`.trim();
  button.type = "button";
  button.dataset.backtestAction = action;
  button.dataset.jobId = String(jobId || "");
  button.textContent = label;
  return button;
}

function appendTextCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = String(value || "-");
  row.append(cell);
}

function emptyRow(message) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 6;
  cell.textContent = message;
  row.append(cell);
  return row;
}

function renderLoading(nodes, append) {
  if (append) {
    nodes.cursorState.textContent = translate("backtests.status.loading");
    return;
  }
  nodes.body.replaceChildren(emptyRow(translate("backtests.history.loading")));
}

function syncPagination(nodes, state) {
  nodes.loadMore.disabled = !state.nextCursor || state.loading;
  nodes.pageCount.textContent = translate("backtests.history.page_count", {
    count: state.items.length,
  });
  nodes.cursorState.textContent = state.nextCursor
    ? translate("backtests.history.cursor_more")
    : translate("backtests.history.cursor_done");
}

function showError(nodes, error) {
  nodes.error.hidden = false;
  nodes.error.textContent = error?.message || translate("js.error.network");
}

function clearError(nodes) {
  nodes.error.hidden = true;
  nodes.error.textContent = "";
}

function hasActiveJobs(items) {
  return items.some((item) => ACTIVE_STATES.has(String(item.state || "")));
}

function formatDate(value) {
  return formatDateTime(value, { locale: getCurrentLocale(), empty: "-" });
}

function compactHash(value) {
  const text = String(value || "");
  return text.length <= 16 ? text : `${text.slice(0, 10)}...${text.slice(-6)}`;
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function requireData(node, name) {
  const value = node.dataset[name];
  if (!value) {
    throw new Error(`Missing data attribute: ${name}`);
  }
  return value;
}
