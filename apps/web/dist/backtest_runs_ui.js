import {
  ACTIVE_JOB_STATES,
  TERMINAL_JOB_STATES,
  STATUS_POLL_INTERVAL_MS,
  TOP_POLL_INTERVAL_MS,
  asRecord,
  buildCell,
  buildHttpError,
  clearPageError,
  compareStableStrings,
  normalizeError,
  parsePositiveInt,
  renderPathTemplate,
  requireDataAttr,
  showPageError,
} from "./backtest_jobs_ui.js";

const BACKTEST_RUNS_PAGE_SELECTOR = "[data-backtest-runs-page]";
const SERVER_ORDER_SORT = "server_order";
const LOCAL_SORT_NOTE = "Local sort reorders loaded summary rows only.";

document.addEventListener("DOMContentLoaded", () => {
  const pageRoot = document.querySelector(BACKTEST_RUNS_PAGE_SELECTOR);
  if (pageRoot === null) {
    return;
  }

  const pageType = String(pageRoot.dataset.backtestRunsPage || "").trim().toLowerCase();
  if (pageType === "history") {
    initHistoryPage(pageRoot);
    return;
  }
  if (pageType === "summary") {
    initRunSummaryPage(pageRoot);
  }
});

function initHistoryPage(pageRoot) {
  const runsPath = requireDataAttr(pageRoot, "apiRunsPath");
  const detailsPathTemplate = requireDataAttr(pageRoot, "detailsPathTemplate");
  const defaultLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultLimit"), 50);

  const stateFilter = pageRoot.querySelector("#runs-history-state");
  const limitSelect = pageRoot.querySelector("#runs-history-limit");
  const cursorValue = pageRoot.querySelector("#runs-history-cursor-value");
  const cursorNote = pageRoot.querySelector("#runs-history-cursor-note");
  const refreshButton = pageRoot.querySelector("#runs-history-refresh");
  const prevButton = pageRoot.querySelector("#runs-history-prev");
  const nextButton = pageRoot.querySelector("#runs-history-next");
  const tableBody = pageRoot.querySelector("#runs-history-table-body");

  if (
    stateFilter === null
    || limitSelect === null
    || cursorValue === null
    || cursorNote === null
    || refreshButton === null
    || prevButton === null
    || nextButton === null
    || tableBody === null
  ) {
    return;
  }

  limitSelect.value = String(defaultLimit);

  const state = {
    pageCursors: [""],
    pageIndex: 0,
    nextCursor: null,
    isLoading: false,
    requestToken: 0,
  };

  const currentCursor = () => String(state.pageCursors[state.pageIndex] || "");

  const updatePagerControls = () => {
    stateFilter.disabled = false;
    limitSelect.disabled = false;
    refreshButton.disabled = false;
    cursorValue.disabled = false;
    prevButton.disabled = state.isLoading || state.pageIndex <= 0;
    nextButton.disabled = state.isLoading || state.nextCursor === null;
  };

  const updateCursorUi = () => {
    const current = currentCursor();
    cursorValue.value = current;
    cursorNote.textContent = state.nextCursor === null
      ? "next_cursor: none"
      : `next_cursor: ${state.nextCursor}`;
  };

  const renderRows = (items) => {
    tableBody.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
      tableBody.innerHTML = "<tr><td colspan=\"10\">No runs found.</td></tr>";
      return;
    }

    items.forEach((item) => {
      const record = asRecord(item);
      const runId = String(record.run_id || "").trim();
      const row = document.createElement("tr");
      if (runId.length > 0) {
        row.dataset.runId = runId;
        row.classList.add("row-clickable");
      }

      row.appendChild(buildCell(runId));
      row.appendChild(buildBadgeCell(String(record.state || "")));
      row.appendChild(buildCell(formatValue(record.execution_mode)));
      row.appendChild(buildCell(formatValue(record.market_id)));
      row.appendChild(buildCell(formatValue(record.symbol)));
      row.appendChild(buildCell(formatValue(record.timeframe)));
      row.appendChild(buildCell(formatValue(record.requested_top_n)));
      row.appendChild(buildCell(formatValue(record.created_at)));
      row.appendChild(buildCell(formatValue(record.updated_at)));
      row.appendChild(
        buildCell(
          `${formatValue(record.processed_units)}/${formatValue(record.total_units)}`,
        ),
      );
      tableBody.appendChild(row);
    });
  };

  const loadRuns = async () => {
    const token = state.requestToken + 1;
    state.requestToken = token;
    state.isLoading = true;
    updatePagerControls();
    updateCursorUi();
    clearPageError(pageRoot);

    if (tableBody.children.length === 0) {
      tableBody.innerHTML = "<tr><td colspan=\"10\">Loading backtest history...</td></tr>";
    }

    try {
      const requestUrl = new URL(runsPath, window.location.origin);
      const stateValue = String(stateFilter.value || "").trim();
      if (stateValue.length > 0) {
        requestUrl.searchParams.set("state", stateValue);
      }
      requestUrl.searchParams.set(
        "limit",
        String(parsePositiveInt(String(limitSelect.value || ""), defaultLimit)),
      );
      const cursor = currentCursor().trim();
      if (cursor.length > 0) {
        requestUrl.searchParams.set("cursor", cursor);
      }

      const response = await fetch(requestUrl.toString(), {
        credentials: "include",
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      if (token !== state.requestToken) {
        return;
      }

      const items = Array.isArray(payload.items) ? payload.items : [];
      const rawNextCursor = String(payload.next_cursor || "").trim();
      state.nextCursor = rawNextCursor.length > 0 ? rawNextCursor : null;
      renderRows(items);
      updateCursorUi();
    } catch (error) {
      if (token !== state.requestToken) {
        return;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      tableBody.innerHTML = "<tr><td colspan=\"10\">Failed to load backtest history.</td></tr>";
      state.nextCursor = null;
      updateCursorUi();
    } finally {
      if (token === state.requestToken) {
        state.isLoading = false;
        updatePagerControls();
      }
    }
  };

  const resetPagination = () => {
    state.pageCursors = [""];
    state.pageIndex = 0;
    state.nextCursor = null;
    updateCursorUi();
  };

  stateFilter.addEventListener("change", async () => {
    resetPagination();
    await loadRuns();
  });

  limitSelect.addEventListener("change", async () => {
    const parsedLimit = parsePositiveInt(String(limitSelect.value || ""), defaultLimit);
    limitSelect.value = String(parsedLimit);
    resetPagination();
    await loadRuns();
  });

  refreshButton.addEventListener("click", async () => {
    await loadRuns();
  });

  prevButton.addEventListener("click", async () => {
    if (state.pageIndex <= 0) {
      return;
    }
    state.pageIndex -= 1;
    await loadRuns();
  });

  nextButton.addEventListener("click", async () => {
    if (state.nextCursor === null) {
      return;
    }
    if (state.pageIndex === state.pageCursors.length - 1) {
      state.pageCursors.push(state.nextCursor);
    }
    state.pageIndex += 1;
    await loadRuns();
  });

  tableBody.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const row = target.closest("tr[data-run-id]");
    if (row === null) {
      return;
    }

    const runId = String(row.getAttribute("data-run-id") || "").trim();
    if (runId.length === 0) {
      return;
    }
    window.location.assign(renderPathTemplate(detailsPathTemplate, encodeURIComponent(runId)));
  });

  updateCursorUi();
  updatePagerControls();
  loadRuns();
}

function initRunSummaryPage(pageRoot) {
  const runId = requireDataAttr(pageRoot, "runId");
  const runsPathPrefix = requireDataAttr(pageRoot, "apiRunsPathPrefix");
  const topPathTemplate = requireDataAttr(pageRoot, "apiTopPathTemplate");
  const runtimeDefaultsPath = requireDataAttr(pageRoot, "apiRuntimeDefaultsPath");
  const defaultTopLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultTopLimit"), 50);

  const copyIdButton = pageRoot.querySelector("#run-copy-id");
  const refreshStatusButton = pageRoot.querySelector("#run-refresh-status");
  const refreshSummaryButton = pageRoot.querySelector("#run-refresh-summary");
  const sortColumnSelect = pageRoot.querySelector("#run-summary-sort-column");
  const sortDirectionSelect = pageRoot.querySelector("#run-summary-sort-direction");
  const summaryNote = pageRoot.querySelector("#run-summary-note");
  const tableHead = pageRoot.querySelector("#run-summary-table-head");
  const tableBody = pageRoot.querySelector("#run-summary-table-body");
  const progressBar = pageRoot.querySelector("#run-progress-bar");
  const progressCaption = pageRoot.querySelector("#run-progress-caption");

  if (
    copyIdButton === null
    || refreshStatusButton === null
    || refreshSummaryButton === null
    || sortColumnSelect === null
    || sortDirectionSelect === null
    || summaryNote === null
    || tableHead === null
    || tableBody === null
    || progressBar === null
    || progressCaption === null
  ) {
    return;
  }

  const fieldMap = {
    mode: pageRoot.querySelector("#run-field-mode"),
    state: pageRoot.querySelector("#run-field-state"),
    stage: pageRoot.querySelector("#run-field-stage"),
    executionMode: pageRoot.querySelector("#run-field-execution-mode"),
    marketId: pageRoot.querySelector("#run-field-market-id"),
    symbol: pageRoot.querySelector("#run-field-symbol"),
    timeframe: pageRoot.querySelector("#run-field-timeframe"),
    requestedTopN: pageRoot.querySelector("#run-field-requested-top-n"),
    createdAt: pageRoot.querySelector("#run-field-created-at"),
    updatedAt: pageRoot.querySelector("#run-field-updated-at"),
    startedAt: pageRoot.querySelector("#run-field-started-at"),
    finishedAt: pageRoot.querySelector("#run-field-finished-at"),
    cancelRequestedAt: pageRoot.querySelector("#run-field-cancel-requested-at"),
    progress: pageRoot.querySelector("#run-field-progress"),
    rankingPrimaryMetric: pageRoot.querySelector("#run-field-ranking-primary-metric"),
    rankingSecondaryMetric: pageRoot.querySelector("#run-field-ranking-secondary-metric"),
  };

  if (Object.values(fieldMap).some((node) => node === null)) {
    return;
  }

  const state = {
    runtimeDefaults: null,
    status: null,
    topRowsOriginal: [],
    topRowsSorted: [],
    sortableColumns: [],
    visibleColumns: [],
    topLimit: defaultTopLimit,
    sortColumn: SERVER_ORDER_SORT,
    sortDirection: "desc",
    statusTimerId: 0,
    topTimerId: 0,
    finalRefreshDone: false,
    statusRequestToken: 0,
    topRequestToken: 0,
  };

  const statusPath = `${runsPathPrefix}${encodeURIComponent(runId)}`;
  const renderTopPath = () => {
    const templatePath = renderPathTemplate(topPathTemplate, encodeURIComponent(runId));
    const requestUrl = new URL(templatePath, window.location.origin);
    requestUrl.searchParams.set("limit", String(state.topLimit));
    return requestUrl.toString();
  };

  const buildSortableColumnsFromRuntime = () => {
    const contracts = asRecord(asRecord(state.runtimeDefaults).contracts);
    const summaryContract = asRecord(contracts.summary);
    return Array.isArray(summaryContract.sortable_columns)
      ? summaryContract.sortable_columns
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
      : [];
  };

  const updateSortControls = () => {
    const previousValue = String(sortColumnSelect.value || SERVER_ORDER_SORT).trim();
    sortColumnSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.value = SERVER_ORDER_SORT;
    defaultOption.textContent = SERVER_ORDER_SORT;
    sortColumnSelect.appendChild(defaultOption);

    state.sortableColumns.forEach((column) => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      sortColumnSelect.appendChild(option);
    });

    if ([SERVER_ORDER_SORT, ...state.sortableColumns].includes(previousValue)) {
      sortColumnSelect.value = previousValue;
    } else {
      sortColumnSelect.value = SERVER_ORDER_SORT;
    }
    state.sortColumn = String(sortColumnSelect.value || SERVER_ORDER_SORT).trim();
    sortDirectionSelect.disabled = state.sortColumn === SERVER_ORDER_SORT;
  };

  const updateTopLimitFromStatus = () => {
    const requestedTopN = parsePositiveInt(
      String(asRecord(state.status).requested_top_n || ""),
      defaultTopLimit,
    );
    state.topLimit = requestedTopN;
  };

  const renderStatus = (rawStatus) => {
    const status = asRecord(rawStatus);
    state.status = status;
    updateTopLimitFromStatus();

    const processedUnits = Number(status.processed_units || 0);
    const totalUnits = Number(status.total_units || 0);
    const ratio = totalUnits > 0 ? Math.min(Math.max(processedUnits / totalUnits, 0), 1) : 0;

    setTextContent(fieldMap.mode, formatValue(status.mode));
    setBadgeContent(fieldMap.state, String(status.state || ""));
    setTextContent(fieldMap.stage, formatValue(status.stage));
    setTextContent(fieldMap.executionMode, formatValue(status.execution_mode));
    setTextContent(fieldMap.marketId, formatValue(status.market_id));
    setTextContent(fieldMap.symbol, formatValue(status.symbol));
    setTextContent(fieldMap.timeframe, formatValue(status.timeframe));
    setTextContent(fieldMap.requestedTopN, formatValue(status.requested_top_n));
    setTextContent(fieldMap.createdAt, formatValue(status.created_at));
    setTextContent(fieldMap.updatedAt, formatValue(status.updated_at));
    setTextContent(fieldMap.startedAt, formatValue(status.started_at));
    setTextContent(fieldMap.finishedAt, formatValue(status.finished_at));
    setTextContent(fieldMap.cancelRequestedAt, formatValue(status.cancel_requested_at));
    setTextContent(fieldMap.progress, `${processedUnits}/${totalUnits}`);
    setTextContent(fieldMap.rankingPrimaryMetric, formatValue(status.ranking_primary_metric));
    setTextContent(fieldMap.rankingSecondaryMetric, formatValue(status.ranking_secondary_metric));

    progressBar.style.width = `${Math.round(ratio * 100)}%`;
    progressCaption.textContent = `Progress: ${Math.round(ratio * 100)}% (${processedUnits}/${totalUnits})`;
  };

  const buildVisibleColumns = () => {
    if (!Array.isArray(state.topRowsOriginal) || state.topRowsOriginal.length === 0) {
      return ["total_return_pct", "best_tp_pct", "best_sl_pct"];
    }

    const rows = state.topRowsOriginal;
    const runtimeColumns = state.sortableColumns.length > 0
      ? state.sortableColumns
      : ["total_return_pct", "best_tp_pct", "best_sl_pct"];

    return runtimeColumns.filter((column) => {
      if (column === "total_return_pct" || column === "best_tp_pct" || column === "best_sl_pct") {
        return true;
      }
      return rows.some((row) => {
        const metrics = asRecord(asRecord(row).summary_metrics_json);
        return Object.prototype.hasOwnProperty.call(metrics, column);
      });
    });
  };

  const buildMetricValue = (row, column) => {
    const record = asRecord(row);
    if (column === "total_return_pct") {
      return readFiniteNumber(record.total_return_pct);
    }
    if (column === "best_tp_pct") {
      return readFiniteNumber(record.best_tp_pct);
    }
    if (column === "best_sl_pct") {
      return readFiniteNumber(record.best_sl_pct);
    }
    return readFiniteNumber(asRecord(record.summary_metrics_json)[column]);
  };

  const compareTopRows = (leftRow, rightRow) => {
    const left = asRecord(leftRow);
    const right = asRecord(rightRow);
    const leftMetric = buildMetricValue(left, state.sortColumn);
    const rightMetric = buildMetricValue(right, state.sortColumn);
    const leftVariantKey = String(left.variant_key || "").trim();
    const rightVariantKey = String(right.variant_key || "").trim();

    if (leftMetric === null && rightMetric !== null) {
      return 1;
    }
    if (leftMetric !== null && rightMetric === null) {
      return -1;
    }
    if (leftMetric !== null && rightMetric !== null && leftMetric !== rightMetric) {
      if (state.sortDirection === "asc") {
        return leftMetric < rightMetric ? -1 : 1;
      }
      return leftMetric > rightMetric ? -1 : 1;
    }

    const variantComparison = compareStableStrings(leftVariantKey, rightVariantKey);
    if (variantComparison !== 0) {
      return variantComparison;
    }

    return Number(left.rank || 0) - Number(right.rank || 0);
  };

  const renderTopTable = () => {
    state.visibleColumns = buildVisibleColumns();
    renderTopHead(tableHead, state.visibleColumns);
    const emptyColspan = Math.max(state.visibleColumns.length + 3, 3);

    tableBody.innerHTML = "";
    if (state.topRowsSorted.length === 0) {
      tableBody.innerHTML = (
        `<tr><td colspan="${emptyColspan}">No persisted summary rows yet.</td></tr>`
      );
      return;
    }

    state.topRowsSorted.forEach((rawRow) => {
      const row = asRecord(rawRow);
      const tr = document.createElement("tr");
      tr.appendChild(buildCell(formatValue(row.rank)));
      tr.appendChild(buildCell(formatValue(row.variant_key)));
      tr.appendChild(buildCell(formatValue(row.indicator_variant_key)));
      state.visibleColumns.forEach((column) => {
        tr.appendChild(buildCell(formatValue(readRenderedMetric(row, column))));
      });
      tableBody.appendChild(tr);
    });
  };

  const applyLocalTopSort = () => {
    if (state.sortColumn === SERVER_ORDER_SORT) {
      state.topRowsSorted = state.topRowsOriginal.slice();
      summaryNote.innerHTML = "First render keeps server order <code>rank ASC, variant_key ASC</code>.";
      sortDirectionSelect.disabled = true;
      renderTopTable();
      return;
    }

    // Local sort reorders already-loaded rows only; do not refetch `/top` here.
    state.topRowsSorted = state.topRowsOriginal.slice().sort(compareTopRows);
    summaryNote.innerHTML = [
      "First render keeps server order <code>rank ASC, variant_key ASC</code>.",
      `${LOCAL_SORT_NOTE} Tie-break stays <code>variant_key ASC</code>.`,
    ].join(" ");
    sortDirectionSelect.disabled = false;
    renderTopTable();
  };

  const loadRuntimeDefaults = async () => {
    try {
      const response = await fetch(runtimeDefaultsPath, {
        credentials: "include",
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      state.runtimeDefaults = asRecord(await response.json());
      state.sortableColumns = buildSortableColumnsFromRuntime();
      updateSortControls();
      applyLocalTopSort();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  const stopPolling = () => {
    if (state.statusTimerId !== 0) {
      window.clearInterval(state.statusTimerId);
      state.statusTimerId = 0;
    }
    if (state.topTimerId !== 0) {
      window.clearInterval(state.topTimerId);
      state.topTimerId = 0;
    }
  };

  const startPolling = () => {
    if (state.statusTimerId === 0) {
      state.statusTimerId = window.setInterval(() => {
        loadStatus();
      }, STATUS_POLL_INTERVAL_MS);
    }
    if (state.topTimerId === 0) {
      state.topTimerId = window.setInterval(() => {
        loadTop();
      }, TOP_POLL_INTERVAL_MS);
    }
  };

  const runFinalRefresh = async () => {
    await Promise.all([
      loadStatus({ skipTransitionHandling: true }),
      loadTop(),
    ]);
    stopPolling();
  };

  const handleStatusTransition = async () => {
    const statusState = String(asRecord(state.status).state || "").trim().toLowerCase();
    if (ACTIVE_JOB_STATES.has(statusState)) {
      state.finalRefreshDone = false;
      startPolling();
      return;
    }
    if (!TERMINAL_JOB_STATES.has(statusState)) {
      stopPolling();
      return;
    }
    stopPolling();
    if (!state.finalRefreshDone) {
      state.finalRefreshDone = true;
      await runFinalRefresh();
    }
  };

  const loadStatus = async ({ skipTransitionHandling = false } = {}) => {
    const token = state.statusRequestToken + 1;
    state.statusRequestToken = token;

    try {
      const response = await fetch(statusPath, {
        credentials: "include",
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      if (token !== state.statusRequestToken) {
        return null;
      }

      renderStatus(payload);
      if (!skipTransitionHandling) {
        await handleStatusTransition();
      }
      return payload;
    } catch (error) {
      if (token !== state.statusRequestToken) {
        return null;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      return null;
    }
  };

  const loadTop = async () => {
    const token = state.topRequestToken + 1;
    state.topRequestToken = token;

    try {
      const response = await fetch(renderTopPath(), {
        credentials: "include",
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      if (token !== state.topRequestToken) {
        return null;
      }

      state.topRowsOriginal = Array.isArray(payload.items) ? payload.items.slice() : [];
      applyLocalTopSort();
      return payload;
    } catch (error) {
      if (token !== state.topRequestToken) {
        return null;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      const fallbackColspan = Math.max(state.visibleColumns.length + 3, 3);
      tableBody.innerHTML = (
        `<tr><td colspan="${fallbackColspan}">Failed to load summary rows.</td></tr>`
      );
      return null;
    }
  };

  const copyRunIdToClipboard = async () => {
    clearPageError(pageRoot);
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(runId);
        return;
      }

      const helper = document.createElement("textarea");
      helper.value = runId;
      helper.setAttribute("readonly", "readonly");
      helper.style.position = "fixed";
      helper.style.left = "-10000px";
      document.body.appendChild(helper);
      helper.select();
      document.execCommand("copy");
      helper.remove();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  copyIdButton.addEventListener("click", async () => {
    await copyRunIdToClipboard();
  });

  refreshStatusButton.addEventListener("click", async () => {
    clearPageError(pageRoot);
    await loadStatus();
  });

  refreshSummaryButton.addEventListener("click", async () => {
    clearPageError(pageRoot);
    await loadTop();
  });

  sortColumnSelect.addEventListener("change", () => {
    state.sortColumn = String(sortColumnSelect.value || SERVER_ORDER_SORT).trim();
    applyLocalTopSort();
  });

  sortDirectionSelect.addEventListener("change", () => {
    state.sortDirection = String(sortDirectionSelect.value || "desc").trim() === "asc"
      ? "asc"
      : "desc";
    if (state.sortColumn !== SERVER_ORDER_SORT) {
      applyLocalTopSort();
    }
  });

  const bootstrap = async () => {
    clearPageError(pageRoot);
    await loadRuntimeDefaults();
    await loadStatus();
    await loadTop();
  };

  updateSortControls();
  bootstrap();
}

function buildBadgeCell(text) {
  const cell = document.createElement("td");
  cell.appendChild(buildStateBadge(text));
  return cell;
}

function setBadgeContent(node, text) {
  node.innerHTML = "";
  node.appendChild(buildStateBadge(text));
}

function buildStateBadge(text) {
  const value = String(text || "").trim().toLowerCase();
  const badge = document.createElement("span");
  badge.className = "state-badge";
  badge.textContent = value.length > 0 ? value : "-";
  if (value.length > 0) {
    badge.classList.add(`state-badge--${value}`);
  }
  return badge;
}

function setTextContent(node, value) {
  node.textContent = value;
}

function renderTopHead(tableHead, visibleColumns) {
  tableHead.innerHTML = "";
  const row = document.createElement("tr");

  const staticHeaders = ["rank", "variant_key", "indicator_variant_key"];
  staticHeaders.forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    row.appendChild(th);
  });

  visibleColumns.forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    row.appendChild(th);
  });
  tableHead.appendChild(row);
}

function readRenderedMetric(row, column) {
  const record = asRecord(row);
  if (column === "total_return_pct") {
    return record.total_return_pct;
  }
  if (column === "best_tp_pct") {
    return record.best_tp_pct;
  }
  if (column === "best_sl_pct") {
    return record.best_sl_pct;
  }
  return asRecord(record.summary_metrics_json)[column];
}

function readFiniteNumber(value) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function formatValue(value) {
  if (value === null || typeof value === "undefined" || String(value).trim().length === 0) {
    return "-";
  }
  return String(value);
}
