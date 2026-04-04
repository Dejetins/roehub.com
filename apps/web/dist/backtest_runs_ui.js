import {
  ACTIVE_JOB_STATES,
  TERMINAL_JOB_STATES,
  STATUS_POLL_INTERVAL_MS,
  TOP_POLL_INTERVAL_MS,
  asRecord,
  buildActionButton,
  buildCell,
  buildHttpError,
  buildStrategyPrefillPayload,
  clearPageError,
  compareStableStrings,
  normalizeError,
  parsePositiveInt,
  persistStrategyPrefillAndNavigate,
  renderMarkdownToSafeHtml,
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
    return;
  }
  if (pageType === "detail") {
    initVariantDetailPage(pageRoot);
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
      row.appendChild(buildCell(buildHistoryExecutionModeSummary(record)));
      row.appendChild(buildCell(formatValue(record.market_id)));
      row.appendChild(buildCell(formatValue(record.symbol)));
      row.appendChild(buildCell(formatValue(record.timeframe)));
      row.appendChild(buildCell(formatValue(record.requested_top_n)));
      row.appendChild(buildCell(formatValue(record.created_at)));
      row.appendChild(buildCell(formatValue(record.updated_at)));
      row.appendChild(buildCell(buildHistoryProgressSummary(record)));
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
  const marketsPath = requireDataAttr(pageRoot, "apiMarketsPath");
  const variantDetailPathTemplate = requireDataAttr(pageRoot, "detailPathTemplate");
  const strategyBuilderPath = requireDataAttr(pageRoot, "strategyBuilderPath");
  const prefillQueryParam = requireDataAttr(pageRoot, "prefillQueryParam");
  const prefillStorage = requireDataAttr(pageRoot, "prefillStorage");
  const defaultTopLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultTopLimit"), 50);

  const copyIdButton = pageRoot.querySelector("#run-copy-id");
  const refreshStatusButton = pageRoot.querySelector("#run-refresh-status");
  const refreshSummaryButton = pageRoot.querySelector("#run-refresh-summary");
  const sortColumnSelect = pageRoot.querySelector("#run-summary-sort-column");
  const sortDirectionSelect = pageRoot.querySelector("#run-summary-sort-direction");
  const summaryNote = pageRoot.querySelector("#run-summary-note");
  const tableHead = pageRoot.querySelector("#run-summary-table-head");
  const tableBody = pageRoot.querySelector("#run-summary-table-body");
  const progressTrack = pageRoot.querySelector(".job-progress-track[role=\"progressbar\"]");
  const progressBar = pageRoot.querySelector("#run-progress-bar");
  const progressCaption = pageRoot.querySelector("#run-progress-caption");
  const progressMeta = pageRoot.querySelector("#run-progress-meta");

  if (
    copyIdButton === null
    || refreshStatusButton === null
    || refreshSummaryButton === null
    || sortColumnSelect === null
    || sortDirectionSelect === null
    || summaryNote === null
    || tableHead === null
    || tableBody === null
    || progressTrack === null
    || progressBar === null
    || progressCaption === null
    || progressMeta === null
  ) {
    return;
  }

  const fieldMap = {
    mode: pageRoot.querySelector("#run-field-mode"),
    state: pageRoot.querySelector("#run-field-state"),
    stage: pageRoot.querySelector("#run-field-stage"),
    executionMode: pageRoot.querySelector("#run-field-execution-mode"),
    executionProfileMode: pageRoot.querySelector("#run-field-execution-profile-mode"),
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
    progressPercent: pageRoot.querySelector("#run-field-progress-percent"),
    etaSeconds: pageRoot.querySelector("#run-field-eta-seconds"),
    rankingPrimaryMetric: pageRoot.querySelector("#run-field-ranking-primary-metric"),
    rankingSecondaryMetric: pageRoot.querySelector("#run-field-ranking-secondary-metric"),
  };

  if (Object.values(fieldMap).some((node) => node === null)) {
    return;
  }

  const resolveRunStrategyContext = createRunStrategyContextResolver({ marketsPath });
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
    const progressPercent = readProgressPercent(status, {
      processedUnits,
      totalUnits,
    });
    const executionProfileMode = formatValue(status.execution_profile_mode);
    const etaSeconds = readFiniteInteger(status.eta_seconds);

    setTextContent(fieldMap.mode, formatValue(status.mode));
    setBadgeContent(fieldMap.state, String(status.state || ""));
    setTextContent(fieldMap.stage, formatValue(status.stage));
    setTextContent(fieldMap.executionMode, formatValue(status.execution_mode));
    setTextContent(fieldMap.executionProfileMode, executionProfileMode);
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
    setTextContent(fieldMap.progressPercent, formatPercentValue(progressPercent));
    setTextContent(fieldMap.etaSeconds, formatEtaSeconds(etaSeconds));
    setTextContent(fieldMap.rankingPrimaryMetric, formatValue(status.ranking_primary_metric));
    setTextContent(fieldMap.rankingSecondaryMetric, formatValue(status.ranking_secondary_metric));

    progressTrack.setAttribute("aria-valuenow", String(progressPercent));
    progressBar.style.width = `${progressPercent}%`;
    progressCaption.textContent = `Progress: ${progressPercent}% (${processedUnits}/${totalUnits})`;
    progressMeta.textContent = [
      `Stage: ${formatValue(status.stage)}`,
      `Profile: ${executionProfileMode}`,
      `ETA: ${etaSeconds === null ? "-" : `~${formatEtaSeconds(etaSeconds)}`}`,
    ].join(" | ");
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

  const saveRowAsStrategy = async (rawRow) => {
    clearPageError(pageRoot);
    try {
      const runContext = await resolveRunStrategyContext(state.status);
      const prefillPayload = buildStrategyPrefillPayload({
        variantPayload: asRecord(rawRow).payload,
        runContext,
      });
      persistStrategyPrefillAndNavigate({
        pageRoot,
        strategyBuilderPath,
        prefillQueryParam,
        prefillStorage,
        prefillPayload,
      });
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  const renderTopTable = () => {
    state.visibleColumns = buildVisibleColumns();
    renderTopHead(tableHead, state.visibleColumns, { includeActions: true });
    const emptyColspan = Math.max(state.visibleColumns.length + 4, 4);

    tableBody.innerHTML = "";
    if (state.topRowsSorted.length === 0) {
      tableBody.innerHTML = (
        `<tr><td colspan="${emptyColspan}">No persisted summary rows yet.</td></tr>`
      );
      return;
    }

    state.topRowsSorted.forEach((rawRow) => {
      const row = asRecord(rawRow);
      const variantKey = String(row.variant_key || "").trim();
      const tr = document.createElement("tr");
      tr.appendChild(buildCell(formatValue(row.rank)));
      tr.appendChild(buildCell(formatValue(variantKey)));
      tr.appendChild(buildCell(formatValue(row.indicator_variant_key)));
      state.visibleColumns.forEach((column) => {
        tr.appendChild(buildCell(formatValue(readRenderedMetric(row, column))));
      });

      const actionsCell = document.createElement("td");
      const detailLink = document.createElement("a");
      detailLink.className = "button-link";
      detailLink.textContent = "Open detail";
      detailLink.href = buildVariantDetailPath({
        pathTemplate: variantDetailPathTemplate,
        runId,
        variantKey,
      });
      if (variantKey.length === 0) {
        detailLink.setAttribute("aria-disabled", "true");
        detailLink.classList.add("button-link--disabled");
        detailLink.href = "#";
        detailLink.addEventListener("click", (event) => {
          event.preventDefault();
        });
      }
      actionsCell.appendChild(detailLink);
      actionsCell.appendChild(
        buildActionButton({
          label: "Save as Strategy",
          className: "button-link--secondary",
          disabled: variantKey.length === 0,
          onClick: async () => {
            await saveRowAsStrategy(rawRow);
          },
        }),
      );
      tr.appendChild(actionsCell);
      tableBody.appendChild(tr);
    });
  };

  const applyLocalTopSort = () => {
    if (state.sortColumn === SERVER_ORDER_SORT) {
      state.topRowsSorted = state.topRowsOriginal.slice();
      summaryNote.innerHTML = [
        "First render keeps server order <code>rank ASC, variant_key ASC</code>.",
        "Detail, chart, and trades stay on the dedicated variant page only.",
      ].join(" ");
      sortDirectionSelect.disabled = true;
      renderTopTable();
      return;
    }

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
      const fallbackColspan = Math.max(state.visibleColumns.length + 4, 4);
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

function initVariantDetailPage(pageRoot) {
  const runId = requireDataAttr(pageRoot, "runId");
  const variantKey = requireDataAttr(pageRoot, "variantKey");
  const runsPathPrefix = requireDataAttr(pageRoot, "apiRunsPathPrefix");
  const topPathTemplate = requireDataAttr(pageRoot, "apiTopPathTemplate");
  const variantReportPathTemplate = requireDataAttr(pageRoot, "apiVariantReportPathTemplate");
  const marketsPath = requireDataAttr(pageRoot, "apiMarketsPath");
  const strategyBuilderPath = requireDataAttr(pageRoot, "strategyBuilderPath");
  const prefillQueryParam = requireDataAttr(pageRoot, "prefillQueryParam");
  const prefillStorage = requireDataAttr(pageRoot, "prefillStorage");
  const defaultTopLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultTopLimit"), 50);

  const refreshButton = pageRoot.querySelector("#variant-refresh-detail");
  const saveButton = pageRoot.querySelector("#variant-save-strategy");
  const includeTradesToggle = pageRoot.querySelector("#variant-include-trades");
  const loadingNode = pageRoot.querySelector("#variant-detail-loading");
  const missingBanner = pageRoot.querySelector("#variant-missing-banner");
  const metricsNode = pageRoot.querySelector("#variant-detail-metrics");
  const markdownNode = pageRoot.querySelector("#variant-detail-markdown");
  const chartNode = pageRoot.querySelector("#variant-detail-chart");
  const tradesNode = pageRoot.querySelector("#variant-detail-trades");
  const selectedIndicatorsNode = pageRoot.querySelector("#variant-selected-indicators");

  if (
    refreshButton === null
    || saveButton === null
    || includeTradesToggle === null
    || loadingNode === null
    || missingBanner === null
    || metricsNode === null
    || markdownNode === null
    || chartNode === null
    || tradesNode === null
    || selectedIndicatorsNode === null
  ) {
    return;
  }

  const fieldMap = {
    runState: pageRoot.querySelector("#variant-field-run-state"),
    executionMode: pageRoot.querySelector("#variant-field-execution-mode"),
    marketId: pageRoot.querySelector("#variant-field-market-id"),
    symbol: pageRoot.querySelector("#variant-field-symbol"),
    timeframe: pageRoot.querySelector("#variant-field-timeframe"),
    rank: pageRoot.querySelector("#variant-field-rank"),
    indicatorVariantKey: pageRoot.querySelector("#variant-field-indicator-variant-key"),
    totalReturnPct: pageRoot.querySelector("#variant-field-total-return-pct"),
    directionMode: pageRoot.querySelector("#variant-field-direction-mode"),
    sizingMode: pageRoot.querySelector("#variant-field-sizing-mode"),
  };

  if (Object.values(fieldMap).some((node) => node === null)) {
    return;
  }

  const resolveRunStrategyContext = createRunStrategyContextResolver({ marketsPath });
  const state = {
    status: null,
    topRows: [],
    selectedRow: null,
    topLimit: defaultTopLimit,
    statusRequestToken: 0,
    topRequestToken: 0,
    reportRequestToken: 0,
    reportCacheByKey: new Map(),
    isLoadingReport: false,
  };

  const statusPath = `${runsPathPrefix}${encodeURIComponent(runId)}`;
  const renderTopPath = () => {
    const templatePath = renderPathTemplate(topPathTemplate, encodeURIComponent(runId));
    const requestUrl = new URL(templatePath, window.location.origin);
    requestUrl.searchParams.set("limit", String(state.topLimit));
    return requestUrl.toString();
  };
  const variantReportPath = renderPathTemplate(
    variantReportPathTemplate,
    encodeURIComponent(runId),
  );

  const setLoadingState = (isLoading) => {
    state.isLoadingReport = isLoading;
    loadingNode.classList.toggle("hidden", !isLoading);
    refreshButton.disabled = isLoading;
    includeTradesToggle.disabled = isLoading;
    saveButton.disabled = isLoading || state.selectedRow === null;
  };

  const renderMissingVariant = (message) => {
    missingBanner.textContent = message;
    missingBanner.classList.remove("hidden");
    metricsNode.innerHTML = "";
    markdownNode.innerHTML = "";
    chartNode.innerHTML = "";
    tradesNode.innerHTML = "";
    selectedIndicatorsNode.innerHTML = "";
    saveButton.disabled = true;
  };

  const clearMissingVariant = () => {
    missingBanner.textContent = "";
    missingBanner.classList.add("hidden");
    saveButton.disabled = state.isLoadingReport || state.selectedRow === null;
  };

  const renderStatus = (rawStatus) => {
    const status = asRecord(rawStatus);
    state.status = status;
    state.topLimit = parsePositiveInt(String(status.requested_top_n || ""), defaultTopLimit);
    setBadgeContent(fieldMap.runState, String(status.state || ""));
    setTextContent(fieldMap.executionMode, formatValue(status.execution_mode));
    setTextContent(fieldMap.marketId, formatValue(status.market_id));
    setTextContent(fieldMap.symbol, formatValue(status.symbol));
    setTextContent(fieldMap.timeframe, formatValue(status.timeframe));
  };

  const renderSelectedVariantSummary = () => {
    const row = asRecord(state.selectedRow);
    const payload = asRecord(row.payload);
    setTextContent(fieldMap.rank, formatValue(row.rank));
    setTextContent(fieldMap.indicatorVariantKey, formatValue(row.indicator_variant_key));
    setTextContent(fieldMap.totalReturnPct, formatValue(row.total_return_pct));
    setTextContent(fieldMap.directionMode, formatValue(payload.direction_mode));
    setTextContent(fieldMap.sizingMode, formatValue(payload.sizing_mode));
    renderSelectedIndicators(payload);
  };

  const renderSelectedIndicators = (payload) => {
    const selections = Array.isArray(asRecord(payload).indicator_selections)
      ? asRecord(payload).indicator_selections
      : [];
    selectedIndicatorsNode.innerHTML = "";
    if (selections.length === 0) {
      const emptyNode = document.createElement("li");
      emptyNode.textContent = "No indicator_selections payload.";
      selectedIndicatorsNode.appendChild(emptyNode);
      return;
    }

    selections.forEach((item) => {
      const selection = asRecord(item);
      const indicatorId = String(selection.indicator_id || "").trim();
      const node = document.createElement("li");
      node.textContent = [
        indicatorId.length > 0 ? indicatorId : "unknown_indicator",
        `inputs=${JSON.stringify(asRecord(selection.inputs))}`,
        `params=${JSON.stringify(asRecord(selection.params))}`,
      ].join(" ");
      selectedIndicatorsNode.appendChild(node);
    });
  };

  const renderMetricRows = (rows) => {
    metricsNode.innerHTML = "";
    if (!Array.isArray(rows) || rows.length === 0) {
      const emptyNode = document.createElement("p");
      emptyNode.className = "muted-text";
      emptyNode.textContent = "No detail rows returned.";
      metricsNode.appendChild(emptyNode);
      return;
    }

    const list = document.createElement("dl");
    list.className = "detail-metrics-grid";
    rows.forEach((row) => {
      const record = asRecord(row);
      const metric = document.createElement("dt");
      metric.textContent = String(record.metric || "");
      const value = document.createElement("dd");
      value.textContent = String(record.value || "");
      list.appendChild(metric);
      list.appendChild(value);
    });
    metricsNode.appendChild(list);
  };

  const renderMarkdownTable = (tableMarkdown) => {
    const normalized = String(tableMarkdown || "").trim();
    markdownNode.innerHTML = "";
    if (normalized.length === 0) {
      const emptyNode = document.createElement("p");
      emptyNode.className = "muted-text";
      emptyNode.textContent = "No table_md returned.";
      markdownNode.appendChild(emptyNode);
      return;
    }

    const content = document.createElement("div");
    content.className = "markdown-report";
    content.innerHTML = renderMarkdownToSafeHtml(normalized);
    markdownNode.appendChild(content);
  };

  const renderTrades = (trades) => {
    tradesNode.innerHTML = "";
    if (!Array.isArray(trades) || trades.length === 0) {
      const emptyNode = document.createElement("p");
      emptyNode.className = "muted-text";
      emptyNode.textContent = includeTradesToggle.checked
        ? "No trades returned for this variant."
        : "Trades disabled. Enable include_trades to inspect the trade list.";
      tradesNode.appendChild(emptyNode);
      return;
    }

    const tableScroll = document.createElement("div");
    tableScroll.className = "table-scroll";
    const table = document.createElement("table");
    table.className = "data-table detail-trades-table";
    table.innerHTML = [
      "<thead>",
      "<tr>",
      "<th scope=\"col\">trade_id</th>",
      "<th scope=\"col\">direction</th>",
      "<th scope=\"col\">entry_bar_index</th>",
      "<th scope=\"col\">exit_bar_index</th>",
      "<th scope=\"col\">entry_fill_price</th>",
      "<th scope=\"col\">exit_fill_price</th>",
      "<th scope=\"col\">net_pnl_quote</th>",
      "<th scope=\"col\">exit_reason</th>",
      "</tr>",
      "</thead>",
      "<tbody></tbody>",
    ].join("");
    const tbody = table.querySelector("tbody");
    trades.forEach((trade) => {
      const record = asRecord(trade);
      const row = document.createElement("tr");
      row.appendChild(buildCell(formatValue(record.trade_id)));
      row.appendChild(buildCell(formatValue(record.direction)));
      row.appendChild(buildCell(formatValue(record.entry_bar_index)));
      row.appendChild(buildCell(formatValue(record.exit_bar_index)));
      row.appendChild(buildCell(formatValue(record.entry_fill_price)));
      row.appendChild(buildCell(formatValue(record.exit_fill_price)));
      row.appendChild(buildCell(formatValue(record.net_pnl_quote)));
      row.appendChild(buildCell(formatValue(record.exit_reason)));
      tbody?.appendChild(row);
    });
    tableScroll.appendChild(table);
    tradesNode.appendChild(tableScroll);
  };

  const renderTradesChart = (trades) => {
    chartNode.innerHTML = "";
    chartNode.appendChild(buildTradesChartNode({ trades, includeTrades: includeTradesToggle.checked }));
  };

  const renderReport = (rawReport) => {
    const report = asRecord(rawReport);
    renderMetricRows(Array.isArray(report.rows) ? report.rows : []);
    renderMarkdownTable(report.table_md);
    renderTradesChart(Array.isArray(report.trades) ? report.trades : []);
    renderTrades(Array.isArray(report.trades) ? report.trades : []);
  };

  const resolveSelectedRow = () => {
    const matchedRow = state.topRows.find((item) => String(asRecord(item).variant_key || "") === variantKey);
    state.selectedRow = matchedRow || null;
    if (matchedRow === null) {
      renderMissingVariant(
        `variant_key ${variantKey} was not found in persisted summary rows for run ${runId}.`,
      );
      return false;
    }

    clearMissingVariant();
    renderSelectedVariantSummary();
    return true;
  };

  const loadStatus = async () => {
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
      state.topRows = Array.isArray(payload.items) ? payload.items.slice() : [];
      resolveSelectedRow();
      return payload;
    } catch (error) {
      if (token !== state.topRequestToken) {
        return null;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      renderMissingVariant("Failed to load persisted summary rows for this run.");
      return null;
    }
  };

  const currentReportCacheKey = () => (
    includeTradesToggle.checked ? "include_trades:true" : "include_trades:false"
  );

  const loadVariantReport = async ({ forceReload = false } = {}) => {
    clearPageError(pageRoot);
    if (state.selectedRow === null) {
      renderMissingVariant(
        `variant_key ${variantKey} was not found in persisted summary rows for run ${runId}.`,
      );
      return;
    }

    const cacheKey = currentReportCacheKey();
    if (!forceReload && state.reportCacheByKey.has(cacheKey)) {
      renderReport(state.reportCacheByKey.get(cacheKey));
      return;
    }

    setLoadingState(true);
    try {
      const requestPayload = {
        variant: cloneJsonValue(asRecord(asRecord(state.selectedRow).payload)),
        include_trades: includeTradesToggle.checked,
      };
      const response = await fetch(variantReportPath, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      state.reportCacheByKey.set(cacheKey, payload);
      renderReport(payload);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    } finally {
      setLoadingState(false);
    }
  };

  const saveSelectedVariant = async () => {
    clearPageError(pageRoot);
    if (state.selectedRow === null) {
      renderMissingVariant(
        `variant_key ${variantKey} was not found in persisted summary rows for run ${runId}.`,
      );
      return;
    }

    try {
      const runContext = await resolveRunStrategyContext(state.status);
      const prefillPayload = buildStrategyPrefillPayload({
        variantPayload: asRecord(state.selectedRow).payload,
        runContext,
      });
      persistStrategyPrefillAndNavigate({
        pageRoot,
        strategyBuilderPath,
        prefillQueryParam,
        prefillStorage,
        prefillPayload,
      });
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  refreshButton.addEventListener("click", async () => {
    await loadVariantReport({ forceReload: true });
  });

  includeTradesToggle.addEventListener("change", async () => {
    await loadVariantReport();
  });

  saveButton.addEventListener("click", async () => {
    await saveSelectedVariant();
  });

  const bootstrap = async () => {
    clearPageError(pageRoot);
    await loadStatus();
    const topPayload = await loadTop();
    if (topPayload !== null && state.selectedRow !== null) {
      await loadVariantReport();
    }
  };

  setLoadingState(false);
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

function renderTopHead(tableHead, visibleColumns, { includeActions = false } = {}) {
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

  if (includeActions) {
    const actionsHeader = document.createElement("th");
    actionsHeader.scope = "col";
    actionsHeader.textContent = "actions";
    row.appendChild(actionsHeader);
  }
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

function readFiniteInteger(value) {
  const numberValue = Number(value);
  return Number.isInteger(numberValue) ? numberValue : null;
}

function clampProgressPercent(value) {
  const numberValue = readFiniteNumber(value);
  if (numberValue === null) {
    return 0;
  }
  return Math.min(Math.max(Math.round(numberValue), 0), 100);
}

function readProgressPercent(status, { processedUnits, totalUnits }) {
  const explicitPercent = readFiniteNumber(asRecord(status).progress_percent);
  if (explicitPercent !== null) {
    return clampProgressPercent(explicitPercent);
  }
  if (totalUnits <= 0) {
    return 0;
  }
  return clampProgressPercent((processedUnits / totalUnits) * 100);
}

function formatPercentValue(value) {
  const normalizedValue = readFiniteNumber(value);
  if (normalizedValue === null) {
    return "-";
  }
  return `${clampProgressPercent(normalizedValue)}%`;
}

function formatEtaSeconds(value) {
  const totalSeconds = readFiniteInteger(value);
  if (totalSeconds === null || totalSeconds < 0) {
    return "-";
  }
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0 && seconds === 0) {
    return `${minutes}m`;
  }
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function buildHistoryExecutionModeSummary(record) {
  const executionMode = formatValue(record.execution_mode);
  const executionProfileMode = formatValue(record.execution_profile_mode);
  if (executionProfileMode === "-") {
    return executionMode;
  }
  return `${executionMode} / ${executionProfileMode}`;
}

function buildHistoryProgressSummary(record) {
  const processedUnits = Number(record.processed_units || 0);
  const totalUnits = Number(record.total_units || 0);
  const progressPercent = readProgressPercent(record, {
    processedUnits,
    totalUnits,
  });
  const etaSeconds = readFiniteInteger(record.eta_seconds);
  return [
    formatValue(record.stage),
    `${progressPercent}%`,
    `${formatValue(record.processed_units)}/${formatValue(record.total_units)}`,
    etaSeconds === null ? null : `ETA ~${formatEtaSeconds(etaSeconds)}`,
  ].filter((item) => item !== null).join(" | ");
}

function formatValue(value) {
  if (value === null || typeof value === "undefined" || String(value).trim().length === 0) {
    return "-";
  }
  return String(value);
}

function buildVariantDetailPath({ pathTemplate, runId, variantKey }) {
  return String(pathTemplate || "")
    .replace("{run_id}", encodeURIComponent(runId))
    .replace("{variant_key}", encodeURIComponent(variantKey));
}

function createRunStrategyContextResolver({ marketsPath }) {
  const state = {
    marketCatalogPromise: null,
    marketsById: new Map(),
  };

  const loadMarketsById = async () => {
    if (state.marketCatalogPromise !== null) {
      return state.marketCatalogPromise;
    }

    state.marketCatalogPromise = fetch(new URL(marketsPath, window.location.origin).toString(), {
      credentials: "include",
    }).then(async (response) => {
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      const markets = Array.isArray(payload) ? payload : [];
      state.marketsById = new Map(
        markets
          .map((item) => asRecord(item))
          .filter((item) => Number(item.market_id || 0) > 0)
          .map((item) => [Number(item.market_id), item]),
      );
      return state.marketsById;
    }).catch((error) => {
      state.marketCatalogPromise = null;
      throw error;
    });

    return state.marketCatalogPromise;
  };

  return async (rawStatus) => {
    const status = asRecord(rawStatus);
    const marketId = Number(status.market_id || 0);
    const symbol = String(status.symbol || "").trim();
    const timeframe = String(status.timeframe || "").trim();
    if (marketId <= 0 || symbol.length === 0 || timeframe.length === 0) {
      throw new Error("Run status does not contain instrument_id/timeframe for prefill.");
    }

    const marketsById = await loadMarketsById();
    const market = asRecord(marketsById.get(marketId));
    const marketType = String(market.market_type || "").trim();
    const marketCode = String(market.market_code || "").trim();
    if (marketType.length === 0 || marketCode.length === 0) {
      throw new Error(`Market metadata for market_id=${marketId} is unavailable for prefill.`);
    }

    return {
      instrument_id: {
        market_id: marketId,
        symbol,
      },
      timeframe,
      market_type: marketType,
      instrument_key: `${marketCode}:${marketType}:${symbol}`,
    };
  };
}

function cloneJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => cloneJsonValue(item));
  }
  if (value !== null && typeof value === "object") {
    const record = asRecord(value);
    const cloned = {};
    Object.keys(record).sort(compareStableStrings).forEach((key) => {
      cloned[key] = cloneJsonValue(record[key]);
    });
    return cloned;
  }
  return value;
}

function buildTradesChartNode({ trades, includeTrades }) {
  const wrapper = document.createElement("div");
  wrapper.className = "variant-chart-shell";

  if (!includeTrades) {
    const emptyNode = document.createElement("p");
    emptyNode.className = "muted-text";
    emptyNode.textContent = "Chart is disabled while include_trades=false.";
    wrapper.appendChild(emptyNode);
    return wrapper;
  }
  if (!Array.isArray(trades) || trades.length === 0) {
    const emptyNode = document.createElement("p");
    emptyNode.className = "muted-text";
    emptyNode.textContent = "Trade chart is unavailable because the detail payload returned no trades.";
    wrapper.appendChild(emptyNode);
    return wrapper;
  }

  const chartWidth = 760;
  const chartHeight = 240;
  const paddingX = 36;
  const paddingY = 20;
  const points = [{ x: 0, y: 0, tradeId: 0, exitReason: "start" }];
  let cumulativePnl = 0;
  let maxX = 1;
  let minY = 0;
  let maxY = 0;

  trades.forEach((trade) => {
    const record = asRecord(trade);
    const exitBarIndex = Number(record.exit_bar_index || 0);
    const netPnl = Number(record.net_pnl_quote || 0);
    cumulativePnl += Number.isFinite(netPnl) ? netPnl : 0;
    maxX = Math.max(maxX, exitBarIndex);
    minY = Math.min(minY, cumulativePnl);
    maxY = Math.max(maxY, cumulativePnl);
    points.push({
      x: exitBarIndex,
      y: cumulativePnl,
      tradeId: Number(record.trade_id || 0),
      exitReason: String(record.exit_reason || "").trim().toLowerCase(),
    });
  });

  const normalizeX = (value) => {
    const width = chartWidth - (paddingX * 2);
    return paddingX + ((value / Math.max(maxX, 1)) * width);
  };
  const normalizeY = (value) => {
    const height = chartHeight - (paddingY * 2);
    const range = Math.max(maxY - minY, 1);
    return chartHeight - paddingY - (((value - minY) / range) * height);
  };

  const svgNamespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNamespace, "svg");
  svg.setAttribute("viewBox", `0 0 ${chartWidth} ${chartHeight}`);
  svg.setAttribute("class", "variant-chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Trade equity chart by exit_bar_index");

  const horizontalStops = [minY, (minY + maxY) / 2, maxY];
  horizontalStops.forEach((value) => {
    const line = document.createElementNS(svgNamespace, "line");
    const y = normalizeY(value);
    line.setAttribute("x1", String(paddingX));
    line.setAttribute("x2", String(chartWidth - paddingX));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("class", "variant-chart-grid");
    svg.appendChild(line);
  });

  const zeroLine = document.createElementNS(svgNamespace, "line");
  zeroLine.setAttribute("x1", String(paddingX));
  zeroLine.setAttribute("x2", String(chartWidth - paddingX));
  zeroLine.setAttribute("y1", String(normalizeY(0)));
  zeroLine.setAttribute("y2", String(normalizeY(0)));
  zeroLine.setAttribute("class", "variant-chart-zero");
  svg.appendChild(zeroLine);

  const path = document.createElementNS(svgNamespace, "polyline");
  path.setAttribute(
    "points",
    points.map((point) => `${normalizeX(point.x)},${normalizeY(point.y)}`).join(" "),
  );
  path.setAttribute("class", "variant-chart-line");
  svg.appendChild(path);

  points.slice(1).forEach((point) => {
    const marker = document.createElementNS(svgNamespace, "circle");
    marker.setAttribute("cx", String(normalizeX(point.x)));
    marker.setAttribute("cy", String(normalizeY(point.y)));
    marker.setAttribute("r", "4");
    marker.setAttribute("class", "variant-chart-point");
    marker.setAttribute(
      "data-exit-reason",
      colorKeyForExitReason(point.exitReason),
    );
    const title = document.createElementNS(svgNamespace, "title");
    title.textContent = `trade_id=${point.tradeId} exit_reason=${point.exitReason} cumulative_net=${point.y.toFixed(2)}`;
    marker.appendChild(title);
    svg.appendChild(marker);
  });

  wrapper.appendChild(svg);
  wrapper.appendChild(buildExitReasonLegend(trades));
  return wrapper;
}

function buildExitReasonLegend(trades) {
  const legend = document.createElement("div");
  legend.className = "variant-chart-legend";
  const counts = new Map();
  trades.forEach((trade) => {
    const reason = colorKeyForExitReason(String(asRecord(trade).exit_reason || ""));
    counts.set(reason, (counts.get(reason) || 0) + 1);
  });

  Array.from(counts.keys()).sort(compareStableStrings).forEach((reason) => {
    const item = document.createElement("span");
    item.className = "variant-chart-legend-item";
    const marker = document.createElement("span");
    marker.className = "variant-chart-legend-marker";
    marker.dataset.exitReason = reason;
    const label = document.createElement("span");
    label.textContent = `${reason} (${counts.get(reason)})`;
    item.appendChild(marker);
    item.appendChild(label);
    legend.appendChild(item);
  });

  return legend;
}

function colorKeyForExitReason(rawReason) {
  const normalized = String(rawReason || "").trim().toLowerCase();
  if (normalized === "tp") {
    return "tp";
  }
  if (normalized === "sl") {
    return "sl";
  }
  if (normalized === "close_on_end") {
    return "close_on_end";
  }
  if (normalized === "signal_exit") {
    return "signal_exit";
  }
  return normalized.length > 0 ? normalized : "other";
}
