const BACKTEST_JOBS_PAGE_SELECTOR = "[data-backtest-jobs-page]";
const ACTIVE_JOB_STATES = new Set(["queued", "running"]);
const TERMINAL_JOB_STATES = new Set(["succeeded", "failed", "cancelled"]);
const STATUS_POLL_INTERVAL_MS = 2000;
const TOP_POLL_INTERVAL_MS = 3000;
const JOB_CONTEXT_STORAGE_PREFIX = "backtest-job-context:";



document.addEventListener("DOMContentLoaded", () => {
  const pageRoot = document.querySelector(BACKTEST_JOBS_PAGE_SELECTOR);
  if (pageRoot === null) {
    return;
  }

  const pageType = String(pageRoot.dataset.backtestJobsPage || "").trim().toLowerCase();
  if (pageType === "list") {
    initJobsListPage(pageRoot);
    return;
  }
  if (pageType === "details") {
    initJobDetailsPage(pageRoot);
  }
});

function initJobsListPage(pageRoot) {
  const jobsPath = requireDataAttr(pageRoot, "apiJobsPath");
  const detailsPathTemplate = requireDataAttr(pageRoot, "detailsPathTemplate");
  const defaultLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultLimit"), 50);

  const stateFilter = pageRoot.querySelector("#jobs-list-state");
  const limitSelect = pageRoot.querySelector("#jobs-list-limit");
  const cursorValue = pageRoot.querySelector("#jobs-list-cursor-value");
  const cursorNote = pageRoot.querySelector("#jobs-list-cursor-note");
  const refreshButton = pageRoot.querySelector("#jobs-list-refresh");
  const prevButton = pageRoot.querySelector("#jobs-list-prev");
  const nextButton = pageRoot.querySelector("#jobs-list-next");
  const tableBody = pageRoot.querySelector("#jobs-list-table-body");
  const disabledBanner = pageRoot.querySelector("#jobs-list-disabled-banner");

  if (
    stateFilter === null
    || limitSelect === null
    || cursorValue === null
    || cursorNote === null
    || refreshButton === null
    || prevButton === null
    || nextButton === null
    || tableBody === null
    || disabledBanner === null
  ) {
    return;
  }

  limitSelect.value = String(defaultLimit);

  const state = {
    pageCursors: [""],
    pageIndex: 0,
    nextCursor: null,
    jobsDisabled: false,
    isLoading: false,
    requestToken: 0,
  };

  const currentCursor = () => String(state.pageCursors[state.pageIndex] || "");

  const setControlsDisabled = (disabled) => {
    stateFilter.disabled = disabled;
    limitSelect.disabled = disabled;
    refreshButton.disabled = disabled;
    prevButton.disabled = disabled;
    nextButton.disabled = disabled;
    cursorValue.disabled = disabled;
  };

  const updatePagerControls = () => {
    if (state.jobsDisabled) {
      setControlsDisabled(true);
      return;
    }

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
    if (state.nextCursor === null) {
      cursorNote.textContent = "next_cursor: none";
      return;
    }
    cursorNote.textContent = `next_cursor: ${state.nextCursor}`;
  };

  const renderRows = (items) => {
    tableBody.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
      tableBody.innerHTML = "<tr><td colspan=\"7\">No jobs found.</td></tr>";
      return;
    }

    items.forEach((item) => {
      const record = asRecord(item);
      const jobId = String(record.job_id || "").trim();
      const row = document.createElement("tr");
      if (jobId.length > 0) {
        row.dataset.jobId = jobId;
        row.classList.add("row-clickable");
      }
      row.appendChild(buildCell(jobId));
      row.appendChild(buildCell(String(record.mode || "")));
      row.appendChild(buildCell(String(record.state || "")));
      row.appendChild(buildCell(String(record.stage || "")));
      row.appendChild(buildCell(String(record.created_at || "")));
      row.appendChild(buildCell(String(record.updated_at || "")));
      row.appendChild(
        buildCell(
          `${String(record.processed_units ?? "")}/${String(record.total_units ?? "")}`,
        ),
      );
      tableBody.appendChild(row);
    });
  };

  const markJobsDisabled = () => {
    state.jobsDisabled = true;
    state.nextCursor = null;
    disabledBanner.classList.remove("hidden");
    setControlsDisabled(true);
    showPageError(pageRoot, "Jobs disabled by config", []);
    tableBody.innerHTML = "<tr><td colspan=\"7\">Jobs disabled by config</td></tr>";
    updateCursorUi();
  };

  const loadJobs = async () => {
    if (state.jobsDisabled) {
      return;
    }

    const token = state.requestToken + 1;
    state.requestToken = token;
    state.isLoading = true;
    updatePagerControls();
    updateCursorUi();
    clearPageError(pageRoot);

    if (tableBody.children.length === 0) {
      tableBody.innerHTML = "<tr><td colspan=\"7\">Loading jobs...</td></tr>";
    }

    try {
      const requestUrl = new URL(jobsPath, window.location.origin);
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
        credentials: 'include',
      });
      if (response.status === 404) {
        markJobsDisabled();
        return;
      }
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
      tableBody.innerHTML = "<tr><td colspan=\"7\">Failed to load jobs.</td></tr>";
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
    await loadJobs();
  });

  limitSelect.addEventListener("change", async () => {
    const parsedLimit = parsePositiveInt(String(limitSelect.value || ""), defaultLimit);
    limitSelect.value = String(parsedLimit);
    resetPagination();
    await loadJobs();
  });

  refreshButton.addEventListener("click", async () => {
    await loadJobs();
  });

  prevButton.addEventListener("click", async () => {
    if (state.pageIndex <= 0) {
      return;
    }
    state.pageIndex -= 1;
    await loadJobs();
  });

  nextButton.addEventListener("click", async () => {
    if (state.nextCursor === null) {
      return;
    }

    if (state.pageIndex === state.pageCursors.length - 1) {
      state.pageCursors.push(state.nextCursor);
    }
    state.pageIndex += 1;
    await loadJobs();
  });

  tableBody.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const row = target.closest("tr[data-job-id]");
    if (row === null) {
      return;
    }

    const jobId = String(row.getAttribute("data-job-id") || "").trim();
    if (jobId.length === 0) {
      return;
    }

    const targetPath = renderPathTemplate(detailsPathTemplate, encodeURIComponent(jobId));
    window.location.assign(targetPath);
  });

  updateCursorUi();
  updatePagerControls();
  loadJobs();
}

function initJobDetailsPage(pageRoot) {
  const jobId = requireDataAttr(pageRoot, "jobId");
  const jobsPathPrefix = requireDataAttr(pageRoot, "apiJobsPathPrefix");
  const topPathTemplate = requireDataAttr(pageRoot, "apiTopPathTemplate");
  const variantReportPath = requireDataAttr(pageRoot, "apiVariantReportPath");
  const cancelPathTemplate = requireDataAttr(pageRoot, "apiCancelPathTemplate");
  const strategyBuilderPath = requireDataAttr(pageRoot, "strategyBuilderPath");
  const prefillQueryParam = requireDataAttr(pageRoot, "prefillQueryParam");
  const prefillStorage = requireDataAttr(pageRoot, "prefillStorage");
  const defaultTopLimit = parsePositiveInt(requireDataAttr(pageRoot, "defaultTopLimit"), 50);

  const copyIdButton = pageRoot.querySelector("#job-copy-id");
  const refreshStatusButton = pageRoot.querySelector("#job-refresh-status");
  const cancelButton = pageRoot.querySelector("#job-cancel");
  const refreshTopButton = pageRoot.querySelector("#job-refresh-top");
  const topLimitSelect = pageRoot.querySelector("#job-top-limit");
  const topTableBody = pageRoot.querySelector("#job-top-table-body");
  const progressBar = pageRoot.querySelector("#job-progress-bar");
  const progressCaption = pageRoot.querySelector("#job-progress-caption");
  const failedBlock = pageRoot.querySelector("#job-failed-block");
  const failedError = pageRoot.querySelector("#job-failed-error");
  const failedErrorJson = pageRoot.querySelector("#job-failed-error-json");
  const disabledBanner = pageRoot.querySelector("#job-details-disabled-banner");

  if (
    copyIdButton === null
    || refreshStatusButton === null
    || cancelButton === null
    || refreshTopButton === null
    || topLimitSelect === null
    || topTableBody === null
    || progressBar === null
    || progressCaption === null
    || failedBlock === null
    || failedError === null
    || failedErrorJson === null
    || disabledBanner === null
  ) {
    return;
  }

  const fieldMap = {
    mode: pageRoot.querySelector("#job-field-mode"),
    state: pageRoot.querySelector("#job-field-state"),
    stage: pageRoot.querySelector("#job-field-stage"),
    createdAt: pageRoot.querySelector("#job-field-created-at"),
    updatedAt: pageRoot.querySelector("#job-field-updated-at"),
    startedAt: pageRoot.querySelector("#job-field-started-at"),
    finishedAt: pageRoot.querySelector("#job-field-finished-at"),
    cancelRequestedAt: pageRoot.querySelector("#job-field-cancel-requested-at"),
    progressUpdatedAt: pageRoot.querySelector("#job-field-progress-updated-at"),
    progress: pageRoot.querySelector("#job-field-progress"),
    requestHash: pageRoot.querySelector("#job-field-request-hash"),
    engineParamsHash: pageRoot.querySelector("#job-field-engine-params-hash"),
    runtimeConfigHash: pageRoot.querySelector("#job-field-runtime-config-hash"),
    specHash: pageRoot.querySelector("#job-field-spec-hash"),
  };

  if (Object.values(fieldMap).some((node) => node === null)) {
    return;
  }

  const state = {
    status: null,
    topRows: [],
    topReportContext: null,
    topLimit: defaultTopLimit,
    jobsDisabled: false,
    finalRefreshDone: false,
    statusTimerId: 0,
    topTimerId: 0,
    statusRequestToken: 0,
    topRequestToken: 0,
    reportCacheByVariantKey: new Map(),
    reportLoadingKeys: new Set(),
    reportErrorsByVariantKey: new Map(),
    reportCacheHitKeys: new Set(),
  };

  const statusPath = `${jobsPathPrefix}${encodeURIComponent(jobId)}`;
  const renderTopPath = () => {
    const templatePath = renderPathTemplate(topPathTemplate, encodeURIComponent(jobId));
    const requestUrl = new URL(templatePath, window.location.origin);
    requestUrl.searchParams.set("limit", String(state.topLimit));
    return requestUrl.toString();
  };
  const cancelPath = renderPathTemplate(cancelPathTemplate, encodeURIComponent(jobId));

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
    if (state.jobsDisabled) {
      return;
    }
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

  const setControlsDisabled = (disabled) => {
    refreshStatusButton.disabled = disabled;
    refreshTopButton.disabled = disabled;
    topLimitSelect.disabled = disabled;
    cancelButton.disabled = disabled;
  };

  const markJobsDisabled = () => {
    if (state.jobsDisabled) {
      return;
    }
    state.jobsDisabled = true;
    stopPolling();
    setControlsDisabled(true);
    disabledBanner.classList.remove("hidden");
    showPageError(pageRoot, "Jobs disabled by config", []);
  };

  const updateButtonsForState = () => {
    const statusState = String(asRecord(state.status).state || "").trim().toLowerCase();
    if (state.jobsDisabled) {
      cancelButton.disabled = true;
      refreshStatusButton.disabled = true;
      refreshTopButton.disabled = true;
      topLimitSelect.disabled = true;
      return;
    }

    refreshStatusButton.disabled = false;
    refreshTopButton.disabled = false;
    topLimitSelect.disabled = false;

    const terminal = TERMINAL_JOB_STATES.has(statusState);
    cancelButton.disabled = terminal || statusState.length === 0;
  };

  const renderStatus = (rawStatus) => {
    const status = asRecord(rawStatus);
    state.status = status;

    const processedUnits = Number(status.processed_units || 0);
    const totalUnits = Number(status.total_units || 0);
    const ratio = totalUnits > 0 ? Math.min(Math.max(processedUnits / totalUnits, 0), 1) : 0;

    fieldMap.mode.textContent = String(status.mode || "");
    fieldMap.state.textContent = String(status.state || "");
    fieldMap.stage.textContent = String(status.stage || "");
    fieldMap.createdAt.textContent = String(status.created_at || "");
    fieldMap.updatedAt.textContent = String(status.updated_at || "");
    fieldMap.startedAt.textContent = String(status.started_at || "-");
    fieldMap.finishedAt.textContent = String(status.finished_at || "-");
    fieldMap.cancelRequestedAt.textContent = String(status.cancel_requested_at || "-");
    fieldMap.progressUpdatedAt.textContent = String(status.progress_updated_at || "-");
    fieldMap.progress.textContent = `${processedUnits}/${totalUnits}`;
    fieldMap.requestHash.textContent = String(status.request_hash || "");
    fieldMap.engineParamsHash.textContent = String(status.engine_params_hash || "");
    fieldMap.runtimeConfigHash.textContent = String(status.backtest_runtime_config_hash || "");
    fieldMap.specHash.textContent = String(status.spec_hash || "-");

    progressBar.style.width = `${Math.round(ratio * 100)}%`;
    progressCaption.textContent = `Progress: ${Math.round(ratio * 100)}% (${processedUnits}/${totalUnits})`;

    const stateValue = String(status.state || "").trim().toLowerCase();
    if (stateValue === "failed") {
      failedBlock.classList.remove("hidden");
      failedError.textContent = String(status.last_error || "Unknown failed reason.");
      const errorJson = asRecord(status.last_error_json);
      failedErrorJson.textContent = JSON.stringify(errorJson, null, 2);
    } else {
      failedBlock.classList.add("hidden");
      failedError.textContent = "";
      failedErrorJson.textContent = "";
    }

    updateButtonsForState();
  };

  const renderVariantReport = (report) => {
    const reportNode = document.createElement("div");
    if (!report || typeof report !== "object") {
      reportNode.textContent = "No report.";
      return reportNode;
    }

    const rows = Array.isArray(report.rows) ? report.rows : [];
    if (rows.length > 0) {
      const list = document.createElement("ul");
      list.className = "compact-list";
      rows.forEach((row) => {
        const item = document.createElement("li");
        const rowRecord = asRecord(row);
        item.textContent = `${String(rowRecord.metric || "")}: ${String(rowRecord.value || "")}`;
        list.appendChild(item);
      });
      reportNode.appendChild(list);
    }

    const tableMarkdown = String(report.table_md || "").trim();
    if (tableMarkdown.length > 0) {
      const tableDetails = document.createElement("details");
      tableDetails.className = "panel panel--soft";
      const summary = document.createElement("summary");
      summary.textContent = "table_md";
      tableDetails.appendChild(summary);
      const content = document.createElement("div");
      content.className = "markdown-report";
      content.innerHTML = renderMarkdownToSafeHtml(tableMarkdown);
      tableDetails.appendChild(content);
      reportNode.appendChild(tableDetails);
    }

    const trades = Array.isArray(report.trades) ? report.trades : [];
    if (trades.length > 0) {
      const tradesDetails = document.createElement("details");
      tradesDetails.className = "panel panel--soft";
      const summary = document.createElement("summary");
      summary.textContent = `trades (${trades.length})`;
      tradesDetails.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "json-pre";
      pre.textContent = JSON.stringify(trades, null, 2);
      tradesDetails.appendChild(pre);
      reportNode.appendChild(tradesDetails);
    }

    if (reportNode.childElementCount === 0) {
      reportNode.textContent = "Report is empty.";
    }
    return reportNode;
  };

  const readVariantReportStateByKey = (variantKey) => ({
    isLoading: state.reportLoadingKeys.has(variantKey),
    report: state.reportCacheByVariantKey.get(variantKey) || null,
    error: state.reportErrorsByVariantKey.get(variantKey) || null,
    cacheHit: state.reportCacheHitKeys.has(variantKey),
  });

  const hasVariantReportContext = () => (
    state.topReportContext !== null
    && typeof state.topReportContext === "object"
    && Object.keys(asRecord(state.topReportContext)).length > 0
  );

  const buildVariantReportRequestPayload = ({ variantRecord }) => {
    const context = asRecord(state.topReportContext);
    if (Object.keys(context).length === 0) {
      throw new Error("Variant report context is unavailable for this job.");
    }

    const variantPayload = asRecord(variantRecord.payload);
    if (Object.keys(variantPayload).length === 0) {
      throw new Error("Variant payload is unavailable.");
    }

    const payload = {
      time_range: normalizeJsonLikeValue(asRecord(context.time_range)),
      variant: normalizeJsonLikeValue(variantPayload),
      include_trades: Boolean(context.include_trades),
    };

    const strategyId = String(context.strategy_id || "").trim();
    const templatePayload = asRecord(context.template);
    const hasTemplatePayload = Object.keys(templatePayload).length > 0;
    if (strategyId.length > 0 && hasTemplatePayload) {
      throw new Error("Report request mode conflict: both strategy_id and template are set.");
    }
    if (strategyId.length === 0 && !hasTemplatePayload) {
      throw new Error("Report request mode is missing: strategy_id or template is required.");
    }
    if (strategyId.length > 0) {
      payload.strategy_id = strategyId;
    }
    if (hasTemplatePayload) {
      payload.template = normalizeJsonLikeValue(templatePayload);
    }

    const overridesPayload = asRecord(context.overrides);
    if (Object.keys(overridesPayload).length > 0) {
      payload.overrides = normalizeJsonLikeValue(overridesPayload);
    }

    const warmupBars = parsePositiveInt(String(context.warmup_bars || ""), 0);
    if (warmupBars > 0) {
      payload.warmup_bars = warmupBars;
    }
    return payload;
  };

  const loadVariantReport = async (rawRow) => {
    const rowData = asRecord(rawRow);
    const variantKey = String(rowData.variant_key || "").trim();
    if (variantKey.length === 0) {
      showPageError(pageRoot, "variant_key is required for report loading.", []);
      return;
    }

    if (!hasVariantReportContext()) {
      state.reportErrorsByVariantKey.set(variantKey, {
        message: "Variant report context is unavailable for this job.",
        details: [],
      });
      renderTopRows();
      return;
    }

    if (state.reportLoadingKeys.has(variantKey)) {
      return;
    }
    if (state.reportCacheByVariantKey.has(variantKey)) {
      state.reportErrorsByVariantKey.delete(variantKey);
      state.reportCacheHitKeys.add(variantKey);
      renderTopRows();
      return;
    }

    state.reportLoadingKeys.add(variantKey);
    state.reportErrorsByVariantKey.delete(variantKey);
    state.reportCacheHitKeys.delete(variantKey);
    renderTopRows();

    try {
      const reportRequestPayload = buildVariantReportRequestPayload({ variantRecord: rowData });
      const reportResponse = await fetch(variantReportPath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reportRequestPayload),
      });
      if (!reportResponse.ok) {
        throw await buildHttpError(reportResponse);
      }
      const reportPayload = asRecord(await reportResponse.json());
      state.reportCacheByVariantKey.set(variantKey, reportPayload);
    } catch (error) {
      const normalized = normalizeError(error);
      state.reportErrorsByVariantKey.set(variantKey, normalized);
    } finally {
      state.reportLoadingKeys.delete(variantKey);
      renderTopRows();
    }
  };

  const renderTopRows = () => {
    topTableBody.innerHTML = "";
    if (state.topRows.length === 0) {
      topTableBody.innerHTML = "<tr><td colspan=\"6\">No top rows yet.</td></tr>";
      return;
    }

    state.topRows.forEach((rawRow) => {
      const rowData = asRecord(rawRow);
      const variantKey = String(rowData.variant_key || "").trim();
      const reportState = readVariantReportStateByKey(variantKey);
      const row = document.createElement("tr");

      row.appendChild(buildCell(String(rowData.rank ?? "")));
      row.appendChild(buildCell(String(rowData.total_return_pct ?? "")));
      row.appendChild(buildCell(variantKey));
      row.appendChild(buildCell(String(rowData.indicator_variant_key || "")));

      const reportCell = document.createElement("td");
      if (variantKey.length === 0) {
        reportCell.textContent = "variant_key is required for report loading.";
      } else if (!hasVariantReportContext()) {
        reportCell.textContent = "Report context is unavailable for this job.";
      } else if (reportState.isLoading) {
        reportCell.textContent = "Loading report...";
      } else if (reportState.error !== null) {
        reportCell.textContent = String(reportState.error.message || "Report load failed.");
        const errorDetails = Array.isArray(reportState.error.details) ? reportState.error.details : [];
        if (errorDetails.length > 0) {
          const detailsList = document.createElement("ul");
          detailsList.className = "compact-list";
          errorDetails.forEach((detail) => {
            const item = document.createElement("li");
            item.textContent = String(detail);
            detailsList.appendChild(item);
          });
          reportCell.appendChild(detailsList);
        }
      } else if (reportState.report !== null) {
        const cacheLabel = document.createElement("p");
        cacheLabel.className = "muted-text";
        cacheLabel.textContent = reportState.cacheHit
          ? "Loaded from cache by variant_key."
          : "Cached by variant_key.";
        reportCell.appendChild(cacheLabel);
        reportCell.appendChild(renderVariantReport(reportState.report));
      } else {
        reportCell.textContent = "Not loaded. Use Load report action.";
      }
      row.appendChild(reportCell);

      const actionsCell = document.createElement("td");
      const loadReportButton = buildActionButton({
        label: "Load report",
        disabled: variantKey.length === 0 || reportState.isLoading || !hasVariantReportContext(),
        onClick: async () => {
          await loadVariantReport(rowData);
        },
      });
      actionsCell.appendChild(loadReportButton);
      const saveButton = buildActionButton({
        label: "Save as Strategy",
        className: "button-link--secondary",
        onClick: () => {
          saveTopRowAsStrategy(rowData);
        },
      });
      actionsCell.appendChild(saveButton);
      row.appendChild(actionsCell);

      topTableBody.appendChild(row);
    });
  };

  const buildPrefillPayloadFromTopRow = (rawRow) => {
    const row = asRecord(rawRow);
    const payload = asRecord(row.payload);
    const selections = Array.isArray(payload.indicator_selections)
      ? payload.indicator_selections
      : [];

    const indicators = selections.map((item) => {
      const selection = asRecord(item);
      return {
        id: String(selection.indicator_id || "").trim(),
        inputs: copyRecord(asRecord(selection.inputs)),
        params: copyRecord(asRecord(selection.params)),
      };
    }).filter((item) => item.id.length > 0);

    if (indicators.length === 0) {
      throw new Error("Variant payload does not include indicator_selections.");
    }

    const context = asRecord(readJobContext(jobId));
    const contextInstrumentId = asRecord(context.instrument_id);
    const payloadInstrumentId = asRecord(payload.instrument_id);

    const marketId = Number(
      contextInstrumentId.market_id || payloadInstrumentId.market_id || 0,
    );
    const symbol = String(
      contextInstrumentId.symbol || payloadInstrumentId.symbol || "",
    ).trim();
    const timeframe = String(context.timeframe || payload.timeframe || "").trim();
    const marketType = String(context.market_type || payload.market_type || "").trim();
    const instrumentKey = String(
      context.instrument_key || payload.instrument_key || "",
    ).trim();

    if (marketId <= 0 || symbol.length === 0 || timeframe.length === 0) {
      throw new Error(
        "Job context does not contain instrument_id/timeframe for prefill. Recreate from /backtests.",
      );
    }
    if (marketType.length === 0 || instrumentKey.length === 0) {
      throw new Error(
        "Job context does not contain market_type/instrument_key for prefill. Recreate from /backtests.",
      );
    }

    return {
      instrument_id: {
        market_id: marketId,
        symbol,
      },
      instrument_key: instrumentKey,
      market_type: marketType,
      timeframe,
      indicators,
    };
  };

  const saveTopRowAsStrategy = (row) => {
    clearPageError(pageRoot);

    if (prefillStorage !== "sessionStorage" || typeof window.sessionStorage === "undefined") {
      showPageError(pageRoot, "sessionStorage is unavailable in current browser.", []);
      return;
    }

    try {
      const prefillPayload = buildPrefillPayloadFromTopRow(row);
      const prefillId = [
        "prefill",
        Date.now().toString(36),
        Math.random().toString(36).slice(2, 10),
      ].join("-");
      window.sessionStorage.setItem(prefillId, JSON.stringify(prefillPayload));

      const targetUrl = new URL(strategyBuilderPath, window.location.origin);
      targetUrl.searchParams.set(prefillQueryParam, prefillId);
      window.location.assign(targetUrl.pathname + targetUrl.search);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
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
    if (state.jobsDisabled) {
      return null;
    }

    const token = state.statusRequestToken + 1;
    state.statusRequestToken = token;

    try {
      const response = await fetch(statusPath, {
        credentials: 'include',
      });
      if (response.status === 404) {
        markJobsDisabled();
        return null;
      }
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
    if (state.jobsDisabled) {
      return null;
    }

    const token = state.topRequestToken + 1;
    state.topRequestToken = token;

    try {
      const response = await fetch(renderTopPath(), {
        credentials: 'include',
      });
      if (response.status === 404) {
        markJobsDisabled();
        return null;
      }
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      if (token !== state.topRequestToken) {
        return null;
      }

      const items = Array.isArray(payload.items) ? payload.items : [];
      const reportContext = asRecord(payload.report_context);
      state.topRows = items.slice();
      state.topReportContext = Object.keys(reportContext).length > 0 ? reportContext : null;
      renderTopRows();
      return payload;
    } catch (error) {
      if (token !== state.topRequestToken) {
        return null;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      return null;
    }
  };

  const cancelJob = async () => {
    if (state.jobsDisabled) {
      return;
    }

    clearPageError(pageRoot);
    cancelButton.disabled = true;

    try {
      const response = await fetch(cancelPath, {
        method: "POST",
        credentials: 'include',
      });
      if (response.status === 404) {
        markJobsDisabled();
        return;
      }
      if (!response.ok) {
        throw await buildHttpError(response);
      }

      const payload = await response.json();
      renderStatus(payload);
      await loadTop();
      await handleStatusTransition();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    } finally {
      updateButtonsForState();
    }
  };

  const copyJobIdToClipboard = async () => {
    clearPageError(pageRoot);

    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(jobId);
        return;
      }

      const helper = document.createElement("textarea");
      helper.value = jobId;
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
    await copyJobIdToClipboard();
  });

  refreshStatusButton.addEventListener("click", async () => {
    clearPageError(pageRoot);
    await loadStatus();
  });

  refreshTopButton.addEventListener("click", async () => {
    clearPageError(pageRoot);
    await loadTop();
  });

  cancelButton.addEventListener("click", async () => {
    await cancelJob();
  });

  topLimitSelect.value = String(defaultTopLimit);
  topLimitSelect.addEventListener("change", async () => {
    const parsedLimit = parsePositiveInt(String(topLimitSelect.value || ""), defaultTopLimit);
    if (![10, 20, 50, 100].includes(parsedLimit)) {
      topLimitSelect.value = "50";
      state.topLimit = 50;
    } else {
      topLimitSelect.value = String(parsedLimit);
      state.topLimit = parsedLimit;
    }
    await loadTop();
  });

  const bootstrap = async () => {
    clearPageError(pageRoot);
    await loadStatus();
    if (state.topRows.length === 0) {
      await loadTop();
    }
  };

  bootstrap();
}

function readJobContext(jobId) {
  if (typeof window.sessionStorage === "undefined") {
    return null;
  }

  const storageKey = `${JOB_CONTEXT_STORAGE_PREFIX}${jobId}`;
  const rawPayload = window.sessionStorage.getItem(storageKey);
  if (typeof rawPayload !== "string" || rawPayload.length === 0) {
    return null;
  }

  try {
    return asRecord(JSON.parse(rawPayload));
  } catch (_error) {
    return null;
  }
}

function renderMarkdownToSafeHtml(markdown) {
  const content = String(markdown || "");
  if (content.length === 0) {
    return "";
  }

  let rendered = `<p>${escapeHtml(content)}</p>`;
  if (window.marked && typeof window.marked.parse === "function") {
    const renderer = new window.marked.Renderer();
    renderer.html = (token) => {
      if (token !== null && typeof token === "object") {
        if (typeof token.text === "string") {
          return escapeHtml(token.text);
        }
        if (typeof token.raw === "string") {
          return escapeHtml(token.raw);
        }
      }
      return escapeHtml(String(token || ""));
    };
    const parsed = window.marked.parse(content, {
      renderer,
      gfm: true,
      breaks: false,
      async: false,
    });
    rendered = typeof parsed === "string" ? parsed : String(parsed || "");
  }

  if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
    return window.DOMPurify.sanitize(rendered, {
      ALLOWED_TAGS: [
        "a",
        "blockquote",
        "br",
        "code",
        "em",
        "li",
        "ol",
        "p",
        "pre",
        "strong",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
      ],
      ALLOWED_ATTR: ["href", "title", "target", "rel"],
    });
  }
  return rendered;
}

function parsePositiveInt(rawValue, fallback) {
  const parsed = Number.parseInt(String(rawValue || "").trim(), 10);
  if (Number.isNaN(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function renderPathTemplate(pathTemplate, identifier) {
  return String(pathTemplate || "").replace("{job_id}", String(identifier || ""));
}

function buildHttpError(response) {
  return parseApiError(response).then((parsed) => {
    const error = new Error(parsed.message);
    error.details = parsed.details;
    return error;
  });
}

async function parseApiError(response) {
  let message = `Request failed with status ${response.status}`;
  let details = [];
  let payload = null;

  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }

  if (payload !== null) {
    const roehubError = parseRoehubErrorPayload(payload);
    if (roehubError !== null) {
      return roehubError;
    }

    const detail = payload.detail;
    if (typeof detail === "string" && detail.length > 0) {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = response.status === 422 ? "Validation error." : message;
      details = detail
        .map((item) => {
          const itemRecord = asRecord(item);
          if (typeof itemRecord.msg === "string" && itemRecord.msg.length > 0) {
            return itemRecord.msg;
          }
          return buildStableDetailString(item);
        })
        .filter((item) => item.length > 0);
    } else if (detail !== null && typeof detail === "object") {
      const detailRecord = asRecord(detail);
      if (typeof detailRecord.message === "string" && detailRecord.message.length > 0) {
        message = detailRecord.message;
      } else if (response.status === 422) {
        message = "Validation error.";
      }
      if (Array.isArray(detailRecord.errors)) {
        details = detailRecord.errors
          .map((item) => {
            if (typeof item === "string") {
              return item;
            }
            const itemRecord = asRecord(item);
            if (typeof itemRecord.message === "string" && itemRecord.message.length > 0) {
              return itemRecord.message;
            }
            return buildStableDetailString(item);
          })
          .filter((item) => item.length > 0);
      } else if (response.status === 422) {
        const fallbackDetails = [];
        if (typeof detailRecord.error === "string") {
          fallbackDetails.push(`error: ${detailRecord.error}`);
        }
        Object.keys(detailRecord).sort(compareStableStrings).forEach((key) => {
          if (key === "error" || key === "message") {
            return;
          }
          fallbackDetails.push(`${key}: ${buildStableDetailString(detailRecord[key])}`);
        });
        details = fallbackDetails;
      }
    }
  }

  return { message, details };
}

function parseRoehubErrorPayload(payload) {
  const payloadRecord = asRecord(payload);
  const errorRecord = asRecord(payloadRecord.error);
  if (Object.keys(errorRecord).length === 0) {
    return null;
  }

  const errorMessage = String(errorRecord.message || "").trim();
  return {
    message: errorMessage.length > 0 ? errorMessage : "Unexpected backtest operation error.",
    details: parseRoehubErrorDetails(errorRecord.details),
  };
}

function parseRoehubErrorDetails(rawDetails) {
  const detailsRecord = asRecord(rawDetails);
  if (Object.keys(detailsRecord).length === 0) {
    return [];
  }

  if (Array.isArray(detailsRecord.errors)) {
    return detailsRecord.errors
      .map((item) => formatValidationDetailItem(item))
      .filter((item) => item.length > 0)
      .sort(compareStableStrings);
  }

  if (typeof detailsRecord.reason === "string" && detailsRecord.reason.trim().length > 0) {
    return [detailsRecord.reason.trim()];
  }

  return [buildStableDetailString(detailsRecord)];
}

function formatValidationDetailItem(item) {
  if (typeof item === "string") {
    return item.trim();
  }

  const itemRecord = asRecord(item);
  const path = typeof itemRecord.path === "string" ? itemRecord.path.trim() : "";
  const message = typeof itemRecord.message === "string" ? itemRecord.message.trim() : "";
  if (path.length > 0 && message.length > 0) {
    return `${path}: ${message}`;
  }
  if (message.length > 0) {
    return message;
  }
  if (path.length > 0) {
    return path;
  }
  return buildStableDetailString(item);
}

function buildStableDetailString(value) {
  try {
    return JSON.stringify(normalizeJsonLikeValue(value));
  } catch (_error) {
    return String(value);
  }
}

function normalizeJsonLikeValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeJsonLikeValue(item));
  }
  if (typeof value === "bigint") {
    return String(value);
  }
  if (value !== null && typeof value === "object") {
    const record = asRecord(value);
    const normalized = {};
    Object.keys(record).sort(compareStableStrings).forEach((key) => {
      normalized[key] = normalizeJsonLikeValue(record[key]);
    });
    return normalized;
  }
  return value;
}

function showPageError(pageRoot, message, details) {
  const banner = pageRoot.querySelector(".error-banner");
  if (banner !== null) {
    banner.textContent = message;
    banner.classList.remove("hidden");
  }

  const detailsContainer = pageRoot.querySelector(".validation-list");
  if (detailsContainer === null) {
    return;
  }
  if (!Array.isArray(details) || details.length === 0) {
    detailsContainer.innerHTML = "";
    detailsContainer.classList.add("hidden");
    return;
  }

  const list = document.createElement("ul");
  details.forEach((detailItem) => {
    const item = document.createElement("li");
    item.textContent = String(detailItem);
    list.appendChild(item);
  });
  detailsContainer.innerHTML = "";
  detailsContainer.appendChild(list);
  detailsContainer.classList.remove("hidden");
}

function clearPageError(pageRoot) {
  const banner = pageRoot.querySelector(".error-banner");
  if (banner !== null) {
    banner.textContent = "";
    banner.classList.add("hidden");
  }

  const detailsContainer = pageRoot.querySelector(".validation-list");
  if (detailsContainer !== null) {
    detailsContainer.innerHTML = "";
    detailsContainer.classList.add("hidden");
  }
}

function normalizeError(error) {
  if (error instanceof Error) {
    const details = Array.isArray(error.details) ? error.details : [];
    const message = String(error.message || "").trim();
    return {
      message: message.length > 0 ? message : "Unexpected backtest operation error.",
      details,
    };
  }
  return { message: "Unexpected backtest operation error.", details: [] };
}

function requireDataAttr(node, camelCaseName) {
  const value = node.dataset[camelCaseName];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing data attribute: ${camelCaseName}`);
  }
  return value;
}

function asRecord(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
}

function compareStableStrings(left, right) {
  if (left < right) {
    return -1;
  }
  if (left > right) {
    return 1;
  }
  return 0;
}

function copyRecord(record) {
  const output = {};
  Object.keys(record).sort(compareStableStrings).forEach((key) => {
    output[key] = record[key];
  });
  return output;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

function buildCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function buildActionButton({ label, onClick, className = "", disabled = false }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button-link ${className}`.trim();
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}
