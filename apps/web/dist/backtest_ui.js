const BACKTEST_PAGE_SELECTOR = "[data-backtest-page]";
const CHART_SCHEMA = "backtest_chart_overlay_v1";
const POLL_INTERVAL_MS = 1500;
const TRADES_PAGE_SIZE = 25;
const TERMINAL_JOB_STATES = new Set(["succeeded", "failed", "cancelled"]);
const ACTIVE_JOB_STATES = new Set(["queued", "running"]);
const DEFAULT_INDICATOR_ID = "ma.dema";
const DEFAULT_SOURCE = "close";
const LAST_JOB_STORAGE_KEY = "roehub.backtests.selectedJobId";

document.addEventListener("DOMContentLoaded", () => {
  const pageRoot = document.querySelector(BACKTEST_PAGE_SELECTOR);
  if (pageRoot === null) {
    return;
  }
  initBacktestsPage(pageRoot);
});

function initBacktestsPage(pageRoot) {
  const paths = {
    defaults: requireDataAttr(pageRoot, "apiDefaultsPath"),
    preflight: requireDataAttr(pageRoot, "apiPreflightPath"),
    jobs: requireDataAttr(pageRoot, "apiJobsPath"),
    jobTemplate: requireDataAttr(pageRoot, "apiJobPathTemplate"),
    topTemplate: requireDataAttr(pageRoot, "apiTopPathTemplate"),
    variantTemplate: requireDataAttr(pageRoot, "apiVariantPathTemplate"),
    tradesTemplate: requireDataAttr(pageRoot, "apiTradesPathTemplate"),
    cancelTemplate: requireDataAttr(pageRoot, "apiCancelPathTemplate"),
  };

  const nodes = collectBacktestNodes(pageRoot);
  if (Object.values(nodes).some((node) => node === null)) {
    return;
  }

  const state = {
    defaults: null,
    selectedJobId: readLastSelectedJobId(),
    selectedJob: null,
    topRows: [],
    trades: [],
    tradesPage: 0,
    pollTimer: null,
    pollController: null,
    tradesController: null,
  };

  const refreshJobs = async () => {
    await loadJobs({ pageRoot, nodes, paths, state });
  };

  nodes.addIndicatorButton.addEventListener("click", () => {
    addIndicatorRow({ nodes, state });
  });
  nodes.riskMode.addEventListener("change", () => {
    syncRiskControls(nodes);
  });
  nodes.loadSampleButton.addEventListener("click", () => {
    loadSampleRequest({ nodes, state });
  });
  nodes.preflightButton.addEventListener("click", async () => {
    await submitPreflight({ pageRoot, nodes, paths, state });
  });
  nodes.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await createBacktestJob({ pageRoot, nodes, paths, state });
  });
  nodes.refreshJobsButton.addEventListener("click", refreshJobs);
  nodes.jobsBody.addEventListener("click", async (event) => {
    const button = event.target instanceof HTMLElement
      ? event.target.closest("[data-backtest-action]")
      : null;
    if (!(button instanceof HTMLElement)) {
      return;
    }
    const action = button.getAttribute("data-backtest-action");
    const jobId = button.getAttribute("data-job-id");
    if (action === "select-job" && jobId) {
      await selectJob({ pageRoot, nodes, paths, state, jobId });
    }
    if (action === "copy" && jobId) {
      await copyToClipboard(jobId);
    }
  });
  nodes.topBody.addEventListener("click", async (event) => {
    const button = event.target instanceof HTMLElement
      ? event.target.closest("[data-backtest-action]")
      : null;
    if (!(button instanceof HTMLElement)) {
      return;
    }
    const action = button.getAttribute("data-backtest-action");
    if (action === "show-trades") {
      const variantKey = button.getAttribute("data-variant-key");
      if (variantKey && state.selectedJobId) {
        await loadTrades({ pageRoot, nodes, paths, state, variantKey });
      }
      return;
    }
    if (action === "copy") {
      const copyValue = button.getAttribute("data-copy-value") || "";
      await copyToClipboard(copyValue);
    }
  });
  nodes.cancelJobButton.addEventListener("click", async () => {
    await cancelSelectedJob({ pageRoot, nodes, paths, state });
  });
  nodes.tradesPrevButton.addEventListener("click", () => {
    if (state.tradesPage > 0) {
      state.tradesPage -= 1;
      renderTradesTable({ nodes, state });
    }
  });
  nodes.tradesNextButton.addEventListener("click", () => {
    const maxPage = Math.max(0, Math.ceil(state.trades.length / TRADES_PAGE_SIZE) - 1);
    if (state.tradesPage < maxPage) {
      state.tradesPage += 1;
      renderTradesTable({ nodes, state });
    }
  });
  window.addEventListener("beforeunload", () => {
    stopPolling(state);
    abortController(state.pollController);
    abortController(state.tradesController);
  });

  addIndicatorRow({ nodes, state });
  syncRiskControls(nodes);
  syncEmptyJobState({ nodes });
  loadDefaults({ pageRoot, nodes, paths, state })
    .then(refreshJobs)
    .then(async () => {
      if (state.selectedJobId) {
        await selectJob({ pageRoot, nodes, paths, state, jobId: state.selectedJobId });
      }
    })
    .catch((error) => {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    });
}

function collectBacktestNodes(pageRoot) {
  return {
    form: pageRoot.querySelector("#backtest-request-form"),
    refreshJobsButton: pageRoot.querySelector("#backtest-refresh-jobs"),
    addIndicatorButton: pageRoot.querySelector("#backtest-add-indicator"),
    loadSampleButton: pageRoot.querySelector("#backtest-load-sample"),
    preflightButton: pageRoot.querySelector("#backtest-preflight-submit"),
    createButton: pageRoot.querySelector("#backtest-create-submit"),
    cancelJobButton: pageRoot.querySelector("#backtest-cancel-job"),
    indicatorRows: pageRoot.querySelector("#backtest-indicator-rows"),
    exchange: pageRoot.querySelector("#backtest-exchange"),
    marketType: pageRoot.querySelector("#backtest-market-type"),
    symbol: pageRoot.querySelector("#backtest-symbol"),
    timeframe: pageRoot.querySelector("#backtest-timeframe"),
    start: pageRoot.querySelector("#backtest-start"),
    end: pageRoot.querySelector("#backtest-end"),
    topN: pageRoot.querySelector("#backtest-top-n"),
    riskMode: pageRoot.querySelector("#backtest-risk-mode"),
    tpSlGrid: pageRoot.querySelector("#backtest-tp-sl-grid"),
    tpStart: pageRoot.querySelector("#backtest-tp-start"),
    tpStop: pageRoot.querySelector("#backtest-tp-stop"),
    tpStep: pageRoot.querySelector("#backtest-tp-step"),
    slStart: pageRoot.querySelector("#backtest-sl-start"),
    slStop: pageRoot.querySelector("#backtest-sl-stop"),
    slStep: pageRoot.querySelector("#backtest-sl-step"),
    rankingMetric: pageRoot.querySelector("#backtest-ranking-metric"),
    rankingDirection: pageRoot.querySelector("#backtest-ranking-direction"),
    directionMode: pageRoot.querySelector("#backtest-direction-mode"),
    sizingMode: pageRoot.querySelector("#backtest-sizing-mode"),
    feeRate: pageRoot.querySelector("#backtest-fee-rate"),
    slippageRate: pageRoot.querySelector("#backtest-slippage-rate"),
    initialCash: pageRoot.querySelector("#backtest-initial-cash"),
    quoteAmount: pageRoot.querySelector("#backtest-quote-amount"),
    equityPct: pageRoot.querySelector("#backtest-equity-pct"),
    minQuote: pageRoot.querySelector("#backtest-min-quote"),
    maxQuote: pageRoot.querySelector("#backtest-max-quote"),
    profitLockEnabled: pageRoot.querySelector("#backtest-profit-lock-enabled"),
    profitLockPct: pageRoot.querySelector("#backtest-profit-lock-pct"),
    closeOnEnd: pageRoot.querySelector("#backtest-close-on-end"),
    preflightStatus: pageRoot.querySelector("#backtest-preflight-status"),
    preflightSummary: pageRoot.querySelector("#backtest-preflight-summary"),
    preflightJson: pageRoot.querySelector("#backtest-preflight-json"),
    jobsStatus: pageRoot.querySelector("#backtest-jobs-status"),
    jobsBody: pageRoot.querySelector("#backtest-jobs-body"),
    selectedState: pageRoot.querySelector("#backtest-selected-state"),
    progressBar: pageRoot.querySelector("#backtest-progress-bar"),
    jobSummary: pageRoot.querySelector("#backtest-job-summary"),
    topBody: pageRoot.querySelector("#backtest-top-body"),
    tradesStatus: pageRoot.querySelector("#backtest-trades-status"),
    tradesSummary: pageRoot.querySelector("#backtest-trades-summary"),
    tradesChart: pageRoot.querySelector("#backtest-trades-chart"),
    chartFallback: pageRoot.querySelector("#backtest-chart-fallback"),
    chartLegend: pageRoot.querySelector("#backtest-chart-legend"),
    tradesBody: pageRoot.querySelector("#backtest-trades-body"),
    tradesPrevButton: pageRoot.querySelector("#backtest-trades-prev"),
    tradesNextButton: pageRoot.querySelector("#backtest-trades-next"),
    tradesPageLabel: pageRoot.querySelector("#backtest-trades-page-label"),
  };
}

async function loadDefaults({ pageRoot, nodes, paths, state }) {
  clearPageError(pageRoot);
  try {
    const defaults = await apiJson(paths.defaults);
    state.defaults = asRecord(defaults);
    populateDefaults({ nodes, state });
    setStateBadge(nodes.jobsStatus, "ready", "succeeded");
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    setStateBadge(nodes.jobsStatus, "defaults unavailable", "failed");
  }
}

function populateDefaults({ nodes, state }) {
  const defaults = asRecord(state.defaults);
  setSelectOptions(nodes.timeframe, arrayOrEmpty(defaults.supported_timeframes), "15m");
  setSelectOptions(nodes.riskMode, arrayOrEmpty(defaults.risk_modes), "none");
  setSelectOptions(nodes.directionMode, arrayOrEmpty(defaults.direction_modes), "long_short_reversal");
  setSelectOptions(nodes.sizingMode, arrayOrEmpty(defaults.sizing_modes), "fixed_equity_pct");
  setSelectOptions(nodes.rankingMetric, arrayOrEmpty(defaults.ranking_metrics), "total_return_pct");

  const rankingDefault = asRecord(defaults.ranking_default);
  if (rankingDefault.primary_metric) {
    nodes.rankingMetric.value = String(rankingDefault.primary_metric);
  }
  if (rankingDefault.direction) {
    nodes.rankingDirection.value = String(rankingDefault.direction);
  }
  if (Number(defaults.top_n_default || 0) > 0) {
    nodes.topN.value = String(defaults.top_n_default);
  }

  const guardrails = asRecord(defaults.guardrails);
  if (Number(guardrails.max_top_n || 0) > 0) {
    nodes.topN.max = String(guardrails.max_top_n);
  }

  const executionDefaults = asRecord(defaults.execution_defaults);
  setNumberInput(nodes.feeRate, executionDefaults.fee_rate);
  setNumberInput(nodes.slippageRate, executionDefaults.slippage_rate);
  setNumberInput(nodes.initialCash, executionDefaults.initial_cash_quote);
  if (executionDefaults.direction_mode) {
    nodes.directionMode.value = String(executionDefaults.direction_mode);
  }
  nodes.closeOnEnd.checked = Boolean(executionDefaults.close_on_end);

  const sizing = asRecord(executionDefaults.sizing);
  if (sizing.mode) {
    nodes.sizingMode.value = String(sizing.mode);
  }
  setNumberInput(nodes.quoteAmount, sizing.quote_amount);
  setNumberInput(nodes.equityPct, sizing.equity_pct);
  setNumberInput(nodes.minQuote, sizing.min_quote);
  setNumberInput(nodes.maxQuote, sizing.max_quote);

  const profitLock = asRecord(executionDefaults.profit_lock);
  nodes.profitLockEnabled.checked = Boolean(profitLock.enabled);
  setNumberInput(nodes.profitLockPct, profitLock.safe_profit_percent);

  syncIndicatorRows({ nodes, state });
  syncRiskControls(nodes);
}

function loadSampleRequest({ nodes, state }) {
  nodes.exchange.value = "binance";
  nodes.marketType.value = "spot";
  nodes.symbol.value = "BTCUSDT";
  nodes.timeframe.value = selectFallback(nodes.timeframe, "15m");
  nodes.start.value = "2020-01-11T20:08";
  nodes.end.value = "2026-04-11T20:08";
  nodes.topN.value = String(Number(asRecord(state.defaults).top_n_default || 100));
  nodes.riskMode.value = "none";
  nodes.rankingMetric.value = selectFallback(nodes.rankingMetric, "total_return_pct");
  nodes.rankingDirection.value = "desc";
  nodes.directionMode.value = selectFallback(nodes.directionMode, "long_short_reversal");
  nodes.sizingMode.value = selectFallback(nodes.sizingMode, "fixed_equity_pct");
  nodes.feeRate.value = "0.00075";
  nodes.slippageRate.value = "0.0001";
  nodes.initialCash.value = "10000";
  nodes.quoteAmount.value = "1000";
  nodes.equityPct.value = "10";
  nodes.minQuote.value = "100";
  nodes.maxQuote.value = "2500";
  nodes.profitLockEnabled.checked = false;
  nodes.profitLockPct.value = "30";
  nodes.closeOnEnd.checked = true;
  nodes.indicatorRows.innerHTML = "";
  addIndicatorRow({ nodes, state, values: {
    indicator_id: DEFAULT_INDICATOR_ID,
    source: DEFAULT_SOURCE,
    window_start: "5",
    window_stop: "10",
    window_step: "1",
  } });
  syncRiskControls(nodes);
}

function addIndicatorRow({ nodes, state, values = {} }) {
  const row = document.createElement("div");
  row.className = "backtest-indicator-row";

  const indicatorLabel = buildLabel("Indicator", "select");
  const indicatorSelect = document.createElement("select");
  indicatorSelect.setAttribute("aria-label", "Indicator id");
  indicatorSelect.dataset.role = "indicator-id";

  const sourceLabel = buildLabel("Source", "select");
  const sourceSelect = document.createElement("select");
  sourceSelect.setAttribute("aria-label", "Indicator source");
  sourceSelect.dataset.role = "source";

  const startLabel = buildLabel("Window start", "input");
  const startInput = buildNumberInput("window-start", values.window_start || "5");
  const stopLabel = buildLabel("Window stop", "input");
  const stopInput = buildNumberInput("window-stop", values.window_stop || "10");
  const stepLabel = buildLabel("Window step", "input");
  const stepInput = buildNumberInput("window-step", values.window_step || "1");

  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "button-link button-link--secondary";
  removeButton.textContent = "Remove";
  removeButton.addEventListener("click", () => {
    if (nodes.indicatorRows.querySelectorAll(".backtest-indicator-row").length > 1) {
      row.remove();
    }
  });

  row.append(
    indicatorLabel,
    indicatorSelect,
    sourceLabel,
    sourceSelect,
    startLabel,
    startInput,
    stopLabel,
    stopInput,
    stepLabel,
    stepInput,
    removeButton,
  );
  nodes.indicatorRows.appendChild(row);

  populateIndicatorSelect({ selectNode: indicatorSelect, state });
  indicatorSelect.value = selectFallback(indicatorSelect, values.indicator_id || DEFAULT_INDICATOR_ID);
  populateSourceSelect({ indicatorSelect, sourceSelect, state });
  sourceSelect.value = selectFallback(sourceSelect, values.source || DEFAULT_SOURCE);
  indicatorSelect.addEventListener("change", () => {
    populateSourceSelect({ indicatorSelect, sourceSelect, state });
  });
}

function syncIndicatorRows({ nodes, state }) {
  nodes.indicatorRows.querySelectorAll(".backtest-indicator-row").forEach((row) => {
    const indicatorSelect = row.querySelector('[data-role="indicator-id"]');
    const sourceSelect = row.querySelector('[data-role="source"]');
    if (indicatorSelect instanceof HTMLSelectElement) {
      const previousIndicator = indicatorSelect.value;
      populateIndicatorSelect({ selectNode: indicatorSelect, state });
      indicatorSelect.value = selectFallback(indicatorSelect, previousIndicator || DEFAULT_INDICATOR_ID);
    }
    if (
      indicatorSelect instanceof HTMLSelectElement
      && sourceSelect instanceof HTMLSelectElement
    ) {
      const previousSource = sourceSelect.value;
      populateSourceSelect({ indicatorSelect, sourceSelect, state });
      sourceSelect.value = selectFallback(sourceSelect, previousSource || DEFAULT_SOURCE);
    }
  });
}

async function submitPreflight({ pageRoot, nodes, paths }) {
  clearPageError(pageRoot);
  setStateBadge(nodes.preflightStatus, "loading", "running");
  setButtonBusy(nodes.preflightButton, true);
  try {
    const payload = buildRequestPayload(nodes);
    const result = await apiJson(paths.preflight, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderPreflight({ nodes, result });
    setStateBadge(nodes.preflightStatus, "valid", "succeeded");
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    setStateBadge(nodes.preflightStatus, "failed", "failed");
  } finally {
    setButtonBusy(nodes.preflightButton, false);
  }
}

async function createBacktestJob({ pageRoot, nodes, paths, state }) {
  clearPageError(pageRoot);
  setButtonBusy(nodes.createButton, true);
  try {
    const payload = buildRequestPayload(nodes);
    const job = await apiJson(paths.jobs, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await loadJobs({ pageRoot, nodes, paths, state });
    await selectJob({ pageRoot, nodes, paths, state, jobId: String(job.job_id), seedJob: job });
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
  } finally {
    setButtonBusy(nodes.createButton, false);
  }
}

async function loadJobs({ pageRoot, nodes, paths, state }) {
  clearPageError(pageRoot);
  setStateBadge(nodes.jobsStatus, "loading", "running");
  nodes.jobsBody.innerHTML = "<tr><td colspan=\"5\">Loading jobs...</td></tr>";
  try {
    const payload = await apiJson(`${paths.jobs}?limit=20`);
    const jobs = Array.isArray(payload.items) ? payload.items.map((item) => asRecord(item)) : [];
    renderJobsTable({ nodes, jobs, state });
    setStateBadge(nodes.jobsStatus, `${jobs.length} jobs`, "succeeded");
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    nodes.jobsBody.innerHTML = "<tr><td colspan=\"5\">Failed to load jobs.</td></tr>";
    setStateBadge(nodes.jobsStatus, "failed", "failed");
  }
}

async function selectJob({ pageRoot, nodes, paths, state, jobId, seedJob = null }) {
  clearPageError(pageRoot);
  state.selectedJobId = jobId;
  storeLastSelectedJobId(jobId);
  state.selectedJob = seedJob;
  state.topRows = [];
  state.trades = [];
  state.tradesPage = 0;
  renderJob({ nodes, job: seedJob || { job_id: jobId, state: "loading" } });
  renderTopRows({ nodes, rows: [] });
  renderTradesEmpty(nodes);
  await refreshSelectedJob({ pageRoot, nodes, paths, state });
}

async function refreshSelectedJob({ pageRoot, nodes, paths, state }) {
  if (!state.selectedJobId) {
    return;
  }
  abortController(state.pollController);
  state.pollController = new AbortController();
  const jobPath = renderBacktestPath(paths.jobTemplate, { job_id: state.selectedJobId });
  try {
    const job = await apiJson(jobPath, { signal: state.pollController.signal });
    state.selectedJob = asRecord(job);
    renderJob({ nodes, job: state.selectedJob });
    const stateValue = String(state.selectedJob.state || "");
    if (ACTIVE_JOB_STATES.has(stateValue)) {
      schedulePoll({ pageRoot, nodes, paths, state });
      return;
    }
    stopPolling(state);
    if (stateValue === "succeeded") {
      await loadTopRows({ pageRoot, nodes, paths, state });
    } else {
      renderTopRows({ nodes, rows: [] });
    }
  } catch (error) {
    if (isAbortError(error)) {
      return;
    }
    stopPolling(state);
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    setStateBadge(nodes.selectedState, "failed", "failed");
  }
}

function schedulePoll({ pageRoot, nodes, paths, state }) {
  stopPolling(state);
  state.pollTimer = window.setTimeout(() => {
    refreshSelectedJob({ pageRoot, nodes, paths, state });
  }, POLL_INTERVAL_MS);
}

async function loadTopRows({ pageRoot, nodes, paths, state }) {
  if (!state.selectedJobId) {
    return;
  }
  nodes.topBody.innerHTML = "<tr><td colspan=\"6\">Loading top variants...</td></tr>";
  try {
    const topPath = renderBacktestPath(paths.topTemplate, { job_id: state.selectedJobId });
    const payload = await apiJson(topPath);
    state.topRows = Array.isArray(payload.items) ? payload.items.map((item) => asRecord(item)) : [];
    renderTopRows({ nodes, rows: state.topRows });
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    nodes.topBody.innerHTML = "<tr><td colspan=\"6\">Failed to load top variants.</td></tr>";
  }
}

async function loadTrades({ pageRoot, nodes, paths, state, variantKey }) {
  if (!state.selectedJobId) {
    return;
  }
  clearPageError(pageRoot);
  abortController(state.tradesController);
  state.tradesController = new AbortController();
  setStateBadge(nodes.tradesStatus, "loading", "running");
  nodes.tradesBody.innerHTML = "<tr><td colspan=\"6\">Loading trades...</td></tr>";
  try {
    const tradesPath = renderBacktestPath(paths.tradesTemplate, {
      job_id: state.selectedJobId,
      variant_key: variantKey,
    });
    const detail = await apiJson(tradesPath, {
      method: "POST",
      signal: state.tradesController.signal,
    });
    renderTradesDetail({ nodes, state, detail });
  } catch (error) {
    if (isAbortError(error)) {
      return;
    }
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
    setStateBadge(nodes.tradesStatus, "failed", "failed");
    nodes.tradesBody.innerHTML = "<tr><td colspan=\"6\">Failed to load trades.</td></tr>";
  }
}

async function cancelSelectedJob({ pageRoot, nodes, paths, state }) {
  if (!state.selectedJobId) {
    return;
  }
  clearPageError(pageRoot);
  try {
    const cancelPath = renderBacktestPath(paths.cancelTemplate, { job_id: state.selectedJobId });
    await apiJson(cancelPath, { method: "POST" });
    await refreshSelectedJob({ pageRoot, nodes, paths, state });
    await loadJobs({ pageRoot, nodes, paths, state });
  } catch (error) {
    const normalized = normalizeError(error);
    showPageError(pageRoot, normalized.message, normalized.details);
  }
}

function buildRequestPayload(nodes) {
  const riskMode = nodes.riskMode.value;
  return {
    coordinates: {
      exchange: requireText(nodes.exchange, "exchange"),
      market_type: requireText(nodes.marketType, "market_type"),
      symbol: requireText(nodes.symbol, "symbol").toUpperCase(),
    },
    timeframe: requireText(nodes.timeframe, "timeframe"),
    time_range: {
      start: datetimeLocalToUtcString(nodes.start.value, "start"),
      end: datetimeLocalToUtcString(nodes.end.value, "end"),
    },
    indicators: collectIndicatorPayloads(nodes),
    risk: buildRiskPayload(nodes, riskMode),
    execution: {
      direction_mode: requireText(nodes.directionMode, "direction_mode"),
      fee_rate: readNumber(nodes.feeRate, "fee_rate"),
      slippage_rate: readNumber(nodes.slippageRate, "slippage_rate"),
      initial_cash_quote: readNumber(nodes.initialCash, "initial_cash_quote"),
      sizing: buildSizingPayload(nodes),
      profit_lock: buildProfitLockPayload(nodes),
      close_on_end: nodes.closeOnEnd.checked,
    },
    ranking: {
      primary_metric: requireText(nodes.rankingMetric, "primary_metric"),
      direction: requireText(nodes.rankingDirection, "ranking.direction"),
    },
    top_n: readPositiveInt(nodes.topN, "top_n"),
  };
}

function collectIndicatorPayloads(nodes) {
  const rows = Array.from(nodes.indicatorRows.querySelectorAll(".backtest-indicator-row"));
  if (rows.length === 0) {
    throw new Error("At least one indicator is required.");
  }
  return rows.map((row, index) => {
    const indicator = row.querySelector('[data-role="indicator-id"]');
    const source = row.querySelector('[data-role="source"]');
    const start = row.querySelector('[data-role="window-start"]');
    const stop = row.querySelector('[data-role="window-stop"]');
    const step = row.querySelector('[data-role="window-step"]');
    if (
      !(indicator instanceof HTMLSelectElement)
      || !(source instanceof HTMLSelectElement)
      || !(start instanceof HTMLInputElement)
      || !(stop instanceof HTMLInputElement)
      || !(step instanceof HTMLInputElement)
    ) {
      throw new Error(`Indicator row ${index + 1} is incomplete.`);
    }
    return {
      indicator_id: requireText(indicator, `indicators.${index}.indicator_id`),
      sources: [requireText(source, `indicators.${index}.source`)],
      window: {
        start: readPositiveInt(start, `indicators.${index}.window.start`),
        stop: readPositiveInt(stop, `indicators.${index}.window.stop`),
        step: readPositiveInt(step, `indicators.${index}.window.step`),
      },
    };
  });
}

function buildRiskPayload(nodes, riskMode) {
  if (riskMode !== "tp_sl_grid") {
    return { mode: "none" };
  }
  return {
    mode: "tp_sl_grid",
    tp: {
      start_pct: readNumber(nodes.tpStart, "risk.tp.start_pct"),
      stop_pct: readNumber(nodes.tpStop, "risk.tp.stop_pct"),
      step_pct: readNumber(nodes.tpStep, "risk.tp.step_pct"),
    },
    sl: {
      start_pct: readNumber(nodes.slStart, "risk.sl.start_pct"),
      stop_pct: readNumber(nodes.slStop, "risk.sl.stop_pct"),
      step_pct: readNumber(nodes.slStep, "risk.sl.step_pct"),
    },
  };
}

function buildSizingPayload(nodes) {
  const mode = requireText(nodes.sizingMode, "execution.sizing.mode");
  const payload = { mode };
  if (mode === "fixed_quote") {
    payload.quote_amount = readNumber(nodes.quoteAmount, "execution.sizing.quote_amount");
  }
  if (
    mode === "fixed_equity_pct"
    || mode === "fixed_equity_pct_min_quote"
    || mode === "fixed_equity_pct_max_quote"
  ) {
    payload.equity_pct = readNumber(nodes.equityPct, "execution.sizing.equity_pct");
  }
  if (mode === "fixed_equity_pct_min_quote") {
    payload.min_quote = readNumber(nodes.minQuote, "execution.sizing.min_quote");
  }
  if (mode === "fixed_equity_pct_max_quote") {
    payload.max_quote = readNumber(nodes.maxQuote, "execution.sizing.max_quote");
  }
  return payload;
}

function buildProfitLockPayload(nodes) {
  if (!nodes.profitLockEnabled.checked) {
    return { enabled: false };
  }
  return {
    enabled: true,
    safe_profit_percent: readNumber(nodes.profitLockPct, "execution.profit_lock.safe_profit_percent"),
  };
}

function renderPreflight({ nodes, result }) {
  const payload = asRecord(result);
  renderKeyValueGrid(nodes.preflightSummary, [
    ["request_hash", compactHash(payload.request_hash)],
    ["result_config_hash", compactHash(payload.result_config_hash)],
    ["cost_class", asRecord(payload.cost_estimate).cost_class || ""],
    ["indicator_rows", asRecord(payload.cost_estimate).indicator_rows || ""],
    ["candidate_combinations", asRecord(payload.cost_estimate).candidate_combinations || ""],
    ["tp_sl_cells", asRecord(payload.cost_estimate).tp_sl_cells || ""],
    ["warnings", arrayOrEmpty(payload.warnings).length],
    ["errors", arrayOrEmpty(payload.errors).length],
  ]);
  nodes.preflightJson.textContent = JSON.stringify(
    {
      normalized_request: payload.normalized_request,
      artifact_metadata: payload.artifact_metadata,
      cost_estimate: payload.cost_estimate,
      warnings: payload.warnings,
      errors: payload.errors,
    },
    null,
    2,
  );
}

function renderJobsTable({ nodes, jobs, state }) {
  nodes.jobsBody.innerHTML = "";
  if (jobs.length === 0) {
    nodes.jobsBody.innerHTML = "<tr><td colspan=\"5\">No jobs yet.</td></tr>";
    return;
  }

  jobs.forEach((job) => {
    const row = document.createElement("tr");
    if (String(job.job_id || "") === state.selectedJobId) {
      row.className = "row-clickable backtest-row-selected";
    }
    const request = asRecord(job.request);
    const coordinates = asRecord(request.coordinates);
    const progress = asRecord(job.progress);
    appendCell(row, formatTimestamp(job.created_at));
    const stateCell = document.createElement("td");
    stateCell.appendChild(buildStateBadge(String(job.state || "")));
    row.appendChild(stateCell);
    appendCell(row, [
      coordinates.symbol,
      request.timeframe,
      `${asRecord(request.time_range).start || ""} - ${asRecord(request.time_range).end || ""}`,
    ].filter(Boolean).join(" | "));
    appendCell(row, `${progress.pipeline_stage || "-"} ${formatPercent(progress.percent)}`);

    const actionsCell = document.createElement("td");
    const selectButton = buildActionButton({
      action: "select-job",
      label: "Select",
      jobId: String(job.job_id || ""),
    });
    const copyButton = buildActionButton({
      action: "copy",
      label: "Copy ID",
      jobId: String(job.job_id || ""),
      className: "button-link--secondary",
    });
    actionsCell.append(selectButton, copyButton);
    row.appendChild(actionsCell);
    nodes.jobsBody.appendChild(row);
  });
}

function renderJob({ nodes, job }) {
  const payload = asRecord(job);
  const stateValue = String(payload.state || "none");
  const progress = asRecord(payload.progress);
  const percent = Number(progress.percent || 0);
  setStateBadge(nodes.selectedState, stateValue, stateValue);
  nodes.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  nodes.cancelJobButton.disabled = !ACTIVE_JOB_STATES.has(stateValue);
  renderKeyValueGrid(nodes.jobSummary, [
    ["job_id", payload.job_id || ""],
    ["pipeline_stage", progress.pipeline_stage || ""],
    ["percent", formatPercent(progress.percent)],
    ["processed/total", `${progress.processed_units || 0}/${progress.total_units || 0}`],
    ["request_hash", compactHash(payload.request_hash)],
    ["result_config_hash", compactHash(payload.result_config_hash)],
    ["created_at", formatTimestamp(payload.created_at)],
    ["started_at", formatTimestamp(payload.started_at)],
    ["finished_at", formatTimestamp(payload.finished_at)],
    ["requested_top_n", payload.requested_top_n || ""],
  ]);
}

function syncEmptyJobState({ nodes }) {
  renderJob({ nodes, job: { state: "none", progress: { percent: 0 } } });
  renderTopRows({ nodes, rows: [] });
  renderTradesEmpty(nodes);
}

function renderTopRows({ nodes, rows }) {
  nodes.topBody.innerHTML = "";
  if (!Array.isArray(rows) || rows.length === 0) {
    nodes.topBody.innerHTML = "<tr><td colspan=\"6\">No top variants loaded.</td></tr>";
    return;
  }

  rows.forEach((rowPayload) => {
    const row = asRecord(rowPayload);
    const tableRow = document.createElement("tr");
    appendCell(tableRow, String(row.rank || ""));
    appendCell(tableRow, compactRecord(asRecord(row.readable_params)));
    appendCell(tableRow, compactMetrics(asRecord(row.summary_metrics)));
    appendCell(tableRow, formatTpSl(row.best_tp_pct, row.best_sl_pct));

    const variantCell = document.createElement("td");
    variantCell.append(
      buildCodeLine("variant_key", String(row.variant_key || "")),
      buildCodeLine("variant_hash", compactHash(row.variant_hash)),
    );
    tableRow.appendChild(variantCell);

    const actionsCell = document.createElement("td");
    const showTradesButton = buildActionButton({
      action: "show-trades",
      label: "show trades",
      variantKey: String(row.variant_key || ""),
      variantHash: String(row.variant_hash || ""),
    });
    const copyKeyButton = buildActionButton({
      action: "copy",
      label: "Copy key",
      copyValue: String(row.variant_key || ""),
      className: "button-link--secondary",
    });
    const copyHashButton = buildActionButton({
      action: "copy",
      label: "Copy hash",
      copyValue: String(row.variant_hash || ""),
      className: "button-link--secondary",
    });
    actionsCell.append(showTradesButton, copyKeyButton, copyHashButton);
    tableRow.appendChild(actionsCell);
    nodes.topBody.appendChild(tableRow);
  });
}

function renderTradesDetail({ nodes, state, detail }) {
  const payload = asRecord(detail);
  const trades = Array.isArray(payload.trades) ? payload.trades.map((item) => asRecord(item)) : [];
  state.trades = trades;
  state.tradesPage = 0;

  const cache = asRecord(payload.cache);
  const timing = asRecord(payload.timing);
  renderKeyValueGrid(nodes.tradesSummary, [
    ["job_id", payload.job_id || ""],
    ["variant_key", payload.variant_key || ""],
    ["variant_hash", compactHash(payload.variant_hash)],
    ["cache", cache.status || ""],
    ["trade_count", trades.length],
    ["lazy_trades_compute", formatSeconds(timing.lazy_trades_compute)],
    ["lazy_trades_cache_hit", formatSeconds(timing.lazy_trades_cache_hit)],
    ["params", compactRecord(asRecord(payload.readable_params))],
    ["metrics", compactMetrics(asRecord(payload.summary_metrics))],
  ]);
  setStateBadge(nodes.tradesStatus, `${trades.length} trades`, String(cache.status || "succeeded"));
  renderChartOverlay({ nodes, detail: payload, trades });
  renderTradesTable({ nodes, state });
}

function renderChartOverlay({ nodes, detail, trades }) {
  const overlay = asRecord(detail.chart_overlay);
  if (overlay.schema !== CHART_SCHEMA) {
    clearCanvas(nodes.tradesChart);
    nodes.chartFallback.textContent = `Unsupported chart overlay schema: ${String(overlay.schema || "missing")}`;
    renderChartLegend(nodes.chartLegend, []);
    return;
  }

  const markers = Array.isArray(overlay.markers) ? overlay.markers.map((item) => asRecord(item)) : [];
  const segments = Array.isArray(overlay.segments) ? overlay.segments.map((item) => asRecord(item)) : [];
  const chartData = buildChartData({ markers, segments, trades });
  drawTradeChart({ canvas: nodes.tradesChart, chartData });
  nodes.chartFallback.textContent = [
    `${chartData.segments.length} segments and ${chartData.markers.length} markers rendered.`,
    "OHLC candles are not present in the lazy trades payload; this chart uses trade price/time overlay.",
  ].join(" ");
  renderChartLegend(nodes.chartLegend, [
    ["long", "Entry/exit long"],
    ["short", "Entry/exit short"],
    ["tp", "TP exit"],
    ["sl", "SL exit"],
    ["signal", "Signal close"],
  ]);
}

function renderTradesTable({ nodes, state }) {
  const total = state.trades.length;
  const maxPage = Math.max(0, Math.ceil(total / TRADES_PAGE_SIZE) - 1);
  state.tradesPage = Math.max(0, Math.min(state.tradesPage, maxPage));
  const startIndex = state.tradesPage * TRADES_PAGE_SIZE;
  const endIndex = Math.min(total, startIndex + TRADES_PAGE_SIZE);
  const pageItems = state.trades.slice(startIndex, endIndex);

  nodes.tradesBody.innerHTML = "";
  if (pageItems.length === 0) {
    nodes.tradesBody.innerHTML = "<tr><td colspan=\"6\">No trades loaded.</td></tr>";
  } else {
    pageItems.forEach((trade) => {
      const row = document.createElement("tr");
      appendCell(row, String(trade.trade_index ?? ""));
      appendCell(row, String(trade.side || trade.direction || ""));
      appendCell(row, `${formatTimestamp(trade.entry_timestamp)} @ ${formatNumber(trade.entry_price)}`);
      appendCell(row, `${formatTimestamp(trade.exit_timestamp)} @ ${formatNumber(trade.exit_price)}`);
      appendCell(row, formatPercentValue(trade.return_pct));
      appendCell(row, String(trade.exit_reason || ""));
      nodes.tradesBody.appendChild(row);
    });
  }

  nodes.tradesPageLabel.textContent = total === 0 ? "0-0 of 0" : `${startIndex + 1}-${endIndex} of ${total}`;
  nodes.tradesPrevButton.disabled = state.tradesPage <= 0;
  nodes.tradesNextButton.disabled = state.tradesPage >= maxPage;
}

function renderTradesEmpty(nodes) {
  nodes.tradesSummary.innerHTML = "";
  nodes.tradesBody.innerHTML = "<tr><td colspan=\"6\">No trades loaded.</td></tr>";
  nodes.tradesPageLabel.textContent = "0-0 of 0";
  nodes.tradesPrevButton.disabled = true;
  nodes.tradesNextButton.disabled = true;
  setStateBadge(nodes.tradesStatus, "idle", "");
  clearCanvas(nodes.tradesChart);
  nodes.chartFallback.textContent = "Select a top variant and click show trades.";
  renderChartLegend(nodes.chartLegend, []);
}

function buildChartData({ markers, segments, trades }) {
  const derivedSegments = segments.length > 0
    ? segments
    : trades.map((trade) => ({
      trade_index: trade.trade_index,
      side: trade.side || trade.direction,
      entry: {
        timestamp: trade.entry_timestamp,
        bar_index: trade.entry_bar_index,
        price: trade.entry_price,
      },
      exit: {
        timestamp: trade.exit_timestamp,
        bar_index: trade.exit_bar_index,
        price: trade.exit_price,
        reason: trade.exit_reason,
      },
      return_pct: trade.return_pct,
    }));
  const derivedMarkers = markers.length > 0
    ? markers
    : trades.flatMap((trade) => [
      {
        kind: "entry",
        timestamp: trade.entry_timestamp,
        bar_index: trade.entry_bar_index,
        price: trade.entry_price,
        side: trade.side || trade.direction,
      },
      {
        kind: "exit",
        timestamp: trade.exit_timestamp,
        bar_index: trade.exit_bar_index,
        price: trade.exit_price,
        side: trade.side || trade.direction,
        exit_reason: trade.exit_reason,
      },
    ]);
  return {
    markers: derivedMarkers.map(normalizeChartPoint).filter((point) => point.price !== null),
    segments: derivedSegments.map(normalizeChartSegment).filter((segment) => (
      segment.entry.price !== null && segment.exit.price !== null
    )),
  };
}

function drawTradeChart({ canvas, chartData }) {
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return;
  }

  const cssWidth = 1200;
  const cssHeight = 420;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = cssWidth;
  const height = cssHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(0, 0, width, height);

  const allPoints = [
    ...chartData.markers,
    ...chartData.segments.flatMap((segment) => [segment.entry, segment.exit]),
  ];
  if (allPoints.length === 0) {
    ctx.fillStyle = "#5b6577";
    ctx.font = "16px Segoe UI, sans-serif";
    ctx.fillText("No chart points available.", 32, 48);
    return;
  }

  const xValues = allPoints.map((point) => point.x).filter((value) => Number.isFinite(value));
  const yValues = allPoints.map((point) => point.price).filter((value) => Number.isFinite(value));
  const xBounds = paddedBounds(Math.min(...xValues), Math.max(...xValues));
  const yBounds = paddedBounds(Math.min(...yValues), Math.max(...yValues));
  const plot = { left: 62, top: 24, right: width - 28, bottom: height - 48 };

  drawChartGrid({ ctx, plot, width, height });
  const xScale = (value) => {
    const ratio = (value - xBounds.min) / (xBounds.max - xBounds.min);
    return plot.left + ratio * (plot.right - plot.left);
  };
  const yScale = (value) => {
    const ratio = (value - yBounds.min) / (yBounds.max - yBounds.min);
    return plot.bottom - ratio * (plot.bottom - plot.top);
  };

  chartData.segments.forEach((segment) => {
    ctx.beginPath();
    ctx.moveTo(xScale(segment.entry.x), yScale(segment.entry.price));
    ctx.lineTo(xScale(segment.exit.x), yScale(segment.exit.price));
    ctx.strokeStyle = segmentColor(segment);
    ctx.lineWidth = 2.5;
    ctx.stroke();
  });

  chartData.markers.forEach((marker) => {
    const x = xScale(marker.x);
    const y = yScale(marker.price);
    ctx.beginPath();
    ctx.arc(x, y, marker.kind === "entry" ? 4 : 5, 0, Math.PI * 2);
    ctx.fillStyle = markerColor(marker);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  });

  ctx.fillStyle = "#5b6577";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.fillText(formatNumber(yBounds.max), 8, plot.top + 5);
  ctx.fillText(formatNumber(yBounds.min), 8, plot.bottom);
}

function drawChartGrid({ ctx, plot, width, height }) {
  ctx.strokeStyle = "#dfe6f3";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#f8fbff";
  ctx.fillRect(0, 0, width, height);
  for (let index = 0; index <= 4; index += 1) {
    const y = plot.top + ((plot.bottom - plot.top) * index) / 4;
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
  }
  for (let index = 0; index <= 6; index += 1) {
    const x = plot.left + ((plot.right - plot.left) * index) / 6;
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom);
    ctx.stroke();
  }
  ctx.strokeStyle = "#aab8d4";
  ctx.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
}

function renderChartLegend(container, items) {
  container.innerHTML = "";
  items.forEach(([reason, label]) => {
    const item = document.createElement("span");
    item.className = "variant-chart-legend-item";
    const marker = document.createElement("span");
    marker.className = "variant-chart-legend-marker";
    marker.dataset.exitReason = reason;
    const text = document.createElement("span");
    text.textContent = label;
    item.append(marker, text);
    container.appendChild(item);
  });
}

async function apiJson(path, options = {}) {
  const headers = { Accept: "application/json" };
  if (options.body) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    credentials: "include",
    ...options,
    headers: {
      ...headers,
      ...(options.headers || {}),
    },
  });
  if (response.status === 401) {
    window.location.assign(`/login?next=${encodeURIComponent(window.location.pathname)}`);
    throw new Error("Authentication is required.");
  }
  if (!response.ok) {
    throw await buildHttpError(response);
  }
  if (response.status === 204) {
    return {};
  }
  return response.json();
}

async function buildHttpError(response) {
  const parsed = await parseApiError(response);
  const error = new Error(parsed.message);
  error.details = parsed.details;
  error.status = response.status;
  error.code = parsed.code;
  return error;
}

async function parseApiError(response) {
  let message = `Request failed with status ${response.status}`;
  let code = "";
  let details = [];
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
  } else {
    try {
      const text = await response.text();
      if (text.trim().length > 0 && response.status !== 502) {
        details = [text.trim().slice(0, 240)];
      }
    } catch (_error) {
      details = [];
    }
  }

  const errorPayload = asRecord(asRecord(payload).error);
  if (Object.keys(errorPayload).length > 0) {
    code = String(errorPayload.code || "");
    message = humanizeBacktestError({ status: response.status, code, fallback: errorPayload.message });
    const payloadDetails = asRecord(errorPayload.details);
    const errorItems = Array.isArray(payloadDetails.errors) ? payloadDetails.errors : [];
    details = errorItems.map(formatValidationItem).filter((item) => item.length > 0);
  } else if (response.status === 502) {
    code = "api.proxy_unavailable";
    message = "API proxy request failed. Retry after the API service is reachable.";
  } else if (payload !== null) {
    const detail = asRecord(payload).detail;
    if (typeof detail === "string" && detail.length > 0) {
      message = detail;
    }
    if (Array.isArray(detail)) {
      details = detail.map(formatValidationItem).filter((item) => item.length > 0);
      message = response.status === 422 ? "Validation error." : message;
    }
  }

  return { message, details, code };
}

function humanizeBacktestError({ status, code, fallback }) {
  const fallbackText = typeof fallback === "string" && fallback.length > 0
    ? fallback
    : `Request failed with status ${status}`;
  const messages = {
    "backtest.forbidden": "This backtest job or variant belongs to another user.",
    "backtest.not_found": "Backtest job or variant was not found.",
    "backtest.idempotency_key_conflict": "The idempotency key conflicts with another request.",
    "backtest.job_not_cancellable": "This job cannot be cancelled in its current state.",
    "backtest.invalid_request": "Backtest request is invalid.",
    "backtest.tp_sl_grid_not_covered": "Requested TP/SL grid is not covered by published artifacts.",
    "backtest.request_too_expensive": "Backtest request exceeds service guardrails.",
    "backtest.rate_limited": "Backtest rate limit was reached. Retry later.",
    "backtest.artifacts_unavailable": "Backtest artifacts are unavailable. Retry after the service recovers.",
    "backtest.queue_saturated": "Backtest queue is saturated. Retry later.",
  };
  return messages[code] || fallbackText;
}

function showPageError(pageRoot, message, details) {
  const banner = pageRoot.querySelector("#backtest-error-banner");
  if (banner !== null) {
    banner.textContent = message;
    banner.classList.remove("hidden");
  }
  const detailsContainer = pageRoot.querySelector("#backtest-validation-errors");
  if (detailsContainer === null) {
    return;
  }
  detailsContainer.innerHTML = "";
  if (!Array.isArray(details) || details.length === 0) {
    detailsContainer.classList.add("hidden");
    return;
  }
  const list = document.createElement("ul");
  details.forEach((detail) => {
    const item = document.createElement("li");
    item.textContent = String(detail);
    list.appendChild(item);
  });
  detailsContainer.appendChild(list);
  detailsContainer.classList.remove("hidden");
}

function clearPageError(pageRoot) {
  const banner = pageRoot.querySelector("#backtest-error-banner");
  if (banner !== null) {
    banner.textContent = "";
    banner.classList.add("hidden");
  }
  const detailsContainer = pageRoot.querySelector("#backtest-validation-errors");
  if (detailsContainer !== null) {
    detailsContainer.innerHTML = "";
    detailsContainer.classList.add("hidden");
  }
}

function normalizeError(error) {
  if (error instanceof Error) {
    return {
      message: error.message,
      details: Array.isArray(error.details) ? error.details : [],
    };
  }
  return { message: "Unexpected error.", details: [] };
}

function requireDataAttr(node, camelCaseName) {
  const value = node.dataset[camelCaseName];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing data attribute: ${camelCaseName}`);
  }
  return value;
}

function renderBacktestPath(template, values) {
  return template
    .replace("{job_id}", encodeURIComponent(String(values.job_id || "")))
    .replace("{variant_key}", encodeURIComponent(String(values.variant_key || "")));
}

function populateIndicatorSelect({ selectNode, state }) {
  const ids = arrayOrEmpty(asRecord(state.defaults).supported_indicator_ids);
  setSelectOptions(selectNode, ids, DEFAULT_INDICATOR_ID);
}

function populateSourceSelect({ indicatorSelect, sourceSelect, state }) {
  const sourceMap = asRecord(asRecord(state.defaults).indicator_sources);
  const sources = arrayOrEmpty(sourceMap[indicatorSelect.value]);
  setSelectOptions(sourceSelect, sources, DEFAULT_SOURCE);
}

function setSelectOptions(selectNode, values, fallback) {
  const previous = selectNode.value;
  const normalized = values.length > 0 ? values.map(String) : [fallback];
  selectNode.innerHTML = "";
  normalized.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectNode.appendChild(option);
  });
  selectNode.value = normalized.includes(previous) ? previous : selectFallback(selectNode, fallback);
}

function selectFallback(selectNode, fallback) {
  const options = Array.from(selectNode.options).map((option) => option.value);
  if (options.includes(fallback)) {
    return fallback;
  }
  return options[0] || "";
}

function syncRiskControls(nodes) {
  nodes.tpSlGrid.classList.toggle("hidden", nodes.riskMode.value !== "tp_sl_grid");
}

function syncStateClass(node, stateValue) {
  node.className = "state-badge";
  if (stateValue) {
    node.classList.add(`state-badge--${stateValue}`);
  }
}

function setStateBadge(node, label, stateValue) {
  node.textContent = String(label || "");
  syncStateClass(node, String(stateValue || "").replaceAll("_", "-"));
}

function buildStateBadge(stateValue) {
  const span = document.createElement("span");
  setStateBadge(span, stateValue, stateValue);
  return span;
}

function setButtonBusy(button, busy) {
  button.disabled = busy;
  button.setAttribute("aria-busy", busy ? "true" : "false");
}

function stopPolling(state) {
  if (state.pollTimer !== null) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

function abortController(controller) {
  if (controller !== null && typeof controller.abort === "function") {
    controller.abort();
  }
}

function isAbortError(error) {
  return error instanceof DOMException && error.name === "AbortError";
}

function readLastSelectedJobId() {
  try {
    return window.sessionStorage.getItem(LAST_JOB_STORAGE_KEY);
  } catch (_error) {
    return null;
  }
}

function storeLastSelectedJobId(jobId) {
  try {
    window.sessionStorage.setItem(LAST_JOB_STORAGE_KEY, jobId);
  } catch (_error) {
    // Session storage is only a convenience for this page.
  }
}

async function copyToClipboard(value) {
  if (!value || !navigator.clipboard) {
    return;
  }
  await navigator.clipboard.writeText(value);
}

function buildLabel(text) {
  const label = document.createElement("label");
  label.textContent = text;
  return label;
}

function buildNumberInput(role, value) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "1";
  input.min = "1";
  input.value = String(value);
  input.dataset.role = role;
  return input;
}

function buildActionButton({
  action,
  label,
  jobId = "",
  variantKey = "",
  variantHash = "",
  copyValue = "",
  className = "",
}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button-link ${className}`.trim();
  button.textContent = label;
  button.setAttribute("data-backtest-action", action);
  if (jobId) {
    button.setAttribute("data-job-id", jobId);
  }
  if (variantKey) {
    button.setAttribute("data-variant-key", variantKey);
  }
  if (variantHash) {
    button.setAttribute("data-variant-hash", variantHash);
  }
  if (copyValue) {
    button.setAttribute("data-copy-value", copyValue);
  }
  return button;
}

function buildCodeLine(label, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "backtest-code-line";
  const name = document.createElement("span");
  name.className = "muted-text";
  name.textContent = `${label}: `;
  const code = document.createElement("code");
  code.textContent = value;
  wrapper.append(name, code);
  return wrapper;
}

function renderKeyValueGrid(container, entries) {
  container.innerHTML = "";
  entries.forEach(([key, value]) => {
    const term = document.createElement("dt");
    term.textContent = String(key);
    const description = document.createElement("dd");
    description.textContent = value === null || typeof value === "undefined" ? "" : String(value);
    container.append(term, description);
  });
}

function appendCell(row, text) {
  const cell = document.createElement("td");
  cell.textContent = text === null || typeof text === "undefined" ? "" : String(text);
  row.appendChild(cell);
}

function requireText(input, path) {
  const value = String(input.value || "").trim();
  if (value.length === 0) {
    throw new Error(`${path} is required.`);
  }
  return value;
}

function readNumber(input, path) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    throw new Error(`${path} must be a number.`);
  }
  return value;
}

function readPositiveInt(input, path) {
  const value = Number(input.value);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${path} must be a positive integer.`);
  }
  return value;
}

function datetimeLocalToUtcString(value, path) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${path} is required.`);
  }
  const normalized = value.length === 16 ? `${value}:00` : value;
  return `${normalized}Z`;
}

function setNumberInput(input, value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    input.value = String(value);
  }
}

function normalizeChartPoint(rawPoint) {
  const point = asRecord(rawPoint);
  return {
    kind: String(point.kind || ""),
    side: String(point.side || ""),
    exit_reason: String(point.exit_reason || point.reason || ""),
    x: chartXValue(point),
    price: numberOrNull(point.price),
  };
}

function normalizeChartSegment(rawSegment) {
  const segment = asRecord(rawSegment);
  const entry = asRecord(segment.entry);
  const exit = asRecord(segment.exit);
  return {
    side: String(segment.side || ""),
    return_pct: numberOrNull(segment.return_pct),
    entry: {
      x: chartXValue(entry),
      price: numberOrNull(entry.price),
    },
    exit: {
      x: chartXValue(exit),
      price: numberOrNull(exit.price),
      exit_reason: String(exit.reason || exit.exit_reason || ""),
    },
  };
}

function chartXValue(point) {
  const timestamp = Date.parse(String(point.timestamp || ""));
  if (Number.isFinite(timestamp)) {
    return timestamp;
  }
  const barIndex = Number(point.bar_index);
  return Number.isFinite(barIndex) ? barIndex : 0;
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function paddedBounds(min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (min === max) {
    const padding = Math.max(1, Math.abs(min) * 0.01);
    return { min: min - padding, max: max + padding };
  }
  const padding = Math.abs(max - min) * 0.08;
  return { min: min - padding, max: max + padding };
}

function segmentColor(segment) {
  const reason = String(asRecord(segment.exit).exit_reason || "");
  if (reason === "tp") {
    return "#1a8d4b";
  }
  if (reason === "sl") {
    return "#c43737";
  }
  if (String(segment.side || "") === "short") {
    return "#b17d16";
  }
  return "#1f5cff";
}

function markerColor(marker) {
  if (marker.kind === "entry") {
    return marker.side === "short" ? "#b17d16" : "#1f5cff";
  }
  if (marker.exit_reason === "tp") {
    return "#1a8d4b";
  }
  if (marker.exit_reason === "sl") {
    return "#c43737";
  }
  return "#6d7485";
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function formatValidationItem(item) {
  if (typeof item === "string") {
    return item;
  }
  const obj = asRecord(item);
  const path = obj.path || obj.loc || "";
  const message = obj.message || obj.msg || JSON.stringify(item);
  return path ? `${path}: ${message}` : String(message);
}

function compactMetrics(metrics) {
  const preferred = [
    "total_return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "trade_count",
    "sharpe_trades",
    "win_rate_pct",
  ];
  return preferred
    .filter((key) => Object.prototype.hasOwnProperty.call(metrics, key))
    .map((key) => `${key}=${formatMetricValue(metrics[key])}`)
    .join(" | ");
}

function compactRecord(record) {
  return Object.entries(record)
    .map(([key, value]) => `${key}=${formatMetricValue(value)}`)
    .join(" | ");
}

function compactHash(value) {
  const text = String(value || "");
  if (text.length <= 16) {
    return text;
  }
  return `${text.slice(0, 10)}...${text.slice(-6)}`;
}

function formatTpSl(tp, sl) {
  const tpText = tp === null || typeof tp === "undefined" ? "TP -" : `TP ${formatNumber(tp)}%`;
  const slText = sl === null || typeof sl === "undefined" ? "SL -" : `SL ${formatNumber(sl)}%`;
  return `${tpText} / ${slText}`;
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (value === null || typeof value === "undefined") {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "";
  }
  return Number.isInteger(number) ? String(number) : number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatSeconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "";
  }
  return `${number.toFixed(3)}s`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "0%";
  }
  return `${Math.max(0, Math.min(100, number))}%`;
}

function formatPercentValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "";
  }
  return `${formatNumber(number)}%`;
}

function formatTimestamp(value) {
  if (value === null || typeof value === "undefined" || String(value).length === 0) {
    return "";
  }
  return String(value).replace("T", " ").replace("Z", "");
}

function arrayOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function asRecord(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
}
