const BACKTEST_PAGE_SELECTOR = "[data-backtest-page]";
const RANGE_PRESET_DAYS = new Map([
  ["7d", 7],
  ["30d", 30],
  ["90d", 90],
]);

document.addEventListener("DOMContentLoaded", () => {
  const pageRoot = document.querySelector(BACKTEST_PAGE_SELECTOR);
  if (pageRoot === null) {
    return;
  }
  initBacktestPage(pageRoot);
});

function initBacktestPage(pageRoot) {
  const backtestsPath = requireDataAttr(pageRoot, "apiBacktestsPath");
  const backtestVariantReportPath = requireDataAttr(pageRoot, "apiBacktestVariantReportPath");
  const strategiesPath = requireDataAttr(pageRoot, "apiStrategiesPath");
  const marketsPath = requireDataAttr(pageRoot, "apiMarketsPath");
  const instrumentsPath = requireDataAttr(pageRoot, "apiInstrumentsPath");
  const indicatorsPath = requireDataAttr(pageRoot, "apiIndicatorsPath");
  const strategyBuilderPath = requireDataAttr(pageRoot, "strategyBuilderPath");
  const historyPath = requireDataAttr(pageRoot, "historyPath");
  const runSummaryPathTemplate = requireDataAttr(pageRoot, "runSummaryPathTemplate");
  const jobsListPath = requireDataAttr(pageRoot, "jobsListPath");
  const prefillQueryParam = requireDataAttr(pageRoot, "prefillQueryParam");
  const prefillStorage = requireDataAttr(pageRoot, "prefillStorage");
  const jobContextStoragePrefix = requireDataAttr(pageRoot, "jobContextStoragePrefix");
  const runtimeDefaultsPath = requireDataAttr(pageRoot, "apiBacktestRuntimeDefaultsPath");

  const form = pageRoot.querySelector("#backtest-form");
  const modeTemplate = pageRoot.querySelector("input[name=\"backtest-mode\"][value=\"template\"]");
  const modeSaved = pageRoot.querySelector("input[name=\"backtest-mode\"][value=\"saved\"]");
  const templateModeSection = pageRoot.querySelector("#backtest-template-mode");
  const savedModeSection = pageRoot.querySelector("#backtest-saved-mode");
  const launchStatus = pageRoot.querySelector("#backtest-launch-status");
  const launchStatusMessage = pageRoot.querySelector("#backtest-launch-status-message");
  const launchStatusLink = pageRoot.querySelector("#backtest-launch-status-link");
  const runButton = pageRoot.querySelector("#backtest-run-button");
  const runLoading = pageRoot.querySelector("#backtest-run-loading");

  const marketSelect = pageRoot.querySelector("#backtest-market-id");
  const symbolQuery = pageRoot.querySelector("#backtest-symbol-query");
  const symbolValue = pageRoot.querySelector("#backtest-symbol-value");
  const selectedSymbol = pageRoot.querySelector("#backtest-selected-symbol");
  const suggestionsList = pageRoot.querySelector("#backtest-symbol-suggestions");
  const timeframeSelect = pageRoot.querySelector("#backtest-timeframe");
  const addIndicatorButton = pageRoot.querySelector("#backtest-add-indicator");
  const blocksContainer = pageRoot.querySelector("#backtest-indicator-blocks");

  const strategiesSelect = pageRoot.querySelector("#backtest-strategy-id");
  const refreshStrategiesButton = pageRoot.querySelector("#backtest-refresh-strategies");

  const rangePresetSelect = pageRoot.querySelector("#backtest-range-preset");
  const rangeStartInput = pageRoot.querySelector("#backtest-range-start");
  const rangeEndInput = pageRoot.querySelector("#backtest-range-end");

  const directionModeSelect = pageRoot.querySelector("#backtest-direction-mode");
  const sizingModeSelect = pageRoot.querySelector("#backtest-sizing-mode");
  const rankingPrimaryMetricSelect = pageRoot.querySelector("#backtest-ranking-primary-metric");
  const rankingSecondaryMetricSelect = pageRoot.querySelector("#backtest-ranking-secondary-metric");
  const executionInitCash = pageRoot.querySelector("#backtest-exec-init-cash");
  const executionFeePct = pageRoot.querySelector("#backtest-exec-fee-pct");
  const executionSlippagePct = pageRoot.querySelector("#backtest-exec-slippage-pct");
  const executionFixedQuote = pageRoot.querySelector("#backtest-exec-fixed-quote");
  const executionSafeProfitPercent = pageRoot.querySelector("#backtest-exec-safe-profit-percent");
  const riskSlEnabled = pageRoot.querySelector("#backtest-risk-sl-enabled");
  const riskSlMode = pageRoot.querySelector("#backtest-risk-sl-mode");
  const riskSlValues = pageRoot.querySelector("#backtest-risk-sl-values");
  const riskSlStart = pageRoot.querySelector("#backtest-risk-sl-start");
  const riskSlStop = pageRoot.querySelector("#backtest-risk-sl-stop");
  const riskSlStep = pageRoot.querySelector("#backtest-risk-sl-step");
  const riskSlPct = pageRoot.querySelector("#backtest-risk-sl-pct");
  const riskTpEnabled = pageRoot.querySelector("#backtest-risk-tp-enabled");
  const riskTpMode = pageRoot.querySelector("#backtest-risk-tp-mode");
  const riskTpValues = pageRoot.querySelector("#backtest-risk-tp-values");
  const riskTpStart = pageRoot.querySelector("#backtest-risk-tp-start");
  const riskTpStop = pageRoot.querySelector("#backtest-risk-tp-stop");
  const riskTpStep = pageRoot.querySelector("#backtest-risk-tp-step");
  const riskTpPct = pageRoot.querySelector("#backtest-risk-tp-pct");
  const topNInput = pageRoot.querySelector("#backtest-top-n");
  const preselectInput = pageRoot.querySelector("#backtest-preselect");
  const topTradesInput = pageRoot.querySelector("#backtest-top-trades-n");
  const warmupBarsInput = pageRoot.querySelector("#backtest-warmup-bars");
  const runtimeDefaultsHint = pageRoot.querySelector("#backtest-runtime-defaults-hint");

  const resultsPanel = pageRoot.querySelector("#backtest-results-panel");
  const resultsMeta = pageRoot.querySelector("#backtest-results-meta");
  const variantsBody = pageRoot.querySelector("#backtest-variants-body");

  if (
    form === null
    || modeTemplate === null
    || modeSaved === null
    || templateModeSection === null
    || savedModeSection === null
    || launchStatus === null
    || launchStatusMessage === null
    || launchStatusLink === null
    || runButton === null
    || runLoading === null
    || marketSelect === null
    || symbolQuery === null
    || symbolValue === null
    || selectedSymbol === null
    || suggestionsList === null
    || timeframeSelect === null
    || addIndicatorButton === null
    || blocksContainer === null
    || strategiesSelect === null
    || refreshStrategiesButton === null
    || rangePresetSelect === null
    || rangeStartInput === null
    || rangeEndInput === null
    || directionModeSelect === null
    || sizingModeSelect === null
    || rankingPrimaryMetricSelect === null
    || rankingSecondaryMetricSelect === null
    || executionInitCash === null
    || executionFeePct === null
    || executionSlippagePct === null
    || executionFixedQuote === null
    || executionSafeProfitPercent === null
    || riskSlEnabled === null
    || riskSlMode === null
    || riskSlValues === null
    || riskSlStart === null
    || riskSlStop === null
    || riskSlStep === null
    || riskSlPct === null
    || riskTpEnabled === null
    || riskTpMode === null
    || riskTpValues === null
    || riskTpStart === null
    || riskTpStop === null
    || riskTpStep === null
    || riskTpPct === null
    || topNInput === null
    || preselectInput === null
    || topTradesInput === null
    || warmupBarsInput === null
    || runtimeDefaultsHint === null
    || resultsPanel === null
    || resultsMeta === null
    || variantsBody === null
  ) {
    return;
  }

  const state = {
    mode: "template",
    isRunning: false,
    markets: [],
    marketsById: new Map(),
    indicators: [],
    indicatorsById: new Map(),
    indicatorDescriptors: [],
    strategiesById: new Map(),
    blocks: [],
    nextBlockNumber: 1,
    searchDebounceId: 0,
    instrumentsAbortController: null,
    latestRun: null,
    latestRunToken: 0,
    runtimeDefaults: null,
    executionFeeDirty: false,
    applyingFeeDefault: false,
    reportCacheByVariantKey: new Map(),
    reportLoadingKeys: new Set(),
    reportErrorsByVariantKey: new Map(),
    reportCacheHitKeys: new Set(),
  };

  const hideLaunchStatus = () => {
    launchStatus.classList.add("hidden");
    launchStatusMessage.textContent = "";
    launchStatusLink.textContent = "/backtests/history";
    launchStatusLink.href = historyPath;
  };

  const clearSelectedSymbol = () => {
    symbolValue.value = "";
    selectedSymbol.textContent = "Selected symbol: none";
  };

  const showLaunchStatus = ({ message, linkHref, linkLabel }) => {
    launchStatusMessage.textContent = message;
    launchStatusLink.href = linkHref;
    launchStatusLink.textContent = linkLabel;
    launchStatus.classList.remove("hidden");
  };

  const readRuntimeContracts = () => (
    asRecord(asRecord(state.runtimeDefaults).contracts)
  );

  const readRuntimeRequestTimeframesContract = () => (
    asRecord(readRuntimeContracts().request_timeframes)
  );

  const readRuntimeSummaryContract = () => (
    asRecord(readRuntimeContracts().summary)
  );

  const readRuntimeLaunchContract = () => (
    asRecord(readRuntimeContracts().launch)
  );

  const readAllowedTimeframes = () => {
    const contract = readRuntimeRequestTimeframesContract();
    return Array.isArray(contract.allowed)
      ? contract.allowed.map((item) => String(item).trim()).filter((item) => item.length > 0)
      : [];
  };

  const readRankingMetrics = () => {
    const contract = readRuntimeSummaryContract();
    return Array.isArray(contract.ranking_metrics)
      ? contract.ranking_metrics
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
      : [];
  };

  const readSupportedIndicatorIds = () => {
    const contract = readRuntimeLaunchContract();
    return Array.isArray(contract.supported_indicator_ids)
      ? contract.supported_indicator_ids
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
      : [];
  };

  const readSourceCatalogValues = (indicatorId) => {
    const normalizedIndicatorId = String(indicatorId || "").trim();
    if (normalizedIndicatorId.length === 0) {
      return [];
    }
    const contract = readRuntimeLaunchContract();
    const sourceCatalog = asRecord(contract.source_values_by_indicator_id);
    const sourceValues = sourceCatalog[normalizedIndicatorId];
    return Array.isArray(sourceValues)
      ? sourceValues
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
      : [];
  };

  const readTopNMax = () => {
    const contract = readRuntimeSummaryContract();
    return readFiniteInteger(contract.top_n_max);
  };

  const readIndicatorSourceDefaultSelections = ({ descriptor, allowedValues }) => {
    if (allowedValues.length === 0) {
      return [];
    }
    const descriptorInputs = Array.isArray(descriptor.inputs) ? descriptor.inputs : [];
    const sourceSpec = descriptorInputs.find(
      (item) => String(asRecord(item).name || "").trim() === "source",
    );
    if (!sourceSpec) {
      return [allowedValues[0]];
    }
    const defaultSpec = asRecord(asRecord(sourceSpec).default);
    if (defaultSpec.mode !== "explicit" || !Array.isArray(defaultSpec.values)) {
      return [allowedValues[0]];
    }
    const defaultValuesSet = new Set(
      defaultSpec.values
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0),
    );
    const selectedDefaults = allowedValues.filter((item) => defaultValuesSet.has(item));
    return selectedDefaults.length > 0 ? selectedDefaults : [allowedValues[0]];
  };

  const refreshTemplateState = () => {
    hideLaunchStatus();
    updateRunAvailability();
  };

  const updateModeSections = () => {
    const isTemplate = state.mode === "template";
    templateModeSection.classList.toggle("hidden", !isTemplate);
    savedModeSection.classList.toggle("hidden", isTemplate);
  };

  const updateRunAvailability = () => {
    runButton.disabled = true;

    if (state.isRunning) {
      return;
    }
    if (state.runtimeDefaults === null) {
      return;
    }
    if (state.mode === "saved" && String(strategiesSelect.value || "").trim().length === 0) {
      return;
    }
    if (state.mode === "template") {
      const hasMissingSourceSelection = state.blocks.some((block) => {
        const allowedSourceValues = readSourceCatalogValues(block.indicatorId);
        if (allowedSourceValues.length === 0) {
          return false;
        }
        return readSelectedSourceValues({
          allowedValues: allowedSourceValues,
          rawSelectedValues: block.sourceSelections,
        }).length === 0;
      });
      if (
        Number(marketSelect.value || "0") <= 0
        || String(symbolValue.value || "").trim().length === 0
        || String(timeframeSelect.value || "").trim().length === 0
        || state.blocks.length === 0
        || state.indicators.length === 0
        || hasMissingSourceSelection
      ) {
        return;
      }
    }
    runButton.disabled = false;
  };

  const selectLabelForInput = (inputId) => (
    pageRoot.querySelector(`label[for="${inputId}"]`)
  );

  const toggleNodesVisibility = ({ nodes, visible }) => {
    nodes.forEach((node) => {
      if (node !== null) {
        node.classList.toggle("hidden", !visible);
      }
    });
  };

  const riskUiSections = {
    sl: {
      enabledNode: riskSlEnabled,
      modeNode: riskSlMode,
      sharedNodes: [
        selectLabelForInput("backtest-risk-sl-mode"),
        riskSlMode,
        selectLabelForInput("backtest-risk-sl-pct"),
        riskSlPct,
      ],
      explicitNodes: [
        selectLabelForInput("backtest-risk-sl-values"),
        riskSlValues,
      ],
      rangeNodes: [
        selectLabelForInput("backtest-risk-sl-start"),
        riskSlStart,
        selectLabelForInput("backtest-risk-sl-stop"),
        riskSlStop,
        selectLabelForInput("backtest-risk-sl-step"),
        riskSlStep,
      ],
    },
    tp: {
      enabledNode: riskTpEnabled,
      modeNode: riskTpMode,
      sharedNodes: [
        selectLabelForInput("backtest-risk-tp-mode"),
        riskTpMode,
        selectLabelForInput("backtest-risk-tp-pct"),
        riskTpPct,
      ],
      explicitNodes: [
        selectLabelForInput("backtest-risk-tp-values"),
        riskTpValues,
      ],
      rangeNodes: [
        selectLabelForInput("backtest-risk-tp-start"),
        riskTpStart,
        selectLabelForInput("backtest-risk-tp-stop"),
        riskTpStop,
        selectLabelForInput("backtest-risk-tp-step"),
        riskTpStep,
      ],
    },
  };

  const updateRiskUiSectionVisibility = (sectionKey) => {
    const section = riskUiSections[sectionKey];
    if (!section) {
      return;
    }
    const enabled = section.enabledNode.checked;
    const mode = normalizeAxisMode(section.modeNode.value);
    const useRange = enabled && mode === "range";

    toggleNodesVisibility({ nodes: section.sharedNodes, visible: enabled });
    toggleNodesVisibility({ nodes: section.explicitNodes, visible: enabled && !useRange });
    toggleNodesVisibility({ nodes: section.rangeNodes, visible: useRange });
  };

  const updateRiskUiVisibility = () => {
    updateRiskUiSectionVisibility("sl");
    updateRiskUiSectionVisibility("tp");
  };

  const readFiniteNumber = (value) => {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue : null;
  };

  const readFiniteInteger = (value) => {
    const numberValue = readFiniteNumber(value);
    if (numberValue === null) {
      return null;
    }
    return Math.trunc(numberValue);
  };

  const readRuntimeDefaultsFeeMap = () => {
    const defaultsRecord = asRecord(state.runtimeDefaults);
    const executionRecord = asRecord(defaultsRecord.execution);
    return asRecord(executionRecord.fee_pct_default_by_market_id);
  };

  const readRuntimeDefaultFeePct = ({ marketId, allowFallbackToFirst }) => {
    const feeMap = readRuntimeDefaultsFeeMap();
    if (marketId > 0) {
      const marketFee = readFiniteNumber(feeMap[String(marketId)]);
      if (marketFee !== null) {
        return marketFee;
      }
    }
    if (!allowFallbackToFirst) {
      return null;
    }
    const sortedKeys = Object.keys(feeMap).sort(compareStableStrings);
    for (const marketKey of sortedKeys) {
      const fallbackFee = readFiniteNumber(feeMap[marketKey]);
      if (fallbackFee !== null) {
        return fallbackFee;
      }
    }
    return null;
  };

  const setExecutionFeeDefaultValue = (feePct) => {
    state.applyingFeeDefault = true;
    executionFeePct.value = String(feePct);
    state.applyingFeeDefault = false;
  };

  const applyDefaultFeeForSelectedMarket = ({ force, allowFallbackToFirst }) => {
    if (!force && state.executionFeeDirty) {
      return;
    }
    if (state.runtimeDefaults === null) {
      return;
    }
    const marketId = Number(marketSelect.value || "0");
    const defaultFee = readRuntimeDefaultFeePct({ marketId, allowFallbackToFirst });
    if (defaultFee === null) {
      return;
    }
    setExecutionFeeDefaultValue(defaultFee);
  };

  const renderRuntimeDefaultsHint = () => {
    if (state.runtimeDefaults === null) {
      runtimeDefaultsHint.textContent = "";
      runtimeDefaultsHint.classList.add("hidden");
      return;
    }
    const defaultsRecord = asRecord(state.runtimeDefaults);
    const summaryContract = readRuntimeSummaryContract();
    const launchContract = readRuntimeLaunchContract();
    const preselectDefault = readFiniteInteger(defaultsRecord.preselect_default);
    const warmupBarsDefault = readFiniteInteger(defaultsRecord.warmup_bars_default);
    const topNDefault = readFiniteInteger(summaryContract.top_n_default);
    const topNMax = readFiniteInteger(summaryContract.top_n_max);
    const requestTimeframes = readAllowedTimeframes();
    const rankingMetrics = readRankingMetrics();
    runtimeDefaultsHint.textContent = [
      "Runtime defaults:",
      `top_n_default=${topNDefault ?? "-"}`,
      `top_n_max=${topNMax ?? "-"}`,
      `preselect=${preselectDefault ?? "-"}`,
      `warmup_bars=${warmupBarsDefault ?? "-"}`,
      `request_timeframes=${requestTimeframes.join(",") || "-"}`,
      `ranking_metrics=${rankingMetrics.join(",") || "-"}`,
      `auto_preflight_enabled=${String(Boolean(launchContract.auto_preflight_enabled))}`,
      `auto_fallback_to_background_enabled=${String(Boolean(launchContract.auto_fallback_to_background_enabled))}.`,
      "Form top_n maps deterministically to request top_k.",
    ].join(" ");
    runtimeDefaultsHint.classList.remove("hidden");
  };

  const populateTimeframeOptions = () => {
    const allowedTimeframes = readAllowedTimeframes();
    const previousValue = String(timeframeSelect.value || "").trim();
    timeframeSelect.innerHTML = "";
    if (allowedTimeframes.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No request_timeframes available";
      timeframeSelect.appendChild(option);
      return;
    }

    allowedTimeframes.forEach((timeframe) => {
      const option = document.createElement("option");
      option.value = timeframe;
      option.textContent = timeframe;
      timeframeSelect.appendChild(option);
    });
    timeframeSelect.value = allowedTimeframes.includes(previousValue)
      ? previousValue
      : allowedTimeframes[0];
  };

  const repopulateRankingMetricSelect = ({ selectNode, includeEmptyOption, emptyLabel }) => {
    const previousValue = String(selectNode.value || "").trim();
    const metrics = readRankingMetrics();
    selectNode.innerHTML = "";
    if (includeEmptyOption) {
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = emptyLabel;
      selectNode.appendChild(emptyOption);
    }
    metrics.forEach((metric) => {
      const option = document.createElement("option");
      option.value = metric;
      option.textContent = metric;
      selectNode.appendChild(option);
    });
    if (metrics.includes(previousValue)) {
      selectNode.value = previousValue;
    }
  };

  const rebuildIndicatorCatalog = () => {
    if (state.runtimeDefaults === null || state.indicatorDescriptors.length === 0) {
      state.indicators = [];
      state.indicatorsById = new Map();
      blocksContainer.innerHTML = "<p class=\"muted-text\">Waiting for runtime defaults...</p>";
      updateRunAvailability();
      return;
    }

    const descriptorsById = new Map(
      state.indicatorDescriptors
        .map((indicator) => [String(indicator.indicator_id || "").trim(), indicator])
        .filter((entry) => entry[0].length > 0),
    );
    const supportedIndicatorIds = readSupportedIndicatorIds();
    const missingIndicatorIds = supportedIndicatorIds.filter((item) => !descriptorsById.has(item));
    state.indicators = supportedIndicatorIds
      .map((indicatorId) => descriptorsById.get(indicatorId))
      .filter((indicator) => indicator && typeof indicator === "object");
    state.indicatorsById = new Map(
      state.indicators.map((indicator) => [String(indicator.indicator_id), indicator]),
    );
    state.blocks = state.blocks.filter((block) => state.indicatorsById.has(block.indicatorId));

    if (missingIndicatorIds.length > 0) {
      showPageError(
        pageRoot,
        "Runtime defaults reference indicators that are unavailable in /api/indicators.",
        missingIndicatorIds.map((indicatorId) => `missing indicator descriptor: ${indicatorId}`),
      );
    }

    if (state.indicators.length === 0) {
      blocksContainer.innerHTML = "<p class=\"muted-text\">No supported_indicator_ids are available.</p>";
      updateRunAvailability();
      return;
    }

    if (state.blocks.length === 0) {
      addIndicatorBlock();
      return;
    }
    renderIndicatorBlocks();
    updateRunAvailability();
  };

  const applyRuntimeDefaultsToAdvancedFields = () => {
    if (state.runtimeDefaults === null) {
      return;
    }
    const defaultsRecord = asRecord(state.runtimeDefaults);
    const executionRecord = asRecord(defaultsRecord.execution);
    const rankingRecord = asRecord(defaultsRecord.ranking);
    const summaryContract = readRuntimeSummaryContract();

    const warmupBarsDefault = readFiniteInteger(defaultsRecord.warmup_bars_default);
    const topNDefault = readFiniteInteger(summaryContract.top_n_default);
    const topNMax = readFiniteInteger(summaryContract.top_n_max);
    const preselectDefault = readFiniteInteger(defaultsRecord.preselect_default);
    const topTradesDefault = readFiniteInteger(defaultsRecord.top_trades_n_default);

    if (warmupBarsDefault !== null) {
      warmupBarsInput.value = String(warmupBarsDefault);
    }
    if (topNDefault !== null) {
      topNInput.value = String(topNDefault);
    }
    if (topNMax !== null && topNMax > 0) {
      topNInput.max = String(topNMax);
    }
    if (preselectDefault !== null) {
      preselectInput.value = String(preselectDefault);
    }
    if (topTradesDefault !== null) {
      topTradesInput.value = String(topTradesDefault);
    }
    populateTimeframeOptions();
    repopulateRankingMetricSelect({
      selectNode: rankingPrimaryMetricSelect,
      includeEmptyOption: true,
      emptyLabel: "Use runtime default",
    });
    repopulateRankingMetricSelect({
      selectNode: rankingSecondaryMetricSelect,
      includeEmptyOption: true,
      emptyLabel: "None (tie-break by variant_key)",
    });
    const primaryMetricDefault = readOptionalRankingMetricLiteral({
      rawValue: rankingRecord.primary_metric_default,
      fieldLabel: "ranking.primary_metric_default",
    });
    const secondaryMetricDefault = readOptionalRankingMetricLiteral({
      rawValue: rankingRecord.secondary_metric_default,
      fieldLabel: "ranking.secondary_metric_default",
    });
    rankingPrimaryMetricSelect.value = primaryMetricDefault || "";
    if (
      secondaryMetricDefault !== null
      && primaryMetricDefault !== null
      && secondaryMetricDefault === primaryMetricDefault
    ) {
      rankingSecondaryMetricSelect.value = "";
    } else {
      rankingSecondaryMetricSelect.value = secondaryMetricDefault || "";
    }

    const initCashDefault = readFiniteNumber(executionRecord.init_cash_quote_default);
    const fixedQuoteDefault = readFiniteNumber(executionRecord.fixed_quote_default);
    const safeProfitDefault = readFiniteNumber(executionRecord.safe_profit_percent_default);
    const slippageDefault = readFiniteNumber(executionRecord.slippage_pct_default);

    if (initCashDefault !== null) {
      executionInitCash.value = String(initCashDefault);
    }
    if (fixedQuoteDefault !== null) {
      executionFixedQuote.value = String(fixedQuoteDefault);
    }
    if (safeProfitDefault !== null) {
      executionSafeProfitPercent.value = String(safeProfitDefault);
    }
    if (slippageDefault !== null) {
      executionSlippagePct.value = String(slippageDefault);
    }

    applyDefaultFeeForSelectedMarket({ force: true, allowFallbackToFirst: true });
    rebuildIndicatorCatalog();
    renderRuntimeDefaultsHint();
    updateRunAvailability();
  };

  const loadRuntimeDefaults = async () => {
    try {
      const response = await fetch(runtimeDefaultsPath, { credentials: 'include' });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      state.runtimeDefaults = asRecord(await response.json());
      applyRuntimeDefaultsToAdvancedFields();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  const renderSuggestionButtons = (symbols) => {
    suggestionsList.innerHTML = "";
    symbols.forEach((symbol) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button-link button-link--secondary";
      button.textContent = symbol;
      button.addEventListener("click", () => {
        symbolValue.value = symbol;
        symbolQuery.value = symbol;
        selectedSymbol.textContent = `Selected symbol: ${symbol}`;
        suggestionsList.innerHTML = "";
        refreshTemplateState();
      });
      item.appendChild(button);
      suggestionsList.appendChild(item);
    });
  };

  const fetchInstruments = async () => {
    const marketId = Number(marketSelect.value || "0");
    const query = symbolQuery.value.trim();
    if (marketId <= 0 || query.length === 0) {
      suggestionsList.innerHTML = "";
      return;
    }

    if (state.instrumentsAbortController !== null) {
      state.instrumentsAbortController.abort();
    }
    const controller = new AbortController();
    state.instrumentsAbortController = controller;

    const requestUrl = new URL(instrumentsPath, window.location.origin);
    requestUrl.searchParams.set("market_id", String(marketId));
    requestUrl.searchParams.set("q", query);
    requestUrl.searchParams.set("limit", "20");

    try {
      const response = await fetch(requestUrl.toString(), {
        credentials: 'include',
        signal: controller.signal,
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      const items = Array.isArray(payload.items) ? payload.items : [];
      const symbols = items
        .map((item) => String(asRecord(item).symbol || "").trim())
        .filter((symbol) => symbol.length > 0);
      renderSuggestionButtons(symbols);
    } catch (error) {
      if (error && error.name === "AbortError") {
        return;
      }
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  const scheduleInstrumentSearch = () => {
    if (state.searchDebounceId !== 0) {
      window.clearTimeout(state.searchDebounceId);
    }
    state.searchDebounceId = window.setTimeout(() => {
      fetchInstruments();
    }, 220);
  };

  const normalizeAxisMode = (rawMode) => {
    const mode = String(rawMode || "explicit").trim().toLowerCase();
    return mode === "range" ? "range" : "explicit";
  };

  const readDefaultFieldValue = (defaultSpec) => {
    const spec = asRecord(defaultSpec);
    if (spec.mode === "explicit" && Array.isArray(spec.values) && spec.values.length > 0) {
      return String(spec.values[0]);
    }
    if (spec.mode === "range" && typeof spec.start !== "undefined") {
      return String(spec.start);
    }
    return "";
  };

  const readDefaultExplicitValuesCsv = (defaultSpec) => {
    const spec = asRecord(defaultSpec);
    if (spec.mode !== "explicit" || !Array.isArray(spec.values) || spec.values.length === 0) {
      return "";
    }
    return spec.values.map((item) => String(item)).join(",");
  };

  const readDefaultRangeFields = (defaultSpec) => {
    const spec = asRecord(defaultSpec);
    if (spec.mode !== "range") {
      return {
        start: "",
        stopIncl: "",
        step: "",
      };
    }
    return {
      start: typeof spec.start === "undefined" ? "" : String(spec.start),
      stopIncl: typeof spec.stop_incl === "undefined" ? "" : String(spec.stop_incl),
      step: typeof spec.step === "undefined" ? "" : String(spec.step),
    };
  };

  const ensureParamAxisState = ({ block, paramRecord }) => {
    const paramName = String(paramRecord.name || "");
    if (paramName.length === 0) {
      return null;
    }
    if (!block.paramAxes || typeof block.paramAxes !== "object" || Array.isArray(block.paramAxes)) {
      block.paramAxes = {};
    }

    const defaultSpec = asRecord(paramRecord.default);
    const defaultMode = normalizeAxisMode(defaultSpec.mode);
    const defaultRange = readDefaultRangeFields(defaultSpec);
    const enumDefault = Array.isArray(paramRecord.enum_values) && paramRecord.enum_values.length > 0
      ? String(paramRecord.enum_values[0])
      : "";
    const explicitDefault = readDefaultExplicitValuesCsv(defaultSpec);
    const fallbackExplicit = explicitDefault.length > 0 ? explicitDefault : enumDefault;
    const existing = asRecord(block.paramAxes[paramName]);
    const hasExistingMode = typeof existing.mode === "string" && existing.mode.trim().length > 0;

    const axisState = {
      mode: hasExistingMode ? normalizeAxisMode(existing.mode) : defaultMode,
      explicitValues: String(existing.explicitValues || ""),
      rangeStart: String(existing.rangeStart || ""),
      rangeStopIncl: String(existing.rangeStopIncl || ""),
      rangeStep: String(existing.rangeStep || ""),
    };
    if (axisState.explicitValues.trim().length === 0 && fallbackExplicit.length > 0) {
      axisState.explicitValues = fallbackExplicit;
    }
    if (axisState.rangeStart.trim().length === 0 && defaultRange.start.length > 0) {
      axisState.rangeStart = defaultRange.start;
    }
    if (axisState.rangeStopIncl.trim().length === 0 && defaultRange.stopIncl.length > 0) {
      axisState.rangeStopIncl = defaultRange.stopIncl;
    }
    if (axisState.rangeStep.trim().length === 0 && defaultRange.step.length > 0) {
      axisState.rangeStep = defaultRange.step;
    }
    block.paramAxes[paramName] = axisState;
    return axisState;
  };

  const readParamLabels = ({ indicatorId, paramName }) => {
    if (paramName === "window" && indicatorId.startsWith("ma.")) {
      return {
        axisLabel: "window period",
        stepLabel: "window grid step",
      };
    }
    return {
      axisLabel: paramName,
      stepLabel: `${paramName} step`,
    };
  };

  const readSelectedSourceValues = ({ allowedValues, rawSelectedValues }) => {
    if (allowedValues.length === 0) {
      return [];
    }
    const selectedValues = Array.isArray(rawSelectedValues)
      ? rawSelectedValues
        .map((item) => String(item).trim())
        .filter((item) => item.length > 0)
      : [];
    const selectedValuesSet = new Set(selectedValues);
    return allowedValues.filter((item) => selectedValuesSet.has(item));
  };

  const ensureDefaultsForBlock = (block) => {
    const descriptor = state.indicatorsById.get(block.indicatorId);
    if (!descriptor) {
      return;
    }
    const allowedSourceValues = readSourceCatalogValues(block.indicatorId);
    const selectedSourceValues = readSelectedSourceValues({
      allowedValues: allowedSourceValues,
      rawSelectedValues: block.sourceSelections,
    });
    block.sourceSelections = selectedSourceValues.length > 0
      ? selectedSourceValues
      : readIndicatorSourceDefaultSelections({
        descriptor,
        allowedValues: allowedSourceValues,
      });

    const params = Array.isArray(descriptor.params) ? descriptor.params : [];
    const descriptorParamNames = new Set();
    params.forEach((paramSpec) => {
      const record = asRecord(paramSpec);
      const paramName = String(record.name || "");
      if (paramName.length === 0) {
        return;
      }
      descriptorParamNames.add(paramName);
      ensureParamAxisState({ block, paramRecord: record });
    });

    if (block.paramAxes && typeof block.paramAxes === "object" && !Array.isArray(block.paramAxes)) {
      Object.keys(block.paramAxes).forEach((paramName) => {
        if (!descriptorParamNames.has(paramName)) {
          delete block.paramAxes[paramName];
        }
      });
    }
  };

  const addIndicatorBlock = () => {
    if (state.indicators.length === 0) {
      return;
    }
    const firstIndicator = state.indicators[0];
    const block = {
      uid: `backtest-indicator-${state.nextBlockNumber}`,
      indicatorId: String(firstIndicator.indicator_id),
      sourceSelections: [],
      paramAxes: {},
    };
    state.nextBlockNumber += 1;
    ensureDefaultsForBlock(block);
    state.blocks.push(block);
    renderIndicatorBlocks();
    refreshTemplateState();
  };

  const renderIndicatorBlocks = () => {
    blocksContainer.innerHTML = "";
    if (state.blocks.length === 0) {
      const emptyNode = document.createElement("p");
      emptyNode.className = "muted-text";
      emptyNode.textContent = "No indicator grids yet.";
      blocksContainer.appendChild(emptyNode);
      return;
    }

    state.blocks.forEach((block, index) => {
      ensureDefaultsForBlock(block);
      const descriptor = state.indicatorsById.get(block.indicatorId);
      if (!descriptor) {
        return;
      }

      const card = document.createElement("section");
      card.className = "indicator-card";

      const header = document.createElement("div");
      header.className = "indicator-card-header";
      const title = document.createElement("h3");
      title.textContent = `Grid #${index + 1}`;
      header.appendChild(title);

      const controls = document.createElement("div");
      controls.className = "inline-actions";
      controls.appendChild(
        buildActionButton({
          label: "Up",
          disabled: index === 0,
          onClick: () => {
            const moved = state.blocks.splice(index, 1)[0];
            state.blocks.splice(index - 1, 0, moved);
            renderIndicatorBlocks();
            refreshTemplateState();
          },
        }),
      );
      controls.appendChild(
        buildActionButton({
          label: "Down",
          disabled: index === state.blocks.length - 1,
          onClick: () => {
            const moved = state.blocks.splice(index, 1)[0];
            state.blocks.splice(index + 1, 0, moved);
            renderIndicatorBlocks();
            refreshTemplateState();
          },
        }),
      );
      controls.appendChild(
        buildActionButton({
          label: "Remove",
          className: "button-link--danger",
          onClick: () => {
            state.blocks = state.blocks.filter((candidate) => candidate.uid !== block.uid);
            renderIndicatorBlocks();
            refreshTemplateState();
          },
        }),
      );
      header.appendChild(controls);
      card.appendChild(header);

      const indicatorLabel = document.createElement("label");
      indicatorLabel.setAttribute("for", `${block.uid}-indicator-id`);
      indicatorLabel.textContent = "Indicator";
      card.appendChild(indicatorLabel);

      const indicatorSelect = document.createElement("select");
      indicatorSelect.id = `${block.uid}-indicator-id`;
      state.indicators.forEach((indicator) => {
        const option = document.createElement("option");
        option.value = String(indicator.indicator_id);
        option.textContent = `${indicator.indicator_id} - ${indicator.title}`;
        option.selected = String(indicator.indicator_id) === block.indicatorId;
        indicatorSelect.appendChild(option);
      });
      indicatorSelect.addEventListener("change", () => {
        block.indicatorId = indicatorSelect.value;
        block.paramAxes = {};
        block.sourceSelections = [];
        ensureDefaultsForBlock(block);
        renderIndicatorBlocks();
        refreshTemplateState();
      });
      card.appendChild(indicatorSelect);

      const sourceOptions = readSourceCatalogValues(block.indicatorId);
      if (sourceOptions.length > 0) {
        const sourceLabel = document.createElement("label");
        sourceLabel.setAttribute("for", `${block.uid}-source-select`);
        sourceLabel.textContent = "source values";
        card.appendChild(sourceLabel);

        const sourceSelect = document.createElement("select");
        sourceSelect.id = `${block.uid}-source-select`;
        sourceSelect.multiple = true;
        sourceSelect.size = Math.min(Math.max(sourceOptions.length, 2), 4);
        const selectedSourceValues = readSelectedSourceValues({
          allowedValues: sourceOptions,
          rawSelectedValues: block.sourceSelections,
        });
        sourceOptions.forEach((sourceValue) => {
          const option = document.createElement("option");
          option.value = sourceValue;
          option.textContent = sourceValue;
          option.selected = selectedSourceValues.includes(sourceValue);
          sourceSelect.appendChild(option);
        });
        sourceSelect.addEventListener("change", () => {
          block.sourceSelections = readSelectedSourceValues({
            allowedValues: sourceOptions,
            rawSelectedValues: Array.from(sourceSelect.selectedOptions).map((item) => item.value),
          });
          refreshTemplateState();
        });
        card.appendChild(sourceSelect);

        const sourceHint = document.createElement("p");
        sourceHint.className = "muted-text";
        sourceHint.textContent = "Use Cmd/Ctrl to select multiple inputs.source values.";
        card.appendChild(sourceHint);
      }

      const paramsTitle = document.createElement("h4");
      paramsTitle.textContent = "params";
      card.appendChild(paramsTitle);

      const descriptorParams = Array.isArray(descriptor.params) ? descriptor.params : [];
      if (descriptorParams.length === 0) {
        const noParamsNode = document.createElement("p");
        noParamsNode.className = "muted-text";
        noParamsNode.textContent = "No params.";
        card.appendChild(noParamsNode);
      } else {
        descriptorParams.forEach((paramSpec) => {
          const paramRecord = asRecord(paramSpec);
          const paramName = String(paramRecord.name || "");
          const paramKind = String(paramRecord.kind || "string");
          if (!paramName) {
            return;
          }
          const axisState = ensureParamAxisState({ block, paramRecord });
          if (axisState === null) {
            return;
          }
          const labels = readParamLabels({
            indicatorId: block.indicatorId,
            paramName,
          });

          const modeLabel = document.createElement("label");
          modeLabel.setAttribute("for", `${block.uid}-param-${paramName}-mode`);
          modeLabel.textContent = `${labels.axisLabel} axis mode`;
          card.appendChild(modeLabel);

          const modeSelect = document.createElement("select");
          modeSelect.id = `${block.uid}-param-${paramName}-mode`;
          ["explicit", "range"].forEach((modeOption) => {
            const option = document.createElement("option");
            option.value = modeOption;
            option.textContent = modeOption;
            option.selected = axisState.mode === modeOption;
            modeSelect.appendChild(option);
          });
          card.appendChild(modeSelect);

          const explicitLabel = document.createElement("label");
          explicitLabel.setAttribute("for", `${block.uid}-param-${paramName}-values`);
          explicitLabel.textContent = `${labels.axisLabel} (${paramKind}) values (csv)`;
          card.appendChild(explicitLabel);

          const explicitInput = document.createElement("input");
          explicitInput.id = `${block.uid}-param-${paramName}-values`;
          explicitInput.type = "text";
          explicitInput.value = String(axisState.explicitValues || "");
          if (Array.isArray(paramRecord.enum_values) && paramRecord.enum_values.length > 0) {
            explicitInput.placeholder = paramRecord.enum_values.join(",");
          }
          explicitInput.addEventListener("change", () => {
            axisState.explicitValues = explicitInput.value.trim();
            refreshTemplateState();
          });
          card.appendChild(explicitInput);

          const rangeStartLabel = document.createElement("label");
          rangeStartLabel.setAttribute("for", `${block.uid}-param-${paramName}-start`);
          rangeStartLabel.textContent = `${labels.axisLabel} start`;
          card.appendChild(rangeStartLabel);

          const rangeStartInput = document.createElement("input");
          rangeStartInput.id = `${block.uid}-param-${paramName}-start`;
          rangeStartInput.type = "number";
          rangeStartInput.step = "any";
          rangeStartInput.value = String(axisState.rangeStart || "");
          rangeStartInput.addEventListener("change", () => {
            axisState.rangeStart = rangeStartInput.value.trim();
            refreshTemplateState();
          });
          card.appendChild(rangeStartInput);

          const rangeStopLabel = document.createElement("label");
          rangeStopLabel.setAttribute("for", `${block.uid}-param-${paramName}-stop`);
          rangeStopLabel.textContent = `${labels.axisLabel} stop_incl`;
          card.appendChild(rangeStopLabel);

          const rangeStopInput = document.createElement("input");
          rangeStopInput.id = `${block.uid}-param-${paramName}-stop`;
          rangeStopInput.type = "number";
          rangeStopInput.step = "any";
          rangeStopInput.value = String(axisState.rangeStopIncl || "");
          rangeStopInput.addEventListener("change", () => {
            axisState.rangeStopIncl = rangeStopInput.value.trim();
            refreshTemplateState();
          });
          card.appendChild(rangeStopInput);

          const rangeStepLabel = document.createElement("label");
          rangeStepLabel.setAttribute("for", `${block.uid}-param-${paramName}-step`);
          rangeStepLabel.textContent = labels.stepLabel;
          card.appendChild(rangeStepLabel);

          const rangeStepInput = document.createElement("input");
          rangeStepInput.id = `${block.uid}-param-${paramName}-step`;
          rangeStepInput.type = "number";
          rangeStepInput.step = "any";
          rangeStepInput.value = String(axisState.rangeStep || "");
          rangeStepInput.addEventListener("change", () => {
            axisState.rangeStep = rangeStepInput.value.trim();
            refreshTemplateState();
          });
          card.appendChild(rangeStepInput);

          const toggleParamMode = () => {
            const isRangeMode = axisState.mode === "range";
            explicitLabel.classList.toggle("hidden", isRangeMode);
            explicitInput.classList.toggle("hidden", isRangeMode);
            rangeStartLabel.classList.toggle("hidden", !isRangeMode);
            rangeStartInput.classList.toggle("hidden", !isRangeMode);
            rangeStopLabel.classList.toggle("hidden", !isRangeMode);
            rangeStopInput.classList.toggle("hidden", !isRangeMode);
            rangeStepLabel.classList.toggle("hidden", !isRangeMode);
            rangeStepInput.classList.toggle("hidden", !isRangeMode);
          };

          modeSelect.addEventListener("change", () => {
            axisState.mode = normalizeAxisMode(modeSelect.value);
            toggleParamMode();
            refreshTemplateState();
          });
          toggleParamMode();
        });
      }

      blocksContainer.appendChild(card);
    });
  };

  const toLocalDatetimeInputValue = (date) => {
    const pad = (value) => String(value).padStart(2, "0");
    return [
      date.getFullYear(),
      "-",
      pad(date.getMonth() + 1),
      "-",
      pad(date.getDate()),
      "T",
      pad(date.getHours()),
      ":",
      pad(date.getMinutes()),
    ].join("");
  };

  const applyRangePreset = (preset) => {
    if (!RANGE_PRESET_DAYS.has(preset)) {
      return;
    }
    const end = new Date();
    const days = RANGE_PRESET_DAYS.get(preset) || 0;
    const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
    rangeStartInput.value = toLocalDatetimeInputValue(start);
    rangeEndInput.value = toLocalDatetimeInputValue(end);
  };

  const parseTimeRange = () => {
    const rawStart = String(rangeStartInput.value || "").trim();
    const rawEnd = String(rangeEndInput.value || "").trim();
    if (rawStart.length === 0 || rawEnd.length === 0) {
      throw new Error("Please set start and end datetime.");
    }
    const start = new Date(rawStart);
    const end = new Date(rawEnd);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
      throw new Error("Time range is invalid.");
    }
    if (start.getTime() >= end.getTime()) {
      throw new Error("Time range start must be earlier than end.");
    }
    return {
      start: start.toISOString(),
      end: end.toISOString(),
    };
  };

  const parseAxisValuesCsv = (rawValues, kind, fieldLabel) => {
    const values = String(rawValues || "")
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    if (values.length === 0) {
      throw new Error(`${fieldLabel} must include at least one value.`);
    }

    const parsed = values.map((item) => {
      if (kind === "int") {
        const parsedInt = Number.parseInt(item, 10);
        if (Number.isNaN(parsedInt)) {
          throw new Error(`${fieldLabel} contains invalid int value: ${item}`);
        }
        return parsedInt;
      }
      if (kind === "float") {
        const parsedFloat = Number.parseFloat(item);
        if (Number.isNaN(parsedFloat)) {
          throw new Error(`${fieldLabel} contains invalid float value: ${item}`);
        }
        return parsedFloat;
      }
      if (kind === "bool") {
        const normalized = item.toLowerCase();
        if (normalized === "true") {
          return "true";
        }
        if (normalized === "false") {
          return "false";
        }
        throw new Error(`${fieldLabel} contains invalid bool value: ${item}`);
      }
      return item;
    });

    return {
      mode: "explicit",
      values: parsed,
    };
  };

  const parseRangeAxisSpec = ({ kind, startRaw, stopRaw, stepRaw, fieldLabel }) => {
    if (kind !== "int" && kind !== "float") {
      throw new Error(`${fieldLabel} range mode requires numeric param kind.`);
    }

    const rawStart = String(startRaw || "").trim();
    const rawStop = String(stopRaw || "").trim();
    const rawStep = String(stepRaw || "").trim();
    if (rawStart.length === 0 || rawStop.length === 0 || rawStep.length === 0) {
      throw new Error(`${fieldLabel} range requires start, stop_incl, and step.`);
    }

    const parseField = (rawValue, axisField) => {
      if (kind === "int") {
        if (!/^-?\d+$/.test(rawValue)) {
          throw new Error(`${fieldLabel} range ${axisField} must be an int.`);
        }
        return Number.parseInt(rawValue, 10);
      }
      const parsedFloat = Number(rawValue);
      if (!Number.isFinite(parsedFloat)) {
        throw new Error(`${fieldLabel} range ${axisField} must be a float.`);
      }
      return parsedFloat;
    };

    const startValue = parseField(rawStart, "start");
    const stopValue = parseField(rawStop, "stop_incl");
    const stepValue = parseField(rawStep, "step");
    if (stepValue <= 0) {
      throw new Error(`${fieldLabel} range step must be greater than 0.`);
    }
    if (startValue > stopValue) {
      throw new Error(`${fieldLabel} range start must be less than or equal to stop_incl.`);
    }

    return {
      mode: "range",
      start: startValue,
      stop_incl: stopValue,
      step: stepValue,
    };
  };

  const parseRiskAxis = ({ modeNode, valuesNode, startNode, stopNode, stepNode, sideName }) => {
    const mode = String(modeNode.value || "explicit").trim().toLowerCase();
    if (mode === "range") {
      const hasAnyRangeInput = (
        String(startNode.value || "").trim().length > 0
        || String(stopNode.value || "").trim().length > 0
        || String(stepNode.value || "").trim().length > 0
      );
      if (!hasAnyRangeInput) {
        return null;
      }
      const startValue = Number.parseFloat(String(startNode.value || "").trim());
      const stopValue = Number.parseFloat(String(stopNode.value || "").trim());
      const stepValue = Number.parseFloat(String(stepNode.value || "").trim());
      if (
        Number.isNaN(startValue)
        || Number.isNaN(stopValue)
        || Number.isNaN(stepValue)
      ) {
        throw new Error(`risk_grid.${sideName} range requires numeric start/stop/step.`);
      }
      return {
        mode: "range",
        start: startValue,
        stop_incl: stopValue,
        step: stepValue,
      };
    }

    const rawValues = String(valuesNode.value || "").trim();
    if (rawValues.length === 0) {
      return null;
    }
    return parseAxisValuesCsv(rawValues, "float", `risk_grid.${sideName}`);
  };

  const readOptionalNumber = (node, label) => {
    const rawValue = String(node.value || "").trim();
    if (rawValue.length === 0) {
      return null;
    }
    const parsed = Number.parseFloat(rawValue);
    if (Number.isNaN(parsed)) {
      throw new Error(`${label} must be a number.`);
    }
    return parsed;
  };

  const readOptionalPositiveInt = (node, label) => {
    const rawValue = String(node.value || "").trim();
    if (rawValue.length === 0) {
      return null;
    }
    const parsed = Number.parseInt(rawValue, 10);
    if (Number.isNaN(parsed) || parsed <= 0) {
      throw new Error(`${label} must be a positive integer.`);
    }
    return parsed;
  };

  const readOptionalRankingMetricLiteral = ({ rawValue, fieldLabel }) => {
    const metric = String(rawValue || "").trim().toLowerCase();
    if (metric.length === 0) {
      return null;
    }
    const rankingMetrics = readRankingMetrics();
    if (!rankingMetrics.includes(metric)) {
      throw new Error(
        `${fieldLabel} must be one of ${rankingMetrics.join(", ")}.`,
      );
    }
    return metric;
  };

  const readRequestedTopN = () => {
    const parsedTopN = readOptionalPositiveInt(topNInput, "top_n");
    if (parsedTopN === null) {
      return null;
    }
    const topNMax = readTopNMax();
    if (topNMax === null || topNMax <= 0) {
      return parsedTopN;
    }
    const cappedTopN = Math.min(parsedTopN, topNMax);
    if (cappedTopN !== parsedTopN) {
      topNInput.value = String(cappedTopN);
    }
    return cappedTopN;
  };

  const buildExecutionPayload = () => {
    const execution = {};
    const initCash = readOptionalNumber(executionInitCash, "execution.init_cash_quote");
    const feePct = readOptionalNumber(executionFeePct, "execution.fee_pct");
    const slippagePct = readOptionalNumber(executionSlippagePct, "execution.slippage_pct");
    const fixedQuote = readOptionalNumber(executionFixedQuote, "execution.fixed_quote");
    const safeProfitPercent = readOptionalNumber(
      executionSafeProfitPercent,
      "execution.safe_profit_percent",
    );

    if (initCash !== null) {
      execution.init_cash_quote = initCash;
    }
    if (feePct !== null) {
      execution.fee_pct = feePct;
    }
    if (slippagePct !== null) {
      execution.slippage_pct = slippagePct;
    }
    if (fixedQuote !== null) {
      execution.fixed_quote = fixedQuote;
    }
    if (safeProfitPercent !== null) {
      execution.safe_profit_percent = safeProfitPercent;
    }

    return Object.keys(execution).length > 0 ? execution : null;
  };

  const buildRiskGridPayload = () => {
    const slEnabled = riskSlEnabled.checked;
    const tpEnabled = riskTpEnabled.checked;

    const slAxis = slEnabled
      ? parseRiskAxis({
        modeNode: riskSlMode,
        valuesNode: riskSlValues,
        startNode: riskSlStart,
        stopNode: riskSlStop,
        stepNode: riskSlStep,
        sideName: "sl",
      })
      : null;
    const tpAxis = tpEnabled
      ? parseRiskAxis({
        modeNode: riskTpMode,
        valuesNode: riskTpValues,
        startNode: riskTpStart,
        stopNode: riskTpStop,
        stepNode: riskTpStep,
        sideName: "tp",
      })
      : null;
    const slPct = slEnabled ? readOptionalNumber(riskSlPct, "risk_grid.sl_pct") : null;
    const tpPct = tpEnabled ? readOptionalNumber(riskTpPct, "risk_grid.tp_pct") : null;

    if (
      !slEnabled
      && !tpEnabled
      && slAxis === null
      && tpAxis === null
      && slPct === null
      && tpPct === null
    ) {
      return null;
    }

    const riskGrid = {
      sl_enabled: slEnabled,
      tp_enabled: tpEnabled,
    };
    if (slAxis !== null) {
      riskGrid.sl = slAxis;
    }
    if (tpAxis !== null) {
      riskGrid.tp = tpAxis;
    }
    if (slPct !== null) {
      riskGrid.sl_pct = slPct;
    }
    if (tpPct !== null) {
      riskGrid.tp_pct = tpPct;
    }
    return riskGrid;
  };

  const buildAdvancedOptions = () => {
    const directionMode = String(directionModeSelect.value || "").trim();
    const sizingMode = String(sizingModeSelect.value || "").trim();
    const primaryMetric = readOptionalRankingMetricLiteral({
      rawValue: rankingPrimaryMetricSelect.value,
      fieldLabel: "primary_metric",
    });
    const secondaryMetric = readOptionalRankingMetricLiteral({
      rawValue: rankingSecondaryMetricSelect.value,
      fieldLabel: "secondary_metric",
    });
    if (primaryMetric === null && secondaryMetric !== null) {
      throw new Error("primary_metric is required when secondary_metric is provided.");
    }
    if (primaryMetric !== null && secondaryMetric !== null && primaryMetric === secondaryMetric) {
      throw new Error("secondary_metric must be different from primary_metric.");
    }
    return {
      directionMode: directionMode.length > 0 ? directionMode : null,
      sizingMode: sizingMode.length > 0 ? sizingMode : null,
      rankingPrimaryMetric: primaryMetric,
      rankingSecondaryMetric: secondaryMetric,
      execution: buildExecutionPayload(),
      riskGrid: buildRiskGridPayload(),
      topN: readRequestedTopN(),
      preselect: readOptionalPositiveInt(preselectInput, "preselect"),
      topTradesN: readOptionalPositiveInt(topTradesInput, "top_trades_n"),
      warmupBars: readOptionalPositiveInt(warmupBarsInput, "warmup_bars"),
    };
  };

  const buildTemplateIndicatorGrids = () => {
    if (state.blocks.length === 0) {
      throw new Error("Template mode requires at least one indicator grid.");
    }
    return state.blocks.map((block, index) => {
      ensureDefaultsForBlock(block);
      const descriptor = state.indicatorsById.get(block.indicatorId);
      if (!descriptor) {
        throw new Error(`Indicator descriptor is unavailable for block #${index + 1}.`);
      }

      const params = {};
      const descriptorParams = Array.isArray(descriptor.params) ? descriptor.params : [];
      descriptorParams.forEach((paramSpec) => {
        const paramRecord = asRecord(paramSpec);
        const paramName = String(paramRecord.name || "");
        if (!paramName) {
          return;
        }
        const axisState = ensureParamAxisState({ block, paramRecord });
        if (axisState === null) {
          return;
        }
        const paramKind = String(paramRecord.kind || "string");
        const fieldLabel = `indicator ${block.indicatorId} param ${paramName}`;
        if (axisState.mode === "range") {
          params[paramName] = parseRangeAxisSpec({
            kind: paramKind,
            startRaw: axisState.rangeStart,
            stopRaw: axisState.rangeStopIncl,
            stepRaw: axisState.rangeStep,
            fieldLabel,
          });
          return;
        }

        const rawValue = String(axisState.explicitValues || "").trim();
        if (rawValue.length === 0) {
          return;
        }
        params[paramName] = parseAxisValuesCsv(rawValue, paramKind, fieldLabel);
      });

      const grid = {
        indicator_id: block.indicatorId,
        params,
      };

      const allowedSourceValues = readSourceCatalogValues(block.indicatorId);
      const selectedSourceValues = readSelectedSourceValues({
        allowedValues: allowedSourceValues,
        rawSelectedValues: block.sourceSelections,
      });
      if (allowedSourceValues.length > 0 && selectedSourceValues.length === 0) {
        throw new Error(`indicator ${block.indicatorId} source requires one or more selected values.`);
      }
      if (selectedSourceValues.length > 0) {
        grid.source = {
          mode: "explicit",
          values: selectedSourceValues,
        };
      }
      return grid;
    });
  };

  const buildRunRequest = () => {
    const timeRange = parseTimeRange();
    const advanced = buildAdvancedOptions();
    const requestPayload = {
      time_range: timeRange,
    };

    if (advanced.topN !== null) {
      requestPayload.top_k = advanced.topN;
    }
    if (advanced.preselect !== null) {
      requestPayload.preselect = advanced.preselect;
    }
    if (advanced.topTradesN !== null) {
      requestPayload.top_trades_n = advanced.topTradesN;
    }
    if (advanced.warmupBars !== null) {
      requestPayload.warmup_bars = advanced.warmupBars;
    }
    if (advanced.rankingPrimaryMetric !== null) {
      requestPayload.ranking = {
        primary_metric: advanced.rankingPrimaryMetric,
      };
      if (advanced.rankingSecondaryMetric !== null) {
        requestPayload.ranking.secondary_metric = advanced.rankingSecondaryMetric;
      }
    }

    if (state.mode === "template") {
      const marketId = Number(marketSelect.value || "0");
      const market = state.marketsById.get(marketId);
      if (!market) {
        throw new Error("Please select market.");
      }
      const symbol = String(symbolValue.value || "").trim();
      if (symbol.length === 0) {
        throw new Error("Please select symbol from suggestions.");
      }
      const timeframe = String(timeframeSelect.value || "").trim();
      if (!readAllowedTimeframes().includes(timeframe)) {
        throw new Error("timeframe must be one of runtime request_timeframes.allowed.");
      }

      const templatePayload = {
        instrument_id: {
          market_id: market.market_id,
          symbol,
        },
        timeframe,
        indicator_grids: buildTemplateIndicatorGrids(),
      };
      if (advanced.directionMode !== null) {
        templatePayload.direction_mode = advanced.directionMode;
      }
      if (advanced.sizingMode !== null) {
        templatePayload.sizing_mode = advanced.sizingMode;
      }
      if (advanced.execution !== null) {
        templatePayload.execution = advanced.execution;
      }
      if (advanced.riskGrid !== null) {
        templatePayload.risk_grid = advanced.riskGrid;
      }

      requestPayload.template = templatePayload;
      return {
        payload: requestPayload,
        context: {
          mode: "template",
          market,
        },
      };
    }

    const strategyId = String(strategiesSelect.value || "").trim();
    if (strategyId.length === 0) {
      throw new Error("Please select strategy.");
    }
    requestPayload.strategy_id = strategyId;

    const strategy = state.strategiesById.get(strategyId);
    const overrides = {};
    if (advanced.directionMode !== null) {
      overrides.direction_mode = advanced.directionMode;
    }
    if (advanced.sizingMode !== null) {
      overrides.sizing_mode = advanced.sizingMode;
    }
    if (advanced.execution !== null) {
      overrides.execution = advanced.execution;
    }
    if (advanced.riskGrid !== null) {
      overrides.risk_grid = advanced.riskGrid;
    }
    if (Object.keys(overrides).length > 0) {
      requestPayload.overrides = overrides;
    }
    return {
      payload: requestPayload,
      context: {
        mode: "saved",
        strategy,
      },
    };
  };

  const renderStrategyOptions = (strategies) => {
    const previousValue = String(strategiesSelect.value || "").trim();
    strategiesSelect.innerHTML = "<option value=\"\">Select strategy</option>";
    strategies.forEach((strategy) => {
      const record = asRecord(strategy);
      const spec = asRecord(record.spec);
      const instrument = asRecord(spec.instrument_id);
      const strategyId = String(record.strategy_id || "").trim();
      if (strategyId.length === 0) {
        return;
      }
      const option = document.createElement("option");
      option.value = strategyId;
      option.textContent = [
        String(record.name || "strategy"),
        String(instrument.symbol || ""),
        String(spec.timeframe || ""),
      ].filter((item) => item.length > 0).join(" | ");
      strategiesSelect.appendChild(option);
    });
    if (previousValue.length > 0) {
      strategiesSelect.value = previousValue;
    }
    updateRunAvailability();
  };

  const loadStrategies = async () => {
    try {
      const response = await fetch(strategiesPath, { credentials: 'include' });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      const strategies = Array.isArray(payload) ? payload : [];
      state.strategiesById = new Map(
        strategies
          .map((item) => asRecord(item))
          .map((item) => [String(item.strategy_id || "").trim(), item])
          .filter((entry) => entry[0].length > 0),
      );
      renderStrategyOptions(strategies);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  const loadReferences = async () => {
    try {
      const [marketsResponse, indicatorsResponse] = await Promise.all([
        fetch(marketsPath, { credentials: 'include' }),
        fetch(indicatorsPath, { credentials: 'include' }),
      ]);
      if (!marketsResponse.ok) {
        throw await buildHttpError(marketsResponse);
      }
      if (!indicatorsResponse.ok) {
        throw await buildHttpError(indicatorsResponse);
      }

      const marketsPayload = await marketsResponse.json();
      const indicatorsPayload = await indicatorsResponse.json();

      const marketsItems = Array.isArray(marketsPayload.items) ? marketsPayload.items : [];
      state.markets = marketsItems
        .map((item) => asRecord(item))
        .filter((item) => Number(item.market_id || 0) > 0)
        .sort((left, right) => Number(left.market_id) - Number(right.market_id));
      state.marketsById = new Map(
        state.markets.map((market) => [Number(market.market_id), market]),
      );

      marketSelect.innerHTML = "<option value=\"\">Select market</option>";
      state.markets.forEach((market) => {
        const option = document.createElement("option");
        option.value = String(market.market_id);
        option.textContent = `${market.market_code} (${market.market_type})`;
        marketSelect.appendChild(option);
      });

      const indicatorsItems = Array.isArray(indicatorsPayload.items)
        ? indicatorsPayload.items
        : [];
      state.indicatorDescriptors = indicatorsItems
        .map((item) => asRecord(item))
        .filter((item) => String(item.indicator_id || "").trim().length > 0);
      rebuildIndicatorCatalog();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
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

  const buildVariantReportRequestPayload = ({ variantRecord }) => {
    if (state.latestRun === null) {
      throw new Error("Backtest result context is unavailable.");
    }

    const latestRunRecord = asRecord(state.latestRun);
    const runRequestPayload = asRecord(latestRunRecord.requestPayload);
    const runResponsePayload = asRecord(latestRunRecord.response);
    const variantPayload = asRecord(variantRecord.payload);
    if (Object.keys(variantPayload).length === 0) {
      throw new Error("Variant payload is unavailable.");
    }

    const payload = {
      time_range: normalizeJsonLikeValue(asRecord(runRequestPayload.time_range)),
      variant: normalizeJsonLikeValue(variantPayload),
      include_trades: Number(runResponsePayload.top_trades_n || 0) > 0,
    };

    const strategyId = String(runRequestPayload.strategy_id || "").trim();
    const templatePayload = asRecord(runRequestPayload.template);
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

    const overridesPayload = asRecord(runRequestPayload.overrides);
    if (Object.keys(overridesPayload).length > 0) {
      payload.overrides = normalizeJsonLikeValue(overridesPayload);
    }

    const warmupBars = readFiniteInteger(runRequestPayload.warmup_bars);
    if (warmupBars !== null && warmupBars > 0) {
      payload.warmup_bars = warmupBars;
    }
    return payload;
  };

  const loadVariantReport = async (variantIndex) => {
    if (state.latestRun === null) {
      showPageError(pageRoot, "Backtest result is unavailable.", []);
      return;
    }

    const latestRunRecord = asRecord(state.latestRun);
    const response = asRecord(latestRunRecord.response);
    const variants = Array.isArray(response.variants) ? response.variants : [];
    const variant = variants[variantIndex];
    if (!variant) {
      showPageError(pageRoot, "Variant is unavailable.", []);
      return;
    }
    const variantRecord = asRecord(variant);
    const variantKey = String(variantRecord.variant_key || "").trim();
    if (variantKey.length === 0) {
      showPageError(pageRoot, "variant_key is required for report loading.", []);
      return;
    }

    if (state.reportLoadingKeys.has(variantKey)) {
      return;
    }
    if (state.reportCacheByVariantKey.has(variantKey)) {
      state.reportErrorsByVariantKey.delete(variantKey);
      state.reportCacheHitKeys.add(variantKey);
      renderResults(response);
      return;
    }

    const runToken = state.latestRunToken;
    state.reportLoadingKeys.add(variantKey);
    state.reportErrorsByVariantKey.delete(variantKey);
    state.reportCacheHitKeys.delete(variantKey);
    renderResults(response);

    try {
      const reportRequestPayload = buildVariantReportRequestPayload({ variantRecord });
      const reportResponse = await fetch(backtestVariantReportPath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reportRequestPayload),
      });
      if (!reportResponse.ok) {
        throw await buildHttpError(reportResponse);
      }
      const reportPayload = asRecord(await reportResponse.json());
      if (runToken !== state.latestRunToken) {
        return;
      }
      state.reportCacheByVariantKey.set(variantKey, reportPayload);
    } catch (error) {
      if (runToken !== state.latestRunToken) {
        return;
      }
      const normalized = normalizeError(error);
      state.reportErrorsByVariantKey.set(variantKey, normalized);
    } finally {
      if (runToken !== state.latestRunToken) {
        return;
      }
      state.reportLoadingKeys.delete(variantKey);
      renderResults(asRecord(state.latestRun).response);
    }
  };

  const renderResults = (responsePayload) => {
    const response = asRecord(responsePayload);
    const metaPayload = {
      schema_version: response.schema_version,
      run_id: response.run_id || null,
      state: response.state || null,
      execution_mode: response.execution_mode || null,
      mode: response.mode,
      strategy_id: response.strategy_id || null,
      instrument_id: response.instrument_id || {},
      timeframe: response.timeframe || "",
      top_k: response.top_k,
      preselect: response.preselect,
      top_trades_n: response.top_trades_n,
      warmup_bars: response.warmup_bars,
      spec_hash: response.spec_hash || null,
      grid_request_hash: response.grid_request_hash || null,
      engine_params_hash: response.engine_params_hash || null,
    };
    resultsMeta.textContent = JSON.stringify(metaPayload, null, 2);

    variantsBody.innerHTML = "";
    const variants = Array.isArray(response.variants) ? response.variants : [];
    if (variants.length === 0) {
      const emptyMessage = String(response.execution_mode || "").trim() === "background_auto"
        ? "No inline variants yet. background_auto launch is queued."
        : "No variants returned.";
      variantsBody.innerHTML = `<tr><td colspan="6">${escapeHtml(emptyMessage)}</td></tr>`;
    } else {
      variants.forEach((variant, index) => {
        const variantRecord = asRecord(variant);
        const variantKey = String(variantRecord.variant_key || "").trim();
        const reportState = readVariantReportStateByKey(variantKey);
        const row = document.createElement("tr");
        row.appendChild(buildCell(String(variantRecord.variant_index ?? index)));
        row.appendChild(buildCell(String(variantRecord.total_return_pct ?? "")));
        row.appendChild(buildCell(String(variantKey)));
        row.appendChild(buildCell(String(variantRecord.indicator_variant_key || "")));

        const reportCell = document.createElement("td");
        if (variantKey.length === 0) {
          reportCell.textContent = "variant_key is missing.";
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
          disabled: variantKey.length === 0 || reportState.isLoading,
          onClick: () => {
            loadVariantReport(index);
          },
        });
        actionsCell.appendChild(loadReportButton);
        const saveButton = buildActionButton({
          label: "Save as Strategy",
          className: "button-link--secondary",
          onClick: () => {
            saveVariantAsStrategy(index);
          },
        });
        actionsCell.appendChild(saveButton);
        row.appendChild(actionsCell);

        variantsBody.appendChild(row);
      });
    }

    resultsPanel.classList.remove("hidden");
  };

  const buildPrefillPayload = (variant) => {
    if (state.latestRun === null) {
      throw new Error("Backtest result context is unavailable.");
    }
    const runResponse = asRecord(state.latestRun.response);
    const instrument = asRecord(runResponse.instrument_id);
    const payload = asRecord(asRecord(variant).payload);
    const selections = Array.isArray(payload.indicator_selections)
      ? payload.indicator_selections
      : [];

    const indicators = selections.map((selection) => {
      const record = asRecord(selection);
      return {
        id: String(record.indicator_id || ""),
        inputs: copyRecord(asRecord(record.inputs)),
        params: copyRecord(asRecord(record.params)),
      };
    }).filter((item) => item.id.length > 0);

    const context = asRecord(state.latestRun.context);
    let marketType = "";
    let instrumentKey = "";

    if (context.mode === "saved") {
      const strategy = asRecord(context.strategy);
      const strategySpec = asRecord(strategy.spec);
      marketType = String(strategySpec.market_type || "");
      instrumentKey = String(strategySpec.instrument_key || "");
    } else {
      const market = asRecord(context.market);
      const symbol = String(instrument.symbol || "");
      marketType = String(market.market_type || "");
      instrumentKey = `${String(market.market_code || "")}:${marketType}:${symbol}`;
    }

    if (marketType.length === 0 || instrumentKey.length === 0) {
      const market = state.marketsById.get(Number(instrument.market_id || 0));
      if (market) {
        const symbol = String(instrument.symbol || "");
        marketType = String(market.market_type || "");
        instrumentKey = `${String(market.market_code || "")}:${marketType}:${symbol}`;
      }
    }

    return {
      instrument_id: {
        market_id: Number(instrument.market_id || 0),
        symbol: String(instrument.symbol || ""),
      },
      instrument_key: instrumentKey,
      market_type: marketType,
      timeframe: String(runResponse.timeframe || ""),
      indicators,
    };
  };

  const saveVariantAsStrategy = (variantIndex) => {
    if (state.latestRun === null) {
      showPageError(pageRoot, "Backtest result is unavailable.", []);
      return;
    }
    const response = asRecord(state.latestRun.response);
    const variants = Array.isArray(response.variants) ? response.variants : [];
    const variant = variants[variantIndex];
    if (!variant) {
      showPageError(pageRoot, "Variant is unavailable.", []);
      return;
    }

    if (prefillStorage !== "sessionStorage" || typeof window.sessionStorage === "undefined") {
      showPageError(pageRoot, "sessionStorage is unavailable in current browser.", []);
      return;
    }

    try {
      const prefillPayload = buildPrefillPayload(variant);
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

  const buildJobContextPayload = (request) => {
    const requestPayload = asRecord(request.payload);
    const context = asRecord(request.context);

    if (context.mode === "saved") {
      const strategy = asRecord(context.strategy);
      const strategySpec = asRecord(strategy.spec);
      const instrument = asRecord(strategySpec.instrument_id);
      const marketId = Number(instrument.market_id || 0);
      const symbol = String(instrument.symbol || "").trim();
      const timeframe = String(strategySpec.timeframe || "").trim();
      const marketType = String(strategySpec.market_type || "").trim();
      const instrumentKey = String(strategySpec.instrument_key || "").trim();
      if (
        marketId <= 0
        || symbol.length === 0
        || timeframe.length === 0
        || marketType.length === 0
        || instrumentKey.length === 0
      ) {
        return null;
      }
      return {
        mode: "saved",
        instrument_id: {
          market_id: marketId,
          symbol,
        },
        timeframe,
        market_type: marketType,
        instrument_key: instrumentKey,
      };
    }

    const templatePayload = asRecord(requestPayload.template);
    const instrument = asRecord(templatePayload.instrument_id);
    const market = asRecord(context.market);
    const marketId = Number(instrument.market_id || 0);
    const symbol = String(instrument.symbol || "").trim();
    const timeframe = String(templatePayload.timeframe || "").trim();
    const marketType = String(market.market_type || "").trim();
    const marketCode = String(market.market_code || "").trim();
    if (
      marketId <= 0
      || symbol.length === 0
      || timeframe.length === 0
      || marketType.length === 0
      || marketCode.length === 0
    ) {
      return null;
    }

    return {
      mode: "template",
      instrument_id: {
        market_id: marketId,
        symbol,
      },
      timeframe,
      market_type: marketType,
      instrument_key: `${marketCode}:${marketType}:${symbol}`,
    };
  };

  const persistJobContext = ({ jobId, request }) => {
    if (typeof window.sessionStorage === "undefined") {
      return;
    }
    const contextPayload = buildJobContextPayload(request);
    if (contextPayload === null) {
      return;
    }
    const storageKey = `${jobContextStoragePrefix}${jobId}`;
    window.sessionStorage.setItem(storageKey, JSON.stringify(contextPayload));
  };

  const renderRunSummaryPath = (runId) => (
    renderPathTemplate(runSummaryPathTemplate, encodeURIComponent(runId))
  );

  const renderLaunchOutcome = ({ payload, responseStatus }) => {
    const responsePayload = asRecord(payload);
    const runId = String(responsePayload.run_id || "").trim();
    const executionMode = String(responsePayload.execution_mode || "").trim();
    const stateValue = String(responsePayload.state || "").trim();
    const primaryRunLink = runId.length > 0 ? renderRunSummaryPath(runId) : historyPath;
    const primaryRunLabel = runId.length > 0
      ? `Open run ${runId}`
      : "Open backtest history";

    if (responseStatus === 202 || executionMode === "background_auto") {
      showLaunchStatus({
        message: [
          "202 Accepted.",
          "Server auto-preflight queued this launch in background_auto.",
          `run_id=${runId || "-"}.`,
          `state=${stateValue || "queued"}.`,
        ].join(" "),
        linkHref: primaryRunLink,
        linkLabel: primaryRunLabel,
      });
      return;
    }

    if (executionMode === "sync_inline") {
      showLaunchStatus({
        message: [
          "Inline launch completed.",
          `execution_mode=sync_inline.`,
          `run_id=${runId || "-"}.`,
          `state=${stateValue || "-"}.`,
        ].join(" "),
        linkHref: primaryRunLink,
        linkLabel: primaryRunLabel,
      });
      return;
    }

    hideLaunchStatus();
  };

  const runBacktestLaunch = async () => {
    const request = buildRunRequest();
    state.isRunning = true;
    hideLaunchStatus();
    updateRunAvailability();
    runLoading.classList.remove("hidden");
    try {
      const response = await fetch(backtestsPath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.payload),
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      if (response.status === 202) {
        const runId = String(asRecord(payload).run_id || "").trim();
        if (runId.length > 0) {
          persistJobContext({ jobId: runId, request });
        }
      }
      state.latestRunToken += 1;
      state.latestRun = {
        response: payload,
        context: request.context,
        requestPayload: normalizeJsonLikeValue(request.payload),
      };
      state.reportCacheByVariantKey.clear();
      state.reportLoadingKeys.clear();
      state.reportErrorsByVariantKey.clear();
      state.reportCacheHitKeys.clear();
      renderLaunchOutcome({ payload, responseStatus: response.status });
      renderResults(payload);
    } finally {
      state.isRunning = false;
      runLoading.classList.add("hidden");
      updateRunAvailability();
    }
  };

  const runBacktest = async () => {
    clearPageError(pageRoot);
    try {
      await runBacktestLaunch();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  modeTemplate.addEventListener("change", () => {
    if (!modeTemplate.checked) {
      return;
    }
    state.mode = "template";
    updateModeSections();
    updateRunAvailability();
    hideLaunchStatus();
  });
  modeSaved.addEventListener("change", () => {
    if (!modeSaved.checked) {
      return;
    }
    state.mode = "saved";
    updateModeSections();
    updateRunAvailability();
    hideLaunchStatus();
    if (state.strategiesById.size === 0) {
      loadStrategies();
    }
  });

  marketSelect.addEventListener("change", () => {
    clearSelectedSymbol();
    suggestionsList.innerHTML = "";
    applyDefaultFeeForSelectedMarket({ force: false, allowFallbackToFirst: false });
    refreshTemplateState();
  });
  symbolQuery.addEventListener("input", () => {
    clearSelectedSymbol();
    scheduleInstrumentSearch();
    refreshTemplateState();
  });
  timeframeSelect.addEventListener("change", refreshTemplateState);
  addIndicatorButton.addEventListener("click", addIndicatorBlock);
  refreshStrategiesButton.addEventListener("click", loadStrategies);
  strategiesSelect.addEventListener("change", () => {
    hideLaunchStatus();
    updateRunAvailability();
  });
  [riskSlEnabled, riskSlMode, riskTpEnabled, riskTpMode].forEach((node) => {
    node.addEventListener("change", updateRiskUiVisibility);
  });

  const markExecutionFeeDirty = () => {
    if (state.applyingFeeDefault) {
      return;
    }
    state.executionFeeDirty = true;
  };
  executionFeePct.addEventListener("change", markExecutionFeeDirty);
  executionFeePct.addEventListener("input", markExecutionFeeDirty);

  rangePresetSelect.addEventListener("change", () => {
    const preset = String(rangePresetSelect.value || "custom");
    applyRangePreset(preset);
    refreshTemplateState();
  });
  [rangeStartInput, rangeEndInput].forEach((node) => {
    const onRangeInputChanged = () => {
      rangePresetSelect.value = "custom";
      refreshTemplateState();
    };
    node.addEventListener("change", onRangeInputChanged);
    node.addEventListener("input", onRangeInputChanged);
  });

  [
    directionModeSelect,
    sizingModeSelect,
    rankingPrimaryMetricSelect,
    rankingSecondaryMetricSelect,
    executionInitCash,
    executionFeePct,
    executionSlippagePct,
    executionFixedQuote,
    executionSafeProfitPercent,
    riskSlEnabled,
    riskSlMode,
    riskSlValues,
    riskSlStart,
    riskSlStop,
    riskSlStep,
    riskSlPct,
    riskTpEnabled,
    riskTpMode,
    riskTpValues,
    riskTpStart,
    riskTpStop,
    riskTpStep,
    riskTpPct,
    topNInput,
    preselectInput,
    topTradesInput,
    warmupBarsInput,
  ].forEach((node) => {
    const onAdvancedChanged = () => {
      hideLaunchStatus();
      updateRunAvailability();
    };
    node.addEventListener("change", onAdvancedChanged);
    node.addEventListener("input", onAdvancedChanged);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runBacktest();
  });

  applyRangePreset("90d");
  updateRiskUiVisibility();
  updateModeSections();
  hideLaunchStatus();
  updateRunAvailability();
  const bootstrap = async () => {
    await loadRuntimeDefaults();
    await Promise.all([
      loadReferences(),
      loadStrategies(),
    ]);
  };
  void bootstrap();
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
  const banner = pageRoot.querySelector("#backtest-error-banner");
  if (banner !== null) {
    banner.textContent = message;
    banner.classList.remove("hidden");
  }

  const detailsContainer = pageRoot.querySelector("#backtest-validation-errors");
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

function renderPathTemplate(pathTemplate, identifier) {
  return String(pathTemplate || "")
    .replace("{job_id}", String(identifier || ""))
    .replace("{run_id}", String(identifier || ""));
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

function asRecord(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
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
