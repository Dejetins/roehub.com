const BACKTEST_PAGE_SELECTOR = "[data-backtest-page]";
const SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
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
  const estimatePath = requireDataAttr(pageRoot, "apiEstimatePath");
  const strategiesPath = requireDataAttr(pageRoot, "apiStrategiesPath");
  const marketsPath = requireDataAttr(pageRoot, "apiMarketsPath");
  const instrumentsPath = requireDataAttr(pageRoot, "apiInstrumentsPath");
  const indicatorsPath = requireDataAttr(pageRoot, "apiIndicatorsPath");
  const strategyBuilderPath = requireDataAttr(pageRoot, "strategyBuilderPath");
  const prefillQueryParam = requireDataAttr(pageRoot, "prefillQueryParam");
  const prefillStorage = requireDataAttr(pageRoot, "prefillStorage");

  const form = pageRoot.querySelector("#backtest-form");
  const modeTemplate = pageRoot.querySelector("input[name=\"backtest-mode\"][value=\"template\"]");
  const modeSaved = pageRoot.querySelector("input[name=\"backtest-mode\"][value=\"saved\"]");
  const runTypeSync = pageRoot.querySelector("input[name=\"backtest-run-type\"][value=\"sync\"]");
  const runTypeJob = pageRoot.querySelector("input[name=\"backtest-run-type\"][value=\"job\"]");
  const templateModeSection = pageRoot.querySelector("#backtest-template-mode");
  const savedModeSection = pageRoot.querySelector("#backtest-saved-mode");
  const jobNotice = pageRoot.querySelector("#backtest-job-notice");
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
  const preflightButton = pageRoot.querySelector("#backtest-preflight-button");
  const preflightLoading = pageRoot.querySelector("#backtest-preflight-loading");
  const preflightSummary = pageRoot.querySelector("#backtest-preflight-summary");

  const strategiesSelect = pageRoot.querySelector("#backtest-strategy-id");
  const refreshStrategiesButton = pageRoot.querySelector("#backtest-refresh-strategies");

  const rangePresetSelect = pageRoot.querySelector("#backtest-range-preset");
  const rangeStartInput = pageRoot.querySelector("#backtest-range-start");
  const rangeEndInput = pageRoot.querySelector("#backtest-range-end");

  const directionModeSelect = pageRoot.querySelector("#backtest-direction-mode");
  const sizingModeSelect = pageRoot.querySelector("#backtest-sizing-mode");
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
  const topKInput = pageRoot.querySelector("#backtest-top-k");
  const preselectInput = pageRoot.querySelector("#backtest-preselect");
  const topTradesInput = pageRoot.querySelector("#backtest-top-trades-n");
  const warmupBarsInput = pageRoot.querySelector("#backtest-warmup-bars");

  const resultsPanel = pageRoot.querySelector("#backtest-results-panel");
  const resultsMeta = pageRoot.querySelector("#backtest-results-meta");
  const variantsBody = pageRoot.querySelector("#backtest-variants-body");

  if (
    form === null
    || modeTemplate === null
    || modeSaved === null
    || runTypeSync === null
    || runTypeJob === null
    || templateModeSection === null
    || savedModeSection === null
    || jobNotice === null
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
    || preflightButton === null
    || preflightLoading === null
    || preflightSummary === null
    || strategiesSelect === null
    || refreshStrategiesButton === null
    || rangePresetSelect === null
    || rangeStartInput === null
    || rangeEndInput === null
    || directionModeSelect === null
    || sizingModeSelect === null
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
    || topKInput === null
    || preselectInput === null
    || topTradesInput === null
    || warmupBarsInput === null
    || resultsPanel === null
    || resultsMeta === null
    || variantsBody === null
  ) {
    return;
  }

  const state = {
    mode: "template",
    runType: "sync",
    isRunning: false,
    preflightReady: false,
    markets: [],
    marketsById: new Map(),
    indicators: [],
    indicatorsById: new Map(),
    strategiesById: new Map(),
    blocks: [],
    nextBlockNumber: 1,
    searchDebounceId: 0,
    instrumentsAbortController: null,
    latestRun: null,
  };

  const setPreflightSummary = (message) => {
    preflightSummary.textContent = message;
  };

  const clearSelectedSymbol = () => {
    symbolValue.value = "";
    selectedSymbol.textContent = "Selected symbol: none";
  };

  const invalidatePreflight = () => {
    state.preflightReady = false;
    setPreflightSummary("Template mode requires successful preflight before run.");
    updateRunAvailability();
  };

  const updateModeSections = () => {
    const isTemplate = state.mode === "template";
    templateModeSection.classList.toggle("hidden", !isTemplate);
    savedModeSection.classList.toggle("hidden", isTemplate);
    preflightButton.disabled = !isTemplate || state.isRunning;
    if (!isTemplate) {
      setPreflightSummary("Saved mode does not require preflight.");
    } else if (!state.preflightReady) {
      setPreflightSummary("Template mode requires successful preflight before run.");
    }
  };

  const updateRunAvailability = () => {
    const isJob = state.runType === "job";
    jobNotice.classList.toggle("hidden", !isJob);
    runButton.textContent = isJob ? "Run as job" : "Run sync";
    runButton.disabled = true;

    if (state.isRunning) {
      return;
    }
    if (isJob) {
      return;
    }
    if (state.mode === "template" && !state.preflightReady) {
      return;
    }
    if (state.mode === "saved" && String(strategiesSelect.value || "").trim().length === 0) {
      return;
    }
    runButton.disabled = false;
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
        invalidatePreflight();
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

  const ensureDefaultsForBlock = (block) => {
    const descriptor = state.indicatorsById.get(block.indicatorId);
    if (!descriptor) {
      return;
    }

    const sourceSpec = Array.isArray(descriptor.inputs)
      ? descriptor.inputs.find((item) => String(asRecord(item).name || "") === "source")
      : null;
    if (sourceSpec !== null) {
      const sourceRecord = asRecord(sourceSpec);
      if (typeof block.sourceValues !== "string" || block.sourceValues.trim().length === 0) {
        if (
          Array.isArray(sourceRecord.allowed_values)
          && sourceRecord.allowed_values.length > 0
        ) {
          block.sourceValues = String(sourceRecord.allowed_values[0]);
        } else {
          block.sourceValues = readDefaultFieldValue(sourceRecord.default);
        }
      }
    } else {
      block.sourceValues = "";
    }

    const params = Array.isArray(descriptor.params) ? descriptor.params : [];
    params.forEach((paramSpec) => {
      const record = asRecord(paramSpec);
      const paramName = String(record.name || "");
      if (!paramName) {
        return;
      }
      if (
        typeof block.paramValues[paramName] === "string"
        && block.paramValues[paramName].trim().length > 0
      ) {
        return;
      }
      if (Array.isArray(record.enum_values) && record.enum_values.length > 0) {
        block.paramValues[paramName] = String(record.enum_values[0]);
        return;
      }
      block.paramValues[paramName] = readDefaultFieldValue(record.default);
    });
  };

  const addIndicatorBlock = () => {
    if (state.indicators.length === 0) {
      return;
    }
    const firstIndicator = state.indicators[0];
    const block = {
      uid: `backtest-indicator-${state.nextBlockNumber}`,
      indicatorId: String(firstIndicator.indicator_id),
      sourceValues: "",
      paramValues: {},
    };
    state.nextBlockNumber += 1;
    ensureDefaultsForBlock(block);
    state.blocks.push(block);
    renderIndicatorBlocks();
    invalidatePreflight();
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
            invalidatePreflight();
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
            invalidatePreflight();
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
            invalidatePreflight();
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
        block.paramValues = {};
        block.sourceValues = "";
        ensureDefaultsForBlock(block);
        renderIndicatorBlocks();
        invalidatePreflight();
      });
      card.appendChild(indicatorSelect);

      const descriptorInputs = Array.isArray(descriptor.inputs) ? descriptor.inputs : [];
      const sourceSpec = descriptorInputs.find(
        (item) => String(asRecord(item).name || "") === "source",
      );
      if (sourceSpec) {
        const sourceRecord = asRecord(sourceSpec);
        const sourceLabel = document.createElement("label");
        sourceLabel.setAttribute("for", `${block.uid}-source-values`);
        sourceLabel.textContent = "source values (csv)";
        card.appendChild(sourceLabel);

        const sourceInput = document.createElement("input");
        sourceInput.id = `${block.uid}-source-values`;
        sourceInput.type = "text";
        sourceInput.value = String(block.sourceValues || "");
        if (
          Array.isArray(sourceRecord.allowed_values)
          && sourceRecord.allowed_values.length > 0
        ) {
          sourceInput.placeholder = sourceRecord.allowed_values.join(",");
        }
        sourceInput.addEventListener("change", () => {
          block.sourceValues = sourceInput.value.trim();
          invalidatePreflight();
        });
        card.appendChild(sourceInput);
      }

      const paramsTitle = document.createElement("h4");
      paramsTitle.textContent = "params (explicit values)";
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

          const fieldLabel = document.createElement("label");
          fieldLabel.setAttribute("for", `${block.uid}-param-${paramName}`);
          fieldLabel.textContent = `${paramName} (${paramKind}) values`;
          card.appendChild(fieldLabel);

          const inputField = document.createElement("input");
          inputField.id = `${block.uid}-param-${paramName}`;
          inputField.type = "text";
          inputField.value = String(block.paramValues[paramName] || "");
          if (Array.isArray(paramRecord.enum_values) && paramRecord.enum_values.length > 0) {
            inputField.placeholder = paramRecord.enum_values.join(",");
          }
          inputField.addEventListener("change", () => {
            block.paramValues[paramName] = inputField.value.trim();
            invalidatePreflight();
          });
          card.appendChild(inputField);
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

    const slAxis = parseRiskAxis({
      modeNode: riskSlMode,
      valuesNode: riskSlValues,
      startNode: riskSlStart,
      stopNode: riskSlStop,
      stepNode: riskSlStep,
      sideName: "sl",
    });
    const tpAxis = parseRiskAxis({
      modeNode: riskTpMode,
      valuesNode: riskTpValues,
      startNode: riskTpStart,
      stopNode: riskTpStop,
      stepNode: riskTpStep,
      sideName: "tp",
    });
    const slPct = readOptionalNumber(riskSlPct, "risk_grid.sl_pct");
    const tpPct = readOptionalNumber(riskTpPct, "risk_grid.tp_pct");

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
    return {
      directionMode: directionMode.length > 0 ? directionMode : null,
      sizingMode: sizingMode.length > 0 ? sizingMode : null,
      execution: buildExecutionPayload(),
      riskGrid: buildRiskGridPayload(),
      topK: readOptionalPositiveInt(topKInput, "top_k"),
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
        const rawValue = String(block.paramValues[paramName] || "").trim();
        if (rawValue.length === 0) {
          return;
        }
        params[paramName] = parseAxisValuesCsv(
          rawValue,
          String(paramRecord.kind || "string"),
          `indicator ${block.indicatorId} param ${paramName}`,
        );
      });

      const grid = {
        indicator_id: block.indicatorId,
        params,
      };

      const descriptorInputs = Array.isArray(descriptor.inputs) ? descriptor.inputs : [];
      const sourceSpec = descriptorInputs.find(
        (item) => String(asRecord(item).name || "") === "source",
      );
      if (sourceSpec) {
        const rawSource = String(block.sourceValues || "").trim();
        if (rawSource.length > 0) {
          grid.source = parseAxisValuesCsv(
            rawSource,
            "string",
            `indicator ${block.indicatorId} source`,
          );
        }
      }
      return grid;
    });
  };

  const buildEstimateRisk = (riskGrid) => {
    const fallbackAxis = { mode: "explicit", values: [1.0] };
    if (riskGrid === null) {
      return {
        sl: fallbackAxis,
        tp: fallbackAxis,
      };
    }

    const slAxis = (
      riskGrid.sl
      || (typeof riskGrid.sl_pct === "number"
        ? { mode: "explicit", values: [riskGrid.sl_pct] }
        : fallbackAxis)
    );
    const tpAxis = (
      riskGrid.tp
      || (typeof riskGrid.tp_pct === "number"
        ? { mode: "explicit", values: [riskGrid.tp_pct] }
        : fallbackAxis)
    );
    return {
      sl: slAxis,
      tp: tpAxis,
    };
  };

  const buildTemplatePreflightPayload = () => {
    const marketId = Number(marketSelect.value || "0");
    if (marketId <= 0) {
      throw new Error("Please select market.");
    }
    if (String(symbolValue.value || "").trim().length === 0) {
      throw new Error("Please select symbol from suggestions.");
    }

    const timeframe = String(timeframeSelect.value || "").trim();
    if (!SUPPORTED_TIMEFRAMES.includes(timeframe)) {
      throw new Error("Unsupported timeframe selected.");
    }

    const advanced = buildAdvancedOptions();
    return {
      timeframe,
      time_range: parseTimeRange(),
      indicators: buildTemplateIndicatorGrids().map((item) => {
        const payload = {
          indicator_id: item.indicator_id,
          params: item.params,
        };
        if (item.source) {
          payload.source = item.source;
        }
        return payload;
      }),
      risk: buildEstimateRisk(advanced.riskGrid),
    };
  };

  const buildRunRequest = () => {
    const timeRange = parseTimeRange();
    const advanced = buildAdvancedOptions();
    const requestPayload = {
      time_range: timeRange,
    };

    if (advanced.topK !== null) {
      requestPayload.top_k = advanced.topK;
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
      if (!SUPPORTED_TIMEFRAMES.includes(timeframe)) {
        throw new Error("Unsupported timeframe selected.");
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
    clearPageError(pageRoot);
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
      state.indicators = indicatorsItems
        .map((item) => asRecord(item))
        .filter((item) => String(item.indicator_id || "").trim().length > 0)
        .sort((left, right) => compareStableStrings(
          String(left.indicator_id),
          String(right.indicator_id),
        ));
      state.indicatorsById = new Map(
        state.indicators.map((indicator) => [String(indicator.indicator_id), indicator]),
      );

      if (state.blocks.length === 0 && state.indicators.length > 0) {
        addIndicatorBlock();
      } else {
        renderIndicatorBlocks();
      }
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
    return reportNode;
  };

  const renderResults = (responsePayload) => {
    const response = asRecord(responsePayload);
    const metaPayload = {
      schema_version: response.schema_version,
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
      variantsBody.innerHTML = "<tr><td colspan=\"6\">No variants returned.</td></tr>";
    } else {
      variants.forEach((variant, index) => {
        const variantRecord = asRecord(variant);
        const row = document.createElement("tr");
        row.appendChild(buildCell(String(variantRecord.variant_index ?? index)));
        row.appendChild(buildCell(String(variantRecord.total_return_pct ?? "")));
        row.appendChild(buildCell(String(variantRecord.variant_key || "")));
        row.appendChild(buildCell(String(variantRecord.indicator_variant_key || "")));

        const reportCell = document.createElement("td");
        reportCell.appendChild(renderVariantReport(asRecord(variantRecord.report)));
        row.appendChild(reportCell);

        const actionsCell = document.createElement("td");
        const saveButton = buildActionButton({
          label: "Save as Strategy",
          onClick: () => {
            saveVariantAsStrategy(index);
          },
        });
        saveButton.classList.add("button-link--secondary");
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

  const runPreflight = async () => {
    clearPageError(pageRoot);
    if (state.mode !== "template") {
      return;
    }
    try {
      const requestPayload = buildTemplatePreflightPayload();
      state.preflightReady = false;
      updateRunAvailability();
      preflightButton.disabled = true;
      preflightLoading.classList.remove("hidden");

      const response = await fetch(estimatePath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      const totalVariants = Number(payload.total_variants || 0);
      const estimatedMemoryBytes = Number(payload.estimated_memory_bytes || 0);
      state.preflightReady = true;
      setPreflightSummary(
        [
          "Preflight passed.",
          `total_variants=${totalVariants}`,
          `estimated_memory_bytes=${estimatedMemoryBytes}`,
        ].join(" "),
      );
      updateRunAvailability();
    } catch (error) {
      state.preflightReady = false;
      setPreflightSummary("Preflight failed. Fix validation errors before run.");
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      updateRunAvailability();
    } finally {
      preflightLoading.classList.add("hidden");
      preflightButton.disabled = state.mode !== "template" || state.isRunning;
    }
  };

  const runBacktest = async () => {
    clearPageError(pageRoot);
    if (state.runType === "job") {
      showPageError(pageRoot, "Jobs UI is in WEB-EPIC-06.", []);
      return;
    }
    try {
      const request = buildRunRequest();
      state.isRunning = true;
      updateRunAvailability();
      runLoading.classList.remove("hidden");

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
      state.latestRun = {
        response: payload,
        context: request.context,
      };
      renderResults(payload);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    } finally {
      state.isRunning = false;
      runLoading.classList.add("hidden");
      updateRunAvailability();
    }
  };

  modeTemplate.addEventListener("change", () => {
    if (!modeTemplate.checked) {
      return;
    }
    state.mode = "template";
    updateModeSections();
    updateRunAvailability();
  });
  modeSaved.addEventListener("change", () => {
    if (!modeSaved.checked) {
      return;
    }
    state.mode = "saved";
    updateModeSections();
    updateRunAvailability();
    if (state.strategiesById.size === 0) {
      loadStrategies();
    }
  });
  runTypeSync.addEventListener("change", () => {
    if (runTypeSync.checked) {
      state.runType = "sync";
      updateRunAvailability();
    }
  });
  runTypeJob.addEventListener("change", () => {
    if (runTypeJob.checked) {
      state.runType = "job";
      updateRunAvailability();
    }
  });

  marketSelect.addEventListener("change", () => {
    clearSelectedSymbol();
    suggestionsList.innerHTML = "";
    invalidatePreflight();
  });
  symbolQuery.addEventListener("input", () => {
    clearSelectedSymbol();
    scheduleInstrumentSearch();
    invalidatePreflight();
  });
  timeframeSelect.addEventListener("change", invalidatePreflight);
  addIndicatorButton.addEventListener("click", addIndicatorBlock);
  preflightButton.addEventListener("click", runPreflight);
  refreshStrategiesButton.addEventListener("click", loadStrategies);
  strategiesSelect.addEventListener("change", updateRunAvailability);

  rangePresetSelect.addEventListener("change", () => {
    const preset = String(rangePresetSelect.value || "custom");
    applyRangePreset(preset);
    invalidatePreflight();
  });
  [rangeStartInput, rangeEndInput].forEach((node) => {
    const onRangeInputChanged = () => {
      rangePresetSelect.value = "custom";
      invalidatePreflight();
    };
    node.addEventListener("change", onRangeInputChanged);
    node.addEventListener("input", onRangeInputChanged);
  });

  [
    directionModeSelect,
    sizingModeSelect,
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
    topKInput,
    preselectInput,
    topTradesInput,
    warmupBarsInput,
  ].forEach((node) => {
    const onAdvancedChanged = () => {
      if (state.mode === "template") {
        invalidatePreflight();
      } else {
        updateRunAvailability();
      }
    };
    node.addEventListener("change", onAdvancedChanged);
    node.addEventListener("input", onAdvancedChanged);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runBacktest();
  });

  applyRangePreset("90d");
  updateModeSections();
  updateRunAvailability();
  loadReferences();
  loadStrategies();
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
          return JSON.stringify(item);
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
            return JSON.stringify(item);
          })
          .filter((item) => item.length > 0);
      } else if (response.status === 422) {
        const fallbackDetails = [];
        if (typeof detailRecord.error === "string") {
          fallbackDetails.push(`error: ${detailRecord.error}`);
        }
        Object.keys(detailRecord).forEach((key) => {
          if (key === "error" || key === "message") {
            return;
          }
          fallbackDetails.push(`${key}: ${JSON.stringify(detailRecord[key])}`);
        });
        details = fallbackDetails;
      }
    }
  }

  return { message, details };
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
    return { message: error.message, details };
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
