const STRATEGY_PAGE_SELECTOR = "[data-strategy-page]";
const SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"];
const TIMEFRAME_INDEX = new Map(
  SUPPORTED_TIMEFRAMES.map((code, index) => [code, index]),
);

document.addEventListener("DOMContentLoaded", () => {
  const pageRoot = document.querySelector(STRATEGY_PAGE_SELECTOR);
  if (pageRoot === null) {
    return;
  }

  const pageType = pageRoot.getAttribute("data-strategy-page");
  if (pageType === "list") {
    initListPage(pageRoot);
    return;
  }
  if (pageType === "details") {
    initDetailsPage(pageRoot);
    return;
  }
  if (pageType === "builder") {
    initBuilderPage(pageRoot);
  }
});

function initListPage(pageRoot) {
  const listPath = requireDataAttr(pageRoot, "apiListPath");
  const clonePath = requireDataAttr(pageRoot, "apiClonePath");
  const deletePathTemplate = requireDataAttr(pageRoot, "apiDeletePathTemplate");
  const detailsPathTemplate = requireDataAttr(pageRoot, "detailsPathTemplate");

  const tableBody = pageRoot.querySelector("#strategies-table-body");
  const filterSymbol = pageRoot.querySelector("#strategy-filter-symbol");
  const filterMarketType = pageRoot.querySelector("#strategy-filter-market-type");
  const filterTimeframe = pageRoot.querySelector("#strategy-filter-timeframe");
  const filterReset = pageRoot.querySelector("#strategy-filter-reset");

  if (
    tableBody === null
    || filterSymbol === null
    || filterMarketType === null
    || filterTimeframe === null
    || filterReset === null
  ) {
    return;
  }

  const state = {
    strategies: [],
  };

  const applyFiltersAndRender = () => {
    const symbolNeedle = filterSymbol.value.trim().toUpperCase();
    const marketTypeNeedle = filterMarketType.value.trim().toLowerCase();
    const timeframeNeedle = filterTimeframe.value.trim().toLowerCase();

    const filtered = state.strategies.filter((strategy) => {
      const spec = asRecord(strategy.spec);
      const instrument = asRecord(spec.instrument_id);
      const symbol = String(instrument.symbol || "").toUpperCase();
      const marketType = String(spec.market_type || "").toLowerCase();
      const timeframe = String(spec.timeframe || "").toLowerCase();

      if (symbolNeedle && !symbol.includes(symbolNeedle)) {
        return false;
      }
      if (marketTypeNeedle && marketType !== marketTypeNeedle) {
        return false;
      }
      if (timeframeNeedle && timeframe !== timeframeNeedle) {
        return false;
      }
      return true;
    });

    renderStrategiesTable({
      tableBody,
      strategies: filtered,
      detailsPathTemplate,
    });
  };

  const refreshStrategies = async () => {
    clearPageError(pageRoot);
    tableBody.innerHTML = "<tr><td colspan=\"6\">Loading strategies...</td></tr>";

    try {
      const response = await fetch(listPath, { credentials: 'include' });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const payload = await response.json();
      state.strategies = Array.isArray(payload) ? payload.slice() : [];
      rebuildFilterOptions({
        strategies: state.strategies,
        marketTypeSelect: filterMarketType,
        timeframeSelect: filterTimeframe,
      });
      applyFiltersAndRender();
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
      tableBody.innerHTML = "<tr><td colspan=\"6\">Failed to load strategies.</td></tr>";
    }
  };

  tableBody.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.getAttribute("data-action");
    const strategyId = target.getAttribute("data-strategy-id");
    if (action === null || strategyId === null || strategyId.length === 0) {
      return;
    }

    if (action === "view") {
      window.location.assign(renderPathTemplate(detailsPathTemplate, strategyId));
      return;
    }

    clearPageError(pageRoot);

    try {
      if (action === "clone") {
        const cloneResponse = await fetch(clonePath, {
          method: "POST",
          credentials: 'include',
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_strategy_id: strategyId }),
        });
        if (!cloneResponse.ok) {
          throw await buildHttpError(cloneResponse);
        }
        await refreshStrategies();
        return;
      }

      if (action === "delete") {
        const confirmed = window.confirm(
          "Delete selected strategy? This operation archives the strategy.",
        );
        if (!confirmed) {
          return;
        }
        const deletePath = renderPathTemplate(deletePathTemplate, strategyId);
        const deleteResponse = await fetch(deletePath, {
          method: "DELETE",
          credentials: 'include',
        });
        if (!deleteResponse.ok) {
          throw await buildHttpError(deleteResponse);
        }
        await refreshStrategies();
      }
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  });

  filterSymbol.addEventListener("input", applyFiltersAndRender);
  filterMarketType.addEventListener("change", applyFiltersAndRender);
  filterTimeframe.addEventListener("change", applyFiltersAndRender);
  filterReset.addEventListener("click", () => {
    filterSymbol.value = "";
    filterMarketType.value = "";
    filterTimeframe.value = "";
    applyFiltersAndRender();
  });

  refreshStrategies();
}

function initDetailsPage(pageRoot) {
  const strategyId = requireDataAttr(pageRoot, "strategyId");
  const getPathTemplate = requireDataAttr(pageRoot, "apiGetPathTemplate");
  const clonePath = requireDataAttr(pageRoot, "apiClonePath");
  const deletePathTemplate = requireDataAttr(pageRoot, "apiDeletePathTemplate");
  const listPath = requireDataAttr(pageRoot, "listPath");

  const cloneButton = pageRoot.querySelector("#strategy-detail-clone");
  const deleteButton = pageRoot.querySelector("#strategy-detail-delete");
  const loadingNode = pageRoot.querySelector("#strategy-details-loading");
  const contentNode = pageRoot.querySelector("#strategy-details-content");
  const indicatorsList = pageRoot.querySelector("#strategy-indicators-list");
  const rawSpecNode = pageRoot.querySelector("#strategy-raw-spec");

  if (
    cloneButton === null
    || deleteButton === null
    || loadingNode === null
    || contentNode === null
    || indicatorsList === null
    || rawSpecNode === null
  ) {
    return;
  }

  const fields = {
    strategyId: pageRoot.querySelector("#strategy-field-id"),
    name: pageRoot.querySelector("#strategy-field-name"),
    createdAt: pageRoot.querySelector("#strategy-field-created-at"),
    marketId: pageRoot.querySelector("#strategy-field-market-id"),
    symbol: pageRoot.querySelector("#strategy-field-symbol"),
    marketType: pageRoot.querySelector("#strategy-field-market-type"),
    instrumentKey: pageRoot.querySelector("#strategy-field-instrument-key"),
    timeframe: pageRoot.querySelector("#strategy-field-timeframe"),
  };
  if (Object.values(fields).some((node) => node === null)) {
    return;
  }

  const strategyPath = renderPathTemplate(getPathTemplate, strategyId);
  const deletePath = renderPathTemplate(deletePathTemplate, strategyId);

  const renderStrategy = (strategy) => {
    const spec = asRecord(strategy.spec);
    const instrument = asRecord(spec.instrument_id);

    fields.strategyId.textContent = String(strategy.strategy_id || "");
    fields.name.textContent = String(strategy.name || "");
    fields.createdAt.textContent = String(strategy.created_at || "");
    fields.marketId.textContent = String(instrument.market_id || "");
    fields.symbol.textContent = String(instrument.symbol || "");
    fields.marketType.textContent = String(spec.market_type || "");
    fields.instrumentKey.textContent = String(spec.instrument_key || "");
    fields.timeframe.textContent = String(spec.timeframe || "");

    const indicators = Array.isArray(spec.indicators) ? spec.indicators : [];
    indicatorsList.innerHTML = "";
    if (indicators.length === 0) {
      const emptyItem = document.createElement("li");
      emptyItem.textContent = "No indicators configured.";
      indicatorsList.appendChild(emptyItem);
    } else {
      indicators.forEach((indicator, index) => {
        const item = asRecord(indicator);
        const row = document.createElement("li");
        row.className = "indicator-list-item";

        const title = document.createElement("strong");
        title.textContent = `${index + 1}. ${String(item.id || "")}`;
        row.appendChild(title);

        const inputsLabel = document.createElement("div");
        inputsLabel.className = "muted-text";
        inputsLabel.textContent = `inputs: ${JSON.stringify(asRecord(item.inputs))}`;
        row.appendChild(inputsLabel);

        const paramsLabel = document.createElement("div");
        paramsLabel.className = "muted-text";
        paramsLabel.textContent = `params: ${JSON.stringify(asRecord(item.params))}`;
        row.appendChild(paramsLabel);

        indicatorsList.appendChild(row);
      });
    }

    rawSpecNode.textContent = JSON.stringify(spec, null, 2);
  };

  const loadStrategy = async () => {
    clearPageError(pageRoot);
    loadingNode.classList.remove("hidden");
    contentNode.classList.add("hidden");

    try {
      const response = await fetch(strategyPath, { credentials: 'include' });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const strategy = await response.json();
      renderStrategy(strategy);
      loadingNode.classList.add("hidden");
      contentNode.classList.remove("hidden");
    } catch (error) {
      loadingNode.classList.add("hidden");
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  cloneButton.addEventListener("click", async () => {
    clearPageError(pageRoot);
    try {
      const response = await fetch(clonePath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_strategy_id: strategyId }),
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const cloned = await response.json();
      const clonedId = String(cloned.strategy_id || "").trim();
      if (clonedId.length > 0) {
        window.location.assign(`/strategies/${encodeURIComponent(clonedId)}`);
        return;
      }
      window.location.assign(listPath);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  });

  deleteButton.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Delete this strategy? This operation archives the strategy.",
    );
    if (!confirmed) {
      return;
    }

    clearPageError(pageRoot);
    try {
      const response = await fetch(deletePath, {
        method: "DELETE",
        credentials: 'include',
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      window.location.assign(listPath);
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  });

  loadStrategy();
}

function initBuilderPage(pageRoot) {
  const createPath = requireDataAttr(pageRoot, "apiCreatePath");
  const marketsPath = requireDataAttr(pageRoot, "apiMarketsPath");
  const instrumentsPath = requireDataAttr(pageRoot, "apiInstrumentsPath");
  const indicatorsPath = requireDataAttr(pageRoot, "apiIndicatorsPath");
  const detailsPathPrefix = requireDataAttr(pageRoot, "detailsPathPrefix");

  const form = pageRoot.querySelector("#strategy-builder-form");
  const marketSelect = pageRoot.querySelector("#builder-market-id");
  const symbolQuery = pageRoot.querySelector("#builder-symbol-query");
  const symbolValue = pageRoot.querySelector("#builder-symbol-value");
  const selectedSymbol = pageRoot.querySelector("#builder-selected-symbol");
  const suggestionsList = pageRoot.querySelector("#builder-symbol-suggestions");
  const timeframeSelect = pageRoot.querySelector("#builder-timeframe");
  const addIndicatorButton = pageRoot.querySelector("#builder-add-indicator");
  const blocksContainer = pageRoot.querySelector("#builder-indicator-blocks");
  const submitButton = pageRoot.querySelector("#builder-submit");

  if (
    form === null
    || marketSelect === null
    || symbolQuery === null
    || symbolValue === null
    || selectedSymbol === null
    || suggestionsList === null
    || timeframeSelect === null
    || addIndicatorButton === null
    || blocksContainer === null
    || submitButton === null
  ) {
    return;
  }

  const state = {
    markets: [],
    marketsById: new Map(),
    indicators: [],
    indicatorsById: new Map(),
    blocks: [],
    nextBlockNumber: 1,
    searchDebounceId: 0,
    instrumentsAbortController: null,
  };

  const clearSelectedSymbol = () => {
    symbolValue.value = "";
    selectedSymbol.textContent = "Selected symbol: none";
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

  const readDefaultValue = (defaultSpec) => {
    const spec = asRecord(defaultSpec);
    if (spec.mode === "explicit" && Array.isArray(spec.values) && spec.values.length > 0) {
      return spec.values[0];
    }
    if (spec.mode === "range" && typeof spec.start !== "undefined") {
      return spec.start;
    }
    return "";
  };

  const coerceParamValue = (kind, value) => {
    if (kind === "int") {
      const parsedInt = Number.parseInt(String(value), 10);
      return Number.isNaN(parsedInt) ? "" : parsedInt;
    }
    if (kind === "float") {
      const parsedFloat = Number.parseFloat(String(value));
      return Number.isNaN(parsedFloat) ? "" : parsedFloat;
    }
    if (kind === "bool") {
      if (value === true || value === "true") {
        return true;
      }
      if (value === false || value === "false") {
        return false;
      }
      return "";
    }
    return value;
  };

  const ensureDefaultsForBlock = (block) => {
    const descriptor = state.indicatorsById.get(block.indicatorId);
    if (!descriptor) {
      return;
    }
    descriptor.inputs.forEach((inputSpec) => {
      const inputName = String(inputSpec.name || "");
      if (!inputName) {
        return;
      }
      if (typeof block.inputs[inputName] !== "undefined") {
        return;
      }
      if (Array.isArray(inputSpec.allowed_values) && inputSpec.allowed_values.length > 0) {
        block.inputs[inputName] = String(inputSpec.allowed_values[0]);
        return;
      }
      block.inputs[inputName] = readDefaultValue(inputSpec.default);
    });
    descriptor.params.forEach((paramSpec) => {
      const paramName = String(paramSpec.name || "");
      if (!paramName) {
        return;
      }
      if (typeof block.params[paramName] !== "undefined") {
        return;
      }
      if (Array.isArray(paramSpec.enum_values) && paramSpec.enum_values.length > 0) {
        block.params[paramName] = paramSpec.enum_values[0];
        return;
      }
      block.params[paramName] = coerceParamValue(
        String(paramSpec.kind || ""),
        readDefaultValue(paramSpec.default),
      );
    });
  };

  const addIndicatorBlock = () => {
    if (state.indicators.length === 0) {
      return;
    }
    const firstIndicator = state.indicators[0];
    const block = {
      uid: `indicator-block-${state.nextBlockNumber}`,
      indicatorId: firstIndicator.indicator_id,
      inputs: {},
      params: {},
    };
    state.nextBlockNumber += 1;
    ensureDefaultsForBlock(block);
    state.blocks.push(block);
    renderIndicatorBlocks();
  };

  const renderIndicatorBlocks = () => {
    blocksContainer.innerHTML = "";
    if (state.blocks.length === 0) {
      const emptyNode = document.createElement("p");
      emptyNode.className = "muted-text";
      emptyNode.textContent = "No indicator blocks yet.";
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
      title.textContent = `Indicator #${index + 1}`;
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
        option.value = indicator.indicator_id;
        option.textContent = `${indicator.indicator_id} - ${indicator.title}`;
        option.selected = indicator.indicator_id === block.indicatorId;
        indicatorSelect.appendChild(option);
      });
      indicatorSelect.addEventListener("change", () => {
        block.indicatorId = indicatorSelect.value;
        block.inputs = {};
        block.params = {};
        ensureDefaultsForBlock(block);
        renderIndicatorBlocks();
      });
      card.appendChild(indicatorSelect);

      const inputsTitle = document.createElement("h4");
      inputsTitle.textContent = "inputs";
      card.appendChild(inputsTitle);
      if (!Array.isArray(descriptor.inputs) || descriptor.inputs.length === 0) {
        const noInputsNode = document.createElement("p");
        noInputsNode.className = "muted-text";
        noInputsNode.textContent = "No inputs";
        card.appendChild(noInputsNode);
      } else {
        descriptor.inputs.forEach((inputSpec) => {
          const inputName = String(inputSpec.name || "");
          if (!inputName) {
            return;
          }

          const fieldLabel = document.createElement("label");
          fieldLabel.setAttribute("for", `${block.uid}-input-${inputName}`);
          fieldLabel.textContent = inputName;
          card.appendChild(fieldLabel);

          if (
            Array.isArray(inputSpec.allowed_values)
            && inputSpec.allowed_values.length > 0
          ) {
            const inputSelect = document.createElement("select");
            inputSelect.id = `${block.uid}-input-${inputName}`;
            inputSpec.allowed_values.forEach((allowedValue) => {
              const option = document.createElement("option");
              option.value = String(allowedValue);
              option.textContent = String(allowedValue);
              option.selected = String(block.inputs[inputName]) === String(allowedValue);
              inputSelect.appendChild(option);
            });
            inputSelect.addEventListener("change", () => {
              block.inputs[inputName] = inputSelect.value;
            });
            card.appendChild(inputSelect);
          } else {
            const inputField = document.createElement("input");
            inputField.id = `${block.uid}-input-${inputName}`;
            inputField.type = "text";
            inputField.value = String(block.inputs[inputName] || "");
            inputField.addEventListener("change", () => {
              block.inputs[inputName] = inputField.value.trim();
            });
            card.appendChild(inputField);
          }
        });
      }

      const paramsTitle = document.createElement("h4");
      paramsTitle.textContent = "params";
      card.appendChild(paramsTitle);
      if (!Array.isArray(descriptor.params) || descriptor.params.length === 0) {
        const noParamsNode = document.createElement("p");
        noParamsNode.className = "muted-text";
        noParamsNode.textContent = "No params";
        card.appendChild(noParamsNode);
      } else {
        descriptor.params.forEach((paramSpec) => {
          const paramName = String(paramSpec.name || "");
          const paramKind = String(paramSpec.kind || "");
          if (!paramName) {
            return;
          }

          const fieldLabel = document.createElement("label");
          fieldLabel.setAttribute("for", `${block.uid}-param-${paramName}`);
          fieldLabel.textContent = `${paramName} (${paramKind})`;
          card.appendChild(fieldLabel);

          if (Array.isArray(paramSpec.enum_values) && paramSpec.enum_values.length > 0) {
            const enumSelect = document.createElement("select");
            enumSelect.id = `${block.uid}-param-${paramName}`;
            paramSpec.enum_values.forEach((enumValue) => {
              const option = document.createElement("option");
              option.value = String(enumValue);
              option.textContent = String(enumValue);
              option.selected = String(block.params[paramName]) === String(enumValue);
              enumSelect.appendChild(option);
            });
            enumSelect.addEventListener("change", () => {
              block.params[paramName] = enumSelect.value;
            });
            card.appendChild(enumSelect);
            return;
          }

          if (paramKind === "bool") {
            const boolSelect = document.createElement("select");
            boolSelect.id = `${block.uid}-param-${paramName}`;
            ["true", "false"].forEach((literalValue) => {
              const option = document.createElement("option");
              option.value = literalValue;
              option.textContent = literalValue;
              option.selected = String(block.params[paramName]) === literalValue;
              boolSelect.appendChild(option);
            });
            boolSelect.addEventListener("change", () => {
              block.params[paramName] = boolSelect.value === "true";
            });
            card.appendChild(boolSelect);
            return;
          }

          const numericKinds = new Set(["int", "float"]);
          const inputField = document.createElement("input");
          inputField.id = `${block.uid}-param-${paramName}`;
          inputField.type = numericKinds.has(paramKind) ? "number" : "text";
          inputField.value = String(block.params[paramName] ?? "");
          if (inputField.type === "number") {
            if (typeof paramSpec.hard_min === "number") {
              inputField.min = String(paramSpec.hard_min);
            }
            if (typeof paramSpec.hard_max === "number") {
              inputField.max = String(paramSpec.hard_max);
            }
            if (typeof paramSpec.step === "number") {
              inputField.step = String(paramSpec.step);
            }
          }
          inputField.addEventListener("change", () => {
            block.params[paramName] = coerceParamValue(paramKind, inputField.value.trim());
          });
          card.appendChild(inputField);
        });
      }

      blocksContainer.appendChild(card);
    });
  };

  const buildPayload = () => {
    const marketId = Number(marketSelect.value || "0");
    if (marketId <= 0) {
      throw new Error("Please select a market.");
    }
    const market = state.marketsById.get(marketId);
    if (!market) {
      throw new Error("Selected market is unavailable.");
    }

    const symbol = symbolValue.value.trim();
    if (symbol.length === 0) {
      throw new Error("Please select a symbol from the suggestions list.");
    }

    const timeframe = timeframeSelect.value;
    if (!SUPPORTED_TIMEFRAMES.includes(timeframe)) {
      throw new Error("Unsupported timeframe selected.");
    }

    if (state.blocks.length === 0) {
      throw new Error("At least one indicator block is required.");
    }

    const indicators = state.blocks.map((block) => {
      const descriptor = state.indicatorsById.get(block.indicatorId);
      if (!descriptor) {
        throw new Error("Indicator registry is missing selected indicator.");
      }

      const inputs = {};
      descriptor.inputs.forEach((inputSpec) => {
        const name = String(inputSpec.name || "");
        if (!name) {
          return;
        }
        const rawValue = block.inputs[name];
        if (typeof rawValue !== "undefined" && rawValue !== "") {
          inputs[name] = rawValue;
        }
      });

      const params = {};
      descriptor.params.forEach((paramSpec) => {
        const name = String(paramSpec.name || "");
        if (!name) {
          return;
        }
        const rawValue = block.params[name];
        if (typeof rawValue !== "undefined" && rawValue !== "") {
          params[name] = rawValue;
        }
      });

      return {
        id: block.indicatorId,
        inputs,
        params,
      };
    });

    return {
      instrument_id: {
        market_id: market.market_id,
        symbol,
      },
      instrument_key: `${market.market_code}:${market.market_type}:${symbol}`,
      market_type: market.market_type,
      timeframe,
      indicators,
    };
  };

  const initializeBuilder = async () => {
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

      const indicatorsItems = Array.isArray(indicatorsPayload.items)
        ? indicatorsPayload.items
        : [];
      state.indicators = indicatorsItems
        .map((item) => asRecord(item))
        .filter((item) => String(item.indicator_id || "").trim().length > 0)
        .sort((left, right) => {
          const leftId = String(left.indicator_id);
          const rightId = String(right.indicator_id);
          return compareStableStrings(leftId, rightId);
        });
      state.indicatorsById = new Map(
        state.indicators.map((indicator) => [indicator.indicator_id, indicator]),
      );

      marketSelect.innerHTML = "<option value=\"\">Select market</option>";
      state.markets.forEach((market) => {
        const option = document.createElement("option");
        option.value = String(market.market_id);
        option.textContent = `${market.market_code} (${market.market_type})`;
        marketSelect.appendChild(option);
      });

      if (state.indicators.length > 0) {
        addIndicatorBlock();
      }
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    }
  };

  marketSelect.addEventListener("change", () => {
    clearSelectedSymbol();
    suggestionsList.innerHTML = "";
  });
  symbolQuery.addEventListener("input", () => {
    clearSelectedSymbol();
    scheduleInstrumentSearch();
  });
  addIndicatorButton.addEventListener("click", () => {
    addIndicatorBlock();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearPageError(pageRoot);

    try {
      const payload = buildPayload();
      submitButton.setAttribute("disabled", "disabled");
      const response = await fetch(createPath, {
        method: "POST",
        credentials: 'include',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw await buildHttpError(response);
      }
      const created = await response.json();
      const strategyId = String(created.strategy_id || "").trim();
      if (strategyId.length > 0) {
        window.location.assign(`${detailsPathPrefix}${encodeURIComponent(strategyId)}`);
        return;
      }
      window.location.assign("/strategies");
    } catch (error) {
      const normalized = normalizeError(error);
      showPageError(pageRoot, normalized.message, normalized.details);
    } finally {
      submitButton.removeAttribute("disabled");
    }
  });

  initializeBuilder();
}

function renderStrategiesTable({ tableBody, strategies, detailsPathTemplate }) {
  tableBody.innerHTML = "";
  if (strategies.length === 0) {
    tableBody.innerHTML = "<tr><td colspan=\"6\">No strategies found.</td></tr>";
    return;
  }

  strategies.forEach((strategy) => {
    const spec = asRecord(strategy.spec);
    const instrument = asRecord(spec.instrument_id);

    const row = document.createElement("tr");
    row.appendChild(buildCell(String(strategy.created_at || "")));
    row.appendChild(buildCell(String(strategy.name || "")));
    row.appendChild(buildCell(String(instrument.symbol || "")));
    row.appendChild(buildCell(String(spec.market_type || "")));
    row.appendChild(buildCell(String(spec.timeframe || "")));

    const actionsCell = document.createElement("td");
    actionsCell.className = "inline-actions";
    actionsCell.appendChild(
      buildStrategyActionButton({
        strategyId: String(strategy.strategy_id || ""),
        action: "view",
        label: "View",
      }),
    );
    actionsCell.appendChild(
      buildStrategyActionButton({
        strategyId: String(strategy.strategy_id || ""),
        action: "clone",
        label: "Clone",
      }),
    );
    actionsCell.appendChild(
      buildStrategyActionButton({
        strategyId: String(strategy.strategy_id || ""),
        action: "delete",
        label: "Delete",
        className: "button-link--danger",
      }),
    );
    row.appendChild(actionsCell);

    row.setAttribute(
      "data-details-path",
      renderPathTemplate(detailsPathTemplate, String(strategy.strategy_id || "")),
    );
    tableBody.appendChild(row);
  });
}

function rebuildFilterOptions({ strategies, marketTypeSelect, timeframeSelect }) {
  const marketTypes = Array.from(
    new Set(
      strategies
        .map((strategy) => String(asRecord(strategy.spec).market_type || ""))
        .filter((value) => value.length > 0),
    ),
  ).sort(compareStableStrings);

  const timeframes = Array.from(
    new Set(
      strategies
        .map((strategy) => String(asRecord(strategy.spec).timeframe || ""))
        .filter((value) => value.length > 0),
    ),
  ).sort((left, right) => {
    const leftIndex = TIMEFRAME_INDEX.has(left) ? TIMEFRAME_INDEX.get(left) : Number.MAX_VALUE;
    const rightIndex = TIMEFRAME_INDEX.has(right) ? TIMEFRAME_INDEX.get(right) : Number.MAX_VALUE;
    if (leftIndex !== rightIndex) {
      return leftIndex - rightIndex;
    }
    return compareStableStrings(left, right);
  });

  repopulateSelect(marketTypeSelect, marketTypes);
  repopulateSelect(timeframeSelect, timeframes);
}

function repopulateSelect(selectNode, values) {
  const previousValue = selectNode.value;
  selectNode.innerHTML = "<option value=\"\">All</option>";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectNode.appendChild(option);
  });
  selectNode.value = values.includes(previousValue) ? previousValue : "";
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
          const obj = asRecord(item);
          if (typeof obj.msg === "string" && obj.msg.length > 0) {
            return obj.msg;
          }
          return JSON.stringify(item);
        })
        .filter((item) => item.length > 0);
    } else if (detail !== null && typeof detail === "object") {
      const detailObject = asRecord(detail);
      if (typeof detailObject.message === "string" && detailObject.message.length > 0) {
        message = detailObject.message;
      } else if (response.status === 422) {
        message = "Validation error.";
      }
      if (Array.isArray(detailObject.errors)) {
        details = detailObject.errors
          .map((item) => {
            if (typeof item === "string") {
              return item;
            }
            const itemObject = asRecord(item);
            if (typeof itemObject.message === "string") {
              return itemObject.message;
            }
            return JSON.stringify(item);
          })
          .filter((item) => item.length > 0);
      }
    }
  }

  return { message, details };
}

function showPageError(pageRoot, message, details) {
  const banner = pageRoot.querySelector("#strategy-error-banner");
  if (banner !== null) {
    banner.textContent = message;
    banner.classList.remove("hidden");
  }

  const detailsContainer = pageRoot.querySelector("#strategy-validation-errors");
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
  const banner = pageRoot.querySelector("#strategy-error-banner");
  if (banner !== null) {
    banner.textContent = "";
    banner.classList.add("hidden");
  }
  const detailsContainer = pageRoot.querySelector("#strategy-validation-errors");
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

function renderPathTemplate(template, strategyId) {
  return template.replace("{strategy_id}", encodeURIComponent(strategyId));
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

function buildCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function buildStrategyActionButton({ strategyId, action, label, className = "" }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `button-link ${className}`.trim();
  button.textContent = label;
  button.setAttribute("data-strategy-id", strategyId);
  button.setAttribute("data-action", action);
  return button;
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
