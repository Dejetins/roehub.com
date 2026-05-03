import { apiRequest } from "../core/api.js";
import { delegate, qs, ready, setBusy } from "../core/dom.js";
import { formatDateTime } from "../core/formatters.js";
import { getCurrentLocale, translate } from "../core/locale.js";
import { notify } from "../core/notifications.js";

const PAGE_SELECTOR = "[data-strategy-page]";
const TIMEFRAMES = Object.freeze(["1m", "5m", "15m", "30m", "1h", "4h", "1d"]);

ready(() => {
  const root = qs(PAGE_SELECTOR);
  if (!root) {
    return;
  }

  const pageType = root.getAttribute("data-strategy-page");
  if (pageType === "list") {
    initStrategyList(root);
    return;
  }
  if (pageType === "create") {
    initStrategyCreate(root);
    return;
  }
  if (pageType === "detail") {
    initStrategyDetail(root);
  }
});

function initStrategyList(root) {
  const listPath = requiredData(root, "apiListPath");
  const clonePath = requiredData(root, "apiClonePath");
  const deleteTemplate = requiredData(root, "apiDeletePathTemplate");
  const detailsTemplate = requiredData(root, "detailsPathTemplate");
  const tableBody = requiredElement(root, "#strategies-table-body");
  const filterSymbol = requiredElement(root, "#strategy-filter-symbol");
  const filterMarket = requiredElement(root, "#strategy-filter-market-type");
  const filterTimeframe = requiredElement(root, "#strategy-filter-timeframe");
  const filterReset = requiredElement(root, "#strategy-filter-reset");

  const state = {
    strategies: [],
    loading: false,
    mutating: false,
  };

  const render = () => {
    renderFilterOptions({ state, filterMarket, filterTimeframe });
    renderStrategiesTable({
      tableBody,
      strategies: filterStrategies({
        strategies: state.strategies,
        symbol: filterSymbol.value,
        marketType: filterMarket.value,
        timeframe: filterTimeframe.value,
      }),
      detailsTemplate,
    });
  };

  const loadStrategies = async () => {
    if (state.loading) {
      return;
    }
    state.loading = true;
    clearError(root);
    tableBody.innerHTML = rowHtml(6, translate("strategies.state.loading"));
    try {
      const payload = await apiRequest(listPath);
      state.strategies = Array.isArray(payload) ? payload.map(asRecord) : [];
      render();
    } catch (error) {
      showError(root, error);
      tableBody.innerHTML = rowHtml(6, translate("strategies.state.load_failed"));
    } finally {
      state.loading = false;
    }
  };

  filterSymbol.addEventListener("input", render);
  filterMarket.addEventListener("change", render);
  filterTimeframe.addEventListener("change", render);
  filterReset.addEventListener("click", () => {
    filterSymbol.value = "";
    filterMarket.value = "";
    filterTimeframe.value = "";
    render();
  });

  delegate(root, "click", "[data-strategy-action]", async (event, element) => {
    event.preventDefault();
    if (state.mutating) {
      return;
    }
    const strategyId = element.getAttribute("data-strategy-id") || "";
    const action = element.getAttribute("data-strategy-action") || "";
    if (!strategyId) {
      return;
    }
    if (action === "open") {
      window.location.assign(renderPath(detailsTemplate, strategyId));
      return;
    }
    if (action === "clone") {
      state.mutating = true;
      await cloneStrategy({ root, clonePath, strategyId, trigger: element });
      state.mutating = false;
      return;
    }
    if (action === "delete") {
      state.mutating = true;
      const confirmed = window.confirm(translate("strategies.confirm.delete"));
      if (!confirmed) {
        state.mutating = false;
        return;
      }
      setBusy(element, true);
      clearError(root);
      try {
        await apiRequest(renderPath(deleteTemplate, strategyId), { method: "DELETE" });
        notify(translate("strategies.notify.deleted"), { tone: "info" });
        await loadStrategies();
      } catch (error) {
        showError(root, error);
      } finally {
        state.mutating = false;
        setBusy(element, false);
      }
    }
  });

  document.addEventListener("roehub:locale-change", render);
  loadStrategies();
}

function initStrategyCreate(root) {
  const createPath = requiredData(root, "apiCreatePath");
  const marketsPath = requiredData(root, "apiMarketsPath");
  const instrumentsPath = requiredData(root, "apiInstrumentsPath");
  const indicatorsPath = requiredData(root, "apiIndicatorsPath");
  const detailsPrefix = requiredData(root, "detailsPathPrefix");

  const form = requiredElement(root, "#strategy-create-form");
  const marketSelect = requiredElement(root, "#strategy-market-id");
  const symbolQuery = requiredElement(root, "#strategy-symbol-query");
  const symbolValue = requiredElement(root, "#strategy-symbol-value");
  const selectedSymbol = requiredElement(root, "#strategy-selected-symbol");
  const suggestionsList = requiredElement(root, "#strategy-symbol-suggestions");
  const timeframeSelect = requiredElement(root, "#strategy-timeframe");
  const blocksContainer = requiredElement(root, "#strategy-indicator-blocks");
  const addIndicatorButton = requiredElement(root, "#strategy-add-indicator");
  const submitButton = requiredElement(root, "#strategy-create-submit");

  const state = {
    markets: [],
    marketsById: new Map(),
    indicators: [],
    indicatorsById: new Map(),
    blocks: [],
    nextBlockNumber: 1,
    loading: false,
    submitting: false,
    searchDebounceId: 0,
    instrumentsAbortController: null,
  };

  const renderBlocks = () => renderIndicatorBlocks({ state, blocksContainer });
  const clearSelectedSymbol = () => {
    symbolValue.value = "";
    selectedSymbol.textContent = translate("strategies.create.no_symbol");
  };

  const searchInstruments = async () => {
    const marketId = Number(marketSelect.value || "0");
    const query = symbolQuery.value.trim().toUpperCase();
    if (marketId <= 0 || query.length === 0) {
      suggestionsList.replaceChildren();
      return;
    }

    state.instrumentsAbortController?.abort();
    const controller = new AbortController();
    state.instrumentsAbortController = controller;

    const requestUrl = new URL(instrumentsPath, window.location.origin);
    requestUrl.searchParams.set("market_id", String(marketId));
    requestUrl.searchParams.set("q", query);
    requestUrl.searchParams.set("limit", "20");

    try {
      const payload = await apiRequest(`${requestUrl.pathname}${requestUrl.search}`, {
        signal: controller.signal,
      });
      const symbols = Array.isArray(payload?.items)
        ? payload.items
            .map((item) => String(asRecord(item).symbol || "").trim().toUpperCase())
            .filter(Boolean)
        : [];
      renderSymbolSuggestions({ suggestionsList, symbolQuery, symbolValue, selectedSymbol, symbols });
    } catch (error) {
      if (error?.code === "aborted") {
        return;
      }
      showError(root, error);
    }
  };

  const scheduleInstrumentSearch = () => {
    if (state.searchDebounceId !== 0) {
      window.clearTimeout(state.searchDebounceId);
    }
    state.searchDebounceId = window.setTimeout(searchInstruments, 220);
  };

  const loadReferences = async () => {
    if (state.loading) {
      return;
    }
    state.loading = true;
    clearError(root);
    try {
      const [marketsPayload, indicatorsPayload] = await Promise.all([
        apiRequest(marketsPath),
        apiRequest(indicatorsPath),
      ]);
      state.markets = normalizeMarkets(marketsPayload);
      state.marketsById = new Map(
        state.markets.map((market) => [Number(market.market_id), market]),
      );
      state.indicators = normalizeIndicators(indicatorsPayload);
      state.indicatorsById = new Map(
        state.indicators.map((indicator) => [String(indicator.indicator_id), indicator]),
      );
      renderMarketOptions({ marketSelect, markets: state.markets });
      if (state.indicators.length > 0 && state.blocks.length === 0) {
        addIndicatorBlock(state);
      }
      renderBlocks();
    } catch (error) {
      showError(root, error);
      blocksContainer.textContent = translate("strategies.state.reference_failed");
    } finally {
      state.loading = false;
    }
  };

  marketSelect.addEventListener("change", () => {
    clearSelectedSymbol();
    suggestionsList.replaceChildren();
  });
  symbolQuery.addEventListener("input", () => {
    clearSelectedSymbol();
    scheduleInstrumentSearch();
  });
  addIndicatorButton.addEventListener("click", () => {
    addIndicatorBlock(state);
    renderBlocks();
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.submitting) {
      return;
    }
    clearError(root);
    state.submitting = true;
    setBusy(submitButton, true);
    try {
      const payload = buildCreatePayload({
        state,
        marketSelect,
        symbolQuery,
        symbolValue,
        timeframeSelect,
      });
      const created = await apiRequest(createPath, { method: "POST", body: payload });
      const strategyId = String(created?.strategy_id || "").trim();
      window.location.assign(
        strategyId ? `${detailsPrefix}${encodeURIComponent(strategyId)}` : "/strategies",
      );
    } catch (error) {
      showError(root, error);
    } finally {
      state.submitting = false;
      setBusy(submitButton, false);
    }
  });

  document.addEventListener("roehub:locale-change", () => {
    if (symbolValue.value) {
      selectedSymbol.textContent = translate("strategies.create.selected_symbol", {
        symbol: symbolValue.value,
      });
    } else {
      clearSelectedSymbol();
    }
    renderBlocks();
  });
  loadReferences();
}

function initStrategyDetail(root) {
  const strategyId = requiredData(root, "strategyId");
  const getTemplate = requiredData(root, "apiGetPathTemplate");
  const clonePath = requiredData(root, "apiClonePath");
  const deleteTemplate = requiredData(root, "apiDeletePathTemplate");
  const listPath = requiredData(root, "listPath");
  const loadingNode = requiredElement(root, "#strategy-detail-loading");
  const contentNode = requiredElement(root, "[data-strategy-detail-content]");
  const cloneButton = requiredElement(root, "#strategy-detail-clone");
  const deleteButton = requiredElement(root, "#strategy-detail-delete");

  const state = {
    strategy: null,
    loading: false,
    mutating: false,
  };

  const render = () => {
    if (!state.strategy) {
      return;
    }
    renderStrategyDetail(root, state.strategy);
  };

  const loadStrategy = async () => {
    if (state.loading) {
      return;
    }
    state.loading = true;
    clearError(root);
    loadingNode.hidden = false;
    contentNode.hidden = true;
    try {
      state.strategy = asRecord(await apiRequest(renderPath(getTemplate, strategyId)));
      render();
      loadingNode.hidden = true;
      contentNode.hidden = false;
    } catch (error) {
      showError(root, error);
      loadingNode.textContent = translate("strategies.state.load_failed");
    } finally {
      state.loading = false;
    }
  };

  cloneButton.addEventListener("click", async () => {
    if (state.mutating) {
      return;
    }
    state.mutating = true;
    await cloneStrategy({ root, clonePath, strategyId, trigger: cloneButton });
    state.mutating = false;
  });

  deleteButton.addEventListener("click", async () => {
    if (state.mutating) {
      return;
    }
    state.mutating = true;
    const confirmed = window.confirm(translate("strategies.confirm.delete"));
    if (!confirmed) {
      state.mutating = false;
      return;
    }
    setBusy(deleteButton, true);
    clearError(root);
    try {
      await apiRequest(renderPath(deleteTemplate, strategyId), { method: "DELETE" });
      window.location.assign(listPath);
    } catch (error) {
      showError(root, error);
    } finally {
      state.mutating = false;
      setBusy(deleteButton, false);
    }
  });

  document.addEventListener("roehub:locale-change", render);
  loadStrategy();
}

async function cloneStrategy({ root, clonePath, strategyId, trigger }) {
  setBusy(trigger, true);
  clearError(root);
  try {
    const cloned = await apiRequest(clonePath, {
      method: "POST",
      body: { source_strategy_id: strategyId },
    });
    const clonedId = String(cloned?.strategy_id || "").trim();
    notify(translate("strategies.notify.cloned"), { tone: "info" });
    if (clonedId) {
      window.location.assign(`/strategies/${encodeURIComponent(clonedId)}`);
    }
  } catch (error) {
    showError(root, error);
  } finally {
    setBusy(trigger, false);
  }
}

function renderStrategiesTable({ tableBody, strategies, detailsTemplate }) {
  tableBody.replaceChildren();
  if (strategies.length === 0) {
    tableBody.innerHTML = rowHtml(6, translate("strategies.state.empty"));
    return;
  }

  strategies.forEach((strategy) => {
    const spec = readSpec(strategy);
    const row = document.createElement("tr");
    row.append(
      cell(formatDateTime(strategy.created_at, { locale: getCurrentLocale() })),
      nameCell(strategy, detailsTemplate),
      cell(`${spec.instrument_key || "-"} / ${spec.market_type || "-"}`),
      cell(spec.timeframe || "-"),
      cell(String(spec.indicators.length)),
      actionsCell(strategy),
    );
    tableBody.append(row);
  });
}

function nameCell(strategy, detailsTemplate) {
  const td = document.createElement("td");
  const wrapper = document.createElement("span");
  wrapper.className = "rh-strategy-name-cell";
  const link = document.createElement("a");
  link.href = renderPath(detailsTemplate, strategy.strategy_id);
  link.textContent = String(strategy.name || translate("strategies.state.unnamed"));
  const code = document.createElement("code");
  code.textContent = String(strategy.strategy_id || "");
  wrapper.append(link, code);
  td.append(wrapper);
  return td;
}

function actionsCell(strategy) {
  const td = document.createElement("td");
  const row = document.createElement("span");
  row.className = "rh-action-row";
  row.append(
    actionButton({
      strategyId: strategy.strategy_id,
      action: "open",
      label: translate("strategies.actions.open"),
    }),
    actionButton({
      strategyId: strategy.strategy_id,
      action: "clone",
      label: translate("strategies.actions.clone"),
    }),
    actionButton({
      strategyId: strategy.strategy_id,
      action: "delete",
      label: translate("strategies.actions.delete"),
      className: "rh-button--danger",
    }),
  );
  td.append(row);
  return td;
}

function actionButton({ strategyId, action, label, className = "" }) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `rh-button ${className}`.trim();
  button.textContent = label;
  button.setAttribute("data-strategy-action", action);
  button.setAttribute("data-strategy-id", String(strategyId || ""));
  return button;
}

function renderFilterOptions({ state, filterMarket, filterTimeframe }) {
  syncOptions({
    select: filterMarket,
    values: uniqueSorted(state.strategies.map((strategy) => readSpec(strategy).market_type)),
    allLabel: translate("strategies.filters.all"),
  });
  syncOptions({
    select: filterTimeframe,
    values: uniqueSorted(state.strategies.map((strategy) => readSpec(strategy).timeframe)),
    allLabel: translate("strategies.filters.all"),
  });
}

function syncOptions({ select, values, allLabel }) {
  const selected = select.value;
  select.replaceChildren(option("", allLabel));
  values.forEach((value) => select.append(option(value, value)));
  select.value = values.includes(selected) ? selected : "";
}

function filterStrategies({ strategies, symbol, marketType, timeframe }) {
  const normalizedSymbol = symbol.trim().toUpperCase();
  return strategies.filter((strategy) => {
    const spec = readSpec(strategy);
    const instrument = asRecord(spec.instrument_id);
    const matchesSymbol =
      normalizedSymbol.length === 0 ||
      String(instrument.symbol || "").toUpperCase().includes(normalizedSymbol) ||
      String(spec.instrument_key || "").toUpperCase().includes(normalizedSymbol);
    const matchesMarket = marketType.length === 0 || spec.market_type === marketType;
    const matchesTimeframe = timeframe.length === 0 || spec.timeframe === timeframe;
    return matchesSymbol && matchesMarket && matchesTimeframe;
  });
}

function renderMarketOptions({ marketSelect, markets }) {
  marketSelect.replaceChildren(option("", translate("strategies.create.select_market")));
  markets.forEach((market) => {
    marketSelect.append(
      option(
        String(market.market_id),
        `${market.market_code} (${market.market_type})`,
      ),
    );
  });
}

function renderSymbolSuggestions({
  suggestionsList,
  symbolQuery,
  symbolValue,
  selectedSymbol,
  symbols,
}) {
  suggestionsList.replaceChildren();
  symbols.forEach((symbol) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rh-button rh-button--ghost";
    button.textContent = symbol;
    button.addEventListener("click", () => {
      symbolValue.value = symbol;
      symbolQuery.value = symbol;
      selectedSymbol.textContent = translate("strategies.create.selected_symbol", { symbol });
      suggestionsList.replaceChildren();
    });
    item.append(button);
    suggestionsList.append(item);
  });
}

function addIndicatorBlock(state) {
  const descriptor = state.indicators[0];
  if (!descriptor) {
    return;
  }
  const block = {
    uid: `strategy-indicator-${state.nextBlockNumber}`,
    indicatorId: String(descriptor.indicator_id),
    inputs: {},
    params: {},
  };
  state.nextBlockNumber += 1;
  ensureBlockDefaults({ block, state });
  state.blocks.push(block);
}

function renderIndicatorBlocks({ state, blocksContainer }) {
  blocksContainer.replaceChildren();
  if (state.indicators.length === 0) {
    blocksContainer.textContent = translate("strategies.create.no_indicators");
    return;
  }
  if (state.blocks.length === 0) {
    blocksContainer.textContent = translate("strategies.create.no_blocks");
    return;
  }

  state.blocks.forEach((block, index) => {
    ensureBlockDefaults({ block, state });
    const descriptor = state.indicatorsById.get(block.indicatorId);
    if (!descriptor) {
      return;
    }

    const card = document.createElement("section");
    card.className = "rh-indicator-card";
    const header = document.createElement("div");
    header.className = "rh-indicator-card__header";
    const title = document.createElement("h3");
    title.textContent = translate("strategies.create.indicator_number", {
      number: index + 1,
    });
    const actions = document.createElement("span");
    actions.className = "rh-action-row";
    actions.append(
      smallButton(translate("strategies.actions.up"), () => {
        moveBlock(state.blocks, index, index - 1);
        renderIndicatorBlocks({ state, blocksContainer });
      }, index === 0),
      smallButton(translate("strategies.actions.down"), () => {
        moveBlock(state.blocks, index, index + 1);
        renderIndicatorBlocks({ state, blocksContainer });
      }, index === state.blocks.length - 1),
      smallButton(translate("strategies.actions.remove"), () => {
        state.blocks = state.blocks.filter((candidate) => candidate.uid !== block.uid);
        renderIndicatorBlocks({ state, blocksContainer });
      }, false, "rh-button--danger"),
    );
    header.append(title, actions);
    card.append(header);

    const fields = document.createElement("div");
    fields.className = "rh-indicator-card__fields";
    fields.append(renderIndicatorSelect({ state, block, blocksContainer }));
    renderAxisFields({
      container: fields,
      specs: Array.isArray(descriptor.inputs) ? descriptor.inputs : [],
      values: block.inputs,
      uid: block.uid,
      group: "input",
    });
    renderAxisFields({
      container: fields,
      specs: Array.isArray(descriptor.params) ? descriptor.params : [],
      values: block.params,
      uid: block.uid,
      group: "param",
    });
    card.append(fields);
    blocksContainer.append(card);
  });
}

function renderIndicatorSelect({ state, block, blocksContainer }) {
  const label = document.createElement("label");
  label.className = "rh-field";
  label.setAttribute("for", `${block.uid}-indicator`);
  label.append(spanText(translate("strategies.fields.indicator")));
  const select = document.createElement("select");
  select.id = `${block.uid}-indicator`;
  state.indicators.forEach((indicator) => {
    select.append(
      option(
        String(indicator.indicator_id),
        `${indicator.indicator_id} - ${indicator.title || indicator.group || ""}`.trim(),
      ),
    );
  });
  select.value = block.indicatorId;
  select.addEventListener("change", () => {
    block.indicatorId = select.value;
    block.inputs = {};
    block.params = {};
    ensureBlockDefaults({ block, state });
    renderIndicatorBlocks({ state, blocksContainer });
  });
  label.append(select);
  return label;
}

function renderAxisFields({ container, specs, values, uid, group }) {
  specs.forEach((rawSpec) => {
    const spec = asRecord(rawSpec);
    const name = String(spec.name || "");
    if (!name) {
      return;
    }
    const label = document.createElement("label");
    label.className = "rh-field";
    label.setAttribute("for", `${uid}-${group}-${name}`);
    label.append(spanText(`${name}${group === "param" && spec.kind ? ` (${spec.kind})` : ""}`));

    const enumValues = group === "input" ? spec.allowed_values : spec.enum_values;
    if (Array.isArray(enumValues) && enumValues.length > 0) {
      const select = document.createElement("select");
      select.id = `${uid}-${group}-${name}`;
      enumValues.forEach((enumValue) => {
        select.append(option(String(enumValue), String(enumValue)));
      });
      select.value = String(values[name] ?? enumValues[0]);
      select.addEventListener("change", () => {
        values[name] = select.value;
      });
      label.append(select);
      container.append(label);
      return;
    }

    const input = document.createElement("input");
    input.id = `${uid}-${group}-${name}`;
    input.type = ["int", "float"].includes(String(spec.kind || "")) ? "number" : "text";
    input.value = String(values[name] ?? "");
    if (typeof spec.hard_min === "number") {
      input.min = String(spec.hard_min);
    }
    if (typeof spec.hard_max === "number") {
      input.max = String(spec.hard_max);
    }
    if (typeof spec.step === "number") {
      input.step = String(spec.step);
    }
    input.addEventListener("change", () => {
      values[name] = coerceValue(String(spec.kind || ""), input.value.trim());
    });
    label.append(input);
    container.append(label);
  });
}

function buildCreatePayload({ state, marketSelect, symbolQuery, symbolValue, timeframeSelect }) {
  const marketId = Number(marketSelect.value || "0");
  const market = state.marketsById.get(marketId);
  if (!market) {
    throw new Error(translate("strategies.validation.market"));
  }

  const symbol = String(symbolValue.value || symbolQuery.value || "").trim().toUpperCase();
  if (!symbol) {
    throw new Error(translate("strategies.validation.symbol"));
  }

  const timeframe = timeframeSelect.value;
  if (!TIMEFRAMES.includes(timeframe)) {
    throw new Error(translate("strategies.validation.timeframe"));
  }
  if (state.blocks.length === 0) {
    throw new Error(translate("strategies.validation.indicator"));
  }

  return {
    instrument_id: {
      market_id: market.market_id,
      symbol,
    },
    instrument_key: `${market.market_code}:${market.market_type}:${symbol}`,
    market_type: market.market_type,
    timeframe,
    indicators: state.blocks.map((block) => buildIndicatorPayload({ block, state })),
  };
}

function buildIndicatorPayload({ block, state }) {
  const descriptor = state.indicatorsById.get(block.indicatorId);
  if (!descriptor) {
    throw new Error(translate("strategies.validation.indicator"));
  }

  const inputs = {};
  for (const spec of Array.isArray(descriptor.inputs) ? descriptor.inputs : []) {
    const name = String(asRecord(spec).name || "");
    if (name && block.inputs[name] !== undefined && block.inputs[name] !== "") {
      inputs[name] = block.inputs[name];
    }
  }

  const params = {};
  for (const spec of Array.isArray(descriptor.params) ? descriptor.params : []) {
    const name = String(asRecord(spec).name || "");
    if (name && block.params[name] !== undefined && block.params[name] !== "") {
      params[name] = block.params[name];
    }
  }

  return {
    id: block.indicatorId,
    inputs,
    params,
  };
}

function renderStrategyDetail(root, strategy) {
  const spec = readSpec(strategy);
  const instrument = asRecord(spec.instrument_id);
  requiredElement(root, "#strategy-field-id").textContent = String(strategy.strategy_id || "");
  requiredElement(root, "#strategy-field-name").textContent = String(
    strategy.name || translate("strategies.state.unnamed"),
  );
  requiredElement(root, "#strategy-field-created-at").textContent = formatDateTime(
    strategy.created_at,
    { locale: getCurrentLocale() },
  );
  requiredElement(root, "#strategy-field-market").textContent =
    `${spec.instrument_key || "-"} / ${spec.market_type || "-"}`;
  requiredElement(root, "#strategy-field-symbol").textContent = String(instrument.symbol || "-");
  requiredElement(root, "#strategy-field-timeframe").textContent = String(spec.timeframe || "-");
  requiredElement(root, "#strategy-field-signal-template").textContent = String(
    spec.signal_template || "-",
  );
  requiredElement(root, "#strategy-raw-spec").textContent = JSON.stringify(spec.raw, null, 2);
  renderIndicatorList(requiredElement(root, "#strategy-indicators-list"), spec.indicators);
}

function renderIndicatorList(list, indicators) {
  list.replaceChildren();
  if (indicators.length === 0) {
    const item = document.createElement("li");
    item.className = "rh-indicator-list-item";
    item.textContent = translate("strategies.detail.no_indicators");
    list.append(item);
    return;
  }

  indicators.forEach((rawIndicator, index) => {
    const indicator = asRecord(rawIndicator);
    const item = document.createElement("li");
    item.className = "rh-indicator-list-item";
    const title = document.createElement("h3");
    title.textContent = `${index + 1}. ${indicator.name || indicator.kind || indicator.id || "-"}`;
    const definition = document.createElement("dl");
    definition.append(
      detailPair("id", String(indicator.id || indicator.name || indicator.kind || "-")),
      detailPair("inputs", JSON.stringify(asRecord(indicator.inputs))),
      detailPair("params", JSON.stringify(asRecord(indicator.params))),
    );
    item.append(title, definition);
    list.append(item);
  });
}

function detailPair(term, value) {
  const fragment = document.createDocumentFragment();
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  fragment.append(dt, dd);
  return fragment;
}

function ensureBlockDefaults({ block, state }) {
  const descriptor = state.indicatorsById.get(block.indicatorId);
  if (!descriptor) {
    return;
  }
  for (const spec of Array.isArray(descriptor.inputs) ? descriptor.inputs : []) {
    const item = asRecord(spec);
    const name = String(item.name || "");
    if (!name || block.inputs[name] !== undefined) {
      continue;
    }
    block.inputs[name] = defaultAxisValue(item);
  }
  for (const spec of Array.isArray(descriptor.params) ? descriptor.params : []) {
    const item = asRecord(spec);
    const name = String(item.name || "");
    if (!name || block.params[name] !== undefined) {
      continue;
    }
    block.params[name] = coerceValue(String(item.kind || ""), defaultAxisValue(item));
  }
}

function defaultAxisValue(spec) {
  const enumValues = spec.allowed_values || spec.enum_values;
  if (Array.isArray(enumValues) && enumValues.length > 0) {
    return enumValues[0];
  }
  const defaultSpec = asRecord(spec.default);
  if (defaultSpec.mode === "explicit" && Array.isArray(defaultSpec.values)) {
    return defaultSpec.values[0] ?? "";
  }
  if (defaultSpec.mode === "range" && defaultSpec.start !== undefined) {
    return defaultSpec.start;
  }
  return "";
}

function coerceValue(kind, value) {
  if (kind === "int") {
    const parsed = Number.parseInt(String(value), 10);
    return Number.isNaN(parsed) ? "" : parsed;
  }
  if (kind === "float") {
    const parsed = Number.parseFloat(String(value));
    return Number.isNaN(parsed) ? "" : parsed;
  }
  if (kind === "bool") {
    return value === true || value === "true";
  }
  return value;
}

function moveBlock(blocks, fromIndex, toIndex) {
  if (toIndex < 0 || toIndex >= blocks.length) {
    return;
  }
  const [item] = blocks.splice(fromIndex, 1);
  blocks.splice(toIndex, 0, item);
}

function smallButton(label, onClick, disabled = false, className = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `rh-button rh-button--ghost ${className}`.trim();
  button.textContent = label;
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}

function normalizeMarkets(payload) {
  return (Array.isArray(payload?.items) ? payload.items : [])
    .map(asRecord)
    .filter((market) => Number(market.market_id || 0) > 0)
    .sort((left, right) => Number(left.market_id) - Number(right.market_id));
}

function normalizeIndicators(payload) {
  return (Array.isArray(payload?.items) ? payload.items : [])
    .map(asRecord)
    .filter((indicator) => String(indicator.indicator_id || "").trim())
    .sort((left, right) =>
      String(left.indicator_id).localeCompare(String(right.indicator_id), "en"),
    );
}

function readSpec(strategy) {
  const raw = asRecord(strategy.spec);
  const instrument = asRecord(raw.instrument_id);
  return {
    raw,
    instrument_id: instrument,
    instrument_key: String(raw.instrument_key || ""),
    market_type: String(raw.market_type || ""),
    timeframe: String(raw.timeframe || ""),
    signal_template: String(raw.signal_template || ""),
    indicators: Array.isArray(raw.indicators) ? raw.indicators : [],
  };
}

function showError(root, error) {
  const banner = qs("#strategy-error-banner", root);
  if (!banner) {
    return;
  }
  const messages = [error?.message || translate("js.error.network")];
  if (error?.fieldErrors && Object.keys(error.fieldErrors).length > 0) {
    messages.push(...Object.values(error.fieldErrors).map((value) => String(value)));
  }
  banner.textContent = messages.join(" ");
  banner.classList.remove("rh-hidden");
}

function clearError(root) {
  const banner = qs("#strategy-error-banner", root);
  if (!banner) {
    return;
  }
  banner.textContent = "";
  banner.classList.add("rh-hidden");
}

function rowHtml(colspan, message) {
  return `<tr><td colspan="${colspan}">${escapeHtml(message)}</td></tr>`;
}

function cell(value) {
  const td = document.createElement("td");
  td.textContent = String(value || "-");
  return td;
}

function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function spanText(value) {
  const span = document.createElement("span");
  span.textContent = value;
  return span;
}

function uniqueSorted(values) {
  return Array.from(new Set(values.map((value) => String(value || "").trim()).filter(Boolean))).sort(
    (left, right) => left.localeCompare(right, "en"),
  );
}

function renderPath(template, strategyId) {
  return template.replace("{strategy_id}", encodeURIComponent(String(strategyId)));
}

function requiredElement(root, selector) {
  const element = qs(selector, root);
  if (!element) {
    throw new Error(`Strategy UI missing required element: ${selector}`);
  }
  return element;
}

function requiredData(element, key) {
  const value = element.dataset[key];
  if (!value) {
    throw new Error(`Strategy UI missing required data attribute: ${key}`);
  }
  return value;
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    };
    return replacements[char];
  });
}
