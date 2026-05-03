import { apiRequest } from "../core/api.js";
import { ready, setBusy } from "../core/dom.js";
import { translate } from "../core/locale.js";
import { notify } from "../core/notifications.js";

const PAGE_SELECTOR = "[data-backtests-run-page]";
const DEFAULT_INDICATOR_ID = "ma.dema";
const DEFAULT_SOURCE = "close";

ready(() => {
  const root = document.querySelector(PAGE_SELECTOR);
  if (!root) {
    return;
  }
  initRunPage(root);
});

function initRunPage(root) {
  const paths = {
    defaults: requireData(root, "apiDefaultsPath"),
    preflight: requireData(root, "apiPreflightPath"),
    jobs: requireData(root, "apiJobsPath"),
    markets: requireData(root, "apiMarketsPath"),
    instruments: requireData(root, "apiInstrumentsPath"),
    indicators: requireData(root, "apiIndicatorsPath"),
  };
  const nodes = collectNodes(root);
  if (Object.values(nodes).some((node) => node === null)) {
    return;
  }

  const state = {
    defaults: {},
    markets: [],
    indicators: [],
    pendingIdempotency: null,
    instrumentsController: null,
  };

  nodes.addIndicator.addEventListener("click", () => addIndicatorRow({ nodes, state }));
  nodes.loadSample.addEventListener("click", () => loadSample({ nodes, state }));
  nodes.preflight.addEventListener("click", async () => {
    await submitPreflight({ root, paths, nodes });
  });
  nodes.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await createJob({ root, paths, nodes, state });
  });
  nodes.riskMode.addEventListener("change", () => syncRiskControls(nodes));
  nodes.exchange.addEventListener("change", () => void refreshInstruments({ paths, nodes, state }));
  nodes.marketType.addEventListener("change", () => void refreshInstruments({ paths, nodes, state }));
  nodes.symbol.addEventListener("input", () => void refreshInstruments({ paths, nodes, state }));
  nodes.form.addEventListener("input", () => {
    state.pendingIdempotency = null;
  });
  window.addEventListener("beforeunload", () => {
    state.instrumentsController?.abort();
  });

  void initialize({ root, paths, nodes, state });
}

async function initialize({ root, paths, nodes, state }) {
  clearError(nodes);
  try {
    const [defaults, markets, indicators] = await Promise.allSettled([
      apiRequest(paths.defaults),
      apiRequest(paths.markets),
      apiRequest(paths.indicators),
    ]);
    if (defaults.status === "fulfilled") {
      state.defaults = asRecord(defaults.value);
      populateDefaults({ nodes, state });
      setStatus(nodes.defaultsStatus, translate("backtests.status.ready"), "success");
    } else {
      setStatus(nodes.defaultsStatus, translate("backtests.status.failed"), "danger");
      showError(nodes, defaults.reason);
    }
    if (markets.status === "fulfilled") {
      state.markets = Array.isArray(markets.value?.items) ? markets.value.items.map(asRecord) : [];
    }
    if (indicators.status === "fulfilled") {
      state.indicators = Array.isArray(indicators.value?.items)
        ? indicators.value.items.map(asRecord)
        : [];
    }
    renderReference({ nodes, state });
    if (nodes.indicatorRows.children.length === 0) {
      addIndicatorRow({ nodes, state });
    }
    loadSample({ nodes, state });
    await refreshInstruments({ paths, nodes, state });
  } catch (error) {
    showError(nodes, error);
  } finally {
    root.dataset.ready = "true";
  }
}

function collectNodes(root) {
  return {
    form: root.querySelector("[data-backtests-run-form]"),
    error: root.querySelector("[data-backtests-error]"),
    validation: root.querySelector("[data-backtests-validation]"),
    defaultsStatus: root.querySelector("[data-backtests-defaults-status]"),
    defaultsSummary: root.querySelector("[data-backtests-defaults-summary]"),
    referenceList: root.querySelector("[data-backtests-reference-list]"),
    preflightStatus: root.querySelector("[data-backtests-preflight-status]"),
    preflightSummary: root.querySelector("[data-backtests-preflight-summary]"),
    preflightJson: root.querySelector("[data-backtests-preflight-json]"),
    addIndicator: root.querySelector("[data-backtests-add-indicator]"),
    loadSample: root.querySelector("[data-backtests-load-sample]"),
    preflight: root.querySelector("[data-backtests-preflight]"),
    create: root.querySelector("[data-backtests-create]"),
    indicatorRows: root.querySelector("[data-backtests-indicator-rows]"),
    exchange: root.querySelector("#backtest-exchange"),
    marketType: root.querySelector("#backtest-market-type"),
    symbol: root.querySelector("#backtest-symbol"),
    symbolOptions: root.querySelector("#backtest-symbol-options"),
    timeframe: root.querySelector("#backtest-timeframe"),
    start: root.querySelector("#backtest-start"),
    end: root.querySelector("#backtest-end"),
    topN: root.querySelector("#backtest-top-n"),
    riskMode: root.querySelector("#backtest-risk-mode"),
    tpSlGrid: root.querySelector("[data-backtests-tp-sl-grid]"),
    tpStart: root.querySelector("#backtest-tp-start"),
    tpStop: root.querySelector("#backtest-tp-stop"),
    tpStep: root.querySelector("#backtest-tp-step"),
    slStart: root.querySelector("#backtest-sl-start"),
    slStop: root.querySelector("#backtest-sl-stop"),
    slStep: root.querySelector("#backtest-sl-step"),
    rankingMetric: root.querySelector("#backtest-ranking-metric"),
    rankingDirection: root.querySelector("#backtest-ranking-direction"),
    directionMode: root.querySelector("#backtest-direction-mode"),
    sizingMode: root.querySelector("#backtest-sizing-mode"),
    feeRate: root.querySelector("#backtest-fee-rate"),
    slippageRate: root.querySelector("#backtest-slippage-rate"),
    initialCash: root.querySelector("#backtest-initial-cash"),
    quoteAmount: root.querySelector("#backtest-quote-amount"),
    equityPct: root.querySelector("#backtest-equity-pct"),
    minQuote: root.querySelector("#backtest-min-quote"),
    maxQuote: root.querySelector("#backtest-max-quote"),
    profitLockEnabled: root.querySelector("#backtest-profit-lock-enabled"),
    profitLockPct: root.querySelector("#backtest-profit-lock-pct"),
    closeOnEnd: root.querySelector("#backtest-close-on-end"),
  };
}

function populateDefaults({ nodes, state }) {
  const defaults = asRecord(state.defaults);
  setOptions(nodes.timeframe, arrayOrEmpty(defaults.supported_timeframes), "15m");
  setOptions(nodes.riskMode, arrayOrEmpty(defaults.risk_modes), "none");
  setOptions(nodes.directionMode, arrayOrEmpty(defaults.direction_modes), "long_short_reversal");
  setOptions(nodes.sizingMode, arrayOrEmpty(defaults.sizing_modes), "fixed_equity_pct");
  setOptions(nodes.rankingMetric, arrayOrEmpty(defaults.ranking_metrics), "total_return_pct");

  const guardrails = asRecord(defaults.guardrails);
  if (Number(guardrails.max_top_n) > 0) {
    nodes.topN.max = String(guardrails.max_top_n);
  }
  if (Number(defaults.top_n_default) > 0) {
    nodes.topN.value = String(defaults.top_n_default);
  }

  const rankingDefault = asRecord(defaults.ranking_default);
  if (rankingDefault.primary_metric) {
    nodes.rankingMetric.value = selectFallback(nodes.rankingMetric, rankingDefault.primary_metric);
  }
  if (rankingDefault.direction) {
    nodes.rankingDirection.value = selectFallback(nodes.rankingDirection, rankingDefault.direction);
  }

  const execution = asRecord(defaults.execution_defaults);
  setNumber(nodes.feeRate, execution.fee_rate);
  setNumber(nodes.slippageRate, execution.slippage_rate);
  setNumber(nodes.initialCash, execution.initial_cash_quote);
  if (execution.direction_mode) {
    nodes.directionMode.value = selectFallback(nodes.directionMode, execution.direction_mode);
  }
  nodes.closeOnEnd.checked = Boolean(execution.close_on_end);

  const sizing = asRecord(execution.sizing);
  if (sizing.mode) {
    nodes.sizingMode.value = selectFallback(nodes.sizingMode, sizing.mode);
  }
  setNumber(nodes.quoteAmount, sizing.quote_amount);
  setNumber(nodes.equityPct, sizing.equity_pct);
  setNumber(nodes.minQuote, sizing.min_quote);
  setNumber(nodes.maxQuote, sizing.max_quote);

  const profitLock = asRecord(execution.profit_lock);
  nodes.profitLockEnabled.checked = Boolean(profitLock.enabled);
  setNumber(nodes.profitLockPct, profitLock.safe_profit_percent);
  syncIndicatorRows({ nodes, state });
  syncRiskControls(nodes);
  renderDefaultsSummary(nodes, defaults);
}

function addIndicatorRow({ nodes, state, values = {} }) {
  const row = document.createElement("div");
  row.className = "rh-backtests-indicator-row";
  row.append(
    label(translate("backtests.field.indicator")),
    indicatorSelect({ state, value: values.indicator_id }),
    label(translate("backtests.field.source")),
    sourceSelect({ state, indicatorId: values.indicator_id, value: values.source }),
    label(translate("backtests.field.window_start")),
    numberInput("window-start", values.window_start || "5"),
    label(translate("backtests.field.window_stop")),
    numberInput("window-stop", values.window_stop || "10"),
    label(translate("backtests.field.window_step")),
    numberInput("window-step", values.window_step || "1"),
    removeIndicatorButton(nodes),
  );
  nodes.indicatorRows.append(row);
  const indicator = row.querySelector('[data-role="indicator-id"]');
  const source = row.querySelector('[data-role="source"]');
  indicator.addEventListener("change", () => {
    syncSourceOptions({ state, indicator, source });
  });
}

function syncIndicatorRows({ nodes, state }) {
  nodes.indicatorRows.querySelectorAll(".rh-backtests-indicator-row").forEach((row) => {
    const indicator = row.querySelector('[data-role="indicator-id"]');
    const source = row.querySelector('[data-role="source"]');
    if (indicator instanceof HTMLSelectElement && source instanceof HTMLSelectElement) {
      const indicatorValue = indicator.value;
      setOptions(indicator, indicatorIds(state), DEFAULT_INDICATOR_ID);
      indicator.value = selectFallback(indicator, indicatorValue || DEFAULT_INDICATOR_ID);
      syncSourceOptions({ state, indicator, source });
    }
  });
}

function syncSourceOptions({ state, indicator, source }) {
  const sourceValue = source.value;
  setOptions(source, sourcesForIndicator(state, indicator.value), DEFAULT_SOURCE);
  source.value = selectFallback(source, sourceValue || DEFAULT_SOURCE);
}

function loadSample({ nodes, state }) {
  nodes.exchange.value = "binance";
  nodes.marketType.value = "spot";
  nodes.symbol.value = "BTCUSDT";
  nodes.timeframe.value = selectFallback(nodes.timeframe, "15m");
  nodes.start.value = "2020-01-11T20:08";
  nodes.end.value = "2026-04-11T20:08";
  nodes.topN.value = String(Number(asRecord(state.defaults).top_n_default || 100));
  nodes.riskMode.value = selectFallback(nodes.riskMode, "none");
  nodes.rankingMetric.value = selectFallback(nodes.rankingMetric, "total_return_pct");
  nodes.rankingDirection.value = selectFallback(nodes.rankingDirection, "desc");
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
  nodes.indicatorRows.replaceChildren();
  addIndicatorRow({
    nodes,
    state,
    values: {
      indicator_id: DEFAULT_INDICATOR_ID,
      source: DEFAULT_SOURCE,
      window_start: "5",
      window_stop: "10",
      window_step: "1",
    },
  });
  syncRiskControls(nodes);
  clearValidation(nodes);
}

async function submitPreflight({ root, paths, nodes }) {
  clearError(nodes);
  clearValidation(nodes);
  setStatus(nodes.preflightStatus, translate("backtests.status.loading"), "warning");
  setBusy(nodes.preflight, true);
  try {
    const payload = buildRequestPayload(nodes);
    const result = await apiRequest(paths.preflight, { method: "POST", body: payload });
    renderPreflight(nodes, result);
    setStatus(nodes.preflightStatus, translate("backtests.status.valid"), "success");
    root.dataset.lastPreflightHash = String(result?.request_hash || "");
    return result;
  } catch (error) {
    renderValidation(nodes, error);
    setStatus(nodes.preflightStatus, translate("backtests.status.failed"), "danger");
    return null;
  } finally {
    setBusy(nodes.preflight, false);
  }
}

async function createJob({ root, paths, nodes, state }) {
  clearError(nodes);
  clearValidation(nodes);
  setBusy(nodes.create, true);
  try {
    const payload = buildRequestPayload(nodes);
    const preflight = await apiRequest(paths.preflight, { method: "POST", body: payload });
    renderPreflight(nodes, preflight);
    setStatus(nodes.preflightStatus, translate("backtests.status.valid"), "success");
    const signature = JSON.stringify(payload);
    const idempotencyKey = idempotencyKeyForSignature(state, signature);
    const job = await apiRequest(paths.jobs, {
      method: "POST",
      body: payload,
      headers: { "Idempotency-Key": idempotencyKey },
      timeoutMs: 60000,
    });
    state.pendingIdempotency = null;
    root.dataset.createdJobId = String(job?.job_id || "");
    notify(translate("backtests.run.created"), { tone: "info" });
    renderCreatedJob(nodes, job);
  } catch (error) {
    renderValidation(nodes, error);
  } finally {
    setBusy(nodes.create, false);
  }
}

function buildRequestPayload(nodes) {
  const riskMode = textValue(nodes.riskMode, "risk.mode");
  return {
    coordinates: {
      exchange: textValue(nodes.exchange, "coordinates.exchange"),
      market_type: textValue(nodes.marketType, "coordinates.market_type"),
      symbol: textValue(nodes.symbol, "coordinates.symbol").toUpperCase(),
    },
    timeframe: textValue(nodes.timeframe, "timeframe"),
    time_range: {
      start: datetimeLocalToUtc(nodes.start.value, "time_range.start"),
      end: datetimeLocalToUtc(nodes.end.value, "time_range.end"),
    },
    indicators: indicatorPayloads(nodes),
    risk: riskPayload(nodes, riskMode),
    execution: {
      direction_mode: textValue(nodes.directionMode, "execution.direction_mode"),
      fee_rate: numberValue(nodes.feeRate, "execution.fee_rate"),
      slippage_rate: numberValue(nodes.slippageRate, "execution.slippage_rate"),
      initial_cash_quote: numberValue(nodes.initialCash, "execution.initial_cash_quote"),
      sizing: sizingPayload(nodes),
      profit_lock: profitLockPayload(nodes),
      close_on_end: nodes.closeOnEnd.checked,
    },
    ranking: {
      primary_metric: textValue(nodes.rankingMetric, "ranking.primary_metric"),
      direction: textValue(nodes.rankingDirection, "ranking.direction"),
    },
    top_n: positiveIntValue(nodes.topN, "top_n"),
  };
}

function indicatorPayloads(nodes) {
  const rows = Array.from(nodes.indicatorRows.querySelectorAll(".rh-backtests-indicator-row"));
  if (rows.length === 0) {
    throw new Error(translate("backtests.validation.indicator_required"));
  }
  return rows.map((row, index) => {
    const indicator = row.querySelector('[data-role="indicator-id"]');
    const source = row.querySelector('[data-role="source"]');
    const start = row.querySelector('[data-role="window-start"]');
    const stop = row.querySelector('[data-role="window-stop"]');
    const step = row.querySelector('[data-role="window-step"]');
    return {
      indicator_id: textValue(indicator, `indicators.${index}.indicator_id`),
      sources: [textValue(source, `indicators.${index}.source`)],
      window: {
        start: positiveIntValue(start, `indicators.${index}.window.start`),
        stop: positiveIntValue(stop, `indicators.${index}.window.stop`),
        step: positiveIntValue(step, `indicators.${index}.window.step`),
      },
    };
  });
}

function riskPayload(nodes, riskMode) {
  if (riskMode !== "tp_sl_grid") {
    return { mode: "none" };
  }
  return {
    mode: "tp_sl_grid",
    tp: {
      start_pct: numberValue(nodes.tpStart, "risk.tp.start_pct"),
      stop_pct: numberValue(nodes.tpStop, "risk.tp.stop_pct"),
      step_pct: numberValue(nodes.tpStep, "risk.tp.step_pct"),
    },
    sl: {
      start_pct: numberValue(nodes.slStart, "risk.sl.start_pct"),
      stop_pct: numberValue(nodes.slStop, "risk.sl.stop_pct"),
      step_pct: numberValue(nodes.slStep, "risk.sl.step_pct"),
    },
  };
}

function sizingPayload(nodes) {
  const mode = textValue(nodes.sizingMode, "execution.sizing.mode");
  const payload = { mode };
  if (mode === "fixed_quote") {
    payload.quote_amount = numberValue(nodes.quoteAmount, "execution.sizing.quote_amount");
  }
  if (
    mode === "fixed_equity_pct"
    || mode === "fixed_equity_pct_min_quote"
    || mode === "fixed_equity_pct_max_quote"
  ) {
    payload.equity_pct = numberValue(nodes.equityPct, "execution.sizing.equity_pct");
  }
  if (mode === "fixed_equity_pct_min_quote") {
    payload.min_quote = numberValue(nodes.minQuote, "execution.sizing.min_quote");
  }
  if (mode === "fixed_equity_pct_max_quote") {
    payload.max_quote = numberValue(nodes.maxQuote, "execution.sizing.max_quote");
  }
  return payload;
}

function profitLockPayload(nodes) {
  if (!nodes.profitLockEnabled.checked) {
    return { enabled: false };
  }
  return {
    enabled: true,
    safe_profit_percent: numberValue(nodes.profitLockPct, "execution.profit_lock.safe_profit_percent"),
  };
}

function renderPreflight(nodes, result) {
  const payload = asRecord(result);
  const cost = asRecord(payload.cost_estimate);
  renderKeyValues(nodes.preflightSummary, [
    ["request_hash", compactHash(payload.request_hash)],
    ["result_config_hash", compactHash(payload.result_config_hash)],
    ["cost_class", cost.cost_class || "-"],
    ["candidate_combinations", cost.candidate_combinations ?? "-"],
    ["indicator_rows", cost.indicator_rows ?? "-"],
    ["tp_sl_cells", cost.tp_sl_cells ?? "-"],
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

function renderCreatedJob(nodes, job) {
  const payload = asRecord(job);
  renderKeyValues(nodes.preflightSummary, [
    ["job_id", payload.job_id || "-"],
    ["state", payload.state || "-"],
    ["request_hash", compactHash(payload.request_hash)],
    ["result_config_hash", compactHash(payload.result_config_hash)],
    ["idempotent_replay", String(Boolean(payload.idempotent_replay))],
    ["next", `/backtests/${payload.job_id || ""}`],
  ]);
}

function renderDefaultsSummary(nodes, defaults) {
  const guardrails = asRecord(defaults.guardrails);
  renderKeyValues(nodes.defaultsSummary, [
    ["timeframes", arrayOrEmpty(defaults.supported_timeframes).join(", ")],
    ["risk_modes", arrayOrEmpty(defaults.risk_modes).join(", ")],
    ["top_n_default", defaults.top_n_default || "-"],
    ["max_top_n", guardrails.max_top_n || "-"],
    ["hit_times", compactRecord(asRecord(defaults.hit_times_grid))],
  ]);
}

function renderReference({ nodes, state }) {
  nodes.referenceList.replaceChildren();
  const marketLine = document.createElement("div");
  marketLine.append(
    labelText(translate("backtests.run.markets")),
    codeText(String(state.markets.length)),
  );
  const indicatorLine = document.createElement("div");
  indicatorLine.append(
    labelText(translate("backtests.run.indicator_defs")),
    codeText(String(state.indicators.length || indicatorIds(state).length)),
  );
  const presetLine = document.createElement("div");
  presetLine.append(
    labelText("backtest_presets"),
    codeText(translate("backtests.run.presets_not_enabled")),
  );
  nodes.referenceList.append(marketLine, indicatorLine, presetLine);
}

async function refreshInstruments({ paths, nodes, state }) {
  const market = matchingMarket(nodes, state.markets);
  if (!market?.market_id) {
    return;
  }
  state.instrumentsController?.abort();
  state.instrumentsController = new AbortController();
  const params = new URLSearchParams({
    market_id: String(market.market_id),
    q: nodes.symbol.value || "BTC",
    limit: "50",
  });
  try {
    const payload = await apiRequest(`${paths.instruments}?${params.toString()}`, {
      signal: state.instrumentsController.signal,
    });
    const symbols = Array.isArray(payload?.items)
      ? payload.items.map((item) => asRecord(item).symbol).filter(Boolean)
      : [];
    nodes.symbolOptions.replaceChildren(
      ...symbols.map((symbol) => {
        const option = document.createElement("option");
        option.value = String(symbol);
        return option;
      }),
    );
  } catch (_error) {
    nodes.symbolOptions.replaceChildren();
  }
}

function matchingMarket(nodes, markets) {
  const exchange = nodes.exchange.value.trim().toLowerCase();
  const marketType = nodes.marketType.value.trim().toLowerCase();
  return markets.find((market) => (
    String(market.exchange_name || "").toLowerCase() === exchange
    && String(market.market_type || "").toLowerCase() === marketType
  ));
}

function renderValidation(nodes, error) {
  const items = validationItems(error);
  if (items.length === 0) {
    showError(nodes, error);
    return;
  }
  nodes.validation.replaceChildren();
  const list = document.createElement("ul");
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.append(li);
  });
  nodes.validation.append(list);
  nodes.validation.hidden = false;
}

function validationItems(error) {
  const details = error?.payload?.error?.details;
  const items = Array.isArray(details?.errors) ? details.errors : [];
  if (items.length > 0) {
    return items.map((item) => {
      const path = item?.path ? `${item.path}: ` : "";
      return `${path}${item?.message || item?.code || translate("js.error.validation")}`;
    });
  }
  if (error instanceof Error) {
    return [error.message];
  }
  return [];
}

function showError(nodes, error) {
  nodes.error.hidden = false;
  nodes.error.textContent = error?.message || translate("js.error.network");
}

function clearError(nodes) {
  nodes.error.hidden = true;
  nodes.error.textContent = "";
}

function clearValidation(nodes) {
  nodes.validation.hidden = true;
  nodes.validation.replaceChildren();
}

function setStatus(node, label, tone) {
  node.textContent = label;
  node.className = `rh-status-badge rh-status-badge--${tone || "neutral"}`;
}

function renderKeyValues(container, items) {
  container.replaceChildren();
  items.forEach(([key, value]) => {
    const term = document.createElement("dt");
    term.textContent = key;
    const detail = document.createElement("dd");
    detail.textContent = String(value ?? "-");
    container.append(term, detail);
  });
}

function syncRiskControls(nodes) {
  nodes.tpSlGrid.hidden = nodes.riskMode.value !== "tp_sl_grid";
}

function idempotencyKeyForSignature(state, signature) {
  if (state.pendingIdempotency?.signature === signature) {
    return state.pendingIdempotency.key;
  }
  const key = globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  state.pendingIdempotency = { signature, key };
  return key;
}

function indicatorSelect({ state, value }) {
  const select = document.createElement("select");
  select.dataset.role = "indicator-id";
  select.setAttribute("aria-label", translate("backtests.field.indicator"));
  setOptions(select, indicatorIds(state), DEFAULT_INDICATOR_ID);
  select.value = selectFallback(select, value || DEFAULT_INDICATOR_ID);
  return select;
}

function sourceSelect({ state, indicatorId, value }) {
  const select = document.createElement("select");
  select.dataset.role = "source";
  select.setAttribute("aria-label", translate("backtests.field.source"));
  setOptions(select, sourcesForIndicator(state, indicatorId || DEFAULT_INDICATOR_ID), DEFAULT_SOURCE);
  select.value = selectFallback(select, value || DEFAULT_SOURCE);
  return select;
}

function removeIndicatorButton(nodes) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rh-button rh-button--ghost";
  button.textContent = translate("backtests.action.remove");
  button.addEventListener("click", (event) => {
    const row = event.currentTarget.closest(".rh-backtests-indicator-row");
    if (nodes.indicatorRows.children.length > 1) {
      row?.remove();
    }
  });
  return button;
}

function numberInput(role, value) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "1";
  input.value = String(value);
  input.dataset.role = role;
  return input;
}

function label(text) {
  const node = document.createElement("label");
  node.textContent = text;
  return node;
}

function labelText(text) {
  const node = document.createElement("span");
  node.className = "rh-backtests-muted";
  node.textContent = text;
  return node;
}

function codeText(text) {
  const node = document.createElement("code");
  node.textContent = text;
  return node;
}

function setOptions(select, values, fallback) {
  const previous = select.value;
  const options = values.length > 0 ? values.map(String) : [fallback];
  select.replaceChildren(
    ...options.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      return option;
    }),
  );
  select.value = selectFallback(select, previous || fallback);
}

function selectFallback(select, preferred) {
  const options = Array.from(select.options).map((option) => option.value);
  return options.includes(String(preferred)) ? String(preferred) : options[0] || "";
}

function setNumber(input, value) {
  if (value !== null && value !== undefined && Number.isFinite(Number(value))) {
    input.value = String(value);
  }
}

function indicatorIds(state) {
  return arrayOrEmpty(asRecord(state.defaults).supported_indicator_ids);
}

function sourcesForIndicator(state, indicatorId) {
  const sourceMap = asRecord(asRecord(state.defaults).indicator_sources);
  return arrayOrEmpty(sourceMap[indicatorId]);
}

function textValue(node, field) {
  if (!node || typeof node.value !== "string" || !node.value.trim()) {
    throw new Error(translate("backtests.validation.required", { field }));
  }
  return node.value.trim();
}

function numberValue(node, field) {
  const raw = textValue(node, field);
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(translate("backtests.validation.number", { field }));
  }
  return value;
}

function positiveIntValue(node, field) {
  const value = numberValue(node, field);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(translate("backtests.validation.positive_int", { field }));
  }
  return value;
}

function datetimeLocalToUtc(value, field) {
  if (!value) {
    throw new Error(translate("backtests.validation.required", { field }));
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    throw new Error(translate("backtests.validation.datetime", { field }));
  }
  return date.toISOString();
}

function compactHash(value) {
  const text = String(value || "");
  return text.length <= 16 ? text : `${text.slice(0, 10)}...${text.slice(-6)}`;
}

function compactRecord(record) {
  return Object.entries(record)
    .slice(0, 6)
    .map(([key, value]) => `${key}=${compactValue(value)}`)
    .join(", ");
}

function compactValue(value) {
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function arrayOrEmpty(value) {
  return Array.isArray(value) ? value : [];
}

function requireData(node, name) {
  const value = node.dataset[name];
  if (!value) {
    throw new Error(`Missing data attribute: ${name}`);
  }
  return value;
}
