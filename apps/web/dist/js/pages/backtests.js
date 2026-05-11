import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { t } from "../core/locale.js";
import { createPoller } from "../core/poller.js";

const DEFAULT_ENDPOINT = "/api/ui/backtests/workstation";
const DEFAULT_VARIANT_OPEN_DELAY_MS = 140;
const DEFAULT_VARIANT_OPEN_DURATION_MS = 400;
const DEFAULT_VARIANT_PREVIEW_LIMIT = 5;
const REFRESH_PRESETS = {
  off: 0,
  "10s": 10000,
  "15s": 15000,
  "30s": 30000,
  "1m": 60000,
  "5m": 300000,
};

const state = {
  market: "binance",
  market_type: "spot",
  symbol: "BTCUSDT",
  timeframe: "15m",
  direction: "long_short_reversal",
  risk_mode: "none",
  sizing_mode: "fixed_equity_pct",
  ranking_metric: "total_return_pct",
  ranking_order: "desc",
  job_state: "",
  job_exchange: "",
  job_market_type: "",
  job_symbol: "",
  launched_from: "",
  launched_to: "",
  cursor: null,
  nextCursor: null,
  query: "",
  runtimeDefaults: null,
  selectedSymbols: new Set(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
  selectedIndicators: [],
  indicatorCatalog: new Map(),
  indicatorFamily: null,
  jobRows: [],
  loadedAllJobs: false,
  selectedJobId: null,
  selectedVariantKey: null,
  resultSummary: null,
  animateVariantJobId: null,
};

let activeRequest = null;
let activeResultRequest = null;
let poller = null;
let manualRefreshRetrySeconds = 0;
let delayedVariantOpen = null;
let variantAnimationFrame = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function trashIcon(label) {
  return `
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M3 6h18"></path>
      <path d="M8 6V4h8v2"></path>
      <path d="M6 6l1 15h10l1-15"></path>
      <path d="M10 11v6"></path>
      <path d="M14 11v6"></path>
    </svg>
    <span class="backtests-visually-hidden">${escapeHtml(label)}</span>
  `;
}

function localTime(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return "--";
  }
  return new Intl.DateTimeFormat(document.documentElement.lang || "en", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function dateToIso(value, fallback) {
  if (!value) {
    return fallback;
  }
  return `${value}T00:00:00Z`;
}

function percent(value, fractionDigits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${number.toFixed(fractionDigits)}%`;
}

function signedDrawdownPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  const signed = number === 0 ? 0 : -Math.abs(number);
  return percent(signed, 2);
}

function decimalOrDash(value, fractionDigits) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return number.toFixed(fractionDigits);
}

function integerOrDash(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  const rounded = Math.trunc(number);
  const sign = rounded < 0 ? "-" : "";
  return `${sign}${String(Math.abs(rounded)).replace(/\B(?=(\d{3})+(?!\d))/g, " ")}`;
}

function compactMagnitude(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  const sign = number < 0 ? "-" : "";
  const abs = Math.abs(number);
  const units = [
    [1_000_000_000_000, "t"],
    [1_000_000_000, "b"],
    [1_000_000, "m"],
    [1_000, "k"],
  ];
  const unit = units.find(([limit]) => abs >= limit);
  if (!unit) {
    return `${sign}${Math.trunc(abs)}`;
  }
  const scaled = abs / unit[0];
  const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  return `${sign}${scaled.toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*[1-9])0+$/, "$1")}${unit[1]}`;
}

function numberOrDash(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}

function labelForId(value) {
  return String(value || "")
    .replaceAll(".", " ")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function sizingModeLabel(value) {
  const labels = {
    all_in: t("backtests.option.all_in"),
    fixed_quote: t("backtests.option.fixed_quote"),
    fixed_equity_pct: t("backtests.option.fixed_equity_pct"),
    fixed_equity_pct_min_quote: t("backtests.option.fixed_equity_pct_min_quote"),
    fixed_equity_pct_max_quote: t("backtests.option.fixed_equity_pct_max_quote"),
  };
  return labels[value] || labelForId(value);
}

function primaryWindowSpec(catalogItem) {
  return catalogItem?.param_specs?.params?.window || {};
}

function specMin(spec) {
  if (spec?.mode === "explicit") {
    const values = (spec.values || []).map(Number).filter(Number.isFinite);
    return values.length ? Math.min(...values) : 1;
  }
  return rangeDefault(spec, "start", 1);
}

function specMax(spec) {
  if (spec?.mode === "explicit") {
    const values = (spec.values || []).map(Number).filter(Number.isFinite);
    return values.length ? Math.max(...values) : 999;
  }
  return rangeDefault(spec, "stop_incl", 999);
}

function rangeDefault(spec, key, fallback) {
  const value = spec?.[key];
  return value === null || value === undefined ? fallback : value;
}

function countRange(start, stop, step) {
  const from = Number(start);
  const to = Number(stop);
  const stride = Number(step);
  if (!Number.isFinite(from) || !Number.isFinite(to) || !Number.isFinite(stride) || stride <= 0 || from > to) {
    return 0;
  }
  return Math.floor((to - from) / stride) + 1;
}

function formatDate(value) {
  return value ? String(value).slice(0, 10) : "--";
}

function formatDateTimeMinute(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return formatDate(value);
  }
  const pad = (part) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function compactId(value) {
  return value ? String(value).slice(0, 8) : "--";
}

function endpointFromTemplate(template, replacements) {
  let endpoint = template;
  Object.entries(replacements).forEach(([key, value]) => {
    endpoint = endpoint.replaceAll(`{${key}}`, encodeURIComponent(String(value)));
  });
  return endpoint;
}

function formatFieldErrors(errors) {
  if (!errors) {
    return "";
  }
  if (Array.isArray(errors)) {
    return errors
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        const path = Array.isArray(item?.loc)
          ? item.loc.join(".")
          : item?.field || item?.path || item?.name || "";
        const message = item?.msg || item?.message || item?.reason || item?.code || "";
        return [path, message].filter(Boolean).join(": ");
      })
      .filter(Boolean)
      .join("; ");
  }
  if (typeof errors === "object") {
    return Object.entries(errors)
      .map(([field, message]) => `${field}: ${Array.isArray(message) ? message.join(", ") : String(message)}`)
      .join("; ");
  }
  return String(errors);
}

function describeApiError(error) {
  const envelope = error?.payload?.error;
  const detail = error?.payload?.detail;
  const parts = [];
  if (envelope && typeof envelope === "object") {
    if (typeof envelope.message === "string") {
      parts.push(envelope.message);
    }
    const envelopeErrors = formatFieldErrors(envelope.details?.field_errors || envelope.details?.errors);
    if (envelopeErrors) {
      parts.push(envelopeErrors);
    } else if (envelope.details?.reason) {
      parts.push(String(envelope.details.reason));
    } else if (envelope.code) {
      parts.push(String(envelope.code));
    }
  }
  if (typeof detail === "string") {
    parts.push(detail);
  } else if (detail && typeof detail === "object") {
    if (typeof detail.message === "string") {
      parts.push(detail.message);
    }
    if (typeof detail.detail === "string") {
      parts.push(detail.detail);
    }
    const detailErrors = formatFieldErrors(detail.field_errors || detail.errors);
    if (detailErrors) {
      parts.push(detailErrors);
    }
  }
  const payloadErrors = formatFieldErrors(error?.payload?.field_errors || error?.payload?.errors);
  if (payloadErrors) {
    parts.push(payloadErrors);
  }
  if (!parts.length && typeof error?.message === "string" && error.message !== "[object Object]") {
    parts.push(error.message);
  }
  const body = parts.filter(Boolean).join("; ") || t("backtests.status.failed");
  return error?.status ? `HTTP ${error.status}: ${body}` : body;
}

function variantBaseEndpoint(root, jobId, variantKey) {
  const template =
    root.dataset.variantEndpointTemplate || "/api/backtests/jobs/{job_id}/variants/{variant_key}";
  return endpointFromTemplate(template, { job_id: jobId, variant_key: variantKey });
}

function financialClass(value) {
  const number = Number(value);
  if (number > 0) {
    return "rh-financial--positive";
  }
  if (number < 0) {
    return "rh-financial--negative";
  }
  return "rh-financial--neutral";
}

function selectedSymbols(root) {
  const selected = Array.from(state.selectedSymbols || []);
  return selected.length ? selected.slice(0, 1) : [state.symbol || "BTCUSDT"];
}

function buildRequestPayload(root) {
  const start = qs("[data-config-field='start']", root)?.value || "2023-01-01";
  const end = qs("[data-config-field='end']", root)?.value || "2024-01-01";
  const capital = Number(qs("[data-config-field='capital']", root)?.value || 10000);
  const feePercent = Number(qs("[data-config-field='fee']", root)?.value || 0.075);
  const slippagePercent = Number(qs("[data-config-field='slippage']", root)?.value || 0.01);
  const sizing = buildSizingPayload(root);
  const indicators = state.selectedIndicators.length
    ? state.selectedIndicators.map((indicator) => ({
        indicator_id: indicator.indicator_id,
        sources: indicator.sources.length ? indicator.sources : undefined,
        window: {
          start: Number(indicator.window.start),
          stop: Number(indicator.window.stop),
          step: Number(indicator.window.step),
        },
      }))
    : (state.runtimeDefaults?.config_draft?.indicators || [
        {
          indicator_id: "ma.dema",
          sources: ["close"],
          window: { start: 5, stop: 30, step: 2 },
        },
      ]).slice(0, 4);
  const risk = { mode: state.risk_mode };
  if (state.risk_mode === "tp_sl_grid") {
    risk.tp = {
      start_pct: Number(qs("[data-risk-field='tp_start']", root)?.value || 0.5),
      stop_pct: Number(qs("[data-risk-field='tp_stop']", root)?.value || 1),
      step_pct: Number(qs("[data-risk-field='tp_step']", root)?.value || 0.5),
    };
    risk.sl = {
      start_pct: Number(qs("[data-risk-field='sl_start']", root)?.value || 0.5),
      stop_pct: Number(qs("[data-risk-field='sl_stop']", root)?.value || 1),
      step_pct: Number(qs("[data-risk-field='sl_step']", root)?.value || 0.5),
    };
  }

  return {
    coordinates: {
      exchange: state.market,
      market_type: state.market_type,
      symbol: selectedSymbols(root)[0] || state.symbol || "BTCUSDT",
    },
    timeframe: state.timeframe,
    time_range: {
      start: dateToIso(start, "2023-01-01T00:00:00Z"),
      end: dateToIso(end, "2024-01-01T00:00:00Z"),
    },
    indicators,
    risk,
    execution: {
      direction_mode: state.direction,
      fee_rate: feePercent / 100,
      slippage_rate: slippagePercent / 100,
      initial_cash_quote: capital,
      sizing,
      profit_lock: { enabled: false },
      close_on_end: true,
    },
    ranking: {
      primary_metric: state.ranking_metric,
      direction: state.ranking_order,
    },
    top_n: Number(state.runtimeDefaults?.runtime_defaults?.top_n_default || 100),
  };
}

function positiveSizingNumber(root, name, fallback) {
  const value = Number(qs(`[data-sizing-field='${name}']`, root)?.value || fallback);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function buildSizingPayload(root) {
  const mode = state.sizing_mode || "fixed_equity_pct";
  if (mode === "all_in") {
    return { mode };
  }
  if (mode === "fixed_quote") {
    return {
      mode,
      quote_amount: positiveSizingNumber(root, "quote_amount", 1000),
    };
  }
  const equityPct = positiveSizingNumber(root, "equity_pct", 10);
  if (mode === "fixed_equity_pct_min_quote") {
    return {
      mode,
      equity_pct: equityPct,
      min_quote: positiveSizingNumber(root, "min_quote", 100),
    };
  }
  if (mode === "fixed_equity_pct_max_quote") {
    return {
      mode,
      equity_pct: equityPct,
      max_quote: positiveSizingNumber(root, "max_quote", 1000),
    };
  }
  return {
    mode: "fixed_equity_pct",
    equity_pct: equityPct,
  };
}

function updateOptionSelection(root, name, value, label) {
  state[name] = value;
  qsa(`[data-backtest-option='${name}']`, root).forEach((option) => {
    option.setAttribute("aria-selected", option.dataset.value === value ? "true" : "false");
  });
  const current = qs(`[data-current-value='${name}']`, root);
  if (current) {
    current.textContent = label || value || t("backtests.results.all");
  }
  if (["job_state", "job_exchange", "job_market_type"].includes(name)) {
    refreshWorkstation(root, "manual").catch(() => {});
  }
  if (name === "risk_mode") {
    updateRiskPanel(root);
  }
  if (name === "sizing_mode") {
    updateSizingPanel(root);
  }
}

function renderDropdownOptions(root, name, options) {
  const menu = qs(`#backtest-${name}-menu`, root);
  if (!menu || !Array.isArray(options) || !options.length) {
    return;
  }
  menu.innerHTML = options
    .map((option, index) => `
      <button
        class="rh-menu-item"
        type="button"
        role="option"
        aria-selected="${index === 0 ? "true" : "false"}"
        data-backtest-option="${escapeHtml(name)}"
        data-value="${escapeHtml(option.value)}"
      >${escapeHtml(option.label || option.value)}</button>
    `)
    .join("");
  if (!options.some((option) => option.value === state[name])) {
    updateOptionSelection(root, name, options[0].value, options[0].label || options[0].value);
  }
}

function renderRuntimeControls(root, data) {
  const runtime = data?.runtime_defaults || {};
  const universe = data?.instrument_universe || {};
  renderDropdownOptions(root, "market", universe.markets || []);
  renderDropdownOptions(root, "market_type", universe.market_types || []);
  renderDropdownOptions(root, "timeframe", (universe.timeframes || runtime.supported_timeframes || []).map((item) => (
    typeof item === "string" ? { value: item, label: item } : item
  )));
  renderDropdownOptions(root, "direction", (runtime.direction_modes || []).map((value) => ({
    value,
    label: value === "long_only" ? t("backtests.option.long_only") : t("backtests.option.long_short"),
  })));
  renderDropdownOptions(root, "risk_mode", (runtime.risk_modes || []).map((value) => ({
    value,
    label: value === "tp_sl_grid" ? t("backtests.option.tp_sl_grid") : t("backtests.option.no_risk"),
  })));
  renderDropdownOptions(root, "sizing_mode", (runtime.sizing_modes || []).map((value) => ({
    value,
    label: sizingModeLabel(value),
  })));
  renderDropdownOptions(root, "ranking_metric", (runtime.ranking_metrics || []).map((value) => ({
    value,
    label: labelForId(value),
  })));
  seedRiskPanel(root, runtime.hit_times_grid || {});
  seedConfigDraft(root, data?.config_draft || {});
  updateRiskPanel(root);
  updateSizingPanel(root);
}

function seedConfigDraft(root, draft) {
  const sizing = draft?.execution?.sizing || {};
  const fieldValues = {
    symbol: draft?.coordinates?.symbol,
    start: String(draft?.time_range?.start || "").slice(0, 10),
    end: String(draft?.time_range?.end || "").slice(0, 10),
    capital: draft?.execution?.initial_cash_quote,
    fee: Number(draft?.execution?.fee_rate || 0) * 100,
    slippage: Number(draft?.execution?.slippage_rate || 0) * 100,
  };
  Object.entries(fieldValues).forEach(([name, value]) => {
    const field = qs(`[data-config-field='${name}']`, root);
    if (field && value !== undefined && value !== null && value !== "") {
      field.value = String(value);
    }
  });
  const sizingValues = {
    quote_amount: sizing.quote_amount ?? 1000,
    equity_pct: sizing.equity_pct ?? 10,
    min_quote: sizing.min_quote ?? 100,
    max_quote: sizing.max_quote ?? 1000,
  };
  Object.entries(sizingValues).forEach(([name, value]) => {
    const field = qs(`[data-sizing-field='${name}']`, root);
    if (field && value !== undefined && value !== null && value !== "") {
      field.value = String(value);
    }
  });
  if (sizing?.mode) {
    updateOptionSelection(root, "sizing_mode", sizing.mode, sizingModeLabel(sizing.mode));
  }
  if (draft?.timeframe) {
    updateOptionSelection(root, "timeframe", draft.timeframe, draft.timeframe);
  }
  if (draft?.risk?.mode) {
    updateOptionSelection(
      root,
      "risk_mode",
      draft.risk.mode,
      draft.risk.mode === "tp_sl_grid" ? t("backtests.option.tp_sl_grid") : t("backtests.option.no_risk")
    );
  }
}

function seedRiskPanel(root, grid) {
  const tp = grid.tp_levels_pct || [];
  const sl = grid.sl_levels_pct || [];
  const defaults = {
    tp_start: tp[0] ?? 0.5,
    tp_stop: tp[Math.min(1, Math.max(0, tp.length - 1))] ?? tp[0] ?? 1,
    tp_step: tp.length > 1 ? Number(tp[1]) - Number(tp[0]) : tp[0] ?? 0.5,
    sl_start: sl[0] ?? 0.5,
    sl_stop: sl[Math.min(1, Math.max(0, sl.length - 1))] ?? sl[0] ?? 1,
    sl_step: sl.length > 1 ? Number(sl[1]) - Number(sl[0]) : sl[0] ?? 0.5,
  };
  Object.entries(defaults).forEach(([key, value]) => {
    const field = qs(`[data-risk-field='${key}']`, root);
    if (field && !field.value) {
      field.value = String(value);
    }
  });
}

function updateRiskPanel(root) {
  const riskGrid = qs("[data-risk-grid]", root);
  if (riskGrid) {
    riskGrid.hidden = state.risk_mode !== "tp_sl_grid";
  }
}

function updateSizingPanel(root) {
  const mode = state.sizing_mode || "fixed_equity_pct";
  const visibleFields = new Set();
  if (mode === "fixed_quote") {
    visibleFields.add("quote_amount");
  } else if (mode === "fixed_equity_pct") {
    visibleFields.add("equity_pct");
  } else if (mode === "fixed_equity_pct_min_quote") {
    visibleFields.add("equity_pct");
    visibleFields.add("min_quote");
  } else if (mode === "fixed_equity_pct_max_quote") {
    visibleFields.add("equity_pct");
    visibleFields.add("max_quote");
  }
  qsa("[data-sizing-field-row]", root).forEach((row) => {
    row.hidden = !visibleFields.has(row.dataset.sizingFieldRow || "");
  });
  const boundsRow = qs("[data-sizing-bounds-row]", root);
  if (boundsRow) {
    boundsRow.hidden = !visibleFields.has("min_quote") && !visibleFields.has("max_quote");
  }
}

function renderSymbols(root, universe) {
  const target = qs("[data-symbol-list]", root);
  const selectedTarget = qs("[data-selected-symbols]", root);
  const symbols = universe?.symbols || [];
  const firstUniverseSymbol = symbols[0]?.value || "BTCUSDT";
  const requestedSelected =
    Array.from(state.selectedSymbols || [])[0] ||
    universe?.selected_symbols?.[0] ||
    state.symbol ||
    firstUniverseSymbol;
  const availableValues = new Set(symbols.map((symbol) => symbol.value));
  const selectedValue = availableValues.has(requestedSelected) ? requestedSelected : firstUniverseSymbol;
  const selected = new Set([selectedValue]);
  state.symbol = selectedValue;
  state.selectedSymbols = selected;
  const symbolField = qs("[data-config-field='symbol']", root);
  if (symbolField instanceof HTMLInputElement) {
    symbolField.value = selectedValue;
  }
  if (target) {
    target.innerHTML = symbols
      .map((symbol) => `
        <button class="backtests-symbol-row ${selected.has(symbol.value) ? "is-selected" : ""}" type="button" data-symbol-row data-symbol-select="${escapeHtml(symbol.value)}" data-symbol-label="${escapeHtml(symbol.label)}" aria-pressed="${selected.has(symbol.value) ? "true" : "false"}">
          <span>${escapeHtml(symbol.label)}</span>
          <small>${escapeHtml(symbol.status)}</small>
        </button>
      `)
      .join("");
  }
  renderSelectedSymbols(root);
}

function renderSelectedSymbols(root) {
  const selectedTarget = qs("[data-selected-symbols]", root);
  const selected = selectedSymbols(root);
  if (selectedTarget) {
    selectedTarget.innerHTML = selected
      .map((symbol) => `<span class="backtests-chip">${escapeHtml(symbol)}</span>`)
      .join("");
  }
  setText("[data-symbol-count]", t("backtests.instruments.count", { count: selected.length }), root);
}

function filterSymbols(root, query) {
  const normalized = String(query || "").trim().toLowerCase();
  qsa("[data-symbol-row]", root).forEach((row) => {
    const label = `${row.dataset.symbolLabel || ""} ${row.textContent || ""}`.toLowerCase();
    row.hidden = normalized.length > 0 && !label.includes(normalized);
  });
}

function renderIndicators(root, catalog) {
  const items = catalog?.items || Array.from(state.indicatorCatalog.values());
  state.indicatorCatalog = new Map(items.map((item) => [item.indicator_id, item]));
  if (!state.selectedIndicators.length) {
    state.selectedIndicators = (state.runtimeDefaults?.config_draft?.indicators || [])
      .map((indicator) => indicatorStateFromDraft(indicator))
      .filter(Boolean);
  }
  renderIndicatorAddMenu(root, items);
  const target = qs("[data-indicator-rows]", root);
  if (target) {
    const rows = state.selectedIndicators;
    target.innerHTML = rows.length
      ? rows
          .map((row, index) => `
            <tr data-selected-indicator-index="${index}">
              <td>
                <strong>${escapeHtml(row.label)}</strong>
                <small>${escapeHtml(row.indicator_id)}</small>
              </td>
              <td><input class="backtests-input backtests-input--axis" type="number" min="${escapeHtml(row.window.min)}" max="${escapeHtml(row.window.max)}" step="${escapeHtml(row.window.unitStep)}" value="${escapeHtml(row.window.start)}" data-indicator-window="start" aria-label="${escapeHtml(row.label)} from"></td>
              <td><input class="backtests-input backtests-input--axis" type="number" min="${escapeHtml(row.window.min)}" max="${escapeHtml(row.window.max)}" step="${escapeHtml(row.window.unitStep)}" value="${escapeHtml(row.window.stop)}" data-indicator-window="stop" aria-label="${escapeHtml(row.label)} to"></td>
              <td><input class="backtests-input backtests-input--axis" type="number" min="${escapeHtml(row.window.unitStep)}" step="${escapeHtml(row.window.unitStep)}" value="${escapeHtml(row.window.step)}" data-indicator-window="step" aria-label="${escapeHtml(row.label)} step"></td>
              <td>
                <div class="backtests-source-list">
                  ${row.availableSources.length
                    ? row.availableSources
                        .map((source) => `
                          <button
                            class="backtests-source-chip ${row.sources.includes(source) ? "is-selected" : ""}"
                            type="button"
                            data-indicator-source="${escapeHtml(source)}"
                          >${escapeHtml(source)}</button>
                        `)
                        .join("")
                    : `<span class="backtests-muted">--</span>`}
                </div>
              </td>
              <td>
                <button
                  class="rh-button rh-button--secondary rh-button--compact backtests-icon-button"
                  type="button"
                  data-remove-indicator
                  aria-label="${escapeHtml(t("backtests.actions.delete"))}"
                  title="${escapeHtml(t("backtests.actions.delete"))}"
                >${trashIcon(t("backtests.actions.delete"))}</button>
              </td>
            </tr>
          `)
          .join("")
      : `<tr><td colspan="6">${escapeHtml(t("common.unavailable"))}</td></tr>`;
  }
  setText("[data-combinations-count]", compactMagnitude(indicatorCombinationCount()), root);
}

function indicatorStateFromDraft(draft) {
  const catalogItem = state.indicatorCatalog.get(draft?.indicator_id);
  if (!catalogItem) {
    return null;
  }
  const spec = primaryWindowSpec(catalogItem);
  const draftWindow = draft?.window || draft?.params?.window || {};
  const start = draftWindow.start ?? rangeDefault(spec, "start", 5);
  const stop = draftWindow.stop ?? draftWindow.stop_incl ?? rangeDefault(spec, "stop_incl", 30);
  const step = draftWindow.step ?? rangeDefault(spec, "step", 1);
  return {
    indicator_id: catalogItem.indicator_id,
    label: catalogItem.label,
    family: catalogItem.family,
    availableSources: catalogItem.sources || [],
    sources: draft.sources || (catalogItem.sources?.[0] ? [catalogItem.sources[0]] : []),
    window: {
      start,
      stop,
      step,
      min: specMin(spec),
      max: specMax(spec),
      unitStep: rangeDefault(spec, "step", 1),
    },
  };
}

function addIndicator(root, indicatorId) {
  const catalogItem = state.indicatorCatalog.get(indicatorId);
  if (!catalogItem) {
    return;
  }
  state.selectedIndicators.push(indicatorStateFromDraft({ indicator_id: indicatorId }));
  renderIndicators(root, { items: Array.from(state.indicatorCatalog.values()) });
}

function renderIndicatorAddMenu(root, items) {
  const target = qs("[data-indicator-add-menu]", root);
  if (!target) {
    return;
  }
  const groups = new Map();
  items.forEach((item) => {
    const family = item.family || "other";
    groups.set(family, [...(groups.get(family) || []), item]);
  });
  const families = Array.from(groups.keys()).sort();
  if (!state.indicatorFamily || !groups.has(state.indicatorFamily)) {
    state.indicatorFamily = families[0] || null;
  }
  const activeItems = groups.get(state.indicatorFamily) || [];
  target.innerHTML = `
    <div class="backtests-indicator-family-tabs" role="tablist" aria-label="${escapeHtml(t("backtests.indicators.family_tabs"))}">
      ${families
        .map((family) => `
          <button
            class="backtests-family-tab ${family === state.indicatorFamily ? "is-active" : ""}"
            type="button"
            role="tab"
            aria-selected="${family === state.indicatorFamily ? "true" : "false"}"
            data-indicator-family-tab="${escapeHtml(family)}"
          >${escapeHtml(family)}</button>
        `)
        .join("")}
    </div>
    <div class="backtests-indicator-family-list" role="group" aria-label="${escapeHtml(state.indicatorFamily || "")}">
      ${activeItems
        .map((item) => `
          <button class="rh-menu-item" type="button" role="option" data-add-indicator="${escapeHtml(item.indicator_id)}">
            <strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.indicator_id)}</small>
          </button>
        `)
        .join("")}
    </div>
  `;
}

function indicatorCombinationCount() {
  return state.selectedIndicators.reduce((total, indicator) => {
    const windowCount = countRange(indicator.window.start, indicator.window.stop, indicator.window.step);
    const sourceCount = Math.max(1, indicator.sources.length);
    return total * Math.max(1, windowCount * sourceCount);
  }, 1);
}

function renderOptimization(root, overview) {
  const progress = Number(overview?.progress_percent || 0);
  setText(
    "[data-progress-label]",
    overview?.active_job_id
      ? t("backtests.optimization.job", { job: overview.active_job_id.slice(0, 8) })
      : t("backtests.optimization.awaiting"),
    root
  );
  setText("[data-progress-percent]", `${progress}%`, root);
  const bar = qs("[data-progress-bar]", root);
  if (bar) {
    bar.style.width = `${Math.max(0, Math.min(progress, 100))}%`;
  }
  setText("[data-remaining]", overview?.estimated_remaining || "--", root);
  setText("[data-completed]", overview?.completed_jobs ?? 0, root);
  setText("[data-running]", overview?.running_jobs ?? 0, root);
  setText("[data-queued]", overview?.queued_jobs ?? 0, root);
  setText("[data-progress-units]", `${overview?.processed_units || 0}/${overview?.total_units || 0}`, root);
}

function renderJobs(root, table) {
  const target = qs("[data-job-rows]", root);
  if (!target) {
    return;
  }
  const rows = table?.items || [];
  state.jobRows = rows;
  state.nextCursor = state.loadedAllJobs ? null : table?.next_cursor || null;
  renderJobPicker(root, rows);
  renderJobPagination(root);
  if (!rows.length) {
    target.innerHTML = `<tr><td colspan="9">${escapeHtml(table?.degradation_reason || t("backtests.results.empty"))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map((row, index) => renderJobRow(root, row, index))
    .join("");
  queueVariantPanelAnimation(root);
}

function renderJobRow(root, row, index) {
  const selected = state.selectedJobId === row.job_id;
  const canDelete = row.actions?.can_delete;
  return `
    <tr class="${selected ? "is-selected" : ""}" data-job-id="${escapeHtml(row.job_id)}" tabindex="0">
      <td>
        <button class="backtests-job-toggle" type="button" data-select-job="${escapeHtml(row.job_id)}" aria-expanded="${selected ? "true" : "false"}">
          ${selected ? "v" : ">"} ${index + 1}
        </button>
      </td>
      <td>${escapeHtml(row.strategy)}</td>
      <td>${escapeHtml(formatDateTimeMinute(row.created_at))}</td>
      <td>${escapeHtml(row.exchange || "--")}</td>
      <td>${escapeHtml(row.market_type || "--")}</td>
      <td>${escapeHtml(row.symbol || "--")}</td>
      <td>${escapeHtml(row.period)}</td>
      <td>${escapeHtml(row.direction)}</td>
      <td>
        <div class="backtests-status-cell">
          <span>${escapeHtml(row.state)} / ${row.progress_percent}%</span>
          ${row.actions?.can_cancel
            ? `<button class="rh-button rh-button--secondary rh-button--compact backtests-row-action" type="button" data-cancel-job-id="${escapeHtml(row.job_id)}">${escapeHtml(t("backtests.actions.cancel"))}</button>`
            : ""}
          ${canDelete
            ? `<button class="rh-button rh-button--secondary rh-button--compact backtests-row-action backtests-row-action--danger backtests-icon-button" type="button" data-delete-job-id="${escapeHtml(row.job_id)}" aria-label="${escapeHtml(t("backtests.actions.delete"))}" title="${escapeHtml(t("backtests.actions.delete"))}">${trashIcon(t("backtests.actions.delete"))}</button>`
            : ""}
        </div>
      </td>
    </tr>
    ${selected ? renderVariantExpansion(root, row) : ""}
  `;
}

function renderVariantExpansion(root, row) {
  const summary = state.resultSummary?.job?.job_id === row.job_id ? state.resultSummary : null;
  const variants = (summary?.top_variants?.items || []).slice(0, variantPreviewLimit(root));
  const title = t("backtests.variants.title", { job: compactId(row.job_id) });
  const shouldAnimate = state.animateVariantJobId === row.job_id;
  const frameClass = shouldAnimate
    ? "backtests-variant-frame"
    : "backtests-variant-frame backtests-variant-frame--static is-open";
  const body = variants.length
    ? variants.map((variant) => renderVariantRow(root, row.job_id, variant)).join("")
    : `<tr><td colspan="10">${escapeHtml(activeResultRequest ? t("backtests.variants.loading") : t("backtests.variants.empty"))}</td></tr>`;
  return `
    <tr class="backtests-variant-expansion">
      <td class="backtests-variant-cell" colspan="9">
        <div class="${frameClass}" data-variant-frame ${shouldAnimate ? 'data-variant-animate="true"' : ""}>
          <section class="backtests-variant-panel" aria-label="${escapeHtml(title)}">
            <header class="backtests-variant-panel__heading">
              <strong>${escapeHtml(title)}</strong>
              <span>${escapeHtml(row.strategy)} · ${escapeHtml(row.symbol || "--")} · ${escapeHtml(formatDateTimeMinute(row.created_at))}</span>
            </header>
            <div class="backtests-table-wrap backtests-variant-table-wrap">
              <table class="backtests-table backtests-table--variants">
                <thead>
                  <tr>
                    <th>${escapeHtml(t("backtests.variants.rank"))}</th>
                    <th>${escapeHtml(t("backtests.variants.variant"))}</th>
                    <th>${escapeHtml(t("backtests.variants.params"))}</th>
                    <th>${escapeHtml(t("backtests.results.return"))}</th>
                    <th>${escapeHtml(t("backtests.results.sharpe"))}</th>
                    <th>${escapeHtml(t("backtests.results.drawdown"))}</th>
                    <th>${escapeHtml(t("backtests.results.profit_factor"))}</th>
                    <th>${escapeHtml(t("backtests.results.win_rate"))}</th>
                    <th>${escapeHtml(t("backtests.results.trades"))}</th>
                    <th>${escapeHtml(t("backtests.variants.csv"))}</th>
                  </tr>
                </thead>
                <tbody>${body}</tbody>
              </table>
            </div>
          </section>
        </div>
      </td>
    </tr>
  `;
}

function queueVariantPanelAnimation(root) {
  if (variantAnimationFrame) {
    window.cancelAnimationFrame(variantAnimationFrame);
  }
  const frame = qs("[data-variant-frame][data-variant-animate='true']", root);
  if (!frame) {
    variantAnimationFrame = null;
    state.animateVariantJobId = null;
    return;
  }
  frame.style.setProperty("--backtests-variant-open-duration", `${variantOpenDurationMs(root)}ms`);
  variantAnimationFrame = window.requestAnimationFrame(() => {
    frame.style.setProperty("--backtests-variant-height", `${frame.scrollHeight}px`);
    frame.classList.add("is-open");
    frame.removeAttribute("data-variant-animate");
    state.animateVariantJobId = null;
    variantAnimationFrame = null;
  });
}

function renderJobPagination(root) {
  setText(
    "[data-job-page-status]",
    t("backtests.results.page_status", { count: state.jobRows.length }),
    root
  );
  const button = qs("[data-load-more-jobs]", root);
  if (button instanceof HTMLButtonElement) {
    button.hidden = !state.nextCursor;
    button.disabled = Boolean(activeRequest);
  }
}

function renderVariantRow(root, jobId, variant) {
  const metrics = variant?.summary_metrics || {};
  const selected = state.selectedVariantKey === variant.variant_key;
  const href = `${variantBaseEndpoint(root, jobId, variant.variant_key)}/trades.csv`;
  const maxDrawdown = metrics.max_drawdown_pct ?? metrics.avg_drawdown_pct;
  const trades = metrics.trade_count ?? metrics.trades_count;
  return `
    <tr class="${selected ? "is-selected" : ""}" data-result-variant-key="${escapeHtml(variant.variant_key)}" tabindex="0">
      <td>#${numberOrDash(variant.rank)}</td>
      <td>${escapeHtml(compactId(variant.variant_key))}</td>
      <td class="backtests-variant-params">${escapeHtml(formatVariantParams(variant))}</td>
      <td class="${financialClass(metrics.total_return_pct)}">${percent(metrics.total_return_pct)}</td>
      <td class="${financialClass(metrics.sharpe)}">${numberOrDash(metrics.sharpe)}</td>
      <td class="rh-financial--negative">${signedDrawdownPercent(maxDrawdown)}</td>
      <td>${decimalOrDash(metrics.profit_factor, 3)}</td>
      <td>${percent(metrics.win_rate_pct, 2)}</td>
      <td>${integerOrDash(trades)}</td>
      <td><a class="rh-button rh-button--secondary rh-button--compact backtests-download-button" href="${escapeHtml(href)}" aria-label="${escapeHtml(t("backtests.variants.csv"))}" title="${escapeHtml(t("backtests.variants.csv"))}">↓</a></td>
    </tr>
  `;
}

function formatVariantParams(variant) {
  const params = variant?.readable_params || variant?.canonical_variant_params || {};
  const entries = Object.entries(params)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 8)
    .map(([key, value]) => `${key}: ${formatParamValue(value)}`);
  if (variant?.best_tp_pct !== null && variant?.best_tp_pct !== undefined) {
    entries.push(`tp: ${variant.best_tp_pct}`);
  }
  if (variant?.best_sl_pct !== null && variant?.best_sl_pct !== undefined) {
    entries.push(`sl: ${variant.best_sl_pct}`);
  }
  return entries.join(" · ") || "--";
}

function formatParamValue(value) {
  if (Array.isArray(value)) {
    return value.join("/");
  }
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, nested]) => `${key}=${nested}`)
      .join(",");
  }
  return String(value);
}

function renderJobPicker(root, rows) {
  const target = qs("[data-job-picker-menu]", root);
  if (target) {
    target.innerHTML = rows.length
      ? rows
          .map((row) => `
            <button class="rh-menu-item" type="button" role="option" data-pick-job-id="${escapeHtml(row.job_id)}">
              ${escapeHtml(compactId(row.job_id))} · ${escapeHtml(row.symbol || "--")} · ${escapeHtml(row.state)}
            </button>
          `)
          .join("")
      : `<span class="backtests-menu-group">${escapeHtml(t("backtests.results.empty"))}</span>`;
  }
  setText(
    "[data-current-job]",
    state.selectedJobId ? compactId(state.selectedJobId) : t("backtests.results.select_job"),
    root
  );
}

function renderResultSummary(root, summary) {
  state.resultSummary = summary;
  const selectedKey = state.selectedVariantKey || summary?.selected_variant_key;
  state.selectedVariantKey = selectedKey;
  renderJobs(root, { items: state.jobRows });
}

function renderFooter(root, data) {
  const sources = data?.sources || [];
  const availableSources = sources.filter((source) => source.status === "available").length;
  setText(
    "[data-footer-connection]",
    data?.footer_status?.api === "available" ? "connected" : data?.footer_status?.api || "--",
    document
  );
  setText(
    "[data-footer-data]",
    `${availableSources}/${sources.length || 0} ${data?.footer_status?.data || "--"}`,
    document
  );
  setText("[data-footer-api]", data?.footer_status?.api || "--");
  setText("[data-footer-latency]", "--", document);
  setText("[data-footer-time]", localTime(data?.generated_at));
  setText(
    "[data-backtests-freshness]",
    data?.retry_after_seconds
      ? t("dashboard.refresh.rate_limited", { seconds: data.retry_after_seconds })
      : t("backtests.refresh.freshness", {
          status: data?.refresh_status || "unknown",
          time: localTime(data?.generated_at),
        })
  );
  setText("[data-backtests-refresh-status]", data?.refresh_status || t("refresh.idle"), document);
}

function renderWorkstation(root, data, { append = false, preserveLoadedRows = false } = {}) {
  state.runtimeDefaults = data;
  manualRefreshRetrySeconds = Number(data?.retry_after_seconds || 0);
  const renderedData = append
    ? mergeJobTableData(data)
    : preserveLoadedRows
      ? preserveLoadedJobTableData(data)
      : data;
  renderRuntimeControls(root, data);
  renderSymbols(root, data?.instrument_universe);
  renderIndicators(root, data?.indicator_catalog);
  renderOptimization(root, data?.optimization_overview);
  renderJobs(root, renderedData?.job_table);
  renderFooter(root, renderedData);
  const loading = qs("[data-backtests-loading]", root);
  if (loading) {
    loading.hidden = true;
  }
}

function mergeJobTableData(data) {
  const nextRows = data?.job_table?.items || [];
  const seen = new Set(state.jobRows.map((row) => row.job_id));
  const mergedRows = [
    ...state.jobRows,
    ...nextRows.filter((row) => {
      if (seen.has(row.job_id)) {
        return false;
      }
      seen.add(row.job_id);
      return true;
    }),
  ];
  state.loadedAllJobs = !data?.job_table?.next_cursor;
  return {
    ...data,
    job_table: {
      ...(data?.job_table || {}),
      items: mergedRows,
    },
  };
}

function preserveLoadedJobTableData(data) {
  const freshRows = data?.job_table?.items || [];
  if (!state.jobRows.length || state.jobRows.length <= freshRows.length) {
    return data;
  }
  const seen = new Set(freshRows.map((row) => row.job_id));
  const mergedRows = [
    ...freshRows,
    ...state.jobRows.filter((row) => {
      if (seen.has(row.job_id)) {
        return false;
      }
      seen.add(row.job_id);
      return true;
    }),
  ];
  return {
    ...data,
    job_table: {
      ...(data?.job_table || {}),
      items: mergedRows,
      next_cursor: state.loadedAllJobs ? null : data?.job_table?.next_cursor || null,
    },
  };
}

async function loadResultSummary(root, jobId, { render = true } = {}) {
  if (!jobId) {
    return null;
  }
  const template = root.dataset.jobSummaryEndpointTemplate || "/api/backtests/jobs/{job_id}/summary";
  const endpoint = new URL(endpointFromTemplate(template, { job_id: jobId }), window.location.origin);
  endpoint.searchParams.set("top_limit", String(variantPreviewLimit(root)));
  const summary = await apiFetch(`${endpoint.pathname}${endpoint.search}`);
  state.selectedJobId = summary.job?.job_id || jobId;
  if (!state.selectedVariantKey) {
    state.selectedVariantKey = summary.selected_variant_key;
  }
  if (render) {
    renderResultSummary(root, summary);
  }
  return summary;
}

async function loadSelectedResult(root) {
  if (!state.selectedJobId || activeResultRequest) {
    return activeResultRequest;
  }
  activeResultRequest = loadResultSummary(root, state.selectedJobId).finally(() => {
    activeResultRequest = null;
  });
  return activeResultRequest;
}

function clearDelayedVariantOpen() {
  if (delayedVariantOpen) {
    window.clearTimeout(delayedVariantOpen);
    delayedVariantOpen = null;
  }
}

function variantOpenDelayMs(root) {
  const raw = Number(root.dataset.variantOpenDelayMs || DEFAULT_VARIANT_OPEN_DELAY_MS);
  return Number.isFinite(raw) && raw >= 0 ? raw : DEFAULT_VARIANT_OPEN_DELAY_MS;
}

function variantOpenDurationMs(root) {
  const raw = Number(root.dataset.variantOpenDurationMs || DEFAULT_VARIANT_OPEN_DURATION_MS);
  return Number.isFinite(raw) && raw >= 0 ? raw : DEFAULT_VARIANT_OPEN_DURATION_MS;
}

function variantPreviewLimit(root) {
  const raw = Number(root.dataset.variantPreviewLimit || DEFAULT_VARIANT_PREVIEW_LIMIT);
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : DEFAULT_VARIANT_PREVIEW_LIMIT;
}

async function openSelectedJob(root, jobId) {
  if (!jobId) {
    state.selectedJobId = null;
    state.selectedVariantKey = null;
    state.resultSummary = null;
    state.animateVariantJobId = null;
    renderJobs(root, { items: state.jobRows, next_cursor: state.nextCursor });
    return;
  }
  state.selectedVariantKey = null;
  renderJobPicker(root, state.jobRows);
  try {
    activeResultRequest = loadResultSummary(root, jobId, { render: false });
    const summary = await activeResultRequest;
    state.selectedJobId = summary.job?.job_id || jobId;
    state.selectedVariantKey = summary.selected_variant_key;
    state.animateVariantJobId = state.selectedJobId;
    renderResultSummary(root, summary);
  } catch (error) {
    setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
  } finally {
    activeResultRequest = null;
  }
}

function selectJob(root, jobId, { delayed = true } = {}) {
  clearDelayedVariantOpen();
  if (!jobId || !delayed) {
    openSelectedJob(root, jobId);
    return;
  }
  setText("[data-create-status]", t("backtests.status.opening_job", { job: compactId(jobId) }), root);
  delayedVariantOpen = window.setTimeout(() => {
    delayedVariantOpen = null;
    openSelectedJob(root, jobId).catch((error) => {
      setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
    });
  }, variantOpenDelayMs(root));
}

async function refreshWorkstation(root, reason = "manual", { append = false } = {}) {
  if (activeRequest) {
    return activeRequest;
  }
  if (!append && reason !== "auto") {
    state.loadedAllJobs = false;
  }
  const endpoint = root.dataset.workstationEndpoint || DEFAULT_ENDPOINT;
  const params = new URLSearchParams();
  params.set("refresh", reason);
  if (state.job_state) {
    params.set("state", state.job_state);
  }
  if (state.job_exchange) {
    params.set("exchange", state.job_exchange);
  }
  if (state.job_market_type) {
    params.set("market_type", state.job_market_type);
  }
  if (append && state.nextCursor) {
    params.set("cursor", state.nextCursor);
  }
  if (state.query) {
    params.set("query", state.query);
  }
  if (state.job_symbol) {
    params.set("symbol", state.job_symbol);
  }
  if (state.launched_from) {
    params.set("launched_from", state.launched_from);
  }
  if (state.launched_to) {
    params.set("launched_to", state.launched_to);
  }
  activeRequest = apiFetch(`${endpoint}?${params.toString()}`)
    .then((data) => {
      renderWorkstation(root, data, {
        append,
        preserveLoadedRows: !append && reason === "auto" && state.jobRows.length > 0,
      });
      if (state.selectedJobId && reason !== "initial") {
        loadSelectedResult(root).catch(() => {});
      }
      return data;
    })
    .finally(() => {
      activeRequest = null;
      renderJobPagination(root);
    });
  return activeRequest;
}

async function preflight(root) {
  const payload = buildRequestPayload(root);
  const endpoint = root.dataset.preflightEndpoint || "/api/backtests/preflight";
  setText("[data-create-status]", t("backtests.status.preflight"), root);
  const result = await apiFetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  setText(
    "[data-create-status]",
    t("backtests.status.preflight_ok", { hash: String(result.request_hash || "").slice(0, 8) }),
    root
  );
  return { payload, preflight: result };
}

async function createJob(root) {
  const buttons = qsa("[data-create-button], [data-create-button-secondary]", root);
  const isRunning = true;
  buttons.forEach((button) => {
    button.disabled = isRunning;
  });
  try {
    const { payload } = await preflight(root);
    const idempotencyKey = window.crypto?.randomUUID
      ? window.crypto.randomUUID()
      : `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const created = await apiFetch(root.dataset.jobsEndpoint || "/api/backtests/jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify(payload),
    });
    setText("[data-create-status]", t("backtests.status.created", { job: created.job_id.slice(0, 8) }), root);
    state.selectedJobId = created.job_id;
    state.selectedVariantKey = null;
    await refreshWorkstation(root, "manual");
  } catch (error) {
    setText("[data-create-status]", describeApiError(error), root);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

async function cancelJob(root, jobId) {
  if (!jobId) {
    return;
  }
  const endpoint = endpointFromTemplate(
    root.dataset.jobCancelEndpointTemplate || "/api/backtests/jobs/{job_id}/cancel",
    { job_id: jobId }
  );
  setText("[data-create-status]", t("backtests.status.cancelling", { job: compactId(jobId) }), root);
  try {
    const result = await apiFetch(endpoint, { method: "POST" });
    setText(
      "[data-create-status]",
      t("backtests.status.cancelled_job", { job: compactId(result?.job_id || jobId) }),
      root
    );
    if (state.selectedJobId === jobId) {
      state.selectedJobId = null;
      state.selectedVariantKey = null;
    }
    await refreshWorkstation(root, "manual");
  } catch (error) {
    setText("[data-create-status]", describeApiError(error), root);
  }
}

async function deleteJob(root, jobId) {
  if (!jobId) {
    return;
  }
  if (!window.confirm(t("backtests.confirm_delete", { job: compactId(jobId) }))) {
    return;
  }
  const endpoint = endpointFromTemplate(
    root.dataset.jobDeleteEndpointTemplate || "/api/backtests/jobs/{job_id}",
    { job_id: jobId }
  );
  setText("[data-create-status]", t("backtests.status.deleting_job", { job: compactId(jobId) }), root);
  try {
    await apiFetch(endpoint, { method: "DELETE" });
    setText("[data-create-status]", t("backtests.status.deleted_job", { job: compactId(jobId) }), root);
    if (state.selectedJobId === jobId) {
      state.selectedJobId = null;
      state.selectedVariantKey = null;
      state.resultSummary = null;
    }
    state.jobRows = state.jobRows.filter((row) => row.job_id !== jobId);
    renderJobs(root, { items: state.jobRows, next_cursor: state.nextCursor });
    await refreshWorkstation(root, "auto");
  } catch (error) {
    setText("[data-create-status]", describeApiError(error), root);
  }
}

function setAutorefresh(root, presetKey) {
  const intervalMs = REFRESH_PRESETS[presetKey] ?? 0;
  if (poller) {
    poller.stop();
    poller = null;
  }
  if (intervalMs > 0) {
    poller = createPoller(() => refreshWorkstation(root, "auto"), {
      intervalMs,
      hiddenTabPause: true,
    });
    poller.start();
  }
  setText("[data-backtests-refresh-current]", presetKey, document);
  setText("[data-backtests-refresh-status]", intervalMs > 0 ? presetKey : t("refresh.idle"), document);
}

function closeStatusRefreshMenu(statusBar) {
  const dropdown = qs("[data-rh-dropdown]", statusBar);
  const trigger = qs("[data-rh-dropdown-trigger]", statusBar);
  const menu = qs("[data-rh-dropdown-menu]", statusBar);
  if (dropdown instanceof HTMLElement) {
    dropdown.dataset.open = "false";
  }
  if (trigger instanceof HTMLElement) {
    trigger.setAttribute("aria-expanded", "false");
  }
  if (menu instanceof HTMLElement) {
    menu.hidden = true;
  }
}

function openStatusRefreshMenu(statusBar) {
  const dropdown = qs("[data-rh-dropdown]", statusBar);
  const trigger = qs("[data-rh-dropdown-trigger]", statusBar);
  const menu = qs("[data-rh-dropdown-menu]", statusBar);
  if (dropdown instanceof HTMLElement) {
    dropdown.dataset.open = "true";
  }
  if (trigger instanceof HTMLElement) {
    trigger.setAttribute("aria-expanded", "true");
  }
  if (menu instanceof HTMLElement) {
    menu.hidden = false;
  }
}

function toggleStatusRefreshMenu(statusBar) {
  const dropdown = qs("[data-rh-dropdown]", statusBar);
  if (dropdown?.dataset.open === "true") {
    closeStatusRefreshMenu(statusBar);
  } else {
    openStatusRefreshMenu(statusBar);
  }
}

function updateRefreshPresetSelection(statusBar, presetKey) {
  qsa("[data-backtests-refresh-preset]", statusBar).forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.backtestsRefreshPreset === presetKey ? "true" : "false");
  });
}

function bindStatusBar(root) {
  const statusBar = qs("[data-backtests-status-bar]", document);
  if (!statusBar || statusBar.dataset.backtestsBound === "true") {
    return;
  }
  statusBar.dataset.backtestsBound = "true";
  statusBar.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const refreshTrigger = event.target.closest("#backtests-refresh-trigger");
    if (refreshTrigger instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      toggleStatusRefreshMenu(statusBar);
      return;
    }
    const refreshButton = event.target.closest("[data-backtests-refresh]");
    if (refreshButton instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      setText("[data-backtests-refresh-status]", t("refresh.manual"), document);
      if (manualRefreshRetrySeconds > 0) {
        setText("[data-backtests-freshness]", t("dashboard.refresh.rate_limited", { seconds: manualRefreshRetrySeconds }), document);
      }
      refreshWorkstation(root, "manual").catch(() => {});
      return;
    }
    const preset = event.target.closest("[data-backtests-refresh-preset]");
    if (preset instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      const presetKey = preset.dataset.backtestsRefreshPreset || "off";
      updateRefreshPresetSelection(statusBar, presetKey);
      setAutorefresh(root, presetKey);
      closeStatusRefreshMenu(statusBar);
    }
  });
}

function bind(root) {
  bindStatusBar(root);
  root.addEventListener("click", (event) => {
    const option = event.target.closest("[data-backtest-option]");
    if (option instanceof HTMLElement) {
      updateOptionSelection(
        root,
        option.dataset.backtestOption || "",
        option.dataset.value || "",
        (option.textContent || "").trim()
      );
      return;
    }
    const createButton = event.target.closest("[data-create-button], [data-create-button-secondary]");
    if (createButton instanceof HTMLElement) {
      createJob(root).catch(() => {});
      return;
    }
    const preflightButton = event.target.closest("[data-preflight-button]");
    if (preflightButton instanceof HTMLElement) {
      preflight(root).catch((error) => {
        setText("[data-create-status]", describeApiError(error), root);
      });
      return;
    }
    const cancelButton = event.target.closest("[data-cancel-job-id]");
    if (cancelButton instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      cancelJob(root, cancelButton.dataset.cancelJobId || "").catch(() => {});
      return;
    }
    const deleteButton = event.target.closest("[data-delete-job-id]");
    if (deleteButton instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      deleteJob(root, deleteButton.dataset.deleteJobId || "").catch(() => {});
      return;
    }
    const selectJobButton = event.target.closest("[data-select-job]");
    if (selectJobButton instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      selectJob(root, selectJobButton.dataset.selectJob || null);
      return;
    }
    const clearSymbols = event.target.closest("[data-clear-symbols]");
    if (clearSymbols instanceof HTMLElement) {
      const firstSymbol = qs("[data-symbol-select]", root)?.dataset.symbolSelect || "BTCUSDT";
      state.symbol = firstSymbol;
      state.selectedSymbols = new Set([firstSymbol]);
      renderSelectedSymbols(root);
      renderSymbols(root, state.runtimeDefaults?.instrument_universe);
      return;
    }
    const symbolButton = event.target.closest("[data-symbol-select]");
    if (symbolButton instanceof HTMLElement) {
      const symbol = symbolButton.dataset.symbolSelect || "";
      if (symbol) {
        state.symbol = symbol;
        state.selectedSymbols = new Set([symbol]);
        renderSymbols(root, state.runtimeDefaults?.instrument_universe);
      }
      return;
    }
    const addIndicatorButton = event.target.closest("[data-add-indicator]");
    if (addIndicatorButton instanceof HTMLElement) {
      addIndicator(root, addIndicatorButton.dataset.addIndicator || "");
      return;
    }
    const indicatorFamilyTab = event.target.closest("[data-indicator-family-tab]");
    if (indicatorFamilyTab instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      const dropdown = indicatorFamilyTab.closest("[data-rh-dropdown]");
      state.indicatorFamily = indicatorFamilyTab.dataset.indicatorFamilyTab || state.indicatorFamily;
      renderIndicatorAddMenu(root, Array.from(state.indicatorCatalog.values()));
      if (dropdown instanceof HTMLElement) {
        dropdown.dataset.open = "true";
        const trigger = qs("[data-rh-dropdown-trigger]", dropdown);
        const menu = qs("[data-rh-dropdown-menu]", dropdown);
        if (trigger instanceof HTMLElement) {
          trigger.setAttribute("aria-expanded", "true");
        }
        if (menu instanceof HTMLElement) {
          menu.hidden = false;
        }
      }
      return;
    }
    const removeIndicatorButton = event.target.closest("[data-remove-indicator]");
    if (removeIndicatorButton instanceof HTMLElement) {
      const row = removeIndicatorButton.closest("[data-selected-indicator-index]");
      const index = Number(row?.dataset.selectedIndicatorIndex);
      if (Number.isInteger(index)) {
        state.selectedIndicators.splice(index, 1);
        renderIndicators(root, { items: Array.from(state.indicatorCatalog.values()) });
      }
      return;
    }
    const sourceButton = event.target.closest("[data-indicator-source]");
    if (sourceButton instanceof HTMLElement) {
      const row = sourceButton.closest("[data-selected-indicator-index]");
      const index = Number(row?.dataset.selectedIndicatorIndex);
      const source = sourceButton.dataset.indicatorSource || "";
      const indicator = state.selectedIndicators[index];
      if (indicator) {
        const selected = new Set(indicator.sources);
        if (selected.has(source) && selected.size > 1) {
          selected.delete(source);
        } else {
          selected.add(source);
        }
        indicator.sources = Array.from(selected);
        renderIndicators(root, { items: Array.from(state.indicatorCatalog.values()) });
      }
      return;
    }
    const refreshButton = event.target.closest("[data-backtests-refresh]");
    if (refreshButton instanceof HTMLElement) {
      if (manualRefreshRetrySeconds > 0) {
        setText("[data-backtests-freshness]", t("dashboard.refresh.rate_limited", { seconds: manualRefreshRetrySeconds }));
      }
      refreshWorkstation(root, "manual").catch(() => {});
      return;
    }
    const loadMoreButton = event.target.closest("[data-load-more-jobs]");
    if (loadMoreButton instanceof HTMLElement) {
      refreshWorkstation(root, "auto", { append: true }).catch(() => {});
      return;
    }
    const jobRow = event.target.closest("[data-job-id]");
    if (jobRow instanceof HTMLElement) {
      selectJob(root, jobRow.dataset.jobId || null);
      return;
    }
    const jobPick = event.target.closest("[data-pick-job-id]");
    if (jobPick instanceof HTMLElement) {
      selectJob(root, jobPick.dataset.pickJobId || null);
      return;
    }
    const variantButton = event.target.closest("[data-result-variant-key]");
    if (variantButton instanceof HTMLElement) {
      state.selectedVariantKey = variantButton.dataset.resultVariantKey || null;
      renderJobs(root, { items: state.jobRows });
      return;
    }
    const preset = event.target.closest("[data-backtests-refresh-preset]");
    if (preset instanceof HTMLElement) {
      setAutorefresh(root, preset.dataset.backtestsRefreshPreset || "off");
      return;
    }
  });

  qs("[data-job-search]", root)?.addEventListener("input", (event) => {
    state.query = event.target.value || "";
    refreshWorkstation(root, "manual").catch(() => {});
  });
  qs("[data-job-symbol]", root)?.addEventListener("input", (event) => {
    state.job_symbol = (event.target.value || "").trim().toUpperCase();
    refreshWorkstation(root, "manual").catch(() => {});
  });
  qs("[data-job-launched-from]", root)?.addEventListener("change", (event) => {
    state.launched_from = event.target.value || "";
    refreshWorkstation(root, "manual").catch(() => {});
  });
  qs("[data-job-launched-to]", root)?.addEventListener("change", (event) => {
    state.launched_to = event.target.value || "";
    refreshWorkstation(root, "manual").catch(() => {});
  });
  qs("[data-symbol-search]", root)?.addEventListener("input", (event) => {
    filterSymbols(root, event.target.value || "");
  });
  root.addEventListener("change", (event) => {
    const checkbox = event.target.closest("[data-symbol-checkbox]");
    if (checkbox instanceof HTMLInputElement) {
      renderSelectedSymbols(root);
      return;
    }
    const axisInput = event.target.closest("[data-indicator-window]");
    if (axisInput instanceof HTMLInputElement) {
      const row = axisInput.closest("[data-selected-indicator-index]");
      const index = Number(row?.dataset.selectedIndicatorIndex);
      const indicator = state.selectedIndicators[index];
      if (indicator) {
        indicator.window[axisInput.dataset.indicatorWindow || "start"] = Number(axisInput.value);
        setText("[data-combinations-count]", compactMagnitude(indicatorCombinationCount()), root);
      }
    }
  });
}

function init() {
  const root = qs("[data-backtests-root]");
  if (!root) {
    return;
  }
  bind(root);
  state.selectedJobId = root.dataset.initialJobId || null;
  refreshWorkstation(root, "initial").then(() => {
    if (state.selectedJobId) {
      return loadSelectedResult(root);
    }
    return null;
  }).catch((error) => {
    setText("[data-create-status]", describeApiError(error), root);
  });
  setAutorefresh(root, "15s");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
