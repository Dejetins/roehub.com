import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { t } from "../core/locale.js";
import { createPoller } from "../core/poller.js";
import { renderBacktestSeries } from "../charts/backtest_series.js";

const DEFAULT_ENDPOINT = "/api/ui/backtests/workstation";
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
  ranking_metric: "total_return_pct",
  ranking_order: "desc",
  job_state: "",
  job_symbol: "",
  launched_from: "",
  launched_to: "",
  cursor: null,
  query: "",
  runtimeDefaults: null,
  selectedSymbols: new Set(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
  selectedIndicators: [],
  indicatorCatalog: new Map(),
  indicatorFamily: null,
  jobRows: [],
  selectedJobId: null,
  selectedVariantKey: null,
  resultSummary: null,
  tradesPage: 1,
  tradesHasNext: false,
  chartKind: "equity",
};

let activeRequest = null;
let activeResultRequest = null;
let poller = null;
let manualRefreshRetrySeconds = 0;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${number.toFixed(1)}%`;
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
  const selected = qsa("[data-symbol-checkbox]:checked", root).map((item) => item.value);
  if (selected.length) {
    state.selectedSymbols = new Set(selected);
  }
  return selected.length ? selected : Array.from(state.selectedSymbols || [state.symbol || "BTCUSDT"]);
}

function buildRequestPayload(root) {
  const start = qs("[data-config-field='start']", root)?.value || "2023-01-01";
  const end = qs("[data-config-field='end']", root)?.value || "2024-01-01";
  const capital = Number(qs("[data-config-field='capital']", root)?.value || 10000);
  const feePercent = Number(qs("[data-config-field='fee']", root)?.value || 0.075);
  const slippagePercent = Number(qs("[data-config-field='slippage']", root)?.value || 0.01);
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
      sizing: { mode: "fixed_equity_pct", equity_pct: 10.0 },
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

function updateOptionSelection(root, name, value, label) {
  state[name] = value;
  qsa(`[data-backtest-option='${name}']`, root).forEach((option) => {
    option.setAttribute("aria-selected", option.dataset.value === value ? "true" : "false");
  });
  const current = qs(`[data-current-value='${name}']`, root);
  if (current) {
    current.textContent = label || value || t("backtests.results.all");
  }
  if (name === "job_state") {
    refreshWorkstation(root, "manual").catch(() => {});
  }
  if (name === "risk_mode") {
    updateRiskPanel(root);
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
  renderDropdownOptions(root, "ranking_metric", (runtime.ranking_metrics || []).map((value) => ({
    value,
    label: labelForId(value),
  })));
  seedRiskPanel(root, runtime.hit_times_grid || {});
  seedConfigDraft(root, data?.config_draft || {});
  updateRiskPanel(root);
}

function seedConfigDraft(root, draft) {
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

function renderSymbols(root, universe) {
  const target = qs("[data-symbol-list]", root);
  const selectedTarget = qs("[data-selected-symbols]", root);
  const symbols = universe?.symbols || [];
  const selected = state.selectedSymbols?.size
    ? state.selectedSymbols
    : new Set(universe?.selected_symbols || ["BTCUSDT"]);
  state.selectedSymbols = new Set(selected);
  if (target) {
    target.innerHTML = symbols
      .map((symbol) => `
        <label class="backtests-symbol-row" data-symbol-row data-symbol-label="${escapeHtml(symbol.label)}">
          <input type="checkbox" value="${escapeHtml(symbol.value)}" data-symbol-checkbox ${selected.has(symbol.value) ? "checked" : ""}>
          <span>${escapeHtml(symbol.label)}</span>
          <small>${escapeHtml(symbol.status)}</small>
        </label>
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
              <td><button class="rh-button rh-button--secondary rh-button--compact" type="button" data-remove-indicator>&times;</button></td>
            </tr>
          `)
          .join("")
      : `<tr><td colspan="6">${escapeHtml(t("common.unavailable"))}</td></tr>`;
  }
  setText("[data-combinations-count]", indicatorCombinationCount(), root);
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
  renderJobPicker(root, rows);
  if (!rows.length) {
    target.innerHTML = `<tr><td colspan="15">${escapeHtml(table?.degradation_reason || t("backtests.results.empty"))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map(
      (row, index) => `
        <tr data-job-id="${escapeHtml(row.job_id)}" tabindex="0">
          <td>${state.selectedJobId === row.job_id ? "v" : ">"} ${index + 1}</td>
          <td>${escapeHtml(row.strategy)}</td>
          <td>${escapeHtml(formatDate(row.created_at))}</td>
          <td>${escapeHtml(row.symbol || "--")}</td>
          <td>${escapeHtml(row.indicator_summary)}</td>
          <td>${escapeHtml(row.period)}</td>
          <td>${escapeHtml(row.direction)}</td>
          <td>${numberOrDash(row.combinations)}</td>
          <td class="${financialClass(row.best_return_pct)}">${percent(row.best_return_pct)}</td>
          <td class="${financialClass(row.best_sharpe)}">${numberOrDash(row.best_sharpe)}</td>
          <td class="${financialClass(row.avg_drawdown_pct)}">${percent(row.avg_drawdown_pct)}</td>
          <td>${numberOrDash(row.profit_factor)}</td>
          <td>${percent(row.win_rate_pct)}</td>
          <td>${numberOrDash(row.trades_count)}</td>
          <td>
            <div class="backtests-status-cell">
              <span>${escapeHtml(row.state)} / ${row.progress_percent}%</span>
              ${row.actions?.can_cancel
                ? `<button class="rh-button rh-button--secondary rh-button--compact backtests-row-action" type="button" data-cancel-job-id="${escapeHtml(row.job_id)}">${escapeHtml(t("backtests.actions.cancel"))}</button>`
                : ""}
            </div>
          </td>
        </tr>
      `
    )
    .join("");
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
  const selectedVariant = (summary?.top_variants?.items || []).find(
    (item) => item.variant_key === selectedKey
  );
  const resultState = qs("[data-result-state]", root);
  if (resultState) {
    resultState.hidden = !summary;
  }
  setText("[data-result-job]", compactId(summary?.job?.job_id), root);
  setText("[data-result-variant]", compactId(selectedKey), root);
  setText("[data-result-return]", percent(selectedVariant?.summary_metrics?.total_return_pct), root);
  setText("[data-result-sharpe]", numberOrDash(selectedVariant?.summary_metrics?.sharpe), root);
  const csv = qs("[data-result-csv]", root);
  if (csv && summary?.job?.job_id && selectedKey) {
    csv.href = `${variantBaseEndpoint(root, summary.job.job_id, selectedKey)}/trades.csv`;
  }
  renderVariantStrip(root, summary?.top_variants?.items || [], selectedKey);
}

function renderVariantStrip(root, variants, selectedKey) {
  const target = qs("[data-result-variants]", root);
  if (!target) {
    return;
  }
  target.innerHTML = variants
    .map(
      (variant) => `
        <button
          class="rh-button ${variant.variant_key === selectedKey ? "rh-button--primary" : "rh-button--secondary"} rh-button--compact"
          type="button"
          data-result-variant-key="${escapeHtml(variant.variant_key)}"
        >
          #${variant.rank} ${escapeHtml(compactId(variant.variant_key))}
        </button>
      `
    )
    .join("");
}

function renderTrades(root, payload) {
  const target = qs("[data-trades-rows]", root);
  if (!target) {
    return;
  }
  const rows = payload?.items || [];
  target.innerHTML = rows.length
    ? rows
        .map(
          (trade) => `
            <tr>
              <td>${numberOrDash(trade.trade_index)}</td>
              <td>${escapeHtml(trade.side || trade.direction || "--")}</td>
              <td>${escapeHtml(localTime(trade.exit_timestamp))}</td>
              <td class="${financialClass(trade.net_pnl_quote)}">${numberOrDash(trade.net_pnl_quote)}</td>
              <td class="${financialClass(trade.return_pct)}">${percent(trade.return_pct)}</td>
            </tr>
          `
        )
        .join("")
    : `<tr><td colspan="5">${escapeHtml(t("backtests.results.empty"))}</td></tr>`;
  const pagination = payload?.pagination || {};
  state.tradesPage = Number(pagination.page || 1);
  state.tradesHasNext = Boolean(pagination.has_next);
  setText(
    "[data-trades-page]",
    `${numberOrDash(pagination.page)} / ${numberOrDash(Math.ceil((pagination.total || 0) / (pagination.page_size || 1)) || 1)}`,
    root
  );
  const previous = qs("[data-trades-prev]", root);
  const next = qs("[data-trades-next]", root);
  if (previous) {
    previous.disabled = !pagination.has_previous;
  }
  if (next) {
    next.disabled = !pagination.has_next;
  }
}

function renderChart(root, payload) {
  const canvas = qs("[data-result-chart]", root);
  qsa("[data-chart-kind]", root).forEach((button) => {
    const active = button.dataset.chartKind === (payload?.kind || state.chartKind);
    button.classList.toggle("rh-button--primary", active);
    button.classList.toggle("rh-button--secondary", !active);
  });
  const result = renderBacktestSeries(canvas, payload?.points || [], { kind: payload?.kind });
  canvas?.setAttribute("data-chart-nonblank", result.nonblank ? "true" : "false");
  setText(
    "[data-chart-status]",
    `${payload?.kind || state.chartKind}: ${payload?.returned_points || 0}/${payload?.source_points || 0}`,
    root
  );
}

function renderFooter(root, data) {
  const sources = data?.sources || [];
  const availableSources = sources.filter((source) => source.status === "available").length;
  const capital = data?.config_draft?.execution?.initial_cash_quote;
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
  setText("[data-footer-capital]", capital ? `${capital} USDT` : "--", document);
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

function renderWorkstation(root, data) {
  state.runtimeDefaults = data;
  manualRefreshRetrySeconds = Number(data?.retry_after_seconds || 0);
  renderRuntimeControls(root, data);
  renderSymbols(root, data?.instrument_universe);
  renderIndicators(root, data?.indicator_catalog);
  renderOptimization(root, data?.optimization_overview);
  renderJobs(root, data?.job_table);
  renderFooter(root, data);
  const loading = qs("[data-backtests-loading]", root);
  if (loading) {
    loading.hidden = true;
  }
}

async function loadResultSummary(root, jobId) {
  if (!jobId) {
    return null;
  }
  const template = root.dataset.jobSummaryEndpointTemplate || "/api/backtests/jobs/{job_id}/summary";
  const summary = await apiFetch(endpointFromTemplate(template, { job_id: jobId }));
  state.selectedJobId = summary.job?.job_id || jobId;
  if (!state.selectedVariantKey) {
    state.selectedVariantKey = summary.selected_variant_key;
  }
  renderResultSummary(root, summary);
  return summary;
}

async function loadChart(root) {
  if (!state.selectedJobId || !state.selectedVariantKey) {
    return;
  }
  const endpoint = `${variantBaseEndpoint(root, state.selectedJobId, state.selectedVariantKey)}/${state.chartKind}?points=600`;
  const payload = await apiFetch(endpoint);
  renderChart(root, payload);
}

async function loadTrades(root, page = 1) {
  if (!state.selectedJobId || !state.selectedVariantKey) {
    return;
  }
  const endpoint = `${variantBaseEndpoint(root, state.selectedJobId, state.selectedVariantKey)}/trades?page=${page}&page_size=50`;
  const payload = await apiFetch(endpoint);
  renderTrades(root, payload);
}

async function loadSelectedResult(root, { includeChart = true, includeTrades = true } = {}) {
  if (!state.selectedJobId || activeResultRequest) {
    return activeResultRequest;
  }
  activeResultRequest = loadResultSummary(root, state.selectedJobId)
    .then(async () => {
      if (includeChart) {
        await loadChart(root);
      }
      if (includeTrades) {
        await loadTrades(root, state.tradesPage);
      }
    })
    .finally(() => {
      activeResultRequest = null;
    });
  return activeResultRequest;
}

function selectJob(root, jobId) {
  state.selectedJobId = jobId || null;
  state.selectedVariantKey = null;
  state.tradesPage = 1;
  renderJobPicker(root, state.jobRows);
  loadSelectedResult(root).catch((error) => {
    setText("[data-chart-status]", error?.message || t("backtests.status.failed"), root);
  });
}

async function refreshWorkstation(root, reason = "manual") {
  if (activeRequest) {
    return activeRequest;
  }
  const endpoint = root.dataset.workstationEndpoint || DEFAULT_ENDPOINT;
  const params = new URLSearchParams();
  params.set("refresh", reason);
  if (state.job_state) {
    params.set("state", state.job_state);
  }
  if (state.cursor) {
    params.set("cursor", state.cursor);
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
      renderWorkstation(root, data);
      if (state.selectedJobId && reason !== "initial") {
        loadSelectedResult(root, { includeChart: false, includeTrades: false }).catch(() => {});
      }
      return data;
    })
    .finally(() => {
      activeRequest = null;
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
  setText("[data-backtests-refresh-current]", presetKey, root);
}

function bind(root) {
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
    const clearSymbols = event.target.closest("[data-clear-symbols]");
    if (clearSymbols instanceof HTMLElement) {
      qsa("[data-symbol-checkbox]", root).forEach((checkbox) => {
        checkbox.checked = false;
      });
      state.selectedSymbols = new Set();
      renderSelectedSymbols(root);
      return;
    }
    const addIndicatorButton = event.target.closest("[data-add-indicator]");
    if (addIndicatorButton instanceof HTMLElement) {
      addIndicator(root, addIndicatorButton.dataset.addIndicator || "");
      return;
    }
    const indicatorFamilyTab = event.target.closest("[data-indicator-family-tab]");
    if (indicatorFamilyTab instanceof HTMLElement) {
      state.indicatorFamily = indicatorFamilyTab.dataset.indicatorFamilyTab || state.indicatorFamily;
      renderIndicatorAddMenu(root, Array.from(state.indicatorCatalog.values()));
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
      state.tradesPage = 1;
      renderResultSummary(root, state.resultSummary);
      loadSelectedResult(root).catch(() => {});
      return;
    }
    const chartButton = event.target.closest("[data-chart-kind]");
    if (chartButton instanceof HTMLElement) {
      state.chartKind = chartButton.dataset.chartKind || "equity";
      loadChart(root).catch(() => {});
      return;
    }
    const prevTrades = event.target.closest("[data-trades-prev]");
    if (prevTrades instanceof HTMLElement && state.tradesPage > 1) {
      loadTrades(root, state.tradesPage - 1).catch(() => {});
      return;
    }
    const nextTrades = event.target.closest("[data-trades-next]");
    if (nextTrades instanceof HTMLElement && state.tradesHasNext) {
      loadTrades(root, state.tradesPage + 1).catch(() => {});
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
        setText("[data-combinations-count]", indicatorCombinationCount(), root);
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
