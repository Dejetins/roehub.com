import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { getCurrentLocale, t } from "../core/locale.js";
import { createPoller } from "../core/poller.js";
import { renderBacktestSeries } from "../charts/backtest_series.js";

const DEFAULT_ENDPOINT = "/api/ui/backtests/workstation";
const DEFAULT_VARIANT_OPEN_DELAY_MS = 140;
const DEFAULT_VARIANT_OPEN_DURATION_MS = 400;
const DEFAULT_VARIANT_PREVIEW_LIMIT = 5;
const DEFAULT_RESULT_POINTS = 600;
const DEFAULT_TRADES_PAGE_SIZE = 50;
const REFRESH_PRESETS = {
  off: 0,
  "10s": 10000,
  "15s": 15000,
  "30s": 30000,
  "1m": 60000,
  "5m": 300000,
};
const AI_DEFAULT_MODE = "create";
const AI_STATUS_STAGES = ["queued", "preparing_catalog", "generating", "validating", "repairing", "ready"];
const AI_TERMINAL_STATUSES = new Set([
  "ready",
  "needs_clarification",
  "blocked_by_policy",
  "input_too_large",
  "security_review",
  "failed",
]);
const AI_STATUS_ALIASES = {
  running: "generating",
  validating_business: "validating",
  validating_schema: "validating",
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
  symbolQuery: "",
  runtimeDefaults: null,
  selectedSymbols: new Set(["BTCUSDT"]),
  selectedIndicators: [],
  indicatorCatalog: new Map(),
  indicatorFamily: null,
  jobRows: [],
  loadedAllJobs: false,
  selectedJobId: null,
  selectedVariantKey: null,
  resultSummary: null,
  resultDetails: null,
  tradesPage: 1,
  animateVariantJobId: null,
  configSeeded: false,
  ai: {
    enabled: false,
    mode: AI_DEFAULT_MODE,
    activeJobId: null,
    activeToken: 0,
    readyJob: null,
    appliedJobId: null,
    status: "",
    isBusy: false,
    typewriterTimer: null,
  },
};

let activeRequest = null;
let activeResultRequest = null;
let activeVariantResultRequest = null;
let activeVariantResultKey = "";
let activeAiStatusRequest = null;
let aiEventSource = null;
let aiPoller = null;
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

function moneyOrDash(value, fractionDigits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return number.toFixed(fractionDigits);
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

function materializationStatus(payload) {
  return payload?.materialization?.status || payload?.status || "";
}

function isMaterializationPayload(payload) {
  return Boolean(payload?.materialization && materializationStatus(payload));
}

function resultPayloadState(payload) {
  if (!payload) {
    return "pending";
  }
  if (isMaterializationPayload(payload)) {
    const status = materializationStatus(payload);
    if (["queued", "running", "materializing"].includes(status)) {
      return "materializing";
    }
    return status || "materializing";
  }
  if (payload.cache?.status && payload.cache.status !== "hit") {
    return payload.cache.status === "miss" ? "materializing" : "degraded";
  }
  return "available";
}

function resultStatusText(payload, fallback = "pending") {
  if (!payload) {
    return fallback;
  }
  if (isMaterializationPayload(payload)) {
    const status = materializationStatus(payload) || "materializing";
    const retryAfter = Number(payload.materialization?.retry_after_seconds || 0);
    return retryAfter > 0 ? `${status}; retry ${retryAfter}s` : status;
  }
  if (payload.cache?.warning) {
    return `degraded: ${payload.cache.warning}`;
  }
  if (payload.cache?.status) {
    return `cache ${payload.cache.status}`;
  }
  return "available";
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

function aiConfigEndpoint(root, kind, jobId) {
  const templates = {
    job: root.dataset.aiConfigJobEndpointTemplate || "/api/backtests/ai-config/jobs/{job_id}",
    events: root.dataset.aiConfigEventsEndpointTemplate || "/api/backtests/ai-config/jobs/{job_id}/events",
    feedback: root.dataset.aiConfigFeedbackEndpointTemplate || "/api/backtests/ai-config/jobs/{job_id}/feedback",
  };
  return endpointFromTemplate(templates[kind], { job_id: jobId });
}

function normalizeAiStatus(value) {
  const normalized = String(value || "").trim();
  return AI_STATUS_ALIASES[normalized] || normalized;
}

function isAiTerminalStatus(value) {
  return AI_TERMINAL_STATUSES.has(normalizeAiStatus(value));
}

function stopAiJobUpdates() {
  if (aiEventSource) {
    aiEventSource.close();
    aiEventSource = null;
  }
  if (aiPoller) {
    aiPoller.stop();
    aiPoller = null;
  }
}

function clearAiTypewriter() {
  if (state.ai.typewriterTimer) {
    window.clearTimeout(state.ai.typewriterTimer);
    state.ai.typewriterTimer = null;
  }
}

function generateIdempotencyKey(prefix = "ai-config") {
  return window.crypto?.randomUUID
    ? window.crypto.randomUUID()
    : `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setAiControls(root) {
  const disabled = !state.ai.enabled || state.ai.isBusy;
  qsa("[data-ai-mode]", root).forEach((button) => {
    if (button instanceof HTMLButtonElement) {
      button.disabled = disabled;
    }
  });
  const prompt = qs("[data-ai-prompt]", root);
  if (prompt instanceof HTMLInputElement) {
    prompt.disabled = disabled;
  }
  const submit = qs("[data-ai-submit]", root);
  if (submit instanceof HTMLButtonElement) {
    submit.disabled = disabled;
  }
}

function setAiBusy(root, isBusy) {
  state.ai.isBusy = isBusy;
  setAiControls(root);
}

function selectAiMode(root, mode) {
  const nextMode = mode || AI_DEFAULT_MODE;
  state.ai.mode = nextMode;
  qsa("[data-ai-mode]", root).forEach((button) => {
    const isActive = button.dataset.aiMode === nextMode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-checked", isActive ? "true" : "false");
  });
}

function renderAiTimeline(root, currentStatus = "") {
  const timeline = qs("[data-ai-timeline]", root);
  if (!timeline) {
    return;
  }
  const normalized = normalizeAiStatus(currentStatus);
  const statuses = AI_STATUS_STAGES.includes(normalized)
    ? AI_STATUS_STAGES
    : normalized
      ? [...AI_STATUS_STAGES, normalized]
      : AI_STATUS_STAGES;
  const activeIndex = statuses.indexOf(normalized);
  const items = statuses.map((status, index) => {
    const item = document.createElement("li");
    item.dataset.aiStage = status;
    item.textContent = status;
    item.classList.toggle("is-active", status === normalized);
    item.classList.toggle(
      "is-complete",
      activeIndex >= 0 && index < activeIndex && AI_STATUS_STAGES.includes(status)
    );
    return item;
  });
  timeline.replaceChildren(...items);
}

function setAiStatus(root, status, message = "") {
  const normalized = normalizeAiStatus(status);
  state.ai.status = normalized;
  renderAiTimeline(root, normalized);
  setText(
    "[data-ai-status]",
    message || (normalized
      ? t("backtests.ai.status.active", { job: compactId(state.ai.activeJobId), status: normalized })
      : t("backtests.status.ready")),
    root
  );
}

function appendAiMessage(root, role, text) {
  const log = qs("[data-ai-log]", root);
  if (!log) {
    return null;
  }
  const article = document.createElement("article");
  article.dataset.aiMessageRole = role;

  const message = document.createElement("div");
  message.className = "backtests-ai-message";
  message.textContent = String(text || "");

  const time = document.createElement("time");
  time.dateTime = new Date().toISOString();
  time.textContent = localTime(time.dateTime);

  article.append(message, time);
  log.append(article);
  log.scrollTop = log.scrollHeight;
  return message;
}

function appendAiAssistantResult(root, job) {
  clearAiTypewriter();
  const message = appendAiMessage(root, "assistant", "");
  if (!message) {
    return;
  }
  const finalText = String(job?.assistant_message || job?.message || "");
  let index = 0;
  const step = () => {
    message.textContent = finalText.slice(0, index);
    index += Math.max(1, Math.ceil(finalText.length / 48));
    if (index <= finalText.length) {
      state.ai.typewriterTimer = window.setTimeout(step, 24);
      return;
    }
    message.textContent = finalText;
    state.ai.typewriterTimer = null;
    renderAiLoadAction(root, message.closest("article"), job);
  };
  step();
}

function renderAiLoadAction(root, article, job) {
  if (
    !article ||
    normalizeAiStatus(job?.status) !== "ready" ||
    job?.load_action?.enabled !== true ||
    !job?.validated_config
  ) {
    return;
  }
  const actions = document.createElement("div");
  actions.className = "backtests-ai-actions";
  const button = document.createElement("button");
  button.className = "rh-button rh-button--primary rh-button--compact backtests-ai-load";
  button.type = "button";
  button.dataset.aiLoadConfiguration = job.job_id;
  button.textContent = t("backtests.ai.load_configuration");
  actions.append(button);
  article.append(actions);
}

function renderAiConfiguratorState(root, aiState) {
  state.ai.enabled = Boolean(aiState?.enabled);
  const stateLabel = qs("[data-ai-state-label]", root);
  if (stateLabel) {
    stateLabel.textContent = state.ai.enabled ? t("backtests.ai.beta") : String(aiState?.state || "disabled");
  }
  const modes = new Set((aiState?.modes || []).map((mode) => mode?.value).filter(Boolean));
  qsa("[data-ai-mode]", root).forEach((button) => {
    button.hidden = modes.size > 0 && !modes.has(button.dataset.aiMode || "");
  });
  if (!modes.has(state.ai.mode)) {
    selectAiMode(root, modes.has(AI_DEFAULT_MODE) || !modes.size ? AI_DEFAULT_MODE : Array.from(modes)[0]);
  } else {
    selectAiMode(root, state.ai.mode);
  }
  setAiControls(root);
  if (!state.ai.enabled && !state.ai.isBusy) {
    setText(
      "[data-ai-status]",
      aiState?.degradation_reason || t("backtests.ai.disabled"),
      root
    );
    renderAiTimeline(root, "");
  }
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

function variantEndpoint(root, kind, jobId, variantKey, params = {}) {
  const templates = {
    variant:
      root.dataset.variantEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}",
    equity:
      root.dataset.variantEquityEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/equity",
    drawdown:
      root.dataset.variantDrawdownEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
    monthly:
      root.dataset.variantMonthlyEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
    symbol:
      root.dataset.variantSymbolEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
    trades:
      root.dataset.variantTradesEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades?page={page}&page_size={page_size}",
    csv:
      root.dataset.variantTradesCsvEndpointTemplate ||
      "/api/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv",
  };
  return endpointFromTemplate(templates[kind], {
    job_id: jobId,
    variant_key: variantKey,
    ...params,
  });
}

function appendSearchParams(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });
  return `${url.pathname}${url.search}`;
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

function optionLabelForValue(root, name, value) {
  const option = qsa(`[data-backtest-option='${name}']`, root)
    .find((item) => item.dataset.value === value);
  return (option?.textContent || "").trim() || value || t("backtests.results.all");
}

function hasDropdownOption(root, name, value) {
  return qsa(`[data-backtest-option='${name}']`, root)
    .some((option) => option.dataset.value === value);
}

function setDropdownValue(root, name, value, { label = "", validateOptions = false, refresh = false } = {}) {
  if (value === undefined || value === null || value === "") {
    return false;
  }
  const normalized = String(value);
  if (validateOptions && !hasDropdownOption(root, name, normalized)) {
    return false;
  }
  updateOptionSelection(root, name, normalized, label || optionLabelForValue(root, name, normalized), { refresh });
  return true;
}

function updateOptionSelection(root, name, value, label, { refresh = true } = {}) {
  state[name] = value;
  qsa(`[data-backtest-option='${name}']`, root).forEach((option) => {
    option.setAttribute("aria-selected", option.dataset.value === value ? "true" : "false");
  });
  qsa(`[data-current-value='${name}']`, root).forEach((current) => {
    current.textContent = label || value || t("backtests.results.all");
  });
  if (refresh && ["job_state", "job_exchange", "job_market_type"].includes(name)) {
    refreshWorkstation(root, "manual").catch(() => {});
  }
  if (refresh && ["market", "market_type"].includes(name)) {
    state.symbol = "";
    state.selectedSymbols = new Set();
    refreshWorkstation(root, "auto").catch(() => {});
  }
  if (name === "risk_mode") {
    updateRiskPanel(root);
  }
  if (name === "sizing_mode") {
    updateSizingPanel(root);
  }
}

function renderDropdownOptions(root, name, options) {
  const menus = qsa(`[data-backtest-menu='${name}']`, root);
  const fallback = qs(`#backtest-${name}-menu`, root);
  const targets = menus.length ? menus : fallback ? [fallback] : [];
  if (!targets.length || !Array.isArray(options) || !options.length) {
    return;
  }
  const selectedValue = options.some((option) => option.value === state[name])
    ? state[name]
    : options[0].value;
  const selectedLabel = options.find((option) => option.value === selectedValue)?.label || selectedValue;
  const markup = options
    .map((option) => `
      <button
        class="rh-menu-item"
        type="button"
        role="option"
        aria-selected="${option.value === selectedValue ? "true" : "false"}"
        data-backtest-option="${escapeHtml(name)}"
        data-value="${escapeHtml(option.value)}"
      >${escapeHtml(option.label || option.value)}</button>
    `)
    .join("");
  targets.forEach((menu) => {
    menu.innerHTML = markup;
  });
  if (selectedValue !== state[name]) {
    updateOptionSelection(root, name, selectedValue, selectedLabel);
  } else {
    qsa(`[data-current-value='${name}']`, root).forEach((current) => {
      current.textContent = selectedLabel || selectedValue || t("backtests.results.all");
    });
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
  if (!state.configSeeded) {
    seedConfigDraft(root, data?.config_draft || {});
    state.configSeeded = true;
  }
  updateRiskPanel(root);
  updateSizingPanel(root);
}

function seedConfigDraft(root, draft, { validateOptions = false, includeIndicators = false } = {}) {
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
  if (draft?.coordinates?.symbol) {
    state.symbol = String(draft.coordinates.symbol).toUpperCase();
    state.selectedSymbols = new Set([state.symbol]);
  }
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
  setDropdownValue(root, "market", draft?.coordinates?.exchange, { validateOptions });
  setDropdownValue(root, "market_type", draft?.coordinates?.market_type, { validateOptions });
  setDropdownValue(root, "timeframe", draft?.timeframe, {
    label: draft?.timeframe,
    validateOptions,
  });
  setDropdownValue(root, "direction", draft?.execution?.direction_mode, { validateOptions });
  setDropdownValue(root, "sizing_mode", sizing?.mode, {
    label: sizing?.mode ? sizingModeLabel(sizing.mode) : "",
    validateOptions,
  });
  setDropdownValue(root, "ranking_metric", draft?.ranking?.primary_metric, { validateOptions });
  setDropdownValue(root, "ranking_order", draft?.ranking?.direction, { validateOptions });
  if (setDropdownValue(root, "risk_mode", draft?.risk?.mode, {
    label: draft?.risk?.mode === "tp_sl_grid" ? t("backtests.option.tp_sl_grid") : t("backtests.option.no_risk"),
    validateOptions,
  })) {
    const tp = draft?.risk?.tp || {};
    const sl = draft?.risk?.sl || {};
    const riskValues = {
      tp_start: tp.start_pct,
      tp_stop: tp.stop_pct,
      tp_step: tp.step_pct,
      sl_start: sl.start_pct,
      sl_stop: sl.stop_pct,
      sl_step: sl.step_pct,
    };
    Object.entries(riskValues).forEach(([name, value]) => {
      const field = qs(`[data-risk-field='${name}']`, root);
      if (field && value !== undefined && value !== null && value !== "") {
        field.value = String(value);
      }
    });
  }
  if (includeIndicators && Array.isArray(draft?.indicators)) {
    state.selectedIndicators = draft.indicators
      .map((indicator) => indicatorStateFromDraft(indicator))
      .filter(Boolean);
    renderIndicators(root, { items: Array.from(state.indicatorCatalog.values()) });
  }
  renderSelectedSymbols(root);
  updateRiskPanel(root);
  updateSizingPanel(root);
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
        </button>
      `)
      .join("");
    filterSymbols(root, state.symbolQuery);
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
  renderResultCanvases(root);
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
    : "backtests-variant-frame backtests-variant-frame--static";
  const body = variants.length
    ? variants.map((variant) => renderVariantRow(root, row.job_id, variant)).join("")
    : `<tr><td colspan="10">${escapeHtml(activeResultRequest ? t("backtests.variants.loading") : variantEmptyText(row, summary))}</td></tr>`;
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
            ${variants.length ? renderSelectedVariantDetail(root, row.job_id, variants) : renderVariantEmptyDetail(row, summary)}
          </section>
        </div>
      </td>
    </tr>
  `;
}

function variantEmptyText(row, summary) {
  if (isQualityGateEmptyResult(row, summary)) {
    return t("backtests.variants.none_passed_quality_gate");
  }
  return t("backtests.variants.empty");
}

function renderVariantEmptyDetail(row, summary) {
  const message = isQualityGateEmptyResult(row, summary)
    ? t("backtests.variants.none_passed_quality_gate")
    : t("backtests.results.unavailable");
  return `
    <section class="backtests-result-detail" data-result-state="unavailable">
      <div class="backtests-result-state">${escapeHtml(message)}</div>
    </section>
  `;
}

function isQualityGateEmptyResult(row, summary) {
  const topCount = Number(summary?.job?.terminal_summary?.top_variants_count);
  return row?.state === "succeeded" && summary && Number.isFinite(topCount) && topCount === 0;
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
    renderResultCanvases(root);
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

function selectedVariant(summaryVariants) {
  if (!state.selectedVariantKey && summaryVariants.length) {
    state.selectedVariantKey = summaryVariants[0].variant_key;
  }
  return summaryVariants.find((variant) => variant.variant_key === state.selectedVariantKey) || summaryVariants[0] || null;
}

function renderSelectedVariantDetail(root, jobId, summaryVariants) {
  const variant = selectedVariant(summaryVariants);
  const variantKey = variant?.variant_key || state.selectedVariantKey;
  if (!variantKey) {
    return `
      <section class="backtests-result-detail" data-result-state="unavailable">
        <div class="backtests-result-state">${escapeHtml(t("backtests.results.unavailable"))}</div>
      </section>
    `;
  }
  const details =
    state.resultDetails?.jobId === jobId && state.resultDetails?.variantKey === variantKey
      ? state.resultDetails
      : null;
  const stateName = details?.state || "pending";
  const csvHref = variantEndpoint(root, "csv", jobId, variantKey);
  return `
    <section class="backtests-result-detail" data-result-state="${escapeHtml(stateName)}" data-selected-result-variant="${escapeHtml(variantKey)}">
      <header class="backtests-result-detail__heading">
        <div>
          <strong>${escapeHtml(t("backtests.result_detail.title", { variant: compactId(variantKey) }))}</strong>
          <span>${escapeHtml(resultDetailStatus(details))}</span>
        </div>
        <div class="backtests-result-actions">
          <button class="rh-button rh-button--secondary rh-button--compact" type="button" data-result-refresh>${escapeHtml(t("refresh.manual"))}</button>
          <a class="rh-button rh-button--secondary rh-button--compact backtests-download-button" href="${escapeHtml(csvHref)}" data-result-csv>${escapeHtml(t("backtests.variants.csv"))}</a>
        </div>
      </header>
      <div class="backtests-result-metrics">
        ${renderResultMetric(t("backtests.results.return"), percent(variant?.summary_metrics?.total_return_pct), financialClass(variant?.summary_metrics?.total_return_pct))}
        ${renderResultMetric(t("backtests.results.sharpe"), numberOrDash(variant?.summary_metrics?.sharpe), financialClass(variant?.summary_metrics?.sharpe))}
        ${renderResultMetric(t("backtests.results.drawdown"), signedDrawdownPercent(variant?.summary_metrics?.max_drawdown_pct ?? variant?.summary_metrics?.avg_drawdown_pct), "rh-financial--negative")}
        ${renderResultMetric(t("backtests.results.trades"), integerOrDash(variant?.summary_metrics?.trade_count ?? variant?.summary_metrics?.trades_count), "")}
      </div>
      <div class="backtests-result-body">
        <div class="backtests-result-charts">
          ${renderChartShell("equity", t("backtests.result_detail.equity"), details?.equity)}
          ${renderChartShell("drawdown", t("backtests.result_detail.drawdown"), details?.drawdown)}
        </div>
        <div class="backtests-result-stat-tables">
          ${renderStatsTable("monthly", t("backtests.result_detail.monthly"), details?.monthly)}
          ${renderStatsTable("symbol", t("backtests.result_detail.symbol"), details?.symbol)}
        </div>
        ${renderTradesPanel(details)}
      </div>
    </section>
  `;
}

function renderResultMetric(label, value, className) {
  return `
    <div>
      <span>${escapeHtml(label)}</span>
      <strong class="${escapeHtml(className)}">${escapeHtml(value)}</strong>
    </div>
  `;
}

function resultDetailStatus(details) {
  if (!details) {
    return t("backtests.result_detail.pending");
  }
  if (details.message) {
    return details.message;
  }
  return details.state || "available";
}

function renderChartShell(kind, label, payload) {
  const stateName = resultPayloadState(payload);
  const points = Array.isArray(payload?.points) ? payload.points.length : 0;
  const unavailable = !payload || isMaterializationPayload(payload) || points === 0;
  return `
    <section class="backtests-result-chart-panel" data-result-chart-panel="${escapeHtml(kind)}" data-result-state="${escapeHtml(stateName)}">
      <header>
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(unavailable ? resultStatusText(payload, t("backtests.result_detail.pending")) : `${points} pts`)}</span>
      </header>
      <canvas class="backtests-result-chart" data-result-chart="${escapeHtml(kind)}" width="420" height="170"></canvas>
    </section>
  `;
}

function renderStatsTable(kind, label, payload) {
  const items = Array.isArray(payload?.items) ? payload.items.slice(0, 6) : [];
  const body = items.length
    ? items.map((item) => renderStatsRow(kind, item)).join("")
    : `<tr><td colspan="4">${escapeHtml(resultStatusText(payload, t("backtests.results.unavailable")))}</td></tr>`;
  return `
    <section class="backtests-result-stats" data-result-stats="${escapeHtml(kind)}" data-result-state="${escapeHtml(resultPayloadState(payload))}">
      <header>
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(resultStatusText(payload, t("backtests.result_detail.pending")))}</span>
      </header>
      <div class="backtests-table-wrap backtests-result-stats-wrap">
        <table class="backtests-table backtests-table--result-stats">
          <tbody>${body}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderStatsRow(kind, item) {
  if (kind === "monthly") {
    return `
      <tr>
        <td>${escapeHtml(item.month || item.period || "--")}</td>
        <td class="${financialClass(item.total_pnl ?? item.pnl)}">${escapeHtml(moneyOrDash(item.total_pnl ?? item.pnl))}</td>
        <td>${escapeHtml(integerOrDash(item.trades_count ?? item.trade_count))}</td>
        <td>${escapeHtml(percent(item.win_rate_pct, 1))}</td>
      </tr>
    `;
  }
  return `
    <tr>
      <td>${escapeHtml(item.symbol || "--")}</td>
      <td class="${financialClass(item.total_pnl ?? item.pnl)}">${escapeHtml(moneyOrDash(item.total_pnl ?? item.pnl))}</td>
      <td>${escapeHtml(integerOrDash(item.trades_count ?? item.trade_count))}</td>
      <td>${escapeHtml(percent(item.win_rate_pct, 1))}</td>
    </tr>
  `;
}

function renderTradesPanel(details) {
  const payload = details?.trades;
  const items = Array.isArray(payload?.items) ? payload.items : [];
  const pagination = payload?.pagination || {};
  const page = Number(pagination.page || state.tradesPage || 1);
  const totalPages = Number(pagination.total_pages || 1);
  const body = items.length
    ? items.map((item) => renderTradeRow(item)).join("")
    : `<tr><td colspan="6">${escapeHtml(resultStatusText(payload, t("backtests.results.unavailable")))}</td></tr>`;
  return `
    <section class="backtests-result-trades" data-result-trades data-result-state="${escapeHtml(resultPayloadState(payload))}">
      <header>
        <strong>${escapeHtml(t("backtests.result_detail.trades"))}</strong>
        <span>${escapeHtml(t("backtests.result_detail.page", { page, total: totalPages }))}</span>
      </header>
      <div class="backtests-table-wrap backtests-result-trades-wrap">
        <table class="backtests-table backtests-table--result-trades">
          <thead>
            <tr>
              <th>${escapeHtml(t("backtests.result_detail.entry"))}</th>
              <th>${escapeHtml(t("backtests.result_detail.exit"))}</th>
              <th>${escapeHtml(t("backtests.result_detail.side"))}</th>
              <th>${escapeHtml(t("backtests.result_detail.qty"))}</th>
              <th>${escapeHtml(t("backtests.result_detail.pnl"))}</th>
              <th>${escapeHtml(t("backtests.result_detail.reason"))}</th>
            </tr>
          </thead>
          <tbody data-trades-rows>${body}</tbody>
        </table>
      </div>
      <footer class="backtests-result-pagination">
        <button class="rh-button rh-button--secondary rh-button--compact" type="button" data-trades-page="prev" ${page <= 1 ? "disabled" : ""}>${escapeHtml(t("pagination.previous"))}</button>
        <button class="rh-button rh-button--secondary rh-button--compact" type="button" data-trades-page="next" ${page >= totalPages ? "disabled" : ""}>${escapeHtml(t("pagination.next"))}</button>
      </footer>
    </section>
  `;
}

function renderTradeRow(item) {
  return `
    <tr>
      <td>${escapeHtml(formatDateTimeMinute(item.entry_time || item.opened_at || item.timestamp))}</td>
      <td>${escapeHtml(formatDateTimeMinute(item.exit_time || item.closed_at))}</td>
      <td>${escapeHtml(item.side || item.direction || "--")}</td>
      <td>${escapeHtml(decimalOrDash(item.qty ?? item.quantity ?? item.size, 4))}</td>
      <td class="${financialClass(item.pnl ?? item.pnl_quote)}">${escapeHtml(moneyOrDash(item.pnl ?? item.pnl_quote))}</td>
      <td>${escapeHtml(item.exit_reason || item.reason || "--")}</td>
    </tr>
  `;
}

function renderResultCanvases(root) {
  const details = state.resultDetails;
  if (!details) {
    return;
  }
  qsa("[data-result-chart]", root).forEach((canvas) => {
    const kind = canvas.dataset.resultChart;
    const payload = kind === "drawdown" ? details.drawdown : details.equity;
    const result = renderBacktestSeries(canvas, payload?.points || [], { kind });
    canvas.dataset.nonblank = result.nonblank ? "true" : "false";
    canvas.dataset.pointCount = String(result.points || 0);
  });
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

function updateResultRetryHint(payloads) {
  const retryAfter = payloads
    .map((payload) => Number(payload?.materialization?.retry_after_seconds || payload?.retry_after_seconds || 0))
    .filter((value) => Number.isFinite(value) && value > 0);
  if (retryAfter.length) {
    manualRefreshRetrySeconds = Math.max(manualRefreshRetrySeconds, ...retryAfter);
  }
}

function summarizeResultState(payloads, rejectedCount) {
  const fulfilled = payloads.filter(Boolean);
  if (!fulfilled.length && rejectedCount > 0) {
    return {
      state: "unavailable",
      message: t("backtests.result_detail.unavailable"),
    };
  }
  const states = fulfilled.map(resultPayloadState);
  if (states.includes("materializing")) {
    return {
      state: "materializing",
      message: t("backtests.result_detail.materializing"),
    };
  }
  if (rejectedCount > 0 || states.includes("degraded")) {
    return {
      state: "degraded",
      message: t("backtests.result_detail.degraded"),
    };
  }
  if (states.every((item) => item === "pending")) {
    return {
      state: "pending",
      message: t("backtests.result_detail.pending"),
    };
  }
  return {
    state: "available",
    message: t("backtests.result_detail.available"),
  };
}

async function loadSelectedVariantDetails(root, { page = state.tradesPage || 1 } = {}) {
  const jobId = state.selectedJobId;
  const variantKey = state.selectedVariantKey;
  if (!jobId || !variantKey) {
    return null;
  }
  const requestKey = `${jobId}:${variantKey}:${page}`;
  if (activeVariantResultRequest && activeVariantResultKey === requestKey) {
    return activeVariantResultRequest;
  }
  state.resultDetails = {
    jobId,
    variantKey,
    state: "pending",
    message: t("backtests.result_detail.pending"),
  };
  renderJobs(root, { items: state.jobRows });
  const tradesPath = variantEndpoint(root, "trades", jobId, variantKey, {
    page,
    page_size: DEFAULT_TRADES_PAGE_SIZE,
  });
  const requests = {
    variant: apiFetch(variantEndpoint(root, "variant", jobId, variantKey)),
    equity: apiFetch(appendSearchParams(variantEndpoint(root, "equity", jobId, variantKey), { points: DEFAULT_RESULT_POINTS })),
    drawdown: apiFetch(appendSearchParams(variantEndpoint(root, "drawdown", jobId, variantKey), { points: DEFAULT_RESULT_POINTS })),
    monthly: apiFetch(variantEndpoint(root, "monthly", jobId, variantKey)),
    symbol: apiFetch(variantEndpoint(root, "symbol", jobId, variantKey)),
    trades: apiFetch(tradesPath),
  };
  const request = Promise.allSettled(Object.values(requests))
    .then((results) => {
      if (state.selectedJobId !== jobId || state.selectedVariantKey !== variantKey) {
        return state.resultDetails;
      }
      const keys = Object.keys(requests);
      const detail = { jobId, variantKey };
      let rejectedCount = 0;
      results.forEach((result, index) => {
        const key = keys[index];
        if (result.status === "fulfilled") {
          detail[key] = result.value;
        } else {
          rejectedCount += 1;
          detail[`${key}Error`] = result.reason;
        }
      });
      const fulfilledPayloads = keys.map((key) => detail[key]).filter(Boolean);
      updateResultRetryHint(fulfilledPayloads);
      const summary = summarizeResultState(fulfilledPayloads, rejectedCount);
      state.resultDetails = {
        ...detail,
        state: summary.state,
        message: summary.message,
      };
      state.tradesPage = page;
      renderJobs(root, { items: state.jobRows });
      return state.resultDetails;
    })
    .finally(() => {
      if (activeVariantResultRequest === request) {
        activeVariantResultRequest = null;
        activeVariantResultKey = "";
      }
    });
  activeVariantResultRequest = request;
  activeVariantResultKey = requestKey;
  return activeVariantResultRequest;
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
  renderAiConfiguratorState(root, data?.ai_configurator_state || {});
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
  activeResultRequest = loadResultSummary(root, state.selectedJobId)
    .then(() => loadSelectedVariantDetails(root))
    .finally(() => {
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
    state.resultDetails = null;
    state.tradesPage = 1;
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
    await loadSelectedVariantDetails(root, { page: 1 });
  } catch (error) {
    setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
  } finally {
    activeResultRequest = null;
  }
}

function selectJob(root, jobId, { delayed = true } = {}) {
  clearDelayedVariantOpen();
  if (jobId && jobId === state.selectedJobId) {
    openSelectedJob(root, null);
    return;
  }
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
  if (state.market) {
    params.set("instrument_exchange", state.market);
  }
  if (state.market_type) {
    params.set("instrument_market_type", state.market_type);
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

function currentAiPayload(root, message, idempotencyKey) {
  return {
    mode: state.ai.mode || AI_DEFAULT_MODE,
    locale: getCurrentLocale(),
    message,
    idempotency_key: idempotencyKey,
    current_config: buildRequestPayload(root),
    ui_context: {
      source_page: "backtests",
      submitted_at: new Date().toISOString(),
      workstation_generated_at: state.runtimeDefaults?.generated_at || null,
      refresh_status: state.runtimeDefaults?.refresh_status || null,
    },
  };
}

function handleAiEventPayload(root, payload, token) {
  if (token !== state.ai.activeToken) {
    return;
  }
  const status = normalizeAiStatus(payload?.status || payload?.event || payload?.state);
  if (status) {
    setAiStatus(root, status, payload?.message || "");
  }
  if (isAiTerminalStatus(status)) {
    loadAiJobSnapshot(root, state.ai.activeJobId, token).catch((error) => {
      setAiStatus(root, "failed", describeApiError(error));
      setAiBusy(root, false);
    });
  }
}

function handleAiJobSnapshot(root, job, token) {
  if (token !== state.ai.activeToken || !job) {
    return;
  }
  const status = normalizeAiStatus(job.status);
  setAiStatus(root, status, job.message || job.assistant_message || "");
  if (!isAiTerminalStatus(status)) {
    return;
  }
  stopAiJobUpdates();
  setAiBusy(root, false);
  if (status === "ready" && job.load_action?.enabled === true && job.validated_config) {
    state.ai.readyJob = job;
    state.ai.appliedJobId = null;
    setText("[data-ai-status]", t("backtests.ai.status.ready"), root);
    appendAiAssistantResult(root, job);
    return;
  }
  state.ai.readyJob = null;
  appendAiMessage(root, "assistant", job.assistant_message || status || t("backtests.ai.status.error"));
}

async function loadAiJobSnapshot(root, jobId, token = state.ai.activeToken) {
  if (!jobId || activeAiStatusRequest) {
    return activeAiStatusRequest;
  }
  const endpoint = aiConfigEndpoint(root, "job", jobId);
  activeAiStatusRequest = apiFetch(endpoint)
    .then((job) => {
      handleAiJobSnapshot(root, job, token);
      return job;
    })
    .finally(() => {
      activeAiStatusRequest = null;
    });
  return activeAiStatusRequest;
}

function startAiPolling(root, jobId, token) {
  if (aiPoller) {
    aiPoller.stop();
  }
  aiPoller = createPoller(
    () => loadAiJobSnapshot(root, jobId, token),
    { intervalMs: 1500, hiddenTabPause: false }
  );
  aiPoller.start();
}

function parseAiSseEvent(event) {
  try {
    return JSON.parse(event.data || "{}");
  } catch {
    return {};
  }
}

function startAiJobUpdates(root, jobId, token, eventsUrl = "") {
  stopAiJobUpdates();
  const eventEndpoint = eventsUrl || aiConfigEndpoint(root, "events", jobId);
  if (!("EventSource" in window)) {
    startAiPolling(root, jobId, token);
    return;
  }
  let opened = false;
  try {
    aiEventSource = new EventSource(eventEndpoint, { withCredentials: true });
    const eventNames = new Set([
      ...AI_STATUS_STAGES,
      ...AI_TERMINAL_STATUSES,
      ...Object.keys(AI_STATUS_ALIASES),
      "capacity_delayed",
    ]);
    eventNames.forEach((eventName) => {
      aiEventSource.addEventListener(eventName, (event) => {
        opened = true;
        handleAiEventPayload(root, parseAiSseEvent(event), token);
      });
    });
    aiEventSource.onmessage = (event) => {
      opened = true;
      handleAiEventPayload(root, parseAiSseEvent(event), token);
    };
    aiEventSource.onerror = () => {
      if (token !== state.ai.activeToken || isAiTerminalStatus(state.ai.status)) {
        stopAiJobUpdates();
        return;
      }
      if (aiEventSource) {
        aiEventSource.close();
        aiEventSource = null;
      }
      if (!opened) {
        setAiStatus(root, state.ai.status || "queued");
      }
      startAiPolling(root, jobId, token);
    };
  } catch {
    startAiPolling(root, jobId, token);
  }
}

async function submitAiPrompt(root) {
  if (!state.ai.enabled || state.ai.isBusy) {
    return;
  }
  const prompt = qs("[data-ai-prompt]", root);
  const message = prompt instanceof HTMLInputElement ? prompt.value.trim() : "";
  if (!message) {
    return;
  }
  const token = state.ai.activeToken + 1;
  const idempotencyKey = generateIdempotencyKey();
  state.ai.activeToken = token;
  state.ai.activeJobId = null;
  state.ai.readyJob = null;
  state.ai.appliedJobId = null;
  clearAiTypewriter();
  stopAiJobUpdates();
  renderAiTimeline(root, "queued");
  setAiBusy(root, true);
  setText("[data-ai-feedback-status]", "", root);
  appendAiMessage(root, "user", message);
  if (prompt instanceof HTMLInputElement) {
    prompt.value = "";
  }
  try {
    const created = await apiFetch(root.dataset.aiConfigJobsEndpoint || "/api/backtests/ai-config/jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify(currentAiPayload(root, message, idempotencyKey)),
    });
    if (token !== state.ai.activeToken) {
      return;
    }
    if (!created?.job_id) {
      setAiBusy(root, false);
      setAiStatus(root, created?.status || "capacity_delayed", created?.message || "");
      appendAiMessage(root, "assistant", created?.message || t("backtests.ai.status.error"));
      return;
    }
    state.ai.activeJobId = created.job_id;
    setAiStatus(root, created.status || "queued", created.message || "");
    startAiJobUpdates(root, created.job_id, token, created.events_url || "");
    loadAiJobSnapshot(root, created.job_id, token).catch(() => {});
  } catch (error) {
    if (token !== state.ai.activeToken) {
      return;
    }
    setAiBusy(root, false);
    setAiStatus(root, "failed", describeApiError(error));
    appendAiMessage(root, "system", describeApiError(error));
  }
}

async function recordAiFeedback(root, jobId, applied) {
  const endpoint = aiConfigEndpoint(root, "feedback", jobId);
  return apiFetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      applied,
      reason: applied ? "loaded_configuration" : "replaced",
      client_context: {
        surface: "backtests",
        mode: state.ai.mode,
      },
    }),
  });
}

async function applyAiConfiguration(root, jobId) {
  const job = state.ai.readyJob;
  if (
    !job ||
    job.job_id !== jobId ||
    normalizeAiStatus(job.status) !== "ready" ||
    job.load_action?.enabled !== true ||
    !job.validated_config
  ) {
    return;
  }
  seedConfigDraft(root, job.validated_config, { validateOptions: true, includeIndicators: true });
  setText("[data-ai-status]", t("backtests.ai.status.loaded"), root);
  try {
    await recordAiFeedback(root, jobId, true);
    state.ai.appliedJobId = jobId;
    setText("[data-ai-feedback-status]", t("backtests.ai.feedback.applied"), root);
  } catch {
    setText("[data-ai-feedback-status]", t("backtests.ai.feedback.failed"), root);
  }
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
        return;
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
    const aiMode = event.target.closest("[data-ai-mode]");
    if (aiMode instanceof HTMLElement) {
      selectAiMode(root, aiMode.dataset.aiMode || AI_DEFAULT_MODE);
      return;
    }
    const aiSubmit = event.target.closest("[data-ai-submit]");
    if (aiSubmit instanceof HTMLElement) {
      submitAiPrompt(root).catch(() => {});
      return;
    }
    const aiLoad = event.target.closest("[data-ai-load-configuration]");
    if (aiLoad instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      applyAiConfiguration(root, aiLoad.dataset.aiLoadConfiguration || "").catch(() => {});
      return;
    }
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
        return;
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
      state.tradesPage = 1;
      state.resultDetails = null;
      renderJobs(root, { items: state.jobRows });
      loadSelectedVariantDetails(root, { page: 1 }).catch(() => {});
      return;
    }
    const resultRefresh = event.target.closest("[data-result-refresh]");
    if (resultRefresh instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      if (manualRefreshRetrySeconds > 0) {
        setText("[data-backtests-freshness]", t("dashboard.refresh.rate_limited", { seconds: manualRefreshRetrySeconds }));
        return;
      }
      loadSelectedVariantDetails(root, { page: state.tradesPage || 1 }).catch(() => {});
      return;
    }
    const tradesPage = event.target.closest("[data-trades-page]");
    if (tradesPage instanceof HTMLElement) {
      event.preventDefault();
      event.stopPropagation();
      const pagination = state.resultDetails?.trades?.pagination || {};
      const current = Number(pagination.page || state.tradesPage || 1);
      const total = Number(pagination.total_pages || 1);
      const nextPage = tradesPage.dataset.tradesPage === "next"
        ? Math.min(total, current + 1)
        : Math.max(1, current - 1);
      if (nextPage !== current) {
        loadSelectedVariantDetails(root, { page: nextPage }).catch(() => {});
      }
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
    state.symbolQuery = event.target.value || "";
    filterSymbols(root, state.symbolQuery);
  });
  qs("[data-ai-prompt]", root)?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitAiPrompt(root).catch(() => {});
    }
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
