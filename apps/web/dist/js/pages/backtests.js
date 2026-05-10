import { apiFetch } from "../core/api.js";
import { qs, qsa, setText } from "../core/dom.js";
import { t } from "../core/locale.js";
import { createPoller } from "../core/poller.js";

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
  cursor: null,
  query: "",
  runtimeDefaults: null,
};

let activeRequest = null;
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
  return selected.length ? selected : [state.symbol || "BTCUSDT"];
}

function buildRequestPayload(root) {
  const start = qs("[data-config-field='start']", root)?.value || "2023-01-01";
  const end = qs("[data-config-field='end']", root)?.value || "2024-01-01";
  const capital = Number(qs("[data-config-field='capital']", root)?.value || 10000);
  const feePercent = Number(qs("[data-config-field='fee']", root)?.value || 0.075);
  const slippagePercent = Number(qs("[data-config-field='slippage']", root)?.value || 0.01);
  const indicators = (state.runtimeDefaults?.config_draft?.indicators || [
    {
      indicator_id: "ma.dema",
      sources: ["close"],
      window: { start: 5, stop: 30, step: 2 },
    },
  ]).slice(0, 4);

  return {
    coordinates: {
      exchange: state.market,
      market_type: state.market_type,
      symbol: selectedSymbols(root)[0],
    },
    timeframe: state.timeframe,
    time_range: {
      start: dateToIso(start, "2023-01-01T00:00:00Z"),
      end: dateToIso(end, "2024-01-01T00:00:00Z"),
    },
    indicators,
    risk: { mode: state.risk_mode },
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
}

function renderSymbols(root, universe) {
  const target = qs("[data-symbol-list]", root);
  const selectedTarget = qs("[data-selected-symbols]", root);
  const symbols = universe?.symbols || [];
  const selected = new Set(universe?.selected_symbols || ["BTCUSDT"]);
  if (target) {
    target.innerHTML = symbols
      .map((symbol) => `
        <label class="backtests-symbol-row">
          <input type="checkbox" value="${escapeHtml(symbol.value)}" data-symbol-checkbox ${selected.has(symbol.value) ? "checked" : ""}>
          <span>${escapeHtml(symbol.label)}</span>
          <small>${escapeHtml(symbol.status)}</small>
        </label>
      `)
      .join("");
  }
  if (selectedTarget) {
    selectedTarget.innerHTML = Array.from(selected)
      .map((symbol) => `<span class="backtests-chip">${escapeHtml(symbol)}</span>`)
      .join("");
  }
  setText("[data-symbol-count]", t("backtests.instruments.count", { count: selected.size }), root);
}

function renderIndicators(root, catalog) {
  const target = qs("[data-indicator-rows]", root);
  if (target) {
    const rows = catalog?.items || [];
    target.innerHTML = rows.length
      ? rows
          .map((row) => `
            <tr>
              <td>${escapeHtml(row.label)}</td>
              <td>${numberOrDash(row.min_value)}</td>
              <td>${numberOrDash(row.max_value)}</td>
              <td>${numberOrDash(row.step)}</td>
              <td>${escapeHtml((row.sources || []).join(", "))}</td>
            </tr>
          `)
          .join("")
      : `<tr><td colspan="5">${escapeHtml(t("common.unavailable"))}</td></tr>`;
  }
  setText("[data-combinations-count]", catalog?.total_combinations_estimate ?? "--", root);
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
}

function renderJobs(root, table) {
  const target = qs("[data-job-rows]", root);
  if (!target) {
    return;
  }
  const rows = table?.items || [];
  if (!rows.length) {
    target.innerHTML = `<tr><td colspan="13">${escapeHtml(table?.degradation_reason || t("backtests.results.empty"))}</td></tr>`;
    return;
  }
  target.innerHTML = rows
    .map(
      (row, index) => `
        <tr data-job-id="${escapeHtml(row.job_id)}">
          <td>${index + 1}</td>
          <td>${escapeHtml(row.strategy)}</td>
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
          <td>${escapeHtml(row.state)} / ${row.progress_percent}%</td>
        </tr>
      `
    )
    .join("");
}

function renderFooter(root, data) {
  setText("[data-footer-api]", data?.footer_status?.api || "--");
  setText("[data-footer-worker]", data?.footer_status?.worker || "--");
  setText("[data-footer-queue]", data?.footer_status?.queue || "--");
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
}

function renderWorkstation(root, data) {
  state.runtimeDefaults = data;
  manualRefreshRetrySeconds = Number(data?.retry_after_seconds || 0);
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
  activeRequest = apiFetch(`${endpoint}?${params.toString()}`)
    .then((data) => {
      renderWorkstation(root, data);
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
    await refreshWorkstation(root, "manual");
  } catch (error) {
    setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
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
        setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
      });
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
}

function init() {
  const root = qs("[data-backtests-root]");
  if (!root) {
    return;
  }
  bind(root);
  refreshWorkstation(root, "initial").catch((error) => {
    setText("[data-create-status]", error?.message || t("backtests.status.failed"), root);
  });
  setAutorefresh(root, "15s");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
