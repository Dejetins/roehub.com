import { apiRequest } from "../core/api.js";
import { financialClassName, formatDateTime, formatNumber, formatPercent } from "../core/formatters.js";
import { translate } from "../core/locale.js";
import { drawTimeSeries } from "../charts/timeseries.js";

const PAGE_SELECTOR = "[data-backtest-result-page]";
const TRADES_PAGE_SIZE = 50;

document.addEventListener("DOMContentLoaded", () => {
  const root = document.querySelector(PAGE_SELECTOR);
  if (root === null) {
    return;
  }
  initBacktestResultPage(root);
});

function initBacktestResultPage(root) {
  const paths = {
    summary: requiredData(root, "apiSummaryPath"),
    variantTemplate: requiredData(root, "apiVariantPathTemplate"),
    equityTemplate: requiredData(root, "apiEquityPathTemplate"),
    drawdownTemplate: requiredData(root, "apiDrawdownPathTemplate"),
    monthlyTemplate: requiredData(root, "apiMonthlyPathTemplate"),
    symbolTemplate: requiredData(root, "apiSymbolPathTemplate"),
    tradesTemplate: requiredData(root, "apiTradesPathTemplate"),
    csvTemplate: requiredData(root, "apiCsvPathTemplate"),
  };
  const nodes = collectNodes(root);
  const state = {
    selectedVariantKey: null,
    summary: null,
    variant: null,
    tradesPage: 1,
    tradesPagination: null,
    variantController: null,
  };

  nodes.refresh.addEventListener("click", () => loadSummary({ root, nodes, paths, state }));
  nodes.prev.addEventListener("click", () => {
    if (state.selectedVariantKey && Number(state.tradesPagination?.page || 1) > 1) {
      loadTrades({ root, nodes, paths, state, page: Number(state.tradesPagination.page) - 1 });
    }
  });
  nodes.next.addEventListener("click", () => {
    if (state.selectedVariantKey && state.tradesPagination?.has_next === true) {
      loadTrades({ root, nodes, paths, state, page: Number(state.tradesPagination.page) + 1 });
    }
  });
  nodes.variants.addEventListener("click", (event) => {
    const button = event.target instanceof HTMLElement
      ? event.target.closest("[data-variant-key]")
      : null;
    if (!(button instanceof HTMLElement)) {
      return;
    }
    const variantKey = button.getAttribute("data-variant-key");
    if (variantKey) {
      loadVariant({ root, nodes, paths, state, variantKey });
    }
  });
  window.addEventListener("beforeunload", () => abortController(state.variantController));

  loadSummary({ root, nodes, paths, state });
}

function collectNodes(root) {
  return {
    refresh: requireNode(root, "#backtest-result-refresh"),
    csv: requireNode(root, "#backtest-result-csv"),
    error: requireNode(root, "#backtest-result-error"),
    state: requireNode(root, "#backtest-result-state"),
    metrics: requireNode(root, "#backtest-result-metrics"),
    variants: requireNode(root, "#backtest-result-variants"),
    equityChart: requireNode(root, "#backtest-result-equity-chart"),
    drawdownChart: requireNode(root, "#backtest-result-drawdown-chart"),
    equityCount: requireNode(root, "#backtest-result-equity-count"),
    drawdownCount: requireNode(root, "#backtest-result-drawdown-count"),
    monthlyBody: requireNode(root, "#backtest-result-monthly-body"),
    symbolBody: requireNode(root, "#backtest-result-symbol-body"),
    tradesBody: requireNode(root, "#backtest-result-trades-body"),
    tradesPage: requireNode(root, "#backtest-result-trades-page"),
    prev: requireNode(root, "#backtest-result-prev"),
    next: requireNode(root, "#backtest-result-next"),
  };
}

async function loadSummary({ root, nodes, paths, state }) {
  clearError(nodes);
  setStatus(nodes.state, translate("backtest_result.loading"), "info");
  nodes.variants.textContent = translate("backtest_result.loading");
  try {
    const summary = await apiRequest(paths.summary);
    state.summary = asRecord(summary);
    const variants = arrayOfRecords(state.summary.variants);
    renderSummary({ nodes, summary: state.summary });
    renderVariants({ nodes, variants, selectedVariantKey: state.summary.selected_variant_key });
    const selected = state.summary.selected_variant_key || variants[0]?.variant_key || null;
    if (selected) {
      await loadVariant({ root, nodes, paths, state, variantKey: String(selected) });
    } else {
      renderEmptyVariantState(nodes);
    }
  } catch (error) {
    showError(nodes, error);
    setStatus(nodes.state, translate("backtest_result.error"), "danger");
  }
}

async function loadVariant({ root, nodes, paths, state, variantKey }) {
  clearError(nodes);
  abortController(state.variantController);
  state.variantController = new AbortController();
  state.selectedVariantKey = variantKey;
  state.tradesPage = 1;
  setCsvLink({ nodes, paths, variantKey });
  renderVariants({
    nodes,
    variants: arrayOfRecords(asRecord(state.summary).variants),
    selectedVariantKey: variantKey,
  });
  try {
    const signal = state.variantController.signal;
    const [variant, equity, drawdown, monthly, symbol] = await Promise.all([
      apiRequest(renderPath(paths.variantTemplate, { variant_key: variantKey }), { signal }),
      apiRequest(renderPath(paths.equityTemplate, { variant_key: variantKey }), { signal }),
      apiRequest(renderPath(paths.drawdownTemplate, { variant_key: variantKey }), { signal }),
      apiRequest(renderPath(paths.monthlyTemplate, { variant_key: variantKey }), { signal }),
      apiRequest(renderPath(paths.symbolTemplate, { variant_key: variantKey }), { signal }),
    ]);
    state.variant = asRecord(variant);
    renderVariantMetrics({ nodes, summary: state.summary, variant: state.variant });
    renderSeries({ nodes, equity: asRecord(equity), drawdown: asRecord(drawdown) });
    renderMonthly({ nodes, payload: asRecord(monthly) });
    renderSymbols({ nodes, payload: asRecord(symbol) });
    await loadTrades({ root, nodes, paths, state, page: 1 });
  } catch (error) {
    if (isAbortError(error)) {
      return;
    }
    showError(nodes, error);
  }
}

async function loadTrades({ nodes, paths, state, page }) {
  if (!state.selectedVariantKey) {
    return;
  }
  const path = renderPath(paths.tradesTemplate, {
    variant_key: state.selectedVariantKey,
  });
  const separator = path.includes("?") ? "&" : "?";
  const payload = await apiRequest(`${path}${separator}page=${page}&page_size=${TRADES_PAGE_SIZE}`);
  const result = asRecord(payload);
  state.tradesPagination = asRecord(result.pagination);
  renderTrades({ nodes, payload: result });
}

function renderSummary({ nodes, summary }) {
  const job = asRecord(summary.job);
  setStatus(nodes.state, String(job.state || "-"), statusTone(String(job.state || "")));
  renderMetricCards(nodes.metrics, [
    metric("job_id", compactId(job.job_id)),
    metric("state", job.state || "-"),
    metric("top_variants", asRecord(job.terminal_summary).top_variants_count ?? 0),
    metric("request_hash", compactHash(job.request_hash)),
    metric("created", formatDateTime(job.created_at)),
    metric("finished", formatDateTime(job.finished_at)),
  ]);
}

function renderVariantMetrics({ nodes, summary, variant }) {
  const job = asRecord(summary.job);
  const metrics = asRecord(variant.summary_metrics);
  setStatus(nodes.state, String(job.state || "-"), statusTone(String(job.state || "")));
  renderMetricCards(nodes.metrics, [
    metric("job_id", compactId(job.job_id)),
    metric("variant_key", compactId(variant.variant_key)),
    metric("variant_hash", compactHash(variant.variant_hash)),
    metric("total_return", formatPercent(metrics.total_return_pct), metrics.total_return_pct),
    metric("max_drawdown", formatPercent(metrics.max_drawdown_pct), metrics.max_drawdown_pct),
    metric("trade_count", formatNumber(metrics.trade_count, { digits: 0 })),
    metric("profit_factor", formatNumber(metrics.profit_factor)),
    metric("win_rate", formatPercent(metrics.win_rate_pct)),
  ]);
}

function renderVariants({ nodes, variants, selectedVariantKey }) {
  nodes.variants.innerHTML = "";
  if (variants.length === 0) {
    nodes.variants.textContent = translate("backtest_result.empty_variants");
    return;
  }
  variants.forEach((variant) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rh-variant-pill";
    button.dataset.variantKey = String(variant.variant_key || "");
    button.dataset.active = String(variant.variant_key || "") === selectedVariantKey ? "true" : "false";
    const metrics = asRecord(variant.summary_metrics);
    button.innerHTML = `
      <span>#${escapeHtml(String(variant.rank || ""))}</span>
      <strong>${escapeHtml(compactId(variant.variant_key))}</strong>
      <em class="${financialClassName(metrics.total_return_pct)}">${escapeHtml(formatPercent(metrics.total_return_pct))}</em>
    `;
    nodes.variants.appendChild(button);
  });
}

function renderSeries({ nodes, equity, drawdown }) {
  const equityRendered = drawTimeSeries(nodes.equityChart, equity.points, {
    color: cssVar("--rh-financial-positive"),
    emptyText: translate("backtest_result.empty_chart"),
  });
  const drawdownRendered = drawTimeSeries(nodes.drawdownChart, drawdown.points, {
    color: cssVar("--rh-financial-negative"),
    emptyText: translate("backtest_result.empty_chart"),
  });
  nodes.equityChart.dataset.nonblank = String(equityRendered);
  nodes.drawdownChart.dataset.nonblank = String(drawdownRendered);
  nodes.equityCount.textContent = pointLabel(equity);
  nodes.drawdownCount.textContent = pointLabel(drawdown);
}

function renderMonthly({ nodes, payload }) {
  renderRows({
    body: nodes.monthlyBody,
    items: arrayOfRecords(payload.items),
    emptyColumns: 5,
    emptyText: translate("backtest_result.empty_stats"),
    cells: (item) => [
      item.month || "-",
      formatNumber(item.trade_count, { digits: 0 }),
      formatPercent(item.win_rate_pct),
      formatPercent(item.avg_return_pct),
      formatNumber(item.net_pnl_quote),
    ],
  });
}

function renderSymbols({ nodes, payload }) {
  renderRows({
    body: nodes.symbolBody,
    items: arrayOfRecords(payload.items),
    emptyColumns: 5,
    emptyText: translate("backtest_result.empty_stats"),
    cells: (item) => [
      item.symbol || "-",
      formatNumber(item.trade_count, { digits: 0 }),
      formatNumber(item.long_count, { digits: 0 }),
      formatNumber(item.short_count, { digits: 0 }),
      formatPercent(item.win_rate_pct),
    ],
  });
}

function renderTrades({ nodes, payload }) {
  const items = arrayOfRecords(payload.items);
  renderRows({
    body: nodes.tradesBody,
    items,
    emptyColumns: 7,
    emptyText: translate("backtest_result.empty_trades"),
    cells: (item) => [
      item.trade_index ?? "-",
      item.side || "-",
      `${formatDateTime(item.entry_timestamp)} @ ${formatNumber(item.entry_price)}`,
      `${formatDateTime(item.exit_timestamp)} @ ${formatNumber(item.exit_price)}`,
      formatPercent(item.return_pct),
      formatNumber(item.net_pnl_quote),
      item.exit_reason || "-",
    ],
  });
  const pagination = asRecord(payload.pagination);
  const page = Number(pagination.page || 1);
  const pageSize = Number(pagination.page_size || TRADES_PAGE_SIZE);
  const total = Number(pagination.total || 0);
  const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(total, page * pageSize);
  nodes.tradesPage.textContent = `${start}-${end} / ${total}`;
  nodes.prev.disabled = pagination.has_previous !== true;
  nodes.next.disabled = pagination.has_next !== true;
}

function renderRows({ body, items, emptyColumns, emptyText, cells }) {
  body.innerHTML = "";
  if (items.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = emptyColumns;
    cell.textContent = emptyText;
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("tr");
    cells(item).forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function renderMetricCards(container, items) {
  container.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "rh-result-metric";
    const label = document.createElement("span");
    label.textContent = item.label;
    const value = document.createElement("strong");
    value.textContent = String(item.value ?? "-");
    if (item.financialValue !== undefined) {
      value.classList.add(financialClassName(item.financialValue));
    }
    card.append(label, value);
    container.appendChild(card);
  });
}

function renderEmptyVariantState(nodes) {
  nodes.metrics.innerHTML = "";
  nodes.monthlyBody.innerHTML = "";
  nodes.symbolBody.innerHTML = "";
  nodes.tradesBody.innerHTML = "";
  nodes.tradesPage.textContent = "0-0 / 0";
  nodes.prev.disabled = true;
  nodes.next.disabled = true;
  nodes.csv.setAttribute("aria-disabled", "true");
}

function setCsvLink({ nodes, paths, variantKey }) {
  nodes.csv.href = renderPath(paths.csvTemplate, { variant_key: variantKey });
  nodes.csv.setAttribute("aria-disabled", "false");
}

function setStatus(node, label, tone) {
  node.textContent = label;
  node.className = `rh-status-badge rh-status-badge--${tone || "neutral"}`;
}

function showError(nodes, error) {
  nodes.error.hidden = false;
  nodes.error.textContent = error?.message || translate("backtest_result.error");
}

function clearError(nodes) {
  nodes.error.hidden = true;
  nodes.error.textContent = "";
}

function metric(label, value, financialValue) {
  return { label, value, financialValue };
}

function pointLabel(payload) {
  return `${arrayOfRecords(payload.points).length}/${Number(payload.total_points || 0)}`;
}

function statusTone(state) {
  if (state === "succeeded") {
    return "success";
  }
  if (state === "failed") {
    return "danger";
  }
  if (state === "running" || state === "queued") {
    return "warning";
  }
  return "neutral";
}

function renderPath(template, values) {
  return Object.entries(values).reduce(
    (path, [key, value]) => path.replace(`{${key}}`, encodeURIComponent(String(value))),
    template,
  );
}

function requiredData(root, key) {
  const value = root.dataset[key];
  if (!value) {
    throw new Error(`Missing ${key}`);
  }
  return value;
}

function requireNode(root, selector) {
  const node = root.querySelector(selector);
  if (node === null) {
    throw new Error(`Missing result page node ${selector}`);
  }
  return node;
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function arrayOfRecords(value) {
  return Array.isArray(value) ? value.map((item) => asRecord(item)) : [];
}

function compactId(value) {
  const text = String(value || "");
  if (text.length <= 18) {
    return text || "-";
  }
  return `${text.slice(0, 10)}...${text.slice(-6)}`;
}

function compactHash(value) {
  const text = String(value || "");
  if (text.length <= 16) {
    return text || "-";
  }
  return `${text.slice(0, 8)}...${text.slice(-6)}`;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function abortController(controller) {
  if (controller) {
    controller.abort();
  }
}

function isAbortError(error) {
  return error?.name === "AbortError" || error?.code === "aborted";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
