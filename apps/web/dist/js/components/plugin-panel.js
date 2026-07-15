const SVG_NS = "http://www.w3.org/2000/svg";
const SUPPORTED_RENDERERS = new Set([
  "trading-time-series",
  "analytics-series",
  "analytics-table",
  "research-summary",
]);

function element(selector, root) {
  return root.querySelector(selector);
}

function clear(node) {
  if (node) {
    node.replaceChildren();
  }
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function columnMap(frame) {
  return new Map(frame.columns.map((column) => [column.key, column]));
}

function unitLabel(column) {
  return column?.unit?.symbol || "—";
}

function formatCell(value, column) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (column?.data_type === "number" || column?.data_type === "integer") {
    const numericValue = Number(value);
    const fractionDigits = column.data_type === "integer" ? 0 : 2;
    const formatted = new Intl.NumberFormat(document.documentElement.lang || "en", {
      maximumFractionDigits: fractionDigits,
    }).format(numericValue * (column.unit?.scale || 1));
    if (column.unit?.kind === "percent") {
      return `${formatted}${column.unit.symbol}`;
    }
    return column.unit ? `${formatted} ${column.unit.symbol}` : formatted;
  }
  if (column?.data_type === "timestamp") {
    const timestamp = new Date(String(value));
    return Number.isNaN(timestamp.valueOf())
      ? String(value)
      : timestamp.toLocaleString(document.documentElement.lang || "en", {
          dateStyle: "short",
          timeStyle: "short",
        });
  }
  return String(value);
}

function validateContribution(contribution) {
  if (
    contribution?.contract !== "RoehubPanelContribution/v1" ||
    !SUPPORTED_RENDERERS.has(contribution.renderer) ||
    typeof contribution?.presentation !== "object"
  ) {
    throw new Error("Unsupported declarative panel contribution");
  }
}

function validateFrame(frame) {
  if (
    frame?.contract !== "RoehubDataFrame/v1" ||
    !Array.isArray(frame.columns) ||
    !Array.isArray(frame.rows) ||
    frame.rows.length > 1000
  ) {
    throw new Error("Unsupported or unbounded RoehubDataFrame/v1 payload");
  }
}

function renderUnits(root, frame) {
  const units = element("[data-panel-units]", root);
  clear(units);
  frame.columns
    .filter((column) => column.unit)
    .forEach((column) => {
      const badge = document.createElement("span");
      badge.className = "plugin-panel__unit";
      badge.textContent = `${column.label}: ${unitLabel(column)}`;
      units?.append(badge);
    });
}

function renderSummary(root, frame) {
  const summary = element("[data-panel-summary]", root);
  clear(summary);
  const columns = frame.columns.filter((column) => column.role === "measure").slice(0, 3);
  columns.forEach((column) => {
    const values = frame.rows
      .map((row) => row[column.key])
      .filter((value) => typeof value === "number");
    const latest = values.at(-1) ?? null;
    const metric = document.createElement("div");
    metric.className = "plugin-panel__metric";
    const label = document.createElement("span");
    label.textContent = column.label;
    const value = document.createElement("strong");
    value.textContent = formatCell(latest, column);
    metric.append(label, value);
    summary?.append(metric);
  });
}

function renderTable(root, contribution, frame) {
  const head = element("[data-panel-table-head]", root);
  const body = element("[data-panel-table-body]", root);
  clear(head);
  clear(body);
  const columns = columnMap(frame);
  const requestedColumns = contribution.presentation.table_columns
    .map((key) => columns.get(key))
    .filter(Boolean);
  const row = document.createElement("tr");
  requestedColumns.forEach((column) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = column.unit
      ? `${column.label} (${column.unit.symbol})`
      : column.label;
    row.append(cell);
  });
  head?.append(row);
  frame.rows.forEach((frameRow) => {
    const tableRow = document.createElement("tr");
    requestedColumns.forEach((column) => {
      const cell = document.createElement("td");
      cell.textContent = formatCell(frameRow[column.key], column);
      tableRow.append(cell);
    });
    body?.append(tableRow);
  });
}

function chartCoordinates(values, { width, left, right, top, bottom }) {
  const numeric = values.filter((value) => Number.isFinite(value));
  let minimum = Math.min(...numeric);
  let maximum = Math.max(...numeric);
  if (minimum === maximum) {
    minimum -= Math.abs(minimum || 1) * 0.05;
    maximum += Math.abs(maximum || 1) * 0.05;
  }
  const x = (index) =>
    left + (index / Math.max(values.length - 1, 1)) * (width - left - right);
  const y = (value) =>
    bottom - ((value - minimum) / (maximum - minimum)) * (bottom - top);
  return { minimum, maximum, x, y };
}

function renderChart(root, contribution, frame) {
  const visual = element("[data-panel-visual]", root);
  const chart = element("[data-panel-chart]", root);
  const drilldown = element("[data-panel-drilldown] span", root);
  clear(chart);
  if (contribution.renderer === "analytics-table" || contribution.renderer === "research-summary") {
    if (visual) visual.hidden = true;
    return;
  }
  if (visual) visual.hidden = false;
  const columns = columnMap(frame);
  const xColumn = columns.get(contribution.presentation.x_column);
  const yColumns = contribution.presentation.y_columns
    .map((key) => columns.get(key))
    .filter((column) => column?.role === "measure")
    .slice(0, 6);
  if (!chart || !xColumn || yColumns.length === 0) {
    throw new Error("Panel presentation does not match the data frame");
  }
  const width = Math.max(280, Math.round(chart.getBoundingClientRect().width || 960));
  const height = 320;
  const padding = {
    top: 24,
    right: width < 480 ? 16 : 30,
    bottom: 38,
    left: width < 480 ? 52 : 64,
  };
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const plotHeight = height - padding.top - padding.bottom;
  const laneGap = 24;
  const laneHeight = (plotHeight - laneGap * (yColumns.length - 1)) / yColumns.length;
  yColumns.forEach((column, seriesIndex) => {
    const points = frame.rows
      .map((row, rowIndex) => ({ row, rowIndex, value: Number(row[column.key]) }))
      .filter((point) => Number.isFinite(point.value));
    const laneTop = padding.top + seriesIndex * (laneHeight + laneGap);
    const laneBottom = laneTop + laneHeight;
    const coordinates = chartCoordinates(
      points.map((point) => point.value),
      {
        width,
        left: padding.left,
        right: padding.right,
        top: laneTop,
        bottom: laneBottom,
      },
    );
    for (let gridIndex = 0; gridIndex <= 2; gridIndex += 1) {
      const y = laneTop + (gridIndex / 2) * laneHeight;
      chart.append(
        svgElement("line", {
          x1: padding.left,
          x2: width - padding.right,
          y1: y,
          y2: y,
          class: "plugin-panel__grid-line",
        }),
      );
      if (gridIndex !== 1) {
        const value = gridIndex === 0 ? coordinates.maximum : coordinates.minimum;
        const scaleLabel = svgElement("text", {
          x: padding.left - 8,
          y: y + 3,
          "text-anchor": "end",
          class: "plugin-panel__axis-label",
        });
        scaleLabel.textContent = formatCell(value, column);
        chart.append(scaleLabel);
      }
    }
    const laneLabel = svgElement("text", {
      x: padding.left,
      y: laneTop - 7,
      class: "plugin-panel__axis-label plugin-panel__axis-label--series",
    });
    laneLabel.textContent = `${column.label} (${unitLabel(column)})`;
    chart.append(laneLabel);
    const path = points
      .map((point) => `${coordinates.x(point.rowIndex)},${coordinates.y(point.value)}`)
      .join(" ");
    chart.append(
      svgElement("polyline", {
        points: path,
        class: `plugin-panel__series${seriesIndex ? " plugin-panel__series--secondary" : ""}`,
      }),
    );
    points.forEach((point) => {
      const marker = svgElement("circle", {
        cx: coordinates.x(point.rowIndex),
        cy: coordinates.y(point.value),
        r: 5,
        class: "plugin-panel__point",
        tabindex: "0",
        role: "button",
        "aria-label": `${column.label}: ${formatCell(point.value, column)}, ${formatCell(
          point.row[xColumn.key],
          xColumn,
        )}`,
      });
      const selectPoint = () => {
        if (drilldown) {
          drilldown.textContent = `${formatCell(point.row[xColumn.key], xColumn)} · ${column.label}: ${formatCell(
            point.value,
            column,
          )}`;
        }
      };
      marker.addEventListener("click", selectPoint);
      marker.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectPoint();
        }
      });
      chart.append(marker);
    });
  });
  const firstLabel = svgElement("text", {
    x: padding.left,
    y: height - 10,
    class: "plugin-panel__axis-label",
  });
  firstLabel.textContent = formatCell(frame.rows[0]?.[xColumn.key], xColumn);
  const lastLabel = svgElement("text", {
    x: width - padding.right,
    y: height - 10,
    "text-anchor": "end",
    class: "plugin-panel__axis-label",
  });
  lastLabel.textContent = formatCell(frame.rows.at(-1)?.[xColumn.key], xColumn);
  chart.append(firstLabel, lastLabel);
}

function renderNotices(root, frame) {
  const notices = element("[data-panel-notices]", root);
  clear(notices);
  [...frame.notices, ...frame.errors].forEach((notice) => {
    const message = document.createElement("p");
    message.className = "plugin-panel__notice";
    message.textContent = `${notice.code}: ${notice.message}`;
    notices?.append(message);
  });
}

export function setHostPanelState(root, state, message = "") {
  const panel = element("[data-host-panel]", root);
  const stateLabel = root.dataset[`label${state[0].toUpperCase()}${state.slice(1)}`] || state;
  const stateBadge = element("[data-panel-state]", root);
  const stateMessage = element("[data-panel-state-message]", root);
  const content = element("[data-panel-content]", root);
  if (panel) {
    panel.dataset.hostPanelState = state;
    panel.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  }
  if (stateBadge) stateBadge.textContent = stateLabel;
  if (state === "error") {
    [
      "[data-panel-summary]",
      "[data-panel-units]",
      "[data-panel-chart]",
      "[data-panel-table-head]",
      "[data-panel-table-body]",
      "[data-panel-notices]",
    ].forEach((selector) => clear(element(selector, root)));
  }
  if (state === "loading" || state === "empty" || state === "error") {
    if (stateMessage) {
      stateMessage.hidden = false;
      stateMessage.setAttribute("role", state === "error" ? "alert" : "status");
      stateMessage.replaceChildren();
      if (state === "loading") {
        const spinner = document.createElement("span");
        spinner.className = "plugin-panel__spinner";
        spinner.setAttribute("aria-hidden", "true");
        stateMessage.append(spinner);
      }
      const text = document.createElement("strong");
      text.textContent = message || stateLabel;
      stateMessage.append(text);
    }
    if (content) content.hidden = true;
  } else {
    if (stateMessage) stateMessage.hidden = true;
    if (content) content.hidden = false;
  }
}

export function renderHostPanel(root, contribution, frame) {
  validateContribution(contribution);
  validateFrame(frame);
  const title = element("[data-panel-title]", root);
  const description = element("[data-panel-description]", root);
  const source = element("[data-panel-source]", root);
  const freshness = element("[data-panel-freshness]", root);
  if (title) title.textContent = contribution.title;
  if (description) description.textContent = contribution.description;
  if (source) source.textContent = frame.metadata.source_label;
  if (freshness) {
    freshness.textContent = `${frame.freshness.status} · ${frame.freshness.age_seconds ?? "?"}s`;
  }
  if (frame.rows.length === 0) {
    [
      "[data-panel-summary]",
      "[data-panel-units]",
      "[data-panel-chart]",
      "[data-panel-table-head]",
      "[data-panel-table-body]",
      "[data-panel-notices]",
    ].forEach((selector) => clear(element(selector, root)));
    setHostPanelState(root, "empty", root.dataset.labelNoData || "No data");
    return "empty";
  }
  const content = element("[data-panel-content]", root);
  if (content) content.hidden = false;
  renderSummary(root, frame);
  renderUnits(root, frame);
  renderTable(root, contribution, frame);
  renderChart(root, contribution, frame);
  renderNotices(root, frame);
  const state = frame.partial
    ? "partial"
    : frame.freshness.status === "stale" || frame.notices.length > 0
      ? "degraded"
      : "success";
  setHostPanelState(root, state);
  return state;
}
