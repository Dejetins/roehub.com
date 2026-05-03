export function drawTimeSeries(canvas, points, options = {}) {
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return false;
  }

  const cssWidth = Math.max(320, Math.floor(canvas.clientWidth || canvas.width || 900));
  const cssHeight = Math.max(220, Math.floor(canvas.clientHeight || 260));
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const styles = getComputedStyle(document.documentElement);
  const surface = styles.getPropertyValue("--rh-surface-2").trim() || "#0a1416";
  const grid = styles.getPropertyValue("--rh-chart-grid-line").trim() || "rgba(255,255,255,0.14)";
  const text = styles.getPropertyValue("--rh-muted").trim() || "#918a82";
  const line = options.color || styles.getPropertyValue("--rh-accent-2").trim() || "#ff9d00";
  const zeroLine = styles.getPropertyValue("--rh-line-muted").trim() || "rgba(255,255,255,0.28)";

  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  const prepared = preparePoints(points);
  const plot = { left: 52, top: 18, right: cssWidth - 16, bottom: cssHeight - 34 };
  drawGrid({ ctx, plot, grid, zeroLine, yBounds: bounds(prepared.map((point) => point.y)) });

  if (prepared.length === 0) {
    ctx.fillStyle = text;
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(options.emptyText || "no points", plot.left, plot.top + 20);
    return false;
  }

  const xBounds = bounds(prepared.map((point) => point.x));
  const yBounds = bounds(prepared.map((point) => point.y));
  const xScale = (value) => plot.left + ((value - xBounds.min) / (xBounds.max - xBounds.min)) * (plot.right - plot.left);
  const yScale = (value) => plot.bottom - ((value - yBounds.min) / (yBounds.max - yBounds.min)) * (plot.bottom - plot.top);

  ctx.beginPath();
  prepared.forEach((point, index) => {
    const x = xScale(point.x);
    const y = yScale(point.y);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();

  const last = prepared[prepared.length - 1];
  ctx.beginPath();
  ctx.arc(xScale(last.x), yScale(last.y), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = line;
  ctx.fill();

  ctx.fillStyle = text;
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText(formatAxis(yBounds.max), 8, plot.top + 5);
  ctx.fillText(formatAxis(yBounds.min), 8, plot.bottom);
  return true;
}

function preparePoints(points) {
  return (Array.isArray(points) ? points : [])
    .map((point, index) => {
      const y = Number(point.value);
      if (!Number.isFinite(y)) {
        return null;
      }
      const timestamp = Date.parse(String(point.x || ""));
      const x = Number.isFinite(timestamp) ? timestamp : index;
      return { x, y };
    })
    .filter((point) => point !== null);
}

function bounds(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return { min: 0, max: 1 };
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    const pad = Math.abs(min) * 0.05 || 1;
    return { min: min - pad, max: max + pad };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

function drawGrid({ ctx, plot, grid, zeroLine, yBounds }) {
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = plot.top + ((plot.bottom - plot.top) * index) / 4;
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
  }
  for (let index = 0; index <= 6; index += 1) {
    const x = plot.left + ((plot.right - plot.left) * index) / 6;
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom);
    ctx.stroke();
  }
  if (yBounds.min < 0 && yBounds.max > 0) {
    const zeroY = plot.bottom - ((0 - yBounds.min) / (yBounds.max - yBounds.min)) * (plot.bottom - plot.top);
    ctx.strokeStyle = zeroLine;
    ctx.beginPath();
    ctx.moveTo(plot.left, zeroY);
    ctx.lineTo(plot.right, zeroY);
    ctx.stroke();
  }
}

function formatAxis(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toLocaleString("en", { maximumFractionDigits: 2 });
}
