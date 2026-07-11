const chartInstances = new WeakMap();
const FALLBACK_START_TIME = 946684800;
const FALLBACK_STEP_SECONDS = 86400;

export function renderBacktestSeries(target, series, options = {}) {
  if (!(target instanceof HTMLElement)) {
    return { nonblank: false, points: 0, renderer: "none" };
  }
  const points = normalizeSeriesPoints(series);
  if (!points.length) {
    destroyChart(target);
    renderEmptyState(target);
    return { nonblank: false, points: 0, renderer: "empty" };
  }
  const charts = window.LightweightCharts;
  if (!charts?.createChart || !charts.LineSeries || !charts.BaselineSeries) {
    return renderCanvasFallback(target, points, options);
  }
  return renderLightweightChart(target, points, options, charts);
}

function renderLightweightChart(target, points, options, charts) {
  destroyChart(target);
  target.replaceChildren();
  const kind = options.kind === "drawdown" ? "drawdown" : "equity";
  const mode = options.mode === "baseline" ? "baseline" : "line";
  const height = Math.max(150, Math.round(target.getBoundingClientRect().height || 172));
  const colors = chartColors(kind);
  const chart = charts.createChart(target, {
    autoSize: true,
    height,
    attributionLogo: true,
    handleScale: {
      axisPressedMouseMove: false,
    },
    layout: {
      attributionLogo: true,
      background: { type: charts.ColorType?.Solid || "solid", color: colors.background },
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
      textColor: colors.text,
    },
    grid: {
      horzLines: { color: colors.grid },
      vertLines: { color: colors.grid },
    },
    rightPriceScale: {
      borderColor: colors.divider,
      scaleMargins: { top: 0.16, bottom: 0.12 },
    },
    timeScale: {
      borderColor: colors.divider,
      fixLeftEdge: true,
      fixRightEdge: false,
      rightOffset: 2,
      secondsVisible: false,
      timeVisible: true,
    },
    crosshair: {
      mode: charts.CrosshairMode?.Normal || 1,
      horzLine: { color: colors.crosshair, style: charts.LineStyle?.Dashed || 2, width: 1 },
      vertLine: { color: colors.crosshair, style: charts.LineStyle?.Dashed || 2, width: 1 },
    },
  });
  const seriesApi = mode === "baseline"
    ? chart.addSeries(charts.BaselineSeries, baselineOptions({ kind, colors, points }))
    : chart.addSeries(charts.LineSeries, lineOptions({ kind, colors }));
  seriesApi.setData(points);
  chart.timeScale().fitContent();
  chartInstances.set(target, { chart, seriesApi });
  target.dataset.nonblank = "true";
  target.dataset.pointCount = String(points.length);
  target.dataset.renderer = "lightweight-charts";
  target.dataset.resultChartMode = mode;
  return { nonblank: true, points: points.length, renderer: "lightweight-charts", mode };
}

function normalizeSeriesPoints(series) {
  return (Array.isArray(series) ? series : [])
    .map((point, index) => {
      const value = Number(point?.value);
      if (!Number.isFinite(value)) {
        return null;
      }
      return {
        time: chartTime(point, index),
        value,
      };
    })
    .filter(Boolean)
    .sort((left, right) => Number(left.time) - Number(right.time));
}

function chartTime(point, index) {
  const raw = point?.time ?? point?.timestamp ?? point?.x ?? point?.exit_timestamp;
  if (typeof raw === "string" && raw.trim()) {
    const numeric = Number(raw);
    if (Number.isFinite(numeric)) {
      return numericTime(numeric, index);
    }
    const parsed = Date.parse(raw);
    if (Number.isFinite(parsed)) {
      return Math.floor(parsed / 1000);
    }
  }
  if (Number.isFinite(Number(raw))) {
    return numericTime(Number(raw), index);
  }
  if (Number.isFinite(Number(point?.trade_index))) {
    return FALLBACK_START_TIME + Number(point.trade_index) * FALLBACK_STEP_SECONDS;
  }
  return FALLBACK_START_TIME + index * FALLBACK_STEP_SECONDS;
}

function numericTime(value, index) {
  if (value >= 1000000000) {
    return Math.floor(value > 100000000000 ? value / 1000 : value);
  }
  return FALLBACK_START_TIME + Math.max(0, Math.floor(value || index)) * FALLBACK_STEP_SECONDS;
}

function lineOptions({ kind, colors }) {
  return {
    color: kind === "drawdown" ? colors.danger : colors.accent,
    crosshairMarkerVisible: true,
    lastValueVisible: true,
    lineWidth: 2,
    priceLineVisible: false,
    priceFormat: { type: "price", precision: kind === "drawdown" ? 2 : 0, minMove: kind === "drawdown" ? 0.01 : 1 },
  };
}

function baselineOptions({ kind, colors, points }) {
  const basePrice = kind === "drawdown" ? 0 : points[0]?.value || 0;
  return {
    baseValue: { type: "price", price: basePrice },
    bottomFillColor1: kind === "drawdown" ? rgba(colors.danger, 0.08) : rgba(colors.danger, 0.22),
    bottomFillColor2: kind === "drawdown" ? rgba(colors.danger, 0.32) : rgba(colors.danger, 0.05),
    bottomLineColor: colors.danger,
    lastValueVisible: true,
    lineWidth: 2,
    priceLineVisible: false,
    topFillColor1: rgba(kind === "drawdown" ? colors.accent : colors.success, 0.26),
    topFillColor2: rgba(kind === "drawdown" ? colors.accent : colors.success, 0.04),
    topLineColor: kind === "drawdown" ? colors.accent : colors.success,
    priceFormat: { type: "price", precision: kind === "drawdown" ? 2 : 0, minMove: kind === "drawdown" ? 0.01 : 1 },
  };
}

function renderCanvasFallback(target, points, options) {
  destroyChart(target);
  const canvas = target instanceof HTMLCanvasElement
    ? target
    : fallbackCanvas(target);
  const result = drawCanvasSeries(canvas, points, options);
  target.dataset.nonblank = result.nonblank ? "true" : "false";
  target.dataset.pointCount = String(result.points || 0);
  target.dataset.renderer = "canvas-fallback";
  return { ...result, renderer: "canvas-fallback" };
}

function fallbackCanvas(target) {
  let canvas = target.querySelector("canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    target.replaceChildren();
    canvas = document.createElement("canvas");
    canvas.className = "backtests-result-chart-fallback";
    target.append(canvas);
  }
  return canvas;
}

function drawCanvasSeries(canvas, points, options = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return { nonblank: false, points: 0 };
  }
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(240, Math.round((rect.width || canvas.clientWidth || 420) * dpr));
  const height = Math.max(120, Math.round((rect.height || canvas.clientHeight || 170) * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = cssVar("--rh-bg", "#050807");
  ctx.fillRect(0, 0, width, height);

  const pad = {
    left: 38 * dpr,
    right: 12 * dpr,
    top: 12 * dpr,
    bottom: 24 * dpr,
  };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  drawChartGrid(ctx, { width, height, pad, dpr });
  if (!points.length) {
    drawEmpty(ctx, width, height, dpr);
    return { nonblank: false, points: 0 };
  }

  const values = points.map((point) => point.value);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const color =
    options.kind === "drawdown"
      ? cssVar("--rh-financial-negative", "#ff3b30")
      : cssVar("--rh-accent-2", "#ffb000");

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = pad.left + (index / Math.max(1, points.length - 1)) * plotWidth;
    const y = pad.top + (1 - (point.value - min) / (max - min)) * plotHeight;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  return { nonblank: true, points: points.length };
}

function renderEmptyState(target) {
  target.replaceChildren();
  const placeholder = document.createElement("span");
  placeholder.className = "backtests-result-chart-empty";
  placeholder.textContent = "NO SERIES";
  target.append(placeholder);
  target.dataset.nonblank = "false";
  target.dataset.pointCount = "0";
  target.dataset.renderer = "empty";
}

function destroyChart(target) {
  const instance = chartInstances.get(target);
  if (instance?.chart) {
    instance.chart.remove();
  }
  chartInstances.delete(target);
}

function drawChartGrid(ctx, { width, height, pad, dpr }) {
  ctx.strokeStyle = cssVar("--rh-chart-grid-line", "rgba(255,128,0,0.2)");
  ctx.lineWidth = 1 * dpr;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }
  for (let i = 0; i <= 5; i += 1) {
    const x = pad.left + ((width - pad.left - pad.right) * i) / 5;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, height - pad.bottom);
    ctx.stroke();
  }
}

function drawEmpty(ctx, width, height, dpr) {
  ctx.fillStyle = cssVar("--rh-muted", "#9d9890");
  ctx.font = `${10 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.fillText("NO SERIES", 14 * dpr, height / 2);
}

function chartColors(kind) {
  return {
    accent: cssVar("--rh-accent-2", "#f0a400"),
    background: cssVar("--rh-bg", "#050807"),
    crosshair: cssVar("--rh-muted", "#a39c90"),
    danger: cssVar("--rh-financial-negative", "#ff5b4a"),
    divider: cssVar("--rh-divider", "rgba(255, 255, 255, 0.1)"),
    grid: cssVar("--rh-chart-grid-line", "rgba(255, 176, 0, 0.12)"),
    success: cssVar("--rh-financial-positive", "#62d26f"),
    text: kind === "drawdown" ? cssVar("--rh-muted", "#a39c90") : cssVar("--rh-text", "#f2eee7"),
  };
}

function rgba(color, alpha) {
  if (color.startsWith("#")) {
    const normalized = color.length === 4
      ? `#${color[1]}${color[1]}${color[2]}${color[2]}${color[3]}${color[3]}`
      : color;
    const value = Number.parseInt(normalized.slice(1), 16);
    if (Number.isFinite(value)) {
      const red = (value >> 16) & 255;
      const green = (value >> 8) & 255;
      const blue = value & 255;
      return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
    }
  }
  return color;
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}
