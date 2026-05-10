export function renderBacktestSeries(canvas, series, options = {}) {
  if (!(canvas instanceof HTMLCanvasElement)) {
    return { nonblank: false, points: 0 };
  }
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
  const points = (Array.isArray(series) ? series : [])
    .map((point, index) => ({ x: index, y: Number(point.value) }))
    .filter((point) => Number.isFinite(point.y));

  drawGrid(ctx, { width, height, pad, dpr });
  if (!points.length) {
    drawEmpty(ctx, width, height, dpr);
    return { nonblank: false, points: 0 };
  }

  const values = points.map((point) => point.y);
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
    const y = pad.top + (1 - (point.y - min) / (max - min)) * plotHeight;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  return { nonblank: true, points: points.length };
}

function drawGrid(ctx, { width, height, pad, dpr }) {
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

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}
