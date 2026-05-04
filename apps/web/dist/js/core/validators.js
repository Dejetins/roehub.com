export function validateRequired(value) {
  return String(value ?? "").trim().length > 0;
}

export function validateRefreshInterval(intervalMs, { minMs = 10000, maxMs = 300000 } = {}) {
  return Number.isFinite(intervalMs) && intervalMs === 0 || (intervalMs >= minMs && intervalMs <= maxMs);
}

export function parsePositiveInteger(value) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}
