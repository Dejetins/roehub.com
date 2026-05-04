import { apiFetch } from "./api.js";
import { createPoller } from "./poller.js";

export const REFRESH_PRESETS = {
  off: 0,
  "10s": 10000,
  "15s": 15000,
  "30s": 30000,
  "1m": 60000,
  "5m": 300000,
};

export function resolveRefreshPreset(presetKey, validateInterval = null) {
  if (presetKey in REFRESH_PRESETS) {
    return { key: presetKey, intervalMs: REFRESH_PRESETS[presetKey] };
  }
  const customSeconds = Number(presetKey);
  const intervalMs = Number.isFinite(customSeconds) ? customSeconds * 1000 : 0;
  if (validateInterval && !validateInterval(intervalMs)) {
    return { key: "off", intervalMs: 0 };
  }
  return { key: presetKey, intervalMs };
}

export function createRefreshController({ endpoint, onResult, onStatus, validateInterval } = {}) {
  let activeRequest = null;
  let poller = null;

  async function refresh(reason = "manual") {
    if (activeRequest) {
      return activeRequest;
    }
    onStatus?.("running");
    activeRequest = apiFetch(`${endpoint}?refresh=${encodeURIComponent(reason)}`)
      .then((result) => {
        onResult?.(result);
        onStatus?.(result?.retry_after_seconds ? "rate_limited" : "ready");
        return result;
      })
      .catch((error) => {
        onStatus?.(error.code || "failed");
        throw error;
      })
      .finally(() => {
        activeRequest = null;
      });
    return activeRequest;
  }

  function setAutorefresh(presetKey) {
    const preset = resolveRefreshPreset(presetKey, validateInterval);
    if (poller) {
      poller.stop();
      poller = null;
    }
    if (preset.intervalMs > 0) {
      poller = createPoller(() => refresh("auto"), {
        intervalMs: preset.intervalMs,
        hiddenTabPause: true,
      });
      poller.start();
    }
    onStatus?.(preset.key === "off" ? "idle" : "scheduled");
    return preset;
  }

  return {
    refresh,
    setAutorefresh,
    get isRunning() {
      return Boolean(activeRequest);
    },
  };
}
