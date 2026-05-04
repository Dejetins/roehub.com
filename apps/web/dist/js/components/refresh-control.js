import { createRefreshController } from "../core/refresh.js";
import { qsa } from "../core/dom.js";

export function initRefreshControls(root = document) {
  qsa("[data-refresh-control]", root).forEach((control) => {
    const endpoint = control.dataset.refreshEndpoint;
    if (!endpoint) {
      return;
    }
    const trigger = control.querySelector("[data-refresh-trigger]");
    const status = control.querySelector("[data-refresh-status]");
    const current = control.querySelector("[data-refresh-current]");
    const controller = createRefreshController({
      endpoint,
      validateInterval: (intervalMs) => intervalMs === 0 || intervalMs >= 10000,
      onStatus: (state) => {
        if (status instanceof HTMLElement) {
          status.textContent = state;
        }
        if (trigger instanceof HTMLButtonElement) {
          trigger.disabled = state === "running";
        }
      },
    });

    if (trigger instanceof HTMLButtonElement) {
      trigger.addEventListener("click", () => {
        controller.refresh("manual").catch(() => null);
      });
    }

    qsa("[data-refresh-preset]", control).forEach((item) => {
      item.addEventListener("click", () => {
        const preset = controller.setAutorefresh(item.dataset.refreshPreset || "off");
        if (current instanceof HTMLElement) {
          current.textContent = preset.key;
        }
      });
    });

    control.roehubRefresh = controller;
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initRefreshControls());
} else {
  initRefreshControls();
}
