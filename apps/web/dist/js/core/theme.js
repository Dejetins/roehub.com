import { dispatchRoehubEvent, qsa, setText } from "./dom.js";

export const THEME_STORAGE_KEY = "roehub_theme";
export const SUPPORTED_THEMES = ["terminal-orange", "graphite"];
export const DEFAULT_THEME = "terminal-orange";

export function normalizeTheme(value) {
  return SUPPORTED_THEMES.includes(value) ? value : DEFAULT_THEME;
}

export function getStoredTheme() {
  try {
    return normalizeTheme(window.localStorage.getItem(THEME_STORAGE_KEY) || "");
  } catch {
    return DEFAULT_THEME;
  }
}

export function applyTheme(theme, { persist = true } = {}) {
  const normalizedTheme = normalizeTheme(theme);
  document.documentElement.dataset.theme = normalizedTheme;
  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, normalizedTheme);
    } catch {
      // Storage can be unavailable in private or locked-down contexts.
    }
  }
  qsa("[data-theme-option]").forEach((option) => {
    const isSelected = option.dataset.themeValue === normalizedTheme;
    option.setAttribute("aria-selected", isSelected ? "true" : "false");
  });
  setText("[data-theme-current]", normalizedTheme);
  dispatchRoehubEvent("theme-change", { theme: normalizedTheme });
  return normalizedTheme;
}

export function initThemeSwitcher(root = document) {
  applyTheme(getStoredTheme(), { persist: false });
  qsa("[data-theme-option]", root).forEach((option) => {
    option.addEventListener("click", () => {
      applyTheme(option.dataset.themeValue || DEFAULT_THEME);
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initThemeSwitcher());
} else {
  initThemeSwitcher();
}
