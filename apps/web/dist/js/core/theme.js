import { dispatchRoehubEvent, qsa, setText } from "./dom.js";

export const THEME_STORAGE_KEY = "roehub_theme";
export const SUPPORTED_THEMES = ["abyss", "graphite", "slate", "frost", "paper", "sand"];
export const DEFAULT_THEME = "graphite";
const LEGACY_THEME_MAP = {
  "terminal-orange": "abyss",
  "matrix-green": "slate",
  "high-contrast": "paper",
};
const THEME_LABELS = {
  abyss: "Abyss",
  graphite: "Graphite",
  slate: "Slate",
  frost: "Frost",
  paper: "Paper",
  sand: "Sand",
};

export function normalizeTheme(value) {
  const mappedValue = LEGACY_THEME_MAP[value] || value;
  return SUPPORTED_THEMES.includes(mappedValue) ? mappedValue : DEFAULT_THEME;
}

export function themeDisplayName(theme) {
  const normalizedTheme = normalizeTheme(theme);
  return THEME_LABELS[normalizedTheme] || normalizedTheme;
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
  setText("[data-theme-current]", themeDisplayName(normalizedTheme));
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
