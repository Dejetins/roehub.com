import { delegate, qsa, rootElement } from "./dom.js";

export const DEFAULT_THEME = "terminal-orange";
export const THEME_STORAGE_KEY = "roehub.theme";
export const SUPPORTED_THEMES = Object.freeze([
  "terminal-orange",
  "graphite",
  "matrix-green",
  "high-contrast",
]);

export function normalizeTheme(rawTheme) {
  if (typeof rawTheme !== "string") {
    return null;
  }
  const normalized = rawTheme.trim();
  return SUPPORTED_THEMES.includes(normalized) ? normalized : null;
}

export function initTheme(options = {}) {
  const documentRef = options.document || document;
  const storage = options.storage || window.localStorage;
  const root = rootElement(documentRef);
  const serverTheme = normalizeTheme(root.dataset.theme);
  const storedTheme = normalizeTheme(readStoredTheme(storage));
  const initialTheme =
    serverTheme && serverTheme !== DEFAULT_THEME
      ? serverTheme
      : storedTheme || serverTheme || DEFAULT_THEME;

  applyTheme(initialTheme, { document: documentRef, persist: false, storage });

  delegate(documentRef, "click", "[data-theme-option]", (event, element) => {
    const theme = normalizeTheme(element.getAttribute("data-theme-option"));
    if (!theme) {
      return;
    }
    event.preventDefault();
    applyTheme(theme, { document: documentRef, persist: true, storage });
    element.closest("details")?.removeAttribute("open");
  });

  return {
    get theme() {
      return root.dataset.theme || DEFAULT_THEME;
    },
    setTheme: (theme) => applyTheme(theme, { document: documentRef, persist: true, storage }),
  };
}

export function applyTheme(rawTheme, options = {}) {
  const theme = normalizeTheme(rawTheme) || DEFAULT_THEME;
  const documentRef = options.document || document;
  const root = rootElement(documentRef);

  root.dataset.theme = theme;
  updateThemeControls(documentRef, theme);

  if (options.persist) {
    writeStoredTheme(options.storage || window.localStorage, theme);
  }

  documentRef.dispatchEvent(new CustomEvent("roehub:theme-change", { detail: { theme } }));
  return theme;
}

export function readFinancialTokens(documentRef = document) {
  const styles = documentRef.defaultView?.getComputedStyle(rootElement(documentRef));
  return {
    positive: styles?.getPropertyValue("--rh-financial-positive").trim() || "",
    negative: styles?.getPropertyValue("--rh-financial-negative").trim() || "",
    neutral: styles?.getPropertyValue("--rh-financial-neutral").trim() || "",
  };
}

function updateThemeControls(documentRef, theme) {
  qsa("[data-theme-option]", documentRef).forEach((element) => {
    const active = element.getAttribute("data-theme-option") === theme;
    element.dataset.active = String(active);
    element.classList.toggle("rh-theme-menu__item--active", active);
    if (active) {
      element.setAttribute("aria-current", "true");
    } else {
      element.removeAttribute("aria-current");
    }
  });
}

function readStoredTheme(storage) {
  try {
    return storage.getItem(THEME_STORAGE_KEY);
  } catch (_error) {
    return null;
  }
}

function writeStoredTheme(storage, theme) {
  try {
    storage.setItem(THEME_STORAGE_KEY, theme);
  } catch (_error) {
    // A denied localStorage write should not block the accessible theme control.
  }
}
