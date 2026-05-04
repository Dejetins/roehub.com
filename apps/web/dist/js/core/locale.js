import { dispatchRoehubEvent, qsa, setText } from "./dom.js";

export const LOCALE_STORAGE_KEY = "roehub_locale";
export const LOCALE_COOKIE_NAME = "roehub_locale";
export const SUPPORTED_LOCALES = ["en", "ru"];
export const DEFAULT_LOCALE = "en";

let localeCatalogs = {};

export function normalizeLocale(value) {
  return SUPPORTED_LOCALES.includes(value) ? value : DEFAULT_LOCALE;
}

function parseCatalogs() {
  const source = document.getElementById("roehub-locale-catalogs");
  if (!source) {
    return {};
  }
  try {
    const parsed = JSON.parse(source.textContent || "{}");
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

export function getCurrentLocale() {
  return normalizeLocale(document.documentElement.dataset.locale || document.documentElement.lang);
}

export function setLocaleCookie(locale) {
  document.cookie = `${LOCALE_COOKIE_NAME}=${normalizeLocale(locale)}; path=/; max-age=31536000; samesite=lax`;
}

export function persistLocale(locale) {
  const normalizedLocale = normalizeLocale(locale);
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, normalizedLocale);
  } catch {
    // Storage is optional; the cookie keeps SSR aligned.
  }
  setLocaleCookie(normalizedLocale);
  return normalizedLocale;
}

export function applyLocale(locale, { persist = true } = {}) {
  const normalizedLocale = normalizeLocale(locale);
  document.documentElement.lang = normalizedLocale;
  document.documentElement.dataset.locale = normalizedLocale;
  if (persist) {
    persistLocale(normalizedLocale);
  }
  qsa("[data-locale-option]").forEach((option) => {
    const isSelected = option.dataset.localeOption === normalizedLocale;
    option.setAttribute("aria-selected", isSelected ? "true" : "false");
  });
  setText("[data-locale-current]", normalizedLocale.toUpperCase());
  dispatchRoehubEvent("locale-change", { locale: normalizedLocale });
  return normalizedLocale;
}

export function t(key, replacements = {}, locale = getCurrentLocale()) {
  const normalizedLocale = normalizeLocale(locale);
  const catalog = localeCatalogs[normalizedLocale] || {};
  const fallbackCatalog = localeCatalogs[DEFAULT_LOCALE] || {};
  let value = catalog[key] || fallbackCatalog[key] || key;
  Object.entries(replacements).forEach(([name, replacement]) => {
    value = value.replaceAll(`{${name}}`, String(replacement));
  });
  return value;
}

export function initLocale(root = document) {
  localeCatalogs = parseCatalogs();
  applyLocale(getCurrentLocale(), { persist: false });
  qsa("[data-locale-option]", root).forEach((option) => {
    option.addEventListener("click", () => {
      const nextLocale = applyLocale(option.dataset.localeOption || DEFAULT_LOCALE);
      const targetUrl = option.dataset.localeUrl;
      if (targetUrl) {
        window.location.assign(targetUrl);
      } else if (nextLocale !== getCurrentLocale()) {
        window.location.reload();
      }
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initLocale());
} else {
  initLocale();
}
