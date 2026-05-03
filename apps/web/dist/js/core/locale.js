import { delegate, qsa, rootElement } from "./dom.js";

export const DEFAULT_LOCALE = "en";
export const SUPPORTED_LOCALES = Object.freeze(["en", "ru"]);
export const LOCALE_STORAGE_KEY = "roehub.locale";
export const LOCALE_COOKIE_NAME = "roehub_locale";

const FALLBACK_CATALOG = Object.freeze({
  en: {
    "js.error.conflict": "The request conflicts with the current state.",
    "js.error.forbidden": "You do not have permission to perform this action.",
    "js.error.network": "Network request failed.",
    "js.error.timeout": "Request timed out.",
    "js.error.unauthorized": "Your session has expired.",
    "js.error.validation": "Check the highlighted fields.",
    "js.validation.number": "Enter a valid number.",
    "js.validation.required": "This field is required.",
  },
  ru: {
    "js.error.conflict": "Запрос конфликтует с текущим состоянием.",
    "js.error.forbidden": "Недостаточно прав для этого действия.",
    "js.error.network": "Сетевой запрос не выполнен.",
    "js.error.timeout": "Время ожидания запроса истекло.",
    "js.error.unauthorized": "Сессия истекла.",
    "js.error.validation": "Проверьте выделенные поля.",
    "js.validation.number": "Введите корректное число.",
    "js.validation.required": "Поле обязательно.",
  },
});

let catalogs = FALLBACK_CATALOG;
let currentLocale = DEFAULT_LOCALE;

export function normalizeLocale(rawLocale) {
  if (typeof rawLocale !== "string") {
    return null;
  }
  const normalized = rawLocale.trim().toLowerCase().replace("_", "-").split("-")[0];
  return SUPPORTED_LOCALES.includes(normalized) ? normalized : null;
}

export function getCurrentLocale() {
  return currentLocale;
}

export function translate(key, params = {}, locale = currentLocale) {
  const resolvedLocale = normalizeLocale(locale) || DEFAULT_LOCALE;
  const message =
    catalogs[resolvedLocale]?.[key] ?? catalogs[DEFAULT_LOCALE]?.[key] ?? key;
  return Object.entries(params).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
    message,
  );
}

export function initLocale(options = {}) {
  const documentRef = options.document || document;
  const root = rootElement(documentRef);
  catalogs = readCatalogs(documentRef);

  const initialLocale =
    normalizeLocale(root.dataset.locale) ||
    normalizeLocale(readStorage(options.storage || window.localStorage)) ||
    DEFAULT_LOCALE;
  applyLocale(initialLocale, { document: documentRef, persist: false });

  delegate(documentRef, "click", "[data-locale-option]", (event, element) => {
    const locale = normalizeLocale(element.getAttribute("data-locale-option"));
    if (!locale) {
      return;
    }
    event.preventDefault();
    applyLocale(locale, { document: documentRef, persist: true });
  });

  return {
    get locale() {
      return currentLocale;
    },
    setLocale: (locale) => applyLocale(locale, { document: documentRef, persist: true }),
    t: translate,
  };
}

export function applyLocale(rawLocale, options = {}) {
  const locale = normalizeLocale(rawLocale) || DEFAULT_LOCALE;
  const documentRef = options.document || document;
  const root = rootElement(documentRef);

  currentLocale = locale;
  root.lang = locale;
  root.dataset.locale = locale;
  updateTranslatedElements(documentRef, locale);
  updateLocaleControls(documentRef, locale);

  if (options.persist) {
    writeStorage(window.localStorage, locale);
    writeCookie(documentRef, locale);
  }

  documentRef.dispatchEvent(
    new CustomEvent("roehub:locale-change", { detail: { locale } }),
  );
  return locale;
}

export function readCatalogs(documentRef = document) {
  const node = documentRef.getElementById("rh-locale-catalogs");
  if (!node?.textContent) {
    return FALLBACK_CATALOG;
  }
  try {
    const parsed = JSON.parse(node.textContent);
    if (!parsed.en || !parsed.ru) {
      return FALLBACK_CATALOG;
    }
    return parsed;
  } catch (_error) {
    return FALLBACK_CATALOG;
  }
}

function updateTranslatedElements(documentRef, locale) {
  qsa("[data-i18n]", documentRef).forEach((element) => {
    element.textContent = translate(element.getAttribute("data-i18n"), {}, locale);
  });
  qsa("[data-i18n-aria-label]", documentRef).forEach((element) => {
    element.setAttribute(
      "aria-label",
      translate(element.getAttribute("data-i18n-aria-label"), {}, locale),
    );
  });
  qsa("[data-i18n-title]", documentRef).forEach((element) => {
    element.setAttribute("title", translate(element.getAttribute("data-i18n-title"), {}, locale));
  });
  qsa("[data-current-locale]", documentRef).forEach((element) => {
    element.textContent = locale.toUpperCase();
  });
}

function updateLocaleControls(documentRef, locale) {
  qsa("[data-locale-option]", documentRef).forEach((element) => {
    const active = element.getAttribute("data-locale-option") === locale;
    element.dataset.active = String(active);
    element.classList.toggle("rh-locale-switcher__option--active", active);
    if (active) {
      element.setAttribute("aria-current", "true");
    } else {
      element.removeAttribute("aria-current");
    }
  });
}

function readStorage(storage) {
  try {
    return storage.getItem(LOCALE_STORAGE_KEY);
  } catch (_error) {
    return null;
  }
}

function writeStorage(storage, locale) {
  try {
    storage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch (_error) {
    // Browsers can deny storage in private contexts; cookie sync remains enough.
  }
}

function writeCookie(documentRef, locale) {
  documentRef.cookie = `${LOCALE_COOKIE_NAME}=${locale}; Max-Age=31536000; Path=/; SameSite=Lax`;
}
