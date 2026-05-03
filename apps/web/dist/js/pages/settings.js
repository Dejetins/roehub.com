import { apiRequest } from "../core/api.js";
import { qs, qsa, ready, setBusy } from "../core/dom.js";
import { formatDateTime } from "../core/formatters.js";
import { applyLocale, getCurrentLocale, translate } from "../core/locale.js";
import { applyTheme, readFinancialTokens } from "../core/theme.js";
import { notify } from "../core/notifications.js";

const EMPTY_VALUE = "--";
let state = {
  preferences: null,
  sessionsCursor: null,
  auditCursor: null,
  applyingPreferenceSideEffects: false,
};

ready(() => {
  const root = qs("[data-settings]");
  if (!root) {
    return;
  }

  bindForms(root);
  bindPagination(root);
  bindLocalePersistence(root);
  loadInitialState(root);
});

async function loadInitialState(root) {
  setStatus(root, "settings.status.loading", "warning");
  await Promise.allSettled([
    loadProfile(root),
    loadLimits(root),
    loadPreferences(root),
    loadNotifications(root),
    loadIntegrations(root),
    loadExchangeKeys(root),
    loadSessions(root, { append: false }),
    loadAudit(root, { append: false }),
  ]);
  verifyFinancialTokenInvariance(root);
  setStatus(root, "settings.status.ready", "success");
}

function bindForms(root) {
  bindSubmit(root, "profile", saveProfile);
  bindSubmit(root, "preferences", savePreferences);
  bindSubmit(root, "notifications", saveNotifications);
  bindSubmit(root, "integrations", saveIntegrations);
  bindSubmit(root, "exchange-key", saveExchangeKey);
}

function bindSubmit(root, formName, handler) {
  const form = qs(`[data-settings-form="${formName}"]`, root);
  if (!form) {
    return;
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (form.dataset.inFlight === "true") {
      return;
    }
    form.dataset.inFlight = "true";
    setBusy(form, true);
    try {
      await handler(root, form);
    } catch (error) {
      notify(errorMessage(error), { tone: "danger", role: "alert" });
    } finally {
      form.dataset.inFlight = "false";
      setBusy(form, false);
    }
  });
}

function bindPagination(root) {
  qs("[data-load-sessions]", root)?.addEventListener("click", () => {
    loadSessions(root, { append: true });
  });
  qs("[data-load-audit]", root)?.addEventListener("click", () => {
    loadAudit(root, { append: true });
  });
}

function bindLocalePersistence(root) {
  document.addEventListener("roehub:locale-change", async (event) => {
    const locale = event.detail?.locale;
    if (!state.preferences || state.applyingPreferenceSideEffects) {
      return;
    }
    if (locale === state.preferences.locale) {
      return;
    }
    try {
      await updatePreferences(root, { locale });
    } catch (error) {
      notify(errorMessage(error), { tone: "danger", role: "alert" });
    }
  });
}

async function loadProfile(root) {
  const payload = await apiRequest(root.dataset.profileEndpoint);
  setText(root, "[data-profile-user-id]", payload.user_id);
  setText(root, "[data-profile-plan]", payload.paid_level);
  setText(root, "[data-profile-timezone]", payload.timezone);
  const form = qs('[data-settings-form="profile"]', root);
  if (form) {
    form.elements.display_name.value = payload.display_name || "";
    form.elements.timezone.value = payload.timezone || "UTC";
  }
}

async function saveProfile(root, form) {
  const payload = await apiRequest(root.dataset.profileEndpoint, {
    method: "PUT",
    body: {
      display_name: form.elements.display_name.value || null,
      timezone: form.elements.timezone.value || "UTC",
    },
  });
  setText(root, "[data-profile-timezone]", payload.timezone);
  notify(translate("settings.saved.profile"), { tone: "success" });
  await loadAudit(root, { append: false });
}

async function loadLimits(root) {
  const payload = await apiRequest(root.dataset.limitsEndpoint);
  setText(root, "[data-exchange-count]", `${payload.exchange_keys_used}/${payload.exchange_keys_limit}`);
}

async function loadPreferences(root) {
  const payload = await apiRequest(root.dataset.preferencesEndpoint);
  state.preferences = payload;
  const form = qs('[data-settings-form="preferences"]', root);
  if (form) {
    form.elements.theme.value = payload.theme;
    form.elements.locale.value = payload.locale;
    form.elements.density.value = payload.density;
  }
  setText(root, "[data-preferences-updated]", formatDate(payload.updated_at));
  applyPreferenceSideEffects(payload);
}

async function savePreferences(root, form) {
  const preferences = await updatePreferences(root, {
    theme: form.elements.theme.value,
    locale: form.elements.locale.value,
    density: form.elements.density.value,
  });
  applyPreferenceSideEffects(preferences);
  notify(translate("settings.saved.preferences"), { tone: "success" });
  await loadAudit(root, { append: false });
}

async function updatePreferences(root, updates) {
  const payload = await apiRequest(root.dataset.preferencesEndpoint, {
    method: "PUT",
    body: updates,
  });
  state.preferences = payload;
  const form = qs('[data-settings-form="preferences"]', root);
  if (form) {
    form.elements.theme.value = payload.theme;
    form.elements.locale.value = payload.locale;
    form.elements.density.value = payload.density;
  }
  setText(root, "[data-preferences-updated]", formatDate(payload.updated_at));
  return payload;
}

function applyPreferenceSideEffects(preferences) {
  state.applyingPreferenceSideEffects = true;
  try {
    applyTheme(preferences.theme, { persist: true });
    applyLocale(preferences.locale, { persist: true });
  } finally {
    state.applyingPreferenceSideEffects = false;
  }
}

async function loadNotifications(root) {
  const payload = await apiRequest(root.dataset.notificationsEndpoint);
  renderNotifications(root, payload);
}

async function saveNotifications(root, form) {
  const payload = await apiRequest(root.dataset.notificationsEndpoint, {
    method: "PUT",
    body: {
      email_notifications_enabled: form.elements.email_notifications_enabled.checked,
      trade_alerts_enabled: form.elements.trade_alerts_enabled.checked,
      product_updates_enabled: form.elements.product_updates_enabled.checked,
    },
  });
  renderNotifications(root, payload);
  notify(translate("settings.saved.notifications"), { tone: "success" });
  await loadAudit(root, { append: false });
}

function renderNotifications(root, payload) {
  const form = qs('[data-settings-form="notifications"]', root);
  if (form) {
    form.elements.email_notifications_enabled.checked = payload.email_notifications_enabled;
    form.elements.trade_alerts_enabled.checked = payload.trade_alerts_enabled;
    form.elements.product_updates_enabled.checked = payload.product_updates_enabled;
  }
  setText(root, "[data-notifications-updated]", formatDate(payload.updated_at));
}

async function loadIntegrations(root) {
  const payload = await apiRequest(root.dataset.integrationsEndpoint);
  renderIntegrations(root, payload.integrations || []);
}

async function saveIntegrations(root, form) {
  const integrations = qsa("[data-integration-provider]", form).map((input) => ({
    provider: input.dataset.integrationProvider,
    enabled: input.checked,
  }));
  const payload = await apiRequest(root.dataset.integrationsEndpoint, {
    method: "PUT",
    body: { integrations },
  });
  renderIntegrations(root, payload.integrations || []);
  notify(translate("settings.saved.integrations"), { tone: "success" });
  await loadAudit(root, { append: false });
}

function renderIntegrations(root, integrations) {
  const byProvider = new Map(integrations.map((item) => [item.provider, item]));
  qsa("[data-integration-provider]", root).forEach((input) => {
    input.checked = byProvider.get(input.dataset.integrationProvider)?.enabled || false;
  });
  const enabledCount = integrations.filter((item) => item.enabled).length;
  setText(root, "[data-integrations-status]", String(enabledCount));
}

async function loadExchangeKeys(root) {
  const payload = await apiRequest(root.dataset.exchangeKeysEndpoint);
  renderExchangeKeys(root, Array.isArray(payload) ? payload : []);
  await loadLimits(root);
}

async function saveExchangeKey(root, form) {
  const body = {
    exchange_name: form.elements.exchange_name.value,
    market_type: form.elements.market_type.value,
    permissions: form.elements.permissions.value,
    label: form.elements.label.value || null,
    api_key: form.elements.api_key.value,
    api_secret: form.elements.api_secret.value,
    passphrase: form.elements.passphrase.value || null,
  };
  try {
    await apiRequest(root.dataset.exchangeKeysEndpoint, {
      method: "POST",
      body,
    });
  } catch (error) {
    if (extractApiErrorCode(error) === "exchange_key_already_exists") {
      notify(translate("settings.exchange.duplicate"), { tone: "warning", role: "alert" });
      return;
    }
    throw error;
  } finally {
    form.elements.api_key.value = "";
    form.elements.api_secret.value = "";
    form.elements.passphrase.value = "";
  }
  form.elements.label.value = "";
  notify(translate("settings.saved.exchange_key"), { tone: "success" });
  await loadExchangeKeys(root);
}

function renderExchangeKeys(root, keys) {
  const tbody = qs("[data-exchange-keys-list]", root);
  if (!tbody) {
    return;
  }
  tbody.replaceChildren();
  if (!keys.length) {
    tbody.append(tableMessageRow(6, translate("settings.exchange.empty")));
    setText(root, "[data-exchange-count]", "0");
    return;
  }
  keys.forEach((key) => {
    const row = document.createElement("tr");
    row.append(
      tableCell(`${key.exchange_name} / ${key.market_type}`),
      tableCell(key.label || EMPTY_VALUE),
      tableCell(key.permissions),
      tableCell(key.api_key),
      tableCell(formatDate(key.updated_at)),
      actionCell(deleteExchangeKeyButton(root, key.key_id)),
    );
    tbody.append(row);
  });
  setText(root, "[data-exchange-count]", String(keys.length));
}

function deleteExchangeKeyButton(root, keyId) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rh-button rh-button--ghost rh-settings-table__action";
  button.textContent = translate("settings.action.delete");
  button.addEventListener("click", async () => {
    if (!window.confirm(translate("settings.exchange.delete_confirm"))) {
      return;
    }
    setBusy(button, true);
    try {
      await apiRequest(`${root.dataset.exchangeKeysEndpoint}/${keyId}`, { method: "DELETE" });
      notify(translate("settings.saved.exchange_key_deleted"), { tone: "success" });
      await loadExchangeKeys(root);
    } catch (error) {
      notify(errorMessage(error), { tone: "danger", role: "alert" });
    } finally {
      setBusy(button, false);
    }
  });
  return button;
}

async function loadSessions(root, { append }) {
  const url = buildPageUrl(root.dataset.sessionsEndpoint, append ? state.sessionsCursor : null);
  const payload = await apiRequest(url);
  state.sessionsCursor = payload.next_cursor || null;
  renderSessions(root, payload.items || [], { append });
  const loadMore = qs("[data-load-sessions]", root);
  if (loadMore) {
    loadMore.hidden = !state.sessionsCursor;
  }
}

function renderSessions(root, sessions, { append }) {
  const tbody = qs("[data-sessions-list]", root);
  if (!tbody) {
    return;
  }
  if (!append) {
    tbody.replaceChildren();
  }
  if (!sessions.length && !append) {
    tbody.append(tableMessageRow(4, translate("settings.sessions.empty")));
    return;
  }
  sessions.forEach((session) => {
    const row = document.createElement("tr");
    row.append(
      tableCell(formatDate(session.created_at)),
      tableCell(formatDate(session.last_seen_at)),
      tableCell(translate(`settings.sessions.status.${session.status}`)),
      tableCell(formatDate(session.idle_expires_at)),
    );
    tbody.append(row);
  });
}

async function loadAudit(root, { append }) {
  const url = buildPageUrl(root.dataset.auditEndpoint, append ? state.auditCursor : null);
  const payload = await apiRequest(url);
  state.auditCursor = payload.next_cursor || null;
  renderAudit(root, payload.items || [], { append });
  const loadMore = qs("[data-load-audit]", root);
  if (loadMore) {
    loadMore.hidden = !state.auditCursor;
  }
}

function renderAudit(root, events, { append }) {
  const tbody = qs("[data-audit-list]", root);
  if (!tbody) {
    return;
  }
  if (!append) {
    tbody.replaceChildren();
  }
  if (!events.length && !append) {
    tbody.append(tableMessageRow(3, translate("settings.audit.empty")));
    return;
  }
  events.forEach((event) => {
    const row = document.createElement("tr");
    row.append(
      tableCell(formatDate(event.created_at)),
      tableCell(event.event_type),
      tableCell(formatAuditMetadata(event.metadata)),
    );
    tbody.append(row);
  });
}

function buildPageUrl(baseUrl, cursor) {
  const url = new URL(baseUrl, window.location.origin);
  url.searchParams.set("limit", "5");
  if (cursor) {
    url.searchParams.set("cursor", cursor);
  }
  return `${url.pathname}${url.search}`;
}

function tableCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text || EMPTY_VALUE;
  return cell;
}

function actionCell(child) {
  const cell = document.createElement("td");
  cell.append(child);
  return cell;
}

function tableMessageRow(colspan, message) {
  const row = document.createElement("tr");
  const cell = tableCell(message);
  cell.colSpan = colspan;
  row.append(cell);
  return row;
}

function setStatus(root, key, tone) {
  const status = qs("[data-settings-status]", root);
  if (!status) {
    return;
  }
  status.textContent = translate(key);
  status.className = `rh-status-badge rh-status-badge--${tone}`;
}

function setText(root, selector, value) {
  const node = qs(selector, root);
  if (node) {
    node.textContent = value || EMPTY_VALUE;
  }
}

function formatDate(value) {
  return formatDateTime(value, { locale: getCurrentLocale(), empty: EMPTY_VALUE });
}

function formatAuditMetadata(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return EMPTY_VALUE;
  }
  return Object.entries(metadata)
    .map(([key, value]) => `${key}:${Array.isArray(value) ? value.join("|") : value}`)
    .join(" ");
}

function extractApiErrorCode(error) {
  return error?.payload?.detail?.error || error?.payload?.error?.code || error?.code;
}

function errorMessage(error) {
  const code = extractApiErrorCode(error);
  if (code === "exchange_key_already_exists") {
    return translate("settings.exchange.duplicate");
  }
  return error?.message || translate("js.error.network");
}

function verifyFinancialTokenInvariance(root) {
  const before = readFinancialTokens();
  const themes = ["terminal-orange", "graphite", "matrix-green", "high-contrast"];
  const currentTheme = document.documentElement.dataset.theme || "terminal-orange";
  const invariant = themes.every((theme) => {
    applyTheme(theme, { persist: false });
    const after = readFinancialTokens();
    return (
      after.positive === before.positive &&
      after.negative === before.negative &&
      after.neutral === before.neutral
    );
  });
  applyTheme(currentTheme, { persist: false });
  root.dataset.financialTokensInvariant = String(invariant);
}
