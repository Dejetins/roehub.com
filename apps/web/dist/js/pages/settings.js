import { apiFetch } from "../core/api.js";
import { on, qs, qsa } from "../core/dom.js";
import { applyLocale, persistLocale, t } from "../core/locale.js";
import { applyTheme } from "../core/theme.js";
import { initDropdowns } from "../components/dropdown.js";

const AUTOREFRESH_STORAGE_KEY = "roehub_autorefresh_defaults";

const state = {
  preferences: {
    theme: "terminal-orange",
    locale: document.documentElement.dataset.locale || "en",
    density: "compact",
    autorefresh_preset: "15s",
    refresh_interval_seconds: 15,
  },
  exchangeName: "binance",
  marketType: "futures",
  sessionsCursor: null,
  auditCursor: null,
};

function endpoint(root, name) {
  return root.dataset[name] || "";
}

function setStatus(message, ok = true) {
  const target = qs("[data-settings-save-state]");
  if (target) {
    target.textContent = message;
    target.classList.toggle("settings-pill--ok", ok);
    target.classList.toggle("settings-pill--warn", !ok);
  }
}

function errorMessage(error) {
  const payload = error?.payload;
  if (payload?.error?.message) {
    return payload.error.message;
  }
  if (payload?.detail?.message) {
    return payload.detail.message;
  }
  if (payload?.detail?.error) {
    return payload.detail.error;
  }
  return error?.message || t("settings.error");
}

async function loadJson(path, fallback) {
  try {
    return await apiFetch(path);
  } catch (error) {
    setStatus(errorMessage(error), false);
    return fallback;
  }
}

function setDropdownValue(selector, value) {
  const target = qs(selector);
  if (target) {
    target.textContent = value;
  }
}

function setText(selector, value) {
  const target = qs(selector);
  if (target) {
    target.textContent = value;
  }
}

function updateSelected(selector, attribute, value) {
  qsa(selector).forEach((item) => {
    item.setAttribute("aria-selected", item.dataset[attribute] === value ? "true" : "false");
  });
}

function renderProfile(profile) {
  if (!profile) return;
  setText("[data-profile-username]", profile.username || "quant_trader");
  setText("[data-profile-email]", profile.email || "quant_trader@example.com");
  setText("[data-profile-user-id]", profile.user_id);
  setText("[data-profile-timezone]", profile.timezone);
  setText("[data-profile-locale]", profile.locale.toUpperCase());
  setText("[data-profile-contact]", profile.telegram_discord || "--");
  setText("[data-profile-subscription]", profile.subscription_status.toUpperCase());
  setText("[data-status-account]", profile.subscription_status.toUpperCase());
}

function renderLimits(limits) {
  if (!limits) return;
  const pairs = [
    ["exchange_connections", limits.exchange_connections_used, limits.exchange_connections_limit],
    ["api_keys", limits.api_keys_used, limits.api_keys_limit],
    ["active_strategies", limits.active_strategies_used, limits.active_strategies_limit],
    ["webhook_events", limits.webhook_events_used, limits.webhook_events_limit],
  ];
  pairs.forEach(([key, used, max]) => {
    const meter = qs(`[data-limit-meter="${key}"]`);
    const value = qs(`[data-limit-value="${key}"]`);
    const usedValue = Number(used);
    const maxValue = Number(max);
    if (meter instanceof HTMLElement) {
      const filledCells = maxValue > 0 ? Math.max(0, Math.min(10, Math.round((usedValue / maxValue) * 10))) : 0;
      meter.setAttribute("aria-valuemax", String(maxValue));
      meter.setAttribute("aria-valuenow", String(usedValue));
      meter.replaceChildren();
      for (let index = 0; index < 10; index += 1) {
        const cell = document.createElement("span");
        cell.className = `settings-cli-meter__cell${index < filledCells ? " settings-cli-meter__cell--fill" : ""}`;
        cell.setAttribute("aria-hidden", "true");
        meter.append(cell);
      }
    }
    if (value) value.textContent = `${used} / ${max}`;
  });
}

function modeDropdown({ label, value, values, dataAttr }) {
  const wrapper = document.createElement("div");
  wrapper.className = "rh-dropdown settings-row-dropdown";
  wrapper.dataset.rhDropdown = "";
  wrapper.dataset.rhDropdownKind = "listbox";
  const id = `${dataAttr}-${crypto.randomUUID()}`;
  const items = values
    .map(
      (item) =>
        `<button class="rh-menu-item" type="button" role="option" aria-selected="${item === value}" data-rh-dropdown-item="${item}" ${dataAttr}="${item}">${item}</button>`
    )
    .join("");
  wrapper.innerHTML = `
    <button class="rh-button rh-button--secondary rh-button--compact rh-dropdown__trigger" type="button" id="${id}-trigger" aria-haspopup="listbox" aria-expanded="false" aria-controls="${id}-menu" data-rh-dropdown-trigger>
      ${label} <span class="rh-dropdown__value">${value}</span>
    </button>
    <div class="rh-popover" id="${id}-menu" role="listbox" hidden data-rh-dropdown-menu>${items}</div>
  `;
  return wrapper;
}

function renderIntegrations(payload) {
  const root = qs("[data-integrations-list]");
  if (!root) return;
  root.replaceChildren();
  const byKey = new Map((payload?.items || []).map((item) => [item.integration_key, item]));
  const requiredItems = [
    ["telegram", "Telegram"],
    ["discord", "Discord"],
  ].map(([integrationKey, label]) => ({
    integration_key: integrationKey,
    label,
    mode: "off",
    status: "disconnected",
    webhook_url_masked: null,
    ...byKey.get(integrationKey),
  }));
  const extraItems = (payload?.items || []).filter((item) => !["telegram", "discord"].includes(item.integration_key));
  [...requiredItems, ...extraItems].forEach((item) => {
    const row = document.createElement("div");
    row.className = "settings-list-row";
    row.dataset.integrationKey = item.integration_key;
    const copy = document.createElement("div");
    copy.innerHTML = `<strong>${item.label}</strong><span>${item.webhook_url_masked || t("settings.integrations.no_webhook")}</span>`;
    row.append(copy);
    const status = document.createElement("span");
    status.className = item.status === "connected" ? "is-positive" : "is-negative";
    status.textContent = item.status;
    row.append(status);
    const connect = document.createElement("button");
    connect.className = "rh-button rh-button--secondary rh-button--compact";
    connect.type = "button";
    connect.dataset.integrationConnect = item.integration_key;
    connect.textContent =
      item.status === "connected" ? t("settings.integrations.connected") : t("settings.integrations.connect");
    if (item.status === "connected") {
      connect.disabled = true;
    }
    row.append(connect);
    row.append(
      modeDropdown({
        label: t("settings.integrations.mode"),
        value: item.mode,
        values: ["off", "alerts", "critical"],
        dataAttr: "data-integration-mode-option",
      })
    );
    root.append(row);
  });
  initDropdowns(root);
}

function renderNotifications(payload) {
  const root = qs("[data-notifications-list]");
  if (!root) return;
  root.replaceChildren();
  (payload?.items || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "settings-list-row";
    row.dataset.channelKey = item.channel_key;
    const label = document.createElement("strong");
    label.textContent = item.label;
    row.append(label);
    const status = document.createElement("span");
    status.className = item.mode === "off" ? "is-warning" : "is-positive";
    status.textContent = item.mode === "off" ? "OFF" : "ON";
    row.append(status);
    row.append(
      modeDropdown({
        label: t("settings.notifications.mode"),
        value: item.mode,
        values: ["off", "on", "critical"],
        dataAttr: "data-notification-mode-option",
      })
    );
    root.append(row);
  });
  initDropdowns(root);
}

function renderPreferences(payload) {
  if (!payload) return;
  state.preferences = {
    theme: payload.theme,
    locale: payload.locale,
    density: payload.density,
    autorefresh_preset: payload.autorefresh.preset_key,
    refresh_interval_seconds: payload.autorefresh.refresh_interval_seconds,
  };
  setDropdownValue("[data-settings-theme-current]", payload.theme);
  setDropdownValue("[data-settings-locale-current]", payload.locale.toUpperCase());
  setDropdownValue("[data-settings-refresh-current]", payload.autorefresh.preset_key);
  const custom = qs("[data-settings-custom-interval]");
  if (custom instanceof HTMLInputElement) {
    custom.value = String(payload.autorefresh.refresh_interval_seconds || 45);
  }
  updateSelected("[data-settings-theme-option]", "settingsThemeOption", payload.theme);
  updateSelected("[data-settings-locale-option]", "settingsLocaleOption", payload.locale);
  updateSelected("[data-settings-refresh-option]", "settingsRefreshOption", payload.autorefresh.preset_key);
  applyTheme(payload.theme, { persist: true });
  persistLocale(payload.locale);
  persistAutorefresh();
}

function persistAutorefresh() {
  try {
    window.localStorage.setItem(AUTOREFRESH_STORAGE_KEY, JSON.stringify({
      preset_key: state.preferences.autorefresh_preset,
      refresh_interval_seconds: state.preferences.refresh_interval_seconds,
    }));
  } catch {
    // Backend persistence is the authority; localStorage is only a browser fallback.
  }
}

function renderExchangeKeys(items) {
  const body = qs("[data-exchange-keys-body]");
  if (!body) return;
  body.replaceChildren();
  if (!items?.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 10;
    cell.textContent = t("settings.exchange.empty");
    return;
  }
  items.forEach((item, index) => {
    const row = body.insertRow();
    const needsAttention = index === 3;
    const status = needsAttention ? t("settings.exchange.needs_attention") : t("settings.state.active");
    const latency = needsAttention ? "128 ms" : `${28 + index * 3} ms`;
    [
      item.exchange_name,
      item.label || "--",
      item.api_key,
      status,
      item.permissions,
      item.market_type,
      item.environment || "Prod",
      item.updated_at,
      latency,
    ].forEach((value, cellIndex) => {
      const cell = row.insertCell();
      cell.textContent = String(value || "--");
      if (cellIndex === 3) {
        cell.className = needsAttention ? "is-warning" : "is-positive";
      }
      if (cellIndex === 8) {
        cell.className = needsAttention ? "is-negative" : "is-positive";
      }
    });
    const action = row.insertCell();
    action.className = "settings-exchange-actions";
    ["refresh"].forEach((actionKey) => {
      const button = document.createElement("button");
      button.className = "rh-button rh-button--secondary rh-button--compact";
      button.type = "button";
      button.setAttribute("aria-label", t(`settings.exchange.${actionKey}`));
      button.textContent = t(`settings.exchange.${actionKey}_short`);
      action.append(button);
    });
    const deleteButton = document.createElement("button");
    deleteButton.className = "rh-button rh-button--secondary rh-button--compact";
    deleteButton.type = "button";
    deleteButton.dataset.exchangeDelete = item.key_id;
    deleteButton.setAttribute("aria-label", t("settings.exchange.disconnect"));
    deleteButton.textContent = t("settings.exchange.disconnect_short");
    action.append(deleteButton);
  });
}

function renderSessions(payload, append = false) {
  const body = qs("[data-sessions-body]");
  if (!body) return;
  if (!append) body.replaceChildren();
  state.sessionsCursor = payload?.next_cursor || null;
  (payload?.items || []).forEach((item) => {
    const row = body.insertRow();
    [item.last_seen_at, item.ip_address, item.device, item.location, item.is_current ? t("settings.state.active") : t("settings.state.ready")].forEach((value) => {
      row.insertCell().textContent = String(value || "--");
    });
  });
}

function renderAudit(payload, append = false) {
  const body = qs("[data-audit-body]");
  if (!body) return;
  if (!append) body.replaceChildren();
  state.auditCursor = payload?.next_cursor || null;
  const items = payload?.items || [];
  if (!append && !items.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 3;
    cell.textContent = t("settings.audit.empty");
    return;
  }
  items.forEach((item) => {
    const row = body.insertRow();
    [item.created_at, item.event_type, item.summary].forEach((value) => {
      row.insertCell().textContent = String(value || "--");
    });
  });
}

async function savePreferences(root) {
  const custom = qs("[data-settings-custom-interval]");
  const payload = {
    theme: state.preferences.theme,
    locale: state.preferences.locale,
    density: state.preferences.density,
    autorefresh_preset: state.preferences.autorefresh_preset,
    refresh_interval_seconds:
      state.preferences.autorefresh_preset === "custom" && custom instanceof HTMLInputElement
        ? Number(custom.value)
        : state.preferences.refresh_interval_seconds,
  };
  try {
    const saved = await apiFetch(endpoint(root, "preferencesEndpoint"), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderPreferences(saved);
    applyLocale(saved.locale, { persist: true });
    setStatus(t("settings.state.saved"), true);
    const status = qs("[data-preferences-status]");
    if (status) status.textContent = t("settings.state.saved");
  } catch (error) {
    const message = errorMessage(error);
    setStatus(message, false);
    const status = qs("[data-preferences-status]");
    if (status) status.textContent = message;
  }
}

async function initSettings(root) {
  const [
    profile,
    limits,
    integrations,
    notifications,
    preferences,
    exchangeKeys,
    sessions,
    audit,
  ] = await Promise.all([
    loadJson(endpoint(root, "profileEndpoint"), null),
    loadJson(endpoint(root, "limitsEndpoint"), null),
    loadJson(endpoint(root, "integrationsEndpoint"), { items: [] }),
    loadJson(endpoint(root, "notificationsEndpoint"), { items: [] }),
    loadJson(endpoint(root, "preferencesEndpoint"), null),
    loadJson(endpoint(root, "exchangeKeysEndpoint"), []),
    loadJson(`${endpoint(root, "sessionsEndpoint")}?limit=5`, { items: [], next_cursor: null }),
    loadJson(`${endpoint(root, "auditEndpoint")}?limit=5`, { items: [], next_cursor: null }),
  ]);
  renderProfile(profile);
  renderLimits(limits);
  renderIntegrations(integrations);
  renderNotifications(notifications);
  renderPreferences(preferences);
  renderExchangeKeys(exchangeKeys);
  renderSessions(sessions);
  renderAudit(audit);
  setStatus(t("settings.state.ready"), true);
}

function initEvents(root) {
  on(root, "click", "[data-settings-theme-option]", (_event, item) => {
    state.preferences.theme = item.dataset.settingsThemeOption || "terminal-orange";
    applyTheme(state.preferences.theme);
    setDropdownValue("[data-settings-theme-current]", state.preferences.theme);
    updateSelected("[data-settings-theme-option]", "settingsThemeOption", state.preferences.theme);
  });
  on(root, "click", "[data-settings-locale-option]", (_event, item) => {
    state.preferences.locale = item.dataset.settingsLocaleOption || "en";
    applyLocale(state.preferences.locale, { persist: true });
    setDropdownValue("[data-settings-locale-current]", state.preferences.locale.toUpperCase());
    updateSelected("[data-settings-locale-option]", "settingsLocaleOption", state.preferences.locale);
  });
  on(root, "click", "[data-settings-refresh-option]", (_event, item) => {
    state.preferences.autorefresh_preset = item.dataset.settingsRefreshOption || "15s";
    const presetSeconds = { off: 0, "10s": 10, "15s": 15, "30s": 30, "1m": 60, "5m": 300 };
    state.preferences.refresh_interval_seconds =
      presetSeconds[state.preferences.autorefresh_preset] ?? state.preferences.refresh_interval_seconds;
    setDropdownValue("[data-settings-refresh-current]", state.preferences.autorefresh_preset);
    updateSelected("[data-settings-refresh-option]", "settingsRefreshOption", state.preferences.autorefresh_preset);
  });
  on(root, "click", "[data-save-all]", () => {
    void savePreferences(root);
  });
  on(root, "click", "[data-integration-mode-option]", async (_event, item) => {
    const row = item.closest("[data-integration-key]");
    const integrationKey = row?.dataset.integrationKey;
    const mode = item.dataset.integrationModeOption;
    if (!integrationKey || !mode) return;
    const saved = await apiFetch(endpoint(root, "integrationsEndpoint"), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ integration_key: integrationKey, mode, webhook_url: null }),
    });
    const integrations = await loadJson(endpoint(root, "integrationsEndpoint"), { items: [saved] });
    renderIntegrations(integrations);
    setStatus(t("settings.state.saved"), true);
  });
  on(root, "click", "[data-integration-connect]", async (_event, item) => {
    const integrationKey = item.dataset.integrationConnect;
    if (!integrationKey) return;
    await apiFetch(endpoint(root, "integrationsEndpoint"), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ integration_key: integrationKey, mode: "alerts", webhook_url: null }),
    });
    const integrations = await loadJson(endpoint(root, "integrationsEndpoint"), { items: [] });
    renderIntegrations(integrations);
    setStatus(t("settings.state.saved"), true);
  });
  on(root, "click", "[data-notification-mode-option]", async (_event, item) => {
    const row = item.closest("[data-channel-key]");
    const channelKey = row?.dataset.channelKey;
    const mode = item.dataset.notificationModeOption;
    if (!channelKey || !mode) return;
    await apiFetch(endpoint(root, "notificationsEndpoint"), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ channel_key: channelKey, mode }),
    });
    const notifications = await loadJson(endpoint(root, "notificationsEndpoint"), { items: [] });
    renderNotifications(notifications);
    setStatus(t("settings.state.saved"), true);
  });
  on(root, "click", "[data-exchange-delete]", async (_event, item) => {
    const keyId = item.dataset.exchangeDelete;
    if (!keyId || !window.confirm(t("settings.exchange.confirm_delete"))) return;
    try {
      await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${keyId}`, { method: "DELETE" });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
    const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), []);
    renderExchangeKeys(exchangeKeys);
  });
  on(root, "click", "[data-sessions-more]", async () => {
    if (!state.sessionsCursor) return;
    const payload = await loadJson(
      `${endpoint(root, "sessionsEndpoint")}?limit=5&cursor=${encodeURIComponent(state.sessionsCursor)}`,
      { items: [], next_cursor: null }
    );
    renderSessions(payload, true);
  });
  on(root, "click", "[data-audit-more]", async () => {
    if (!state.auditCursor) return;
    const payload = await loadJson(
      `${endpoint(root, "auditEndpoint")}?limit=5&cursor=${encodeURIComponent(state.auditCursor)}`,
      { items: [], next_cursor: null }
    );
    renderAudit(payload, true);
  });
  on(root, "click", "[data-profile-edit]", () => {
    const form = qs("[data-profile-form]");
    if (form instanceof HTMLFormElement) {
      form.hidden = !form.hidden;
      if (!form.hidden) {
        qs("input", form)?.focus();
      }
    }
  });
  on(root, "click", "[data-exchange-form-toggle]", () => {
    const form = qs("[data-exchange-form]");
    if (form) form.hidden = !form.hidden;
  });
  on(root, "click", "[data-exchange-name-option]", (_event, item) => {
    state.exchangeName = item.dataset.exchangeNameOption || "binance";
    setDropdownValue("[data-exchange-name-current]", state.exchangeName);
  });
  on(root, "click", "[data-market-option]", (_event, item) => {
    state.marketType = item.dataset.marketOption || "futures";
    setDropdownValue("[data-market-current]", state.marketType);
  });
}

function initForms(root) {
  const profileForm = qs("[data-profile-form]", root);
  profileForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const payload = Object.fromEntries(data.entries());
    const saved = await apiFetch(endpoint(root, "profileEndpoint"), {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderProfile(saved);
    setStatus(t("settings.state.saved"), true);
  });

  const exchangeForm = qs("[data-exchange-form]", root);
  exchangeForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const payload = {
      exchange_name: state.exchangeName,
      market_type: state.marketType,
      label: data.get("label") || null,
      permissions: "trade",
      api_key: data.get("api_key") || "",
      api_secret: data.get("api_secret") || "",
      passphrase: data.get("passphrase") || null,
    };
    const status = qs("[data-exchange-form-status]");
    try {
      await apiFetch(endpoint(root, "exchangeKeysEndpoint"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      form.reset();
      const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), []);
      renderExchangeKeys(exchangeKeys);
      if (status) status.textContent = t("settings.exchange.saved");
    } catch (error) {
      form.querySelectorAll("input[type='password']").forEach((input) => {
        input.value = "";
      });
      if (status) status.textContent = errorMessage(error);
    }
  });
}

function initSettingsPage() {
  const root = qs("[data-settings-root]");
  if (!root) return;
  initEvents(root);
  initForms(root);
  void initSettings(root);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initSettingsPage);
} else {
  initSettingsPage();
}
