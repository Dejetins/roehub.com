import { apiFetch } from "../core/api.js";
import { on, qs } from "../core/dom.js";
import { t } from "../core/locale.js";
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
  environment: "mainnet",
  permissions: "read",
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

function closeDropdownForItem(item) {
  const dropdown = item.closest("[data-rh-dropdown]");
  const trigger = dropdown?.querySelector("[data-rh-dropdown-trigger]");
  const menu = dropdown?.querySelector("[data-rh-dropdown-menu]");
  if (dropdown instanceof HTMLElement) {
    dropdown.dataset.open = "false";
  }
  if (trigger instanceof HTMLElement) {
    trigger.setAttribute("aria-expanded", "false");
  }
  if (menu instanceof HTMLElement) {
    menu.hidden = true;
  }
}

function setText(selector, value) {
  const target = qs(selector);
  if (target) {
    target.textContent = value;
  }
}

function formatTimestampToSeconds(value) {
  const text = String(value || "").trim();
  if (!text) return "--";
  return text.replace("T", " ").replace(/\.\d+/, "").replace(/Z$/, "");
}

function clearSecretInputs(form) {
  form.querySelectorAll("[data-secret-input]").forEach((input) => {
    input.value = "";
  });
}

function secretInputValue(form, selector) {
  const input = qs(selector, form);
  return input instanceof HTMLInputElement ? input.value : "";
}

function exchangeItems(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.items)) return payload.items;
  return [];
}

function statusClass(item) {
  if (item?.status === "disabled") return "is-warning";
  const validationStatus = item?.validation_status || "";
  if (validationStatus === "valid_readonly" || validationStatus === "valid_trade_enabled") {
    return "is-positive";
  }
  if (validationStatus === "skipped_external_validation") return "is-warning";
  if (validationStatus.startsWith("invalid_") || validationStatus === "unsupported_account_mode") {
    return "is-negative";
  }
  return "";
}

function readableStatus(value) {
  return String(value || "--").replaceAll("_", " ");
}

function formatPlan(value) {
  const normalized = String(value || "free").trim().toLowerCase();
  return normalized ? normalized.charAt(0).toUpperCase() + normalized.slice(1) : "Free";
}

function renderProfile(profile) {
  if (!profile) return;
  const locale = profile.locale || profile.language || "en";
  const contact = profile.telegram_discord || profile.telegram || "--";
  const subscription = profile.subscription_status || profile.subscription || "free";
  setText("[data-profile-username]", profile.username || "quant_trader");
  setText("[data-profile-email]", profile.email || "quant_trader@example.com");
  setText("[data-profile-user-id]", profile.user_id);
  setText("[data-profile-timezone]", profile.timezone);
  setText("[data-profile-locale]", locale.toUpperCase());
  setText("[data-profile-contact]", contact);
  setText("[data-profile-subscription]", formatPlan(subscription));
  setText("[data-status-account]", formatPlan(subscription));
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

function renderExchangeKeys(payload) {
  const body = qs("[data-exchange-keys-body]");
  if (!body) return;
  const items = exchangeItems(payload);
  body.replaceChildren();
  if (!items?.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 11;
    cell.textContent = t("settings.exchange.empty");
    return;
  }
  items.forEach((item) => {
    const row = body.insertRow();
    const rowClass = statusClass(item);
    [
      item.exchange_name,
      item.label || "--",
      item.api_key,
      readableStatus(item.status),
      readableStatus(item.validation_status),
      item.permissions || "--",
      item.market_type,
      item.environment || "--",
      readableStatus(item.ip_restriction_status),
      formatTimestampToSeconds(item.last_validated_at),
    ].forEach((value, cellIndex) => {
      const cell = row.insertCell();
      cell.textContent = String(value || "--");
      if (cellIndex === 3 || cellIndex === 4 || cellIndex === 8) {
        cell.className = rowClass;
      }
    });
    const action = row.insertCell();
    action.className = "settings-exchange-actions";
    [
      ["validate", "settings.exchange.validate", "settings.exchange.validate_short"],
      ["rotate", "settings.exchange.rotate", "settings.exchange.rotate_short"],
      ["disable", "settings.exchange.disable", "settings.exchange.disable_short"],
    ].forEach(([actionKey, labelKey, shortLabelKey]) => {
      const button = document.createElement("button");
      button.className = "rh-button rh-button--secondary rh-button--compact";
      button.type = "button";
      button.dataset[`exchange${actionKey.charAt(0).toUpperCase()}${actionKey.slice(1)}`] =
        item.connection_id || item.key_id || "";
      button.disabled = actionKey !== "validate" && item.status === "disabled";
      button.setAttribute("aria-label", t(labelKey));
      button.textContent = t(shortLabelKey);
      action.append(button);
    });
  });
}

function renderRotateRow(row, connectionId) {
  const existing = qs(`[data-rotate-row="${connectionId}"]`);
  if (existing) {
    existing.remove();
    return;
  }
  const rotateRow = row.parentElement?.insertRow(row.sectionRowIndex + 1);
  if (!rotateRow) return;
  rotateRow.dataset.rotateRow = connectionId;
  const cell = rotateRow.insertCell();
  cell.colSpan = 11;
  cell.innerHTML = `
    <form class="settings-rotate-form" data-rotate-form="${connectionId}" autocomplete="off" data-lpignore="true" data-1p-ignore="true" data-bwignore="true" data-form-type="other">
      <input class="settings-secret-input" name="rotate_public_token" type="text" autocomplete="off" autocapitalize="none" spellcheck="false" inputmode="text" placeholder="API key" aria-label="API key" aria-autocomplete="none" data-secret-input data-rotate-api-key-input data-lpignore="true" data-1p-ignore="true" data-bwignore="true" data-form-type="other">
      <input class="settings-secret-input" name="rotate_private_token" type="text" autocomplete="off" autocapitalize="none" spellcheck="false" inputmode="text" placeholder="API secret" aria-label="API secret" aria-autocomplete="none" data-secret-input data-rotate-api-secret-input data-lpignore="true" data-1p-ignore="true" data-bwignore="true" data-form-type="other">
      <button class="rh-button rh-button--primary rh-button--compact" type="submit">${t("settings.exchange.rotate_short")}</button>
      <output class="settings-inline-status" data-rotate-status aria-live="polite"></output>
    </form>
  `;
}

function renderSessions(payload, append = false) {
  const body = qs("[data-sessions-body]");
  if (!body) return;
  if (!append) body.replaceChildren();
  state.sessionsCursor = payload?.next_cursor || null;
  (payload?.items || []).forEach((item) => {
    const row = body.insertRow();
    [formatTimestampToSeconds(item.last_seen_at), item.ip_address, item.device, item.location, item.is_current ? t("settings.state.active") : t("settings.state.ready")].forEach((value) => {
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
    [formatTimestampToSeconds(item.created_at), item.event_type, item.summary].forEach((value) => {
      row.insertCell().textContent = String(value || "--");
    });
  });
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
  on(root, "click", "[data-exchange-validate]", async (_event, item) => {
    const connectionId = item.dataset.exchangeValidate;
    if (!connectionId) return;
    await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/validate`, {
      method: "POST",
    });
    const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), { items: [] });
    renderExchangeKeys(exchangeKeys);
  });
  on(root, "click", "[data-exchange-rotate]", (_event, item) => {
    const connectionId = item.dataset.exchangeRotate;
    const row = item.closest("tr");
    if (!connectionId || !(row instanceof HTMLTableRowElement)) return;
    renderRotateRow(row, connectionId);
  });
  on(root, "click", "[data-exchange-disable]", async (_event, item) => {
    const connectionId = item.dataset.exchangeDisable;
    if (!connectionId) return;
    const confirmation = window.prompt(t("settings.exchange.confirm_disable"));
    if (confirmation !== "DISABLE") return;
    await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/disable`, {
      method: "POST",
    });
    const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), { items: [] });
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
    closeDropdownForItem(item);
  });
  on(root, "click", "[data-market-option]", (_event, item) => {
    state.marketType = item.dataset.marketOption || "futures";
    setDropdownValue("[data-market-current]", state.marketType);
    closeDropdownForItem(item);
  });
  on(root, "click", "[data-environment-option]", (_event, item) => {
    state.environment = item.dataset.environmentOption || "mainnet";
    setDropdownValue("[data-environment-current]", state.environment);
    closeDropdownForItem(item);
  });
  on(root, "click", "[data-permissions-option]", (_event, item) => {
    state.permissions = item.dataset.permissionsOption || "read";
    setDropdownValue("[data-permissions-current]", state.permissions);
    closeDropdownForItem(item);
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
      environment: state.environment,
      label: data.get("label") || null,
      permissions: state.permissions,
      api_key: secretInputValue(form, "[data-exchange-api-key-input]"),
      api_secret: secretInputValue(form, "[data-exchange-api-secret-input]"),
    };
    const status = qs("[data-exchange-form-status]");
    try {
      await apiFetch(endpoint(root, "exchangeKeysEndpoint"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      clearSecretInputs(form);
      form.reset();
      const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), { items: [] });
      renderExchangeKeys(exchangeKeys);
      if (status) status.textContent = t("settings.exchange.saved");
    } catch (error) {
      clearSecretInputs(form);
      if (status) status.textContent = errorMessage(error);
    }
  });

  on(root, "submit", "[data-rotate-form]", async (event, form) => {
    event.preventDefault();
    const connectionId = form.dataset.rotateForm;
    const status = qs("[data-rotate-status]", form);
    if (!connectionId) return;
    const payload = {
      api_key: secretInputValue(form, "[data-rotate-api-key-input]"),
      api_secret: secretInputValue(form, "[data-rotate-api-secret-input]"),
    };
    try {
      await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/rotate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      clearSecretInputs(form);
      const exchangeKeys = await loadJson(endpoint(root, "exchangeKeysEndpoint"), { items: [] });
      renderExchangeKeys(exchangeKeys);
    } catch (error) {
      clearSecretInputs(form);
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
