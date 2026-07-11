import { apiFetch } from "../core/api.js";
import { on, qs } from "../core/dom.js";
import { t } from "../core/locale.js";
import { initDropdowns } from "../components/dropdown.js";

const AUTOREFRESH_STORAGE_KEY = "roehub_autorefresh_defaults";
const DEFAULT_MARKET_TYPES = ["spot", "futures"];
let hadLoadError = false;

const state = {
  activeTab: "profile",
  preferences: {
    theme: "graphite",
    locale: document.documentElement.dataset.locale || "en",
    density: "compact",
    autorefresh_preset: "15s",
    refresh_interval_seconds: 15,
  },
  exchangeName: "binance",
  marketTypes: [...DEFAULT_MARKET_TYPES],
  environment: "mainnet",
  exchangeStatusFilter: "active",
  sessionsCursor: null,
  auditCursor: null,
  notificationScoped: null,
};

const SETTINGS_TABS = new Set(["profile", "api", "integrations", "security"]);

function endpoint(root, name) {
  return root.dataset[name] || "";
}

function exchangeKeysPath(root, status = state.exchangeStatusFilter) {
  const base = endpoint(root, "exchangeKeysEndpoint");
  const effectiveStatus = status === "history" ? "all" : status || "active";
  return `${base}?status=${encodeURIComponent(effectiveStatus)}`;
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
  if (error?.outcomeUnknown) {
    return t("settings.mutation.unknown");
  }
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
    hadLoadError = true;
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

function initialSettingsTab() {
  const hashTab = window.location.hash.replace(/^#/, "");
  return SETTINGS_TABS.has(hashTab) ? hashTab : "profile";
}

function setSettingsTab(root, tab) {
  const nextTab = SETTINGS_TABS.has(tab) ? tab : "profile";
  state.activeTab = nextTab;
  root.dataset.settingsActiveTab = nextTab;
  root.querySelectorAll("[data-settings-tab-button]").forEach((button) => {
    const selected = button.dataset.settingsTabButton === nextTab;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  });
  root.querySelectorAll("[data-settings-tab-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.settingsTabPanel !== nextTab;
  });
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

function openExchangeFormModal() {
  const modal = qs("[data-exchange-form-modal]");
  if (!(modal instanceof HTMLElement)) return;
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("settings-exchange-modal-lock");
  const status = qs("[data-exchange-form-status]", modal);
  if (status) status.textContent = "";
  resetMarketResults();
  syncMarketTypesFromControls(modal);
  setEnvironmentValue(modal, state.environment);
  const firstControl = qs("[data-exchange-form] [data-rh-dropdown-trigger], [data-exchange-form] input, [data-exchange-form] button", modal);
  if (firstControl instanceof HTMLElement) {
    firstControl.focus();
  }
}

function closeExchangeFormModal({ clearSecrets = true } = {}) {
  const modal = qs("[data-exchange-form-modal]");
  const form = qs("[data-exchange-form]");
  if (form instanceof HTMLFormElement && clearSecrets) {
    clearSecretInputs(form);
  }
  if (modal instanceof HTMLElement) {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
  }
  document.body.classList.remove("settings-exchange-modal-lock");
  const trigger = qs("[data-exchange-form-toggle]");
  if (trigger instanceof HTMLElement) {
    trigger.focus();
  }
}

function secretInputValue(form, selector) {
  const input = qs(selector, form);
  return input instanceof HTMLInputElement ? input.value : "";
}

function selectedMarketTypes(root = document) {
  const selected = Array.from(root.querySelectorAll("[data-market-type-checkbox]"))
    .filter((item) => item instanceof HTMLInputElement && item.checked)
    .map((item) => item.value)
    .filter((value) => DEFAULT_MARKET_TYPES.includes(value));
  return [...new Set(selected)];
}

function syncMarketTypesFromControls(root = document) {
  const selected = selectedMarketTypes(root);
  state.marketTypes = selected.length ? selected : [];
  return state.marketTypes;
}

function setEnvironmentValue(root, value) {
  const nextEnvironment = ["mainnet", "testnet"].includes(value) ? value : "mainnet";
  state.environment = nextEnvironment;
  root.querySelectorAll("[data-environment-option]").forEach((button) => {
    const selected = button.dataset.environmentOption === nextEnvironment;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-checked", selected ? "true" : "false");
  });
}

function savedExchangeItems(saved) {
  if (Array.isArray(saved?.items)) return saved.items;
  return saved?.connection_id ? [saved] : [];
}

function isTradingReady(item) {
  return (
    item?.status === "active" &&
    item?.effective_capability === "trading" &&
    item?.connection_readiness === "ready_for_trading"
  );
}

function marketLabel(value) {
  if (value === "spot") return t("settings.exchange.market_spot");
  if (value === "futures") return t("settings.exchange.market_futures");
  return readableStatus(value);
}

function resetMarketResults() {
  const target = qs("[data-market-results]");
  if (!target) return;
  target.replaceChildren();
  target.hidden = true;
}

function renderMarketResults(saved) {
  const target = qs("[data-market-results]");
  if (!target) return;
  const results = Array.isArray(saved?.market_results)
    ? saved.market_results
    : savedExchangeItems(saved).map((connection) => ({
        market_type: connection.market_type,
        status: "created",
        connection,
      }));
  target.replaceChildren();
  if (!results.length) {
    target.hidden = true;
    return;
  }
  results.forEach((result) => {
    const row = document.createElement("div");
    const connection = result.connection;
    const ready = isTradingReady(connection);
    row.className = `settings-market-result${
      result.status === "failed" ? " is-negative" : ready ? " is-positive" : " is-warning"
    }`;
    const label = document.createElement("strong");
    label.textContent = marketLabel(result.market_type);
    const message = document.createElement("span");
    message.textContent =
      result.status === "failed"
        ? result.error_message || readableStatus(result.error_code)
        : readinessMessage(connection);
    row.append(label, message);
    target.append(row);
  });
  target.hidden = false;
}

function exchangeItems(payload, status = state.exchangeStatusFilter) {
  const items = Array.isArray(payload) ? payload : Array.isArray(payload?.items) ? payload.items : [];
  if (status === "active") {
    return items.filter((item) => item?.status === "active");
  }
  if (status === "history") {
    return items.filter((item) => item?.status !== "active");
  }
  return items;
}

function statusClass(item) {
  if (item?.status === "disabled" || item?.status === "archived") return "is-warning";
  const validationStatus = item?.validation_status || "";
  if (validationStatus === "permission_mismatch") return "is-negative";
  if (validationStatus === "valid_readonly" || validationStatus === "valid_trade_enabled") {
    return "is-positive";
  }
  if (validationStatus === "skipped_external_validation") return "is-warning";
  if (validationStatus.startsWith("invalid_") || validationStatus === "unsupported_account_mode") {
    return "is-negative";
  }
  return "";
}

function setExchangeStatusFilter(root, status) {
  const effectiveStatus = ["active", "history"].includes(status) ? status : "active";
  state.exchangeStatusFilter = effectiveStatus;
  root.querySelectorAll("[data-exchange-status-filter]").forEach((button) => {
    const selected = button.dataset.exchangeStatusFilter === effectiveStatus;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", selected ? "true" : "false");
  });
}

function readableStatus(value) {
  return String(value || "--").replaceAll("_", " ");
}

function readinessMessage(item) {
  const readiness = item?.connection_readiness || "";
  const reason = item?.connection_readiness_reason || item?.validation_reason || "";
  const reasonKey = reason ? `settings.exchange.reason.${reason}` : "";
  if (readiness === "ready_for_trading") return t("settings.exchange.ready_for_trading");
  if (reasonKey) return t(reasonKey);
  if (readiness) return readableStatus(readiness);
  return readableStatus(item?.status);
}

function compactReadinessMessage(item) {
  if (item?.connection_readiness === "ready_for_trading") return t("settings.state.ready");
  return readinessMessage(item);
}

function capabilityDisplay(item) {
  const effective = item?.effective_capability || "none";
  const requested = item?.requested_capability || "trading";
  if (effective === "trading") return t("settings.exchange.capability_trading");
  return `${readableStatus(requested)} / ${readableStatus(effective)} · ${readinessMessage(item)}`;
}

function marketAvailabilityKey(item) {
  return [
    item?.exchange_name || "",
    item?.label || "",
    item?.api_key || "",
    item?.environment || "",
  ].join("\u001f");
}

function marketAvailabilityRank(item) {
  if (!item) return 0;
  if (
    item.status === "active" &&
    item.connection_readiness === "ready_for_trading" &&
    item.effective_capability === "trading"
  ) {
    return 40;
  }
  if (item.status === "active") return 30;
  if (item.status === "disabled") return 20;
  if (item.status === "archived") return 10;
  return 1;
}

function buildMarketAvailability(items) {
  const groups = new Map();
  items.forEach((item) => {
    if (!item?.market_type) return;
    const key = marketAvailabilityKey(item);
    const group = groups.get(key) || {};
    const existing = group[item.market_type];
    if (marketAvailabilityRank(item) >= marketAvailabilityRank(existing)) {
      group[item.market_type] = item;
    }
    groups.set(key, group);
  });
  return groups;
}

function representativeExchangeItems(items, status = state.exchangeStatusFilter) {
  if (status !== "active") return items;
  const representatives = new Map();
  items.forEach((item) => {
    const key = marketAvailabilityKey(item);
    const existing = representatives.get(key);
    if (marketAvailabilityRank(item) > marketAvailabilityRank(existing)) {
      representatives.set(key, item);
      return;
    }
    if (
      marketAvailabilityRank(item) === marketAvailabilityRank(existing) &&
      item?.market_type === "spot"
    ) {
      representatives.set(key, item);
    }
  });
  return [...representatives.values()];
}

function renderMarketAvailability(group = {}, currentMarketType = "") {
  const root = document.createElement("div");
  root.className = "settings-market-availability";
  DEFAULT_MARKET_TYPES.forEach((marketType) => {
    const item = group[marketType];
    const row = document.createElement("span");
    row.className = "settings-market-availability__item";
    if (marketType === currentMarketType) {
      row.classList.add("is-current");
      row.setAttribute("aria-current", "true");
    }
    const label = document.createElement("span");
    label.className = "settings-market-availability__label";
    label.textContent = marketLabel(marketType);
    const value = document.createElement("span");
    value.className = item ? statusClass(item) : "is-warning";
    value.textContent = item
      ? compactReadinessMessage(item)
      : t("settings.exchange.market_not_connected");
    row.append(label, value);
    root.append(row);
  });
  return root;
}

function canConfigureAccount(item) {
  return (
    item?.status === "active" &&
    item?.market_type === "futures" &&
    item?.environment === "testnet" &&
    item?.effective_capability === "trading" &&
    item?.connection_readiness === "ready_for_trading"
  );
}

function appendTextCell(row, value, className = "") {
  const cell = row.insertCell();
  cell.textContent = String(value || "--");
  if (className) cell.className = className;
  return cell;
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

function renderScopedNotifications(payload) {
  const root = qs("[data-notification-scoped]");
  if (!root) return;
  const fallback = {
    mode: "off",
    route_status: "disabled",
    telegram_binding: { is_confirmed: false },
    delivery_counters: { telegram_sent_total: 0, telegram_sent_last_24h: 0, last_telegram_sent_at: null },
    report_schedule: { weekly_enabled: false, monthly_enabled: false, timezone: "UTC" },
    available_modes: ["off", "critical_only", "signals", "trades", "reports", "all"],
  };
  const model = { ...fallback, ...(payload || {}) };
  model.telegram_binding = { ...fallback.telegram_binding, ...(payload?.telegram_binding || {}) };
  model.delivery_counters = { ...fallback.delivery_counters, ...(payload?.delivery_counters || {}) };
  model.report_schedule = { ...fallback.report_schedule, ...(payload?.report_schedule || {}) };
  state.notificationScoped = model;

  const bindingStatus = qs("[data-telegram-binding-status]");
  const bindingPill = qs("[data-telegram-binding-pill]");
  const isBound = Boolean(model.telegram_binding.is_confirmed);
  if (bindingStatus) {
    bindingStatus.textContent = isBound
      ? t("settings.notifications.telegram_bound")
      : t("settings.notifications.telegram_unbound");
  }
  if (bindingPill) {
    bindingPill.textContent = isBound ? t("settings.state.active") : t("settings.state.paused");
    bindingPill.classList.toggle("settings-pill--ok", isBound);
    bindingPill.classList.toggle("settings-pill--warn", !isBound);
  }

  const sentCount = qs("[data-notification-sent-count]");
  const sentLast = qs("[data-notification-sent-last]");
  if (sentCount) {
    sentCount.textContent = String(model.delivery_counters.telegram_sent_total || 0);
  }
  if (sentLast) {
    const lastSentAt = model.delivery_counters.last_telegram_sent_at;
    sentLast.textContent = lastSentAt
      ? `${t("settings.notifications.last_sent")}: ${new Date(lastSentAt).toLocaleString()}`
      : t("settings.notifications.no_sent_messages");
  }

  const modeRow = qs("[data-scoped-mode-row]");
  if (modeRow) {
    modeRow.replaceChildren();
    const copy = document.createElement("div");
    copy.innerHTML = `<strong>${t("settings.notifications.scoped_mode")}</strong><span>${t(`settings.notifications.mode.${model.mode}`)}</span>`;
    modeRow.append(copy);
    modeRow.append(
      modeDropdown({
        label: t("settings.notifications.mode"),
        value: model.mode,
        values: model.available_modes,
        dataAttr: "data-scoped-mode-option",
      })
    );
  }

  const timezone = qs("[data-report-schedule-timezone]");
  if (timezone) {
    timezone.textContent = model.report_schedule.timezone || "UTC";
  }
  root.querySelectorAll("[data-report-schedule-toggle]").forEach((button) => {
    const key = button.dataset.reportScheduleToggle;
    const enabled = key === "weekly"
      ? Boolean(model.report_schedule.weekly_enabled)
      : Boolean(model.report_schedule.monthly_enabled);
    button.textContent = `${t(`settings.notifications.${key}`)}: ${enabled ? "ON" : "OFF"}`;
    button.classList.toggle("rh-button--primary", enabled);
    button.classList.toggle("rh-button--secondary", !enabled);
    button.setAttribute("aria-pressed", enabled ? "true" : "false");
  });
  initDropdowns(root);
}

async function saveScopedNotifications(root, patch) {
  const current = state.notificationScoped || {
    mode: "off",
    report_schedule: { weekly_enabled: false, monthly_enabled: false, timezone: "UTC" },
  };
  const schedule = current.report_schedule || {};
  const saved = await apiFetch(endpoint(root, "notificationScopedEndpoint"), {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      mode: patch.mode || current.mode || "off",
      weekly_enabled: patch.weekly_enabled ?? Boolean(schedule.weekly_enabled),
      monthly_enabled: patch.monthly_enabled ?? Boolean(schedule.monthly_enabled),
      timezone: patch.timezone || schedule.timezone || "UTC",
    }),
  });
  renderScopedNotifications(saved);
  setStatus(t("settings.state.saved"), true);
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
  const marketAvailability = buildMarketAvailability(items);
  const renderedItems = representativeExchangeItems(items);
  body.replaceChildren();
  if (!renderedItems?.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 11;
    cell.textContent = t("settings.exchange.empty");
    return;
  }
  renderedItems.forEach((item) => {
    const row = body.insertRow();
    const rowClass = statusClass(item);
    const availabilityGroup = marketAvailability.get(marketAvailabilityKey(item)) || {};
    const futuresItem = availabilityGroup.futures;
    appendTextCell(row, item.exchange_name);
    appendTextCell(row, item.label || "--");
    appendTextCell(row, item.api_key);
    appendTextCell(row, readinessMessage(item), rowClass);
    appendTextCell(row, readableStatus(item.validation_status), rowClass);
    appendTextCell(row, capabilityDisplay(item));
    const markets = row.insertCell();
    markets.append(
      renderMarketAvailability(availabilityGroup, item.market_type)
    );
    appendTextCell(row, item.environment || "--");
    appendTextCell(row, readableStatus(item.ip_restriction_status), rowClass);
    appendTextCell(row, formatTimestampToSeconds(item.last_validated_at));
    const action = row.insertCell();
    action.className = "settings-exchange-actions";
    const activeStrategyBindings = Number(item.active_strategy_bindings_count || 0);
    const usedByStrategies = Number(item.used_by_strategies_count || activeStrategyBindings);
    if (usedByStrategies > 0) {
      const usage = document.createElement("span");
      usage.className = "settings-exchange-usage";
      usage.textContent = t("settings.exchange.used_by_strategies", {
        count: String(usedByStrategies),
      });
      action.append(usage);
    }
    const actions =
      item.status === "active"
        ? [
            ["validate", "settings.exchange.recheck", "settings.exchange.recheck_short"],
            ["rotate", "settings.exchange.rotate", "settings.exchange.rotate_short"],
            ...(canConfigureAccount(futuresItem)
              ? [["configureAccount", "settings.exchange.configure_account", "settings.exchange.configure_account_short"]]
              : []),
            ["disable", "settings.exchange.disconnect", "settings.exchange.disconnect_short"],
          ]
        : item.status === "disabled"
          ? [["archive", "settings.exchange.archive", "settings.exchange.archive_short"]]
          : [];
    actions.forEach(([actionKey, labelKey, shortLabelKey]) => {
      const button = document.createElement("button");
      button.className = "rh-button rh-button--secondary rh-button--compact";
      button.type = "button";
      button.dataset[`exchange${actionKey.charAt(0).toUpperCase()}${actionKey.slice(1)}`] =
        actionKey === "configureAccount"
          ? futuresItem?.connection_id || futuresItem?.key_id || ""
          : item.connection_id || item.key_id || "";
      button.setAttribute("aria-label", t(labelKey));
      button.textContent = t(shortLabelKey);
      if (actionKey === "configureAccount") {
        button.dataset.exchangeConfigureExchange = futuresItem?.exchange_name || item.exchange_name || "";
      }
      if (actionKey === "disable" && activeStrategyBindings > 0) {
        button.disabled = true;
        button.title = t("settings.exchange.disconnect_blocked", {
          count: String(activeStrategyBindings),
        });
      }
      action.append(button);
    });
    if (!actions.length) {
      action.textContent = "--";
    }
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
  const moreButton = qs("[data-sessions-more]");
  if (moreButton instanceof HTMLButtonElement) moreButton.disabled = !state.sessionsCursor;
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
  const moreButton = qs("[data-audit-more]");
  if (moreButton instanceof HTMLButtonElement) moreButton.disabled = !state.auditCursor;
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
  hadLoadError = false;
  if (root.dataset.settingsScope === "connections") {
    const [exchangeKeys, audit] = await Promise.all([
      loadJson(exchangeKeysPath(root, "active"), []),
      loadJson(`${endpoint(root, "auditEndpoint")}?limit=5`, { items: [], next_cursor: null }),
    ]);
    setExchangeStatusFilter(root, "active");
    renderExchangeKeys(exchangeKeys);
    renderAudit(audit);
    if (!hadLoadError) setStatus(t("settings.state.ready"), true);
    return;
  }
  const [
    profile,
    limits,
    integrations,
    notifications,
    notificationScoped,
    preferences,
    exchangeKeys,
    sessions,
    audit,
  ] = await Promise.all([
    loadJson(endpoint(root, "profileEndpoint"), null),
    loadJson(endpoint(root, "limitsEndpoint"), null),
    loadJson(endpoint(root, "integrationsEndpoint"), { items: [] }),
    loadJson(endpoint(root, "notificationsEndpoint"), { items: [] }),
    loadJson(endpoint(root, "notificationScopedEndpoint"), null),
    loadJson(endpoint(root, "preferencesEndpoint"), null),
    loadJson(exchangeKeysPath(root, "active"), []),
    loadJson(`${endpoint(root, "sessionsEndpoint")}?limit=5`, { items: [], next_cursor: null }),
    loadJson(`${endpoint(root, "auditEndpoint")}?limit=5`, { items: [], next_cursor: null }),
  ]);
  renderProfile(profile);
  renderLimits(limits);
  renderIntegrations(integrations);
  renderNotifications(notifications);
  renderScopedNotifications(notificationScoped);
  renderPreferences(preferences);
  setExchangeStatusFilter(root, "active");
  renderExchangeKeys(exchangeKeys);
  renderSessions(sessions);
  renderAudit(audit);
  if (!hadLoadError) setStatus(t("settings.state.ready"), true);
}

function initEvents(root) {
  on(root, "click", "[data-settings-tab-button]", (event, item) => {
    event.preventDefault();
    const tab = item.dataset.settingsTabButton || "profile";
    setSettingsTab(root, tab);
    if (window.location.hash.replace(/^#/, "") !== tab) {
      window.history.replaceState(null, "", `#${tab}`);
    }
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
  on(root, "click", "[data-scoped-mode-option]", async (_event, item) => {
    const mode = item.dataset.scopedModeOption;
    if (!mode) return;
    closeDropdownForItem(item);
    await saveScopedNotifications(root, { mode });
  });
  on(root, "click", "[data-report-schedule-toggle]", async (_event, item) => {
    const key = item.dataset.reportScheduleToggle;
    const schedule = state.notificationScoped?.report_schedule || {};
    if (key === "weekly") {
      await saveScopedNotifications(root, { weekly_enabled: !Boolean(schedule.weekly_enabled) });
      return;
    }
    if (key === "monthly") {
      await saveScopedNotifications(root, { monthly_enabled: !Boolean(schedule.monthly_enabled) });
    }
  });
  on(root, "click", "[data-exchange-validate]", async (_event, item) => {
    const connectionId = item.dataset.exchangeValidate;
    if (!connectionId) return;
    item.disabled = true;
    try {
      await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/validate`, {
        method: "POST",
      });
    } catch (error) {
      setStatus(errorMessage(error), false);
    }
    const exchangeKeys = await loadJson(exchangeKeysPath(root), { items: [] });
    renderExchangeKeys(exchangeKeys);
    item.disabled = false;
  });
  on(root, "click", "[data-exchange-configure-account]", async (_event, item) => {
    const connectionId = item.dataset.exchangeConfigureAccount;
    const exchangeName = item.dataset.exchangeConfigureExchange || "";
    if (!connectionId || !exchangeName) return;
    const symbol = window.prompt(t("settings.exchange.configure_symbol_prompt"), "BTCUSDT");
    const normalizedSymbol = String(symbol || "").trim().toUpperCase();
    if (!normalizedSymbol) return;
    item.disabled = true;
    try {
      const result = await apiFetch(
        `${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/account-config`,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            instrument_key: `${exchangeName}:futures:${normalizedSymbol}`,
            margin_mode: "isolated",
            leverage: "1",
          }),
        }
      );
      setStatus(
        t("settings.exchange.account_configured", {
          margin: result.observed_margin_mode || result.target_margin_mode,
          leverage: result.observed_leverage || result.target_leverage,
        }),
        result.sync_status === "fresh"
      );
      const exchangeKeys = await loadJson(exchangeKeysPath(root), { items: [] });
      renderExchangeKeys(exchangeKeys);
    } catch (error) {
      setStatus(errorMessage(error), false);
    } finally {
      item.disabled = false;
    }
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
    if (confirmation !== "DISCONNECT") return;
    item.disabled = true;
    try {
      await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/disable`, {
        method: "POST",
      });
    } catch (error) {
      setStatus(errorMessage(error), false);
    }
    const exchangeKeys = await loadJson(exchangeKeysPath(root), { items: [] });
    renderExchangeKeys(exchangeKeys);
    item.disabled = false;
  });
  on(root, "click", "[data-exchange-archive]", async (_event, item) => {
    const connectionId = item.dataset.exchangeArchive;
    if (!connectionId) return;
    const confirmation = window.prompt(t("settings.exchange.confirm_archive"));
    if (confirmation !== "ARCHIVE") return;
    item.disabled = true;
    try {
      await apiFetch(`${endpoint(root, "exchangeKeysEndpoint")}/${connectionId}/archive`, {
        method: "POST",
      });
    } catch (error) {
      setStatus(errorMessage(error), false);
    }
    const exchangeKeys = await loadJson(exchangeKeysPath(root), { items: [] });
    renderExchangeKeys(exchangeKeys);
    item.disabled = false;
  });
  on(root, "click", "[data-exchange-status-filter]", async (_event, item) => {
    const status = item.dataset.exchangeStatusFilter || "active";
    setExchangeStatusFilter(root, status);
    const exchangeKeys = await loadJson(exchangeKeysPath(root, status), { items: [] });
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
    openExchangeFormModal();
  });
  on(root, "click", "[data-exchange-form-close]", () => {
    closeExchangeFormModal();
  });
  document.addEventListener("keydown", (event) => {
    const modal = qs("[data-exchange-form-modal]");
    if (event.key === "Escape" && modal instanceof HTMLElement && !modal.hidden) {
      event.preventDefault();
      closeExchangeFormModal();
      return;
    }
    if (event.key === "Tab" && modal instanceof HTMLElement && !modal.hidden) {
      const focusable = Array.from(modal.querySelectorAll(
        'button:not([disabled]):not([tabindex="-1"]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
      )).filter((node) => node instanceof HTMLElement && !node.hidden);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  on(root, "click", "[data-exchange-name-option]", (_event, item) => {
    state.exchangeName = item.dataset.exchangeNameOption || "binance";
    setDropdownValue("[data-exchange-name-current]", state.exchangeName);
    closeDropdownForItem(item);
  });
  on(root, "change", "[data-market-type-checkbox]", (_event, item) => {
    const form = item.closest("[data-exchange-form]") || root;
    syncMarketTypesFromControls(form);
    resetMarketResults();
  });
  on(root, "click", "[data-environment-option]", (_event, item) => {
    setEnvironmentValue(root, item.dataset.environmentOption || "mainnet");
    resetMarketResults();
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
    const marketTypes = syncMarketTypesFromControls(form);
    const status = qs("[data-exchange-form-status]");
    if (!marketTypes.length) {
      resetMarketResults();
      if (status) status.textContent = t("settings.exchange.market_required");
      return;
    }
    const submitButton = qs('button[type="submit"]', form);
    if (submitButton instanceof HTMLButtonElement) submitButton.disabled = true;
    const payload = {
      exchange_name: state.exchangeName,
      market_type: marketTypes[0],
      market_types: marketTypes,
      environment: state.environment,
      label: data.get("label") || null,
      permissions: "trade",
      api_key: secretInputValue(form, "[data-exchange-api-key-input]"),
      api_secret: secretInputValue(form, "[data-exchange-api-secret-input]"),
    };
    try {
      const saved = await apiFetch(endpoint(root, "exchangeKeysEndpoint"), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      clearSecretInputs(form);
      const savedItems = savedExchangeItems(saved);
      const hasReadyItem = savedItems.some((item) => isTradingReady(item));
      const allReady = savedItems.length > 0 && savedItems.every((item) => isTradingReady(item));
      const nextStatus = hasReadyItem ? "active" : "history";
      setExchangeStatusFilter(root, nextStatus);
      const exchangeKeys = await loadJson(exchangeKeysPath(root, nextStatus), { items: [] });
      renderExchangeKeys(exchangeKeys);
      renderMarketResults(saved);
      if (status) {
        status.textContent = allReady
          ? t("settings.exchange.saved")
          : t("settings.exchange.validation_complete");
      }
      if (marketTypes.length === 1 && allReady) {
        form.reset();
        syncMarketTypesFromControls(form);
        resetMarketResults();
        closeExchangeFormModal({ clearSecrets: false });
      }
    } catch (error) {
      clearSecretInputs(form);
      resetMarketResults();
      if (status) status.textContent = errorMessage(error);
      if (error?.outcomeUnknown) {
        const exchangeKeys = await loadJson(exchangeKeysPath(root, "all"), { items: [] });
        setExchangeStatusFilter(root, "history");
        renderExchangeKeys(exchangeKeys);
        closeExchangeFormModal({ clearSecrets: false });
        setStatus(t("settings.mutation.unknown"), false);
      }
    } finally {
      if (submitButton instanceof HTMLButtonElement) submitButton.disabled = false;
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
      const exchangeKeys = await loadJson(exchangeKeysPath(root), { items: [] });
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
  setSettingsTab(root, initialSettingsTab());
  initEvents(root);
  initForms(root);
  void initSettings(root);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initSettingsPage);
} else {
  initSettingsPage();
}
