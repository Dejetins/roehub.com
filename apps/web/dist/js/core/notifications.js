import { qs } from "./dom.js";

export function notify(message, options = {}) {
  const documentRef = options.document || document;
  const region = options.region || qs("[data-notification-region]", documentRef);
  if (!region) {
    return null;
  }

  const item = documentRef.createElement("div");
  item.className = `rh-notification rh-notification--${options.tone || "info"}`;
  item.setAttribute("role", options.role || "status");
  item.textContent = message;
  region.append(item);

  const timeoutMs = options.timeoutMs ?? 5000;
  if (timeoutMs > 0) {
    window.setTimeout(() => item.remove(), timeoutMs);
  }
  return item;
}

export function clearNotifications(options = {}) {
  const documentRef = options.document || document;
  const region = options.region || qs("[data-notification-region]", documentRef);
  region?.replaceChildren();
}
