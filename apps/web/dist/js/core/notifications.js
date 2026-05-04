export function notify(message, { tone = "info", timeoutMs = 5000 } = {}) {
  const region = getNotificationRegion();
  const item = document.createElement("div");
  item.className = `rh-notification rh-notification--${tone}`;
  item.setAttribute("role", tone === "danger" ? "alert" : "status");
  item.textContent = message;
  region.append(item);
  if (timeoutMs > 0) {
    window.setTimeout(() => item.remove(), timeoutMs);
  }
  return item;
}

function getNotificationRegion() {
  let region = document.querySelector("[data-notification-region]");
  if (region instanceof HTMLElement) {
    return region;
  }
  region = document.createElement("div");
  region.className = "rh-notification-region";
  region.dataset.notificationRegion = "true";
  document.body.append(region);
  return region;
}
