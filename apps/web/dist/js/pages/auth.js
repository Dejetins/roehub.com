const modal = document.querySelector("[data-auth-modal]");
let lastFocusedElement = null;

function isFocusable(element) {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  if (element.hidden || element.getAttribute("aria-hidden") === "true") {
    return false;
  }
  return !element.hasAttribute("disabled") && element.tabIndex >= 0;
}

function getFocusableElements(root) {
  return Array.from(
    root.querySelectorAll("a[href], button, input, select, textarea, [tabindex]")
  ).filter(isFocusable);
}

function openModal(trigger) {
  if (!modal) {
    return;
  }
  lastFocusedElement = trigger instanceof HTMLElement ? trigger : document.activeElement;
  modal.classList.add("auth-modal--open");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("auth-modal-lock");
  const dialog = modal.querySelector("[role='dialog']");
  const primary = modal.querySelector("[data-auth-primary]");
  if (primary instanceof HTMLElement) {
    primary.focus();
  } else if (dialog instanceof HTMLElement) {
    dialog.focus();
  }
}

function closeModal() {
  if (!modal) {
    return;
  }
  modal.classList.remove("auth-modal--open");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("auth-modal-lock");
  if (lastFocusedElement instanceof HTMLElement) {
    lastFocusedElement.focus();
  }
}

function trapFocus(event) {
  if (!modal || !modal.classList.contains("auth-modal--open") || event.key !== "Tab") {
    return;
  }
  const focusableElements = getFocusableElements(modal);
  if (focusableElements.length === 0) {
    event.preventDefault();
    return;
  }
  const first = focusableElements[0];
  const last = focusableElements[focusableElements.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

document.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) {
    return;
  }
  const openTrigger = event.target.closest("[data-auth-open]");
  if (openTrigger) {
    event.preventDefault();
    openModal(openTrigger);
    return;
  }
  if (event.target.closest("[data-auth-close]")) {
    event.preventDefault();
    closeModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
    return;
  }
  trapFocus(event);
});

document.querySelectorAll("[data-locale-option]").forEach((option) => {
  option.addEventListener("click", () => {
    if (option instanceof HTMLElement) {
      window.localStorage.setItem("roehub_locale", option.dataset.localeOption || "en");
    }
  });
});

const logoutPanel = document.querySelector("[data-auth-logout]");
if (logoutPanel instanceof HTMLElement) {
  const redirectPath = logoutPanel.dataset.logoutRedirect || "/login";
  fetch("/api/auth/logout", { method: "POST", credentials: "include" })
    .catch(() => null)
    .finally(() => {
      window.location.assign(redirectPath);
    });
}

if (modal?.dataset.openOnLoad === "true") {
  openModal(null);
}
