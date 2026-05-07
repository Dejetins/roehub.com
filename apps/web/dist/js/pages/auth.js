const modal = document.querySelector("[data-auth-modal]");
const loginForm = document.querySelector("[data-auth-login-form]");
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
  const initial = modal.querySelector("[data-auth-initial]");
  const primary = modal.querySelector("[data-auth-primary]");
  if (initial instanceof HTMLElement) {
    initial.focus();
  } else if (primary instanceof HTMLElement) {
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

if (loginForm instanceof HTMLFormElement) {
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(loginForm);
    const username = String(formData.get("username") || "").trim();
    const password = String(formData.get("password") || "");
    const nextPath = String(formData.get("next") || "/dashboard");
    const errorElement = loginForm.querySelector("[data-auth-error]");
    const submitButton = loginForm.querySelector("[data-auth-primary]");
    const defaultError = loginForm.dataset.authErrorDefault || "Unable to sign in";

    if (errorElement instanceof HTMLElement) {
      errorElement.hidden = true;
      errorElement.textContent = "";
    }

    if (!username || !password) {
      if (errorElement instanceof HTMLElement) {
        errorElement.textContent = defaultError;
        errorElement.hidden = false;
      }
      return;
    }

    if (submitButton instanceof HTMLButtonElement) {
      submitButton.disabled = true;
    }

    try {
      const response = await fetch("/api/auth/password-login", {
        method: "POST",
        credentials: "include",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
          next: nextPath,
        }),
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch (_error) {
        payload = null;
      }
      if (!response.ok) {
        const detail = payload && typeof payload === "object" ? payload.detail : null;
        const message =
          detail && typeof detail === "object" && typeof detail.message === "string"
            ? detail.message
            : defaultError;
        throw new Error(message);
      }
      const redirectPath =
        payload && typeof payload === "object" && typeof payload.next === "string"
          ? payload.next
          : "/dashboard";
      window.location.assign(redirectPath);
    } catch (error) {
      if (errorElement instanceof HTMLElement) {
        errorElement.textContent = error instanceof Error ? error.message : defaultError;
        errorElement.hidden = false;
      }
      if (submitButton instanceof HTMLButtonElement) {
        submitButton.disabled = false;
      }
    }
  });
}

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
