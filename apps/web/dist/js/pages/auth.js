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
  fetch("/api/auth/local/logout", {
    method: "POST",
    credentials: "include",
    headers: csrfHeaders(),
  })
    .catch(() => null)
    .finally(() => {
      window.location.assign(redirectPath);
    });
}

if (modal?.dataset.openOnLoad === "true") {
  openModal(null);
}

function decodeBase64Url(value) {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const decoded = window.atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "="));
  return Uint8Array.from(decoded, (character) => character.charCodeAt(0));
}

function encodeBase64Url(value) {
  const bytes = new Uint8Array(value);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return window.btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function creationOptions(publicKey) {
  return {
    ...publicKey,
    challenge: decodeBase64Url(publicKey.challenge),
    user: { ...publicKey.user, id: decodeBase64Url(publicKey.user.id) },
    excludeCredentials: (publicKey.excludeCredentials || []).map((item) => ({
      ...item,
      id: decodeBase64Url(item.id),
    })),
  };
}

function requestOptions(publicKey) {
  return {
    ...publicKey,
    challenge: decodeBase64Url(publicKey.challenge),
    allowCredentials: (publicKey.allowCredentials || []).map((item) => ({
      ...item,
      id: decodeBase64Url(item.id),
    })),
  };
}

function serializeCredential(credential) {
  const response = {
    clientDataJSON: encodeBase64Url(credential.response.clientDataJSON),
  };
  if (credential.response.attestationObject) {
    response.attestationObject = encodeBase64Url(credential.response.attestationObject);
    response.transports = credential.response.getTransports?.() || [];
  }
  if (credential.response.authenticatorData) {
    response.authenticatorData = encodeBase64Url(credential.response.authenticatorData);
    response.signature = encodeBase64Url(credential.response.signature);
    response.userHandle = credential.response.userHandle
      ? encodeBase64Url(credential.response.userHandle)
      : null;
  }
  return {
    id: credential.id,
    rawId: encodeBase64Url(credential.rawId),
    type: credential.type,
    response,
    clientExtensionResults: credential.getClientExtensionResults(),
    authenticatorAttachment: credential.authenticatorAttachment,
  };
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {
    credentials: "include",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail?.error || "request_failed");
  }
  return payload;
}

function csrfHeaders() {
  const cookiePart = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith("roehub_csrf="));
  return cookiePart
    ? { "x-csrf-token": decodeURIComponent(cookiePart.slice("roehub_csrf=".length)) }
    : {};
}

const localAuth = document.querySelector("[data-local-auth]");
if (localAuth instanceof HTMLElement) {
  const loginPanel = localAuth.querySelector("[data-local-login]");
  const bootstrapForm = localAuth.querySelector("[data-local-bootstrap]");
  const status = localAuth.querySelector("[data-local-auth-status]");
  const recoveryPanel = localAuth.querySelector("[data-recovery-codes]");
  const recoveryList = localAuth.querySelector("[data-recovery-code-list]");
  const nextPath = localAuth.dataset.nextPath || "/dashboard";

  const setStatus = (message) => {
    if (status instanceof HTMLElement) {
      status.textContent = message;
    }
  };
  const finishLogin = () => window.location.assign(nextPath);

  fetchJson("/api/auth/local/status")
    .then((payload) => {
      if (bootstrapForm instanceof HTMLFormElement) {
        bootstrapForm.hidden = !payload.bootstrap_required;
      }
      if (loginPanel instanceof HTMLElement) {
        loginPanel.hidden = payload.bootstrap_required;
      }
    })
    .catch(() => setStatus("Сервис входа временно недоступен."));

  const oidcPanel = localAuth.querySelector("[data-oidc-login]");
  fetchJson("/api/auth/oidc/status")
    .then((payload) => {
      const providerName = oidcPanel?.querySelector("[data-oidc-provider-name]");
      if (providerName instanceof HTMLElement) {
        providerName.textContent = payload.display_name;
      }
      if (oidcPanel instanceof HTMLElement) {
        oidcPanel.hidden = false;
      }
    })
    .catch(() => {
      if (oidcPanel instanceof HTMLElement) {
        oidcPanel.hidden = true;
      }
    });

  localAuth.querySelector("[data-passkey-login]")?.addEventListener("click", async () => {
    try {
      setStatus("Подтвердите вход на устройстве.");
      const options = await fetchJson("/api/auth/local/passkey/options", { method: "POST" });
      const credential = await navigator.credentials.get({
        publicKey: requestOptions(options.publicKey),
      });
      await fetchJson("/api/auth/local/passkey/complete", {
        method: "POST",
        body: JSON.stringify({
          challenge_id: options.challenge_id,
          credential: serializeCredential(credential),
        }),
      });
      finishLogin();
    } catch (_error) {
      setStatus("Вход не выполнен. Повторите проверку ключа доступа.");
    }
  });

  if (bootstrapForm instanceof HTMLFormElement) {
    bootstrapForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const requestBody = Object.fromEntries(new FormData(bootstrapForm).entries());
      const ticketFile = requestBody.ticket_file;
      if (!(ticketFile instanceof File) || ticketFile.size === 0) {
        setStatus("Выберите файл одноразового кода, созданный roehubctl.");
        return;
      }
      requestBody.ticket = (await ticketFile.text()).trim();
      delete requestBody.ticket_file;
      if (!requestBody.password) {
        delete requestBody.password;
      }
      try {
        setStatus("Создайте ключ доступа на устройстве.");
        const options = await fetchJson("/api/auth/local/bootstrap/options", {
          method: "POST",
          body: JSON.stringify(requestBody),
        });
        bootstrapForm.reset();
        const credential = await navigator.credentials.create({
          publicKey: creationOptions(options.publicKey),
        });
        const result = await fetchJson("/api/auth/local/bootstrap/complete", {
          method: "POST",
          body: JSON.stringify({
            challenge_id: options.challenge_id,
            credential: serializeCredential(credential),
          }),
        });
        bootstrapForm.hidden = true;
        if (recoveryList instanceof HTMLOListElement) {
          recoveryList.replaceChildren(
            ...result.recovery_codes.map((code) => {
              const item = document.createElement("li");
              item.textContent = code;
              return item;
            })
          );
        }
        if (recoveryPanel instanceof HTMLElement) {
          recoveryPanel.hidden = false;
        }
        setStatus("Владелец создан. Сохраните коды восстановления.");
      } catch (_error) {
        bootstrapForm.reset();
        setStatus("Настройка не завершена. Получите новый одноразовый код и повторите.");
      }
    });
  }

  localAuth.querySelector("[data-recovery-ack]")?.addEventListener("click", () => {
    recoveryList?.replaceChildren();
    if (recoveryPanel instanceof HTMLElement) {
      recoveryPanel.hidden = true;
    }
    finishLogin();
  });

  localAuth.querySelector("[data-password-login]")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    try {
      await fetchJson("/api/auth/local/password", {
        method: "POST",
        body: JSON.stringify(Object.fromEntries(new FormData(form).entries())),
      });
      form.reset();
      finishLogin();
    } catch (_error) {
      form.reset();
      setStatus("Вход не выполнен. Проверьте данные или повторите позже.");
    }
  });

  localAuth.querySelector("[data-recovery-login]")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    try {
      await fetchJson("/api/auth/local/recovery", {
        method: "POST",
        body: JSON.stringify(Object.fromEntries(new FormData(form).entries())),
      });
      form.reset();
      finishLogin();
    } catch (_error) {
      form.reset();
      setStatus("Восстановление не выполнено. Проверьте данные или повторите позже.");
    }
  });
}
