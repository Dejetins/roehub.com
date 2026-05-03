import { translate } from "./locale.js";

export const DEFAULT_TIMEOUT_MS = 15000;
export const MUTATION_METHODS = Object.freeze(["DELETE", "PATCH", "POST", "PUT"]);

let csrfTokenProvider = () => null;

export class RoehubApiError extends Error {
  constructor({ status, code, message, payload = null, fieldErrors = {}, retryable = false }) {
    super(message);
    this.name = "RoehubApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
    this.fieldErrors = fieldErrors;
    this.retryable = retryable;
  }
}

export function setCsrfTokenProvider(provider) {
  if (typeof provider !== "function") {
    throw new TypeError("CSRF token provider must be a function");
  }
  csrfTokenProvider = provider;
}

export async function apiRequest(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const headers = new Headers(options.headers || {});

  let timedOut = false;
  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  applyCsrfHeader({ headers, method });

  try {
    const response = await fetch(path, {
      method,
      credentials: "include",
      headers,
      body: serializeBody(options.body),
      signal: controller.signal,
    });
    const payload = await parsePayload(response);
    if (!response.ok) {
      throw buildHttpError(response, payload, options);
    }
    return payload;
  } catch (error) {
    if (error instanceof RoehubApiError) {
      throw error;
    }
    if (timedOut || error?.name === "AbortError") {
      throw new RoehubApiError({
        status: 0,
        code: timedOut ? "timeout" : "aborted",
        message: timedOut ? translate("js.error.timeout") : translate("js.error.network"),
        retryable: timedOut,
      });
    }
    throw new RoehubApiError({
      status: 0,
      code: "network_error",
      message: translate("js.error.network"),
      retryable: true,
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function applyCsrfHeader({ headers, method }) {
  if (!MUTATION_METHODS.includes(method)) {
    return;
  }
  const provided = csrfTokenProvider({ method, headers });
  if (!provided) {
    return;
  }
  if (typeof provided === "string") {
    headers.set("X-CSRF-Token", provided);
    return;
  }
  if (provided.headerName && provided.headerValue) {
    headers.set(provided.headerName, provided.headerValue);
  }
}

function serializeBody(body) {
  if (body === undefined || body === null || typeof body === "string" || body instanceof FormData) {
    return body;
  }
  return JSON.stringify(body);
}

async function parsePayload(response) {
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function buildHttpError(response, payload, options) {
  const status = response.status;
  const code = mapStatusCode(status);
  const message = extractErrorMessage(payload, code);
  const fieldErrors = extractFieldErrors(payload);

  if (status === 401 && options.redirectOnUnauthorized !== false) {
    redirectToLogin();
  }

  return new RoehubApiError({
    status,
    code,
    message,
    payload,
    fieldErrors,
    retryable: status === 429 || status >= 500,
  });
}

function mapStatusCode(status) {
  if (status === 401) {
    return "unauthorized";
  }
  if (status === 403) {
    return "forbidden";
  }
  if (status === 409) {
    return "conflict";
  }
  if (status === 422) {
    return "validation_error";
  }
  return status >= 500 ? "server_error" : "request_error";
}

function extractErrorMessage(payload, code) {
  const envelopeMessage = payload?.error?.message || payload?.message || payload?.detail;
  if (typeof envelopeMessage === "string" && envelopeMessage.trim()) {
    return envelopeMessage;
  }
  if (code === "unauthorized") {
    return translate("js.error.unauthorized");
  }
  if (code === "forbidden") {
    return translate("js.error.forbidden");
  }
  if (code === "conflict") {
    return translate("js.error.conflict");
  }
  if (code === "validation_error") {
    return translate("js.error.validation");
  }
  return translate("js.error.network");
}

function extractFieldErrors(payload) {
  const fields = payload?.error?.field_errors || payload?.field_errors;
  return fields && typeof fields === "object" ? fields : {};
}

function redirectToLogin() {
  if (!window.location) {
    return;
  }
  const next = `${window.location.pathname}${window.location.search}`;
  window.location.assign(`/login?next=${encodeURIComponent(next)}`);
}
