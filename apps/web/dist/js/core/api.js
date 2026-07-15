export class RoehubApiError extends Error {
  constructor(message, { status = 0, code = "request_failed", payload = null, retryAfterSeconds = null, outcomeUnknown = false } = {}) {
    super(message);
    this.name = "RoehubApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
    this.retryAfterSeconds = retryAfterSeconds;
    this.outcomeUnknown = outcomeUnknown;
  }
}

export function createCsrfProvider(provider) {
  let csrfProvider = typeof provider === "function" ? provider : null;
  return {
    set(nextProvider) {
      csrfProvider = typeof nextProvider === "function" ? nextProvider : null;
    },
    get() {
      return csrfProvider ? csrfProvider() : null;
    },
  };
}

export const csrf = createCsrfProvider();

function isMutation(method) {
  return !["GET", "HEAD", "OPTIONS"].includes(method.toUpperCase());
}

function normalizeErrorPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return {};
  }
  const source = payload.error && typeof payload.error === "object" ? payload.error : payload;
  return {
    code: source.code || source.error_code || (typeof source.error === "string" ? source.error : null) || "request_failed",
    message: source.message || source.detail || "Request failed",
    fieldErrors: source.field_errors || source.errors || null,
    retryAfterSeconds: source.retry_after_seconds ?? null,
  };
}

async function parsePayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

export async function apiFetch(path, options = {}) {
  const method = options.method || "GET";
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 15000;
  const controller = new AbortController();
  const externalSignal = options.signal || null;
  let removeExternalAbort = null;
  let abortKind = "";
  const timeoutId = window.setTimeout(() => {
    abortKind = "timeout";
    controller.abort(new DOMException("Request timed out", "AbortError"));
  }, timeoutMs);
  const headers = new Headers(options.headers || {});

  if (externalSignal) {
    const abortFromExternalSignal = () => {
      abortKind = abortKind || "aborted";
      controller.abort(new DOMException("Request aborted", "AbortError"));
    };
    if (externalSignal.aborted) {
      abortFromExternalSignal();
    } else {
      externalSignal.addEventListener("abort", abortFromExternalSignal, { once: true });
      removeExternalAbort = () => externalSignal.removeEventListener("abort", abortFromExternalSignal);
    }
  }

  if (isMutation(method)) {
    if (!headers.has("x-request-id")) {
      headers.set(
        "x-request-id",
        globalThis.crypto?.randomUUID?.() || `web-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      );
    }
    const token = csrf.get();
    if (token) {
      headers.set("x-csrf-token", token);
    }
  }

  try {
    const response = await fetch(path, {
      ...options,
      method,
      headers,
      credentials: "include",
      signal: controller.signal,
    });

    const payload = await parsePayload(response);
    if (response.ok) {
      return payload;
    }

    const normalized = normalizeErrorPayload(payload);
    const retryAfter = response.headers.get("retry-after");
    const retryAfterSeconds = normalized.retryAfterSeconds ?? (retryAfter ? Number(retryAfter) : null);
    const codeByStatus = {
      401: "unauthenticated",
      403: "forbidden",
      409: "conflict",
      422: "validation_error",
    };
    const errorCode =
      normalized.code && normalized.code !== "request_failed"
        ? normalized.code
        : codeByStatus[response.status] || "request_failed";
    throw new RoehubApiError(normalized.message, {
      status: response.status,
      code: errorCode,
      payload,
      retryAfterSeconds,
      outcomeUnknown: isMutation(method) && response.status >= 500,
    });
  } catch (error) {
    if (error instanceof RoehubApiError) {
      if (error.status === 401) {
        document.dispatchEvent(new CustomEvent("roehub:auth-required", { detail: error }));
      }
      throw error;
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      const isTimeout = abortKind === "timeout";
      throw new RoehubApiError(isTimeout ? "Request timed out" : "Request aborted", {
        code: isTimeout ? "timeout" : "aborted",
        outcomeUnknown: isMutation(method),
      });
    }
    throw new RoehubApiError(error?.message || "Network request failed", {
      code: "network_error",
      outcomeUnknown: isMutation(method),
    });
  } finally {
    window.clearTimeout(timeoutId);
    removeExternalAbort?.();
  }
}
