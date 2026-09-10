import type { ApiFailureKind, FieldIssue, SessionIdentity } from '@roehub/web-contracts';
import { z } from 'zod';

const admissionSchema = z.object({ limit_scope: z.string().optional(), limit: z.number().optional(), requested: z.number().optional(), used: z.number().optional() });

export class ApiError extends Error {
  constructor(
    public readonly kind: ApiFailureKind,
    public readonly status: number | null,
    public readonly outcome: 'failed' | 'unknown',
    public readonly retryAfterSeconds: number | null = null,
    public readonly issues: readonly FieldIssue[] = [],
    public readonly code: string | null = null,
    public readonly admission: z.infer<typeof admissionSchema> | null = null,
  ) { super(kind); }
}

const errorEnvelope = z.object({ error: z.object({ code: z.string().optional(), details: z.object({
  ...admissionSchema.shape,
  retry_after_seconds: z.number().finite().nonnegative().optional(),
  errors: z.array(z.object({
    path: z.string(), code: z.string(), message: z.string(),
  })).optional(),
}).optional() }).optional() });

function failureKind(status: number): ApiFailureKind {
  const kinds: Partial<Record<number, ApiFailureKind>> = {401: 'unauthenticated', 403: 'forbidden', 404: 'not-found',
    409: 'conflict', 422: 'validation', 429: 'rate-limited'};
  return kinds[status] ?? 'unavailable';
}

export interface ApiReply<T> { status: number; data: T; retryAfterSeconds: number | null }
export interface RequestOptions {
  method?: 'GET' | 'POST' | 'DELETE' | 'PATCH' | 'PUT';
  body?: unknown;
  signal?: AbortSignal;
  idempotencyKey?: string;
  timeoutMs?: number;
}

/** Exactly one same-origin request. Commands are never retried here. */
export async function requestJson<T>(
  path: string, schema: z.ZodType<T>, options: RequestOptions = {},
): Promise<ApiReply<T>> {
  // Only relative API paths; reject normalization that could leave the proxy.
  const url = new URL(path, window.location.origin);
  if (!path.startsWith('/api/') || url.origin !== window.location.origin ||
      !url.pathname.startsWith('/api/') || /[\\\r\n]/.test(path)) {
    throw new TypeError('Expected a same-origin /api/ path');
  }
  const method = options.method ?? 'GET';
  const command = method !== 'GET';
  const timeout = options.timeoutMs ?? 15_000;
  if (!Number.isFinite(timeout) || timeout < 1 || timeout > 60_000) {
    throw new TypeError('Request timeout must be between 1 and 60000 ms');
  }
  const controller = new AbortController();
  const abort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, {
      method, credentials: 'same-origin', cache: 'no-store', redirect: 'error',
      signal: controller.signal,
      headers: { Accept: 'application/json',
        ...(options.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(options.idempotencyKey ? { 'Idempotency-Key': options.idempotencyKey } : {}),
      },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
    });
    // Status alone is authoritative for session expiry; a broken/hanging body
    // must not delay closing the private UI or turn a known 401 into transport.
    if (response.status === 401) {
      window.dispatchEvent(new Event('roehub:unauthenticated'));
      controller.abort();
      throw new ApiError('unauthenticated', 401, 'failed');
    }
    const rawDelay = response.headers.get('Retry-After');
    const delay = rawDelay !== null && /^\d+$/.test(rawDelay) ? Number(rawDelay) : null;
    const retryAfter = delay !== null && Number.isFinite(delay) ? delay : null;
    let payload: unknown;
    if (response.status !== 204) {
      try { payload = await response.json(); }
      catch (error) { if (!(error instanceof SyntaxError)) throw error; }
    }
    if (controller.signal.aborted) throw controller.signal.reason;
    if (!response.ok) {
      const parsed = errorEnvelope.safeParse(payload);
      throw new ApiError(failureKind(response.status), response.status,
        command && response.status >= 500 ? 'unknown' : 'failed',
        Math.max(retryAfter ?? 0, parsed.success ? parsed.data.error?.details?.retry_after_seconds ?? 0 : 0) || null,
        parsed.success ? parsed.data.error?.details?.errors ?? [] : [],
        parsed.success ? parsed.data.error?.code ?? null : null,
        parsed.success && parsed.data.error?.details ? admissionSchema.parse(parsed.data.error.details) : null);
    }
    const parsed = schema.safeParse(payload);
    if (!parsed.success) {
      throw new ApiError('invalid-response', response.status, command ? 'unknown' : 'failed');
    }
    return { status: response.status, data: parsed.data, retryAfterSeconds: retryAfter };
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (!command && options.signal?.aborted) throw error;
    throw new ApiError('transport', null, command ? 'unknown' : 'failed');
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', abort);
  }
}

const sessionSchema = z.object({ user_id: z.string().min(1), paid_level: z.string().min(1) });
export async function readSession(signal?: AbortSignal): Promise<SessionIdentity> {
  return (await requestJson('/api/auth/current-user', sessionSchema, { signal })).data;
}

/** Organization is intentionally absent: current-user cannot establish replay scope. */
export function loginContinuation(path: string): string {
  const url = new URL(path, window.location.origin);
  const safe = path.startsWith('/') && !path.startsWith('//') &&
    !/[\\\r\n]/.test(path) && url.origin === window.location.origin
    ? `${url.pathname}${url.search}` : '/backtests';
  return `/login?${new URLSearchParams({ next: safe })}`;
}
