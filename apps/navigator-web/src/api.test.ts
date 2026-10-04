import { afterEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { loginContinuation, requestJson } from './api';
import { createQueryClient } from './query-client';

afterEach(() => vi.unstubAllGlobals());
describe('single-request transport', () => {
  it('uses same-origin cookies and propagates 202 / delay', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{"pending":true}', { status: 202, headers: { 'Retry-After': '4' } }));
    vi.stubGlobal('fetch', fetch);
    expect(await requestJson('/api/backtests/jobs/x', z.object({ pending: z.boolean() })))
      .toEqual({ status: 202, data: { pending: true }, retryAfterSeconds: 4 });
    expect(fetch.mock.calls[0][1]).toMatchObject({ credentials: 'same-origin', redirect: 'error', cache: 'no-store' });
  });
  it.each(['https://elsewhere.test/api/x', '//elsewhere.test/api/x', '/api/../auth', '/api/\\evil'])('rejects a path outside the proxy: %s', async path => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await expect(requestJson(path, z.unknown())).rejects.toThrow('same-origin');
    expect(fetch).not.toHaveBeenCalled();
  });
  it.each([401, 403, 404, 409, 422, 429, 503])('preserves HTTP %i without retrying', async status => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}', { status })); vi.stubGlobal('fetch', fetch);
    await expect(requestJson('/api/backtests/jobs', z.unknown())).rejects.toMatchObject({ status, outcome: 'failed' });
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it('lost command response is unknown and never retried', async () => {
    const fetch = vi.fn().mockRejectedValue(new TypeError('network')); vi.stubGlobal('fetch', fetch);
    await expect(requestJson('/api/backtests/jobs', z.unknown(), { method: 'POST', body: {}, idempotencyKey: 'logical-attempt' }))
      .rejects.toMatchObject({ kind: 'transport', outcome: 'unknown' });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(createQueryClient().getDefaultOptions().mutations?.retry).toBe(false);
  });
  it('invalid command response retains unknown outcome', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not JSON', { status: 201 })));
    await expect(requestJson('/api/backtests/jobs', z.object({ job_id: z.string() }), { method: 'POST' }))
      .rejects.toMatchObject({ kind: 'invalid-response', outcome: 'unknown' });
  });
  it('forwards cancellation and bounds a stalled read', async () => {
    const fetch = vi.fn((_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })); vi.stubGlobal('fetch', fetch);
    await expect(requestJson('/api/backtests/jobs', z.unknown(), { timeoutMs: 5 })).rejects.toMatchObject({ kind: 'transport' });
    const controller = new AbortController();
    const read = requestJson('/api/backtests/jobs', z.unknown(), { signal: controller.signal });
    controller.abort();
    await expect(read).rejects.toMatchObject({ name: 'AbortError' });
  });
  it('keeps encoded variant and drops unsafe continuation', () => {
    expect(new URLSearchParams(loginContinuation('/backtests/123?variant=a%2Fb').split('?')[1]).get('next')).toBe('/backtests/123?variant=a%2Fb');
    expect(loginContinuation('//evil.test/')).toBe('/login?next=%2Fbacktests');
  });
});

it('retains the real dotted-path 422 field contract and strips raw input/context', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: {
    code: 'validation_error', message: 'Validation failed', details: { errors: [
      { path: 'body.indicators.0.window.start', code: 'greater_than', message: 'Input should be greater than 0', input: -1, ctx: { gt: 0 } },
      { path: 'body.name', code: 'missing', message: 'Field required' },
    ] },
  } }), { status: 422 })));
  await expect(requestJson('/api/backtests/preflight', z.unknown(), { method: 'POST' }))
    .rejects.toMatchObject({ status: 422, code: 'validation_error', issues: [
      { path: 'body.indicators.0.window.start', code: 'greater_than', message: 'Input should be greater than 0' },
      { path: 'body.name', code: 'missing', message: 'Field required' },
    ] });
});

it('cancels a read during body consumption without returning a late result', async () => {
  const controller = new AbortController();
  vi.stubGlobal('fetch', vi.fn().mockImplementation(async (_url, options) => ({
    status: 200, ok: true, headers: new Headers(),
    json: () => new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      controller.abort();
    }),
  })));
  await expect(requestJson('/api/backtests/jobs', z.unknown(), { signal: controller.signal }))
    .rejects.toMatchObject({ name: 'AbortError' });
});
it('honors admission-body retry delay when there is no Retry-After header', async () => {
 const fetch=vi.fn().mockResolvedValue(new Response(JSON.stringify({error:{code:'backtest.rate_limited',details:{retry_after_seconds:60,limit_scope:'tier',limit:2,used:2}}}),{status:429}));vi.stubGlobal('fetch',fetch);
 await expect(requestJson('/api/backtests/preflight',z.unknown(),{method:'POST',body:{}})).rejects.toMatchObject({status:429,retryAfterSeconds:60,outcome:'failed'});expect(fetch).toHaveBeenCalledTimes(1);
});
it('signals a known 401 before touching a rejected response body',async()=>{
 const json=vi.fn().mockRejectedValue(new TypeError('response body disconnected'));
 vi.stubGlobal('fetch',vi.fn().mockResolvedValue({status:401,ok:false,headers:new Headers(),json}));const expired=vi.fn();window.addEventListener('roehub:unauthenticated',expired);
 try {await expect(requestJson('/api/backtests/jobs',z.unknown(),{method:'POST',body:{}})).rejects.toMatchObject({kind:'unauthenticated',status:401});expect(expired).toHaveBeenCalledTimes(1);expect(json).not.toHaveBeenCalled();}
 finally {window.removeEventListener('roehub:unauthenticated',expired);}
});
