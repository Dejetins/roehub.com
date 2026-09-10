import '@testing-library/jest-dom/vitest';
import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, useLocation, useNavigate } from 'react-router';
import { I18nextProvider } from 'react-i18next';
import { App } from './app';
import { createQueryClient } from './query-client';
import { createI18n } from './i18n';
import { jobSchema, libraryParams } from './library-api';
import { ApiError } from './api';

const id = '10000000-0000-4000-8000-000000000001';
const secondId = '10000000-0000-4000-8000-000000000002';
const now = new Date().toISOString();
const job = { job_id: id, state: 'succeeded', created_at: now, updated_at: now, generated_at: now,
  refresh_status: 'terminal', next_allowed_refresh_at: now, retry_after_seconds: 0, cancel_requested_at: null,
  progress: { percent: 100, processed_units: 1, total_units: 1, updated_at: now, pipeline_stage: 'done' },
  request: { coordinates: { exchange: 'binance', market_type: 'spot', symbol: 'BTCUSDT' },
    timeframe: '15m', time_range: { start: now, end: now }, risk_mode: 'none', ui_metadata: { strategy_name: 'First research' } } };
const workstation = { generated_at: now, refresh_status: 'fresh', next_allowed_refresh_at: now, retry_after_seconds: 0,
  sources: [{ name: 'backtest_jobs', status: 'unavailable' }], job_table: { state: 'unavailable' } };
const session = { user_id: 'actor', paid_level: 'free' };
function response(value: unknown, status = 200) { return new Response(JSON.stringify(value), { status }); }
function mockApi(handler?: (url: URL, init: RequestInit) => Promise<Response> | Response | undefined) {
  const fetch = vi.fn((input: URL, init: RequestInit) => handler?.(input, init) ?? Promise.resolve(response(
    input.pathname.includes('current-user') ? session : input.pathname.includes('workstation') ? workstation :
      input.pathname.endsWith('/jobs') ? { items: [job], next_cursor: null } : job)));
  vi.stubGlobal('fetch', fetch); return fetch;
}
function NavigationProbe() {
  const location = useLocation(); const navigate = useNavigate();
  return <><output data-testid="url">{location.pathname}{location.search}</output>
    <button onClick={() => navigate(`/backtests/${secondId}`)}>Switch job</button></>;
}
function mount(path = '/backtests', locale: 'en' | 'ru' = 'en') {
  const client = createQueryClient();
  render(<I18nextProvider i18n={createI18n(locale)}><QueryClientProvider client={client}>
    <MemoryRouter initialEntries={[path]}><App bootstrap={{ locale, subject: 'actor' }} /><NavigationProbe /></MemoryRouter>
  </QueryClientProvider></I18nextProvider>);
  return client;
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('sanitizes unsupported/duplicate filters without modifying opaque valid cursors', () => {
  expect(libraryParams(new URLSearchParams('state=running&state=failed&risk_mode=none&limit=001&cursor=abc_=-&query=secret&evil=1')).toString())
    .toBe('state=running&risk_mode=none&limit=1&cursor=abc_%3D-');
  expect(libraryParams(new URLSearchParams('state=oops&risk_mode=x&limit=251&cursor=%0A')).toString()).toBe('');
});
it('strips raw metadata, organization and unused nested request data from the cache contract', () => {
  const parsed = jobSchema.parse({ ...job, artifact_metadata: { private: 'secret' }, organization_id: 'private',
    request: { ...job.request, execution: { private: 'secret' } } });
  expect(JSON.stringify(parsed)).not.toContain('secret'); expect(parsed).not.toHaveProperty('organization_id');
});
it('renders direct real-shaped jobs while extended filtering truthfully stays unavailable', async () => {
  mockApi(); mount();
  expect(await screen.findByRole('link', { name: 'First research' })).toBeVisible();
  expect(screen.queryByText(/Stale snapshot/)).toBeNull();
  await userEvent.click(screen.getByText('Search, instrument and date filters'));
  expect(screen.getByRole('searchbox')).toBeDisabled();
  expect(screen.getByText(/server job projection is unavailable/)).toBeVisible();
  expect(screen.getByRole('link', { name: 'New backtest' })).toHaveAttribute('href', '/backtests/new');
});
it('canonicalizes route parameters and preserves an encoded variant on valid job entry', async () => {
  mockApi(); mount(`/backtests/${id}?state=failed&variant=a%2Fb&token=secret`);
  expect(await screen.findByRole('heading', { name: 'First research' })).toBeVisible();
  expect(screen.getByTestId('url')).toHaveTextContent(`/backtests/${id}?state=failed&variant=a%2Fb`);
});
it('reads selected job independently of an empty filtered list', async () => {
  mockApi(url => url.pathname.endsWith('/jobs') ? response({ items: [], next_cursor: null }) : undefined);
  mount(`/backtests/${id}?state=failed`);
  expect(await screen.findByRole('heading', { name: 'First research' })).toBeVisible();
  expect(screen.getByText('No jobs on this page')).toBeVisible();
});
it('does not fetch invalid job IDs', async () => {
  const fetch = mockApi(); mount('/backtests/not-a-uuid');
  expect(await screen.findByText(/Invalid job ID/)).toBeVisible();
  expect(fetch.mock.calls.some(([url]) => url.pathname.endsWith('/not-a-uuid'))).toBe(false);
});
it('keeps next-page navigation on an empty risk-filtered server page', async () => {
  const fetch = mockApi(url => url.pathname.endsWith('/jobs') ? response({ items: [], next_cursor: 'abc_=' }) : undefined);
  mount('/backtests?risk_mode=tp_sl_grid&limit=1');
  expect(await screen.findByText('No matching jobs on this page')).toBeVisible();
  await userEvent.click(screen.getByRole('button', { name: 'Next page' }));
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.searchParams.get('cursor') === 'abc_=')).toBe(true));
});
it.each([403, 404, 503, 422])('presents a job read %i without a false successful detail', async status => {
  mockApi(url => url.pathname.endsWith(`/${id}`) ? response({}, status) : undefined); mount(`/backtests/${id}`);
  expect(await screen.findByRole('alert')).toBeVisible();
  expect(screen.queryByRole('heading', { name: 'First research' })).toBeNull();
});
it('retains last valid list on a transient refresh error with a stale label', async () => {
  let failed = false;
  mockApi(url => url.pathname.endsWith('/jobs') && failed ? response({}, 503) : undefined);
  const client = mount(); await screen.findByRole('link', { name: 'First research' }); failed = true;
  await act(() => client.refetchQueries({ queryKey: ['private', 'actor', 'jobs'] }));
  expect(screen.getByRole('link', { name: 'First research' })).toBeVisible();
  expect(await screen.findByText(/Stale snapshot/)).toBeVisible();
});
it('hides formerly cached list after authorization denial', async () => {
  let denied = false;
  mockApi(url => url.pathname.endsWith('/jobs') && denied ? response({}, 403) : undefined);
  const client = mount(); await screen.findByRole('link', { name: 'First research' }); denied = true;
  await act(() => client.refetchQueries({ queryKey: ['private', 'actor', 'jobs'] }));
  await waitFor(() => expect(screen.queryByRole('link', { name: 'First research' })).toBeNull());
});
it('honors HTTP 429 cooldown without automatic request retry', async () => {
  const fetch = mockApi(url => url.pathname.endsWith('/jobs') ? new Response('{}', { status: 429, headers: { 'Retry-After': '120' } }) : undefined);
  mount(); await screen.findByRole('alert');
  expect(screen.getAllByRole('button', { name: 'Refresh' })[0]).toBeDisabled();
  expect(fetch.mock.calls.filter(([url]) => url.pathname.endsWith('/jobs'))).toHaveLength(1);
});
it('aborts an obsolete selection and discards even a late successful response', async () => {
  let resolveOld!: (value: Response) => void; let oldSignal: AbortSignal | undefined;
  mockApi((url, init) => url.pathname.endsWith(`/${id}`) ? new Promise(resolve => { resolveOld = resolve; oldSignal = init.signal as AbortSignal; }) :
    url.pathname.endsWith(`/${secondId}`) ? response({ ...job, job_id: secondId, request: { ...job.request, ui_metadata: { strategy_name: 'Second research' } } }) : undefined);
  const client = mount(`/backtests/${id}`);
  await screen.findByText('Loading selected job…');
  await userEvent.click(screen.getByRole('button', { name: 'Switch job' }));
  expect(await screen.findByRole('heading', { name: 'Second research' })).toBeVisible();
  expect(oldSignal?.aborted).toBe(true);
  await act(async () => resolveOld(response(job)));
  expect(screen.queryByRole('heading', { name: 'First research' })).toBeNull();
  expect(client.getQueryData(['private', 'actor', 'job', id])).toBeUndefined();
});
it.each(['changed', 'expired'])('clears all private query state when session is %s and stops reads', async kind => {
  let invalid = false;
  const fetch = mockApi(url => invalid && url.pathname.includes('current-user') ? response(kind === 'changed' ? { ...session, user_id: 'other' } : {}, kind === 'changed' ? 200 : 401) : undefined);
  const client = mount(); await screen.findByRole('link', { name: 'First research' }); invalid = true;
  await act(() => client.refetchQueries({ queryKey: ['session'] }));
  await screen.findByRole('alert');
  expect(client.getQueryCache().findAll({ queryKey: ['private'] })).toHaveLength(0);
  const before = fetch.mock.calls.length;
  await act(() => client.refetchQueries({ queryKey: ['private'] }));
  expect(fetch.mock.calls).toHaveLength(before);
});
it('treats a protected API 401 as session expiry and clears cached siblings', async () => {
  mockApi(); const client = mount(); await screen.findByRole('link', { name: 'First research' });
  await act(async () => { await client.fetchQuery({ queryKey: ['private', 'probe'], queryFn: () => { throw new ApiError('unauthenticated', 401, 'failed'); } }).catch(() => undefined); });
  expect(await screen.findByRole('link', { name: 'Sign in' })).toBeVisible();
  expect(client.getQueryCache().findAll({ queryKey: ['private'] })).toHaveLength(0);
});

it('keeps old generated snapshots visible and honors the server refresh deadline', async () => {
  mockApi(url => url.pathname.endsWith('/jobs') ? response({ items: [{ ...job, generated_at: new Date(Date.now() - 120000).toISOString(),
    retry_after_seconds: 120, next_allowed_refresh_at: new Date(Date.now() + 120000).toISOString() }], next_cursor: null }) : undefined);
  mount(); await screen.findByRole('link', { name: 'First research' });
  expect(screen.getAllByText(/Stale snapshot/).length).toBeGreaterThan(0);
  expect(screen.getAllByRole('button', { name: 'Refresh' })[0]).toBeDisabled();
});
it('cancels an in-flight private read on subject change and rejects its late response', async () => {
  let resolveOld!: (value: Response) => void; let signal: AbortSignal | undefined; let changed = false;
  mockApi((url, init) => url.pathname.endsWith('/jobs') ? new Promise(resolve => { resolveOld = resolve; signal = init.signal as AbortSignal; }) :
    url.pathname.includes('current-user') && changed ? response({ ...session, user_id: 'other' }) : undefined);
  const client = mount(); await screen.findByText('Loading jobs…'); changed = true;
  await act(() => client.refetchQueries({ queryKey: ['session'] })); await screen.findByRole('alert');
  expect(signal?.aborted).toBe(true);
  await act(async () => resolveOld(response({ items: [job], next_cursor: null })));
  expect(client.getQueryCache().findAll({ queryKey: ['private'] })).toHaveLength(0);
  expect(screen.queryByRole('link', { name: 'First research' })).toBeNull();
});
