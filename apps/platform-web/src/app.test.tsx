import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router';
import { I18nextProvider } from 'react-i18next';
import { App } from './app';
import { createQueryClient } from './query-client';
import { createI18n } from './i18n';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function renderApp() {
  render(<I18nextProvider i18n={createI18n('en')}><QueryClientProvider client={createQueryClient()}>
    <MemoryRouter initialEntries={['/backtests/job?variant=a%2Fb']}><App bootstrap={{ locale: 'en', subject: 'actor' }} /></MemoryRouter>
  </QueryClientProvider></I18nextProvider>);
}
it('shows re-authentication only for 401, preserving the deep link', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })));
  renderApp();
  const link = await screen.findByRole('link', { name: 'Sign in' });
  expect(link.getAttribute('href')).toBe('/login?next=%2Fbacktests%2Fjob%3Fvariant%3Da%252Fb');
});
it('identity outage does not offer logout or expose protected content', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 503 })));
  renderApp();
  expect((await screen.findByRole('alert')).textContent).toContain('unavailable');
  expect(screen.queryByRole('link', { name: 'Sign in' })).toBeNull();
  expect(screen.queryByRole('navigation')).toBeNull();
});
it('a changed subject prevents reuse of the original account view', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"user_id":"other","paid_level":"free"}')));
  renderApp();
  expect((await screen.findByRole('alert')).textContent).toContain('account has changed');
  expect(screen.queryByRole('navigation')).toBeNull();
});
