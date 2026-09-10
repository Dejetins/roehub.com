import 'vite/modulepreload-polyfill';
import { createRoot } from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createBrowserRouter, RouterProvider } from 'react-router';
import { I18nextProvider } from 'react-i18next';
import { z } from 'zod';
import { App } from './app';
import { createI18n } from './i18n';
import { createQueryClient } from './query-client';
import './style.css';

const bootstrap = z.object({ locale: z.enum(['ru', 'en']), subject: z.string().min(1) })
  .parse(JSON.parse(document.getElementById('platform-bootstrap')!.textContent!));
const queryClient = createQueryClient();
// Pages are fully reloaded for SSR/logout. Do not retain queries in the bfcache.
window.addEventListener('pagehide', () => { void queryClient.cancelQueries(); queryClient.clear(); });
window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload(); });
createRoot(document.getElementById('platform-root')!).render(
  <I18nextProvider i18n={createI18n(bootstrap.locale)}>
    <QueryClientProvider client={queryClient}><RouterProvider router={createBrowserRouter([{path: "*", element: <App bootstrap={bootstrap} />}])}/></QueryClientProvider>
  </I18nextProvider>,
);
