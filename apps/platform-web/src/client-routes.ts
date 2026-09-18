import type { ClientBootstrap } from '@roehub/web-contracts';
export function clientRoute(path: string, bootstrap: ClientBootstrap) {
  const url = new URL(path, window.location.origin);
  if (url.origin !== window.location.origin) return false;
  const routes = bootstrap.client_routes ?? ['/backtests'];
  if (/^\/backtests(?:\/[^/]+)?\/?$/.test(url.pathname)) return routes.includes('/backtests');
  return routes.includes('/strategies') && /^\/strategies(?:\/[^/]+)?\/?$/.test(url.pathname) &&
    url.pathname.replace(/\/$/, '') !== '/strategies/new' && url.searchParams.getAll('view').at(-1) !== 'classic' &&
    ['', 'dashboard'].includes(url.searchParams.getAll('mode').at(-1) ?? '');
}
