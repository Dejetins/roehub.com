import type { NavigatorBootstrap } from './navigator-bootstrap';
export function clientRoute(path: string, bootstrap: NavigatorBootstrap) {
  const url = new URL(path, window.location.origin);
  if (url.origin !== window.location.origin) return false;
  const routes = bootstrap.client_routes ?? ['/backtests'];
  if (url.pathname === '/dashboard') return routes.includes('/dashboard');
  if (/^\/settings\/(profile|preferences|notifications|security)\/?$/.test(url.pathname)) return routes.includes('/settings');
  if (url.pathname === '/connections') return routes.includes('/connections');
  if (/^\/data(?:\/ingestion)?\/?$/.test(url.pathname)) return routes.includes('/data');
  if (/^\/monitoring(?:\/[^/]+)?\/?$/.test(url.pathname)) return routes.includes('/monitoring');
  if (/^\/backtests(?:\/[^/]+)?\/?$/.test(url.pathname)) return routes.includes('/backtests');
  return routes.includes('/strategies') && /^\/strategies(?:\/[^/]+)?\/?$/.test(url.pathname) &&
    url.pathname.replace(/\/$/, '') !== '/strategies/new' && url.searchParams.getAll('view').at(-1) !== 'classic' &&
    ['', 'dashboard'].includes(url.searchParams.getAll('mode').at(-1) ?? '');
}
