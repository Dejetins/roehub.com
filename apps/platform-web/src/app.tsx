import {clearOperationRecovery} from './operation-recovery';
import { StrategiesPage } from './strategies-page';
import { clientRoute } from './client-routes';
import { MotionLink as Link, useDisclosureMotion } from './motion';
import { clearStrategyRecovery, loadStrategyRecovery } from './strategy-recovery';
import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router';
import { ChartNoAxesCombined, Database, LayoutDashboard, LogOut, Settings, SquareChartGantt } from 'lucide-react';
import type { ClientBootstrap } from '@roehub/web-contracts';
import { ApiError, loginContinuation, readSession } from './api';
import { BacktestsPage } from './backtests-page';
import { clearRecovery, loadRecovery } from './recovery';

export function App({ bootstrap }: { bootstrap: ClientBootstrap }) {
  useDisclosureMotion();
  const { t } = useTranslation();
  const location = useLocation();
  const client = useQueryClient();
  const [expired, setExpired] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const session = useQuery({ queryKey: ['session', bootstrap.subject],
    queryFn: ({ signal }) => readSession(signal), enabled: !expired && !leaving,
    refetchInterval: query => expired || leaving || query.state.error ||
      (query.state.data && query.state.data.user_id !== bootstrap.subject) ? false : 30_000,
    refetchOnWindowFocus: query => !expired && !leaving && !query.state.error &&
      (!query.state.data || query.state.data.user_id === bootstrap.subject), retry: false });
  const strategies = location.pathname.startsWith('/strategies');
  const title = strategies ? 'strategies' : 'title';
  const routeAllowed = clientRoute(`${location.pathname}${location.search}`, bootstrap);
  useEffect(() => { if (!routeAllowed) window.location.replace(`${location.pathname}${location.search}`); }, [routeAllowed, location.pathname, location.search]);
  useEffect(() => { document.title = `${t(title)} · RoeHub`; }, [t, title]);
  const next = `${location.pathname}${location.search}`;
  const changed = !!session.data && session.data.user_id !== bootstrap.subject;
  const blocked = expired || leaving || changed || !!session.error;
  useEffect(() => {
    const stop = () => setExpired(true);
    window.addEventListener('roehub:unauthenticated', stop);
    return () => window.removeEventListener('roehub:unauthenticated', stop);
  }, []);
  useEffect(() => { if (changed || leaving) { clearRecovery(); clearStrategyRecovery(); clearOperationRecovery(); } else if (session.data && !blocked) { loadRecovery(bootstrap.subject); loadStrategyRecovery(bootstrap.subject); } }, [changed, leaving, session.data, blocked, bootstrap.subject]);
  useEffect(() => client.getQueryCache().subscribe(event => {
    if (event.type === 'updated' && event.query.state.error instanceof ApiError &&
      event.query.state.error.kind === 'unauthenticated') setExpired(true);
  }), [client]);
  useEffect(() => {
    if (blocked) {
      void client.cancelQueries();
      client.removeQueries({ predicate: query => query.queryKey[0] !== 'session' });
    }
  }, [blocked, client]);
  useEffect(() => {
    // Route changes focus the new reading context; filter changes leave focus on the control.
    if (!location.pathname.startsWith('/backtests')) document.getElementById('workspace-heading')?.focus({preventScroll:true});
  }, [location.pathname, session.isPending]);
  if (session.isPending) return <main className="session-panel"><p role="status">{t('loading')}</p></main>;
  if (blocked) return <main className="session-panel"><h1>{t('title')}</h1><p role="alert">{
    t(expired || (session.error instanceof ApiError && session.error.kind === 'unauthenticated')
      ? 'expired' : changed ? 'changed' : leaving ? 'signingOut' : 'unavailable')
  }</p>{(expired || (session.error instanceof ApiError && session.error.kind === 'unauthenticated')) &&
    <a href={loginContinuation(next)}>{t('login')}</a>}</main>;
  if (!routeAllowed) return null;
  return <div className="app" data-platform-client>
    <a className="skip-link" href="#workspace-heading">{t('skip')}</a>
    <header className="topbar"><span className="brand">RoeHub</span>{title!=='strategies'&&<span>{t(title)}</span>}
      <nav aria-label={t('locale')} className="languages">
        {(['ru', 'en'] as const).map(locale => <a key={locale} lang={locale}
          aria-current={bootstrap.locale === locale ? 'true' : undefined}
          href={`/locale?${new URLSearchParams({ locale, next })}`}>
          {locale === 'ru' ? 'Русский' : 'English'}
        </a>)}
      </nav>
    </header>
    <aside className="sidebar"><nav aria-label={t('navigation')}>
      <a href="/dashboard"><LayoutDashboard aria-hidden="true" />{t('dashboard')}</a>
      <Link reloadDocument={!clientRoute('/backtests', bootstrap)} data-client-link={clientRoute('/backtests', bootstrap)} to="/backtests" aria-current={!strategies ? 'page' : undefined}><SquareChartGantt aria-hidden="true" />{t('title')}</Link>
      <Link reloadDocument={!clientRoute('/strategies', bootstrap)} to="/strategies" aria-current={strategies ? 'page' : undefined}><ChartNoAxesCombined aria-hidden="true" />{t('strategies')}</Link>
      <span className="nav-unavailable" aria-disabled="true" title={t('dataUnavailable')}><Database aria-hidden="true" />{t('data')}<small>{t('soon')}</small></span>
      <a href="/settings"><Settings aria-hidden="true" />{t('settings')}</a>
    </nav><a className="logout" href="/logout" onClick={() => {
      clearRecovery(); clearStrategyRecovery(); clearOperationRecovery(); setLeaving(true); void client.cancelQueries(); client.clear();
    }}><LogOut aria-hidden="true" />{t('logout')}</a></aside>
    <main className="workspace">{strategies ? <StrategiesPage subject={bootstrap.subject} /> : <BacktestsPage subject={bootstrap.subject} />}</main>
    <footer>{t(strategies ? 'strategy.workspace' : 'artifactOnly')}<span>{t(strategies?'strategy.liveFooter':'manualReads')}</span></footer>
  </div>;
}
