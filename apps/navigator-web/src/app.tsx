import {restoreTemporaryReads,sessionReadInterval,temporarySessionError} from './session-recovery';
import { OverviewPage } from './overview-page';
import { LoadingData } from './loading-data';
import {clearOperationRecovery} from './operation-recovery';
import { StrategiesPage } from './strategies-page';
import { clientRoute } from './client-routes';
import { MotionLink as Link, useDisclosureMotion } from './motion';
import { clearStrategyRecovery, loadStrategyRecovery } from './strategy-recovery';
import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router';
import { ChartNoAxesCombined, Database, LayoutDashboard, LogOut, Activity, Plug, Settings, SquareChartGantt } from 'lucide-react';
import type { NavigatorBootstrap } from './navigator-bootstrap';
import {SettingsPage} from './settings-page';
import {AccountPreferences} from './account-preferences';
import {ConnectionsPage} from './connections-page';
import {MonitoringPage} from './monitoring-page';
import {DataPage} from './data-page';
import {LegacyDownloadsRedirect} from './data-downloads';
import { ApiError, loginContinuation, readSession } from './api';
import { BacktestsPage } from './backtests-page';
import { clearRecovery, loadRecovery } from './recovery';

export function App({ bootstrap }: { bootstrap: NavigatorBootstrap }) {
  useDisclosureMotion();
  const { t } = useTranslation();
  const location = useLocation();
  const client = useQueryClient();
  const recovering=useRef(false);
  const [expired, setExpired] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const session = useQuery({ queryKey: ['session', bootstrap.subject],
    queryFn: ({ signal }) => readSession(signal), enabled: !expired && !leaving,
    refetchInterval: query => expired || leaving ||
      (query.state.data && query.state.data.user_id !== bootstrap.subject) ? false : sessionReadInterval(query.state.error,query.state.errorUpdateCount),
    refetchOnReconnect: false,
    refetchOnWindowFocus: query => !expired && !leaving && !query.state.error &&
      (!query.state.data || query.state.data.user_id === bootstrap.subject), retry: false });
  const strategies = location.pathname.startsWith('/strategies');
  const overview = location.pathname === '/dashboard';
  const settings = location.pathname.startsWith('/settings/');
  const connections = location.pathname === '/connections';
  const monitoring = location.pathname.startsWith('/monitoring');
  const data = location.pathname === '/data';
  const ingestion = location.pathname === '/data/ingestion';
  const title = monitoring ? 'monitoring' : data||ingestion ? 'data' : connections ? 'connections' : settings ? 'settings' : overview ? 'dashboard' : strategies ? 'strategies' : 'title';
  const routeAllowed = clientRoute(`${location.pathname}${location.search}`, bootstrap);
  useEffect(() => { if (!routeAllowed) window.location.replace(`${location.pathname}${location.search}`); }, [routeAllowed, location.pathname, location.search]);
  useEffect(() => { document.title = `${t(title)} · RoeHub`; }, [t, title]);
  const next = `${location.pathname}${location.search}`;
  const changed = !!session.data && session.data.user_id !== bootstrap.subject;
  const blocked = expired || leaving || changed || !!session.error;
  const temporary = !expired&&!leaving&&!changed&&temporarySessionError(session.error);
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
    if (blocked&&!temporary) {
      void client.cancelQueries();
      client.removeQueries({ predicate: query => query.queryKey[0] !== 'session' });
    }
  }, [blocked, temporary, client]);
  useEffect(() => {
    if(temporary){recovering.current=true;return;}
    if(blocked||!recovering.current)return;
    recovering.current=false;
    // Restore failed reads after identity is verified; never touch command fences.
    return restoreTemporaryReads(client);
  },[temporary,blocked,client]);
  useEffect(() => {
    // Route changes focus the new reading context; filter changes leave focus on the control.
    if (!location.pathname.startsWith('/backtests')) document.getElementById('workspace-heading')?.focus({preventScroll:true});
  }, [location.pathname, session.isPending]);
  if (session.isPending) return <main className="session-panel"><LoadingData /></main>;
  const sessionPanel = <main className="session-panel"><h1><span className="nav-label">{t(title)}</span></h1><p role="alert">{
    t(expired || (session.error instanceof ApiError && session.error.kind === 'unauthenticated')
      ? 'expired' : changed ? 'changed' : leaving ? 'signingOut' : temporary ? 'sessionRecovering' : 'unavailable')
  }</p>{(expired || (session.error instanceof ApiError && session.error.kind === 'unauthenticated')) &&
    <a href={loginContinuation(next)}>{t('login')}</a>}</main>;
  if (blocked&&(!temporary||!session.data)) return sessionPanel;
  if (!routeAllowed) return null;
  if (ingestion) return <LegacyDownloadsRedirect/>;
  return <>{blocked&&sessionPanel}<div className="app navigator-app" data-platform-client hidden={blocked} inert={blocked} style={blocked?{display:'none'}:undefined}>
    {bootstrap.client_routes?.includes('/settings')&&<AccountPreferences subject={bootstrap.subject}/>}
    <a className="skip-link" href="#workspace-heading">{t('skip')}</a>
    <aside className="sidebar"><span className="navigator-brand" title="RoeHub">R</span><nav aria-label={t('navigation')}>
      <Link reloadDocument={!clientRoute('/dashboard', bootstrap)} to="/dashboard" title={t('dashboard')} aria-current={overview ? 'page' : undefined}><LayoutDashboard aria-hidden="true" /><span className="nav-label">{t('dashboard')}</span></Link>
      <Link reloadDocument={!clientRoute('/backtests', bootstrap)} data-client-link={clientRoute('/backtests', bootstrap)} to="/backtests" title={t('title')} aria-current={location.pathname.startsWith('/backtests') ? 'page' : undefined}><SquareChartGantt aria-hidden="true" /><span className="nav-label">{t('title')}</span></Link>
      <Link reloadDocument={!clientRoute('/strategies', bootstrap)} to="/strategies" title={t('strategies')} aria-current={strategies ? 'page' : undefined}><ChartNoAxesCombined aria-hidden="true" /><span className="nav-label">{t('strategies')}</span></Link>
      <Link to="/data" reloadDocument={!clientRoute('/data',bootstrap)} title={t('data')} aria-current={data||ingestion?'page':undefined}><Database aria-hidden="true" /><span className="nav-label">{t('data')}</span></Link>
      <Link to="/connections" reloadDocument={!clientRoute('/connections',bootstrap)} title={t('connections')} aria-current={connections?'page':undefined}><Plug aria-hidden="true"/><span className="nav-label">{t('connections')}</span></Link>
      <Link to="/monitoring" reloadDocument={!clientRoute('/monitoring',bootstrap)} title={t('monitoring')} aria-current={monitoring?'page':undefined}><Activity aria-hidden="true"/><span className="nav-label">{t('monitoring')}</span></Link>
      <Link to="/settings/profile" reloadDocument={!clientRoute('/settings/profile',bootstrap)} title={t('settings')} aria-current={settings?'page':undefined}><Settings aria-hidden="true" /><span className="nav-label">{t('settings')}</span></Link>
    </nav><details className="navigator-languages"><summary aria-label={t('locale')}>文</summary><nav aria-label={t('locale')} className="languages">
        {(['ru', 'en'] as const).map(locale => <a key={locale} lang={locale}
          aria-current={bootstrap.locale === locale ? 'true' : undefined}
          href={`/locale?${new URLSearchParams({ locale, next })}`}>
          {locale === 'ru' ? 'Русский' : 'English'}
        </a>)}
      </nav> </details><a className="logout" title={t('logout')} href="/logout" onClick={() => {
      clearRecovery(); clearStrategyRecovery(); clearOperationRecovery(); setLeaving(true); void client.cancelQueries(); client.clear();
    }}><LogOut aria-hidden="true" /><span className="nav-label">{t('logout')}</span></a></aside>
    <main className="workspace"><div className="navigator-page-label">{t(title)}<span>Navigator</span></div><div className={`navigator-grid${settings||connections||data||ingestion||monitoring?' workpages-grid':''}${data?' data-workspace':''}`}>{monitoring?<MonitoringPage subject={bootstrap.subject}/>:data?<DataPage subject={bootstrap.subject}/>:connections?<ConnectionsPage subject={bootstrap.subject}/>:settings?<SettingsPage subject={bootstrap.subject}/>:overview ? <OverviewPage subject={bootstrap.subject} /> : strategies ? <StrategiesPage subject={bootstrap.subject} /> : <BacktestsPage subject={bootstrap.subject} />}</div></main>
    <footer>{overview ? (bootstrap.locale === 'ru' ? 'Демонстрационный портфель · данные синтетические' : 'Demo portfolio · synthetic data') : (settings||connections||data||ingestion||monitoring ? t(title) : t(strategies ? 'strategy.workspace' : 'artifactOnly'))}<span>{overview ? 'UTC · USDT' : (settings||connections||data||ingestion||monitoring ? 'UTC' : t(strategies?'strategy.liveFooter':'manualReads'))}</span></footer>
  </div></>;
}
