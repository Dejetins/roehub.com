import { MotionSettings } from './motion';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Route, Routes, useLocation, useNavigate } from 'react-router';
import { useTranslation } from 'react-i18next';
import { Builder, type BuilderSummary } from './builder';
import { BacktestsWorkspace } from './library';
import { libraryParams } from './library-api';

/** History and the draft retain their state while the editor folds inside Jobs. */
export function BacktestsPage({ subject }: { subject: string }) {
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const [demoPreset] = useState(() => ['localhost','127.0.0.1'].includes(window.location.hostname) && new URLSearchParams(location.search).get('preset') === 'synthetic');
  const newRoute = /^\/backtests\/new\/?$/.test(location.pathname);
  const history = useRef({ ...location, pathname: newRoute ? '/backtests' : location.pathname });
  if (!newRoute) history.current = location;
  const [expanded, setExpanded] = useState(newRoute);
  const [visited, setVisited] = useState(newRoute);
  const [generation, setGeneration] = useState(0);
  const [draft, setDraft] = useState<BuilderSummary | null>(null);
  const submitted = useRef(false);
  const receiveSummary = useCallback((summary: BuilderSummary) => { if (!submitted.current) setDraft(summary); }, []);
  const trigger = useRef<HTMLButtonElement>(null);
  const previousPath = useRef(location.pathname);
  useEffect(() => {
    const prior = previousPath.current; previousPath.current = location.pathname;
    if (newRoute) {
      if (submitted.current) { submitted.current = false; setGeneration(value => value + 1); setDraft(null); }
      setVisited(true); setExpanded(true);
    }
    else if (prior !== location.pathname) {
      setExpanded(false);
      (document.getElementById('selected-job-heading') ?? trigger.current)?.focus({ preventScroll: true });
    }
  }, [location.pathname, newRoute]);
  function open() {
    if (submitted.current) { submitted.current = false; setGeneration(value => value + 1); setDraft(null); }
    setVisited(true); setExpanded(true);
    trigger.current?.focus({ preventScroll: true });
    trigger.current?.scrollIntoView?.({ behavior: 'auto', block: 'nearest' });
  }
  function close() {
    trigger.current?.focus({ preventScroll: true }); setExpanded(false);
    if (newRoute) navigate(`${history.current.pathname}${history.current.search}${history.current.hash}`);
  }
  function date(value: string) {
    const parsed = new Date(value);
    return Number.isFinite(parsed.getTime()) ? new Intl.DateTimeFormat(i18n.language, { dateStyle: 'medium', timeZone: 'UTC' }).format(parsed) : '—';
  }
  const configuration = <section className="config-disclosure" aria-labelledby="configure-toggle">
      <div className="panel-head disclosure-head">
        <h3><button ref={trigger} id="configure-toggle" type="button" className="disclosure-toggle" aria-expanded={expanded} aria-controls="configuration-body"
          onClick={() => expanded ? close() : open()}><ChevronDown aria-hidden="true" />{t(expanded ? 'builder.collapse' : draft?.dirty ? 'builder.resume' : 'new')}</button></h3>
        <p className="draft-summary">{draft ? <>{draft.symbol} · {draft.timeframe}<span>{date(draft.start)} — {date(draft.end)}</span><span>{t(`builder.${draft.status}`)}</span></> : t('builder.configure')}</p>

      </div>
      <div id="configuration-body" className="disclosure-body" data-expanded={expanded} aria-hidden={!expanded} inert={!expanded}
        onKeyDown={event => { if (event.key === 'Escape' && !event.defaultPrevented) { event.preventDefault(); close(); } }}>
        <div className="disclosure-clip">
          {visited && <Builder demoPreset={demoPreset} key={generation} subject={subject} embedded onSummary={receiveSummary} onClose={close} onCreated={id => {
            submitted.current = true; setExpanded(false); setDraft(null);
            const filters = libraryParams(new URLSearchParams(history.current.search)).toString();
            navigate(`/backtests/${id}${filters ? `?${filters}` : ''}`);
          }} />}
        </div>
      </div>
    </section>;
  return <div className="backtests-page">
    <div className="workspace-title"><h1 id="workspace-heading" tabIndex={-1}>{t('title')}</h1><span className="scope-label">{t('research')}</span><MotionSettings /></div>
    <div id="panel-history">
      <Routes location={history.current}>
        <Route path="/backtests" element={<BacktestsWorkspace subject={subject} embedded active={!newRoute} configuration={configuration} onNew={open} />} />
        <Route path="/backtests/:jobId" element={<BacktestsWorkspace subject={subject} mode="detail" embedded active={!newRoute} configuration={configuration} onNew={open} />} />
      </Routes>
    </div>
  </div>;
}
