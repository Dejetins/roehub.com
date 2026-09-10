import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useMotionState } from './motion';

/** Expand in place: the mounted charts, zoom and timeframe survive both directions. */
export function ExpandableOverview({ controls, children }: { controls: ReactNode; children: ReactNode }) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useMotionState(false, 'layout');
  const surface = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useLayoutEffect(() => {
    if (!expanded || !surface.current) return;
    const element = surface.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Inert siblings along the ancestor chain without reparenting/remounting charts.
    const siblings: Array<[HTMLElement, boolean]> = [];
    let branch: HTMLElement = element;
    while (branch.parentElement) {
      for (const sibling of branch.parentElement.children) {
        if (sibling !== branch && sibling instanceof HTMLElement) { siblings.push([sibling, sibling.inert]); sibling.inert = true; }
      }
      branch = branch.parentElement;
      if (branch === document.body) break;
    }
    trigger.current?.focus({ preventScroll: true });
    return () => {
      for (const [sibling, inert] of siblings) sibling.inert = inert;
      document.body.style.overflow = previousOverflow;
      trigger.current?.focus({ preventScroll: true });
    };
  }, [expanded]);
  return <div ref={surface} className={`overview-charts expandable-overview${expanded ? ' overview-expanded' : ''}`}
    role={expanded ? 'dialog' : undefined} aria-modal={expanded ? true : undefined} aria-label={expanded ? t('results.overview') : undefined}
    onKeyDown={event => {
      if (!expanded) return;
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setExpanded(false); }
      if (event.key !== 'Tab') return;
      const focusable = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled),select:not(:disabled),input:not(:disabled),a[href],summary,[tabindex="0"]')].filter(el => el.getClientRects().length && !el.closest('[inert]'));
      const first = focusable[0], last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }}>
    <div className="overview-toolbar">{controls}<button ref={trigger} className="overview-expand" aria-label={t(expanded ? 'results.exitFullscreen' : 'results.fullscreen')} title={t(expanded ? 'results.exitFullscreen' : 'results.fullscreen')} onClick={() => setExpanded(value => !value)}>
      {expanded ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}
    </button></div>
    {children}
  </div>;
}
