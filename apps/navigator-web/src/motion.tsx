import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { flushSync } from 'react-dom';
import { Link, useNavigate, type LinkProps } from 'react-router';

export function motionDuration() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return 0;
  const configured = Number.parseFloat(document.documentElement.style.getPropertyValue('--motion-duration'));
  return Number.isFinite(configured) ? configured : 320;
}
let running: ViewTransition | undefined;
/** One interruptible transition coordinator for structural UI changes. Never wraps network commands. */
export function transitionUI(update: () => void, kind: 'content' | 'layout' = 'content') {
  running?.skipTransition();
  if (kind === 'content') {
    // Update immediately: rapid tab clicks must never wait for a document snapshot.
    flushSync(update);
    return;
  }
  if (!document.startViewTransition || motionDuration() === 0) { update(); return; }
  document.documentElement.dataset.motionKind = kind;
  const transition = document.startViewTransition(() => { flushSync(update); });
  running = transition;
  // Skipping a superseded animation is expected, not an application failure.
  void transition.ready.catch(() => {});
  void transition.finished.catch(() => {}).finally(() => { if (running === transition) { running = undefined; delete document.documentElement.dataset.motionKind; } });
}
export function useMotionState<T>(initial: T | (() => T), kind: 'content' | 'layout' = 'content'): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState(initial);
  const set = useCallback<Dispatch<SetStateAction<T>>>(next => transitionUI(() => setValue(next), kind), [kind]);
  return [value, set];
}
/** Client navigation uses the same coordinator as local tabs and panel geometry. */
export function MotionLink({ onClick, ...props }: LinkProps) {
  const navigate = useNavigate();
  return <Link {...props} onClick={event => {
    onClick?.(event);
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || props.reloadDocument || (props.target && props.target !== '_self')) return;
    event.preventDefault();
    transitionUI(() => { void navigate(props.to, { replace: props.replace, state: props.state, preventScrollReset: props.preventScrollReset, relative: props.relative }); });
  }} />;
}
/** Native details retain their semantics and use the same layout transition. */
export function useDisclosureMotion() {
  useEffect(() => {
    const disclosures = new Map<HTMLDetailsElement, { animation: Animation; opening: boolean }>();
    const click = (event: MouseEvent) => {
      if (event.defaultPrevented || !(event.target instanceof Element)) return;
      const summary = event.target.closest('summary');
      const details = summary?.parentElement;
      if (!(details instanceof HTMLDetailsElement) || !details.closest('[data-platform-client]')) return;
      if (event.target.closest('a,button,input,select')) return;
      event.preventDefault();
      if (details.closest('.operations-technical') && details.animate && motionDuration()) {
        const previous = disclosures.get(details);
        const opening = !(previous?.opening ?? details.open);
        const start = details.getBoundingClientRect().height;
        previous?.animation.cancel();
        details.open = opening;
        const end = details.getBoundingClientRect().height;
        // Keep content rendered while closing; commit the native state at the end.
        details.open = true;
        const animation = details.animate([
          { height: `${start}px`, overflow: 'hidden' },
          { height: `${end}px`, overflow: 'hidden' },
        ], { duration: motionDuration(), easing: 'cubic-bezier(.22,1,.36,1)' });
        disclosures.set(details, { animation, opening });
        animation.onfinish = () => {
          details.open = opening;
          disclosures.delete(details);
        };
        return;
      }
      transitionUI(() => { details.open = !details.open; });
    };
    const cancel = (event: Event) => {
      if (event.defaultPrevented || !(event.target instanceof HTMLDialogElement) || !event.target.closest('[data-platform-client]')) return;
      event.preventDefault();
      const dialog = event.target;
      transitionUI(() => dialog.close());
    };
    document.addEventListener('click', click);
    document.addEventListener('cancel', cancel, true);
    return () => { for (const { animation } of disclosures.values()) animation.cancel(); document.removeEventListener('click', click); document.removeEventListener('cancel', cancel, true); };
  }, []);
}
