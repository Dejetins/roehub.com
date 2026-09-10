import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { flushSync } from 'react-dom';
import { Link, useNavigate, type LinkProps } from 'react-router';
import { useTranslation } from 'react-i18next';

export const motionSpeeds = { off: 0, fast: 180, normal: 320, slow: 520 } as const;
export type MotionSpeed = keyof typeof motionSpeeds;
// Retain the existing preference so upgrading does not reset the user's choice.
const preferenceKey = 'roehub.backtests.motion';
export function readMotionSpeed(): MotionSpeed {
  try { const value = localStorage.getItem(preferenceKey); if (value && Object.hasOwn(motionSpeeds, value)) return value as MotionSpeed; } catch { /* Optional storage. */ }
  return 'normal';
}
export function motionDuration() {
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return 0;
  const configured = Number.parseFloat(document.documentElement.style.getPropertyValue('--motion-duration'));
  return Number.isFinite(configured) ? configured : motionSpeeds[readMotionSpeed()];
}
let running: ViewTransition | undefined;
let contentAnimation: Animation | undefined;
/** One interruptible transition coordinator for structural UI changes. Never wraps network commands. */
export function transitionUI(update: () => void, kind: 'content' | 'layout' = 'content') {
  running?.skipTransition();
  contentAnimation?.cancel();
  if (kind === 'content') {
    // Update immediately: rapid tab clicks must never wait for a document snapshot.
    flushSync(update);
    const target = document.querySelector<HTMLElement>('[aria-modal=true] [data-motion-content]') ?? document.querySelector<HTMLElement>('[data-motion-content]');
    if (motionDuration() && target?.animate) contentAnimation = target.animate(
      [{ opacity: .2, transform: 'translateY(3px)' }, { opacity: 1, transform: 'translateY(0)' }],
      { duration: motionDuration(), easing: 'cubic-bezier(.22,1,.36,1)' });
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
export function MotionSettings() {
  const { t } = useTranslation();
  const [speed, setSpeed] = useState(readMotionSpeed);
  useEffect(() => {
    document.documentElement.style.setProperty('--motion-duration', `${motionSpeeds[speed]}ms`);
  }, [speed]);
  useEffect(() => {
    const sync = () => setSpeed(readMotionSpeed());
    window.addEventListener('storage', sync);
    return () => window.removeEventListener('storage', sync);
  }, []);
  return <label className="motion-control">{t('builder.animation')}<select aria-label={t('builder.animation')} value={speed} onChange={event => {
    const next = event.target.value as MotionSpeed;
    try { localStorage.setItem(preferenceKey, next); } catch { /* CSS still applies for this visit. */ }
    setSpeed(next);
  }}>{(Object.keys(motionSpeeds) as MotionSpeed[]).map(value => <option key={value} value={value}>{t(`builder.motion${value}`)}</option>)}</select></label>;
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
    const click = (event: MouseEvent) => {
      if (event.defaultPrevented || !(event.target instanceof Element)) return;
      const summary = event.target.closest('summary');
      const details = summary?.parentElement;
      if (!(details instanceof HTMLDetailsElement) || !details.closest('[data-platform-client]')) return;
      if (event.target.closest('a,button,input,select')) return;
      event.preventDefault();
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
    return () => { document.removeEventListener('click', click); document.removeEventListener('cancel', cancel, true); };
  }, []);
}
