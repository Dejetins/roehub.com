import { useLayoutEffect, useRef } from 'react';

/** One committed view, never a second cache. Callers provide a complete, identity-checked
 * bundle. Scope is the authorization boundary; restricted reads must set blocked.
 * Keep controls outside the retained view and disable commands while retained. */
export function useReadSnapshot<T>(scope: string, key: string, candidate: T | undefined, blocked = false) {
  const previous = useRef<{scope: string; key: string; data: T} | undefined>(undefined);
  const current = !blocked && candidate !== undefined ? {scope, key, data: candidate} : undefined;
  const displayed = blocked ? undefined : current ?? (previous.current?.scope === scope ? previous.current : undefined);
  useLayoutEffect(() => {
    if (blocked || previous.current?.scope !== scope) previous.current = undefined;
    if (current) previous.current = current;
  });
  return { data: displayed?.data, retained: !!displayed && !current, displayedKey: displayed?.key };
}
