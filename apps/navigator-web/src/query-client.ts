import { QueryClient } from '@tanstack/react-query';

export const READ_CACHE = { gcTime: 120_000, maxInactiveEntries: 64, maxInactiveBytes: 16 * 1024 * 1024 } as const;

export function createQueryClient() {
  const client = new QueryClient({ defaultOptions: {
    queries: { retry: false, staleTime: 15_000, gcTime: READ_CACHE.gcTime, refetchOnWindowFocus: false },
    mutations: { retry: false, gcTime: READ_CACHE.gcTime },
  } });
  client.setQueryDefaults(['data-job-action'], {gcTime:300_000});
  const cache = client.getQueryCache();
  const sizes = new WeakMap<object, number>();
  function size(data: unknown) {
    if (data && typeof data === 'object' && sizes.has(data)) return sizes.get(data)!;
    let bytes: number;
    try { bytes = (JSON.stringify(data)?.length ?? 0) * 2; } catch { bytes = READ_CACHE.maxInactiveBytes + 1; }
    if (data && typeof data === 'object') sizes.set(data, bytes);
    return bytes;
  }
  // Inactive read results are disposable. Session and command recovery are not read results.
  // Coalesce notifications, and never evict an observed or in-flight query.
  const touched = new Map<string, number>();
  let queued = false;
  cache.subscribe(event => {
    if (event.type === 'removed') { touched.delete(event.query.queryHash); return; }
    if (event.type === 'observerAdded' || event.type === 'observerRemoved' || event.type === 'updated') touched.set(event.query.queryHash, Date.now());
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      const reads = cache.getAll().filter(query => query.getObserversCount() === 0 && query.state.fetchStatus === 'idle' && query.queryKey[0] !== 'session' && query.queryKey[0] !== 'data-job-action' && !['cancel','delete'].includes(String(query.queryKey[2])));
      reads.sort((a,b) => (touched.get(a.queryHash) ?? 0) - (touched.get(b.queryHash) ?? 0));
      let count = reads.length, bytes = reads.reduce((sum,query) => sum + size(query.state.data), 0);
      for (const query of reads) {
        if (count <= READ_CACHE.maxInactiveEntries && bytes <= READ_CACHE.maxInactiveBytes) break;
        count--; bytes -= size(query.state.data); cache.remove(query);
      }
    });
  });
  return client;
}
