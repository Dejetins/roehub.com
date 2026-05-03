export function createEventStream(url, options = {}) {
  const EventSourceFactory = options.EventSourceFactory || window.EventSource;
  if (!EventSourceFactory) {
    options.onDowngrade?.({ reason: "eventsource_unavailable", url });
    return { close: () => undefined, readyState: "downgraded" };
  }

  const source = new EventSourceFactory(url, {
    withCredentials: options.withCredentials ?? true,
  });

  source.onopen = (event) => options.onOpen?.(event);
  source.onmessage = (event) => options.onMessage?.(event);
  source.onerror = (event) => {
    options.onError?.(event);
    if (source.readyState === EventSourceFactory.CLOSED) {
      options.onDowngrade?.({ reason: "closed", url });
    }
  };

  if (options.events) {
    Object.entries(options.events).forEach(([name, handler]) => {
      source.addEventListener(name, handler);
    });
  }

  return {
    close: () => source.close(),
    get readyState() {
      return source.readyState;
    },
    source,
  };
}
