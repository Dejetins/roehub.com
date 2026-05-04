export function createSseClient(url, { onMessage, onError, onDowngrade } = {}) {
  let source = null;

  function connect() {
    if (!("EventSource" in window)) {
      onDowngrade?.("eventsource_unavailable");
      return null;
    }
    source = new EventSource(url, { withCredentials: true });
    source.onmessage = (event) => onMessage?.(event);
    source.onerror = (event) => {
      onError?.(event);
      if (source?.readyState === EventSource.CLOSED) {
        onDowngrade?.("eventsource_closed");
      }
    };
    return source;
  }

  function close() {
    source?.close();
    source = null;
  }

  return { connect, close };
}
