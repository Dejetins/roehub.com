export function createPoller(callback, options = {}) {
  return new Poller(callback, options);
}

export class Poller {
  constructor(callback, options = {}) {
    if (typeof callback !== "function") {
      throw new TypeError("Poller callback must be a function");
    }
    this.callback = callback;
    this.intervalMs = options.intervalMs ?? 5000;
    this.hiddenIntervalMs = options.hiddenIntervalMs ?? 5000;
    this.maxBackoffMs = options.maxBackoffMs ?? 30000;
    this.backoffFactor = options.backoffFactor ?? 1.6;
    this.documentRef = options.document || document;
    this.timer = null;
    this.running = false;
    this.inFlight = false;
    this.currentDelayMs = this.intervalMs;
    this.abortController = null;
    this.visibilityHandler = () => this.handleVisibilityChange();
  }

  start({ immediate = true } = {}) {
    if (this.running) {
      return;
    }
    this.running = true;
    this.documentRef.addEventListener("visibilitychange", this.visibilityHandler);
    this.schedule(immediate ? 0 : this.intervalMs);
  }

  stop() {
    this.running = false;
    this.documentRef.removeEventListener("visibilitychange", this.visibilityHandler);
    window.clearTimeout(this.timer);
    this.timer = null;
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  async tick() {
    if (!this.running || this.inFlight) {
      return;
    }
    if (this.documentRef.hidden) {
      this.schedule(this.hiddenIntervalMs);
      return;
    }

    this.inFlight = true;
    this.abortController = new AbortController();
    try {
      await this.callback({ signal: this.abortController.signal });
      this.currentDelayMs = this.intervalMs;
    } catch (_error) {
      this.currentDelayMs = Math.min(
        Math.ceil(this.currentDelayMs * this.backoffFactor),
        this.maxBackoffMs,
      );
    } finally {
      this.inFlight = false;
      this.abortController = null;
      this.schedule(this.currentDelayMs);
    }
  }

  schedule(delayMs) {
    if (!this.running) {
      return;
    }
    window.clearTimeout(this.timer);
    this.timer = window.setTimeout(() => {
      void this.tick();
    }, delayMs);
  }

  handleVisibilityChange() {
    if (!this.running) {
      return;
    }
    if (this.documentRef.hidden) {
      window.clearTimeout(this.timer);
      this.schedule(this.hiddenIntervalMs);
      return;
    }
    if (!this.inFlight) {
      this.schedule(0);
    }
  }
}
