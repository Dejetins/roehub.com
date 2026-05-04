export class RoehubPoller {
  constructor(task, options = {}) {
    this.task = task;
    this.intervalMs = options.intervalMs || 30000;
    this.hiddenTabPause = options.hiddenTabPause !== false;
    this.backoffMs = 0;
    this.retryAfterUntil = 0;
    this.timer = null;
    this.running = false;
    this.stopped = true;
    this.lastResult = null;
  }

  start() {
    if (!this.stopped) {
      return;
    }
    this.stopped = false;
    this.schedule(0);
  }

  stop() {
    this.stopped = true;
    if (this.timer) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
  }

  schedule(delayMs = this.intervalMs) {
    if (this.stopped) {
      return;
    }
    if (this.timer) {
      window.clearTimeout(this.timer);
    }
    this.timer = window.setTimeout(() => this.tick(), delayMs);
  }

  async tick() {
    if (this.stopped) {
      return;
    }
    if (this.hiddenTabPause && document.hidden) {
      this.schedule(this.intervalMs);
      return;
    }
    if (this.running) {
      this.schedule(this.intervalMs);
      return;
    }
    const now = Date.now();
    if (this.retryAfterUntil > now) {
      this.schedule(this.retryAfterUntil - now);
      return;
    }

    this.running = true;
    try {
      this.lastResult = await this.task();
      this.backoffMs = 0;
      const retryAfterSeconds = Number(this.lastResult?.retry_after_seconds || 0);
      if (retryAfterSeconds > 0) {
        this.retryAfterUntil = Date.now() + retryAfterSeconds * 1000;
      }
    } catch (error) {
      if (error?.status === 401) {
        this.stop();
        document.dispatchEvent(new CustomEvent("roehub:auth-required", { detail: error }));
        return;
      }
      this.backoffMs = Math.min(this.backoffMs ? this.backoffMs * 2 : 1000, 30000);
    } finally {
      this.running = false;
    }
    this.schedule(Math.max(this.intervalMs, this.backoffMs));
  }
}

export function createPoller(task, options = {}) {
  return new RoehubPoller(task, options);
}
