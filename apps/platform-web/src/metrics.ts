export type MetricName =
  | "acknowledgement"
  | "dispatch"
  | "responseToPaint"
  | "sseToPaint"
  | "themeToPaint"
  | "resizeToPaint";

export interface PrototypeMetricSnapshot {
  metrics: Record<MetricName, number[]>;
  inpDurations: number[];
  longTasks: number[];
  frameIntervals: number[];
  restAbortCount: number;
}

const metricNames: MetricName[] = [
  "acknowledgement",
  "dispatch",
  "responseToPaint",
  "sseToPaint",
  "themeToPaint",
  "resizeToPaint",
];

const metricValues = Object.fromEntries(
  metricNames.map((name) => [name, [] as number[]]),
) as Record<MetricName, number[]>;
const inpDurations: number[] = [];
const longTasks: number[] = [];
const frameIntervals: number[] = [];
let restAbortCount = 0;
let pendingRestInteractionAt: number | null = null;
let pendingRestResponseAt: number | null = null;
let previousFrameAt: number | null = null;

function append(target: number[], value: number): void {
  if (Number.isFinite(value) && value >= 0) {
    target.push(Number(value.toFixed(3)));
    if (target.length > 2000) target.shift();
  }
}

export function recordMetric(name: MetricName, value: number): void {
  append(metricValues[name], value);
}

export function recordAfterNextPaint(
  name: MetricName,
  startedAt: number,
): void {
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() =>
      recordMetric(name, performance.now() - startedAt),
    );
  } else {
    setTimeout(() => recordMetric(name, performance.now() - startedAt), 0);
  }
}

export function beginRestInteraction(startedAt: number): void {
  pendingRestInteractionAt = startedAt;
  recordAfterNextPaint("acknowledgement", startedAt);
}

export function recordRestDispatch(): void {
  if (pendingRestInteractionAt === null) return;
  recordMetric("dispatch", performance.now() - pendingRestInteractionAt);
  pendingRestInteractionAt = null;
}

export function recordRestResponse(): void {
  pendingRestResponseAt = performance.now();
}

export function recordRestPaint(): void {
  if (pendingRestResponseAt === null) return;
  const responseAt = pendingRestResponseAt;
  pendingRestResponseAt = null;
  recordAfterNextPaint("responseToPaint", responseAt);
}

export function recordRestAbort(): void {
  restAbortCount += 1;
}

export function resetMetrics(): void {
  for (const name of metricNames) metricValues[name].length = 0;
  inpDurations.length = 0;
  longTasks.length = 0;
  frameIntervals.length = 0;
  restAbortCount = 0;
  pendingRestInteractionAt = null;
  pendingRestResponseAt = null;
  previousFrameAt = null;
}

export function snapshotMetrics(): PrototypeMetricSnapshot {
  return {
    metrics: Object.fromEntries(
      metricNames.map((name) => [name, [...metricValues[name]]]),
    ) as Record<MetricName, number[]>,
    inpDurations: [...inpDurations],
    longTasks: [...longTasks],
    frameIntervals: [...frameIntervals],
    restAbortCount,
  };
}

function observePerformance(): void {
  if (typeof PerformanceObserver === "undefined") return;
  const supported = PerformanceObserver.supportedEntryTypes ?? [];

  if (supported.includes("event")) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const eventEntry = entry as PerformanceEntry & {
          interactionId?: number;
        };
        if ((eventEntry.interactionId ?? 0) > 0)
          append(inpDurations, entry.duration);
      }
    });
    observer.observe({
      type: "event",
      buffered: true,
      durationThreshold: 0,
    } as PerformanceObserverInit);
  }

  if (supported.includes("longtask")) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) append(longTasks, entry.duration);
    });
    observer.observe({ type: "longtask", buffered: true });
  }
}

function sampleFrames(now: number): void {
  if (previousFrameAt !== null) append(frameIntervals, now - previousFrameAt);
  previousFrameAt = now;
  requestAnimationFrame(sampleFrames);
}

declare global {
  interface Window {
    __ROEHUB_PROTOTYPE_METRICS__: {
      reset: typeof resetMetrics;
      snapshot: typeof snapshotMetrics;
    };
  }
}

if (typeof window !== "undefined") {
  window.__ROEHUB_PROTOTYPE_METRICS__ = {
    reset: resetMetrics,
    snapshot: snapshotMetrics,
  };
  observePerformance();
  if (typeof requestAnimationFrame === "function")
    requestAnimationFrame(sampleFrames);
}
