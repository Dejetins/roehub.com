import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

function percentile(values: number[], quantile: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.ceil(sorted.length * quantile) - 1,
  );
  return Number(sorted[index].toFixed(3));
}

function distribution(values: number[]) {
  return {
    samples: values.length,
    p50: percentile(values, 0.5),
    p75: percentile(values, 0.75),
    p95: percentile(values, 0.95),
  };
}

test("measures client latency, INP, long tasks, and frame cadence", async ({
  page,
}) => {
  await page.goto("/__prototype/react/");
  await expect(page.getByTestId("sse-state")).toHaveText("open");
  await expect(page.getByTestId("revision")).toContainText(/rest-|sse-/);
  await page.evaluate(() => window.__ROEHUB_PROTOTYPE_METRICS__.reset());

  for (let index = 0; index < 30; index += 1) {
    const expectedSamples = index + 1;
    await page.getByTestId("rest-reload").click();
    await page.waitForFunction(
      (minimum) =>
        window.__ROEHUB_PROTOTYPE_METRICS__.snapshot().metrics.responseToPaint
          .length >= minimum,
      expectedSamples,
    );
  }

  for (let index = 0; index < 32; index += 1) {
    const theme = ["abyss", "graphite", "frost", "paper"][index % 4];
    await page.getByRole("button", { name: theme, exact: true }).click();
  }

  const resizer = page.getByTestId("panel-resizer");
  await resizer.focus();
  for (let index = 0; index < 32; index += 1) {
    await resizer.press(index % 2 === 0 ? "ArrowRight" : "ArrowLeft");
  }

  await page.waitForFunction(
    () =>
      window.__ROEHUB_PROTOTYPE_METRICS__.snapshot().metrics.sseToPaint
        .length >= 20,
  );
  await page.waitForTimeout(800);
  const snapshot = await page.evaluate(() =>
    window.__ROEHUB_PROTOTYPE_METRICS__.snapshot(),
  );

  const longTasksOver50Ms = snapshot.longTasks.filter(
    (duration) => duration > 50,
  );
  const droppedFrameIntervals = snapshot.frameIntervals.filter(
    (duration) => duration > 25,
  );
  const stableFramePercent = Number(
    (
      (snapshot.frameIntervals.filter((duration) => duration <= 20).length /
        snapshot.frameIntervals.length) *
      100
    ).toFixed(2),
  );
  const report = {
    measuredAt: new Date().toISOString(),
    browser: await page.evaluate(() => navigator.userAgent),
    viewport: page.viewportSize(),
    method:
      "30 sequential 36ms deterministic REST fixtures, live 120ms SSE, 32 theme clicks, 32 keyboard resizes; PerformanceObserver(event,longtask) plus requestAnimationFrame",
    acknowledgementMs: distribution(snapshot.metrics.acknowledgement),
    dispatchMs: distribution(snapshot.metrics.dispatch),
    responseToPaintMs: distribution(snapshot.metrics.responseToPaint),
    sseToPaintMs: distribution(snapshot.metrics.sseToPaint),
    themeToPaintMs: distribution(snapshot.metrics.themeToPaint),
    resizeToPaintMs: distribution(snapshot.metrics.resizeToPaint),
    inpEventDurationMs: distribution(snapshot.inpDurations),
    inpCandidateMs: percentile(snapshot.inpDurations, 0.98),
    longTasks: {
      samples: snapshot.longTasks.length,
      over50Ms: longTasksOver50Ms,
    },
    frameCadence: {
      intervalMs: distribution(snapshot.frameIntervals),
      medianFps:
        percentile(snapshot.frameIntervals, 0.5) === null
          ? null
          : Number(
              (1000 / (percentile(snapshot.frameIntervals, 0.5) ?? 1)).toFixed(
                2,
              ),
            ),
      stableFramePercent,
      intervalsOver25Ms: droppedFrameIntervals.length,
    },
  };

  const evidenceRoot = path.resolve(
    process.cwd(),
    "output/playwright/linear-frontend-architecture-spike",
  );
  await mkdir(evidenceRoot, { recursive: true });
  await writeFile(
    path.join(evidenceRoot, "performance-results.json"),
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );

  expect(report.dispatchMs.p95).not.toBeNull();
  expect(report.dispatchMs.p95 ?? Infinity).toBeLessThanOrEqual(50);
  expect(report.acknowledgementMs.p95 ?? Infinity).toBeLessThanOrEqual(100);
  expect(report.responseToPaintMs.p95 ?? Infinity).toBeLessThanOrEqual(200);
  expect(report.sseToPaintMs.p95 ?? Infinity).toBeLessThanOrEqual(200);
  expect(report.inpCandidateMs).not.toBeNull();
  expect(report.inpCandidateMs ?? Infinity).toBeLessThanOrEqual(200);
  expect(longTasksOver50Ms).toEqual([]);
  expect(stableFramePercent).toBeGreaterThanOrEqual(90);
});
