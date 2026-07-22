import { mkdir } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

const evidenceRoot = path.resolve(
  process.cwd(),
  "output/playwright/linear-frontend-architecture-spike",
);

test("proves bounded React coexistence and the current SSR return path", async ({
  page,
}) => {
  await mkdir(path.join(evidenceRoot, "screenshots"), { recursive: true });
  const reactConsoleErrors: string[] = [];
  const reactRequestFailures: string[] = [];
  let reactPhase = true;
  page.on("console", (message) => {
    if (reactPhase && message.type() === "error")
      reactConsoleErrors.push(message.text());
  });
  page.on("requestfailed", (request) => {
    if (reactPhase && !request.url().includes("/__prototype/api/backtests")) {
      reactRequestFailures.push(`${request.method()} ${request.url()}`);
    }
  });

  const entryResponse = await page.goto("/__prototype/react/");
  expect(entryResponse?.headers()["x-roehub-prototype"]).toBe("true");
  await expect(page.getByText("PROTOTYPE", { exact: true })).toBeVisible();
  await expect(page.getByTestId("query-authority")).toHaveText(
    "TanStack Query cache",
  );
  await expect(page.getByTestId("sse-state")).toHaveText("open");
  await expect(page.getByTestId("revision")).toContainText(/rest-|sse-/);

  for (const theme of ["abyss", "graphite", "frost", "paper"] as const) {
    await page.getByRole("button", { name: theme, exact: true }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await page.screenshot({
      path: path.join(
        evidenceRoot,
        "screenshots",
        `theme-${theme}-1440x900.png`,
      ),
      fullPage: false,
    });
  }

  const resizer = page.getByTestId("panel-resizer");
  const initialWidth = Number(await resizer.getAttribute("aria-valuenow"));
  await resizer.focus();
  await resizer.press("ArrowRight");
  await expect(resizer).toHaveAttribute(
    "aria-valuenow",
    String(initialWidth + 8),
  );
  const resizerBox = await resizer.boundingBox();
  expect(resizerBox).not.toBeNull();
  if (resizerBox) {
    await page.mouse.move(
      resizerBox.x + resizerBox.width / 2,
      resizerBox.y + 100,
    );
    await page.mouse.down();
    await page.mouse.move(304, resizerBox.y + 100, { steps: 8 });
    await page.mouse.up();
  }
  await expect(resizer).toHaveAttribute("aria-valuenow", "304");
  await resizer.dblclick();
  await expect(resizer).toHaveAttribute("aria-valuenow", "240");

  await page.getByTestId("rest-slow").click();
  await expect(page.getByText("Fetching", { exact: true })).toBeVisible();
  await page.getByTestId("rest-cancel").click();
  await page.waitForFunction(
    () => window.__ROEHUB_PROTOTYPE_METRICS__.snapshot().restAbortCount >= 1,
  );
  await expect(page.getByTestId("cancel-message")).toContainText(
    "Cancellation requested through TanStack Query",
  );

  expect(reactConsoleErrors).toEqual([]);
  expect(reactRequestFailures).toEqual([]);

  await page.setViewportSize({ width: 820, height: 900 });
  await page.screenshot({
    path: path.join(evidenceRoot, "screenshots", "responsive-820x900.png"),
    fullPage: false,
  });
  await page.setViewportSize({ width: 1440, height: 900 });

  reactPhase = false;
  await page.getByTestId("ssr-return").click();
  await page.waitForURL("**/backtests?from=react-prototype");
  await expect(page.locator("[data-backtests-root]")).toBeVisible();
  await expect(page.locator("h1")).toContainText("Backtests");
  expect(page.url()).not.toContain("/__prototype/react/");

  await page.goBack();
  await page.waitForURL("**/__prototype/react/");
  await expect(page.getByText("PROTOTYPE", { exact: true })).toBeVisible();
});
