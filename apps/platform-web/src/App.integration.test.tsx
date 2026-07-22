import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { createPrototypeQueryClient, PrototypeApp } from "./App";
import { UiStore } from "./state/uiStore";

class MockEventSource {
  static instances: MockEventSource[] = [];

  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;

  constructor(public readonly url: string | URL) {
    MockEventSource.instances.push(this);
    queueMicrotask(() => this.onopen?.(new Event("open")));
  }

  close(): void {}

  emit(payload: object): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(payload) }),
    );
  }
}

describe("Prototype state and transport integration", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            revision: "rest-test",
            source: "mock-rest",
            serverAuthorization: "fixture-server-projection",
            rows: [
              {
                id: "bt-safe-001",
                name: "BTC daily fixture",
                status: "running",
                returnPct: 8.42,
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it("keeps theme in MobX and applies SSE only to the Query projection", async () => {
    const store = new UiStore();
    render(
      <QueryClientProvider client={createPrototypeQueryClient()}>
        <PrototypeApp uiStore={store} />
      </QueryClientProvider>,
    );
    const user = userEvent.setup();

    expect(await screen.findByText("BTC daily fixture")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "paper" }));
    expect(store.theme).toBe("paper");
    expect(document.documentElement).toHaveAttribute("data-theme", "paper");

    MockEventSource.instances[0]?.emit({
      revision: "sse-test",
      rowId: "bt-safe-001",
      status: "completed",
    });

    await waitFor(() =>
      expect(screen.getByTestId("revision")).toHaveTextContent("sse-test"),
    );
    expect(screen.getByTestId("status-bt-safe-001")).toHaveTextContent(
      "completed",
    );
    expect(store).not.toHaveProperty("rows");
  });
});
