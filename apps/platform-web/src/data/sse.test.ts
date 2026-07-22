import { QueryClient } from "@tanstack/react-query";

import type { BacktestSnapshot } from "./api";
import {
  applyBacktestSseEvent,
  BACKTESTS_QUERY_KEY,
  parseBacktestSseEvent,
} from "./sse";

describe("SSE remote-state boundary", () => {
  it("updates the TanStack Query cache without a MobX server-state copy", () => {
    const queryClient = new QueryClient();
    const initial: BacktestSnapshot = {
      revision: "rest-0001",
      source: "mock-rest",
      serverAuthorization: "fixture-server-projection",
      rows: [
        { id: "bt-safe-001", name: "Fixture", status: "running", returnPct: 1 },
      ],
    };
    queryClient.setQueryData(BACKTESTS_QUERY_KEY, initial);

    applyBacktestSseEvent(
      queryClient,
      parseBacktestSseEvent(
        JSON.stringify({
          revision: "sse-0001",
          rowId: "bt-safe-001",
          status: "completed",
        }),
      ),
    );

    expect(
      queryClient.getQueryData<BacktestSnapshot>(BACKTESTS_QUERY_KEY),
    ).toEqual(
      expect.objectContaining({
        revision: "sse-0001",
        source: "mock-sse",
        rows: [expect.objectContaining({ status: "completed" })],
      }),
    );
  });
});
