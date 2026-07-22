import type { QueryClient } from "@tanstack/react-query";

import { recordAfterNextPaint } from "../metrics";
import type { BacktestSnapshot } from "./api";

export const BACKTESTS_QUERY_KEY = ["prototype", "backtests"] as const;

export interface BacktestSseEvent {
  revision: string;
  rowId: string;
  status: "queued" | "running" | "completed";
}

export function parseBacktestSseEvent(raw: string): BacktestSseEvent {
  const payload = JSON.parse(raw) as Partial<BacktestSseEvent>;
  if (
    typeof payload.revision !== "string" ||
    typeof payload.rowId !== "string" ||
    !["queued", "running", "completed"].includes(payload.status ?? "")
  ) {
    throw new Error("Invalid prototype SSE event");
  }
  return payload as BacktestSseEvent;
}

export function applyBacktestSseEvent(
  queryClient: QueryClient,
  event: BacktestSseEvent,
): void {
  const receivedAt = performance.now();
  queryClient.setQueryData<BacktestSnapshot>(
    BACKTESTS_QUERY_KEY,
    (snapshot) => {
      if (!snapshot) return snapshot;
      return {
        ...snapshot,
        revision: event.revision,
        source: "mock-sse",
        rows: snapshot.rows.map((row) =>
          row.id === event.rowId ? { ...row, status: event.status } : row,
        ),
      };
    },
  );
  recordAfterNextPaint("sseToPaint", receivedAt);
}
