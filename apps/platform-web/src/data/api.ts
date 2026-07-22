import {
  recordRestAbort,
  recordRestDispatch,
  recordRestResponse,
} from "../metrics";

export interface BacktestRow {
  id: string;
  name: string;
  status: "queued" | "running" | "completed";
  returnPct: number;
}

export interface BacktestSnapshot {
  revision: string;
  source: "mock-rest" | "mock-sse";
  serverAuthorization: "fixture-server-projection";
  rows: BacktestRow[];
}

function isSnapshot(payload: unknown): payload is BacktestSnapshot {
  if (typeof payload !== "object" || payload === null) return false;
  const candidate = payload as Partial<BacktestSnapshot>;
  return (
    typeof candidate.revision === "string" &&
    (candidate.source === "mock-rest" || candidate.source === "mock-sse") &&
    candidate.serverAuthorization === "fixture-server-projection" &&
    Array.isArray(candidate.rows)
  );
}

export async function fetchBacktestSnapshot(
  signal: AbortSignal,
  latencyMs: number,
): Promise<BacktestSnapshot> {
  signal.addEventListener("abort", recordRestAbort, { once: true });
  recordRestDispatch();
  const response = await fetch(
    `/__prototype/api/backtests?latency_ms=${latencyMs}`,
    {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    },
  );
  if (!response.ok)
    throw new Error(`Prototype REST failed with ${response.status}`);
  const payload: unknown = await response.json();
  if (!isSnapshot(payload))
    throw new Error("Prototype REST returned an invalid snapshot");
  recordRestResponse();
  return payload;
}
