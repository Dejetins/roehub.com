import { QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { observer } from "mobx-react-lite";
import { useEffect, useMemo, useRef, useState } from "react";
import styled from "styled-components";

import { fetchBacktestSnapshot } from "./data/api";
import {
  applyBacktestSseEvent,
  BACKTESTS_QUERY_KEY,
  parseBacktestSseEvent,
} from "./data/sse";
import {
  beginRestInteraction,
  recordAfterNextPaint,
  recordRestPaint,
} from "./metrics";
import {
  PANEL_KEYBOARD_STEP,
  PANEL_MAX_WIDTH,
  PANEL_MIN_WIDTH,
  THEME_IDS,
  type ThemeId,
  UiStore,
} from "./state/uiStore";

const Shell = styled.div<{ $panelWidth: number }>`
  display: grid;
  grid-template-columns: ${({ $panelWidth }) => $panelWidth}px 8px minmax(
      0,
      1fr
    );
  height: 100vh;
  background: var(--canvas);
`;

const Sidebar = styled.aside`
  min-width: 0;
  padding: var(--space-4);
  overflow: hidden;
  background: var(--surface);
  border-right: 1px solid var(--border);
`;

const ResizeHandle = styled.div`
  position: relative;
  z-index: 2;
  margin-left: -4px;
  margin-right: -4px;
  cursor: col-resize;

  &::after {
    content: "";
    position: absolute;
    inset: 0 3px;
    background: var(--border);
  }

  &:hover::after,
  &:focus-visible::after {
    background: var(--accent);
  }
`;

const Main = styled.main`
  min-width: 0;
  overflow: auto;
  padding: 20px 24px 32px;
`;

const Card = styled.section`
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
`;

const Button = styled.button`
  min-height: 32px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-raised);
  color: var(--text);
  cursor: pointer;

  &[aria-pressed="true"] {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
`;

const RowButton = styled.button`
  display: grid;
  grid-template-columns: minmax(140px, 1fr) 100px 100px;
  width: 100%;
  padding: 10px 12px;
  border: 0;
  border-bottom: 1px solid var(--border);
  background: transparent;
  color: var(--text);
  text-align: left;
  cursor: pointer;

  &[aria-current="true"] {
    background: color-mix(in srgb, var(--accent) 14%, transparent);
  }
`;

function setTheme(store: UiStore, theme: ThemeId): void {
  const startedAt = performance.now();
  store.setTheme(theme);
  document.documentElement.dataset.theme = theme;
  recordAfterNextPaint("themeToPaint", startedAt);
}

function useServerProjection() {
  const latencyRef = useRef(36);
  const query = useQuery({
    queryKey: BACKTESTS_QUERY_KEY,
    queryFn: ({ signal }) => fetchBacktestSnapshot(signal, latencyRef.current),
    staleTime: 30_000,
  });
  return { query, latencyRef };
}

export interface PrototypeAppProps {
  uiStore: UiStore;
}

export const PrototypeApp = observer(function PrototypeApp({
  uiStore,
}: PrototypeAppProps) {
  const queryClient = useQueryClient();
  const { query, latencyRef } = useServerProjection();
  const [sseState, setSseState] = useState<"connecting" | "open" | "error">(
    "connecting",
  );
  const [cancelMessage, setCancelMessage] = useState(
    "No cancellation requested",
  );
  const pointerActive = useRef(false);
  const lastRestRevision = useRef<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = uiStore.theme;
  }, [uiStore.theme]);

  useEffect(() => {
    const events = new EventSource("/__prototype/events");
    events.onopen = () => setSseState("open");
    events.onerror = () => setSseState("error");
    events.onmessage = (message) => {
      applyBacktestSseEvent(queryClient, parseBacktestSseEvent(message.data));
    };
    return () => events.close();
  }, [queryClient]);

  useEffect(() => {
    const revision = query.data?.revision ?? null;
    if (
      revision?.startsWith("rest-") &&
      revision !== lastRestRevision.current
    ) {
      lastRestRevision.current = revision;
      recordRestPaint();
    }
  }, [query.data?.revision]);

  const selected = useMemo(
    () => query.data?.rows.find((row) => row.id === uiStore.selectedBacktestId),
    [query.data?.rows, uiStore.selectedBacktestId],
  );

  const reload = (latencyMs: number): void => {
    const startedAt = performance.now();
    latencyRef.current = latencyMs;
    setCancelMessage("Request in flight; Query owns its AbortSignal");
    beginRestInteraction(startedAt);
    void query.refetch({ cancelRefetch: true });
  };

  const cancel = (): void => {
    void queryClient.cancelQueries({ queryKey: BACKTESTS_QUERY_KEY });
    setCancelMessage("Cancellation requested through TanStack Query");
  };

  const resizeFromPointer = (clientX: number): void => {
    const startedAt = performance.now();
    uiStore.resizePanel(clientX);
    recordAfterNextPaint("resizeToPaint", startedAt);
  };

  return (
    <Shell $panelWidth={uiStore.panelWidth} data-testid="prototype-shell">
      <Sidebar aria-label="Prototype local navigation">
        <p className="prototype-mark">PROTOTYPE</p>
        <h1>React coexistence spike</h1>
        <p className="muted">
          Route boundary: <code>/__prototype/react/</code>
        </p>
        <nav aria-label="Prototype sections">
          <a href="#state">State boundary</a>
          <a href="#performance">Performance</a>
          <a href="/backtests?from=react-prototype" data-testid="ssr-return">
            Return to current SSR /backtests
          </a>
        </nav>
        <dl className="boundary-list">
          <div>
            <dt>MobX</dt>
            <dd>theme · panel width · selected row</dd>
          </div>
          <div>
            <dt>TanStack Query</dt>
            <dd>REST snapshot · SSE projection · cancellation</dd>
          </div>
          <div>
            <dt>Authorization</dt>
            <dd>server projection only; no client decision</dd>
          </div>
        </dl>
      </Sidebar>

      <ResizeHandle
        role="separator"
        aria-label="Resize prototype navigation"
        aria-orientation="vertical"
        aria-valuemin={PANEL_MIN_WIDTH}
        aria-valuemax={PANEL_MAX_WIDTH}
        aria-valuenow={uiStore.panelWidth}
        tabIndex={0}
        data-testid="panel-resizer"
        onDoubleClick={() => uiStore.resetPanel()}
        onPointerDown={(event) => {
          pointerActive.current = true;
          document.body.dataset.resizing = "true";
          event.currentTarget.setPointerCapture(event.pointerId);
          resizeFromPointer(event.clientX);
        }}
        onPointerMove={(event) => {
          if (pointerActive.current) resizeFromPointer(event.clientX);
        }}
        onPointerUp={(event) => {
          pointerActive.current = false;
          delete document.body.dataset.resizing;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
        }}
        onKeyDown={(event) => {
          const startedAt = performance.now();
          if (event.key === "ArrowLeft")
            uiStore.resizePanelBy(-PANEL_KEYBOARD_STEP);
          else if (event.key === "ArrowRight")
            uiStore.resizePanelBy(PANEL_KEYBOARD_STEP);
          else if (event.key === "Home") uiStore.resetPanel();
          else return;
          event.preventDefault();
          recordAfterNextPaint("resizeToPaint", startedAt);
        }}
      />

      <Main>
        <header className="page-header">
          <div>
            <p className="eyebrow">ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE</p>
            <h2>Read-only backtest projection</h2>
            <p className="muted">
              Safe fixture data. No trading action or production credential path
              exists.
            </p>
          </div>
          <div className="theme-controls" aria-label="Theme switcher">
            {THEME_IDS.map((theme) => (
              <Button
                key={theme}
                type="button"
                aria-pressed={uiStore.theme === theme}
                onClick={() => setTheme(uiStore, theme)}
              >
                {theme}
              </Button>
            ))}
          </div>
        </header>

        <div className="status-grid" id="state">
          <Card>
            <span className="label">Remote authority</span>
            <strong data-testid="query-authority">TanStack Query cache</strong>
            <small data-testid="revision">
              {query.data?.revision ?? "loading"}
            </small>
          </Card>
          <Card>
            <span className="label">SSE transport</span>
            <strong data-testid="sse-state">{sseState}</strong>
            <small>{query.data?.source ?? "awaiting snapshot"}</small>
          </Card>
          <Card>
            <span className="label">Local presentation</span>
            <strong>{uiStore.theme}</strong>
            <small data-testid="panel-width">
              {uiStore.panelWidth}px panel
            </small>
          </Card>
        </div>

        <Card className="work-surface">
          <div className="toolbar">
            <div>
              <span className="label">Typed REST fixture</span>
              <strong>{query.isFetching ? "Fetching" : "Stable"}</strong>
            </div>
            <div className="button-row">
              <Button
                type="button"
                data-testid="rest-reload"
                onClick={() => reload(36)}
              >
                Reload REST (36ms mock)
              </Button>
              <Button
                type="button"
                data-testid="rest-slow"
                onClick={() => reload(900)}
              >
                Start cancellable REST
              </Button>
              <Button type="button" data-testid="rest-cancel" onClick={cancel}>
                Cancel through Query
              </Button>
            </div>
          </div>
          <p className="muted" aria-live="polite" data-testid="cancel-message">
            {cancelMessage}
          </p>
          {query.error ? <p role="alert">{query.error.message}</p> : null}
          <div
            className="table"
            role="table"
            aria-label="Safe backtest fixtures"
          >
            <div className="table-head" role="row">
              <span>Name</span>
              <span>Status</span>
              <span>Return</span>
            </div>
            {query.data?.rows.map((row) => (
              <RowButton
                key={row.id}
                type="button"
                role="row"
                aria-current={uiStore.selectedBacktestId === row.id}
                onClick={() => uiStore.selectBacktest(row.id)}
              >
                <span>{row.name}</span>
                <span data-testid={`status-${row.id}`}>{row.status}</span>
                <span>{row.returnPct.toFixed(2)}%</span>
              </RowButton>
            ))}
          </div>
        </Card>

        <Card id="performance">
          <span className="label">Selected local state</span>
          <strong>{selected?.name ?? "Select a row"}</strong>
          <p className="muted">
            Browser harness exports dispatch, acknowledgement,
            response/SSE-to-paint, INP event durations, long tasks, and
            requestAnimationFrame cadence through
            <code> window.__ROEHUB_PROTOTYPE_METRICS__</code>.
          </p>
        </Card>
      </Main>
    </Shell>
  );
});

export function createPrototypeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 60_000 },
    },
  });
}
