import { apiFetch } from "../core/api.js";
import { renderHostPanel, setHostPanelState } from "../components/plugin-panel.js";

const root = document.querySelector("[data-plugin-panel-lab]");

if (root instanceof HTMLElement) {
  let fixtureState = "success";
  let renderer = "trading-time-series";
  let activeRequest = null;

  const contribution = () => ({
    contract: "RoehubPanelContribution/v1",
    contribution_id: "roehub.qa.pnl",
    title: root.querySelector("[data-panel-title]")?.textContent || "Panel",
    description: root.querySelector("[data-panel-description]")?.textContent || "Panel",
    renderer,
    source: {
      instance_id: root.dataset.instanceId,
      query: {
        contract: "DataSourceQuery/v1",
        dataset: "portfolio.pnl",
        dimensions: ["timestamp"],
        measures: ["pnl", "drawdown"],
        filters: [],
        row_limit: 200,
        byte_limit: 262144,
        point_limit: 1000,
        timeout_ms: 3000,
        read_only: true,
      },
    },
    presentation: {
      x_column: "timestamp",
      y_columns: ["pnl", "drawdown"],
      table_columns: ["timestamp", "pnl", "drawdown"],
      default_view: renderer === "analytics-table" ? "table" : "visual",
    },
  });

  async function loadPanel() {
    activeRequest?.abort();
    activeRequest = new AbortController();
    setHostPanelState(root, "loading");
    try {
      const endpoint = new URL(root.dataset.queryEndpoint, window.location.origin);
      endpoint.searchParams.set("fixture_state", fixtureState);
      const frame = await apiFetch(`${endpoint.pathname}${endpoint.search}`, {
        method: "POST",
        signal: activeRequest.signal,
        timeoutMs: 3500,
        headers: { "content-type": "application/json" },
        body: JSON.stringify(contribution().source.query),
      });
      renderHostPanel(root, contribution(), frame);
    } catch (error) {
      if (error?.code === "aborted") return;
      setHostPanelState(root, "error", error?.message || root.dataset.labelError);
    }
  }

  root.querySelector("[data-panel-refresh]")?.addEventListener("click", loadPanel);
  root.querySelectorAll("[data-panel-renderer]").forEach((button) => {
    button.addEventListener("click", () => {
      renderer = button.dataset.panelRenderer;
      root.querySelectorAll("[data-panel-renderer]").forEach((candidate) => {
        candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
      });
      loadPanel();
    });
  });
  root.querySelectorAll("[data-panel-fixture-state]").forEach((button) => {
    button.addEventListener("click", () => {
      fixtureState = button.dataset.panelFixtureState;
      root.querySelectorAll("[data-panel-fixture-state]").forEach((candidate) => {
        candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
      });
      loadPanel();
    });
  });

  loadPanel();
}
