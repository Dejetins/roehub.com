import {
  PANEL_DEFAULT_WIDTH,
  PANEL_MAX_WIDTH,
  PANEL_MIN_WIDTH,
  UiStore,
} from "./uiStore";

describe("UiStore local presentation authority", () => {
  it("keeps theme, selection, and bounded panel geometry locally", () => {
    const store = new UiStore();

    store.setTheme("paper");
    store.selectBacktest("bt-safe-001");
    store.resizePanel(999);

    expect(store.theme).toBe("paper");
    expect(store.selectedBacktestId).toBe("bt-safe-001");
    expect(store.panelWidth).toBe(PANEL_MAX_WIDTH);

    store.resizePanel(-1);
    expect(store.panelWidth).toBe(PANEL_MIN_WIDTH);

    store.resetPanel();
    expect(store.panelWidth).toBe(PANEL_DEFAULT_WIDTH);
  });
});
