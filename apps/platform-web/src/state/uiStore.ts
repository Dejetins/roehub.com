import { makeAutoObservable } from "mobx";

export const THEME_IDS = ["abyss", "graphite", "frost", "paper"] as const;
export type ThemeId = (typeof THEME_IDS)[number];

export const PANEL_DEFAULT_WIDTH = 240;
export const PANEL_MIN_WIDTH = 208;
export const PANEL_MAX_WIDTH = 320;
export const PANEL_KEYBOARD_STEP = 8;

export class UiStore {
  theme: ThemeId = "graphite";
  panelWidth = PANEL_DEFAULT_WIDTH;
  selectedBacktestId: string | null = null;

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });
  }

  setTheme(theme: ThemeId): void {
    this.theme = theme;
  }

  selectBacktest(backtestId: string): void {
    this.selectedBacktestId = backtestId;
  }

  resizePanel(width: number): void {
    this.panelWidth = Math.min(
      PANEL_MAX_WIDTH,
      Math.max(PANEL_MIN_WIDTH, width),
    );
  }

  resizePanelBy(delta: number): void {
    this.resizePanel(this.panelWidth + delta);
  }

  resetPanel(): void {
    this.panelWidth = PANEL_DEFAULT_WIDTH;
  }
}
