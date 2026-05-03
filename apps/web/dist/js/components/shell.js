import { ready } from "../core/dom.js";
import { initLocale } from "../core/locale.js";
import { initTheme } from "../core/theme.js";

ready(() => {
  initTheme();
  initLocale();
});
