import { qsa } from "../core/dom.js";

const LISTBOX_INIT_FLAG = "__roehubListboxDelegatesInitialized";

export function initListboxes(root = document) {
  qsa("[role='listbox']", root);
  if (window[LISTBOX_INIT_FLAG]) {
    return;
  }
  window[LISTBOX_INIT_FLAG] = true;

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const option = event.target.closest("[role='option']");
    const listbox = option?.closest("[role='listbox']");
    if (!(option instanceof HTMLElement) || !(listbox instanceof HTMLElement)) {
      return;
    }
    qsa("[role='option']", listbox).forEach((candidate) => {
      candidate.setAttribute("aria-selected", candidate === option ? "true" : "false");
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initListboxes());
} else {
  initListboxes();
}
