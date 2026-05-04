import { qsa } from "../core/dom.js";

export function initListboxes(root = document) {
  qsa("[role='listbox']", root).forEach((listbox) => {
    qsa("[role='option']", listbox).forEach((option) => {
      option.addEventListener("click", () => {
        qsa("[role='option']", listbox).forEach((candidate) => {
          candidate.setAttribute("aria-selected", candidate === option ? "true" : "false");
        });
      });
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initListboxes());
} else {
  initListboxes();
}
