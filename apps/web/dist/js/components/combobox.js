import { qsa } from "../core/dom.js";

export function initComboboxes(root = document) {
  qsa("[data-rh-combobox]", root).forEach((combobox) => {
    const input = combobox.querySelector("[role='combobox']");
    const options = qsa("[role='option']", combobox);
    if (!(input instanceof HTMLInputElement)) {
      return;
    }
    input.addEventListener("input", () => {
      const query = input.value.trim().toLowerCase();
      options.forEach((option) => {
        option.hidden = query !== "" && !(option.textContent || "").toLowerCase().includes(query);
      });
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initComboboxes());
} else {
  initComboboxes();
}
