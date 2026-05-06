import { getFocusable, qsa } from "../core/dom.js";

const DROPDOWN_INIT_FLAG = "__roehubDropdownDelegatesInitialized";

function closeDropdown(dropdown) {
  const trigger = dropdown.querySelector("[data-rh-dropdown-trigger]");
  const menu = dropdown.querySelector("[data-rh-dropdown-menu]");
  if (trigger instanceof HTMLElement) {
    trigger.setAttribute("aria-expanded", "false");
  }
  if (menu instanceof HTMLElement) {
    menu.hidden = true;
  }
  dropdown.dataset.open = "false";
}

function openDropdown(dropdown) {
  qsa("[data-rh-dropdown]").forEach((candidate) => {
    if (candidate !== dropdown) {
      closeDropdown(candidate);
    }
  });
  const trigger = dropdown.querySelector("[data-rh-dropdown-trigger]");
  const menu = dropdown.querySelector("[data-rh-dropdown-menu]");
  if (trigger instanceof HTMLElement) {
    trigger.setAttribute("aria-expanded", "true");
  }
  if (menu instanceof HTMLElement) {
    menu.hidden = false;
    const selected = menu.querySelector("[aria-selected='true']");
    if (selected instanceof HTMLElement) {
      selected.focus();
    } else {
      getFocusable(menu)[0]?.focus();
    }
  }
  dropdown.dataset.open = "true";
}

function moveFocus(menu, direction) {
  const items = getFocusable(menu);
  if (items.length === 0) {
    return;
  }
  const index = Math.max(items.indexOf(document.activeElement), 0);
  const nextIndex = (index + direction + items.length) % items.length;
  items[nextIndex].focus();
}

function focusByTypeahead(menu, key) {
  const normalizedKey = key.toLowerCase();
  const item = getFocusable(menu).find((candidate) =>
    (candidate.textContent || "").trim().toLowerCase().startsWith(normalizedKey)
  );
  item?.focus();
}

export function initDropdowns(root = document) {
  qsa("[data-rh-dropdown]", root).forEach((dropdown) => {
    if (!dropdown.dataset.open) {
      dropdown.dataset.open = "false";
    }
  });

  if (window[DROPDOWN_INIT_FLAG]) {
    return;
  }
  window[DROPDOWN_INIT_FLAG] = true;

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const trigger = event.target.closest("[data-rh-dropdown-trigger]");
    if (trigger instanceof HTMLElement) {
      const dropdown = trigger.closest("[data-rh-dropdown]");
      if (dropdown instanceof HTMLElement) {
        if (dropdown.dataset.open === "true") {
          closeDropdown(dropdown);
        } else {
          openDropdown(dropdown);
        }
      }
      return;
    }
    if (event.target instanceof Element && event.target.closest("[data-rh-dropdown]")) {
      return;
    }
    qsa("[data-rh-dropdown]").forEach(closeDropdown);
  });

  document.addEventListener("keydown", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const dropdown = event.target.closest("[data-rh-dropdown]");
    if (!(dropdown instanceof HTMLElement)) {
      return;
    }
    const trigger = dropdown.querySelector("[data-rh-dropdown-trigger]");
    const menu = dropdown.querySelector("[data-rh-dropdown-menu]");
    if (!(trigger instanceof HTMLElement) || !(menu instanceof HTMLElement)) {
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      closeDropdown(dropdown);
      trigger.focus();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      if (menu.hidden) {
        openDropdown(dropdown);
      } else {
        moveFocus(menu, 1);
      }
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(menu, -1);
    } else if (event.key.length === 1 && !event.metaKey && !event.ctrlKey) {
      focusByTypeahead(menu, event.key);
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initDropdowns());
} else {
  initDropdowns();
}
