import { getFocusable, qsa } from "../core/dom.js";

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
    const trigger = dropdown.querySelector("[data-rh-dropdown-trigger]");
    const menu = dropdown.querySelector("[data-rh-dropdown-menu]");
    if (!(trigger instanceof HTMLElement) || !(menu instanceof HTMLElement)) {
      return;
    }
    trigger.addEventListener("click", () => {
      if (dropdown.dataset.open === "true") {
        closeDropdown(dropdown);
      } else {
        openDropdown(dropdown);
      }
    });
    dropdown.addEventListener("keydown", (event) => {
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
  });

  document.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest("[data-rh-dropdown]")) {
      return;
    }
    qsa("[data-rh-dropdown]").forEach(closeDropdown);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initDropdowns());
} else {
  initDropdowns();
}
