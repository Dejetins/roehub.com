const SIDEBAR_STORAGE_KEY = "roehub_workbench_sidebar";
const dialogOpeners = new WeakMap();

function openDialog(dialog, opener) {
  if (!(dialog instanceof HTMLDialogElement)) return;
  dialogOpeners.set(dialog, opener || document.activeElement);
  if (!dialog.open) dialog.showModal();
  const preferredFocus = dialog.querySelector("[data-workbench-command-input]");
  window.requestAnimationFrame(() => preferredFocus?.focus());
}

function closeDialog(dialog) {
  if (!(dialog instanceof HTMLDialogElement) || !dialog.open) return;
  dialog.close();
}

function restoreDialogFocus(dialog) {
  const opener = dialogOpeners.get(dialog);
  if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
  dialogOpeners.delete(dialog);
}

function initDialogs(root = document) {
  root.querySelectorAll("[data-workbench-dialog-open]").forEach((opener) => {
    opener.addEventListener("click", () => {
      const dialogId = opener.dataset.workbenchDialogOpen;
      if (dialogId) openDialog(document.getElementById(dialogId), opener);
    });
  });

  root.querySelectorAll("dialog.workbench-dialog").forEach((dialog) => {
    dialog.addEventListener("close", () => restoreDialogFocus(dialog));
    dialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeDialog(dialog);
      }
    });
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) closeDialog(dialog);
    });
  });
}

function initCommandPalette(root = document) {
  const dialog = root.getElementById("command-dialog");
  const input = root.querySelector("[data-workbench-command-input]");
  const results = [...root.querySelectorAll("[data-workbench-command-results] a")];
  if (!(dialog instanceof HTMLDialogElement) || !(input instanceof HTMLInputElement)) return;

  const filterResults = () => {
    const query = input.value.trim().toLocaleLowerCase();
    results.forEach((item) => {
      item.hidden = Boolean(query) && !String(item.dataset.commandLabel || "").includes(query);
    });
  };

  input.addEventListener("input", filterResults);
  dialog.addEventListener("close", () => {
    input.value = "";
    filterResults();
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      openDialog(dialog, document.activeElement);
    }
  });
}

function readSidebarPreference() {
  try {
    return window.localStorage.getItem(SIDEBAR_STORAGE_KEY) === "collapsed";
  } catch {
    return false;
  }
}

function writeSidebarPreference(collapsed) {
  try {
    window.localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? "collapsed" : "expanded");
  } catch {
    // Local preference is optional when storage is unavailable.
  }
}

function initSidebar(root = document) {
  const shell = root.querySelector("[data-workbench-shell]");
  const toggle = root.querySelector("[data-workbench-collapse]");
  if (!(shell instanceof HTMLElement) || !(toggle instanceof HTMLButtonElement)) return;

  const setCollapsed = (collapsed) => {
    shell.classList.toggle("is-sidebar-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    toggle.textContent = collapsed ? "›" : "‹";
    writeSidebarPreference(collapsed);
  };

  setCollapsed(readSidebarPreference());
  toggle.addEventListener("click", () => setCollapsed(!shell.classList.contains("is-sidebar-collapsed")));
}

function initShell() {
  initDialogs();
  initCommandPalette();
  initSidebar();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initShell, { once: true });
} else {
  initShell();
}
