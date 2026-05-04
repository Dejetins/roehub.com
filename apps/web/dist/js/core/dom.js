export function qs(selector, root = document) {
  const element = root.querySelector(selector);
  return element instanceof HTMLElement ? element : null;
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector)).filter(
    (element) => element instanceof HTMLElement
  );
}

export function on(root, eventName, selector, handler) {
  root.addEventListener(eventName, (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const matched = event.target.closest(selector);
    if (matched instanceof HTMLElement) {
      handler(event, matched);
    }
  });
}

export function isFocusable(element) {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  if (element.hidden || element.getAttribute("aria-hidden") === "true") {
    return false;
  }
  return !element.hasAttribute("disabled") && element.tabIndex >= 0;
}

export function getFocusable(root) {
  return qsa("a[href], button, input, textarea, [tabindex]", root).filter(isFocusable);
}

export function setText(selector, text, root = document) {
  qsa(selector, root).forEach((element) => {
    element.textContent = text;
  });
}

export function dispatchRoehubEvent(name, detail = {}) {
  document.dispatchEvent(new CustomEvent(`roehub:${name}`, { bubbles: true, detail }));
}
