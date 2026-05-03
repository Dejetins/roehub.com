const ROOT_SELECTOR = "html";

export function rootElement(documentRef = document) {
  const root = documentRef.querySelector(ROOT_SELECTOR);
  if (!root) {
    throw new Error("Roehub DOM root is missing");
  }
  return root;
}

export function qs(selector, scope = document) {
  return scope.querySelector(selector);
}

export function qsa(selector, scope = document) {
  return Array.from(scope.querySelectorAll(selector));
}

export function on(target, type, handler, options) {
  target.addEventListener(type, handler, options);
  return () => target.removeEventListener(type, handler, options);
}

export function delegate(root, type, selector, handler, options) {
  return on(
    root,
    type,
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) {
        return;
      }

      const matched = target.closest(selector);
      if (matched && root.contains(matched)) {
        handler(event, matched);
      }
    },
    options,
  );
}

export function setAttributes(element, attributes) {
  Object.entries(attributes).forEach(([name, value]) => {
    if (value === false || value === null || value === undefined) {
      element.removeAttribute(name);
      return;
    }
    if (value === true) {
      element.setAttribute(name, "");
      return;
    }
    element.setAttribute(name, String(value));
  });
}

export function setBusy(element, busy) {
  element.setAttribute("aria-busy", String(busy));
  if ("disabled" in element) {
    element.disabled = busy;
  }
}

export function ready(callback, documentRef = document) {
  if (documentRef.readyState === "loading") {
    documentRef.addEventListener("DOMContentLoaded", callback, { once: true });
    return;
  }
  callback();
}
