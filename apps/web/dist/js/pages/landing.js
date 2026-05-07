const root = document.querySelector("[data-landing-root]");

if (root instanceof HTMLElement) {
  root.dataset.landingReady = "true";
}
