import { ready } from "../core/dom.js";

ready(() => {
  const landingPage = document.querySelector("[data-landing-page]");
  if (!landingPage) {
    return;
  }

  landingPage.dataset.ready = "true";
  document.dispatchEvent(new CustomEvent("roehub:landing-ready"));
});
