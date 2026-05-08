import { t } from "../core/locale.js";

const root = document.querySelector("[data-landing-root]");

const STREAM = [
  { kind: "command", key: "landing.cli.command_init" },
  { kind: "response", key: "landing.cli.response_shell" },
  { kind: "command", key: "landing.cli.command_sync" },
  { kind: "response", key: "landing.cli.response_exchange" },
  { kind: "command", key: "landing.cli.command_backtest" },
  { kind: "response", key: "landing.cli.response_variants" },
  { kind: "response", key: "landing.cli.response_strategy" },
  { kind: "command", key: "landing.cli.command_deploy" },
  { kind: "response", key: "landing.cli.response_ready" },
];

const MAX_VISIBLE_LINES = 7;
const TYPE_DELAY_MS = 28;
const COMMAND_PAUSE_MS = 380;
const RESPONSE_PAUSE_MS = 620;
const RESTART_PAUSE_MS = 3400;

if (root instanceof HTMLElement) {
  root.dataset.landingReady = "true";
  initCliStream(root);
}

function initCliStream(container) {
  const log = container.querySelector("[data-cli-log]");
  const state = container.querySelector("[data-cli-state]");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  if (!(log instanceof HTMLOListElement)) {
    return;
  }

  let timer = 0;
  let itemIndex = 0;
  let charIndex = 0;
  let activeCommand = "";
  let activeLine = null;

  const clearTimer = () => {
    if (timer) {
      window.clearTimeout(timer);
      timer = 0;
    }
  };

  const schedule = (callback, delay) => {
    clearTimer();
    timer = window.setTimeout(callback, delay);
  };

  const trimLog = () => {
    while (log.children.length > MAX_VISIBLE_LINES) {
      log.firstElementChild?.remove();
    }
  };

  const appendLine = (kind, text) => {
    const line = document.createElement("li");
    line.className = `landing-cli__line landing-cli__line--${kind}`;
    line.textContent = text;
    log.append(line);
    trimLog();
    return line;
  };

  const renderStatic = () => {
    clearTimer();
    log.replaceChildren();
    STREAM.slice(-MAX_VISIBLE_LINES).forEach((item) => {
      appendLine(item.kind, t(item.key));
    });
    if (state instanceof HTMLElement) {
      state.textContent = t("landing.cli.state_ready");
    }
  };

  const typeNextCharacter = () => {
    if (document.hidden) {
      clearTimer();
      return;
    }
    if (!(activeLine instanceof HTMLElement)) {
      return;
    }
    activeLine.textContent = activeCommand.slice(0, charIndex);
    charIndex += 1;
    if (charIndex <= activeCommand.length) {
      schedule(typeNextCharacter, TYPE_DELAY_MS);
      return;
    }
    activeLine.classList.remove("landing-cli__line--typing");
    activeCommand = "";
    activeLine = null;
    schedule(runNextItem, COMMAND_PAUSE_MS);
  };

  const runNextItem = () => {
    if (document.hidden) {
      clearTimer();
      return;
    }
    if (itemIndex >= STREAM.length) {
      if (state instanceof HTMLElement) {
        state.textContent = t("landing.cli.state_ready");
      }
      schedule(() => {
        itemIndex = 0;
        log.replaceChildren();
        if (state instanceof HTMLElement) {
          state.textContent = t("landing.cli.state_booting");
        }
        runNextItem();
      }, RESTART_PAUSE_MS);
      return;
    }

    const item = STREAM[itemIndex];
    itemIndex += 1;
    const text = t(item.key);

    if (item.kind === "command") {
      activeCommand = text;
      activeLine = appendLine("command", "");
      activeLine.classList.add("landing-cli__line--typing");
      charIndex = 0;
      typeNextCharacter();
      return;
    }

    appendLine("response", text);
    schedule(runNextItem, RESPONSE_PAUSE_MS);
  };

  const start = () => {
    if (reducedMotion.matches) {
      renderStatic();
      return;
    }
    log.replaceChildren();
    if (state instanceof HTMLElement) {
      state.textContent = t("landing.cli.state_booting");
    }
    itemIndex = 0;
    schedule(runNextItem, 240);
  };

  if (typeof reducedMotion.addEventListener === "function") {
    reducedMotion.addEventListener("change", start);
  } else {
    reducedMotion.addListener(start);
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearTimer();
      return;
    }
    if (reducedMotion.matches) {
      renderStatic();
      return;
    }
    if (activeCommand && activeLine instanceof HTMLElement) {
      typeNextCharacter();
      return;
    }
    schedule(runNextItem, 240);
  });

  start();
}
