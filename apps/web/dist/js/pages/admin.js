import { apiFetch, RoehubApiError } from "../core/api.js";
import { t } from "../core/locale.js";

const root = document.querySelector("[data-admin-root]");

if (root instanceof HTMLElement) {
  const state = {
    organizationId: "",
    snapshot: null,
    pendingConfirmation: null,
    currentOperationId: "",
    currentOperationKey: "",
    currentOperationUrl: "",
    pollGeneration: 0,
  };

  const endpoint = (name, replacements = {}) => {
    let value = root.dataset[name] || "";
    Object.entries(replacements).forEach(([key, replacement]) => {
      value = value.replace(`{${key}}`, encodeURIComponent(String(replacement)));
    });
    return value;
  };

  const text = (selector, value) => {
    const target = root.querySelector(selector);
    if (target) target.textContent = String(value ?? "—");
  };

  const shortId = (value) => {
    const normalized = String(value || "");
    return normalized.length > 16 ? `${normalized.slice(0, 8)}…${normalized.slice(-6)}` : normalized;
  };

  const readable = (value) => String(value || "—").replaceAll("_", " ").replaceAll(".", " ");

  const setConsoleState = (kind, title, detail) => {
    text("[data-admin-state]", title);
    text("[data-admin-state-detail]", detail);
    const presence = root.querySelector("[data-admin-presence]");
    if (presence) presence.className = `admin-console__presence${kind ? ` is-${kind}` : ""}`;
  };

  const setPanel = (name) => {
    root.querySelectorAll("[data-admin-section]").forEach((button) => {
      const selected = button.dataset.adminSection === name;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    root.querySelectorAll("[data-admin-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.adminPanel !== name;
    });
  };

  const appendCell = (row, value, { code = false } = {}) => {
    const cell = document.createElement("td");
    const content = code ? document.createElement("code") : document.createElement("span");
    content.textContent = String(value ?? "—");
    cell.append(content);
    row.append(cell);
    return cell;
  };

  const renderCapabilities = (capabilities = {}) => {
    const target = root.querySelector("[data-admin-capabilities]");
    if (!target) return;
    target.replaceChildren();
    ["providers", "backups", "updates", "services", "observability"].forEach((key) => {
      const card = document.createElement("article");
      const value = capabilities[key] || "degraded";
      card.className = `admin-metric is-${value}`;
      const label = document.createElement("span");
      label.textContent = t(`admin.capability.${key}`);
      const status = document.createElement("strong");
      status.textContent = t(`admin.capability.${value}`);
      card.append(label, status);
      target.append(card);
    });
  };

  const roleOptions = ["owner", "admin", "operator", "trader", "viewer"];

  const renderMembers = (members = []) => {
    const target = root.querySelector("[data-admin-members]");
    if (!target) return;
    target.replaceChildren();
    const canManage = state.snapshot?.permissions?.includes("roles.manage") && state.snapshot?.recent_auth;
    members.forEach((member) => {
      const row = document.createElement("tr");
      const userCell = appendCell(row, shortId(member.user_id), { code: true });
      userCell.title = member.user_id;
      const roleCell = document.createElement("td");
      const select = document.createElement("select");
      select.className = "admin-member-role";
      select.setAttribute("aria-label", t("admin.members.role_for", { user: shortId(member.user_id) }));
      roleOptions.forEach((role) => {
        const option = document.createElement("option");
        option.value = role;
        option.textContent = t(`admin.role.${role}`);
        option.selected = role === member.role;
        select.append(option);
      });
      select.disabled = !canManage;
      roleCell.append(select);
      row.append(roleCell);
      appendCell(row, t(`admin.member_status.${member.status}`));
      const actionCell = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "rh-button rh-button--compact rh-button--secondary";
      button.textContent = t("admin.members.save");
      button.disabled = !canManage;
      button.addEventListener("click", () => {
        const nextRole = select.value;
        openConfirmation({
          title: t("admin.confirm.role_title"),
          impact: t("admin.confirm.role_impact", {
            user: shortId(member.user_id),
            from: t(`admin.role.${member.role}`),
            to: t(`admin.role.${nextRole}`),
          }),
          execute: async () => {
            await apiFetch(
              endpoint("memberEndpointTemplate", {
                organization_id: state.organizationId,
                user_id: member.user_id,
              }),
              {
                method: "PATCH",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ role: nextRole }),
              },
            );
            await loadSnapshot();
          },
        });
      });
      actionCell.append(button);
      row.append(actionCell);
      target.append(row);
    });
  };

  const renderPlugins = (installations = [], operations = []) => {
    const target = root.querySelector("[data-admin-plugins]");
    if (target) {
      target.replaceChildren();
      if (!installations.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 4;
        cell.textContent = t("admin.plugins.empty");
        row.append(cell);
        target.append(row);
      }
      installations.forEach((plugin) => {
        const row = document.createElement("tr");
        appendCell(row, plugin.plugin_id, { code: true });
        appendCell(row, readable(plugin.status));
        appendCell(row, plugin.granted_permissions.join(", ") || "—");
        appendCell(row, plugin.rollback_available ? t("admin.answer.yes") : t("admin.answer.no"));
        target.append(row);
      });
    }
    const operationTarget = root.querySelector("[data-admin-plugin-operations]");
    if (!operationTarget) return;
    operationTarget.replaceChildren();
    if (!operations.length) {
      const empty = document.createElement("p");
      empty.textContent = t("admin.operations.empty");
      operationTarget.append(empty);
    }
    operations.forEach((operation) => {
      const item = document.createElement("div");
      item.className = "admin-operation-list__item";
      [operation.kind, operation.target_id, readable(operation.status)].forEach((value) => {
        const span = document.createElement("span");
        span.textContent = value;
        item.append(span);
      });
      operationTarget.append(item);
    });
  };

  const renderEvents = (events = []) => {
    const target = root.querySelector("[data-admin-events]");
    if (!target) return;
    target.replaceChildren();
    if (!events.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = t("admin.audit.empty");
      row.append(cell);
      target.append(row);
    }
    events.forEach((event) => {
      const row = document.createElement("tr");
      appendCell(row, new Date(event.created_at).toLocaleString());
      appendCell(row, event.category);
      appendCell(row, event.action, { code: true });
      appendCell(row, readable(event.outcome));
      target.append(row);
    });
  };

  const renderOperationalHealth = (health = {}) => {
    const overallState = health.overall_state || "unknown";
    text("[data-admin-health-state]", t(`admin.health.state.${overallState}`));
    text("[data-admin-health-profile]", health.profile || "unknown");
    text(
      "[data-admin-health-updated]",
      health.generated_at ? new Date(health.generated_at).toLocaleString() : "—",
    );
    const status = root.querySelector("[data-admin-health-state]");
    status?.classList.toggle("rh-status--success", overallState === "ready");
    const target = root.querySelector("[data-admin-health-services]");
    if (!target) return;
    target.replaceChildren();
    const services = Array.isArray(health.services) ? health.services : [];
    if (!services.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.textContent = t("admin.health.empty");
      row.append(cell);
      target.append(row);
      return;
    }
    const canOperate = state.snapshot?.permissions?.includes("operations.execute")
      && state.snapshot?.recent_auth
      && state.snapshot?.capabilities?.services === "ready";
    services.forEach((service) => {
      const row = document.createElement("tr");
      appendCell(row, service.service_id, { code: true });
      appendCell(row, readable(service.capability));
      const stateCell = appendCell(row, t(`admin.health.state.${service.state}`));
      stateCell.className = `admin-health-state is-${service.state}`;
      appendCell(row, new Date(service.observed_at).toLocaleString());
      const runbookCell = document.createElement("td");
      const runbookLink = document.createElement("a");
      runbookLink.href = service.runbook_path;
      runbookLink.textContent = service.runbook_id;
      runbookLink.className = "admin-health-link";
      runbookCell.append(runbookLink);
      row.append(runbookCell);
      const actionCell = document.createElement("td");
      const actionButton = document.createElement("button");
      actionButton.type = "button";
      actionButton.className = "rh-button rh-button--compact rh-button--secondary";
      actionButton.textContent = t(
        service.action_ref === "restart_service"
          ? "admin.health.restart"
          : "admin.health.diagnostics",
      );
      actionButton.disabled = !canOperate
        || service.state !== "stopped"
        || service.action_ref !== "restart_service";
      actionButton.addEventListener("click", () => submitOperation("restart", [service.service_id]));
      actionCell.append(actionButton);
      row.append(actionCell);
      target.append(row);
    });
  };

  const renderSnapshot = (snapshot) => {
    state.snapshot = snapshot;
    text("[data-admin-organization-name]", snapshot.organization_name);
    text("[data-admin-organization-id]", shortId(snapshot.organization_id));
    const organizationId = root.querySelector("[data-admin-organization-id]");
    if (organizationId) organizationId.title = snapshot.organization_id;
    text("[data-admin-recent-auth]", snapshot.recent_auth ? t("admin.answer.yes") : t("admin.answer.no"));
    text("[data-admin-permission-count]", snapshot.permissions.length);
    text("[data-admin-role]", t(`admin.role.${snapshot.role}`));
    const roleStatus = root.querySelector("[data-admin-role]");
    roleStatus?.classList.toggle("rh-status--success", snapshot.recent_auth);
    const reauth = root.querySelector("[data-admin-recent-auth-banner]");
    if (reauth) reauth.hidden = snapshot.recent_auth;
    const canManagePlugins = snapshot.permissions.includes("plugins.manage") && snapshot.recent_auth;
    root.querySelectorAll("[data-admin-plugin-form] input, [data-plugin-submit]").forEach((control) => {
      control.disabled = !canManagePlugins;
    });
    const canOperate = snapshot.permissions.includes("operations.execute") && snapshot.recent_auth;
    root.querySelectorAll("[data-admin-operation]").forEach((button) => {
      const installationOnly = button.hasAttribute("data-installation-operation");
      button.disabled = !canOperate
        || snapshot.capabilities.services !== "ready"
        || (installationOnly && !snapshot.installation_owner);
    });
    renderCapabilities(snapshot.capabilities);
    renderMembers(snapshot.members);
    renderPlugins(snapshot.plugin_installations, snapshot.plugin_operations);
    renderEvents(snapshot.events);
    renderOperationalHealth(snapshot.operational_health);
    setConsoleState("ready", t("admin.state.ready"), t("admin.state.ready_desc", { organization: snapshot.organization_name }));
  };

  async function loadSnapshot() {
    const snapshot = await apiFetch(
      endpoint("snapshotEndpointTemplate", { organization_id: state.organizationId }),
    );
    renderSnapshot(snapshot);
  }

  async function loadAdmin() {
    setConsoleState("", t("admin.state.loading"), t("admin.state.loading_desc"));
    try {
      const organizations = await apiFetch(endpoint("organizationsEndpoint"));
      if (!Array.isArray(organizations) || !organizations.length) {
        throw new RoehubApiError(t("admin.state.no_organization"), { code: "organization_missing" });
      }
      state.organizationId = organizations[0].organization.organization_id;
      await loadSnapshot();
    } catch (error) {
      const message = error instanceof RoehubApiError ? error.message : t("admin.state.failed_desc");
      setConsoleState("error", t("admin.state.failed"), message);
    }
  }

  const confirmationDialog = root.querySelector("[data-admin-confirm]");
  const confirmationInput = root.querySelector("[data-confirm-input]");
  const confirmationSubmit = root.querySelector("[data-confirm-submit]");

  function openConfirmation({ title, impact, execute }) {
    if (!(confirmationDialog instanceof HTMLDialogElement) || !(confirmationInput instanceof HTMLInputElement)) return;
    const phrase = state.snapshot?.organization_name || "CONFIRM";
    state.pendingConfirmation = execute;
    text("[data-confirm-title]", title);
    text("[data-confirm-impact]", impact);
    text("[data-confirm-phrase]", phrase);
    text("[data-confirm-status]", "");
    const reauth = root.querySelector("[data-confirm-reauth]");
    if (reauth) reauth.hidden = Boolean(state.snapshot?.recent_auth);
    confirmationInput.value = "";
    if (confirmationSubmit) confirmationSubmit.disabled = true;
    confirmationDialog.showModal();
    confirmationInput.focus();
  }

  confirmationInput?.addEventListener("input", () => {
    if (!(confirmationInput instanceof HTMLInputElement) || !confirmationSubmit) return;
    confirmationSubmit.disabled = confirmationInput.value !== (state.snapshot?.organization_name || "CONFIRM");
  });

  root.querySelector("[data-admin-confirm-form]")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (event.submitter?.value === "cancel") {
      state.pendingConfirmation = null;
      confirmationDialog?.close();
      return;
    }
    if (!state.pendingConfirmation || confirmationSubmit?.disabled) return;
    confirmationSubmit.disabled = true;
    text("[data-confirm-status]", t("admin.confirm.executing"));
    try {
      await state.pendingConfirmation();
      state.pendingConfirmation = null;
      confirmationDialog?.close();
    } catch (error) {
      const message = error instanceof RoehubApiError ? error.message : t("admin.confirm.failed");
      text("[data-confirm-status]", message);
      if (confirmationSubmit) confirmationSubmit.disabled = false;
    }
  });

  const renderPermissionDiff = ({ current, requested, declared }) => {
    const target = root.querySelector("[data-admin-plugin-diff]");
    if (!target) return;
    target.replaceChildren();
    const added = requested.filter((permission) => !current.includes(permission));
    [
      [t("admin.plugins.current"), current],
      [t("admin.plugins.requested"), requested],
      [t("admin.plugins.added"), added],
      [t("admin.plugins.declared"), declared],
    ].forEach(([labelValue, values], index) => {
      const row = document.createElement("div");
      row.className = `admin-diff__row${index === 2 && values.length ? " admin-diff__added" : ""}`;
      const label = document.createElement("strong");
      label.textContent = labelValue;
      const value = document.createElement("span");
      value.textContent = values.join(", ") || "—";
      row.append(label, value);
      target.append(row);
    });
    return added;
  };

  root.querySelector("[data-admin-plugin-form]")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!(event.currentTarget instanceof HTMLFormElement)) return;
    const form = event.currentTarget;
    const formData = new FormData(form);
    const bundleId = String(formData.get("bundle_id") || "").trim();
    const instanceName = String(formData.get("instance_name") || "").trim();
    const requested = formData.getAll("permissions").map(String).sort();
    try {
      setConsoleState("", t("admin.plugins.validating"), bundleId);
      const validated = await apiFetch(
        endpoint("pluginValidateTemplate", { organization_id: state.organizationId }),
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ bundle_id: bundleId }),
        },
      );
      const currentInstallation = state.snapshot?.plugin_installations?.find(
        (plugin) => plugin.plugin_id === validated.plugin_id,
      );
      const current = currentInstallation?.granted_permissions || [];
      const added = renderPermissionDiff({ current, requested, declared: validated.permissions });
      setConsoleState("ready", t("admin.plugins.validated"), validated.plugin_id);
      openConfirmation({
        title: t("admin.confirm.plugin_title"),
        impact: t("admin.confirm.plugin_impact", {
          plugin: validated.plugin_id,
          added: added.length ? added.join(", ") : t("admin.plugins.none"),
        }),
        execute: async () => {
          await apiFetch(
            endpoint("pluginInstallTemplate", { organization_id: state.organizationId }),
            {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "Idempotency-Key": `admin-plugin-${Date.now()}`,
              },
              body: JSON.stringify({
                bundle_id: bundleId,
                instance_name: instanceName,
                permissions: requested,
                config: {},
              }),
            },
          );
          form.reset();
          await loadSnapshot();
        },
      });
    } catch (error) {
      setConsoleState("error", t("admin.plugins.failed"), error instanceof Error ? error.message : String(error));
    }
  });

  const progressByState = { accepted: 18, running: 58, succeeded: 100, failed: 100, rejected: 100, unknown: 72 };

  const renderOperation = (operation, idempotencyKey = state.currentOperationKey) => {
    const panel = root.querySelector("[data-admin-operation-status]");
    if (panel) panel.hidden = false;
    text("[data-operation-title]", `${readable(operation.action)} · ${shortId(operation.operation_id)}`);
    text("[data-operation-state]", t(`admin.operation_state.${operation.state}`));
    text("[data-operation-detail]", readable(operation.detail_code));
    const bar = root.querySelector("[data-operation-progress]");
    if (bar instanceof HTMLElement) bar.style.width = `${progressByState[operation.state] || 12}%`;
    const reconcile = root.querySelector("[data-operation-reconcile]");
    if (reconcile) reconcile.hidden = operation.state !== "unknown";
    state.currentOperationId = operation.operation_id;
    state.currentOperationKey = idempotencyKey;
    state.currentOperationUrl = `${endpoint("operationEndpointTemplate", { organization_id: state.organizationId })}/${encodeURIComponent(operation.operation_id)}`;
  };

  const pollOperation = async (generation, attempts = 0) => {
    if (generation !== state.pollGeneration || !state.currentOperationUrl || attempts >= 20) return;
    try {
      const operation = await apiFetch(state.currentOperationUrl, {
        headers: { "Idempotency-Key": state.currentOperationKey },
      });
      renderOperation(operation);
      if (["accepted", "running"].includes(operation.state)) {
        window.setTimeout(() => pollOperation(generation, attempts + 1), 1000);
      } else {
        await loadSnapshot();
      }
    } catch (error) {
      text("[data-operation-detail]", error instanceof Error ? error.message : String(error));
    }
  };

  const submitOperation = async (action, services = []) => {
    const releaseInput = root.querySelector("[data-admin-release-version]");
    const isRelease = ["install", "update", "rollback"].includes(action);
    const releaseVersion = releaseInput instanceof HTMLInputElement ? releaseInput.value.trim() : "";
    if (isRelease && (!releaseVersion || !releaseInput?.checkValidity())) {
      releaseInput?.reportValidity();
      return;
    }
    openConfirmation({
      title: t(`admin.confirm.operation_title`, { action: t(`admin.operation.${action}`) }),
      impact: t("admin.confirm.operation_impact", { action: t(`admin.operation.${action}`) }),
      execute: async () => {
        const idempotencyKey = `admin-${action}-${Date.now()}`;
        const operation = await apiFetch(
          endpoint("operationEndpointTemplate", { organization_id: state.organizationId }),
          {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "Idempotency-Key": idempotencyKey,
            },
            body: JSON.stringify({
              action,
              profile: "base",
              services,
              release_version: isRelease ? releaseVersion : null,
            }),
          },
        );
        renderOperation(operation, idempotencyKey);
        setPanel("updates");
        state.pollGeneration += 1;
        window.setTimeout(() => pollOperation(state.pollGeneration), 600);
      },
    });
  };

  root.querySelectorAll("[data-admin-operation]").forEach((button) => {
    button.addEventListener("click", () => submitOperation(button.dataset.adminOperation || ""));
  });

  root.querySelector("[data-operation-reconcile]")?.addEventListener("click", async () => {
    if (!state.currentOperationUrl) return;
    try {
      const operation = await apiFetch(`${state.currentOperationUrl}:reconcile`, {
        method: "POST",
        headers: { "Idempotency-Key": state.currentOperationKey },
      });
      renderOperation(operation);
      if (["accepted", "running"].includes(operation.state)) {
        state.pollGeneration += 1;
        window.setTimeout(() => pollOperation(state.pollGeneration), 600);
      }
    } catch (error) {
      text("[data-operation-detail]", error instanceof Error ? error.message : String(error));
    }
  });

  root.querySelectorAll("[data-admin-section]").forEach((button) => {
    button.addEventListener("click", () => setPanel(button.dataset.adminSection || "overview"));
  });
  root.querySelector("[data-admin-reload]")?.addEventListener("click", loadAdmin);
  loadAdmin();
}
