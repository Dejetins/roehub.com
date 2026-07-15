---
prompt_name: 07-oidc-provider-integration
repo: roehub.com
scope: "Implement provider-neutral OIDC for greenfield installations while retaining local owner recovery and stable internal user identity."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "07", prerequisites: ["06"], previous_stage_gate: "Stage 06 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: auth and browser contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: greenfield local/OIDC decision}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: local auth evidence}
  task_entrypoints:
    - {path: docs/architecture/identity/identity-keycloak-auth-model-v1.md, why: current implementation baseline only; no identity import}
    - {path: src/trading/contexts/identity/, why: identity provider adapters}
    - {path: apps/api/, why: OIDC callback and sessions}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: architecture-design, timing: before provider boundary, reason: generic OIDC adapter and identity ownership}
  - {skill: contract-impact-analysis, timing: before identity changes, reason: issuer/subject/session/user identity semantics}
  - {skill: browser-qa-evidence, timing: verification, reason: local and OIDC end-to-end flows}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/identity/, apps/api/, apps/web/, migrations/postgres/, configs/, tests/, docs/architecture/identity/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve historical accepted evidence and unrelated auth work; never read current production identities or credentials
file_manifest:
  expected_primary_touches: [src/trading/contexts/identity/, apps/api/, migrations/postgres/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/07-oidc-provider-integration.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [apps/web/, configs/, docs/architecture/identity/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: browser, acceptance_surfaces: [generic OIDC discovery/JWKS/callback, invitation or provisioning link, duplicate/conflict rejection, local owner fallback, session revocation, provider outage degradation]}
proof_boundary: {label: N/A, exclusions: [current Keycloak identity import, production credentials, migrating passwords or TOTP secrets]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Make external OIDC optional and provider-neutral for a new installation. Keycloak and Pocket ID are ordinary compatible providers, not migration sources. Local `installation_owner` recovery remains available.

# Requirements

- Implement verified issuer/discovery/JWKS, authorization code flow, state/nonce/PKCE, strict redirect and subject binding.
- `AuthenticationProvider/v1` owns the OIDC boundary. Default network budgets are 3 seconds to connect, 10 seconds for a response and 15 seconds overall; configuration may tighten but not disable the deadline.
- Discovery/JWKS GET may retry at most twice with bounded jitter and validated cache semantics. A token POST with an unknown result is never blindly retried; create no Roehub session, record a redacted failure and restart the login ceremony.
- External OIDC outage degrades only that provider. Local login and existing valid Roehub sessions remain available; expose redacted discovery/JWKS/token metrics, a prolonged-outage alert and the linked `ops.roehub.io/v1` runbook.
- Map external identities by `(provider_id, issuer, subject)` to stable internal `user_id` created in the new Roehub database.
- Require an invitation, explicit provisioning rule or authenticated linking flow before a new external identity gains membership. Reject issuer/subject conflicts and account takeover.
- Clean install defaults to local authentication with zero configured OIDC providers. Enabling OIDC must not disable the local owner recovery path.
- Do not import current `keycloak_subject`, passwords, TOTP secrets, sessions or users. Do not connect to the current production identity store.
- Keep Pocket ID and Keycloak as generic OIDC compatibility fixtures, not mandatory containers.

# Browser authentication and evidence safety

- Use disposable local and disposable OIDC-provider identities created inside the isolated target installation. Current `smoke_e2e_keycloak` credentials are not applicable to this greenfield acceptance surface.
- Credentials, authorization codes, cookies, tokens, WebAuthn material and raw identity-provider responses are forbidden in screenshots, traces, logs, stage reports and the ledger. If the disposable provider or bootstrap path is unavailable, mark the affected browser boundary `blocked`.

# Validation and stop rules

Use a disposable OIDC/Keycloak-compatible fixture and real browser flows for local login, OIDC login, invitation/provisioning, linking, duplicate/conflict rejection, logout, revoked sessions and local fallback. Inject discovery, JWKS and token timeouts plus malformed/stale keys and prove the retry/degradation/error contract above. Block on subject takeover, issuer confusion, changed `user_id`, disabled local recovery, blind token retry, indefinite wait, dependency on current production identities or any need for production credentials. Update docs current-vs-target status only with observed evidence.
