---
prompt_name: 06-local-auth-sessions-and-recovery
repo: roehub.com
scope: "Implement passkey-first local authentication, optional password/TOTP, server sessions, recent-auth, bootstrap and recovery."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "06", prerequisites: ["05"], previous_stage_gate: "Stage 05 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: auth/browser/security rules}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: accepted local auth contract}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: predecessor evidence}
  task_entrypoints:
    - {path: src/trading/contexts/identity/, why: sessions and users}
    - {path: apps/api/routes/, why: auth API}
    - {path: apps/web/, why: login and recovery UI}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: contract-impact-analysis, timing: before auth changes, reason: cookies, sessions, API and identity semantics}
  - {skill: backend-quality-gates, timing: verification, reason: backend gates}
  - {skill: browser-qa-evidence, timing: after implementation, reason: real login/recovery/recent-auth proof}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/identity/, apps/api/, apps/web/, migrations/postgres/, tests/, docs/architecture/identity/, docs/architecture/platform/, docs/runbooks/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated browser and identity changes
file_manifest:
  expected_primary_touches: [src/trading/contexts/identity/, apps/api/, apps/web/, migrations/postgres/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/06-local-auth-sessions-and-recovery.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [docs/architecture/identity/, docs/runbooks/, docs/architecture/README.md]
validation_strategy: {depth: browser, acceptance_surfaces: [passkey registration/login, optional password and TOTP, logout/revocation, recovery, bootstrap, CSRF, recent-auth, closed registration]}
proof_boundary: {label: N/A, exclusions: [external OIDC, production authentication]}
authority: {implementation_write: true, git_publish: false, production_mutation: false}
---

# Objective

Deliver a secure local authentication mode that does not require Keycloak and remains recoverable through a one-time `roehubctl` bootstrap procedure.

# Requirements

- Use WebAuthn/passkeys as the preferred method; password and TOTP are optional and use current approved cryptographic libraries only.
- Store opaque server-side sessions; enforce secure cookie, CSRF, rotation, revocation, expiry and `recent-auth` semantics.
- Registration is closed by default. Bootstrap the first `installation_owner` through a single-use local operation; invitations create later users.
- Recovery codes are one-time, hashed, redacted and auditable. No secret is returned after initial display.
- Define rate limits, lockout/abuse behavior and safe generic errors without account enumeration.
- Preserve organization/RBAC semantics from Stage `05`.

# Browser authentication and evidence safety

- This clean local-auth flow must use a disposable local installation identity created through the Stage `06` bootstrap path. The legacy username `smoke_e2e_keycloak` is intentionally inapplicable to primary local-auth acceptance.
- If a comparison against the existing Keycloak-backed runtime is necessary, use only username `smoke_e2e_keycloak`; obtain its password from host-local `ROEHUB_SMOKE_E2E_PASSWORD` in `/Users/daniildegtyarev/.config/roehub/roehub.env`, never from chat.
- Disposable local credentials, WebAuthn material, recovery codes, passwords, cookies and tokens must not appear in prompt files, command output, screenshots, traces, reports or ledgers. Capture only redacted state and outcome.

# Validation

Run focused backend gates, database integration and real browser flows for bootstrap, passkey register/login, optional fallback as implemented, logout, recovery, expired/revoked sessions, CSRF and privileged `recent-auth`. Check console/network and mobile/desktop basic layout. Never store credentials in traces or reports.

# Stop rules

Block on an unavailable browser/authenticator surface required for acceptance, account enumeration, raw credential logging, unrecoverable owner state, weak fallback, or any path that grants `recent-auth` without a fresh ceremony. Update ledger after evidence.
