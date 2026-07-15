---
prompt_name: 13-data-sources-panels-and-design-system
repo: roehub.com
scope: "Implement data-source/panel/app contributions, RoehubDataFrame/v1 and a host-owned accessible visualization system."
language: {implementation: project_default, agent_report: ru}
prompt_pack_execution: {mode: goal_driven, plan_doc: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, prompt_pack_dir: .codex/agents/generated/roehub-self-hosted-oss-platform-v1/, stage_ledger: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, goal_mode_optional: true, goal_artifact_required: false}
stage: {id: "13", prerequisites: ["09", "12"], previous_stage_gate: "Stages 09 and 12 accepted."}
branch_policy: {default_branch: main, separate_branch_allowed: false, stage_specific_branches_forbidden: true, worktree_allowed: false, stash_allowed: false}
context_sources:
  always_read:
    - {path: .codex/AGENTS.md, why: UI/browser/plugin contracts}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md, why: DataFrame and visual-host decisions}
    - {path: docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md, why: research/plugin prerequisites}
    - {path: docs/architecture/apps/web/web-ui-design-manifest-v1.md, why: existing design contract}
  task_entrypoints:
    - {path: apps/web/, why: current SSR/HTMX/JavaScript UI}
    - {path: apps/api/dto/, why: current visualization read models}
    - {path: apps/web/dist/js/pages/dashboard.js, why: current custom SVG charts}
    - {path: apps/web/dist/js/pages/backtests.js, why: current Canvas result charts}
skill_routing:
  - {skill: staged-plan-runner, timing: before actions, reason: ledger gate}
  - {skill: ui-ux-pro-max, timing: before UI implementation, reason: accessible institutional design system and chart states}
  - {skill: contract-impact-analysis, timing: before DataFrame/panel API freeze, reason: public UI/data contracts}
  - {skill: browser-qa-evidence, timing: verification, reason: real browser, responsive and accessibility evidence}
change_ownership:
  parallel_main_expected: true
  owned_change_scope: [src/trading/contexts/extensions/, src/trading/integration/, apps/api/, apps/web/, sdk/, schemas/plugins/, tests/, docs/architecture/apps/web/, docs/architecture/platform/, docs/architecture/README.md]
  foreign_changes_policy: preserve unrelated redesign work and existing accepted UI contracts
file_manifest:
  expected_primary_touches: [src/trading/integration/, apps/api/, apps/web/, schemas/plugins/, tests/, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/13-data-sources-panels-and-design-system.md, docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/roehub-self-hosted-oss-platform-v1-stage-ledger.md]
  possible_secondary_touches: [src/trading/contexts/extensions/, sdk/, docs/architecture/apps/web/, docs/architecture/README.md]
validation_strategy: {depth: browser, acceptance_surfaces: [RoehubDataFrame/v1, query limits/cancellation, declarative panels, real browser interaction, keyboard/focus, light/dark, reduced motion, responsive viewports, degraded data source]}
proof_boundary: {label: N/A, exclusions: [arbitrary same-origin plugin JavaScript, production dashboard replacement]}
authority: {implementation_write: true, git_publish: false, production_mutation: false, dependency_addition: requires explicit repository-compatible justification}
---

# Objective

Let signed data-source plugins feed consistent, high-quality panels while Roehub retains layout, visual language, accessibility and security.

# Requirements

- Define `RoehubDataFrame/v1` fields, types, units, labels, rows/columns, metadata, freshness, notices, partial status and bounded errors.
- Data-source queries derive organization from session; enforce read-only default, max time/rows/bytes/points, cancellation and redaction.
- Provide declarative `panel` and compositional `app` contributions. Do not execute plugin JavaScript in the main origin.
- Build host render adapters, initially for trading time series, general analytics and declarative research; public API must not depend on a library.
- Preserve existing Roehub design tokens and evolve toward a dense institutional drill-down UI with loading/empty/error/degraded/success states and table alternatives.
- Do not anonymously embed Grafana or expose data-source credentials to the browser.

# Browser authentication and evidence safety

- Prefer a disposable local-auth identity in the isolated target installation. If the check runs against the still-Keycloak-backed current Roehub runtime, use username `smoke_e2e_keycloak` and read the password only from host-local `ROEHUB_SMOKE_E2E_PASSWORD` in `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Never request credentials in chat or persist passwords, cookies, tokens, authorization headers, secret-bearing queries or private data-source rows in screenshots, traces, logs, reports or the ledger. Use sanitized fixtures and redacted artifacts.

# Validation

Run API/contract/source gates and real browser QA at 375, 768, 1024 and 1440 widths. Cover keyboard/focus, light/dark, reduced motion, loading/empty/error/degraded/partial states, chart units/table alternative, console errors and failed network requests. Use a controlled external database fixture and prove query limits/cancellation and cross-org denial.

# Stop rules

Block on same-origin arbitrary plugin code, browser secrets, unbounded query, inaccessible chart-only information, layout overflow, cross-org data or a library-specific public contract. Update ledger after evidence.
