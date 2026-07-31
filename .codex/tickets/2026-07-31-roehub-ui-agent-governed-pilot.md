---
ticket_id: ROEHUB-UI-AGENT-GOVERNED-PILOT-2026-07-31
status: active
owner: codex
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22
evidence:
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-figma-page-scaffold.json
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-preparation-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-1-better-interface-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-2-better-interface-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-2-owner-rejection-audit.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-3-source-contamination-rejection.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-4-action-gate-rejection.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-5-linear-native-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-selection-candidate.json
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-selection-candidate-dir-003.json
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-tahoe-v3-owner-rejection.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-tahoe-v4-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-attempt-tahoe-v4-owner-feedback.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v5-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v6-owner-feedback.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v6-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v7-better-interface-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v8-owner-refinement.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-07-31-direction-linear-black-v8-interface-review.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-refinement.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-direction-linear-black-v9-owner-acceptance.md
  - .codex/delivery/evidence/roehub-ui-agent-governed-pilot/2026-08-01-backtests-figma-pilot-handoff-cold-head-review.md
acceptance_authority: product_owner
agent_self_acceptance: prohibited
current_checkpoint: library_slice_review
---

# Prove the agent-governed Roehub Figma delivery loop

## Outcome

One small Backtests candidate proves that Codex can coordinate Figma work, reject structurally
invalid output, perform no more than two bounded automatic repair attempts per owner-feedback
cycle, and show the product owner only a gated review packet. Explicit product-owner refinements
start a new bounded direction cycle; they are not inferred automatic repairs.

## Authoritative sources

- `docs/architecture/ui/roehub-agent-governed-figma-delivery-standard-v2.md`;
- `docs/architecture/ui/roehub-ui-agent-delivery-contract-v1.json`;
- `docs/architecture/ui/roehub-ui-component-registry-schema-v1.json`;
- `docs/architecture/ui/roehub-ui-composition-manifest-schema-v1.json`;
- `docs/architecture/ui/roehub-backtests-process-pilot-brief-v1.md` after
  `pilot_brief_approval`;
- current Roehub product, route, authorization, and Backtests contracts selected during brief
  preparation.

The former Figma file `GBzmB9evtzqnAYNjp9W1sr`, rejected React visuals, historical design-tool work,
and previous Figma frames are prohibited inputs.

## Executor handoff

The standalone cross-chat executor prompt is
`.codex/agents/generated/roehub-backtests-figma-pilot-v1/task.md`. It carries this ticket from
`library_slice_review` through `pilot_final_approval`, but it does not replace this ticket or grant
Git publication, runtime implementation, or product-owner acceptance by inference.

The prompt carries
`docs/architecture/ui/roehub-linear-black-authenticated-workspace-visual-standard-v1.md` as a
`proposed_for_library_slice_review` translation of the accepted v9 direction. It is not a reusable
cross-screen authority until the product owner accepts the exact named revision at that checkpoint.

## Scope

- Establish the two-file page scaffold and baseline inventory.
- Prepare one neutral-black, compact, optically symmetric Workbench refinement under the
  product-owner-specified `linear_black_backtests_workbench_v9` owner-refinement evidence outside
  the reusable library boundary.
- Record explicit product-owner acceptance of v9 at `direction_selection`.
- Create only the tokens and components needed for:
  - one Backtests toolbar;
  - one representative job row;
  - one detail-dock header;
  - one loading or degraded state.
- Validate the accepted library slice, request product-owner publication and product-file
  enablement in Figma, then verify the published keys read-only.
- Create one composition manifest and render one isolated product candidate from library instances.
- Run a deliberate negative test that the validator must reject before repair.
- Run structural and independent visual audits automatically.
- Present the gated candidate for `pilot_final_approval`.

## Checkpoints

1. `pilot_brief_approval`: product owner confirms the bounded slice and review vocabulary.
2. `direction_selection`: product owner selects one named visual direction.
3. `library_slice_review`: product owner reviews the gated token/component specimen; Codex cannot
   publish or treat it as accepted before this decision.
4. `composed_candidate_review`: product owner reviews the manifest-built Backtests slice.
5. `pilot_final_approval`: product owner accepts or rejects the named evidence packet.

All routine structural and visual QA runs before each checkpoint without a separate user request.

## Owned repository paths

- this ticket and its evidence;
- a ticket-owned product/UI brief, component registry, composition manifest, and audit reports under
  `docs/architecture/ui/` or `.codex/delivery/evidence/`;
- `tools/design/validate_roehub_ui_delivery.py` and its focused tests;
- current routing documents only for exact identities or rules introduced by this pilot.

## Owned Figma boundaries

- `Roehub UI Library` (`rgbNUPCuV7q2pARG4Cml8V`): task-created nodes on `80 Audit Sandbox`, then
  exact accepted assets on their owner pages;
- `Roehub Authenticated Platform UI` (`nzKVsXuCmoTbHJGckHfK3T`): task-created exploration nodes on
  `01 Direction Review`, sandbox nodes on `80 Audit Sandbox`, and exactly one candidate on
  `02 Candidate`;
- no mutation of `03 Accepted` before explicit product-owner acceptance;
- no access to the historical file as task input.

## Executor constraints

- Figma MCP Plugin API is the default canonical writer.
- Library publication and enablement are manual product-owner actions in the Figma UI; Codex owns
  the post-action read-only verification.
- A Figma-native agent is not required and may not mutate library masters or accepted nodes.
- Every write uses an exact parent node and returns created or changed IDs.
- Unknown components, variants, actions, fields, states, and raw UI nodes fail preflight.
- A failed candidate gets at most two bounded automatic repair attempts.

## Acceptance gates

- repository contract and JSON Schema validate;
- Figma account, plan, file keys, page IDs, and page names match the contract;
- component registry and manifest contain no unknown or duplicate identities;
- library assets bind approved tokens and styles and contain no detached or placeholder content;
- product candidate uses published library instances only;
- required Backtests content and state are present;
- mutation stayed inside the exact task-owned boundary;
- screenshot review passes hierarchy, density, typography, alignment, contrast, clipping, and
  content-extreme checks;
- deliberate invalid fixture is rejected;
- final packet names exact IDs, checks, screenshots, unresolved gaps, and the product-owner decision.

## Non-goals

- No complete Roehub design system, full Backtests screen, implementation, browser proof, API change,
  route change, publishing to Git, deployment, or acceptance by agent inference.
- No attempt to reproduce or repair the previous design state.

## Proof boundary

The pilot proves the governance and Figma structure of one small design slice. It does not prove a
production-ready library, a complete screen, runtime behavior, accessibility implementation, or
implementation readiness.
