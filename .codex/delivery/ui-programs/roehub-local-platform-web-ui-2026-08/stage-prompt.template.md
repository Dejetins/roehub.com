---
artifact_kind: ui_design_program_stage_prompt
stage_instance_id: <gate>@<target>-r<revision>
gate_id: <G0-G6>
target_id: <exact-program-family-or-wave-revision>
title: <stage-title>
report_path: <stage-report-path>
proof_boundary: <observed-proof-boundary>
expected_touch_zones: <comma-separated-paths-or-named-zones>
blocker_policy: needs_input_for_owner_questions_hard_block_only_for_unsafe_or_unrecoverable_conditions
transition_receipt: <exact-transition-receipt-path>
incoming_transition_receipt: <exact-hash-pinned-incoming-receipt-for-pending-G3-or-G4>
incoming_transition_receipt_sha256: <sha256>
runtime_profile: codex.ui-stage-runtime-profiles/v1@1.6.0
execution_mode: manual_sequential
goal_artifact_required: false
current_stage: <gate>@<target>-r<revision>
rendered_review_target: <exact-G3-G6-runnable-review-path-or-none-for-G0-G2>
known_stop_resolution: none
owner_review_target: <non_visual_summary_for_G0-G2|finished_visuals_only_for_G3-G6>

visual_authority:
  source_visual_ref: <exact-current-source-visual-path>
  source_visual_sha256: <exact-current-source-visual-sha256>
  source_evidence_mode: <renderable_html|native_image|figma_adapter>
  owner_decision_ref: <exact-current-owner-decision-ref>
  screen_acceptance_scope: <exact-screen-acceptance-scope>
  visual_language_scope: <exact-visual-language-scope>
  reusable_foundation_scope: <exact-reusable-foundation-scope>
  inheritance_policy: <required-or-explicit-exception>
  mobile_scope: <unauthorized-or-authorized>

prompt_pack_execution:
  plan_doc: <plan-doc>
  prompt_pack_dir: <prompt-pack-dir>
  stage_ledger: <stage-ledger>

context:
  always_read: [<repository-AGENTS.md>, <stage-ledger>, <latest-accepted-evidence>]
  task_entrypoints: [<entrypoint-1>, <entrypoint-2>]
  conditional_bundles: []
  consult_if_needed: []

skills:
  primary: ui-design-program
  companions: []

validation_strategy:
  proof_boundary: <observed-proof-boundary>
  evidence_target: <stage-report-path>
  commands: []

file_manifest:
  expected_touch_zones: [<expected-touch-zone>]
  foreign_changes_excluded: true

safety:
  recoverable_input_policy: needs_input_and_resume_same_stage
  hard_blocker_policy: terminal_blocked_only_for_unsafe_or_unrecoverable_state
  mobile_scope: unauthorized_unless_exact_user_authorization_is_cited
  agent_self_acceptance: prohibited
  repeated_write_authorization: prohibited_when_current_task_already_authorizes_owned_paths

owner_interaction:
  raw_json_review: prohibited_by_default
  full_screen_inventory_review: prohibited_by_default
  max_questions_per_checkpoint: 3
  magic_acceptance_string_required: false
---

# Task

Execute only `<gate>@<target>-r<revision>` from the current authorized UI
program. G0 may operate on the active draft created during bootstrap; later
gates require the accepted inputs named by their runtime profile.

## Context / Current State

Name the exact current program revision, ledger state, accepted inputs required
by this gate, responsive-Web boundary, and mobile authorization.

## Requirements

Generate the stage-specific contract before work:

```text
python3 /Users/daniildegtyarev/.codex/skills/ui-design-program/scripts/ui_program_context.py --ledger <stage-ledger> --project-root <root>
```

Follow its objective, ownership, outputs, allowed unresolved values, forbidden
actions, companion references, owner surface, validation profile, and adjacent
preflight. Preserve accepted product/design decisions and use only allowed
provenance.

# Context acquisition protocol

Read `always_read`, bounded entrypoints, the generated runtime context, and
only its named companion references. Expand only for an explicit condition.

# Reading manifest

Keep baseline context to about eight files and 35k–50k tokens unless a named
blocker requires more.

# Work plan

1. Validate entry state and claim through `staged-plan-runner`.
2. Execute only the generated runtime profile.
3. Validate the current artifact and observed proof boundary.
4. Validate ledger synchronization and adjacent-stage transition.
5. Update ledger and report the owner surface.

# Acceptance criteria

The current profile passes; no known next-stage trap remains; no unauthorized
mobile work or self-acceptance occurs; `next_stage_allowed` matches the live
transition receipt.

# Implementation constraints

Do not change product semantics, hand-edit generated evidence, repeat existing
write authorization, or create a revision for an ordinary owner answer.

# Quality gates

Run the profile-emitted artifact validator, then:

```text
python3 /Users/daniildegtyarev/.codex/skills/ui-design-program/scripts/validate_stage_ledger.py --project-root <root> <stage-ledger>
python3 /Users/daniildegtyarev/.codex/skills/ui-design-program/scripts/validate_stage_transition.py --project-root <root> --ledger <stage-ledger> <transition-receipt>
```

# Adjacent-stage preflight

Use the generated `next_preflight` and all seven required live checks.

# Final output

Lead with finished visuals/result, material exceptions, and at most three
questions. Link machine evidence and state one next action.
