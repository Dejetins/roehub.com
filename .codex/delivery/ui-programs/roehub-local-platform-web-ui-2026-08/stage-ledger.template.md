---
artifact_kind: ui_design_program_stage_ledger
ledger_status: draft
execution_mode: manual_sequential
goal_artifact_required: false
plan_doc: <plan-doc>
prompt_pack_dir: <prompt-pack-dir>
stage_ledger: <stage-ledger>
current_stage: none
Next stage allowed: false
---

# UI Design Program Stage Ledger

| Stage instance | Gate | Target ID | Prompt | Status | Dependencies | Evidence | Transition receipt | Owner decision | Executor claim | Claimed at |
|---|---|---|---|---|---|---|---|---|---|---|
| G0@<program-id>-r1 | G0 | `<program-id>` | `<prompt-path>` | pending | — | `<evidence-path>` | `<transition-receipt>` | N/A | — | — |
| G1@atlas-r1 | G1 | atlas-r1 | `<prompt-path>` | pending | G0@<program-id>-r1 | `<evidence-path>` | `<transition-receipt>` | N/A | — | — |
| G2@program-r1 | G2 | program-r1 | `<prompt-path>` | pending | G1@atlas-r1 | `<evidence-path>` | `<transition-receipt>` | N/A | — | — |
| G3@<program-id>-r1 | G3 | `<program-id>` | `<prompt-path>` | pending | G2@program-r1 | `<evidence-path>` | `<transition-receipt>` | required | — | — |
| G4@<family-id>-r1 | G4 | `<family-id>-r1` | `<prompt-path>` | pending | G3@<program-id>-r1 | `<evidence-path>` | `<transition-receipt>` | required | — | — |
| G5@<wave-id>-r1 | G5 | `<wave-id>-r1` | `<prompt-path>` | pending | `G4@<all-current-family-ids>-r1, G5@<prerequisite-wave-ids>-r1` | `<evidence-path>` | `<transition-receipt>` | required | — | — |
| G6@<program-id>-r1 | G6 | `<program-id>` | `<prompt-path>` | pending | `G5@<required-wave-id>-r1` | `<evidence-path>` | `<completion-receipt>` | required | — | — |

Create one G4 row per accepted family target and one G5 row per wave target.
Every G5 row depends on every current-authority G4 row plus its declared
prerequisite waves. The first G5 claim is allowed only after all current G4
rows are accepted.
`needs_input` is non-terminal and resumes the same row only after the recorded
decision packet is answered through a valid `owner-input-response` receipt.
`accepted`, hard `blocked`, and `superseded` are
terminal. Create a new revision only when target, scope, dependencies, accepted
sources, prompt requirements, or acceptance criteria materially change.

## Stage details

Create exactly one section like this for every table row. Keep it after the row
reaches a terminal state; this is stage history, not a replaceable “latest”
summary.

### `G0@<program-id>-r1`

- title: `<human-readable-stage-title>`
- report_path: `<stage-report-path>`
- expected_touch_zones: `<comma-separated-paths-or-named-zones>`
- proof_boundary: `<exact-proof-boundary>`
- blocker_policy: `needs_input_for_owner_questions_hard_block_only_for_unsafe_or_unrecoverable_conditions`
- decision_packet: `none`
- resume_condition: `none`
- resume_evidence_ref: `none`
- resume_evidence_sha256: `none`
- transition_receipt: `<transition-receipt>`
- transition_receipt_sha256: `none`
- execution_allowed: `true`
- replaces_stage: `none`
- repair_evidence_ref: `none`
- historical_outcome: `none`
- current_authority: `true`
- invalidated_by_ref: `none`
- superseded_by_stage: `none`
- supersedes_stage: `none`
- incoming_transition_receipt: `none`
- incoming_transition_receipt_sha256: `none`

Repeat this exact field set for every G1-G6 row. Set `execution_allowed: true`
only when the prompt exists and dependencies can be evaluated. For a repaired
hard blocker, create a new stage instance and set `replaces_stage` plus a
readable `repair_evidence_ref`; never reopen the blocked row in place.

For a revision-gated correction of previously terminal evidence, preserve the
historical row and its `historical_outcome`, set `current_authority: false`,
record a readable `invalidated_by_ref` plus `superseded_by_stage`, and make the
new row point back with `supersedes_stage`. This does not use the hard-blocker
`replaces_stage` field.

## Current owner input

- Decision packet: `<path-or-none>`
- Questions: `<zero-to-three-concise-questions>`
- Resume condition: `<exact-condition-or-none>`

## Current blockers

- `<blocker-or-none>`

## Latest validation

- Commands: `<exact-commands>`
- Observed boundary: `<boundary>`
- Result: `<passed-or-blocked>`

## File manifest summary

- created: []
- modified: []
- deleted: []
- outside_expected_paths: []
- foreign_changes_excluded: true

## Handoff

- Residual risk: `<risk>`
- Next executor must know: `<handoff>`
- Transition receipt: `<validated-receipt-path>`
- Next stage allowed: false
