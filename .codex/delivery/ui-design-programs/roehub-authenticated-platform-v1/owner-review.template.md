---
artifact_kind: ui_design_owner_review
program_id: <program-id>
gate_id: <gate-id>
validation_profile: <atlas_gate|structure_gate|program_ready|handoff_ready>
review_revision: <exact-revision>
source_hashes: []
browser_receipts: []
mobile_scope: unauthorized
agent_self_acceptance: prohibited
---

# Owner review: <gate-title>

## What is being reviewed

- Exact program/screen/family/wave revision: `<revision>`
- Exact manifest validation profile: `<validation-profile>`
- Included surfaces and states: `<scope>`
- Responsive-web anchors: `<anchors>`
- Mobile-specific design: `<authorized with exact reference | not authorized>`

## Automatic gates

| Gate | Evidence | Result |
|---|---|---|
| Contract and provenance | `<receipt>` | `<passed-or-blocked>` |
| Geometry and overflow | `<receipt>` | `<passed-or-blocked>` |
| Raster comparison | `<receipt>` | `<passed-or-blocked>` |
| Accessibility and keyboard | `<receipt>` | `<passed-or-blocked>` |
| Console and network | `<receipt>` | `<passed-or-blocked>` |

## Known exclusions and residual risks

- `<exclusion-or-risk>`

## Exact owner decision

The owner chooses one and names the unchanged revision:

- `accept <revision>`
- `reject <revision>: <reason>`
- `request changes to <revision>: <bounded changes>`

No agent may fill or infer the owner decision.

## Decision receipt record

After the owner decides, record the decision in an owner-controlled JSON
receipt or registry:

```json
{
  "decision_id": "<exact-id>",
  "status": "accepted",
  "revision": 1,
  "source_user_message_ref": "user-message://<task-id>/<message-id>",
  "accepted_value_sha256s": ["<canonical-value-sha256>"]
}
```

Hash the receipt file only after recording the decision. Downstream origins
must pin its SHA-256, JSON Pointer, and the exact accepted value hash.

For a program-gate decision, put the canonical SHA-256 of exactly one of these
identities in `accepted_value_sha256s`:

```json
{"program_id": "<program-id>", "revision": 1}
```

Use the two-field identity only for an existing compatible `atlas_gate`
revision. For `structure_gate`, `program_ready`, and `handoff_ready`, use:

```json
{
  "program_id": "<program-id>",
  "revision": 2,
  "validation_profile": "structure_gate"
}
```

Do not reuse an earlier gate's accepted value hash.
