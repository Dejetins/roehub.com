# UI program input decision packet

- Intake: `.codex/delivery/ui-programs/roehub-local-platform-web-ui-2026-08/ui-program-intake.json`
- Status: `needs_input`
- Gate: `G0`

## Agent instructions

Before asking the owner, inspect the named repository sources and accepted pilot, measure all derivable visual values, and update the contracts yourself. Ask only questions whose answer changes product semantics, authority, scope, or an intentional baseline exception. The owner answers in natural language; the agent writes JSON.

## Owner questions

### 1. Product model and authority

Confirm the missing product facts: what the platform does end-to-end, who uses it, which release slices are in scope, and which product sources are authoritative.

Helpful answer format: Describe the user's starting point, main outcome, domain entities, roles/permissions, success/failure/recovery paths, exclusions, locales and themes.

Default recommendation: First let the agent resolve everything available from repository sources and the accepted pilot.

### 2. Screens, states, and every action

Describe the unresolved screens and controls: purpose, data, states, every visible action/button/menu item, its availability, result, feedback, failure and recovery.

Helpful answer format: For each named screen, walk through what is visible, what can be clicked or typed, what changes, where navigation goes, and what happens on empty/loading/error/denied states.

Default recommendation: First let the agent resolve everything available from repository sources and the accepted pilot.

### 3. Accepted pilot and fixed platform baseline

Confirm the unresolved pilot/baseline decisions: accepted pilot identity, allowed inheritance scope, Web width range, and any intentional shell exception such as login.

Helpful answer format: Do not redesign. Name the accepted pilot, the exact exception screens, and only the differences that are intentionally allowed. Measurements should be taken by the agent.

Default recommendation: First let the agent resolve everything available from repository sources and the accepted pilot.

## Machine findings

The detailed validator findings stay here for the agent; the owner does not need to edit JSON.

### Product model and authority

- `$.product.primary_user_outcomes has fewer items than minItems`
- `$.product.domain_entities has fewer items than minItems`
- `$.product.domain_terms has fewer items than minItems`
- `$.program_scope.included_release_slices has fewer items than minItems`
- `$.authoritative_inventory.sha256 must have type ['string']`
- `$.pilot.reviewed_theme_ids has fewer items than minItems`
- `intake_ready requires status complete`
- `intake_ready forbids placeholder at $.product.purpose`
- `intake_ready forbids placeholder at $.product.operating_model`
- `intake_ready forbids placeholder at $.program_scope.source_ref`
- `intake_ready forbids placeholder at $.authoritative_inventory.path`
- `$.program_scope.source_ref must be exact and non-placeholder`
- `intake_ready requires at least one role, permission profile, locale, and theme`
- `authoritative_inventory.path must be exact and non-placeholder`

### Screens, states, and every action

- `$.pilot.reviewed_state_ids has fewer items than minItems`
- `$.pilot.represented_screen_ids has fewer items than minItems`
- `intake_ready forbids placeholder at $.pilot.screen_acceptance_scope`

### Accepted pilot and fixed platform baseline

- `$.pilot.source_visual_sha256 must have type ['string']`
- `$.pilot.native_viewport.width must have type ['integer']`
- `$.pilot.native_viewport.height must have type ['integer']`
- `$.baseline_contract.sha256 does not match '^[0-9a-f]{64}$'`
- `intake_ready forbids placeholder at $.pilot.source_visual_ref`
- `intake_ready forbids placeholder at $.pilot.owner_decision_ref`
- `intake_ready forbids placeholder at $.pilot.visual_language_scope`
- `intake_ready forbids placeholder at $.pilot.reusable_foundation_scope`
- `intake_ready forbids placeholder at $.baseline_contract.sha256`
- `pilot.source_visual.path must be exact and non-placeholder`
- `pilot.owner_decision_ref must be an exact owner-decision reference`
- `baseline_contract.sha256 must be an exact SHA-256`
