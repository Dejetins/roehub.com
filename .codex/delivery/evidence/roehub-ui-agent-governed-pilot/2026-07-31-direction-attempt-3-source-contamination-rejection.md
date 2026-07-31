# Direction attempt 3 — source-contamination rejection

## Outcome

- Status: `rejected_before_figma`.
- Generated base:
  `/Users/daniildegtyarev/.codex/generated_images/019fb4fa-5eba-7fa0-a94f-ab5e05510b4d/exec-bdde29ac-4aaf-4c6d-89ac-3632c8ba88f4.png`.
- Density repair:
  `/Users/daniildegtyarev/.codex/generated_images/019fb4fa-5eba-7fa0-a94f-ab5e05510b4d/exec-568d99b4-529b-4011-b9d6-53175a472f09.png`.
- Figma mutation: none.
- Reuse: prohibited.

## Failed gates

The first raster failed the `44-52px` worklist-row target. The bounded repair improved density and
populated the inspector, but the overall composition remained an obvious derivative of rejected
`DIR-001`: full-width one-row list, familiar toolbar geometry, and a large empty main canvas.

## Root cause and correction

The generator received the rejected screenshot as a content-reference image. Despite explicit
negative instructions, the visual model retained its composition. This confirms that rejected
frames and rasters cannot safely be provided even for content inventory.

The next attempt receives no visual input from `DIR-001` or this failed attempt. Roehub content is
provided only as text from the approved brief; the single visual reference is limited to
interaction density and workspace relationships.
