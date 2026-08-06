---
doc: supporting-visual-reference
version: "1.0"
status: accepted_supporting_owner_input
language: en
program_id: roehub-local-platform-web-ui-2026-08
evidence_mode: renderable_html_observed_via_loopback
authority_effect: supporting_only
---

# Custometry Pilot Candidate v2 — Supporting Roehub Application Analogue

## Source identity

The owner identified this pilot as close in substance to the desired Roehub UI:

- Source:
  `/Users/daniildegtyarev/Projects/Custometry/.codex/delivery/ui-design-programs/custometry-v2/evidence/pilot-candidate-v2/ru/source.html`
- Source SHA-256:
  `c816ac191dd9cdb4806d13cc0e742aa342df445ea8add9b638f698e4d56bdb08`
- Local runtime asset: Apache ECharts `6.1.0`
- Asset SHA-256:
  `b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0`
- Custometry candidate status at inspection: `candidate_review`, with
  `authority_effect: none_until_owner_acceptance` inside the Custometry
  program.

Roehub uses this exact observed revision only as a supporting application
analogue. It does not replace the accepted Linear native-image visual-language
authority, does not grant source fidelity, and does not import Custometry
product semantics.

The captured PNG review aids remain local and are intentionally ignored by the
public Roehub repository. Their hashes and observed coverage are tracked here.
The Custometry source is not copied or mutated by this Roehub program.

## Reference hierarchy

1. **Linear screenshots:** normative visual and interaction grammar.
2. **This Custometry candidate:** a concrete translation of that grammar into
   a dense analytical Web application.
3. **Roehub v23:** Roehub-specific Backtest domain and workbench relationship
   evidence.
4. **Current Roehub UI code:** implementation, compatibility and current-state
   evidence only.

## Browser observation

The exact source was served from loopback and inspected in Chromium with
`playwright-cli`. No external or data-changing action was exercised.

| Capture | SHA-256 | Coverage |
|---|---|---|
| `overview-1440x900.png` | `4c6753c30d12a0960d6d7d4aa1de90b9669323648615c4e821487968d89f6a5a` | raised workspace, grouped navigation, three-level header, KPI property rail, chart and table hierarchy |
| `chart-settings-1440x900.png` | `d72092aa261ac572c05e61bd641a97162167e123aff7d44ddd393547313012ba` | quiet icon action and anchored settings popover |
| `inspector-1440x900.png` | `80fbe0987cb900475b4a505ab33843e62b77e6de08dcbab5d3408f598dcd0fa4` | docked context inspector and main-content reflow |
| `overview-1024x900.png` | `971d8be76817a5062a006fad8fd352ac4449bbe68d938325d58605cb51c44670` | automatic navigation rail, retained workspace identity, zero root overflow |
| `overview-820x900.png` | `8ded75a4a161b71743908e9eccd5dde5b87857a73944407120471afb1977da55` | minimum Roehub Web anchor, local analytical overflow, zero root overflow |
| `filter-popover-820x900.png` | `917a4524307c255af123065e423fa78e994fa143d9f97405b3c9a080dee76a2c` | bounded one-column filter popover at the minimum anchor |

Observed layout facts:

- `1440×900`: full grouped sidebar and raised workspace.
- `1024×900`: `58 px` navigation rail, `958 px` workspace, zero root
  horizontal overflow.
- `820×900`: `52 px` navigation rail, `763 px` workspace, zero root
  horizontal overflow.
- Browser console after the inspected states: zero errors and zero warnings.

## What Roehub should inherit

- The shell construction: dark canvas, navigation underneath, one rounded
  workspace plane with a restrained perimeter.
- Compact header, local tabs and exactly one context toolbar.
- A flat decision-metric rail instead of KPI cards.
- One primary analytical surface followed by a dense evidence table.
- Contextual icon actions and anchored popovers that do not add permanent
  toolbar noise.
- A docked inspector that preserves the selected object and causes deliberate
  content reflow.
- Navigation-rail transformation at narrow Web widths without changing
  navigation identity.
- Local overflow for analytical rows/tables instead of a mobile redesign.

## Roehub-specific corrections

- Do not copy Custometry navigation labels, report structure, metrics,
  comparison language, fixture data, or cyan/gold chart semantics.
- Roehub results need the explicit layers `Overview → Variants → Compare →
  Variant detail → Raw`, which the single-report Custometry pilot does not
  contain.
- Backtest creation and Backtest-to-Strategy promotion remain separate Roehub
  journeys; this pilot supplies no authority for them.
- At 820 px, truncated metric and table content must expose an obvious local
  scroll affordance or adaptive column priority; clipping alone is not enough.
- The inspector should dock at wide widths, but become a bounded drawer or
  focused subview when opening it would make the analytical canvas too narrow.
- Keep the practical text floor above the pilot's smallest uppercase labels
  where long-session readability would suffer.

## Proof boundary

This evidence proves only the observed local render and interactions of the
hash-pinned Custometry candidate. It does not prove Roehub screen fidelity,
backend compatibility, data semantics, accessibility conformance, production
runtime, persistence, authorization, deployment, or owner acceptance of any
future Roehub screen.
