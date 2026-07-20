# Evidence — ROEHUB-LOCAL-UI-DESIGN-SYSTEM-CONTRACT-2026-07-20

## Terminal verdict

`accepted` for the documentation-contract ticket only. The three new
design-system deliverables remain `ready_for_product_review`; this verdict does
not accept their visual values, create a Penpot artifact, or authorize product
implementation.

## Boundary and sources

- Work branch: `codex/roehub-local-ui-design-system-contract`, created from
  clean local `main` at `7314c1febeb4ccc50bba585f513921d27711f59a`.
- `bec15b7a` was confirmed as an ancestor of local `main` before the branch was
  created.
- Read authority/context: root and `.codex` `AGENTS.md`, selected ticket,
  accepted local IA, screen registry, access/route contract, UI delivery
  architecture, product requirements, Web entrypoint/template asset inventory,
  and historical `prototypes/roehub-v2/` source/QA material.
- No Penpot file, Penpot connector or product-code path was read or written.
  The prototype was treated as historical evidence, not target authority.

## Created contract surface

- `docs/architecture/apps/web/roehub-local-platform-design-system-contract-v1.md`
- `docs/architecture/apps/web/roehub-local-platform-design-token-contract-v1.json`
- `docs/architecture/apps/web/roehub-local-platform-component-registry-v1.json`
- `tests/unit/docs/test_roehub_local_platform_design_system_contract.py`

The generated architecture index and project map were refreshed only for these
new documentation artifacts.

## Coverage observed by focused tests

- Exact themes: `abyss`, `graphite`, `slate`, `frost`, `paper`, `sand`.
- Exact local-platform widths: `820`, `1024`, `1440`; `390` explicitly
  excluded.
- 29 visual local screen IDs have exact required-state lists and at least one
  registered component family. The 5 non-visual contracts and one historical
  prototype entry are explicitly excluded from UI composition.
- Token contrast pairs, keyboard/focus/reduced-motion/localization declarations,
  stable future `@roehub/*` bindings, and a future-only `Roehub — Design System`
  library structure are covered.
- `ChartSpec/v1` is declarative and blocks raw JavaScript callbacks, executable
  formatters, HTML/secret-bearing tooltips, arbitrary plugins, renderers,
  transforms and unrestricted ECharts options. Units, timezone, source,
  freshness, accessible summary and table alternative are required.
- Job UI separates queue wait/ETA from measured execution/ETA; low or
  insufficient confidence suppresses a duration, and terminal/unknown states
  cannot produce a false active 100% or success claim.

## Checks run

```text
python -m pytest -q tests/unit/docs/test_roehub_local_platform_information_architecture.py tests/unit/docs/test_roehub_local_platform_design_system_contract.py
8 passed, 1 warning

python -m tools.docs.generate_docs_index --check
OK: docs/architecture/README.md is up-to-date.

python -m tools.docs.generate_project_map --check
OK: project map up-to-date (5 artifacts)

git diff --check
exit 0
```

The only warning is an existing pytest configuration warning about unknown
`asyncio_default_fixture_loop_scope`; it does not fail either focused suite.

## Cold self-review

Reviewed the generated contract names, exact JSON state coverage, route/role
non-mutation invariants, chart/ETA safety boundaries, changed-path list and
generated index/map after the first green run. Verdict: no Penpot mutation,
runtime implementation claim, public-site composition, new route, role,
capability or hidden product decision was found.

## Residual risk and next allowed step

Product owner may still reject or refine visual values/component anatomy. Future
server projections may need typed freshness, chart and progress fields, and
future ECharts renderer selection needs measured runtime proof. The next
allowed step is product-owner review of the three `ready_for_product_review`
deliverables. This accepted ticket alone does not authorize Penpot or Web UI
implementation.
