---
evidence_id: ROEHUB-DELIVERY-MODEL-CONSOLIDATION-2026-07-22
ticket_id: ROEHUB-DELIVERY-MODEL-CONSOLIDATION-2026-07-22
verdict: local_validation_passed
observed_at: 2026-07-22T13:09:37Z
---

# Roehub delivery-model consolidation evidence

## Delivered model

- Replaced the two former delivery graphs with
  `.codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json`.
- Preserved all 15 ticket IDs and their real `depends_on` relationships;
  a direct comparison against both replaced graph versions returned
  `ticket_set_equal=True` and `dependency_map_equal=True`.
- Updated every graph ticket front matter to the unified graph. The four
  factual `accepted` tickets remain unchanged in status and retain their
  existing evidence.
- The only `ready` ticket is
  `ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20`; no graph ticket is
  `active`. The remaining priority queue has the requested 11 tickets.
- Removed status and priority duplication from the UI migration registry. It is
  now migration context only; Linear references are functional-structure input,
  not a tracker or authority for task status, dependency, or priority.
- Replaced `.codex/PLANS.md` with a historical stub and added a GitHub CI step
  plus a standard-library validator for delivery-model drift.

## Local verification

- `python -m tools.delivery.validate_roehub_delivery_model` — passed.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_delivery_model.py`
  — `4 passed in 0.04s`.
- Focused Ruff format/check and focused Pyright for the validator and its test
  — passed, `0 errors, 0 warnings, 0 informations`.
- `uv run ruff check .` — passed.
- `python .codex/hooks/tests/run_tests.py` — all active hook regressions
  passed.
- Changed JSON files parse successfully: unified graph, UI registry, and
  generated project map.
- `python -m tools.docs.generate_docs_index --check` and
  `python -m tools.docs.generate_project_map --check` — passed; the architecture
  index was refreshed under the required lock and remained unchanged.
- `python tools/release/oss_metadata.py --check` — passed.
- `git diff --check` — passed.
- CI classification selected code/docs/full test shards because CI workflow
  changed; `web_image_changed=false`.

## Cold review

- Mode: cold self-review with `architecture-review`; no independent subagent was
  started because this ticket prohibits parallel execution flows.
- Verdict: `Release` after removing the migration-registry and evidence status
  duplicates found during review.
- Reviewed: source-of-truth separation, dependency preservation, queue order,
  accepted evidence preservation, CI routing, and the no-tracker boundary.

## Residual risk and proof boundary

- The unscoped `uv run pyright` reports pre-existing `149 errors, 2 warnings`
  under untracked `local_artifacts/rl_trading/**`; the changed paths pass focused
  Pyright and no local-artifact file was modified.
- This evidence proves repository delivery metadata and local validation only.
  It does not claim browser, runtime, deployment, image-publication, or Linear
  interaction proof. GitHub Actions remains the published-result boundary.
