---
{
  "schema_version": "stage-prompt/v1",
  "prompt_pack_execution": {
    "plan_doc": "implementation-plan.md",
    "prompt_pack_dir": ".",
    "stage_ledger": "iteration-ledger.md"
  },
  "stage_contract": {
    "id": "S07",
    "title": "Remove legacy implementation and reconcile every reference",
    "prompt_path": "07-remove-clickhouse-and-docs-audit.md",
    "report_path": "reports/S07.md",
    "receipt_dir": "receipts/S07",
    "depends_on": [
      "S06"
    ],
    "expected_touches": [
      "../../../../src",
      "../../../../apps",
      "../../../../tools",
      "../../../../scripts",
      "../../../../tests",
      "../../../../migrations/clickhouse",
      "../../../../infra",
      "../../../../configs",
      "../../../../schemas",
      "../../../../pyproject.toml",
      "../../../../uv.lock",
      "../../../../docs"
    ],
    "acceptance_criteria": [
      "CH adapters/DDL/client dependency and executable maintenance/benchmark paths are removed or ported; no product/test-tool runtime import remains.",
      "Every residual inventory match has a narrow documented historical or rejection-test justification; all active normative docs describe PG.",
      "Dependency lock, generated docs/map/schemas/topology and CI semantic coverage mapping are consistent."
    ],
    "proof_boundary": "Repository-wide dependency/reference closure and static/build/import gates, not complete functional performance.",
    "validation": {
      "profile": "prompt-manager/bundled-v1 plus repository boundary-specific gates",
      "checks": [
        "Follow the stage work/acceptance and section6 proof matrix; record actual commands and output paths.",
        "Reread current repository CI routing/package scripts; run only relevant checks, plus changed-boundary real infrastructure/browser/performance evidence.",
        "Run bundled draft/entry and receipt checks for transitions; artifact structure alone is not implementation proof."
      ],
      "requires_user_acceptance": false
    },
    "entry_inputs": [
      {
        "path": "../../../../pyproject.toml",
        "producer_stage": null
      },
      {
        "path": "../../../../uv.lock",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/ci/route_changes.py",
        "producer_stage": null
      },
      {
        "path": "../../../../.github/workflows/ci.yml",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/docs/generate_project_map.py",
        "producer_stage": null
      },
      {
        "path": "../../../../tools/qa/postgres_storage_prototype.py",
        "producer_stage": null
      },
      {
        "path": "implementation-plan.md",
        "producer_stage": null
      },
      {
        "path": "reports/S06.md",
        "producer_stage": "S06"
      }
    ]
  }
}
---

# S07 — Remove legacy implementation and reconcile every reference

## Scope and current state

Execute this stage only when the owner selects it or an explicitly selected full-pack run
covers it. Authoring the package did not authorize execution. Read the journal for actual
state/claims; never infer status from this prompt. Read the plan's D1–D7 and relevant
workstream/proof sections. Preserve the accepted PG-only direction and existing provider,
Redis, artifact, API/security and user-settings boundaries. Development market history may
be rebuilt; that is not authority for whole-database/volume deletion or publication.

## Context route

Always read `AGENTS.md`, this prompt, `implementation-plan.md` and `iteration-ledger.md`.
Repository-relative task entrypoints:
- `pyproject.toml`.
- `uv.lock`.
- `tools/ci/route_changes.py`.
- `.github/workflows/ci.yml`.
- `tools/docs/generate_project_map.py`.
- `tools/qa/postgres_storage_prototype.py`.
Read the predecessor report when declared; it is a deferred output, not an authoring gap.
Use project-map slices for affected callers, then inspect actual source. Refresh dirty-tree
ownership and the initial Appendix inventory; follow caller/config/schema references beyond
literal brand matches. Do not reread unrelated historical packs as current instructions.

## Work

Delete unreachable CH adapter packages, exporter, DDL manifests, dev XML and CH-only ops scripts; port meaningful tests/notebooks/QA/RL scripts. Remove clickhouse-connect through the project lock workflow; ensure no transitive/runtime need remains. Preserve measured prototype bytes under immutable non-shipped evidence then replace/remove its CH-dependent product tooling, without invalidating benchmark history. Remove stale comments/source labels from active ports/docs. Run a bounded brand/path/dialect/config/dependency scan and review all residuals individually. Current plan and immutable reports may mention CH historically; do not mass-rewrite old accepted ledgers or broad-allowlist active directories. Update doc navigation and generated files through their owners.

## Required acceptance

- CH adapters/DDL/client dependency and executable maintenance/benchmark paths are removed or ported; no product/test-tool runtime import remains.
- Every residual inventory match has a narrow documented historical or rejection-test justification; all active normative docs describe PG.
- Dependency lock, generated docs/map/schemas/topology and CI semantic coverage mapping are consistent.

## Verification and decision policy

Proof boundary: Repository-wide dependency/reference closure and static/build/import gates, not complete functional performance.
Use section6 and current CI/package routes; retain commands, failures, corrections and
raw sanitized evidence beneath `reports/S07/`. Report unperformed/unavailable checks
honestly; do not mark provider, browser, migration or performance checks passed from mocks.
Do not weaken acceptance silently, invent owner approval or touch unrelated changes.
Resolve routine reversible implementation choices. A real missing material decision or
hard boundary blocker must name the narrow affected work, evidence and resume condition;
continue independent authorized work rather than resetting previous accepted stages.

Skills: `backend-quality-gates` for Python verification; `contract-impact-analysis` when
changing relied-upon DTO/schema/config/backup boundaries; `backend-performance-evidence`
for timing/query/resource claims; `browser-qa-evidence` and the user's chosen browser route
when UI behavior changes; `architecture-review`/`production-risk-review` for the one
policy-required independent review. Load only crossed companions. No publication skill
unless the owner separately grants Git publication, which this prompt does not do.

## Claim, journal and handoff

Before side effects, validate current stage entry with the bundled validator and claim via
`python .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S07 --executor <unique-id>`.
The ID is executor-owned, not an owner approval token. Never steal a claim or edit active
JSON by hand. `status`, `resume`, `accept`, `needs-input` and `block` use that same copied
updater and canonical lifecycle; hard-blocked stages require a revisioned replacement,
not status reset. The updater is bound to this directory; do not use an old pack helper.

Write `reports/S07.md` with owned created/modified/deleted paths, foreign changes excluded,
actual verification results and file links, contracts, data/resource effects, remaining
risk and next-stage inputs. If all required checks pass, accept with `--checks-passed`
and actual `--evidence` files. The updater generates a new immutable receipt and validates
the adjacent prompt before allowing its claim. Do not forge a receipt before evidence.
At the last stage completion, all required obligations must be accepted before ledger
completion. Eligibility of the next row is distinct from continuation authorization.

Final report to the owner is Russian, concise and self-contained, with proof limits and
links. Stop after this stage unless a current explicit full-pack continuation covers the next.
