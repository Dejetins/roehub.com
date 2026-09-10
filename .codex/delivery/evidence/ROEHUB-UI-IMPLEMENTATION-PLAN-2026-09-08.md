# UI implementation plan and Backtests pack — authoring evidence

Date: 2026-09-08. Scope: local plan/prompt authoring only, under the user's accepted
six-stage decomposition. No implementation, dependency installation, stage activation,
Git publication or runtime delivery was performed. This is an authoring receipt,
not a `prompt-pack-receipt/v1` execution transition or another control-state journal.

## Delivered artifacts

- [General plan](../../../docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md):
  accepted decomposition, source/target boundaries, full retained UI backlog,
  dependency/compatibility/rollback/proof requirements and technical prerequisites.
- [Backtests ledger](../../agents/generated/roehub-backtests-client-v1/stage-ledger.md)
  and six exact stage prompts: foundation; shell/library; configure/submit/recovery;
  execution/cancel; results/export/saved strategy/delete; integrated acceptance.
- Canonical pointers in the Backtests ticket, functional contract and Web architecture;
  generated documentation index and project map updated.
- Coverage check: all 44 registry records and 18 journey IDs occur exactly once in
  the plan ownership tables; the 43 source surfaces remain covered through those records.

## Review

Cold-head review: completed.
Mode: independent subagent (`review_backtests_pack`), read-only; no nested reviewer.
Scope/source: the saved plan/pack/ledger and directly linked current ticket, functional
contract/registry, repo guidance, Prompt Manager artifact/lifecycle/receipt contracts,
and Architecture Review's `cold-head-plan-prompt-pack-review.md`.
Verdict: **Release for authored draft; Block for execution entry**.
New material findings: none. Open technical finding: **High — PACK-CLAIM**.
Local follow-up: verified final draft lifecycle, source/path mappings, preserved v23,
foreign-change hash and exact prompt/ledger contracts. No required fixes were found.

PACK-CLAIM is not missing product approval. Current repo guidance/tools and the
installed staged-plan-runner do not supply a verified supported exclusive atomic
ledger updater. Neither a filesystem read/write sequence nor the bundled validator
provides that capability. The draft correctly has no allowed rows/claims/current stage.
Resolve the updater and bind current evidence before enabling only S1 and rechecking
entry. Do not create an improvised lock or bypass the ledger to execute a prompt.

## Validation

All paths below are from the repository root, except the installed skill validator.

| Check | Result / boundary |
|---|---|
| `python3 /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py --root /Users/daniildegtyarev/Projects/roehub.com --ledger /Users/daniildegtyarev/Projects/roehub.com/.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md --check draft` | PASS, exit 0, JSON `status=pass`, structure/file binding only |
| Same validator with `--check entry --stage S1` | FAIL, exit 1: `Entry lacks a resolved claim capability; keep the draft non-runnable`. Expected named entry gap; not an execution pass |
| `python3 -B -m tools.docs.generate_docs_index --check` | PASS after generation |
| `python3 -B -m tools.docs.generate_project_map --check` | PASS after generation |
| `git diff --check` | PASS |
| Read-only Python mapping/link/hash check | PASS: 44 records, 43 surfaces, 18 journeys; S1–S6 chain; exact prompt/ledger contracts; existing context paths/Markdown links; draft/pending/unclaimed rows; pilot and foreign AGENTS unchanged |

Repository runtime/frontend/API tests were not run merely to validate prompt
construction; their future required commands and evidence are encoded in each stage.
Validator/review success does not prove client functionality or runtime readiness.

## Preservation and handoff

Existing foreign `.codex/AGENTS.md` changes were not modified; its task baseline SHA-256
is `40e787ff60e4eb02e61db193dca56a2faab89dbb8a2e0798771f5cb6b0087a41`.
The v23 HTML remains at its accepted path with SHA-256
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
No historical ledger, prompt or evidence was reset. Generated files are derived outputs.

Authoring state: `draft_valid`; S1 is the intended first stage, **not entry_ready**.
Execution remains a later request after PACK-CLAIM resolution. Target role/default
cutover also retains the existing Backtests authz dependencies; later product blocks
remain in the general plan. No new visual approval is needed for accepted v23.

## Saved artifact bindings

These hashes bind this authoring review/check receipt to the saved documents.
They are historical evidence, not a mutable status source or execution receipts.

| Artifact | SHA-256 |
|---|---|
| `docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md` | `03cb1a7cea12accd19eac2958938316573fb08b8b2c0e7df931043a30ca12821` |
| `.codex/agents/generated/roehub-backtests-client-v1/01-client-foundation.md` | `4cbc8bff01d900aecc363ad941e8bf3ec3245948d479c99fe0236428c3891397` |
| `.codex/agents/generated/roehub-backtests-client-v1/02-shell-and-library.md` | `2a6dbb36e986cb2b48bf3cb58c259bc945f7434b71ddc0d3e6b603e3916b99f2` |
| `.codex/agents/generated/roehub-backtests-client-v1/03-configure-and-submit.md` | `1aa5a6f0022e1ab9d990d72809a00f9a1d85d2d902eae6f887bd752a033c7893` |
| `.codex/agents/generated/roehub-backtests-client-v1/04-execution-and-cancel.md` | `f14879015c04fe93c197c4fd8987508b8bc40c7d5fe634c917ae9b43ac83951f` |
| `.codex/agents/generated/roehub-backtests-client-v1/05-results-and-strategy.md` | `4af0d21b9529347ff46301aa0e812f2fd5b3456f80573c84b33de416319fc818` |
| `.codex/agents/generated/roehub-backtests-client-v1/06-complete-journey-acceptance.md` | `b6d957fbaf30b288e72ef27de512fe41c0b6d31f1ca3b268e9531d14c5b523c2` |
| `.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md` | `9467a9505598910478289e5717253e0353411b7c3b5a5f77054a254019c34938` |
| `.codex/tickets/2026-09-08-roehub-backtests-client.md` | `4b6db6340a42fc38fe607b8bcc84aa540f3f0689785dd66d1275665eb46d160b` |
| `docs/architecture/apps/web/roehub-ui-functional-contract-v1.md` | `3dd63f9c2b2a7505a1c5c3229058ca1e8eff0b9fc9fac16307e9cea9843f454e` |
| `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md` | `66321ece0805c4b2c9837ae27d252b474d417736e168b357a6add48eebeec9ea` |
