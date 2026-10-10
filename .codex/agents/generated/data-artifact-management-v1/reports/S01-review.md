# S01 independent architecture review

Date: 2026-10-08. Reviewer: independent subagent `s01_architecture_review`.
Skill: `architecture-review`. Scope: S01 report D1–D7 and R01–R13 against the selected stage/plan and narrow current sources. No product, canonical documentation, plan, prompt or ledger mutation; no recursive review.

## Verdict

**Ready within the S01 proof boundary.** One Medium contract finding and one Low wording finding were returned to the executor, corrected in the unaccepted report, and re-read. No unresolved S01 acceptance blocker remains. This verdict supports contract-stage acceptance only; it does not establish implementation, deployment, capacity, numeric equivalence or runtime security.

## Findings and closure

### F1 — Medium — bulk command identity conflicted with request uniqueness (closed)

Observed initial D2: each request carried one coordinate/kind and unique `(organization_id,idempotency_key)`. D6 allowed one atomic 1–50-coordinate command and `/commands/{idempotency_key}` recovery. Inference: multiple child requests could not share the declared command identity; original whole-batch response recovery lacked an explicit persistence owner. Affected boundary: persistence/command idempotency, R08/R11.

Smallest requested correction: separate command envelope with unique organization/key, canonical body hash and original complete response; child request FK and item-level uniqueness; distinct system/scheduled dedupe identity. Verification: re-read corrected S01 D2/D6. The report now declares `backtest_artifact_commands`, unique `(organization_id,idempotency_key)`, original bounded response/status, child `command_id`, unique `(command_id,exchange,market_type,symbol,kind)`, organization binding and one admission transaction encompassing response. Correction closes the contract inconsistency. S02/S06 still require PostgreSQL concurrent duplicate/bulk rollback and lost-response tests; prose is not those tests.

### F2 — Low — existing control seam described as existing storage capability (closed)

Observed initial D6 wording called the storage capability existing, although baseline correctly said it was absent. `OperationAction` in `src/trading/contexts/operations/contracts.py` contains no artifact-storage action; `apps/control_agent/server.py` validates the closed request and authorizes service identity before dispatch. Inference: ambiguous wording could encourage bypassing or assuming a nonexistent action. Affected boundary: privileged integration contract.

The corrected report now explicitly extends the existing typed control-agent **boundary with a new storage capability**. This is accurate. S06 owns the implementation and negative authorization/path tests.

## Assessment and source-backed observations

| Area | Assessment | Evidence / implication |
|---|---|---|
| D1/D4 catalog, profiles and compatibility | OK for contract | `contracts.py` timeframe constants and root-manifest schema checks agree with 11 prices / 10 mappings and optional schema-2 risk semantics. Frozen catalog evidence records 40 IDs/400 targets. Profiles preserve selected IDs, finite grids and semantic recipe hashes; no duplicate runtime registry is introduced. |
| D2 ownership and shared demand | OK for contract | `migrations/postgres/0028_backtest_input_recipe_v1.sql` keys slot ownership by coordinate/slot, excludes org/generation from mutex, and separates organization readers from internal publisher readers. Report preserves this and separates tenant requests from physical union builds. Incompatible identities serialize; correction/prefix proof controls reuse. |
| D3 short history/source | OK with explicit S03 proof gate | Notification is a hint across ClickHouse/PostgreSQL; bounded reconciliation repairs loss. Long-timeframe zero bars are insufficient, not ready. Subset publication is conditional on reader tests; preserving the old pointer is allowed until proven. This is accurately assigned to S03 in the approved plan, not a missing S01 owner decision. |
| D5 schedules/resource limits | OK for contract | Scheduler wiring forwards enabled and uses actual 03:05 Europe/Moscow defaults. Report defines calendar 1/3/7-day anchors, DST, manual-run independence, bounded admission and fairness. Limits are proposals/estimates, not capacity claims; native ceilings and all-history intent remain explicit. |
| D6 installation authorization | OK for contract | `organizations.py::require_installation_control` independently checks installation ownership; organization `operations.execute` alone cannot replace it. Typed control-agent boundary exists, storage commands do not. Report requires both checks and recent auth for mutations. |
| D6 cleanup/storage | OK for contract | First versioned-input/ownership section of `backtest-service-artifact-runtime-v1.ru.md` requires slot_a then slot_b, writer reservations, queued/active/quarantined pins and exact liveness recovery. Report retains those rules, disallows TTL reclaim, revalidates preview and root identity, and excludes relocation. Actual filesystem race/crash behavior remains unproven. |
| D7 / R01–R13 traceability | OK for contract | All 13 requirements map to owners, contract sections, baseline test anchors and explicit future proof. `builder-api.ts` confirms missing start bound/input_readiness client exposure. Browser observation is clearly a baseline screenshot rather than flow or runtime acceptance. |
| Activation/rollback | OK for contract | Additive schema/readers precede writers; legacy demand preserves implicit precompute without assigning it to the first tenant; scheduler ownership handoff and compatible rollback are explicit. Installed/external reader inventory remains unknown and is not falsely inferred from source inventory. |

Existing components to reuse: native builder/validator and signal catalogs; slot reservations/pins and immutable recipes; identity installation-owner gate; authenticated typed control-agent; current coherent UI loading contract. None warrants a replacement broker, financial identity, private filesystem API or new lock protocol.

## Independent checks and limits

- Read selected S01 prompt and relevant approved-plan D1–D7/R01–R13, repository AGENTS, architecture entrypoints and report/evidence.
- Compared narrow actual sources: operation contracts/server, identity organization use cases, scheduler wiring, Navigator builder schemas, artifact constants/schema validation, existing ownership migration and normative runtime ownership documentation.
- Independently recomputed SHA-256 for all 269 paths in `S01-baseline.json` using `python3`; result: `changed_since_baseline: []`. Initial `python` command was unavailable (exit 127); rerun with `python3` succeeded. No inference of missing evidence from that command failure.
- No product tests, database migration, browser flow, storage probe, provider request, build, benchmark or expensive computation performed by this reviewer. Catalog parity/runtime observations are executor evidence, with narrow source consistency inspection here; the reviewer did not independently rerun every catalog import or browser observation.
- Current S01 product compatibility remains `none`. Proposed management APIs/schema are conditional `compatible-change`; schema-1-only readers against schema-2 are `breaking-change`; missing-timeframe consumers and actual installed versions remain `unknown` until their assigned proofs.

Future gates remain mandatory: real PostgreSQL command/claim contention; exact all-timeframe and short-history fixtures; scheduled/on-demand numerical equivalence; storage negative permissions/symlink/TOCTOU/worker-mount failures; cross-org pin/publication/cleanup crash drills; browser regressions. These are assigned S02–S09 implementation proofs, not grounds to claim them passed or to block a complete read-only S01 contract stage.

No canonical documentation was changed: proposed names remain in the stage handoff until implementation updates the affected runtime/market-data contracts. Reviewer-owned change is this new report only. Ledger acceptance and receipt binding remain the executor's responsibility.
