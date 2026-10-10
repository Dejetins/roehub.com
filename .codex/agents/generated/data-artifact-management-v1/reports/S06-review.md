# S06 independent production-risk review

Reviewer: independent `s06_security_review` agent using `production-risk-review`.
Scope: S06 API, organization/installation authorization, typed control-agent transport,
storage service, migration 0031, and disposable test evidence. No product edits,
publication, runtime activation, or recursive review were performed.

## Current disposition

No unresolved material security/filesystem/API findings in the reviewed S06 code.
All nine findings, including the same-review follow-up, were corrected and re-reviewed. Stage acceptance still depends
on completion of required executor gates and immutable lifecycle evidence.

## Findings and corrections reviewed

- F1 (high): First configuration of a candidate root did not inspect the existing
  `backtest-primary` when settings were absent. Corrected: primary contents are
  checked and `relocation_required` preserves the mapping/data. Disposable test
  covers legacy primary contents with a clean candidate.
- F2 (high): Cleanup checked settings before durable writer reservation but not
  again in the destructive transaction. Corrected: settings version/root and
  preview root inode are rechecked immediately before ownership verification/IO.
- F3 (high): Manifest/current reads reopened ordinary paths after no-follow
  traversal. Corrected: bounded descriptor-relative, no-follow regular-file
  reads; slot identity and manifest hash are retained and rechecked.
- F4 (high): Base requests could expose foreign derivative union progress from
  shared build events. Corrected: base phase allowlist; shared derivative events
  collapse to a generic phase without timeframe or union counts. Own organization
  request/event lookup and coordinate checks remain server-side.
- F5 (medium): Control unavailability mapped to HTTP 422. Corrected: sanitized
  HTTP 503 with Retry-After, and known server storage failures are sanitized.
- F6 (medium): Storage inspection required a version that a freshly loaded UI
  could not discover. Corrected: GET returns current version without CAS;
  mutation CAS remains required.
- F7 (medium): Inaccessible catalog rows failed the whole inventory/history page.
  Corrected: inaccessible rows are omitted and pagination advances over the
  bounded scanned page. Consumers must follow next_cursor even for an empty page.
- F8 (high): Raw YAML pointer parsing accepted an unknown active_slot.
  Corrected using existing strict ArtifactCurrentPointerV2, real active manifest
  digest/generation validation, and positive integer inactive generation. Four
  malformed-pointer cases prove preservation for invalid slot, boolean generation,
  invalid digest and unsupported schema.

An early cross-installation concern was withdrawn after inspecting migration
0011: `identity_installations.singleton_key` has UNIQUE and CHECK constraints
enforcing one installation per database. It is not an unresolved finding.
Default profile already contains catalog_revision/catalog_indicators; the initial
catalog completeness question does not require a separate catalog endpoint.

## Verification reviewed

- Reviewer independently ran `.venv/bin/pytest -q
  tests/unit/apps/api/test_artifact_storage_routes.py
  tests/unit/apps/control_agent/test_auth.py`: **4 passed**.
- Executor `reports/S06-real.json`: exit 0, **55 passed in 17.62s**, covering
  test_artifact_http_storage.py, test_artifact_management.py and
  test_artifact_dispatch.py. Inspected tests use disposable PostgreSQL schemas,
  real filesystem writes/cleanup, and actual HTTP admin -> API client -> signed
  Unix -> storage -> PostgreSQL/filesystem composition. HTTP authentication uses
  the real OrganizationAccessService over an in-memory identity repository;
  privileged service independently rechecks installation ownership in real PG.
- Negative evidence includes organization roles versus installation owner,
  origin/recent auth, closed/oversized DTO, CAS/idempotency/bulk replay,
  foreign-org reads, symlink/traversal, changed current/new cross-org pins,
  first-root/relocation preservation, real chmod denial, injected ENOSPC/EACCES,
  durable quarantine/unknown replay, and four malformed-pointer cases. Injected
  ENOSPC does not prove a physically full disk.
- Executor `reports/S06-types.json`: exit 0, **0 errors, 0 warnings**.
- At follow-up time `reports/S06-lint.json` records two test-format findings;
  executor is resolving remaining gates. This review does not relabel that
  result as passed or establish stage acceptance.
- No unrelated scope changes identified in reviewed S06 additions. Additive API,
  typed control capability and storage persistence are **compatible-change**
  within existing supported authorization/ownership contracts.

## Handoff details

Production wiring registers only configured backtest-primary. Arbitrary root
paths, worker owner tokens and receipts are rejected. Shared derivative progress
intentionally has unknown counts and no foreign-only timeframe. Cleanup is one
explicit inactive slot per preview using existing durable writer ownership and
reader protection. First setup preserves capability_enabled=false, paused=true,
legacy_handoff=pending; configuration does not activate workers or perform owner
handoff. Preserve reader-first migration and quarantine/reconciliation rules.

## Proof limits

No production database, production owner identity, real schedule activation,
S3, production worker mount, browser UI, or deployment was exercised. Destructive
checks operate only on disposable roots. Independent review of application
composition is code evidence; only the recorded exercised boundaries constitute
runtime evidence. Failed cleanup retains durable intent/quarantine and unknown
replay; no automatic reclaim or retrying deletion is authorized.

## Same-review follow-up: policy-only commands and request pagination

Reviewed the executor's policy-only persistence addition. Internal
PolicyItem.admit_request defaults true, is excluded from historical hashes when
true, and is not accepted by public DTOs. update_now explicitly rejects false.
Policy-only writes retain organization authorization, coordinate checks, immutable
profile validation, CAS, admission lock and the atomic command response. They do
not create a base/derivative child request; the explicit policy_updated outcome
with nullable request_id/state must be consumed by S07. Existing internal
callers retain default admission behavior and historical digest/replay semantics.
No material security issue found in this addition.

F9 (medium, corrected): request pagination advanced using a filtered row, then
rechecked its coordinate permission during continuation and returned 403. The
reviewed repair selects only created_at by organization_id/request_id inside the
authorized transaction. It never serializes the hidden request, retains
foreign-organization cursor 404, and orders consistently by created_at/request_id.
The new hidden-last-row regression exercises empty first page and successful
continuation to an authorized coordinate. No unresolved follow-up code finding.

Follow-up tests added: policy disable/schedule edit creates no base refresh,
historical command hashes replay, and pagination continues past a revoked
coordinate. At follow-up review, executor is rerunning final gates; current real
evidence records 60 passed from the preceding run. The newest test additions
must be included in the final executor evidence before stage acceptance.
