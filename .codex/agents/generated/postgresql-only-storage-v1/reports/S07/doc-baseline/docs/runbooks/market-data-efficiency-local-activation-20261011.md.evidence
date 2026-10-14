# Market Data efficiency: local activation, 2026-10-11

## Scope and outcome

The owner authorized local S01–S04 activation with data/settings preserved and a
local Git commit only. Publication, remote deployment and network changes were
not requested. The owner additionally approved including necessary pre-existing
source dependencies after review exposed an uncommitted implementation baseline.

All eight `roehub-local` services run and pass their configured health checks.
The six application services use candidate image
`sha256:00f854bea20f9a4e0e57895f0f1f9672a2ae3ce606ffa4bca0050f79d9b59f05`.
**End-to-end live collection is not verified:** Binance historical requests
returned HTTP 451 during restart recovery. The collector preserved failed repair
receipts and the UI shows an error; container health is not domain readiness.
Do not interpret the activation as a successful four-stream live acceptance.

## Preserved installation and rollout

Evidence is owner-local under
`.local_artifacts/platform-local/activation-20261011/`; private backups are under
`.local_artifacts/platform-local/backups/efficiency-20261011/` and must not be
committed. PostgreSQL, artifact files and installation/private configuration were
backed up. Existing ClickHouse/PostgreSQL volumes and resource ceilings remain.
No candle tables, jobs, leases, quota records or failure receipts were reset.

The old image did not include earlier in-container repairs. A read-only inventory
preserved 92 runtime files from the existing API/web/worker containers, with no
conflicting versions or deleted runtime files. A proper baseline image was built
from the old image plus those existing files; then the selected stage source was
layered into the candidate. No source files were copied into running containers.
Baseline rollback image:
`sha256:ba3de94bd5501e40084badd6b5248fcf9a414592a95bb8047fe76e9538dd9e52`
(`roehub/local-runtime:pre-efficiency-20261011`).

The signed owner CLI gracefully stopped the participants. ClickHouse moved through
`provisioned` to `active`; the controller was regenerated/recreated with each image
binding. A stop attempted after a local tag changed was rejected with
`control_agent.image_digest_mismatch`; its failed operation was inspected, the
binding corrected, and a separate stop then succeeded. The final managed start
succeeded at journal sequence 243. No unknown-outcome command was blindly replayed.

Only PostgreSQL phase `market-data-provider-admission-0034` was applied under the
normal advisory lock through its migration helper/checksum marker. This adopted
populated installation cannot replay the greenfield bootstrap. All 54 prior
markers were retained and the new marker raised the count to 55. The installed
market-data YAML changed only four limiter modes to `shared_conservative` and
added common scope `local-installation-egress`; all other parsed values match the
backup. No proxy variables were present in the participating containers.

Preservation checks:

- All 102 stored coordinates remain, with no reduced indexed minute coverage.
- All 129 existing jobs remain, with no reduced completed-unit counters.
- Exact row-content hashes match for instrument selections, artifact installation
  settings, policies, profiles and legacy demands.
- The same four selections remain: spot `0GUSDT`, `AAVEUSDT`, `ACTUSDT`, and futures
  `0GUSDT`. Existing ready artifact files remain readable in the UI.

## Activation defects fixed

1. The official ClickHouse entrypoint removes `default` for `CLICKHOUSE_USER=proof`.
   The S02 networks-only default fragment recreated a user without authentication,
   producing code 347 and preventing startup. Explicit `--default-user-disabled`
   omits that fragment only for this verified entrypoint mode. The prior XML was
   checked and privately backed up; all four generated passwords and the existing
   `proof` identity were retained. Actual server users are `proof` and the four
   classes; remote `default` returns 401. Private server XML is mode 0600 and owned
   by `clickhouse`. This is a `compatible-change` for the verified installation;
   the default provisioner behavior remains unchanged.
2. The managed publisher reused the API reference factory's `interactive` class
   while mounting only its `artifacts` credential. A trusted composition keyword
   now selects `artifacts` there; API callers retain `interactive`. No credential
   fallback or cross-class secret mount was added. This is a `compatible-change`
   to internal composition; API/DTO and persistence shapes are unchanged (`none`).

Both regressions were observed before repair. Focused tests: 10 installation tests
and 11 API/publisher tests passed, with scoped Ruff passing. Independent read-only
review found no blocker for the disabled-default fix under its explicit preflight;
the publisher composition fix also passed independent review. The review covered
activation ordering, security and required baseline dependencies without recursive
review. An isolated candidate tree initially passed 129 unit tests with one
missing-baseline-config failure; adding the existing observability configuration
resolved it (all three affected tests passed). Thus all 130 selected unit cases
passed across the focused runs. Runtime module imports passed in that isolated
tree; the staged source is checked against those tested bytes. No benchmarks or
full real-infrastructure suites were repeated for these composition-only fixes.

## Real-runtime observations and limits

- **S01:** the actual publisher completed three indexed queries returning 50, 2
  and 50 rows (102 coordinates), each reading only
  `market_data.canonical_candles_1m_stats`, with observed durations 9, 3 and 35 ms.
  This establishes index-path use, not a new performance benchmark.
- **S02:** actual class settings match the four budgets; all four identities
  appear in the query log. Stream writes reached raw/canonical storage. A single
  class admission rejection (202) was observed during startup; limits were not
  increased. Credentials were readable by runtime UID 501 without broadening modes.
- **S03:** actual REST calls charged shared PostgreSQL scope/windows; completed
  calls released permits. A narrow kline read returned 200, while actual historical
  repair calls reproduced `source_http_451`. Failed repairs were not mass rearmed,
  deleted or masked. No alternate endpoint, proxy or routing bypass was attempted.
- **S04:** the new async collector is deployed and starts successfully, but the
  historical-readiness gate blocks WebSocket admission after failed repairs.
  New closed canonical minutes across all four streams, async WS receipt batching
  and sustained backlog drain therefore remain unproven in this installation.
  Source-stage real-infrastructure tests are separate evidence, not substituted
  for this missing live acceptance.

Browser smoke on `http://localhost:20120`: the candle catalog loads and reports
stream errors; the artifact catalog loads existing ready entries. No warning/error
console messages were observed during those reads. No settings, selections or
requests were submitted through the browser. Detailed browser network capture was
not performed; server query logs provide the storage-side evidence.

## Recovery and rollback

The next live acceptance requires restored permitted Binance access. Then diagnose
and explicitly resume the retained repair receipts, preserving their identity and
history, and observe at least two newly closed canonical minutes for every selected
stream with empty/draining recovery and no stranded provider permits. Do not clear
failures automatically or claim that healthy containers prove fresh data.

If rollback is requested/required: stop and drain all participating new clients,
restore the baseline image **and** the backed-up market-data YAML, render S02
`provisioned`, refresh the controller binding, and restart through the owner CLI.
Retain additive users, passwords, migration, quota spend, cooldowns and all data.
Runtime rollback was prepared but not exercised. No long-duration capacity or
uninterrupted exchange availability claim is made.

Local commit scope: 143 explicitly selected files, including the owner-approved
implementation baseline and dependency closure. Frontend changes, unrelated
identity implementation, private installation files and generated stage evidence
are excluded. Migration 0032 is included solely because the selected migration
manifest/loader requires that SQL file. Prior unrelated working-tree changes
remain unstaged. No push or pull request was performed.
