# S05 independent production-risk review

Reviewer: independent delegated review agent. Date: 2026-10-14.

## Scope and method

Read-only review of the S05 consumer migration, with focused follow-up reviews of
artifact source identities, publication/recovery fencing, legacy targets, memmap
consumers, bounded DNS lifecycle, and partition privileges. The reviewer read the
production-risk-review skill, S05 prompt, relevant plan requirements, PostgreSQL
READERS.md/INGESTION.md, and the affected implementation and tests. No recursive
review was requested. This record is the reviewer's only file mutation.

## Findings and resolutions

1. **P1: stale builds could restore ready after source reset/correction.** The
   original management completion and recovery paths fenced ownership but not
   generation/content/funding revisions. The implementation now holds a source
   revision guard across the final pointer switch and managed ready projection.
   Managed completion saves its receipt inside that guard. Publication and managed
   repositories share the same control gateway, so nested transactions use
   savepoints and retain the outer locks. Long-running computation precedes this
   transaction. The derivative publisher uses the same base publication path.
2. **P1: legacy source targets remained executable.** SourceTarget now rejects an
   incomplete revision tuple; dispatch explicitly rejects legacy queued targets.
   Additive migration 0039 invalidates legacy requests/source state and quarantines
   affected running builds. Publication journals carry source identity and recovery
   checks it when reconciling an active published build.
3. **P1: revision rejection could permanently quarantine an unpublished build.**
   The initial recovery guard required the obsolete revision even when the pointer
   had never changed. Recovery now permits cleanup of unpublished work under exact
   dead-owner fencing. Demonstrably prepublication source failures become
   ArtifactPublicationNotStarted instead of permanent ambiguous quarantine. Reviewed
   regression tests check candle correction, funding correction and generation
   change after computation, unchanged pointer, removed journal, and successful
   subsequent publication.
4. **P1: killing a WS writer could leave its DNS resolver running.** The resolver
   subprocess now watches its parent PID and absolute deadline. Reviewed tests cover
   resolver timeout and both terminate/kill of its parent. The resolver receives
   hostname/port rather than credentials; prepared conninfo preserves the hostname
   and TLS options alongside resolved addresses.
5. **P2: full-history hashing introduced avoidable large copies.** freeze_source
   now hashes final arrays in 4096-row slices. Its existing finite full-history
   compute allocation remains; this does not establish constant downstream RSS.
   PostgreSQL publication forces full rebuilds to avoid reusing a stale incremental
   prefix.
6. **Privilege-contract concern: application partition provisioning.** The proposed
   application EXECUTE grant for ensure_partitions was removed. Maintenance-only
   provisioning is retained. The inspected SQL function uses fixed search_path,
   quoted identifiers/literals and a bounded table/range scope; no arbitrary SQL
   privilege-escalation path was identified.

## Conclusion and evidence boundary

The findings above were closed by source inspection and review of the associated
regression tests. No remaining blocker was identified within these inspected
boundaries at the last follow-up review. This is not a blanket approval of every
dirty workspace change or a declaration that all S05 acceptance obligations passed.

The reviewer read `consumers-attempt8.log` (11 passed) and `dns-lifecycle.log`
(4 passed), but did not independently execute those tests. The executor subsequently
reported 15 passing real-PG consumer tests, 33 passing repair/alert unit tests, and
passing Ruff/Pyright checks. Those later results, and later batch-cap/reference
lookup chunking and alert/runbook changes, were not independently rerun or reviewed
by this agent.

The executor also reported browser evidence showing real PostgreSQL S05USDT data in
the accepted Backtests form, a ready workspace response, and a bounds request after
symbol selection (unavailable without an artifact). The reviewer did not perform
that browser session and does not claim independent UI verification.

Installation-wide physical connection accounting, simultaneous handshakes, all
control/recovery/lease sessions, and the S06 budget handoff remain outside this
focused review's proof. Tests against PostgreSQL max_connections=48 alone do not
establish that complete installation envelope. No local-installation cutover,
publication, production delivery or S09 workload acceptance is claimed.
