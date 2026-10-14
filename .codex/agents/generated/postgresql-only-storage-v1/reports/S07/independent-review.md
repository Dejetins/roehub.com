# S07 independent review

Reviewer: `/root/s07_review`, one independent production-risk review with followups;
no recursive review. Final review inspected the owned changes and concrete logs.

Closed findings: provider raw fields/null normalization and DTO array assertions;
CI SCRAM role auth and real-infra route; stale docs SQL and actual/calendarExpected
coverage denominator; immutable receipt-bound S01 contract; real PG composition,
exact-source request grouping and preclaim storage admission in dispatch fixtures.
`dispatch-fixes2.log` reports 3 corrected cases passed.

No remaining material product defects found in the reviewed S07 deltas. This is
not a complete functional/performance or production readiness verdict.

P1 structural acceptance blocker: S08 directory entry_inputs fail validator
is_file, and canonical updater preflights S08 before S07 acceptance. The
orchestration owner is handling a narrow metadata migration under exclusive
lock/CAS, preserving statuses, claim and accepted receipt bindings. S07 executor
must not bypass validation or alter ledger/updater while repair runs.

P2 open migration diagnostic: handshake test expects connect_timeout but actual
code is postgres_retry_exhausted. Gateway matches the S07 baseline. Independent
blackhole reproducer returned that code after 0.205 s at a 0.2 s deadline without
surviving blackhole thread; this supports bounded completion only. The original
test remains failed, with classification/contract resolution due in S08 before
functional recovery acceptance. See timeout-attribution.json and semantic-suite.log.

The final dispatch-file run is separately recorded in dispatch-final.log. Prior
123 passed cases remain current evidence because their code and fixture did not
change after that run; corrected test definitions are covered by the final file.

Closure: the orchestration owner repaired pending entry locators. S07 focused
sanity review (transition-preflight.log) verified exact unchanged bodies/scope,
six accepted receipts, updater bytes and successful projected S08 entry. Structural
P1 is closed; P2 remains explicitly open. Final dispatch file: 21 passed.
