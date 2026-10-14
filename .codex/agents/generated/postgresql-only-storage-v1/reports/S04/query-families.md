# D5 query family mapping

Every captured SELECT has exact SQL, parameters and EXPLAIN ANALYZE BUFFERS JSON in
`query-plans.json`; its zero-based plan index and SQL hash join `query-catalogue.csv`.
Rows/buffers are observed values, not optimizer estimates. Each captured operation
ran its result assertion before plans were collected. Plans are warm/reused-fixture
observations, not cold-cache or mixed-workload comparisons.

| Family / port | Predicate/order and cardinality | Expected access / observed proof |
|---|---|---|
| Q02 canonical DTO/arrays | One market/symbol; half-open two-year UTC range, ts_open ascending;1,052,640 output rows | Ordered PK scans on24 monthly partitions, binary COPY/server cursor; full timestamp/OHLCV/meta assertions |
| Q03 bounds/before/tail | One instrument and exclusive before; bounds2 values, tail100 rows | Summary + clipped actual masks; tail ordered PK LIMIT100 within occupied timestamp range |
| Q04 daily/distinct | One instrument, bounded days;731 counts or explicit range timestamps | Coverage-day PK and clipped bit masks, no canonical fallback |
| Q05 catalog coverage/latest | Batched market/symbol keys and UTC window | Key enumeration then one batched count query; compact day rows; latest delegates verified index |
| Q06 instrument coverage | One instrument, explicit UTC range | Generation/summary plus batched day count; actual percentage, explicit unknown state |
| Q07 history actual/absence/gap | Batched keys or one range; dense/sparse2year histories | Policy-tagged masks, actual precedence, frontier then day-level series731 rows; first gap LIMIT1 |
| Q08 verifier | One aligned seven-day range, current generation/lease | Generation/lease locking and at most10080 canonical rows; day-mask updates; contiguous bounded prefix |
| Q09 stored observations | Keyset coordinate page LIMIT50; exact before clipped inside minute | Compact summary + partial-day masks; bounded page; exact count/first/last against independent fixture |
| Q10 market references | Explicit market IDs | Small reference relation; set parity |
| Q11 instrument references | One market and1000 explicit symbols | Composite key/index lookup or justified compact reference scan; full fields/key parity |
| Q12 enabled/search/readiness | Enabled markets/instruments; escaped prefix and LIMIT100; BTC key | Reference tables/indexes, sorted prefix output;10,002 enabled rows,4 readiness states |
| Q13 funding universe/get | Explicit markets or exact instrument | Composite primary key/reference transport; no arbitrary universe output cap; real9000-row test |
| Q15 funding latest/range | Exact instrument, half-open2year range;17,544 hourly events | Partition/PK ordered access; complete field/interval parity; per-event interval coverage |
| Q16 explicit rebuild | Expected generation; at most7day checkpoint step | Bounded canonical range; persistent progress; writer exclusion until ready |

Q01 candle writes and Q14 funding writes remain implemented/proven in S03; Q08/Q16
mutation correctness is covered by real transactional tests, not merely by SELECT
plans. Q09 independent baseline canonical aggregation occurs only in the evidence
harness, never the product observer. Source/predicate detail for every plan remains
in the raw JSON so the CSV cannot hide a full scan behind an index label.

Probe workload/deadline mapping: bulk30s (Q02/Q15-range), reconcile10s
(Q07/Q08/Q09/Q16), reference5s(Q10/Q11/Q13), interactive1s(other queries),
all lock_timeout250ms. Plan replay itself uses30s to capture evidence; runtime
operations already executed through the shorter applicable gateway deadlines.
