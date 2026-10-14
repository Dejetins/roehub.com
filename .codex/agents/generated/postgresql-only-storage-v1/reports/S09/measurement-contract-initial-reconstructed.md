<!-- Reconstructed verbatim initial wording from session tool output; not an original contemporaneous file hash. Original failed pilot samples and source snapshots are retained. -->
# S09 measurement contract — frozen before pilots and timed regimes

Scope: production PG16 adapters on a disposable target; no installation/live-provider claim.
Source: immutable monthly Parquet files in `.local_artifacts/pg-comparison-20261013/candles`, verified against snapshot.json SHA256, 15,130,115 canonical rows / 102 instruments. Every source row is loaded through PostgresRawKlineWriter using roehub_app. Input byte hashes, full independent normalized 17-field row comparisons/output hashes and full artifact-array hashes are required before long runs. Raw is the production adapter's typed observation representation, not a provider payload archive.

Baseline: PostgreSQL 2 CPU / 4 GiB, shared_buffers256MiB/work_mem4MiB, max_connections48, fsync/synchronous_commit/full_page_writes on. Separate Linux client 4CPU/6GiB; actual CPU/RSS measured separately from server. Protected shared client registry 16market/24control/8admin; server has no registry mount. API and every application lane use production guarded gateways. No shared installed runtime writes. Capture Docker image IDs, mount/resource configuration, Linux/cgroup/hardware, source hashes and dirty revision.

Mix, fixed offered cadence (open-loop due times; retain queue delay, all failed and late samples):
- Stream: 5 rows each second normal; 100 rows each second stress.
- History: 1000 rows every2s in both modes.
- Latest100: every1s normal, every0.5s stress.
- 102+ instrument 7-day catalog and bounded7-day gap: every2s.
- One-year artifact input and full-largest-instrument array export: each every10s, two independent bulk slots, full output byte hashes every observation.
- Real authenticated existing `/market-data/markets` HTTP route: every1s, production persisted sessions/users and reference adapter on the same PG.
- Existing auth rate-limit write/read/clear: every2s.
- Existing work request submit/claim/checkpoint/finish/read under coordinator ownership: every5s.
- Existing async recovery journal remember/pending/ack plus connection publication: every2s.

Durations: pilot sanity first; >=3 normal and >=3 stress runs, each >=300s wall time, alternating normal1/stress1/normal2/stress2/normal3/stress3. >=30 observations of each large-read lane per regime. Then >=3600s stress soak with the same concurrent ingestion/query/recovery/artifact/control mix. Never stop/restart because a tool observation call times out. Record live session handles and progressive samples.

Timing: adapter operation includes fixture preparation and validation overhead (conservative); report array hash cost separately. HTTP is separate and never replaces adapter latency. Report due/start/end and queue wait; gateway pool.get wait measured independently. Preserve failures, throughput offered/completed/receipted rows and all samples, including first-use cold sessions. No OS cache flush: dataset import and parity verification warm filesystem caches. Initial new session observations are session-cold, not cold disk. Same schedule each run; largest immutable instrument and year remain fixed. Each run writes a fresh deterministic S09 symbol namespace; prior successful receipts are retained. Thus incremental raw/WAL/control growth is explicit, not an identical reset-state claim. Compare regimes with their measured starting relation sizes and ordering; no causal tuning win without a matched pre/post baseline.

Guardrails unchanged: latest/catalog/gap p95<=100ms; stream p95<=100ms; 1000-history batch p95<=250ms; year artifact input median<=2s; full-instrument p95<=10s. Each run and pooled regime reported. No unexpected errors/OOM/lost successful receipts/divergent results/unbounded queues or unexplained growth. Completed rows must equal offered rows; observe cadence lag separately, never treat drain as on-time throughput. Explicit fault/cancellation/saturation samples have their own labelled expected-failure phase and do not hide timed-load errors.

Resource proof: periodic pg_stat_activity/waits, checkpoint/WAL/autovacuum/dead tuples, relation heap/index/raw/total and WAL directory; server cgroup CPU/memory anon/file/peak plus process RSS, client CPU/RSS/cgroup separately. No raw retention deletion introduced; growth attributed to retained successful raw observations, metadata and WAL recycling. Capture plans with ANALYZE/BUFFERS after representative ANALYZE. Explicit guarded saturation with real admin DML, cancellation and recoverable fault checks preserve physical limits and unknown-record quarantine.

Acceptance requires complete proof and independent focused review. Setup, partial import, pilot or S08 outcomes cannot satisfy S09. If measurements identify defects, profile and narrowly repair, preserve failed results and rerun honest comparable regimes. Resource changes require explicit new envelope plus retained baseline outcomes.

Pre-run fidelity amendment (no timed pilot/regime has run): add active stream-symbol latest100 every1s, checking every returned field against the deterministic offered fixture while the same symbol is concurrently written. Seed exactly one known row before the timing horizon; report that one baseline raw observation separately. Main immutable latest/catalog/gap outputs receive full independently-derived expected hashes. All lanes are mandatory, lifecycle exceptions are retained and missing samples fail validation. End timestamps and completions inside the offered horizon are separate from eventual drain. Job mutations invoke the actual coordinator fence.
