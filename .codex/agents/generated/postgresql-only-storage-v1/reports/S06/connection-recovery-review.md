# S06 connection lifecycle and security review

Independent reviewer: `/root/security_review`, focused read-only review. Final
result: no new blocking security findings in the reviewed boundaries. The reviewer
read the final cancellation/native/maintenance logs and six RUN JSON results;
they did not independently rerun the full runtime matrix. No recursive review.

## Implemented boundary

External PostgreSQL retains host/port/database/credentials_ref/tls. No server-side
shared filesystem, server.lock, custom server token or connection_authority input
is required. All installation clients share one protected POSIX registry; admission
is 16 market / 24 control / 8 admin, before the first socket, including preauth.

Direct PG16 transport completion requires raw EOF plus Linux TCP_CLOSE_WAIT before
local shutdown. Consumed RST, timeout and absent pg_stat_activity rows never free
slots. Idle TLS sessions use bounded rollback/self-termination before PQfinish;
active/COPY/pipeline paths fail closed. Native utilities use one counted upstream,
verified TLS to the original hostname and a trusted local plaintext loopback.

Resolved endpoint is saved with the initial reservation. Enrollment pins a root-owned
IP:port mapping, SQL system/epoch identity, Docker daemon/container/data identity and
the actual backend-owned TCP socket. Recovery locks all 48 slots, rejects foreign or
unverifiable records, persists stopping intent, disables restart and proves the
selected server stopped before clearing existing inodes. Resume uses a counted admin
connection and rotates epoch. Ambiguous stop response permits an idempotent retry.

## Findings corrected

- Reclamation based on statistics absence and raw EOF alone was rejected.
- Descriptor/connection configuration ownership and concurrent finalization fixed.
- Dedicated PID namespace, root binding, protected enrollment, durable stopping and
  resuming states, correct pidfile readiness and bounded administrative calls added.
- Native option denylist replaced with a closed grammar: GNU abbreviations and short
  option clusters cannot replace the counted endpoint or enable parallel sessions.
- Exact remote pgpass host/port lookup occurs before selecting the loopback port.
- Native TLS no longer exhausts slots through normal close_notify/RST interaction.
- Every socket attempt records its destination; known pre-socket cancellation frees
  only unstarted reservations. Thread-owned escrow survives repeated async cancellation.
- Monitor receives only schema USAGE needed for metadata lookup; application DDL
  privileges are not expanded. Maintenance rotates bounded request batches.

## Evidence

- `budget-runtime8.log`: 48 established physical sessions, 16/24/8 split, bounded
  overflow and admin work at saturation. Transport algorithm then gained the
  separately tested TLS and cancellation fixes below.
- `transport-eof3.log`, `transport-negative1.log`: FIN/RST distinction and concurrent
  finalization, ordinary sync/async/startup-failure lifecycles.
- `operator-recovery3.log`: 96 killed preauth owners, two full resets, live-owner
  rejection, lost stop response/retry, 120 subsequent closes, no server registry mount.
- `native-tls3.log`, `native-cli-runtime1.log`: actual PG16 dump/restore/psql through
  the guard with verify-full, exact passfile lookup and 14 CLI override rejections.
- `tls-endpoint-negative1.log`: wrong endpoint rejected before socket without slot
  loss; 30 sync and 30 async TLS closes; wrong certificate hostname rejected.
- `async-cancellation2.log`: 12 prepare/opening single/repeated-cancellation cases,
  no leaked or quarantined reservations.
- `maintenance-runtime5.log`: real maintenance-role DDL, 64-request rotation,
  12-month bound, six partition families, operational metrics.

Earlier failed experiments remain evidence of failures, not acceptance claims.

## Limits

Linux; direct PostgreSQL endpoint with stable ownership of the enrolled IP:port;
shared protected client registry; trusted operator host; exclusive maintenance
control. Transparent failover, same-address reassignment and terminating proxies
are not proven. The supplied recovery adapter controls a dedicated remote Docker
PostgreSQL over an administrative channel; other control planes need an equivalent
verified adapter. Malformed/foreign records are not silently reclaimed. Crashed or
unknown transport remains quarantined until verified operator recovery. Restart
policy remains no until the operator restores the intended policy after resume.

These limits and concrete commands are documented in
`docs/runbooks/storage-bootstrap-and-migrations.md`. This review is evidence for
S06, not a release, deployment or current-installation readiness assertion.
