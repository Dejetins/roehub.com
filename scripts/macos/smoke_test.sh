#!/usr/bin/env bash
set -Eeuo pipefail

pg_isready -h 127.0.0.1 -p 15433
redis-cli -h 127.0.0.1 -p 16379 PING
curl -I http://127.0.0.1:18124/ping
curl -i http://127.0.0.1:18000/auth/current-user
curl -I http://127.0.0.1:13000
curl -I http://127.0.0.1:19090
curl -fsS http://127.0.0.1:19201/metrics >/tmp/roehub-test-metrics-19201.txt
curl -fsS http://127.0.0.1:19202/metrics >/tmp/roehub-test-metrics-19202.txt
