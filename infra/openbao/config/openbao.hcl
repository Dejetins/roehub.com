ui = false

api_addr     = "http://openbao:8200"
cluster_addr = "http://openbao:8201"

storage "raft" {
  path    = "/openbao/file"
  node_id = "roehub-openbao-1"
}

listener "tcp" {
  address         = "0.0.0.0:8200"
  cluster_address = "0.0.0.0:8201"
  tls_disable     = true

  telemetry {
    unauthenticated_metrics_access = true
  }
}

telemetry {
  prometheus_retention_time = "24h"
  disable_hostname          = true
}

audit "file" "roehub" {
  description = "Roehub secret-safe audit log"

  options {
    file_path = "/openbao/logs/audit.log"
    log_raw   = "false"
    mode      = "0600"
  }
}
