ui = false
disable_mlock = true

api_addr = "http://127.0.0.1:8200"
cluster_addr = "http://127.0.0.1:8201"

storage "file" {
  path = "/opt/roehub/state/openbao/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = true

  telemetry {
    unauthenticated_metrics_access = true
  }
}

telemetry {
  prometheus_retention_time = "24h"
  disable_hostname = true
}
