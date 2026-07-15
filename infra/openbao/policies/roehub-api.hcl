# API can inspect health and secret metadata only. Value and Transit access are denied.
path "sys/health" {
  capabilities = ["read"]
}

path "kv/metadata/roehub/*" {
  capabilities = ["read"]
}

path "kv/data/roehub/*" {
  capabilities = ["deny"]
}

path "transit/decrypt/*" {
  capabilities = ["deny"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
