path "kv/data/roehub/*" {
  capabilities = ["create", "update", "read", "delete"]
}

path "kv/metadata/roehub/*" {
  capabilities = ["read", "list"]
}

path "kv/delete/roehub/*" {
  capabilities = ["update"]
}

path "kv/undelete/roehub/*" {
  capabilities = ["update"]
}

# Destruction is intentionally absent: rollback remains possible inside retention.

path "auth/token/renew-self" {
  capabilities = ["update"]
}
