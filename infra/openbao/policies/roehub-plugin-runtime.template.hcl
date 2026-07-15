# Render only after validating both identifiers against ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$.
path "kv/data/roehub/plugins/${organization_id}/${instance_id}" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
