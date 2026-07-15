path "kv/data/roehub/telegram/providers/*" {
  capabilities = ["read"]
}

path "kv/data/roehub/telegram/recipients/*" {
  capabilities = ["create", "update", "read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
