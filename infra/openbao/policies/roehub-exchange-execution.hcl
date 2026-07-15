path "kv/data/roehub/exchange/*" {
  capabilities = ["read"]
}

path "transit/encrypt/roehub-exchange-credentials" {
  capabilities = ["update"]
}

path "transit/decrypt/roehub-exchange-credentials" {
  capabilities = ["update"]
}

path "transit/hmac/roehub-exchange-credentials/*" {
  capabilities = ["update"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
