path "transit/encrypt/roehub-exchange-credentials" {
  capabilities = ["update"]
}

path "transit/decrypt/roehub-exchange-credentials" {
  capabilities = ["update"]
}

path "transit/hmac/roehub-exchange-credentials/*" {
  capabilities = ["update"]
}
