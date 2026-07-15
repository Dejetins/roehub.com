path "sys/storage/raft/snapshot" {
  capabilities = ["read", "sudo"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}
