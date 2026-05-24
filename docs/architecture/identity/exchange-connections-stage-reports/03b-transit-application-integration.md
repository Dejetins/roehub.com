# Stage 3B: Transit Application Integration

Дата проверки: 2026-05-24.

Статус: accepted for application integration. Stage 3A OpenBao/Vault runtime
evidence is accepted and repeated after the `exchange-control` application
secret boundary was wired to Transit-compatible config.

## Scope

Stage 3B adds the application secret cipher port, Transit adapter, redacted
secret DTOs, product fail-closed config and deterministic tests inside the
`exchange-control` boundary. It does not provision OpenBao/Vault, migrate
legacy `identity_exchange_keys`, add Binance/Bybit validation, expose plaintext
credentials to `apps/api`, or implement order execution.

## Prerequisite Evidence

| Component | Secret operation | Command / test | Expected result | Observed result | Blocker |
|---|---|---|---|---|---|
| Stage 3A report | Runtime prerequisite | `docs/architecture/identity/exchange-connections-stage-reports/03a-openbao-vault-runtime-provisioning.md` | Stage 3A is accepted before Stage 3B starts. | Accepted: OpenBao 2.5.4, `OPENBAO_ADDR`, Transit key, scoped tokens, Monit, Prometheus and recovery evidence are recorded. | None |
| Branch gate | Delivery safety | `test "$(git branch --show-current)" = main` | Work starts on `main`. | Passed before implementation: current branch is `main`. | None |
| Fast-forward gate | Delivery safety | `git pull --ff-only origin main` | Local checkout can fast-forward from `origin/main`. | Passed before implementation: `Already up to date.` | None |

## Secret Boundary Evidence

| Component | Secret operation | Command / test | Expected result | Observed result | Blocker |
|---|---|---|---|---|---|
| Secret cipher port | Encrypt/decrypt/fingerprint contract | `src/trading/contexts/exchange_control/application/secret_cipher.py` | `ExchangeSecretCipher` protocol exists inside `exchange-control`. | Added `ExchangeSecretCipher`, `ExchangeCredentialSecret`, `ExchangeCredentialCiphertext`, `ExchangeCredentialFingerprint` and sanitized `ExchangeSecretCipherError`. | None |
| Transit adapter | OpenBao/Vault-compatible HTTP calls | `src/trading/contexts/exchange_control/adapters/outbound/openbao_transit.py` | Adapter uses Stage 3A env/runtime contract and key `roehub-exchange-credentials`. | Added `OpenBaoTransitExchangeSecretCipher` using Transit encrypt, decrypt and HMAC endpoints with `X-Vault-Token` from `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`. | None |
| Test/dev cipher | Deterministic tests | `test_deterministic_test_cipher_encrypts_and_fingerprints_without_decrypt_path` | Tests can cover secret operations without live OpenBao or real secrets. | Passed: deterministic encrypt/fingerprint output and no decrypt path in fake cipher. | None |
| Redaction | Repr/error safety | `test_secret_value_objects_redact_repr`; `test_openbao_transit_adapter_sanitizes_http_errors` | Plaintext, ciphertext detail and provider error body are not exposed. | Passed: reprs redact values and HTTP errors normalize to sanitized messages. | None |
| `apps/api` boundary | No decrypt path | `rg -n "ExchangeSecretCipher|OpenBaoTransit|openbao_transit_v1|vault_transit_v1|ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN|ROEHUB_API_TRANSIT_TOKEN|transit/decrypt" apps/api \|\| true` | `apps/api` does not import the secret cipher, Transit adapter or token env names. | Passed: no matches. | None |

## Config And ACL Evidence

| Service | Token / env | Operation | Expected result | Observed result | Blocker |
|---|---|---|---|---|---|
| `exchange-control` | `ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER=openbao_transit_v1` | Product startup config | `ROEHUB_ENV=prod` requires Transit backend and rejects dev/in-memory mode. | Passed in `test_prod_runtime_fails_closed_without_transit_config`. | None |
| `exchange-control` | `OPENBAO_ADDR` | Product startup config | Product config fails closed when the runtime address is missing. | Passed in focused runtime tests. | None |
| `exchange-control` | `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` | Product startup config | Product config fails closed when the exchange-control Transit token is missing. | Passed in focused runtime tests; token value is not stored in config docs. | None |
| `apps/api` | `ROEHUB_API_TRANSIT_TOKEN` | Product startup guard and deny evidence | API token is required only as a configured deny-evidence identity; `apps/api` receives no decrypt adapter. | Passed in focused runtime tests and `apps/api` grep evidence. | None |
| Transit key | `ROEHUB_EXCHANGE_CONTROL_TRANSIT_KEY` | Product startup config | Key name must remain `roehub-exchange-credentials`. | Passed: wrong key fails config validation. | None |
| Mac Studio OpenBao | `OPENBAO_ADDR` | `ssh macstudio '... curl -fsS "$OPENBAO_ADDR/v1/sys/health"'` | Runtime is initialized and unsealed. | Passed: `initialized=true`, `sealed=false`, version `2.5.4`. | None |
| `exchange-control` | `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN` | `curl -fsS -X POST "$OPENBAO_ADDR/v1/transit/encrypt/roehub-exchange-credentials" ... '{"plaintext":"U1RBR0UzQl9TTU9LRQ=="}'` | Scoped identity can encrypt through Transit. | Passed on Mac Studio: sanitized marker `exchange_control_encrypt=ok`; response body discarded. | None |
| `apps/api` | `ROEHUB_API_TRANSIT_TOKEN` | `curl -i -X POST "$OPENBAO_ADDR/v1/transit/decrypt/roehub-exchange-credentials" ... '{"ciphertext":"vault:v1:stage3b-placeholder"}'` | API identity cannot decrypt. | Passed on Mac Studio: HTTP `403`, recorded as `api_decrypt_denied_http=403`. | None |

## Quality Gates

| Gate | Expected result | Observed result | Blocker |
|---|---|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations` | Focused exchange-control and migration tests pass. | Passed: `21 passed in 0.46s`. | None |
| `uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control` | Lint passes for changed secret boundary and tests. | Passed: `All checks passed!`. | None |
| `uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control` | Type check passes. | Passed: `0 errors, 0 warnings, 0 informations`. | None |
| `python -m tools.docs.generate_docs_index --check` | Docs index is current after Markdown changes. | Passed after docs update. | None |
| `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE\|api_secret\|passphrase" logs output .playwright-cli \|\| true` | No test secret markers in logs/output/browser artifacts; missing artifact dirs are acceptable. | Passed with no committed artifact evidence; `logs` directory is absent. | None |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `none` | Existing `apps/api` routes and DTOs are unchanged. |
| Port / application contract | `compatible-change` | New `ExchangeSecretCipher` port and Transit adapter are additive inside `exchange-control`. |
| DTO / repr safety | `compatible-change` | New secret value objects redact reprs and normalize secret-cipher errors; no public response fields changed. |
| Persisted schema | `none` | No database migration, table shape or backfill is introduced in Stage 3B. |
| Config schema | `compatible-change` | `exchange-control` product mode now requires `ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER`, `OPENBAO_ADDR`, `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`, `ROEHUB_API_TRANSIT_TOKEN` and fixed key `roehub-exchange-credentials`; dev default remains explicit test/dev fake. |
| Ops / runtime | `compatible-change` | Existing launchd runtime now exports the Transit backend selector and fixed key while sourcing token/address values from host-local env. |
| `apps/api` decrypt capability | `none` | No `apps/api` code path imports the cipher or Transit adapter; runtime deny evidence remains HTTP `403`. |
| Persistence identity / request hash / cache key | `none` | No cache, request hash or persistence identity semantics are changed. |

## Recovery And Rollback

| Component | Procedure | Expected result | Observed result | Blocker |
|---|---|---|---|---|
| Application rollback | Revert Stage 3B code/config and remove `ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER` / `ROEHUB_EXCHANGE_CONTROL_TRANSIT_KEY` from exchange-control launchd command. | `exchange-control` returns to no-secret-backend Stage 2 behavior. | Documented; no database rollback required. | None |
| Token rotation | Update host-local `/Users/daniildegtyarev/.config/roehub/roehub.env`, restart `exchange-control`, rerun ACL smoke. | New token works without repo changes. | Documented in runbook. | None |
| Key rotation / rewrap | Use Transit rotate/rewrap command shapes; do not decrypt through `apps/api`. | Existing ciphertexts remain Transit-managed. | Runbook updated with command design for future Stage 4+ stored ciphertexts. | None |
| Runtime unavailable | Product config and Transit adapter fail closed with sanitized errors. | Secret operations do not fall back to in-memory fake in `ROEHUB_ENV=prod`. | Covered by focused tests. | None |

## Stage 3C Handoff Facts

- `ExchangeSecretCipher` is implemented under `exchange-control`; Stage 3C can
  expose local-only internal command endpoints that use this boundary.
- Product mode requires `openbao_transit_v1` or `vault_transit_v1`; `in_memory_dev`
  remains test/dev only and is rejected in `ROEHUB_ENV=prod`.
- Runtime env names are `OPENBAO_ADDR`,
  `ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN`, `ROEHUB_API_TRANSIT_TOKEN`,
  `ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER` and
  `ROEHUB_EXCHANGE_CONTROL_TRANSIT_KEY`; values remain host-local.
- Transit key remains `roehub-exchange-credentials`; application config rejects
  any other key name.
- `apps/api` still has no decrypt adapter/import path and decrypt remains denied
  by runtime ACL with HTTP `403`.
- Stage 3C may add the internal command API/client only after Stage 3B direct-main
  delivery and CI/deploy verification complete. Stage 4/5 remain blocked until
  Stage 3C is accepted.
