---
doc: rl-trading-agent-platform-v1-stage-09b-local-artifact-backup-restore
status: accepted
stage: 09B
updated_at: 2026-07-02
---

# Stage 09B: local artifact backup and restore drill

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

## Pre-Edit Gate

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/09b-local-artifact-backup-restore.md` |
| Prompt sha256 | `c3b0cb160b04ed32ce284a1ae77115d5bf82e0c3793ca3429eebbce350677534` |
| Ledger state observed before work | `current_stage=09B`; Stage `09` accepted; Stage `09B` pending/current |
| Prerequisite verdict | accepted Stage `09`; implementation may proceed; acceptance requires target-host backup/restore evidence |
| `.codex/agents/.context/promt_manager_state.yaml` | read; treated as stale prompt-generation state where it conflicts with current `.codex/AGENTS.md`/ledger direct-main and local stage policy |
| Browser/auth | `N/A`; username `smoke_e2e_keycloak` was not used and `ROEHUB_SMOKE_E2E_PASSWORD` was not read |
| Exchange/provider effects | `N/A`; no exchange SDK, order submit, paper/testnet/live/mainnet path, provider credential or raw provider payload surface is in scope |

## Planned Concrete File List Before Edit

| Path | Planned state | Reason |
|---|---|---|
| `src/trading/contexts/rl_trading/domain/artifact_backup.py` | create | Stage `09B` backup manifest, registry metadata dump, restore validation, rollback dry-run metadata and retention contract. |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modify | Export the Stage `09B` domain surface for CLI/tests. |
| `scripts/rl_trading/stage09b_local_artifact_backup_restore.py` | create | Operator-facing local backup/restore drill and rollback dry-run command. |
| `tests/unit/contexts/rl_trading/domain/test_artifact_backup.py` | create | Focused hash/path/restore/tamper/metadata tests. |
| `docs/runbooks/mac-studio-native-backend-operations.md` | modify | Add Stage `09B` operator command, output paths, retention and residual single-host risk note. |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/09b-local-artifact-backup-restore.md` | create | This stage report. |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modify | Record Stage `09B` status/evidence/blocker and downstream allowance. |
| `docs/architecture/README.md` | modify if docs index requires regeneration | Docs index sync after adding this report. |

Initial blockers: none for implementation. During validation, non-interactive `macstudio` SSH was temporarily blocked by passphrase/agent availability for `/Users/daniildegtyarev/.ssh/macstudio_ed25519`. The operator loaded the key into the agent, and `ssh -o BatchMode=yes macstudio 'true'` then passed with exit code `0`.

## Scope

Implemented and accepted Stage `09B` local backup/restore support around accepted RL metadata artifacts:

- deterministic backup manifest for accepted compact manifests/summaries;
- registry metadata dump with active-champion metadata, source manifest references, calibration pre-Stage-`10` status and retention policy;
- restore drill that restores to a separate path and validates hashes after restore;
- rollback dry-run command that validates current/previous model ids against the registry dump and never deletes artifacts;
- runbook commands and retention guidance.

Pre-main Mac Studio artifact evidence completed from the checkout `/Users/daniildegtyarev/Projects/roehub.com` against `/opt/roehub/state/rl_trading/`. The Stage `09B` code snapshot was copied to the Mac Studio checkout for a non-production artifact drill. Boundary label: `target_host_readiness_pre_main`. This evidence proves host-local artifact backup/restore command behavior only; it is not production validation for the target revision, not `/opt/roehub/app` deploy validation, and not `post_main_production_runtime_proof`.

Proof boundary: `post_main_production_runtime_proof` would require the target revision to be on `main`, green GitHub Actions/CI, deployment or verified sync into `/opt/roehub/app`, and then the appropriate production runtime smoke. Stage `09B` does not perform or claim that boundary.

Not in scope:

- production DB migration apply or registry write;
- `/opt/roehub/app` deploy or production runtime smoke;
- active model load, paper/testnet/live/mainnet readiness;
- cloud/S3/off-host backup.

## Business Impact

Stage `09B` снижает операционный риск перед будущей калибровкой и активацией RL-модели: accepted candidate metadata теперь можно локально восстановить из backup manifest и registry dump, а rollback dry-run показывает, что операторская команда не удаляет артефакты и проверяет ожидаемые model ids.

Пользовательский продуктовый эффект на этом stage отсутствует: UI, API, тарифы, paper/testnet/live/mainnet execution, exchange submit и runtime activation не менялись. Бизнес-результат stage — не запуск торговли, а снятие блокера “единственная копия критичных ML metadata” перед Stage `10`.

`N/A`: customer-facing copy, billing, entitlement changes, order execution, provider calls, browser-visible behavior and production deployment.

## Observed State

Mac Studio source artifact availability:

| Artifact | Observed sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json` | `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7` |
| `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json` | `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08l_reward_warm_start_research_v1/stage08l_reward_warm_start_99a00ffa43c83b9ac553/stage08l_reward_warm_start_research_summary.json` | `5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1` |

The implemented CLI intentionally copies compact manifests/summaries only. It does not copy large split artifacts, raw checkpoint tensors, secrets, provider payloads, cookies or credentials.

Calibration state: Stage `10` has not produced per-ticker calibration packs yet. Stage `09B` records `calibration_pack.status=not_created_pre_stage10` in the registry metadata dump and keeps runtime activation blocked until later accepted calibration/promotion/runtime stages.

Rollback state: because no real production champion history exists yet, the rollback evidence is a non-production metadata drill using `stage09b_previous_accepted_champion_restore_drill` as the previous-champion fixture. It proves command/reference semantics only; it does not claim a real production rollback target.

## File Manifest

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/artifact_backup.py` | created | Stage `09B` artifact specs, backup manifest, registry dump, restore validation, rollback dry-run metadata and retention policy. | `compatible-change` additive internal Python domain surface |
| `src/trading/contexts/rl_trading/domain/__init__.py` | modified | Export Stage `09B` backup/restore helpers. | `compatible-change` additive internal Python exports |
| `scripts/rl_trading/stage09b_local_artifact_backup_restore.py` | created | Operator-facing `run-drill` and `rollback-dry-run` command. | `compatible-change` host-local CLI |
| `tests/unit/contexts/rl_trading/domain/test_artifact_backup.py` | created | Focused unit coverage for full drill, tamper detection and path containment. | `none` test-only |
| `docs/runbooks/mac-studio-native-backend-operations.md` | modified | Operator commands, output paths, retention policy and residual single-host disk risk. | `compatible-change` runbook |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/09b-local-artifact-backup-restore.md` | created | Stage report, evidence and handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Record Stage `09B` accepted state and open Stage `10`. | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | modified | Docs index includes this report. | `compatible-change` docs index |

Deleted: none.

Outside expected paths: none.

Runtime artifacts created on `macstudio` under `/opt/roehub/state/rl_trading/`:

| Artifact | sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/backups/stage09b_local_artifact_backup_restore_v1/stage09b_macstudio_20260702t203000z/stage09b_backup_manifest.json` | `a0c9508f72fe2a0267ab1bc3307025e2551a7f3e606d9549b42b0fbeb13f18a5` |
| `/opt/roehub/state/rl_trading/backups/stage09b_local_artifact_backup_restore_v1/stage09b_macstudio_20260702t203000z/metadata/stage09b_registry_metadata_dump.json` | `854e909139c7397819b08253c308b091ffff8ebd15b12591c41e7e572af9449d` |
| `/opt/roehub/state/rl_trading/restore_drills/stage09b_local_artifact_backup_restore_v1/stage09b_macstudio_20260702t203000z/stage09b_restore_report.json` | `36d05e1dcd09885523109bb19bddd6984ea054bbb6bd47d5533796b89ed41d5e` |
| `/opt/roehub/state/rl_trading/backups/stage09b_local_artifact_backup_restore_v1/stage09b_macstudio_20260702t203000z/metadata/stage09b_rollback_manifest.json` | `6af5fe0ac5e8074ba2cb5518a7d846cb72c9b9e6020a2fb33686ccdbe438c877` |

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API route, HTTP payload or UI read model changed. |
| Port contract | `none` | No existing port signature changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration or existing persisted schema changed in Stage `09B`. |
| Config schema/defaults | `none` | No env/YAML/default resolution changed. |
| Request hash/cache key/persistence identity | `none` | Existing request/cache identities are unchanged. |
| Host-local CLI contract | `compatible-change` | Adds `stage09b_local_artifact_backup_restore.py run-drill` and `rollback-dry-run`; no existing command changed. |
| Artifact manifest/report semantics | `compatible-change` | Adds backup manifest, registry metadata dump, restore report and rollback manifest schemas under `/opt/roehub/state/rl_trading/`. |
| Service-call auth/timeout/retry/error semantics | `none` | No service call or external adapter changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider submit, Redis publish, DB write execution or production mutation. |
| Logs/metrics/traces/audit/redaction | `compatible-change` | Adds sanitized local artifact reports; no secrets, raw provider payloads or raw tensors are written. |
| Alert/runbook semantics | `compatible-change` | Adds operator runbook commands and residual single-host risk note. |
| Benchmark/rollout gates | `compatible-change` | Stage `09B` is accepted after `target_host_readiness_pre_main` artifact backup/restore evidence; Stage `10` may start, while runtime activation remains blocked by later stages. |
| Browser-visible behavior | `none` | Browser/auth surface is `N/A`. |
| Performance hot path | `none` | Offline metadata/file-copy command only; no inference/training hot path is wired. |

## Quality Gates And Evidence

| Gate | Result |
|---|---|
| Focused pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_artifact_backup.py` -> `3 passed` |
| Focused ruff | passed: `uv run ruff check src/trading/contexts/rl_trading/domain/artifact_backup.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage09b_local_artifact_backup_restore.py tests/unit/contexts/rl_trading/domain/test_artifact_backup.py` |
| Focused pyright | passed: `uv run pyright src/trading/contexts/rl_trading/domain/artifact_backup.py src/trading/contexts/rl_trading/domain/__init__.py scripts/rl_trading/stage09b_local_artifact_backup_restore.py tests/unit/contexts/rl_trading/domain/test_artifact_backup.py` -> `0 errors` |
| CLI parse smoke | passed locally and on `macstudio`: `uv run python scripts/rl_trading/stage09b_local_artifact_backup_restore.py --help` |
| Local rollback dry-run parse smoke | passed: `uv run python scripts/rl_trading/stage09b_local_artifact_backup_restore.py rollback-dry-run --expected-current-model-version-id stage08m_a3823cbd01143878_fd7c614b --to-model-version-id stage09b_previous_accepted_champion_restore_drill --reason stage09b_restore_drill --generated-at-utc 2026-07-02T20:30:00Z` -> `status=accepted` |
| Prompt-level ruff | passed: `uv run ruff check src/trading/contexts/rl_trading apps tests` |
| Prompt-level pyright | passed: `uv run pyright src/trading/contexts/rl_trading apps tests` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` -> `455 passed, 3 warnings` |
| Docs index | passed: `uv run python -m tools.docs.generate_docs_index`; `uv run python -m tools.docs.generate_docs_index --check` |
| Non-interactive Mac Studio SSH | passed after operator loaded key: `ssh -o BatchMode=yes macstudio 'true'` |
| Mac Studio checkout/tooling | `target_host_readiness_pre_main`: observed `HEAD=d878e30c`; `uv=/opt/homebrew/bin/uv`; Stage `09B` files were synced to `/Users/daniildegtyarev/Projects/roehub.com` for this pre-main artifact drill |
| Mac Studio artifact backup/restore drill | `target_host_readiness_pre_main` passed: `uv run python scripts/rl_trading/stage09b_local_artifact_backup_restore.py run-drill --run-id stage09b_macstudio_20260702t203000z --generated-at-utc 2026-07-02T20:30:00Z` -> `status=accepted`, `artifact_count=8`, `drill_result_hash=31076351a474e89973c7008c5b33fb8ea900c3f2e116c336cf420415a2b83f7c` |
| Mac Studio restore report schema check | `target_host_readiness_pre_main` passed: backup manifest `artifact_count=8`, `backup_entries=8`; restore report `restored_artifact_count=8`; `reference_validation.status=accepted` |
| Mac Studio rollback dry-run | `target_host_readiness_pre_main` passed: `uv run python scripts/rl_trading/stage09b_local_artifact_backup_restore.py rollback-dry-run --registry-metadata-dump /opt/roehub/state/rl_trading/backups/stage09b_local_artifact_backup_restore_v1/stage09b_macstudio_20260702t203000z/metadata/stage09b_registry_metadata_dump.json --expected-current-model-version-id stage08m_a3823cbd01143878_fd7c614b --to-model-version-id stage09b_previous_accepted_champion_restore_drill --reason stage09b_restore_drill --generated-at-utc 2026-07-02T20:30:00Z` -> `status=accepted`, `rollback_dry_run_hash=880519724ed6c3ac7e045b045c8fc9bbe4be5655c41a0b6718493cfc4974a5c9` |
| Whitespace | passed: `git diff --check` |

The generated registry dump records:

- `active_champion.model_version_id=stage08m_a3823cbd01143878_fd7c614b`;
- `calibration_pack.status=not_created_pre_stage10`;
- `same_physical_disk=true`;
- `source_manifest_entries=2` in restore reference validation.

## Retention Policy

Retain forever:

- accepted champion manifest and scorecard;
- Stage `08J` source dataset manifest;
- Stage `08L` research source summary;
- backup manifest;
- registry metadata dump;
- rollback manifest;
- previous-champion metadata fixture.

Retain for `30` days by default:

- restore drill copies;
- calibration-status drill metadata until Stage `10` supersedes it with real calibration artifacts.

Removable after evidence capture:

- scratch/temporary outputs that are not referenced by backup manifest, restore report, registry metadata dump, rollback manifest or this report.

## Residual Risks

- Backup/restore roots are on the same physical disk (`same_physical_disk=true`; `/System/Volumes/Data` was observed for `/opt/roehub/state/rl_trading/`). This protects against operator error and accidental local deletion, but it is not disaster recovery. Stage `19`/`21` must either prove a second-disk/off-host backup path or explicitly accept the residual single-host disk risk.
- Stage `10` calibration artifacts do not exist yet; runtime activation remains fail-closed until later accepted calibration, promotion and runtime stages.
- Real previous champion history does not exist yet; rollback proof is limited to a metadata dry-run fixture until at least two accepted champions exist.
- Delivery state is `local-only` with a synced Mac Studio checkout snapshot for artifact evidence. No `origin/main` commit, CI/deploy, `/opt/roehub/app` deploy or production runtime validation is claimed by this stage.

## Cold Self-Review

Mode: `cold self-review fallback`. Independent subagent review was not used because subagent spawning requires an explicit user request.

Verdict: `Release`.

Checked:

- stage continuity: Stage `09` accepted, Stage `09B` is accepted, and Stage `10` is now the next executable stage;
- proof boundary: Mac Studio evidence is explicitly `target_host_readiness_pre_main` for a non-production artifact drill, not production validation for the target revision; `post_main_production_runtime_proof` would require `main`, green GitHub Actions/CI, deploy or verified sync into `/opt/roehub/app`, and production runtime smoke;
- contract impact: new surface is additive local CLI/artifact metadata; API/UI/exchange/persisted DB contracts unchanged;
- secrets/redaction: implementation records paths/hashes/counts only and tests check that model weights are not copied into the registry dump;
- validation: focused and prompt-level local gates passed; Mac Studio `run-drill`, restore hash validation and `rollback-dry-run` passed.

## Next-Stage Handoff

Stage `09B` is accepted. Ledger `current_stage` moves to `10`.

Next allowed prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/10-per-ticker-calibration.md`.

Stage `10A`, `13`, paper/testnet/live and mainnet work remain closed until their own prerequisites are accepted. Runtime activation must continue to fail closed while `calibration_pack.status=not_created_pre_stage10`.
