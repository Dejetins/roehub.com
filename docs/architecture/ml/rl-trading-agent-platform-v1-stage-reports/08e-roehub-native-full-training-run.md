---
doc: rl-trading-agent-platform-v1-stage-08e-roehub-native-full-training-run
stage: "08E"
status: in_progress
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08E: Roehub-Native Full Training Run

Status: `in_progress`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08E` started after checking the ledger: Stage `08D` is `accepted` for
methodology execution, Stage `06` is `accepted`, and `current_stage=08E`.
Browser/auth QA is `N/A` for this offline training stage; the Roehub smoke
Keycloak username and host-local password source were not used.

This is `target_host_non_production_training_pre_main` evidence only. No
production `/opt/roehub/app` sync, service reload, browser/auth proof, registry
write, promotion, activation, exchange side effect, paper/testnet/live run, or
mainnet submit was performed.

The full Mac Studio run is managed and resumable, but it is not complete yet.
Stage `08E` must remain `in_progress` until the completed
`roehub_native_candidate_manifest.json` exists with `best` and `final`
checkpoint hashes. Stage `08F` remains blocked.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/75af2b01-6ebc-4d6d-8a9e-9518fe1920d6/pasted-text.txt` |
| Prompt sha256 | `970bbeaa0a840b1c631089494acd907a13848dd1f227258831209b7a3514bdf6` |
| Repo prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08e-roehub-native-full-training-run.md` |
| Repo prompt sha256 | `970bbeaa0a840b1c631089494acd907a13848dd1f227258831209b7a3514bdf6` |
| Previous-stage gate | passed: Stage `08D` is `accepted`, Stage `06` is `accepted`, and `current_stage=08E` |
| Stage `06` manifest | `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json` |
| Stage `06` manifest sha256 | `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` |
| Training candidate level | `roehub_native_candidate` |
| Full run id | `stage08e_roehub_native_full` |
| Full run PID | CPU launch `38545` was stopped by operator request; MPS resume parent `42087`, child Python `42088` at latest process check |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/roehub_native_training.py` | - | - | Stage `08E` Roehub-native trainer wrapper: reuses upstream-compatible environment rollout mechanics with native candidate metadata, warning register, safety payload, progress, resume, `best`/`final` checkpoint and manifest names; now releases MPS cache after validation/checkpoint/final-save boundaries. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage08e_roehub_native_full_training_run.py` | - | - | Opt-in operator CLI for accepted Stage `06` sessionized train/validation artifact loading, strict manifest/file hash checks, status/resume, and full Mac Studio launch. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_roehub_native_training.py` | - | - | Focused unit coverage for `roehub_native_candidate`, Stage `08E` metadata, environment rollout, `best`/`final`, and no HF data use. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08e_roehub_native_training.py` | - | - | Tiny CLI smoke over fixture Stage `06` split artifacts and sanitized manifest output. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08e-roehub-native-full-training-run.md` | - | - | This in-progress Stage `08E` report. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/hf_original_training.py` | - | Allows the existing training config payload to carry `stage=08E` while preserving default `08C` behavior; uses the shared MPS cache release boundary for long MPS runs. | `compatible-change` additive config metadata and GPU resource handling for offline training |
| - | `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | Adds `TorchD3qnPerAgent.release_device_cache()` and a no-op-on-CPU MPS `synchronize`/`empty_cache` helper for long offline training runs. | `compatible-change` internal GPU resource handling |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08E` identifiers and runner. | `compatible-change` additive Python export |
| - | `apps/worker/rl_trading_trainer/main/main.py` | - | Add `stage08e` dispatch to the existing trainer worker entrypoint. | `compatible-change` additive worker subcommand |
| - | `tests/unit/apps/worker/test_rl_trading_trainer.py` | - | Cover `stage08e status` worker dispatch. | `none` test-only |
| - | `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | Covers CPU no-op and MPS cache release helper behavior. | `none` test-only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08E` `in_progress` and keep `08F` blocked until completion. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index |

Outside expected paths: none in git.

Runtime artifacts (`proof_boundary=target_host_non_production_training_pre_main`):

| Path | Host | Reason | sha256 / state |
|---|---|---|---|
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/code_snapshot/` | Mac Studio | Non-production code snapshot used because the Mac Studio git checkout is dirty with older stage work. This avoids mutating `/Users/daniildegtyarev/Projects/roehub.com` or `/opt/roehub/app`; updated with MPS cache-release rework before GPU resume. | updated source hashes: `upstream_methodology.py` `27bb53413998af8c46270ccd839e89ffa65aeb1ddd697c54c0b6e72e4a69d8b6`, `hf_original_training.py` `b36e8d1710c7639b3e2189a524ae0da3095695bca1e44d377591169f9878e390`, `roehub_native_training.py` `40412e317cdf4165fcbf0151afcd52e0576efb576bd3d870ba74971820a0e53a` |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/macstudio_smoke/stage08e_macstudio_smoke/roehub_native_candidate_manifest.json` | Mac Studio | Tiny real Stage `06` target-host smoke with strict manifest/file hash checks; not the full candidate. | file sha256 `32b36f1df3220d46950ee00fe6cbbfeefc695a5543f7e2d014870d0e521034f7`; manifest hash `a1904303d62f4d5ecc772523395af65ff1bc551df669187c73b014c4ea05ae01` |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/macstudio_smoke/stage08e_macstudio_smoke/progress.jsonl` | Mac Studio | Tiny smoke durable progress. | `d17f61e3927ede8548cd3f90da08a755d11a61a6dbb322d8747d10eaa1e57211` |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/mps_debug/stage08e_mps_cacheflush_canary_1200/roehub_native_candidate_manifest.json` | Mac Studio | MPS cache-release canary over accepted Stage `06` artifacts with four validation/checkpoint boundaries; not the full candidate. | completed; manifest hash `cf4806fec9db2cc568248d60a177bd824e9a4c46b36dd8f162e695c64078e3eb`; latest status sha256 `7bbdfd1e85520b79641dd11ffeb68ba65178578ae0ac88fb6f79968d0d5317f8`; `1200/1200`, `device=mps`, elapsed `96.843836833s`, RSS `780.78125 MiB` |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/` | Mac Studio | Managed full native training run directory. | `running`; candidate manifest not yet present |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/latest_status.json` | Mac Studio | Fresh full-run MPS running event at report poll. | observed status sha256 `5e2f76e6b5f602777c82f043d6dbe602f9e6533cab8de37ab90c290420e3f19d`; `24100/55000` episodes, `241000/550000` env steps, `43.8181818182%`, `device=mps`; file continues to change while running |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/progress.jsonl` | Mac Studio | Durable episode/env-step progress. | file continues to append while running; latest observed row at `2026-06-24T20:45:54Z` |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/latest_checkpoint.json` | Mac Studio | Resume checkpoint pointer after MPS validation/checkpoint boundary. | observed sha256 `c8d4d58b1470e4f55eb11a9a47f98aa5b3815a90d1ca83e34120f5ce00dae9aa`; checkpoint at `24000` episodes and `240000` env steps |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/checkpoints/latest_resume.pth` | Mac Studio | Managed resumable checkpoint with policy/target/optimizer/PER state after MPS resume. | observed sha256 `d8c1c8255bf07fdd0823d4e10aa356a5c13ea855d8312f5848a64bac74a60522`; file can change at later checkpoints |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/mps_resume_20260624T203429Z.stdout.log` | Mac Studio | MPS resume stdout log. | empty at latest check |
| `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/mps_resume_20260624T203429Z.stderr.log` | Mac Studio | MPS resume stderr log. | empty at latest check |

## Implemented Training Path

| Area | Result |
|---|---|
| Architecture | Uses Stage `08B` `roehub_d3qn_cnn_dueling_v1`; historical MLP/scripted-transition path is not used. |
| Data | CLI loads only accepted Stage `06` sessionized split artifacts. It does not load Stage `04` HF NPZ files, six-symbol fallback, old `215`-symbol subset, or external HF data. |
| Split use | Current full run uses `hf_period_rebuild_current_trading:train` for training and `hf_period_rebuild_current_trading:validation` for validation-selected `best.pth`. Stage `08F` owns test/backtest evaluation. |
| Progress | Writes `progress.jsonl` and `latest_status.json` with completed episodes, planned episodes, completed env steps, elapsed, ETA, device and resource snapshot. |
| Resume | Writes `latest_resume.pth` plus `latest_checkpoint.json`; resume restores policy/target/optimizer and PER buffer state. |
| Candidate manifest | Will be written only on completion as `roehub_native_candidate_manifest.json`. It is not present yet. |

## Adaptation Diff From `hf_original_candidate`

| Surface | `08C` HF-original branch | `08E` Roehub-native branch |
|---|---|---|
| Dataset source | Stage `04` external HF `train_data.npz` and `val_data.npz`. | Accepted Stage `06` sessionized Binance Futures split artifacts only. |
| Dataset hash | HF split file hashes from Stage `04`. | Stage `06` manifest sha256 `61995c61228705090a9cd5d868776c14435ae421bdf35677a7f5c654af71ac08` plus per-split feature file hash checks. |
| Symbol/session scope | HF original split symbols and source windows. | Roehub-native Binance Futures USDT perpetual sessions from Stage `06`; full accepted train/validation splits in the managed run. |
| Normalization | Train-only stats from HF train split. | Train-only stats from Stage `06` train split only; no validation/test/backtest stats in normalization. |
| Device policy | Stage `08C` started MPS and resumed CPU after an Apple Metal command-buffer error. | Stage `08E` initially launched CPU-only, then was stopped by operator request and resumed on `mps_preferred_cpu_fallback` after adding MPS cache release at validation/checkpoint/final-save boundaries. The MPS canary and the full-run `24000` checkpoint boundary passed, but full completion is still pending. |
| Evaluation | Stage `08D` owns HF test/backtest. | Stage `08F` owns native test/backtest; Stage `08E` does not evaluate candidate quality. |

## Warning Register Carried From `08D`

These warnings do not block `08E`, but they must inform `08F` interpretation:

| Warning | Evidence |
|---|---|
| Weak untuned HF-demo profitability | HF candidate net PnL after costs `2064.37744919`. |
| Simple baseline outperformed HF candidate | Simple baseline net PnL after costs `4508.37753925`. |
| Low positive-session ratio | `0.0324699`. |
| Missing Optuna/tuning | Stage `08D` did not run Optuna/tuning. |
| Demo `30/10` profile | `agent_history_len=30`, `agent_session_len=10`; stronger `90/60` or larger-profile training remains later research hardening. |

## Target-Host Evidence

Tiny real Stage `06` smoke:

| Field | Value |
|---|---|
| Run id | `stage08e_macstudio_smoke` |
| Manifest path | `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/macstudio_smoke/stage08e_macstudio_smoke/roehub_native_candidate_manifest.json` |
| File sha256 | `32b36f1df3220d46950ee00fe6cbbfeefc695a5543f7e2d014870d0e521034f7` |
| Manifest hash | `a1904303d62f4d5ecc772523395af65ff1bc551df669187c73b014c4ea05ae01` |
| Completed episodes / env steps | `2` / `20` |
| Device | `cpu` |
| Dataset source | accepted Stage `06` manifest with strict split feature hash checks |

MPS cache-release canary:

| Field | Value |
|---|---|
| Run id | `stage08e_mps_cacheflush_canary_1200` |
| Manifest path | `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/mps_debug/stage08e_mps_cacheflush_canary_1200/roehub_native_candidate_manifest.json` |
| Manifest hash | `cf4806fec9db2cc568248d60a177bd824e9a4c46b36dd8f162e695c64078e3eb` |
| Latest status sha256 | `7bbdfd1e85520b79641dd11ffeb68ba65178578ae0ac88fb6f79968d0d5317f8` |
| Completed episodes / env steps | `1200` / `12000` |
| Validation/checkpoint points | `4` / `4` |
| Device | `mps` |
| Elapsed | `96.843836833s` |
| RSS at completion | `780.78125 MiB` |

Full managed run (`proof_boundary=target_host_non_production_training_pre_main`):

| Field | Value |
|---|---|
| Run id | `stage08e_roehub_native_full` |
| CPU PID at original launch | `38545`; stopped before completion by operator request |
| MPS resume PID | parent `42087`, child Python `42088` at latest process check |
| Run dir | `/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full` |
| Status at latest poll | `running` |
| Latest timestamp | `2026-06-24T20:45:54Z` |
| Completed episodes / planned | `24100/55000` |
| Completed env steps / planned | `241000/550000` |
| Progress | `43.8181818182%` |
| Device | `mps` |
| Best validation so far | episode `13000`, metric `2.0106214018` |
| Validation curve points | `24` |
| Learn updates | `231001` |
| RSS at latest poll | `5312.140625 MiB` |
| Latest observed status sha256 | `5e2f76e6b5f602777c82f043d6dbe602f9e6533cab8de37ab90c290420e3f19d` |
| Latest checkpoint | `24000` episodes / `240000` env steps; `latest_checkpoint.json` sha256 `c8d4d58b1470e4f55eb11a9a47f98aa5b3815a90d1ca83e34120f5ce00dae9aa`; `latest_resume.pth` sha256 `d8c1c8255bf07fdd0823d4e10aa356a5c13ea855d8312f5848a64bac74a60522` |
| Stderr | MPS resume stderr log empty at latest check |
| Candidate manifest | not present yet |

Status command:

```bash
ssh macstudio 'cd /opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/code_snapshot && /opt/homebrew/bin/uv run --extra rl-ml python scripts/rl_trading/stage08e_roehub_native_full_training_run.py status --run-dir /opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full'
```

Resume command if the process stops before completion:

```bash
ssh macstudio 'cd /opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/code_snapshot && /opt/homebrew/bin/uv run --extra rl-ml python scripts/rl_trading/stage08e_roehub_native_full_training_run.py run --resume --run-id stage08e_roehub_native_full --stage06-manifest-path /opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/stage06_sessionized_manifest.json --output-root /opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full --device-policy mps_preferred_cpu_fallback --torch-num-threads 1 --torch-num-interop-threads 1'
```

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol or service boundary changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `compatible-change` | Additive Python training config metadata for `stage=08E` and additive `stage08e` worker/CLI subcommand. Existing runtime defaults remain unchanged. |
| Request hash / cache key / persistence identity | `none` | No request/cache/persistence identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, DB, Redis, registry, paper/testnet/live or mainnet side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized progress, status, smoke manifest and in-progress stage ledger evidence under the ML artifact root. |
| Benchmark / rollout gates | `compatible-change` | Stage `08E` is now `in_progress`; `08F` remains blocked until the completed native candidate manifest exists. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline training only; no API or live inference hot path changed. Runtime resource evidence is training-progress evidence, not a production-latency claim. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `08D` and Stage `06` are `accepted`, `current_stage=08E`, and Stage `08E` may run |
| Prompt hash | passed; `970bbeaa0a840b1c631089494acd907a13848dd1f227258831209b7a3514bdf6` |
| Focused local ruff | passed, including MPS cache-release rework |
| Focused local tests | passed; latest focused set `14 passed` |
| Focused local pyright | passed; `0 errors` |
| Mac Studio snapshot focused tests | passed; latest focused set `14 passed` |
| Mac Studio tiny real Stage `06` smoke | passed; manifest file sha256 `32b36f1df3220d46950ee00fe6cbbfeefc695a5543f7e2d014870d0e521034f7` |
| Mac Studio MPS cache-release canary | passed; `stage08e_mps_cacheflush_canary_1200` completed on `device=mps`, manifest hash `cf4806fec9db2cc568248d60a177bd824e9a4c46b36dd8f162e695c64078e3eb`, `1200/1200` episodes, four validation/checkpoint points |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `415 passed, 3 warnings` |
| Full Mac Studio training | `in_progress` on `device=mps`; report-poll status hash `5e2f76e6b5f602777c82f043d6dbe602f9e6533cab8de37ab90c290420e3f19d`; `24100/55000` episodes; latest checkpoint at `24000`; candidate manifest not yet present |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because
the available multi-agent tool contract requires an explicit user request before
spawning subagents.

Review scope: Stage `08E` implementation/report, GPU rework handoff, ledger
handoff, file/runtime manifest, proof-boundary/browser-auth wording, contract
impact, quality gates, warning register and `08F` blocker state.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: Release for the current `in_progress` handoff state; not Release for
Stage `08E` acceptance because the full candidate manifest is still absent.

Blockers fixed: report and ledger both keep `08F` blocked until
`roehub_native_candidate_manifest.json` exists; runtime proof is labeled
`target_host_non_production_training_pre_main`; browser/auth and exchange
side-effect surfaces are explicitly `N/A`; file/runtime manifests are listed in
both report and ledger; CPU-only wording was replaced with MPS resume evidence.

Local follow-up check: completed; docs index check passed after report/ledger
updates.

Residual risks: full training is still running from a non-production code
snapshot on MPS and must be rechecked for completion, final hashes and resource
evidence before Stage `08E` can be accepted. The MPS canary and `24000`
checkpoint boundary passed, but this is still not full-run completion.

## Residual Risks

- Stage `08E` is not accepted yet; full training is still running on MPS and the
  `roehub_native_candidate_manifest.json` is absent.
- `08F` remains blocked until the completed full-run manifest records
  train-only normalization stats, `best`/`final` checkpoint hashes, validation
  curves, resource metrics and progress hash.
- The run uses a non-production code snapshot because the Mac Studio git
  checkout is dirty with older stage work. This is valid
  `target_host_non_production_training_pre_main` evidence, not
  `post_main_production_runtime_proof`.
- MPS execution is now enabled and running, but the Stage `08C` Metal
  command-buffer failure class remains a residual long-run risk until the full
  `55000/55000` run completes and writes final hashes.

## 08F Handoff

Stage `08F` is not allowed yet.

The next executor must first confirm that the full run has completed and that
`/opt/roehub/state/rl_trading/training_runs/stage08e_roehub_native_full_training_run_v1/full/stage08e_roehub_native_full/roehub_native_candidate_manifest.json`
exists with status `completed`. Only then may the ledger advance to `08F`.
