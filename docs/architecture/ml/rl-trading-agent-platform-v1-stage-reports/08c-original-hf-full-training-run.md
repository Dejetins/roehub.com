---
doc: rl-trading-agent-platform-v1-stage-08c-original-hf-full-training-run
stage: "08C"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-24"
---

# Stage 08C: Original HF Full Training Run

Status: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

Stage `08C` started after checking the ledger: Stage `04` is `accepted`,
Stage `08B` is `accepted`, `current_stage=08C`, and Stage `08C` is allowed.
Browser/auth QA is `N/A` for this offline training stage; the Roehub smoke
Keycloak username and host-local password source were not used.

This report records implementation and
`target_host_non_production_training_pre_main` completion on Mac Studio. It is
not `post_main_production_runtime_proof`: no production `/opt/roehub/app` sync,
service reload, browser/auth proof, registry write, promotion, activation,
exchange side effect, paper/testnet/live run, or mainnet submit was performed.
The accepted handoff artifact is the completed `hf_original_candidate` manifest.
The initial MPS run stopped after a Metal Performance Shaders internal
command-buffer error; the same run was resumed from `latest_resume.pth` in
CPU-only mode and completed without a new stderr error. Stage `08D` is now
allowed to consume the completed candidate manifest.

## Source Pinning

| Source | Evidence |
|---|---|
| Prompt path | `/Users/daniildegtyarev/.codex/attachments/6b64868c-bd6d-44e1-8dd7-0764aec8e830/pasted-text.txt` |
| Prompt sha256 | `a745ed404df17a6bb6441a0cf0df4ffa358604ba232a882f48836fbeefb70c39` |
| Repo prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08c-original-hf-full-training-run.md` |
| Previous stage gate | passed: Stage `04` and Stage `08B` are `accepted`; `current_stage=08C` |
| HF dataset source | Stage `04` original HF dataset under `/opt/roehub/state/rl_trading/hf_reproducibility/dataset/ResearchRL/open-rl-trading-binance-dataset` |
| External code vendored | none |
| Raw datasets/checkpoints/provider payloads in git | none |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/rl_trading/domain/hf_original_training.py` | - | - | Stage `08C` HF-original environment-rollout trainer, durable progress writer, best/final checkpoint materialization, resume checkpoint, report/manifest builders and validation metric loop. | `compatible-change` additive internal Python domain surface |
| `scripts/rl_trading/stage08c_original_hf_full_training_run.py` | - | - | Operator CLI for real Stage `04` HF NPZ loading, strict hash checks, status/resume support and full Mac Studio training launch. | `compatible-change` additive opt-in CLI |
| `tests/unit/contexts/rl_trading/domain/test_hf_original_training.py` | - | - | Focused unit coverage for progress, checkpoint policy, manifest and no scripted-transition path. | `none` test-only |
| `tests/perf_smoke/contexts/rl_trading/test_stage08c_original_hf_training.py` | - | - | Tiny CLI fixture smoke for the Stage `08C` operator surface. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08c-original-hf-full-training-run.md` | - | - | This accepted Stage `08C` report. | `compatible-change` docs/report |
| - | `src/trading/contexts/rl_trading/domain/upstream_methodology.py` | - | Add PER buffer state snapshot/restore for resumable full training and ensure `learn()` returns the policy network to train mode after validation/inference. | `compatible-change` additive Python behavior for offline training |
| - | `src/trading/contexts/rl_trading/domain/__init__.py` | - | Export additive Stage `08C` trainer identifiers and helpers. | `compatible-change` additive Python export |
| - | `apps/worker/rl_trading_trainer/main/main.py` | - | Add `stage08c` dispatch to the existing trainer worker entrypoint. | `compatible-change` additive worker subcommand |
| - | `tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py` | - | Cover PER state snapshot/restore. | `none` test-only |
| - | `tests/unit/apps/worker/test_rl_trading_trainer.py` | - | Cover `stage08c status` worker dispatch. | `none` test-only |
| - | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | Mark Stage `08C` `accepted` and advance `current_stage` to `08D`. | `compatible-change` docs/ledger |
| - | `docs/architecture/README.md` | - | Docs index regeneration after adding this report. | `compatible-change` docs index |

Outside expected paths: none in git.

Runtime artifacts (`proof_boundary=target_host_non_production_training_pre_main`):

| Path | Host | Reason | sha256 / state |
|---|---|---|---|
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/code_snapshot/` | Mac Studio | Non-production code snapshot used because the Mac Studio git checkout was already dirty/stale and missing Stage `08B` files. This avoided mutating `/Users/daniildegtyarev/Projects/roehub.com` and `/opt/roehub/app`. | directory artifact |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/macstudio_smoke/stage08c_macstudio_smoke_clean/hf_original_candidate_manifest.json` | Mac Studio | Tiny real-HF target-host smoke with Stage `04` hash checks enabled; not a full candidate. | manifest hash `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_candidate_manifest.json` | Mac Studio | Completed full original-HF candidate manifest. | file sha256 `189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4`; manifest `candidate_manifest_hash` `c144111b5e74246589b55b1160aa869e0e6de9505f1311a12d8dadd452c50abc` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_training_report.json` | Mac Studio | Completed full-run training report. | `f6c0b49e31191500ac305cf3c81875f1f74d686fda766e043ff84f951c2aaf8b` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/training_config.json` | Mac Studio | Full-run sanitized config and source/dataset lineage. | `4be3a8febc02354c5f56fbf11311d5ca58a2c7eb5bcc3bda04e21f87ff97c7f1` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/train_only_normalization_stats.json` | Mac Studio | Train-only normalization stats from original HF train split. | `c4e03bdb28447d789a8a097d44c73c77140348d841edfd9a4de7b752fd60f51e` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/progress.jsonl` | Mac Studio | Durable episode/env-step progress. | `987e8e56f611dcf8096386d258df4779769b607450dec618330dec9be4be096c` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/latest_status.json` | Mac Studio | Final full-run status. | `1ca938d5f482ce2f824c0c1db9d0be4efe06a8e4ff26b990ed0173865181e355` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/latest_checkpoint.json` | Mac Studio | Resume checkpoint pointer. | `4f4f1ef32dfb022e43b9f5a5ce06b3a2cd5764cc958b55364381719de8c5efe7` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/checkpoints/best.pth` | Mac Studio | Validation-selected best checkpoint for Stage `08D` evaluation. | `3538c77abb363f6ade74cc98113fc5a19be78b2f63c5449e675485ee8ce36e0c` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/checkpoints/final.pth` | Mac Studio | Final checkpoint; diagnostic unless selected by evaluation. | `791b7e9d9d9d61ee657886121680844ad6d5b4b9aac124aa838e3a8f6a4fc229` |
| `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/checkpoints/latest_resume.pth` | Mac Studio | Managed resumable checkpoint with model/optimizer/PER state. | `7cc6d8ca2028b9559de7ab29e5628a7e717b118d7b8c83b098941eed0320e0b2` |

Delivery state: `local-only` implementation plus
`target_host_non_production_training_pre_main` managed run. This is not
`post_main_production_runtime_proof`. A future
`post_main_production_runtime_proof` would require the target revision on
`main`, green CI/GitHub Actions for that revision, deploy or verified sync from
the Mac Studio checkout into `/opt/roehub/app`, and then the appropriate
production runtime smoke. None of those production-proof actions were performed
by Stage `08C`.

## Операторский Итог

Stage `08C` дал первый полный кандидат на оригинальном HF dataset в новой
upstream-compatible цепочке. Бизнес-смысл этого шага узкий: теперь можно
оценивать методологически корректный HF-original candidate в Stage `08D`, вместо
того чтобы снова возвращаться к rejected MLP/scripted-transition path. Это еще
не разрешение на registry, promotion, activation, paper/testnet/live execution
или mainnet.

Logging/redaction coverage: only sanitized paths, hashes, metrics, config ids
and resource counters are recorded in this report and ledger. Secrets, raw
provider payloads, credentials, cookies, tokens and exchange side-effect payloads
are `N/A` and were not used.

Alerts/monitoring/runbook coverage: production alerting is `N/A` because this is
an offline non-production training run under `/opt/roehub/state/rl_trading/`, not
a service deployment. Operational follow-up is the Stage `08D` prompt consuming
the candidate manifest.

## Implemented Training Path

| Area | Result |
|---|---|
| Architecture | Uses `roehub_d3qn_cnn_dueling_v1` from Stage `08B`; historical MLP/scripted-transition path is not used. |
| Data | CLI loads original Stage `04` HF `train_data.npz` and `val_data.npz` with strict expected sha256 checks by default. Stage `06` Roehub-native data is not used. |
| Full profile | Defaults to `episodes=55_000`, `batch_size=16`, `learning_rate=1e-4`, `train_start=10_000`, PER capacity `230_000`, validation every `1000` episodes and validation-selected `best.pth` plus `final.pth`. |
| Progress | Writes `progress.jsonl` and `latest_status.json` with completed episodes, planned episodes, completed env steps, elapsed, ETA, device and resource snapshot. |
| Resume | Writes `latest_resume.pth` plus `latest_checkpoint.json`; resume restores policy/target/optimizer and PER buffer state. |
| Candidate manifest | On completion only, writes `hf_original_candidate_manifest.json` with train dataset hashes, config hash, code state, train-only normalization stats hash, best/final checkpoint hashes, validation curves, resource metrics and progress hash. |
| Evaluation boundary | Stage `08D` owns test/backtest acceptance. Stage `08C` does not run evaluation/backtest acceptance. |

## Target-Host Evidence

Tiny real-HF smoke:

| Field | Value |
|---|---|
| Run id | `stage08c_macstudio_smoke_clean` |
| Manifest path | `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/macstudio_smoke/stage08c_macstudio_smoke_clean/hf_original_candidate_manifest.json` |
| Manifest hash | `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98` |
| Train split hash | matched Stage `04` expected hash |
| Validation split hash | matched Stage `04` expected hash |
| Completed episodes / env steps | `2` / `20` |
| Device | `cpu` |
| Wall seconds | `0.070736` |
| Best checkpoint sha256 | `c5d1d319c262d9411c510393001f5920a58bb84832a180cc491264582f9afb5a` |
| Final checkpoint sha256 | `67aa57a35d008c82b6616ab0227c69e36d0ecd70c8a60366d77148e5e9d29191` |
| Progress sha256 | `e6bcc3b175f13d79ff1c0e4c2385178eee594211e29d842fe1ff11d4d84bc3d5` |

Full managed run lifecycle
(`proof_boundary=target_host_non_production_training_pre_main`):

| Event | Evidence |
|---|---|
| Initial launch | PID `74035`; `device_policy=mps_preferred_cpu_fallback`; run dir `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full` |
| Last healthy MPS snapshot | `2026-06-24T07:49:27Z`; `4601/55000` episodes; `46010/550000` env steps; `8.3654545455%`; device `mps`; latest status sha256 `f1403d1b8cc4e8d8dfa44687639166d26d8c1785ce4725a3366eaa658d5f251e` |
| MPS stop reason | `stderr.log` recorded Apple Metal Performance Shaders command-buffer `Internal Error (00000001:Internal Error)`; no completed manifest existed at that point. |
| CPU resume | PID `7465`; resumed from `latest_resume.pth` with `device_policy=cpu_only_deterministic`; no new stderr error was observed in `resume_cpu_20260624T100311Z.stderr.log`. |
| Final status | `completed`; timestamp `2026-06-24T10:28:11Z`; device `cpu`; `55000/55000` episodes; `550000/550000` env steps; `100.0%`; wall seconds `1490.868954959`; RSS after `5160.390625 MiB`; throughput `368.8390507376` env steps/sec. |
| Best checkpoint policy | validation metric `Validation_mean_pnl`; best step `470000`; best metric `49.6091622024`; default evaluation checkpoint `best.pth`. |
| Dataset hashes | Stage `04` train hash matched `1c5cdf179777f0a68a81da915749f50d97826282e1419a5314a67b170e9cb14d`; validation hash matched `1e1e347bd4f842680f8a1781bc1e51f790f5e5865796e9ef3bd69548e20c51f4`. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No route, request, response, auth or browser behavior changed. |
| Port contract | `none` | No application port/protocol or service boundary changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No migration or database schema changed. |
| Config schema/defaults | `compatible-change` | Additive Python training config and additive `stage08c` worker/CLI subcommand. Existing runtime defaults remain unchanged. |
| Request hash / cache key / persistence identity | `none` | No request/cache/persistence identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls or auth surfaces changed. |
| External side effects / unknown-state semantics | `none` | No exchange, DB, Redis, registry, paper/testnet/live or mainnet side effect. |
| Logs / metrics / traces / audit / reports | `compatible-change` | Adds sanitized progress, status, manifest/report and stage ledger evidence under the ML artifact root. |
| Benchmark / rollout gates | `compatible-change` | Stage `08C` is accepted; Stage `08D` is now allowed to consume the completed `hf_original_candidate` manifest. |
| Browser-visible behavior | `none` | Browser/auth QA is `N/A`. |
| Performance hot path | `none` | Offline training only; no API or live inference hot path changed. Runtime resource evidence is training-progress evidence, not a production-latency claim. |

## Quality Gates

| Gate | Result |
|---|---|
| Previous-stage ledger gate | passed; Stage `04` and Stage `08B` are `accepted`, `current_stage=08C`, and Stage `08C` may run |
| Prompt hash | passed; `a745ed404df17a6bb6441a0cf0df4ffa358604ba232a882f48836fbeefb70c39` |
| Focused `08C` tests | passed; `uv run pytest -q tests/unit/contexts/rl_trading/domain/test_hf_original_training.py tests/perf_smoke/contexts/rl_trading/test_stage08c_original_hf_training.py tests/unit/contexts/rl_trading/domain/test_upstream_methodology.py tests/unit/apps/worker/test_rl_trading_trainer.py` -> `16 passed` |
| Focused ruff | passed after fixes |
| Focused pyright | passed; `0 errors` |
| `uv run ruff check src/trading/contexts/rl_trading apps tests` | passed |
| `uv run pyright src/trading/contexts/rl_trading apps tests` | passed; `0 errors` |
| `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` | passed; `411 passed, 3 warnings` |
| Mac Studio tiny real-HF smoke | passed; manifest hash `b500b825f72d5c434d3b97b89476acaeb464af78c09c79522a1ec19158660e98`; train/validation split hashes matched Stage `04` |
| Mac Studio full HF run | passed as `target_host_non_production_training_pre_main`; final status `completed`, `55000/55000` episodes, `550000/550000` env steps, manifest file sha256 `189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration |

## Cold-Head Review

Cold-head review: completed.

Mode: cold self-review fallback. Independent subagent review was not used because
the available multi-agent tool contract requires an explicit user request before
spawning subagents.

Review scope: Stage `08C` implementation/report, ledger handoff, file/runtime
manifest, proof-boundary/browser-auth wording, contract impact, quality gates and
`08D` handoff.

Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`.

Verdict: Release after fixes for Stage `08C` acceptance.

Blockers fixed: the full original-HF run completed, the candidate manifest and
checkpoint/progress/status hashes are recorded, the proof boundary is explicit as
`target_host_non_production_training_pre_main`, and the ledger is advanced to
`08D`.

Local follow-up check: completed. Focused tests, required ruff/pyright/unit gates
and docs index check passed.

Residual risks: Stage `08D` evaluation/backtest has not run yet, and this stage
does not provide `post_main_production_runtime_proof`.

## Residual Risks

- The run is executing from a non-production code snapshot because the Mac Studio
  git checkout was dirty/stale. This is valid `target_host_non_production_training_pre_main`
  evidence, not post-main production runtime proof.
- `best.pth` is validation-selected training output, not an evaluation/backtest
  acceptance result. Stage `08D` must still score it against sanity baselines and
  methodology-parity rules.
- No `08D` evaluation/backtest, registry write, promotion, activation, paper/testnet/live
  execution or mainnet submit was performed.

## 08D Handoff

Stage `08D` is now allowed.

The next executor must first consume:

1. `hf_original_candidate_manifest.json` at `/opt/roehub/state/rl_trading/training_runs/stage08c_original_hf_full_training_run_v1/full/stage08c_hf_original_full/hf_original_candidate_manifest.json`;
2. manifest file sha256 `189370a40c874481a52262902884c1be3bd58b1faa0f7a581d6d04a6ae9e80d4`;
3. final progress `55000/55000` episodes and `550000/550000` env steps;
4. validation-selected `best.pth` sha256 `3538c77abb363f6ade74cc98113fc5a19be78b2f63c5449e675485ee8ce36e0c`;
5. final checkpoint sha256 `791b7e9d9d9d61ee657886121680844ad6d5b4b9aac124aa838e3a8f6a4fc229`;
6. progress sha256 `987e8e56f611dcf8096386d258df4779769b607450dec618330dec9be4be096c`.

The next prompt is
`.codex/agents/generated/rl-trading-agent-platform-v1/08d-original-hf-backtest-evaluation.md`.
